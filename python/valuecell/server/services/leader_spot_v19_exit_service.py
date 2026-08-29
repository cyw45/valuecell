"""V19 stop, protection, retracement, and trend exit engine."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_exit import (
    LeaderSpotV19ExitDecision,
    LeaderSpotV19PositionExitInput,
)
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19Event,
    LeaderSpotV19ExecutionIntent,
    LeaderSpotV19Position,
)


class LeaderSpotV19ExitEngine:
    """Compute V19 exits deterministically; exits are staged before venue submission."""

    def decide(
        self, config: LeaderSpotV19Config, position: LeaderSpotV19PositionExitInput
    ) -> LeaderSpotV19ExitDecision:
        entry = position.entry_price
        current_profit = position.current_bid / entry - 1
        peak = position.peak_price
        status = position.protection_status
        started_at = position.protection_started_at
        loss_active = position.loss_circuit_active
        moving_stop = position.moving_stop_price
        layered_stop = position.layered_exit_price
        break_count = position.trend_break_count

        if status in {"PROTECTION_NONE", "PROTECTION_PENDING"}:
            stop_price = entry * (1 - config.loss.stop_loss_pct)
            hard_stop_confirmed = (
                position.hard_stop_two_source_confirmed
                or position.hard_stop_local_bid_persistent
            )
            if position.current_bid <= stop_price and hard_stop_confirmed:
                return self._decision(position, status, peak, moving_stop, layered_stop, loss_active, break_count, "STOP_LOSS_8PCT", "market")
            if loss_active and position.observed_at - position.entry_time >= timedelta(hours=config.loss.loss_circuit_hours):
                return self._decision(position, status, peak, moving_stop, layered_stop, loss_active, break_count, "LOSS_CIRCUIT_7D", "market")
            if status == "PROTECTION_NONE" and current_profit >= config.profit.protection_profit_pct:
                status = "PROTECTION_PENDING"
                started_at = position.observed_at
            elif status == "PROTECTION_PENDING":
                if current_profit < config.profit.protection_profit_pct:
                    status = "PROTECTION_NONE"
                    started_at = None
                elif started_at is not None and position.observed_at - started_at >= timedelta(seconds=config.profit.protection_hold_seconds):
                    status = "PROTECTION_ACTIVE"
                    loss_active = False
                    peak = max(peak, entry * (1 + config.profit.protection_profit_pct), position.closed_one_minute_high or 0)
            return self._decision(position, status, peak, moving_stop, layered_stop, loss_active, break_count, None, None, protection_started_at=started_at)

        peak = max(peak, position.closed_one_minute_high or 0)
        peak_profit = max(position.peak_profit_pct, peak / entry - 1)
        moving_stop = entry * self._moving_stop_multiplier(config, peak_profit)
        layered_stop = self._layered_stop(config, entry, peak, current_profit)
        moving_triggered = position.current_bid <= moving_stop
        layered_triggered = layered_stop is not None and position.current_bid <= layered_stop
        if moving_triggered:
            return self._decision(position, status, peak, moving_stop, layered_stop, False, break_count, "MOVING_STOP", "market", peak_profit=peak_profit)
        if layered_triggered:
            return self._decision(position, status, peak, moving_stop, layered_stop, False, break_count, "LAYERED_RETRACEMENT", "market", peak_profit=peak_profit)

        trend_exit, break_count = self._trend_exit(config, position, current_profit)
        if trend_exit:
            return self._decision(position, status, peak, moving_stop, layered_stop, False, break_count, "TREND_EXIT", "limit", limit_price=position.current_bid * (1 - config.profit.trend.limit_offset), peak_profit=peak_profit)
        return self._decision(position, status, peak, moving_stop, layered_stop, False, break_count, None, None, peak_profit=peak_profit)

    def decide_and_stage_exit(
        self, session: Session, *, config: LeaderSpotV19Config, position: LeaderSpotV19Position, market_state: str, current_bid: float, closed_one_minute_high: float | None, fifteen_minute_closes: list[float], execution_generation: int, now, hard_stop_two_source_confirmed: bool = False, hard_stop_local_bid_persistent: bool = False, trend_data_valid: bool = False,
    ) -> LeaderSpotV19ExitDecision:
        entry_time = position.entry_time.replace(tzinfo=now.tzinfo) if position.entry_time.tzinfo is None else position.entry_time
        protection_started_at = position.protection_started_at
        if protection_started_at is not None and protection_started_at.tzinfo is None:
            protection_started_at = protection_started_at.replace(tzinfo=now.tzinfo)
        data = LeaderSpotV19PositionExitInput(
            position_id=position.position_id, symbol=position.symbol, entry_price=float(position.entry_price), quantity=float(position.entry_quantity), entry_time=entry_time, protection_status=position.protection_status, protection_started_at=protection_started_at, peak_price=float(position.peak_price), peak_profit_pct=position.peak_profit_pct, moving_stop_price=float(position.moving_stop_price), layered_exit_price=float(position.layered_exit_price) if position.layered_exit_price else None, loss_circuit_active=position.loss_circuit_active, hard_stop_two_source_confirmed=hard_stop_two_source_confirmed, hard_stop_local_bid_persistent=hard_stop_local_bid_persistent, trend_data_valid=trend_data_valid, trend_break_count=position.trend_break_count, current_bid=current_bid, closed_one_minute_high=closed_one_minute_high, fifteen_minute_closes=fifteen_minute_closes, market_state=market_state, observed_at=now,
        )
        decision = self.decide(config, data)
        position.protection_status = decision.protection_status
        position.protection_started_at = now if decision.protection_status == "PROTECTION_PENDING" and position.protection_started_at is None else (None if decision.protection_status == "PROTECTION_NONE" else position.protection_started_at)
        position.peak_price = str(decision.peak_price)
        position.peak_profit_pct = decision.peak_profit_pct
        position.moving_stop_price = str(decision.moving_stop_price)
        position.layered_exit_price = str(decision.layered_exit_price) if decision.layered_exit_price else None
        position.loss_circuit_active = decision.loss_circuit_active
        position.trend_break_count = decision.trend_break_count
        if decision.exit_reason_code is not None:
            self._stage_exit_intent(session, position, execution_generation, decision)
        session.add(LeaderSpotV19Event(event_id=str(uuid4()), tenant_id=position.tenant_id, strategy_id=position.strategy_id, batch_id=position.batch_id, correlation_id=position.position_id, position_id=position.position_id, actor="system", reason_code=decision.exit_reason_code or "protection_updated", before_state={}, after_state=decision.model_dump(mode="json")))
        session.commit()
        return decision

    @staticmethod
    def _moving_stop_multiplier(config, peak_profit):
        return next((tier.stop_multiplier for tier in reversed(config.profit.moving_stop_tiers) if peak_profit >= tier.profit_pct), config.profit.moving_stop_initial_multiplier)

    @staticmethod
    def _layered_stop(config, entry, peak, current_profit):
        tier = next((item for item in config.profit.layered_exit_tiers if current_profit >= item.minimum_profit_pct and (item.maximum_profit_pct is None or current_profit < item.maximum_profit_pct)), None)
        return None if tier is None else max(peak * (1 - tier.retracement_pct), entry * tier.floor_multiplier)

    @staticmethod
    def _ema(values, period):
        multiplier = 2 / (period + 1)
        result = values[0]
        for value in values[1:]: result += (value - result) * multiplier
        return result

    def _trend_exit(self, config, position, current_profit):
        threshold = config.profit.trend.degraded_profit_threshold if position.market_state == "M2" else config.profit.trend.standard_profit_threshold
        values = position.fifteen_minute_closes
        if not position.trend_data_valid or current_profit < threshold or len(values) < config.profit.trend.ema_slow_period + 1:
            return False, 0
        fast = self._ema(values, config.profit.trend.ema_fast_period)
        previous_fast = self._ema(values[:-1], config.profit.trend.ema_fast_period)
        slow = self._ema(values, config.profit.trend.ema_slow_period)
        count = position.trend_break_count + 1 if values[-1] < fast else 0
        return count >= 2 and (fast < previous_fast or values[-1] < slow), count

    @staticmethod
    def _decision(position, status, peak, moving, layered, loss_active, breaks, reason, order_type, *, limit_price=None, peak_profit=None, protection_started_at=None):
        return LeaderSpotV19ExitDecision(position_id=position.position_id, protection_status=status, peak_price=peak, peak_profit_pct=position.peak_profit_pct if peak_profit is None else peak_profit, moving_stop_price=moving, layered_exit_price=layered, loss_circuit_active=loss_active, trend_break_count=breaks, exit_reason_code=reason, order_type=order_type, limit_price=limit_price, observed_at=position.observed_at)

    @staticmethod
    def _stage_exit_intent(session, position, generation, decision):
        key = f"v19-{position.position_id}-{decision.exit_reason_code}"
        if session.query(LeaderSpotV19ExecutionIntent).filter_by(tenant_id=position.tenant_id, idempotency_key=key).first() is None:
            session.add(LeaderSpotV19ExecutionIntent(intent_id=str(uuid4()), tenant_id=position.tenant_id, strategy_id=position.strategy_id, batch_id=position.batch_id, position_id=position.position_id, execution_generation=generation, idempotency_key=key, symbol=position.symbol, side="sell", order_type=decision.order_type, leg_kind="close", requested_quantity=position.entry_quantity, requested_price=str(decision.limit_price) if decision.limit_price else None, lifecycle_state="pending", status="pending", request_payload={"reason_code": decision.exit_reason_code}))
