"""V19 account circuit breakers and local pending-entry cancellation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_account_risk import (
    LeaderSpotV19AccountRiskDecision,
    LeaderSpotV19AccountRiskInput,
    LeaderSpotV19RiskCancellationResult,
)
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19Event,
    LeaderSpotV19ExecutionIntent,
    LeaderSpotV19Position,
    LeaderSpotV19RiskState,
)


class LeaderSpotV19AccountRiskEngine:
    """Apply only V19's daily-loss halt and 15% equity emergency circuit breaker."""

    def decide(
        self,
        config: LeaderSpotV19Config,
        risk_input: LeaderSpotV19AccountRiskInput,
    ) -> LeaderSpotV19AccountRiskDecision:
        observed_at = risk_input.observed_at
        daily_pnl = risk_input.daily_realized_pnl_quote
        reset_at = risk_input.daily_loss_reset_at
        if observed_at >= reset_at:
            daily_pnl = 0.0
            reset_at = self._next_utc_midnight(observed_at)
        drawdown = max(0.0, (risk_input.prior_close_equity_quote - risk_input.equity_quote) / risk_input.prior_close_equity_quote)
        if drawdown >= config.account_risk.equity_drawdown_halt_pct:
            return LeaderSpotV19AccountRiskDecision(
                state="equity_halted", can_open=False, daily_realized_pnl_quote=daily_pnl,
                daily_loss_reset_at=reset_at, equity_drawdown_pct=drawdown,
                halt_until=observed_at + timedelta(hours=config.account_risk.equity_halt_hours),
                reason_code="equity_drawdown_halt", cancel_pending_entries=True,
                force_close_positions=True, observed_at=observed_at,
            )
        if daily_pnl <= -risk_input.daily_loss_limit_quote:
            return LeaderSpotV19AccountRiskDecision(
                state="daily_loss_halted", can_open=False, daily_realized_pnl_quote=daily_pnl,
                daily_loss_reset_at=reset_at, equity_drawdown_pct=drawdown,
                halt_until=reset_at, reason_code="daily_loss_limit_reached",
                cancel_pending_entries=True, force_close_positions=False, observed_at=observed_at,
            )
        return LeaderSpotV19AccountRiskDecision(
            state="normal", can_open=True, daily_realized_pnl_quote=daily_pnl,
            daily_loss_reset_at=reset_at, equity_drawdown_pct=drawdown, halt_until=None,
            reason_code=None, cancel_pending_entries=False, force_close_positions=False,
            observed_at=observed_at,
        )

    def apply(
        self,
        session: Session,
        *,
        config: LeaderSpotV19Config,
        risk: LeaderSpotV19RiskState,
        risk_input: LeaderSpotV19AccountRiskInput,
        execution_generation: int,
    ) -> tuple[LeaderSpotV19AccountRiskDecision, LeaderSpotV19RiskCancellationResult]:
        """Persist risk facts, cancel unsubmitted entries, and stage emergency sells."""

        decision = self.decide(config, risk_input)
        risk.state = decision.state
        risk.daily_realized_pnl_quote = decision.daily_realized_pnl_quote
        risk.daily_loss_reset_at = decision.daily_loss_reset_at
        risk.equity_drawdown_pct = decision.equity_drawdown_pct
        risk.halt_until = decision.halt_until
        risk.reason_code = decision.reason_code
        risk.reason_detail = decision.reason_code
        risk.version += 1
        cancellations = self._cancel_pending_entries(
            session, risk.tenant_id, risk.strategy_id, risk.batch_id
        ) if decision.cancel_pending_entries else LeaderSpotV19RiskCancellationResult()
        if decision.force_close_positions:
            self._stage_emergency_closes(session, risk, execution_generation)
        session.add(LeaderSpotV19Event(
            event_id=str(uuid4()), tenant_id=risk.tenant_id, strategy_id=risk.strategy_id,
            batch_id=risk.batch_id, correlation_id=risk.risk_state_id, actor="system",
            reason_code=decision.reason_code or "account_risk_normal",
            before_state={}, after_state=decision.model_dump(mode="json"),
        ))
        session.commit()
        return decision, cancellations

    @staticmethod
    def _cancel_pending_entries(
        session: Session, tenant_id: str, strategy_id: str, batch_id: str
    ) -> LeaderSpotV19RiskCancellationResult:
        pending = session.query(LeaderSpotV19ExecutionIntent).filter_by(
            tenant_id=tenant_id, strategy_id=strategy_id, batch_id=batch_id,
            side="buy",
        ).filter(LeaderSpotV19ExecutionIntent.status.in_(["pending", "open", "submitted"])).all()
        cancelled: list[str] = []
        preserved: list[str] = []
        for intent in pending:
            if intent.status == "submission_unknown":
                preserved.append(intent.intent_id)
                continue
            intent.status = "cancelled"
            intent.lifecycle_state = "cancelled"
            intent.error_code = "account_risk_cancelled_entry"
            intent.terminal_at = datetime.now(UTC)
            cancelled.append(intent.intent_id)
        return LeaderSpotV19RiskCancellationResult(
            cancelled_intent_ids=cancelled, preserved_intent_ids=preserved
        )

    @staticmethod
    def _stage_emergency_closes(
        session: Session, risk: LeaderSpotV19RiskState, generation: int
    ) -> None:
        positions = session.query(LeaderSpotV19Position).filter_by(
            tenant_id=risk.tenant_id, strategy_id=risk.strategy_id, batch_id=risk.batch_id,
            closed_at=None,
        ).order_by(LeaderSpotV19Position.peak_profit_pct.asc()).all()
        for position in positions:
            key = f"v19-{position.position_id}-EQUITY_DRAWDOWN_HALT"
            if session.query(LeaderSpotV19ExecutionIntent).filter_by(
                tenant_id=risk.tenant_id, idempotency_key=key
            ).first() is not None:
                continue
            session.add(LeaderSpotV19ExecutionIntent(
                intent_id=str(uuid4()), tenant_id=risk.tenant_id,
                strategy_id=risk.strategy_id, batch_id=risk.batch_id,
                position_id=position.position_id, execution_generation=generation,
                idempotency_key=key, symbol=position.symbol, side="sell",
                order_type="market", leg_kind="close",
                requested_quantity=position.entry_quantity, lifecycle_state="pending",
                status="pending", request_payload={"reason_code": "EQUITY_DRAWDOWN_HALT"},
            ))

    @staticmethod
    def _next_utc_midnight(value: datetime) -> datetime:
        current = value.astimezone(UTC)
        return current.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
