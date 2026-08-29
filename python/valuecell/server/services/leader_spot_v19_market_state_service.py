"""Deterministic V19 market state machine and signal-starvation policy."""

from __future__ import annotations


from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.leader_spot_v19_market_state import (
    LeaderSpotV19MarketCondition,
    LeaderSpotV19MarketStateDecision,
    LeaderSpotV19MarketStateInput,
    LeaderSpotV19SignalStarvationPolicy,
)
from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.db.models.leader_spot_v19 import LeaderSpotV19MarketStateDecision as MarketStateDecisionRow


_STARVATION_48H = 48
_STARVATION_72H = 72


class LeaderSpotV19MarketStateEngine:
    """Calculate the V19 M0–M4 state without fetching data or placing orders."""

    def decide(
        self,
        config: LeaderSpotV19Config,
        inputs: LeaderSpotV19MarketStateInput,
    ) -> LeaderSpotV19MarketStateDecision:
        conditions = self._conditions(config, inputs)
        starvation = self._starvation(config, inputs)
        data_unsafe = inputs.data_state == "DATA_UNSAFE"
        halt_reasons = self._halt_reasons(config, inputs, conditions)
        standard_fail_count = sum(
            not condition.passed
            for condition in conditions[:4]
        )
        if data_unsafe:
            state = "M0"
            profile = "halt"
            reasons = ["data_unsafe"]
        elif halt_reasons:
            state = "M1"
            profile = "halt"
            reasons = halt_reasons
        elif inputs.data_state == "DATA_DEGRADED":
            if standard_fail_count == 0:
                state = "M3"
                profile = "standard"
                reasons = ["data_degraded_standard_only"]
            else:
                state = "M1"
                profile = "halt"
                reasons = ["data_degraded_requires_standard_market"]
        elif standard_fail_count == 0:
            if self._strong_trend(config, inputs):
                state = "M4"
                profile = "strong_trend"
                reasons = ["strong_trend_market"]
            else:
                state = "M3"
                profile = "standard"
                reasons = ["standard_market"]
        elif standard_fail_count <= config.market.standard_allow_fail:
            state = "M2"
            profile = "degraded"
            reasons = ["degraded_market_one_standard_condition_failed"]
        else:
            state = "M1"
            profile = "halt"
            reasons = ["two_or_more_standard_conditions_failed"]
        return LeaderSpotV19MarketStateDecision(
            market_state=state,
            entry_profile=profile,
            can_open=state in {"M2", "M3", "M4"},
            reason_codes=reasons,
            conditions=conditions,
            starvation=starvation,
            observed_at=inputs.observed_at,
        )

    def decide_and_persist(
        self,
        session: Session,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        config: LeaderSpotV19Config,
        inputs: LeaderSpotV19MarketStateInput,
    ) -> LeaderSpotV19MarketStateDecision:
        """Persist auditable inputs and decision before candidate filtering begins."""

        decision = self.decide(config, inputs)
        session.add(
            MarketStateDecisionRow(
                decision_id=str(uuid4()),
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                data_state=inputs.data_state,
                market_state=decision.market_state,
                entry_profile=decision.entry_profile,
                can_open=decision.can_open,
                reason_codes=decision.reason_codes,
                input_facts=inputs.model_dump(mode="json"),
                conditions=[condition.model_dump(mode="json") for condition in decision.conditions],
                starvation=decision.starvation.model_dump(mode="json"),
                observed_at=decision.observed_at,
            )
        )
        session.commit()
        return decision

    @staticmethod
    def _conditions(
        config: LeaderSpotV19Config,
        inputs: LeaderSpotV19MarketStateInput,
    ) -> list[LeaderSpotV19MarketCondition]:
        market = config.market
        return [
            LeaderSpotV19MarketCondition(
                code="up_ratio_standard",
                passed=inputs.up_ratio > market.up_ratio_standard,
                actual=inputs.up_ratio,
                threshold=market.up_ratio_standard,
            ),
            LeaderSpotV19MarketCondition(
                code="volume_ratio_standard",
                passed=inputs.volume_ratio_to_5d_average > market.volume_ratio_standard,
                actual=inputs.volume_ratio_to_5d_average,
                threshold=market.volume_ratio_standard,
            ),
            LeaderSpotV19MarketCondition(
                code="fear_greed_standard",
                passed=inputs.fear_greed_index > market.fear_greed_standard,
                actual=inputs.fear_greed_index,
                threshold=market.fear_greed_standard,
            ),
            LeaderSpotV19MarketCondition(
                code="funding_rate_standard",
                passed=market.funding_rate_standard_min
                <= inputs.funding_rate
                <= market.funding_rate_standard_max,
                actual=inputs.funding_rate,
                threshold=(
                    f"[{market.funding_rate_standard_min},"
                    f"{market.funding_rate_standard_max}]"
                ),
            ),
        ]

    @staticmethod
    def _halt_reasons(
        config: LeaderSpotV19Config,
        inputs: LeaderSpotV19MarketStateInput,
        conditions: list[LeaderSpotV19MarketCondition],
    ) -> list[str]:
        market = config.market
        reasons: list[str] = []
        if inputs.daily_loss_limit_reached:
            reasons.append("daily_loss_limit_reached")
        if inputs.up_ratio <= market.up_ratio_halt:
            reasons.append("up_ratio_halt")
        if inputs.fear_greed_index < market.fear_greed_degraded:
            reasons.append("fear_greed_halt")
        if not (
            market.funding_rate_degraded_min
            <= inputs.funding_rate
            <= market.funding_rate_degraded_max
        ):
            reasons.append("funding_rate_halt")
        standard_fails = sum(not condition.passed for condition in conditions)
        if standard_fails >= 2:
            reasons.append("standard_conditions_failed")
        return reasons

    @staticmethod
    def _strong_trend(
        config: LeaderSpotV19Config,
        inputs: LeaderSpotV19MarketStateInput,
    ) -> bool:
        market = config.market
        return (
            inputs.up_ratio > market.strong_trend_up_ratio
            and inputs.volume_ratio_to_5d_average > market.strong_trend_volume_ratio
        )

    @staticmethod
    def _starvation(
        config: LeaderSpotV19Config,
        inputs: LeaderSpotV19MarketStateInput,
    ) -> LeaderSpotV19SignalStarvationPolicy:
        candidate = config.candidate
        recovered = inputs.valid_candidate_count >= candidate.signal_recover_count
        elapsed_hours = 0.0
        if not recovered and inputs.no_valid_candidate_since is not None:
            elapsed_hours = max(
                0.0,
                (inputs.observed_at - inputs.no_valid_candidate_since).total_seconds() / 3600,
            )
        rs_rank = candidate.relative_strength_rank_pct_standard
        liquidity = candidate.liquidity_quote_standard
        score = candidate.score_standard
        if elapsed_hours >= _STARVATION_48H:
            rs_rank = candidate.relative_strength_rank_pct_starved
        if elapsed_hours >= _STARVATION_72H:
            liquidity = candidate.liquidity_quote_starved
            score = candidate.score_starved
        return LeaderSpotV19SignalStarvationPolicy(
            elapsed_hours=elapsed_hours,
            recovered=recovered,
            relative_strength_rank_pct=rs_rank,
            liquidity_quote=liquidity,
            score_threshold=score,
        )
