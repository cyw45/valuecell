"""Ordered fail-closed candidate funnel for the V19 leader strategy."""

from __future__ import annotations


from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_candidate import (
    LeaderSpotV19CandidateDecision,
    LeaderSpotV19CandidateInput,
    LeaderSpotV19CandidateStep,
)
from valuecell.server.api.schemas.leader_spot_v19_market_state import (
    LeaderSpotV19MarketStateDecision,
)
from valuecell.server.db.models.leader_spot_v19 import LeaderSpotV19CandidateSnapshot


class LeaderSpotV19CandidateFilter:
    """Evaluate V19 candidate facts in required order; never infer missing V16.1 data."""

    def evaluate(
        self,
        config: LeaderSpotV19Config,
        market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateDecision:
        steps: list[LeaderSpotV19CandidateStep] = []
        for evaluator in (
            self._entry_state,
            self._liquidity,
            self._new_coin,
            self._relative_strength,
            self._anomaly,
            self._box_breakout,
            self._score,
            self._order_book,
        ):
            step = evaluator(config, market, candidate)
            steps.append(step)
            if not step.passed:
                return LeaderSpotV19CandidateDecision(
                    symbol=candidate.symbol,
                    source_rank=candidate.source_rank,
                    accepted=False,
                    score=candidate.score.total_score if candidate.score else None,
                    reason_code=step.reason_code,
                    steps=steps,
                    observed_at=candidate.observed_at,
                )
        return LeaderSpotV19CandidateDecision(
            symbol=candidate.symbol,
            source_rank=candidate.source_rank,
            accepted=True,
            score=candidate.score.total_score if candidate.score else None,
            reason_code=None,
            steps=steps,
            observed_at=candidate.observed_at,
        )

    def evaluate_and_persist(
        self,
        session: Session,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        snapshot_group_id: str,
        config: LeaderSpotV19Config,
        market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateDecision:
        """Persist every attempted funnel decision, including rejection facts."""

        decision = self.evaluate(config, market, candidate)
        session.add(
            LeaderSpotV19CandidateSnapshot(
                candidate_id=str(uuid4()),
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                snapshot_group_id=snapshot_group_id,
                source="okx",
                symbol=decision.symbol,
                source_rank=decision.source_rank,
                market_state=market.market_state,
                data_state=candidate.data_state,
                funnel_stage=decision.steps[-1].stage,
                accepted=decision.accepted,
                score=decision.score,
                reason_code=decision.reason_code,
                facts={
                    "steps": [step.model_dump(mode="json") for step in decision.steps],
                    "candidate": candidate.model_dump(mode="json"),
                    "market_reason_codes": market.reason_codes,
                },
                observed_at=decision.observed_at,
            )
        )
        session.commit()
        return decision

    @staticmethod
    def _entry_state(
        _config: LeaderSpotV19Config,
        market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateStep:
        passed = market.can_open and candidate.data_state == "DATA_OK"
        return LeaderSpotV19CandidateStep(
            stage="entry_state",
            passed=passed,
            reason_code=None if passed else "entry_state_not_open",
            facts={
                "market_state": market.market_state,
                "data_state": candidate.data_state,
                "can_open": market.can_open,
            },
        )

    @staticmethod
    def _liquidity(
        _config: LeaderSpotV19Config,
        market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateStep:
        threshold = market.starvation.liquidity_quote
        passed = candidate.quote_volume_24h >= threshold
        return LeaderSpotV19CandidateStep(
            stage="liquidity",
            passed=passed,
            reason_code=None if passed else "liquidity_below_threshold",
            facts={"quote_volume_24h": candidate.quote_volume_24h, "threshold": threshold},
        )

    @staticmethod
    def _new_coin(
        config: LeaderSpotV19Config,
        market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateStep:
        age_hours = (candidate.observed_at - candidate.listing_at).total_seconds() / 3600
        if age_hours < config.new_coin.ban_hours:
            return LeaderSpotV19CandidateStep(
                stage="new_coin",
                passed=False,
                reason_code="new_coin_banned",
                facts={"age_hours": age_hours},
            )
        if age_hours < config.new_coin.strict_hours:
            passed = (
                market.market_state in {"M3", "M4"}
                and candidate.strict_new_coin_requirements_met
                and candidate.enhanced_depth_confirmed
            )
            return LeaderSpotV19CandidateStep(
                stage="new_coin",
                passed=passed,
                reason_code=None if passed else "new_coin_strict_requirements_unmet",
                facts={
                    "age_hours": age_hours,
                    "market_state": market.market_state,
                    "strict_requirements_met": candidate.strict_new_coin_requirements_met,
                    "enhanced_depth_confirmed": candidate.enhanced_depth_confirmed,
                },
            )
        return LeaderSpotV19CandidateStep(
            stage="new_coin",
            passed=True,
            facts={"age_hours": age_hours},
        )
    @staticmethod
    def _relative_strength(
        _config: LeaderSpotV19Config,
        market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateStep:
        rs_passed = candidate.relative_strength_rank_pct <= market.starvation.relative_strength_rank_pct
        high_pump = candidate.return_24h_pct > 0.40
        pump_passed = not high_pump or (
            market.market_state in {"M3", "M4"} and candidate.high_pump_retest_confirmed
        )
        passed = rs_passed and pump_passed
        reason = None
        if not rs_passed:
            reason = "relative_strength_below_threshold"
        elif not pump_passed:
            reason = "high_pump_not_retest_confirmed"
        return LeaderSpotV19CandidateStep(
            stage="relative_strength",
            passed=passed,
            reason_code=reason,
            facts={
                "rank_pct": candidate.relative_strength_rank_pct,
                "threshold": market.starvation.relative_strength_rank_pct,
                "return_24h_pct": candidate.return_24h_pct,
                "high_pump_retest_confirmed": candidate.high_pump_retest_confirmed,
            },
        )

    @staticmethod
    def _anomaly(
        config: LeaderSpotV19Config,
        _market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateStep:
        passed = not candidate.needle_detected and candidate.br_value <= config.breakout.needle_br_threshold
        reason = "needle_detected" if candidate.needle_detected else "br_above_threshold"
        return LeaderSpotV19CandidateStep(
            stage="anomaly",
            passed=passed,
            reason_code=None if passed else reason,
            facts={"needle_detected": candidate.needle_detected, "br_value": candidate.br_value},
        )

    @staticmethod
    def _box_breakout(
        config: LeaderSpotV19Config,
        market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateStep:
        evidence = candidate.box
        if evidence is None:
            return LeaderSpotV19CandidateStep(
                stage="box_breakout",
                passed=False,
                reason_code="v16_1_box_parameters_unavailable",
            )
        minimum_volume = 1.0
        age_hours = (candidate.observed_at - candidate.listing_at).total_seconds() / 3600
        if market.market_state == "M2":
            minimum_volume += config.breakout.box_volume_degraded_add
        if age_hours < config.new_coin.degraded_hours:
            minimum_volume += config.new_coin.degraded_volume_add
        passed = (
            evidence.passed
            and evidence.fifteen_minute_close_confirmed
            and evidence.five_minute_close_confirmations >= 2
            and evidence.second_five_minute_volume_confirmed
            and evidence.volume_multiplier >= minimum_volume
        )
        return LeaderSpotV19CandidateStep(
            stage="box_breakout",
            passed=passed,
            reason_code=None if passed else "box_breakout_not_confirmed",
            facts={
                "parameter_fingerprint": evidence.parameter_fingerprint,
                "five_minute_close_confirmations": evidence.five_minute_close_confirmations,
                "volume_multiplier": evidence.volume_multiplier,
                "minimum_volume_multiplier": minimum_volume,
            },
        )

    @staticmethod
    def _score(
        config: LeaderSpotV19Config,
        market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateStep:
        evidence = candidate.score
        if evidence is None:
            return LeaderSpotV19CandidateStep(
                stage="score",
                passed=False,
                reason_code="score_formula_unavailable",
            )
        threshold = market.starvation.score_threshold
        if market.market_state == "M2":
            threshold = 42
        age_hours = (candidate.observed_at - candidate.listing_at).total_seconds() / 3600
        if age_hours < config.new_coin.degraded_hours:
            threshold += config.new_coin.degraded_score_add
        passed = evidence.total_score >= threshold
        return LeaderSpotV19CandidateStep(
            stage="score",
            passed=passed,
            reason_code=None if passed else "score_below_threshold",
            facts={
                "score": evidence.total_score,
                "threshold": threshold,
                "formula_source": evidence.formula_source,
                "formula_fingerprint": evidence.formula_fingerprint,
            },
        )

    @staticmethod
    def _order_book(
        config: LeaderSpotV19Config,
        _market: LeaderSpotV19MarketStateDecision,
        candidate: LeaderSpotV19CandidateInput,
    ) -> LeaderSpotV19CandidateStep:
        book = candidate.order_book
        if book is None:
            return LeaderSpotV19CandidateStep(
                stage="order_book",
                passed=False,
                reason_code="order_book_unavailable",
            )
        available_quote = sum(level.price * level.quantity for level in book.asks)
        passed = available_quote >= config.position.order_amount_quote
        return LeaderSpotV19CandidateStep(
            stage="order_book",
            passed=passed,
            reason_code=None if passed else "five_level_depth_insufficient",
            facts={
                "ask_quote_depth": available_quote,
                "required_quote": config.position.order_amount_quote,
                "estimated_slippage_pct": candidate.estimated_entry_slippage_pct,
            },
        )
