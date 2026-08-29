"""Persist fixed-strategy paper signals without changing the configurable engine."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from valuecell.server.api.schemas.fixed_strategy import (
    FixedEngineInput,
    FixedStrategySignal,
)
from valuecell.server.db.models.rule_strategy import RuleStrategyEvaluationJournal
from valuecell.server.db.repositories.rule_strategy_repository import RuleStrategyRepository
from valuecell.server.services.fixed_strategy_dispatcher import evaluate_fixed_strategy

_FIXED_KINDS = {"dual_ma_trend", "pair_rotation", "leader_breakout"}


class FixedPaperEvaluationService:
    """Record fixed-strategy decisions as durable explainable paper evidence."""

    def __init__(self, repository: RuleStrategyRepository | None = None) -> None:
        self._repository = repository or RuleStrategyRepository()

    def evaluate_and_record(
        self,
        *,
        strategy_id: str,
        tenant_id: str,
        strategy_kind: str,
        batch_id: str | None,
        request: FixedEngineInput,
        btc_request: list[Any] | None = None,
    ) -> tuple[FixedStrategySignal, str]:
        """Evaluate and persist one signal; no order or venue side effect occurs."""
        if strategy_kind not in _FIXED_KINDS:
            raise ValueError(f"Unsupported fixed strategy kind: {strategy_kind}")
        if strategy_kind == "leader_breakout":
            if btc_request is None:
                raise ValueError("leader_breakout requires BTC candle facts")
            signal = evaluate_fixed_strategy(
                strategy_kind, request, btc_candles=btc_request
            )
        else:
            signal = evaluate_fixed_strategy(strategy_kind, request)
        result = {
            "strategy_kind": signal.kind,
            "action": signal.action,
            "reason_code": signal.reason_code,
            "reason": signal.reason,
            "conditions": [
                condition.model_dump(mode="json") for condition in signal.conditions
            ],
            "indicators": signal.indicators,
            "pair": signal.pair,
            "execution_block_reason": signal.execution_block_reason,
            "execution_ledger": "paper_signal_only",
            "paper_fill": False,
        }
        journal = self._repository.append_evaluation(
            RuleStrategyEvaluationJournal(
                evaluation_id=f"fixed_{uuid4().hex}",
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                result=result,
                signals=result["conditions"],
                trades=[],
                funding=[],
            )
        )
        return signal, journal.evaluation_id
