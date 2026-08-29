"""Dispatch registered fixed strategy engines without venue side effects."""

from __future__ import annotations

from collections.abc import Sequence

from valuecell.server.api.schemas.fixed_strategy import (
    FixedCandle,
    FixedEngineInput,
    FixedStrategyKind,
    FixedStrategySignal,
)
from valuecell.server.services.fixed_dual_ma_engine import evaluate_fixed_dual_ma
from valuecell.server.services.fixed_leader_engine import evaluate_leader_candidate
from valuecell.server.services.fixed_pair_rotation_engine import evaluate_pair_rotation


class FixedStrategyEngineUnavailableError(RuntimeError):
    """Raised when a fixed strategy lacks required engine inputs."""


def evaluate_fixed_strategy(
    kind: FixedStrategyKind,
    request: FixedEngineInput,
    *,
    btc_candles: Sequence[FixedCandle] | None = None,
) -> FixedStrategySignal:
    """Evaluate one fixed strategy using only supplied platform facts."""

    if kind == "dual_ma_trend":
        return evaluate_fixed_dual_ma(request)
    if kind == "pair_rotation":
        return evaluate_pair_rotation(request)
    if btc_candles is None:
        raise FixedStrategyEngineUnavailableError(
            "leader_breakout requires a separate final BTC candle series"
        )
    return evaluate_leader_candidate(request, btc_candles)


__all__ = [
    "FixedStrategyEngineUnavailableError",
    "evaluate_fixed_strategy",
]
