from valuecell.server.api.schemas.multi_strategy import StrategyKind
from valuecell.server.services.multi_strategy_registry import (
    fixed_strategy_definitions,
    strategy_code_fingerprint,
    strategy_definition,
)


def test_strategy_registry_resolves_all_kinds() -> None:
    kinds: tuple[StrategyKind, ...] = (
        "configurable_rule",
        "dual_ma_trend",
        "pair_rotation",
        "leader_breakout",
    )
    assert tuple(strategy_definition(kind).kind for kind in kinds) == kinds


def test_fixed_strategy_fingerprint_is_stable() -> None:
    assert strategy_code_fingerprint("dual_ma_trend") == strategy_code_fingerprint("dual_ma_trend")
    assert strategy_code_fingerprint("dual_ma_trend") != strategy_code_fingerprint("pair_rotation")


def test_fixed_registry_excludes_editable_strategy() -> None:
    assert {definition.kind for definition in fixed_strategy_definitions()} == {
        "dual_ma_trend",
        "pair_rotation",
        "leader_breakout",
    }
