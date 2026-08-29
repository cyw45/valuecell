"""Registry for configurable and code-owned multi-strategy identities.

The registry intentionally contains metadata only. Fixed strategy algorithms are
implemented later and cannot be executed merely by appearing in this registry.
"""

from __future__ import annotations
import hashlib
import json

import valuecell.server.api.schemas.rule_strategy as rule_strategy_schema
from valuecell.server.api.schemas.multi_strategy import (
    STRATEGY_DEFINITIONS,
    StrategyDefinition,
    StrategyKind,
)


def strategy_definition(kind: StrategyKind) -> StrategyDefinition:
    """Return one registered definition or fail instead of silently falling back."""

    for definition in STRATEGY_DEFINITIONS:
        if definition.kind == kind:
            return definition
    raise ValueError(f"Unknown strategy kind: {kind}")


def strategy_code_fingerprint(kind: StrategyKind) -> str:
    """Fingerprint stable code-owned metadata before an engine exists."""

    definition = strategy_definition(kind)
    payload = definition.model_dump(mode="json")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

_FIXED_STRATEGY_SYMBOLS: dict[StrategyKind, tuple[str, ...]] = {
    "dual_ma_trend": (
        "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
        "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "TRX-USDT", "LINK-USDT",
        "DOT-USDT", "POL-USDT", "LTC-USDT", "BCH-USDT", "ATOM-USDT",
        "UNI-USDT", "APT-USDT", "FIL-USDT", "ARB-USDT", "OP-USDT",
        "NEAR-USDT", "INJ-USDT", "SUI-USDT", "SEI-USDT", "TIA-USDT",
        "RUNE-USDT", "AAVE-USDT", "MKR-USDT", "GRT-USDT", "STX-USDT",
        "IMX-USDT", "LDO-USDT", "PEPE-USDT", "WLD-USDT", "ORDI-USDT",
        "JUP-USDT", "WIF-USDT", "S-USDT", "ALGO-USDT", "EOS-USDT",
        "XTZ-USDT",
    ),
    "pair_rotation": (
        "AAVE-USDT", "ADA-USDT", "BNB-USDT", "BTC-USDT", "DOGE-USDT",
        "ETH-USDT", "INJ-USDT", "LINK-USDT", "LTC-USDT", "NEAR-USDT",
        "PEPE-USDT", "SOL-USDT",
    ),
    # The leader engine owns its dynamic 40-symbol universe. BTC is a bootstrap
    # symbol required by the persisted config before that engine exists.
    "leader_breakout": ("BTC-USDT",),
}


def fixed_strategy_config(
    kind: StrategyKind,
    *,
    initial_capital_quote: float,
    environment: str = "paper",
    sandbox_connection_id: str | None = None,
) -> rule_strategy_schema.RuleStrategyConfig:
    """Build non-editable storage config without implementing a fixed engine."""

    if kind == "configurable_rule":
        raise ValueError("configurable_rule does not have a fixed configuration")
    execution = rule_strategy_schema.RuleStrategyExecutionConfig(
        environment=environment,
        sandbox_connection_id=sandbox_connection_id,
    )
    risk = rule_strategy_schema.RuleStrategyRiskConfig(
        order_quote_amount=100.0,
        stop_loss_pct=0.05 if kind == "dual_ma_trend" else 0.08 if kind == "leader_breakout" else None,
        max_positions=6,
    )
    return rule_strategy_schema.RuleStrategyConfig(
        initial_capital_quote=initial_capital_quote,
        symbols=list(_FIXED_STRATEGY_SYMBOLS[kind]),
        interval="4h",
        risk=risk,
        execution=execution,
    )


def fixed_strategy_definitions() -> tuple[StrategyDefinition, ...]:
    """Return code-owned strategies without including the editable legacy kind."""

    return tuple(
        definition
        for definition in STRATEGY_DEFINITIONS
        if definition.parameter_source == "code"
    )
