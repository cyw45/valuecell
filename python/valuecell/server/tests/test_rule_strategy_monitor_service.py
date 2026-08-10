from types import SimpleNamespace

from valuecell.server.services.rule_strategy_monitor_service import (
    StrategyMarketMetadata,
    decide_monitor_state,
)


def _row():
    return SimpleNamespace(
        consecutive_low_volume_days=0,
    )


def test_configured_symbol_is_observed_without_listing_or_volume_metadata():
    decision = decide_monitor_state(
        _row(),
        StrategyMarketMetadata(
            listing_age_days=None,
            average_quote_volume_30d=None,
            price_quote=0.12,
            provider="okx",
        ),
        position_quantity=0,
    )

    assert decision.state == "admitted"
    assert decision.reason_code == "monitor_observation_enabled"
    assert decision.protected_held is False


def test_observation_keeps_held_position_protected():
    decision = decide_monitor_state(
        _row(),
        metadata=None,
        position_quantity=2.5,
    )

    assert decision.state == "held"
    assert decision.reason_code == "monitor_position_protected"
    assert decision.protected_held is True
