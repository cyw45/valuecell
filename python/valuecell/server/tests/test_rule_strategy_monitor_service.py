from datetime import datetime, timezone
from types import SimpleNamespace

from valuecell.server.services.rule_strategy_monitor_service import (
    StrategyMarketMetadata,
    decide_monitor_state,
)
from valuecell.server.services.rule_strategy_service import RuleStrategyService


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
    assert decision.reason_code == "market_metadata_unavailable"
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


def test_observation_marks_unavailable_provider_evidence_explicitly():
    decision = decide_monitor_state(_row(), metadata=None, position_quantity=0)

    assert decision.state == "admitted"
    assert decision.reason_code == "market_metadata_unavailable"


class MonitorRepository:
    def __init__(self) -> None:
        self.row = SimpleNamespace(
            id=1,
            symbol="BTC-USDT",
            consecutive_low_volume_days=0,
        )
        self.saved: dict[str, object] = {}

    def monitors(self, _strategy_id: str, _tenant_id: str):
        return [self.row]

    def get_account_state(self, _strategy_id: str, _tenant_id: str):
        return None

    def claim_monitor_lease(self, *_args, **_kwargs):
        return [self.row]

    def update_monitor_state(self, _monitor_id: int, _tenant_id: str, **values):
        self.saved = values
        return self.row


class MarketMetadataService:
    def get_monitor_metadata(self, symbols: list[str]):
        observed_at = datetime(2026, 8, 6, tzinfo=timezone.utc)
        return {
            symbol: SimpleNamespace(
                provider="binance",
                listing_first_tradable_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                listing_age_days=2_410,
                average_quote_volume_30d=12_500_000.0,
                price_quote=98_000.0,
                price_observed_at=observed_at,
            )
            for symbol in symbols
        }


def test_monitor_refresh_fetches_and_persists_exchange_facts():
    repository = MonitorRepository()
    service = RuleStrategyService(
        repository=repository,
        market_service=MarketMetadataService(),
    )

    service._refresh_monitor_admission("strategy-a", "tenant-a", force=True)

    assert repository.saved["metadata_provider"] == "binance"
    assert repository.saved["listing_age_days"] == 2_410
    assert repository.saved["average_quote_volume_30d"] == 12_500_000.0
    assert repository.saved["price_quote"] == 98_000.0
