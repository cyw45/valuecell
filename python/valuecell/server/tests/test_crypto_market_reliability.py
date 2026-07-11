import asyncio
from unittest.mock import AsyncMock

import pytest

from valuecell.server.api.schemas.crypto_market import CryptoCandleData
from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.services.crypto_market_service import (
    CandleFetchResult,
    CryptoMarketService,
)


BASE_TS = 1_700_000_000_000


@pytest.fixture(autouse=True)
def clear_crypto_market_service_state():
    CryptoMarketService._cache.clear()
    CryptoMarketService._inflight.clear()
    CryptoMarketService._provider_health.clear()
    yield
    CryptoMarketService._cache.clear()
    CryptoMarketService._inflight.clear()
    CryptoMarketService._provider_health.clear()


def _fetch_result(
    provider: str,
    symbol: str,
    interval: str,
    lookback: int,
) -> CandleFetchResult:
    candles = [
        CryptoCandleData(
            ts=BASE_TS + index * 60_000,
            open=100.0 + index,
            high=101.0 + index,
            low=99.0 + index,
            close=100.5 + index,
            volume=1_000.0,
        )
        for index in range(lookback)
    ]
    return CandleFetchResult(
        symbol=symbol,
        exchange_symbol=symbol.replace("-", "/"),
        provider=provider,
        candles=candles,
    )


@pytest.mark.asyncio
async def test_concurrent_identical_provider_fetches_share_one_network_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    fetch_started = asyncio.Event()
    joined_inflight = asyncio.Event()
    release_fetch = asyncio.Event()

    class InflightRegistry(dict):
        def get(self, key, default=None):
            task = super().get(key, default)
            if task is not None:
                joined_inflight.set()
            return task

    monkeypatch.setattr(CryptoMarketService, "_inflight", InflightRegistry())

    async def fetch_network(
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
    ) -> CandleFetchResult:
        fetch_started.set()
        await release_fetch.wait()
        return _fetch_result(provider, symbol, interval, lookback)

    network_fetch = AsyncMock(side_effect=fetch_network)
    monkeypatch.setattr(service, "_fetch_uncached", network_fetch)

    first = asyncio.create_task(
        service._fetch_from_provider(
            provider="okx", symbol="BTC-USDT", interval="15m", lookback=2
        )
    )
    await fetch_started.wait()
    second = asyncio.create_task(
        service._fetch_from_provider(
            provider="okx", symbol="BTC-USDT", interval="15m", lookback=2
        )
    )
    await joined_inflight.wait()
    release_fetch.set()

    first_result, second_result = await asyncio.gather(first, second)

    network_fetch.assert_awaited_once_with("okx", "BTC-USDT", "15m", 2)
    assert first_result == second_result
    assert first_result.candles[-1].close == 101.5


@pytest.mark.asyncio
async def test_cached_provider_fetch_returns_result_without_a_second_network_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    network_fetch = AsyncMock(
        side_effect=lambda provider, symbol, interval, lookback: _fetch_result(
            provider, symbol, interval, lookback
        )
    )
    monkeypatch.setattr(service, "_fetch_uncached", network_fetch)

    first = await service._fetch_from_provider(
        provider="okx", symbol="BTC-USDT", interval="1h", lookback=2
    )
    second = await service._fetch_from_provider(
        provider="okx", symbol="BTC-USDT", interval="1h", lookback=2
    )

    network_fetch.assert_awaited_once_with("okx", "BTC-USDT", "1h", 2)
    assert second == first
    assert second.exchange_symbol == "BTC/USDT"


@pytest.mark.asyncio
async def test_failed_provider_enters_cooldown_and_fallback_skips_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("broken", "binance"))
    attempted_providers: list[str] = []

    async def fetch_network(
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
    ) -> CandleFetchResult:
        attempted_providers.append(provider)
        if provider == "broken":
            raise RuntimeError("primary unavailable")
        return _fetch_result(provider, symbol, interval, lookback)

    monkeypatch.setattr(service, "_fetch_uncached", AsyncMock(side_effect=fetch_network))

    first = await service.get_indicators(
        symbols=["BTC-USDT"],
        interval="15m",
        lookback=2,
        providers=["broken", "binance"],
    )
    second = await service.get_indicators(
        symbols=["BTC-USDT"],
        interval="15m",
        lookback=3,
        providers=["broken", "binance"],
    )

    health_by_provider = {
        provider["provider"]: provider for provider in service.get_health()["providers"]
    }
    assert attempted_providers == ["broken", "binance", "binance"]
    assert first.symbols[0].provider == "binance"
    assert second.symbols[0].provider == "binance"
    assert health_by_provider["broken"]["status"] == "cooldown"
    assert health_by_provider["broken"]["consecutive_failures"] == 1
    assert health_by_provider["broken"]["last_error"] == "primary unavailable"
    assert health_by_provider["broken"]["cooldown_remaining_s"] > 0


def test_rule_strategy_config_normalizes_symbols_and_preserves_schedule() -> None:
    config = RuleStrategyConfig(
        symbols=[" btc/usdt ", "BTC-USDT"], interval="15m", decide_interval_s=120
    )

    assert config.symbols == ["BTC-USDT"]
    assert config.interval == "15m"
    assert config.decide_interval_s == 120


def test_rule_strategy_config_defaults_market_schedule() -> None:
    config = RuleStrategyConfig()

    assert config.symbols == ["BTC-USDT"]
    assert config.interval == "1h"
    assert config.decide_interval_s is None
