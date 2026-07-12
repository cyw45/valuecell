from unittest.mock import AsyncMock

import pytest

from valuecell.server.api.schemas.crypto_market import CryptoCandleData
from valuecell.server.config.settings import get_settings
from valuecell.server.services.crypto_market_service import (
    CandleFetchResult,
    CryptoMarketService,
)


@pytest.fixture(autouse=True)
def clear_crypto_market_service_state():
    CryptoMarketService._cache.clear()
    CryptoMarketService._inflight.clear()
    CryptoMarketService._provider_health.clear()
    yield
    CryptoMarketService._cache.clear()
    CryptoMarketService._inflight.clear()
    CryptoMarketService._provider_health.clear()


@pytest.fixture(autouse=True)
def reset_market_provider_attempts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VALUECELL_MARKET_DATA_PROVIDER_ATTEMPTS", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _real_provider_result() -> CandleFetchResult:
    return CandleFetchResult(
        symbol="BTC-USDT",
        exchange_symbol="BTC/USDT",
        provider="okx",
        candles=[
            CryptoCandleData(
                ts=1_700_000_000_000,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1_000.0,
            )
        ],
    )


@pytest.mark.asyncio
async def test_serves_last_successful_provider_candles_during_failure_and_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    first_result = _real_provider_result()
    fetch = AsyncMock(side_effect=[first_result, RuntimeError("temporary outage")])
    monkeypatch.setattr(service, "_fetch_uncached", fetch)

    initial = await service._fetch_with_fallback(
        symbol="BTC-USDT", interval="1h", lookback=1, providers=["okx"]
    )
    cache_key = ("okx", "BTC-USDT", "1h", 1)
    CryptoMarketService._cache[cache_key].expires_at = 0.0

    after_failure = await service._fetch_with_fallback(
        symbol="BTC-USDT", interval="1h", lookback=1, providers=["okx"]
    )
    during_cooldown = await service._fetch_with_fallback(
        symbol="BTC-USDT", interval="1h", lookback=1, providers=["okx"]
    )

    assert initial is first_result
    assert after_failure is first_result
    assert during_cooldown is first_result
    assert fetch.await_count == 2


@pytest.mark.asyncio
async def test_provider_failure_without_prior_candle_cache_is_surfaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    monkeypatch.setattr(
        service,
        "_fetch_uncached",
        AsyncMock(side_effect=RuntimeError("temporary outage")),
    )

    with pytest.raises(RuntimeError, match=r"okx: temporary outage"):
        await service._fetch_with_fallback(
            symbol="BTC-USDT", interval="1h", lookback=1, providers=["okx"]
        )

    assert CryptoMarketService._cache == {}
