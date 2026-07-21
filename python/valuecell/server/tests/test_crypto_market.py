from datetime import datetime, timezone

import asyncio
import math
from unittest.mock import AsyncMock, call

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from valuecell.server.api.routers.crypto_market import create_crypto_market_router
from valuecell.server.config.settings import get_settings
from valuecell.server.api.schemas.crypto_market import (
    BollingerBandData,
    CryptoCandleData,
    CryptoIndicatorPointData,
    CryptoMarketIndicatorsData,
    CryptoSymbolIndicatorsData,
)
from valuecell.server.services import crypto_market_service as crypto_market_module
from valuecell.server.services.crypto_market_service import (
    CandleFetchResult,
    CryptoMarketService,
    DefaultMarketSnapshot,
)


OKX_DELISTED_DEFAULT_SYMBOLS = {
    "MATIC-USDT",
    "FTM-USDT",
    "CAKE-USDT",
    "EOS-USDT",
    "WBTC-USDT",
    "MKR-USDT",
    "KAVA-USDT",
    "OMG-USDT",
    "VET-USDT",
    "WAVES-USDT",
    "OCEAN-USDT",
    "REQ-USDT",
    "GNO-USDT",
    "BAL-USDT",
    "SPELL-USDT",
    "AUDIO-USDT",
    "COTI-USDT",
    "HOT-USDT",
    "KEY-USDT",
    "LOKA-USDT",
    "MBL-USDT",
    "NKN-USDT",
    "OAX-USDT",
    "RIF-USDT",
    "SXP-USDT",
}


def test_default_symbol_catalog_excludes_okx_delisted_markets():
    assert not (
        OKX_DELISTED_DEFAULT_SYMBOLS
        & set(crypto_market_module.SUPPORTED_CRYPTO_SYMBOLS)
    )


@pytest.fixture(autouse=True)
def reset_market_provider_attempts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VALUECELL_MARKET_DATA_PROVIDER_ATTEMPTS", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


BASE_TS = 1_700_000_000_000


def test_public_candle_requests_send_stable_json_headers(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"code":"0","data":[["1","1","1","1","1","1"]]}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(crypto_market_module, "urlopen", fake_urlopen)

    candles = CryptoMarketService()._fetch_rest_candle_page(
        "okx", "BTC-USDT", "1m", 1, None, None
    )

    headers = {key.lower(): value for key, value in captured["request"].header_items()}
    assert headers["user-agent"] == "valuecell-market-data/1.0"
    assert headers["accept"] == "application/json"
    assert len(candles) == 1


def test_symbol_specific_provider_error_does_not_trip_global_circuit_breaker(monkeypatch):
    service = CryptoMarketService()
    service.providers = ("mexc",)
    service._provider_health.clear()
    service._provider_health["mexc"] = crypto_market_module.ProviderHealth()

    async def unsupported_symbol(**_kwargs):
        raise RuntimeError("HTTP Error 400: Bad Request")

    monkeypatch.setattr(service, "_fetch_from_provider", unsupported_symbol)

    with pytest.raises(RuntimeError, match="unsupported_symbol"):
        asyncio.run(
            service._fetch_with_fallback(
                symbol="MATIC-USDT",
                interval="15m",
                lookback=240,
                providers=["mexc"],
            )
        )

    health = service._provider_health["mexc"]
    assert health.consecutive_failures == 0
    assert health.cooldown_until == 0


def _candles(*, start: float = 100.0, count: int = 80) -> list[CryptoCandleData]:
    return [
        CryptoCandleData(
            ts=BASE_TS + index * 60_000,
            open=start + index - 0.5,
            high=start + index + 1.0,
            low=start + index - 1.0,
            close=start + index,
            volume=1_000.0 + index,
        )
        for index in range(count)
    ]


def _sample_market_data() -> CryptoMarketIndicatorsData:
    candles = _candles(count=2)
    return CryptoMarketIndicatorsData(
        interval="15m",
        lookback=2,
        providers=["okx"],
        symbols=[
            CryptoSymbolIndicatorsData(
                symbol="BTC-USDT",
                exchange_symbol="BTC/USDT",
                provider="okx",
                interval="15m",
                candles=candles,
                indicators=[
                    CryptoIndicatorPointData(
                        ts=candles[-1].ts,
                        ma={"ma5": None, "ma10": None, "ma20": None, "ma60": None},
                        rsi=None,
                        bollinger=BollingerBandData(),
                        momentum=None,
                        macd=0.25,
                        macd_signal=0.20,
                        macd_histogram=0.05,
                    )
                ],
                latest_price=candles[-1].close,
            )
        ],
        failed_symbols={},
    )


@pytest.mark.asyncio
async def test_crypto_market_service_returns_candles_and_core_indicators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    candles = _candles()

    async def fake_fetch_uncached(
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
    ) -> CandleFetchResult:
        assert provider == "okx"
        assert symbol == "BTC-USDT"
        assert interval == "15m"
        assert lookback == 80
        return CandleFetchResult(
            symbol=symbol,
            exchange_symbol="BTC/USDT",
            provider=provider,
            candles=candles,
        )

    CryptoMarketService._cache.clear()
    monkeypatch.setattr(service, "_fetch_uncached", fake_fetch_uncached)

    result = await service.get_indicators(
        symbols=["BTC-USDT"],
        interval="15m",
        lookback=80,
        providers=["okx"],
    )

    assert result.failed_symbols == {}
    assert result.interval == "15m"
    assert result.lookback == 80
    assert result.providers == ["okx"]

    btc = result.symbols[0]
    assert btc.symbol == "BTC-USDT"
    assert btc.exchange_symbol == "BTC/USDT"
    assert btc.provider == "okx"
    assert btc.latest_price == candles[-1].close
    assert btc.candles == candles
    assert len(btc.indicators) == len(candles)

    latest = btc.indicators[-1]
    closes = [item.close for item in candles]
    assert latest.ma["ma5"] == pytest.approx(sum(closes[-5:]) / 5)
    assert latest.ma["ma10"] == pytest.approx(sum(closes[-10:]) / 10)
    assert latest.ma["ma20"] == pytest.approx(sum(closes[-20:]) / 20)
    assert latest.ma["ma60"] == pytest.approx(sum(closes[-60:]) / 60)
    assert latest.rsi == pytest.approx(100.0)
    assert latest.momentum == pytest.approx(closes[-1] - closes[-15])

    expected_bb_middle = sum(closes[-20:]) / 20
    expected_bb_std = math.sqrt(
        sum((close - expected_bb_middle) ** 2 for close in closes[-20:]) / 19
    )
    assert latest.bollinger.middle == pytest.approx(expected_bb_middle)
    assert latest.bollinger.upper == pytest.approx(expected_bb_middle + expected_bb_std * 2)
    assert latest.bollinger.lower == pytest.approx(expected_bb_middle - expected_bb_std * 2)
    assert latest.macd is not None
    assert latest.macd_signal is not None
    assert latest.macd_histogram == pytest.approx(latest.macd - latest.macd_signal)


@pytest.mark.asyncio
async def test_crypto_market_service_normalizes_and_returns_multiple_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    seen_symbols: list[str] = []

    async def fake_fetch_uncached(
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
    ) -> CandleFetchResult:
        seen_symbols.append(symbol)
        start = 200.0 if symbol == "ETH-USDT" else 100.0
        return CandleFetchResult(
            symbol=symbol,
            exchange_symbol=symbol.replace("-", "/"),
            provider=provider,
            candles=_candles(start=start, count=lookback),
        )

    CryptoMarketService._cache.clear()
    monkeypatch.setattr(service, "_fetch_uncached", fake_fetch_uncached)

    result = await service.get_indicators(
        symbols=["btc/usdt", "ETH-USDT"],
        interval="1h",
        lookback=65,
        providers=["okx"],
    )

    assert seen_symbols == ["BTC-USDT", "ETH-USDT"]
    assert [item.symbol for item in result.symbols] == ["BTC-USDT", "ETH-USDT"]
    assert [item.exchange_symbol for item in result.symbols] == ["BTC/USDT", "ETH/USDT"]
    assert {item.latest_price for item in result.symbols} == {164.0, 264.0}
    assert result.failed_symbols == {}


@pytest.mark.asyncio
async def test_crypto_market_service_rejects_unsupported_symbols_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    fetch = AsyncMock()
    monkeypatch.setattr(service, "_fetch_uncached", fetch)

    with pytest.raises(ValueError, match="Unsupported crypto symbol: AAPL-USDT"):
        await service.get_indicators(
            symbols=["AAPL-USDT"],
            interval="15m",
            lookback=80,
            providers=["okx"],
        )

    fetch.assert_not_called()


@pytest.mark.asyncio
async def test_crypto_market_service_falls_back_to_second_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("broken", "binance"))
    calls: list[tuple[str, str]] = []

    async def fake_fetch_uncached(
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
    ) -> CandleFetchResult:
        calls.append((provider, symbol))
        if provider == "broken":
            raise RuntimeError("primary provider down")
        return CandleFetchResult(
            symbol=symbol,
            exchange_symbol="BTC/USDT",
            provider=provider,
            candles=_candles(count=lookback),
        )

    CryptoMarketService._cache.clear()
    monkeypatch.setattr(service, "_fetch_uncached", fake_fetch_uncached)

    result = await service.get_indicators(
        symbols=["BTC-USDT"],
        interval="5m",
        lookback=70,
        providers=["broken", "binance"],
    )

    assert calls == [
        ("broken", "BTC-USDT"),
        ("broken", "BTC-USDT"),
        ("binance", "BTC-USDT"),
    ]
    assert result.failed_symbols == {}
    assert result.providers == ["broken", "binance"]
    assert result.symbols[0].provider == "binance"
    assert result.symbols[0].candles[-1].close == 169.0


@pytest.mark.asyncio
async def test_crypto_market_service_reports_symbol_fetch_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))

    async def fake_fetch_uncached(
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
    ) -> CandleFetchResult:
        if symbol == "ETH-USDT":
            raise RuntimeError("exchange returned no candles")
        return CandleFetchResult(
            symbol=symbol,
            exchange_symbol="BTC/USDT",
            provider=provider,
            candles=_candles(count=lookback),
        )

    CryptoMarketService._cache.clear()
    monkeypatch.setattr(service, "_fetch_uncached", fake_fetch_uncached)

    result = await service.get_indicators(
        symbols=["BTC-USDT", "ETH-USDT"],
        interval="5m",
        lookback=70,
        providers=["okx"],
    )

    assert [item.symbol for item in result.symbols] == ["BTC-USDT"]
    assert result.failed_symbols == {
        "ETH-USDT": "okx: empty_response; okx: empty_response",
    }


@pytest.mark.asyncio
async def test_historical_fetch_failure_does_not_open_shared_provider_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))

    async def failing_history_fetch(
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
        time_range: tuple[int | None, int | None],
    ) -> CandleFetchResult:
        raise RuntimeError("historical range unavailable")

    CryptoMarketService._provider_health.clear()
    monkeypatch.setattr(service, "_fetch_history_from_provider", failing_history_fetch)

    result = await service.get_indicators(
        symbols=["BTC-USDT"],
        interval="1h",
        lookback=240,
        providers=["okx"],
        from_ts_ms=BASE_TS,
        to_ts_ms=BASE_TS + 60_000,
    )

    assert result.symbols == []
    assert result.failed_symbols["BTC-USDT"] == (
        "okx: provider_error; okx: provider_error"
    )
    assert CryptoMarketService._provider_health["okx"].consecutive_failures == 0


def test_mexc_history_retrieval_pages_public_rest_candles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("mexc",))
    interval_ms = 3_600_000
    from_ts_ms = BASE_TS
    to_ts_ms = BASE_TS + interval_ms * 1_000
    calls: list[tuple[int | None, int | None]] = []

    def fake_page(
        provider: str,
        symbol: str,
        interval: str,
        limit: int,
        page_from_ts_ms: int | None,
        page_to_ts_ms: int | None,
    ) -> list[CryptoCandleData]:
        assert provider == "mexc"
        assert symbol == "BTC-USDT"
        assert interval == "1h"
        assert limit == 1_000
        calls.append((page_from_ts_ms, page_to_ts_ms))
        assert page_from_ts_ms is not None
        assert page_to_ts_ms is not None
        return [
            CryptoCandleData(
                ts=page_from_ts_ms,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1_000.0,
            ),
            CryptoCandleData(
                ts=page_to_ts_ms,
                open=101.0,
                high=102.0,
                low=100.0,
                close=101.5,
                volume=1_000.0,
            ),
        ]

    monkeypatch.setattr(service, "_fetch_rest_candle_page", fake_page)

    result = service._fetch_provider_candles(
        "mexc", "BTC-USDT", "1h", 1_001, from_ts_ms, to_ts_ms
    )

    assert calls == [
        (from_ts_ms, from_ts_ms + interval_ms * 999),
        (from_ts_ms + interval_ms * 1_000, to_ts_ms),
    ]
    assert result.exchange_symbol == "BTC/USDT"
    assert [candle.ts for candle in result.candles] == [
        from_ts_ms,
        from_ts_ms + interval_ms * 999,
        to_ts_ms,
    ]


def test_crypto_market_router_returns_success_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AsyncMock()
    service.get_indicators = AsyncMock(return_value=_sample_market_data())
    monkeypatch.setattr(
        "valuecell.server.api.routers.crypto_market.get_crypto_market_service",
        lambda: service,
    )
    app = FastAPI()
    app.include_router(create_crypto_market_router())

    response = TestClient(app).get(
        "/crypto-market/indicators",
        params={
            "symbols": "BTC-USDT, ETH-USDT",
            "interval": "15m",
            "lookback": 2,
            "providers": "okx, binance",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["msg"] == "Crypto indicators retrieved successfully"
    assert body["data"]["symbols"][0]["symbol"] == "BTC-USDT"
    service.get_indicators.assert_awaited_once_with(
        symbols=["BTC-USDT", "ETH-USDT"],
        interval="15m",
        lookback=2,
        providers=["okx", "binance"],
    )


@pytest.mark.asyncio
async def test_default_snapshot_refresh_preserves_previous_snapshot_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    sample = _sample_market_data()
    complete_snapshot_data = sample.model_copy(
        update={
            "symbols": [
                sample.symbols[0].model_copy(update={"symbol": symbol})
                for symbol in ("BTC-USDT", "ETH-USDT", "SOL-USDT")
            ]
        }
    )
    get_indicators = AsyncMock(
        side_effect=[complete_snapshot_data, RuntimeError("public market unavailable")]
    )
    monkeypatch.setattr(service, "get_indicators", get_indicators)

    first_snapshot = await service.refresh_default_snapshot()
    retained_snapshot = await service.refresh_default_snapshot()

    assert retained_snapshot is first_snapshot
    assert retained_snapshot.data.symbols == complete_snapshot_data.symbols
    assert retained_snapshot.data.failed_symbols == complete_snapshot_data.failed_symbols
    assert get_indicators.await_args_list == [
        call(
            symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
            interval="1h",
            lookback=240,
        ),
        call(
            symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
            interval="1h",
            lookback=240,
        ),
    ]


@pytest.mark.asyncio
async def test_default_snapshot_refresh_merges_partial_refresh_with_previous_candles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    sample = _sample_market_data()
    initial_symbols = [
        sample.symbols[0].model_copy(
            update={
                "symbol": symbol,
                "candles": _candles(start=start, count=2),
                "latest_price": start + 1,
            }
        )
        for symbol, start in (
            ("BTC-USDT", 100.0),
            ("ETH-USDT", 200.0),
            ("SOL-USDT", 300.0),
        )
    ]
    initial_data = sample.model_copy(update={"symbols": initial_symbols})
    fresh_btc = sample.symbols[0].model_copy(
        update={
            "symbol": "BTC-USDT",
            "candles": _candles(start=400.0, count=2),
            "latest_price": 401.0,
        }
    )
    partial_data = sample.model_copy(
        update={
            "symbols": [fresh_btc],
            "failed_symbols": {
                "ETH-USDT": "okx: provider unavailable",
                "SOL-USDT": "okx: provider unavailable",
            },
        }
    )
    monkeypatch.setattr(
        service,
        "get_indicators",
        AsyncMock(side_effect=[initial_data, partial_data]),
    )

    await service.refresh_default_snapshot()
    refreshed_snapshot = await service.refresh_default_snapshot()

    assert refreshed_snapshot is not None
    symbols = {item.symbol: item for item in refreshed_snapshot.data.symbols}
    assert [item.symbol for item in refreshed_snapshot.data.symbols] == [
        "BTC-USDT",
        "ETH-USDT",
        "SOL-USDT",
    ]
    assert symbols["BTC-USDT"] == fresh_btc
    assert symbols["ETH-USDT"].candles == initial_symbols[1].candles
    assert symbols["SOL-USDT"].candles == initial_symbols[2].candles
    assert refreshed_snapshot.data.failed_symbols == partial_data.failed_symbols


def test_crypto_market_router_serves_default_symbol_subset_from_shared_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CryptoMarketService(providers=("okx",))
    service._default_snapshot = DefaultMarketSnapshot(
        data=_sample_market_data(), fetched_at=datetime.now(timezone.utc)
    )
    service.get_indicators = AsyncMock(
        side_effect=AssertionError("default snapshot request must not fetch")
    )
    monkeypatch.setattr(
        "valuecell.server.api.routers.crypto_market.get_crypto_market_service",
        lambda: service,
    )
    app = FastAPI()
    app.include_router(create_crypto_market_router())

    response = TestClient(app).get(
        "/crypto-market/indicators",
        params={
            "symbols": "BTC-USDT",
            "interval": "1h",
            "lookback": 2,
        },
    )

    assert response.status_code == 200
    assert response.headers["X-ValueCell-Market-Cache"] == "default-snapshot"
    body = response.json()
    assert body["msg"] == "Crypto indicators retrieved from shared market snapshot"
    assert [item["symbol"] for item in body["data"]["symbols"]] == ["BTC-USDT"]
    assert body["data"]["failed_symbols"] == {}
    service.get_indicators.assert_not_awaited()



@pytest.mark.asyncio
async def test_default_snapshot_persists_and_restores_after_service_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    snapshot_path = tmp_path / "market_snapshot.json"
    monkeypatch.setattr(
        CryptoMarketService,
        "_snapshot_path",
        staticmethod(lambda: snapshot_path),
    )
    sample = _sample_market_data()
    service = CryptoMarketService(providers=("okx",))
    monkeypatch.setattr(service, "get_indicators", AsyncMock(return_value=sample))

    persisted_snapshot = await service.refresh_default_snapshot()
    restored_snapshot = CryptoMarketService(providers=("okx",)).get_default_snapshot()

    assert persisted_snapshot is not None
    assert snapshot_path.is_file()
    assert restored_snapshot == persisted_snapshot


def test_default_snapshot_ignores_corrupt_persisted_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    snapshot_path = tmp_path / "market_snapshot.json"
    snapshot_path.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr(
        CryptoMarketService,
        "_snapshot_path",
        staticmethod(lambda: snapshot_path),
    )

    service = CryptoMarketService(providers=("okx",))

    assert service.get_default_snapshot() is None

def test_crypto_market_router_returns_400_for_invalid_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AsyncMock()
    service.get_indicators = AsyncMock(
        side_effect=ValueError("Unsupported crypto symbol: AAPL-USDT")
    )
    monkeypatch.setattr(
        "valuecell.server.api.routers.crypto_market.get_crypto_market_service",
        lambda: service,
    )
    app = FastAPI()
    app.include_router(create_crypto_market_router())

    response = TestClient(app).get(
        "/crypto-market/indicators",
        params={"symbols": "AAPL-USDT", "interval": "15m", "lookback": 2},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported crypto symbol: AAPL-USDT"


@pytest.mark.parametrize("interval", ["1w", "1M", "3M", "1Y"])
def test_crypto_market_router_forwards_historical_range_and_aggregation_interval(
    monkeypatch: pytest.MonkeyPatch, interval: str
) -> None:
    service = AsyncMock()
    service.get_indicators = AsyncMock(return_value=_sample_market_data())
    monkeypatch.setattr(
        "valuecell.server.api.routers.crypto_market.get_crypto_market_service",
        lambda: service,
    )
    app = FastAPI()
    app.include_router(create_crypto_market_router())

    response = TestClient(app).get(
        "/crypto-market/indicators",
        params={
            "symbols": "BTC-USDT",
            "interval": interval,
            "lookback": 24,
            "from_ts_ms": 1_700_000_000_000,
            "to_ts_ms": 1_710_000_000_000,
            "providers": "okx",
        },
    )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    service.get_indicators.assert_awaited_once_with(
        symbols=["BTC-USDT"],
        interval=interval,
        lookback=24,
        providers=["okx"],
        from_ts_ms=1_700_000_000_000,
        to_ts_ms=1_710_000_000_000,
    )
