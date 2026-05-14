import pytest

from valuecell.agents.common.trading.models import MarketType
from valuecell.agents.spot_rsi_ladder_agent import symbol_filter


class _FakeExchange:
    def __init__(self, config, markets=None, error=None):
        self._markets = markets or {}
        self._error = error
        self.closed = False

    async def load_markets(self):
        if self._error is not None:
            raise self._error
        return self._markets

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_filter_supported_spot_symbols_keeps_only_spot_pairs(monkeypatch) -> None:
    markets = {
        "BTC/USDT": {
            "symbol": "BTC/USDT",
            "base": "BTC",
            "quote": "USDT",
            "spot": True,
            "id": "BTCUSDT",
        },
        "ETH/USDT": {
            "symbol": "ETH/USDT",
            "base": "ETH",
            "quote": "USDT",
            "spot": True,
            "id": "ETHUSDT",
        },
        "BTC/USDT:USDT": {
            "symbol": "BTC/USDT:USDT",
            "base": "BTC",
            "quote": "USDT",
            "spot": False,
            "id": "BTCUSDT",
        },
    }

    def _factory(_exchange_id: str):
        return lambda config: _FakeExchange(config, markets=markets)

    monkeypatch.setattr(symbol_filter, "get_exchange_cls", _factory)

    filtered = await symbol_filter.filter_supported_spot_symbols(
        "binance",
        ["BTC-USDT", "ETH-USDT", "MIR-USDT"],
        MarketType.SPOT,
    )

    assert filtered == ["BTC-USDT", "ETH-USDT"]


@pytest.mark.asyncio
async def test_filter_supported_spot_symbols_falls_back_on_market_load_error(
    monkeypatch,
) -> None:
    def _factory(_exchange_id: str):
        return lambda config: _FakeExchange(
            config,
            error=RuntimeError("network blocked"),
        )

    monkeypatch.setattr(symbol_filter, "get_exchange_cls", _factory)

    original = ["BTC-USDT", "MIR-USDT"]
    filtered = await symbol_filter.filter_supported_spot_symbols(
        "okx",
        original,
        MarketType.SPOT,
    )

    assert filtered == original


@pytest.mark.asyncio
async def test_filter_supported_spot_symbols_skips_filter_for_non_spot() -> None:
    original = ["BTC-USDT", "ETH-USDT"]
    filtered = await symbol_filter.filter_supported_spot_symbols(
        "binance",
        original,
        MarketType.SWAP,
    )

    assert filtered == original
