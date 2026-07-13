"""Crypto-only market data and indicator service for strategy dashboards."""

from __future__ import annotations

import asyncio
import time
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from loguru import logger
from valuecell.server.config.settings import get_settings
from valuecell.utils.env import ensure_system_env_dir

from valuecell.server.api.schemas.crypto_market import (
    BollingerBandData,
    CryptoCandleData,
    CryptoIndicatorPointData,
    CryptoMarketIndicatorsData,
    CryptoSymbolCatalogData,
    CryptoSymbolIndicatorsData,
)

DEFAULT_PROVIDERS: tuple[str, ...] = ("gate", "okx", "binance", "mexc")
SOURCE_INTERVALS: set[str] = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"}
AGGREGATED_INTERVALS: set[str] = {"1w", "1M", "3M", "1Y"}
SUPPORTED_INTERVALS = SOURCE_INTERVALS | AGGREGATED_INTERVALS
AGGREGATION_SOURCE_INTERVAL: dict[str, str] = {
    "1w": "1d",
    "1M": "1d",
    "3M": "1d",
    "1Y": "1d",
}
DEFAULT_LOOKBACK = 240
MAX_LOOKBACK = 5_000
CACHE_TTL_S = 12.0
FETCH_TIMEOUT_S = 8.0
MAX_GATE_CANDLES_PER_REQUEST = 1_000
MAX_CONCURRENT_FETCHES = 6
MA_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)
RSI_WINDOW = 14
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
MOMENTUM_WINDOW = 14
CANDLE_FRESHNESS_MAX_AGE_S = 90

INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1_800,
    "1h": 3_600, "4h": 14_400, "1d": 86_400, "1w": 604_800,
    "1M": 2_592_000, "3M": 7_776_000, "1Y": 31_536_000,
}

REST_PROVIDER_PAGE_LIMITS: dict[str, int] = {
    "gate": 1_000,
    "binance": 1_000,
    "mexc": 1_000,
    "okx": 300,
}
REST_PROVIDER_INTERVALS: dict[str, dict[str, str]] = {
    "binance": {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"},
    "mexc": {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "4h": "4h", "1d": "1d"},
    "okx": {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H", "1d": "1Dutc"},
}

MARKET_SNAPSHOT_FILENAME = "market_snapshot.json"

SUPPORTED_CRYPTO_SYMBOLS: tuple[str, ...] = (

    "BTC-USDT",
    "ETH-USDT",
    "BNB-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "ADA-USDT",
    "DOGE-USDT",
    "DOT-USDT",
    "USDC-USDT",
    "LTC-USDT",
    "BCH-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "MATIC-USDT",
    "POL-USDT",
    "UNI-USDT",
    "ATOM-USDT",
    "ETC-USDT",
    "FIL-USDT",
    "AAVE-USDT",
    "SAND-USDT",
    "MANA-USDT",
    "ALGO-USDT",
    "FTM-USDT",
    "NEAR-USDT",
    "GRT-USDT",
    "CAKE-USDT",
    "XLM-USDT",
    "EOS-USDT",
    "TRX-USDT",
    "WBTC-USDT",
    "ARB-USDT",
    "OP-USDT",
    "MKR-USDT",
    "SNX-USDT",
    "CRV-USDT",
    "1INCH-USDT",
    "KAVA-USDT",
    "ZRX-USDT",
    "BAT-USDT",
    "OMG-USDT",
    "QTUM-USDT",
    "ICX-USDT",
    "VET-USDT",
    "THETA-USDT",
    "NEO-USDT",
    "ONT-USDT",
    "ZIL-USDT",
    "RVN-USDT",
    "DASH-USDT",
    "HBAR-USDT",
    "IOTA-USDT",
    "WAVES-USDT",
    "KSM-USDT",
    "RSR-USDT",
    "CELR-USDT",
    "FET-USDT",
    "OCEAN-USDT",
    "REQ-USDT",
    "BNT-USDT",
    "LRC-USDT",
    "GNO-USDT",
    "PAXG-USDT",
    "UMA-USDT",
    "BAL-USDT",
    "SPELL-USDT",
    "AUDIO-USDT",
    "RAY-USDT",
    "CELO-USDT",
    "MASK-USDT",
    "COTI-USDT",
    "CHZ-USDT",
    "ENJ-USDT",
    "GAS-USDT",
    "HOT-USDT",
    "IOST-USDT",
    "KEY-USDT",
    "LOKA-USDT",
    "MBL-USDT",
    "NKN-USDT",
    "OAX-USDT",
    "RIF-USDT",
    "SXP-USDT",
)


@dataclass(frozen=True)
class CandleFetchResult:
    symbol: str
    exchange_symbol: str
    provider: str
    candles: list[CryptoCandleData]


@dataclass
class CacheEntry:
    expires_at: float
    result: CandleFetchResult


@dataclass
class ProviderHealth:
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_success_at: str | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class DefaultMarketSnapshot:
    data: CryptoMarketIndicatorsData
    fetched_at: datetime


class CryptoMarketService:
    """Fetch crypto OHLCV data with provider fallback and compute indicators."""

    _cache: ClassVar[dict[tuple[str, str, str, int], CacheEntry]] = {}
    _inflight: ClassVar[dict[tuple[str, str, str, int], asyncio.Task[CandleFetchResult]]] = {}
    _provider_health: ClassVar[dict[str, ProviderHealth]] = {}
    _semaphore: ClassVar[asyncio.Semaphore | None] = None
    _semaphore_limit: ClassVar[int | None] = None
    _symbol_set: ClassVar[set[str]] = set(SUPPORTED_CRYPTO_SYMBOLS)

    def __init__(self, providers: tuple[str, ...] | None = None) -> None:
        settings = get_settings()
        self.providers = providers or settings.MARKET_DATA_PROVIDERS
        self._default_snapshot = self._load_default_snapshot()
        if self.__class__._semaphore_limit != settings.MARKET_DATA_MAX_CONCURRENT_FETCHES:
            self.__class__._semaphore = asyncio.Semaphore(
                settings.MARKET_DATA_MAX_CONCURRENT_FETCHES
            )
            self.__class__._semaphore_limit = settings.MARKET_DATA_MAX_CONCURRENT_FETCHES

    async def refresh_default_snapshot(self) -> DefaultMarketSnapshot | None:
        """Refresh each default symbol without discarding prior real candles."""
        settings = get_settings()
        try:
            data = await self.get_indicators(
                symbols=list(settings.MARKET_DEFAULT_SYMBOLS),
                interval=settings.MARKET_DEFAULT_INTERVAL,
                lookback=settings.MARKET_DEFAULT_LOOKBACK,
            )
        except Exception as exc:
            logger.warning("Crypto market snapshot refresh failed err={}", exc)
            return self._default_snapshot

        previous_symbols = (
            {item.symbol: item for item in self._default_snapshot.data.symbols}
            if self._default_snapshot is not None
            else {}
        )
        refreshed_symbols = {item.symbol: item for item in data.symbols}
        merged_symbols = [
            refreshed_symbols.get(symbol) or previous_symbols[symbol]
            for symbol in settings.MARKET_DEFAULT_SYMBOLS
            if symbol in refreshed_symbols or symbol in previous_symbols
        ]
        if not merged_symbols:
            logger.warning(
                "Crypto market snapshot refresh has no usable candles failures={}",
                data.failed_symbols,
            )
            return self._default_snapshot

        snapshot = DefaultMarketSnapshot(
            data=CryptoMarketIndicatorsData(
                interval=data.interval,
                lookback=data.lookback,
                providers=data.providers,
                symbols=merged_symbols,
                failed_symbols=data.failed_symbols,
                snapshot_fetched_at=datetime.now(timezone.utc).isoformat(),
            ),
            fetched_at=datetime.now(timezone.utc),
        )
        self._default_snapshot = snapshot
        self._persist_default_snapshot(snapshot)
        if data.failed_symbols:
            logger.warning(
                "Crypto market snapshot refreshed with retained symbols failures={}",
                data.failed_symbols,
            )
        return snapshot

    @staticmethod
    def _snapshot_path() -> Path:
        return ensure_system_env_dir() / MARKET_SNAPSHOT_FILENAME

    def _load_default_snapshot(self) -> DefaultMarketSnapshot | None:
        path = self._snapshot_path()
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            data = CryptoMarketIndicatorsData.model_validate(raw["data"])
            fetched_at = datetime.fromisoformat(raw["fetched_at"])
            logger.info("Loaded persisted public market snapshot from {}", path)
            return DefaultMarketSnapshot(data=data, fetched_at=fetched_at)
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable public market snapshot {}: {}", path, exc)
            return None

    def _persist_default_snapshot(self, snapshot: DefaultMarketSnapshot) -> None:
        path = self._snapshot_path()
        temporary_path = path.with_suffix(".tmp")
        payload = {
            "fetched_at": snapshot.fetched_at.isoformat(),
            "data": snapshot.data.model_dump(mode="json"),
        }
        try:
            temporary_path.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary_path, path)
        except OSError as exc:
            logger.warning("Could not persist public market snapshot {}: {}", path, exc)

    def get_default_snapshot(self) -> DefaultMarketSnapshot | None:
        """Return the latest successful server-owned public market snapshot."""
        return self._default_snapshot

    def get_health(self) -> dict[str, object]:
        """Return non-sensitive public OHLCV provider state for diagnostics."""
        now = time.monotonic()
        return {
            "providers": [
                {
                    "provider": provider,
                    "status": "cooldown"
                    if self._provider_health.get(provider, ProviderHealth()).cooldown_until > now
                    else "ready",
                    "consecutive_failures": self._provider_health.get(
                        provider, ProviderHealth()
                    ).consecutive_failures,
                    "cooldown_remaining_s": max(
                        0.0,
                        self._provider_health.get(provider, ProviderHealth()).cooldown_until - now,
                    ),
                    "last_success_at": self._provider_health.get(
                        provider, ProviderHealth()
                    ).last_success_at,
                    "last_error": self._provider_health.get(
                        provider, ProviderHealth()
                    ).last_error,
                }
                for provider in self.providers
            ],
            "cache_ttl_s": get_settings().MARKET_DATA_CACHE_TTL_S,
            "max_concurrent_fetches": get_settings().MARKET_DATA_MAX_CONCURRENT_FETCHES,
        }

    def get_supported_symbols(self) -> CryptoSymbolCatalogData:
        return CryptoSymbolCatalogData(symbols=list(SUPPORTED_CRYPTO_SYMBOLS))

    async def get_indicators(
        self,
        *,
        symbols: list[str],
        interval: str,
        lookback: int,
        providers: list[str] | None = None,
        from_ts_ms: int | None = None,
        to_ts_ms: int | None = None,
    ) -> CryptoMarketIndicatorsData:
        normalized_symbols = self._normalize_symbols(symbols)
        normalized_interval = self._normalize_interval(interval)
        normalized_lookback = self._normalize_lookback(lookback)
        time_range = self._normalize_time_range(from_ts_ms, to_ts_ms)
        provider_pool = self._normalize_providers(providers)
        source_interval = AGGREGATION_SOURCE_INTERVAL.get(
            normalized_interval, normalized_interval
        )
        source_lookback = (
            MAX_LOOKBACK
            if normalized_interval in AGGREGATED_INTERVALS
            else normalized_lookback
        )
        tasks = [
            self._fetch_symbol_with_indicators(
                symbol=symbol,
                interval=normalized_interval,
                source_interval=source_interval,
                lookback=source_lookback,
                providers=provider_pool,
                time_range=time_range,
            )
            for symbol in normalized_symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        symbol_results: list[CryptoSymbolIndicatorsData] = []
        failed_symbols: dict[str, str] = {}
        for symbol, result in zip(normalized_symbols, results):
            if isinstance(result, Exception):
                failed_symbols[symbol] = str(result)
                continue
            symbol_results.append(result)

        return CryptoMarketIndicatorsData(
            interval=normalized_interval,
            lookback=normalized_lookback,
            providers=provider_pool,
            symbols=symbol_results,
            failed_symbols=failed_symbols,
        )

    async def _fetch_symbol_with_indicators(
        self,
        *,
        symbol: str,
        interval: str,
        source_interval: str,
        lookback: int,
        providers: list[str],
        time_range: tuple[int | None, int | None],
    ) -> CryptoSymbolIndicatorsData:
        fetch_result = await self._fetch_with_fallback(
            symbol=symbol,
            interval=source_interval,
            lookback=lookback,
            providers=providers,
            time_range=time_range,
        )
        candles = self._aggregate_candles(fetch_result.candles, interval)
        if not candles:
            raise RuntimeError("No candles matched the requested time range")
        fetch_result = CandleFetchResult(
            symbol=fetch_result.symbol,
            exchange_symbol=fetch_result.exchange_symbol,
            provider=fetch_result.provider,
            candles=candles,
        )
        indicators = self._compute_indicators(fetch_result.candles)
        latest_price = fetch_result.candles[-1].close if fetch_result.candles else None
        latest_ts_ms = fetch_result.candles[-1].ts if fetch_result.candles else None
        now_ts_ms = int(time.time() * 1000)
        freshness_age_ms = (
            max(0, now_ts_ms - latest_ts_ms) if latest_ts_ms is not None else None
        )
        max_age_ms = INTERVAL_SECONDS[interval] * 2 * 1000
        freshness_status = (
            "fresh"
            if freshness_age_ms is not None and freshness_age_ms <= max_age_ms
            else "stale" if freshness_age_ms is not None else "unknown"
        )

        return CryptoSymbolIndicatorsData(
            symbol=symbol,
            exchange_symbol=fetch_result.exchange_symbol,
            provider=fetch_result.provider,
            interval=interval,
            candles=fetch_result.candles,
            indicators=indicators,
            latest_price=latest_price,
            snapshot_ts_ms=latest_ts_ms,
            freshness_age_ms=freshness_age_ms,
            freshness_status=freshness_status,
        )

    async def _fetch_with_fallback(
        self,
        *,
        symbol: str,
        interval: str,
        lookback: int,
        providers: list[str],
        time_range: tuple[int | None, int | None] = (None, None),
    ) -> CandleFetchResult:
        errors: list[str] = []
        stale_result: CandleFetchResult | None = None
        settings = get_settings()
        is_historical_request = any(time_range)
        for provider in providers:
            health = self._provider_health.setdefault(provider, ProviderHealth())
            cache_key = (provider, symbol, interval, lookback)
            cached = self._cache.get(cache_key)
            if not is_historical_request and health.cooldown_until > time.monotonic():
                errors.append(f"{provider}: cooling down")
                if stale_result is None and cached is not None:
                    stale_result = cached.result
                continue
            for attempt in range(settings.MARKET_DATA_PROVIDER_ATTEMPTS):
                try:
                    result = (
                        await self._fetch_history_from_provider(
                            provider, symbol, interval, lookback, time_range
                        )
                        if any(time_range)
                        else await self._fetch_from_provider(
                            provider=provider,
                            symbol=symbol,
                            interval=interval,
                            lookback=lookback,
                        )
                    )
                    if not is_historical_request:
                        health.consecutive_failures = 0
                        health.cooldown_until = 0.0
                        health.last_error = None
                        health.last_success_at = datetime.now(timezone.utc).isoformat()
                    return result
                except Exception as exc:
                    errors.append(f"{provider}: {exc}")
                    if stale_result is None and cached is not None:
                        stale_result = cached.result
                    if attempt + 1 < settings.MARKET_DATA_PROVIDER_ATTEMPTS:
                        await asyncio.sleep(0.25 * (attempt + 1))
                        continue
                    if not is_historical_request:
                        self._record_provider_failure(provider, exc)
                        logger.warning(
                            "Crypto OHLCV fetch failed provider={} symbol={} interval={} err={}",
                            provider,
                            symbol,
                            interval,
                            str(exc),
                        )
        if stale_result is not None and not any(time_range):
            logger.warning(
                "Serving last successful candles during provider outage symbol={} interval={}",
                symbol,
                interval,
            )
            return stale_result
        raise RuntimeError("; ".join(errors) or "No provider succeeded")
    async def _fetch_history_from_provider(
        self,
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
        time_range: tuple[int | None, int | None],
    ) -> CandleFetchResult:
        return await asyncio.to_thread(
            self._fetch_provider_candles,
            provider,
            symbol,
            interval,
            lookback,
            time_range[0],
            time_range[1],
        )

    async def _fetch_from_provider(
        self,
        *,
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
    ) -> CandleFetchResult:
        cache_key = (provider, symbol, interval, lookback)
        cached = self._cache.get(cache_key)
        now = time.monotonic()
        if cached and cached.expires_at > now:
            return cached.result

        inflight = self._inflight.get(cache_key)
        if inflight is not None:
            return await asyncio.shield(inflight)

        task = asyncio.create_task(
            self._fetch_with_limit(provider, symbol, interval, lookback)
        )
        self._inflight[cache_key] = task
        try:
            result = await asyncio.shield(task)
            self._cache[cache_key] = CacheEntry(
                expires_at=time.monotonic() + get_settings().MARKET_DATA_CACHE_TTL_S,
                result=result,
            )
            return result
        finally:
            if self._inflight.get(cache_key) is task:
                self._inflight.pop(cache_key, None)

    async def _fetch_with_limit(
        self, provider: str, symbol: str, interval: str, lookback: int
    ) -> CandleFetchResult:
        semaphore = self._semaphore
        if semaphore is None:
            raise RuntimeError("Market-data semaphore was not initialized")
        async with semaphore:
            return await asyncio.wait_for(
                self._fetch_uncached(provider, symbol, interval, lookback),
                timeout=FETCH_TIMEOUT_S,
            )

    def _record_provider_failure(self, provider: str, exc: Exception) -> None:
        health = self._provider_health.setdefault(provider, ProviderHealth())
        health.consecutive_failures += 1
        settings = get_settings()
        cooldown_s = min(
            settings.MARKET_DATA_FAILURE_COOLDOWN_MAX_S,
            settings.MARKET_DATA_FAILURE_COOLDOWN_BASE_S
            * (2 ** (health.consecutive_failures - 1)),
        )
        health.cooldown_until = time.monotonic() + cooldown_s
        health.last_error = str(exc)[:500]

    async def _fetch_uncached(
        self,
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
    ) -> CandleFetchResult:
        return await asyncio.to_thread(
            self._fetch_provider_candles,
            provider,
            symbol,
            interval,
            lookback,
            None,
            None,
        )

    def _fetch_provider_candles(
        self,
        provider: str,
        symbol: str,
        interval: str,
        lookback: int,
        from_ts_ms: int | None,
        to_ts_ms: int | None,
    ) -> CandleFetchResult:
        if provider == "gate":
            return self._fetch_gate_public_candles(
                symbol, interval, lookback, from_ts_ms, to_ts_ms
            )
        if provider not in REST_PROVIDER_PAGE_LIMITS:
            raise RuntimeError(f"Unsupported market-data provider: {provider}")
        if interval not in REST_PROVIDER_INTERVALS[provider]:
            raise RuntimeError(f"Provider '{provider}' does not support interval '{interval}'")
        if from_ts_ms is None or to_ts_ms is None:
            candles = self._fetch_rest_candle_page(
                provider,
                symbol,
                interval,
                min(lookback, REST_PROVIDER_PAGE_LIMITS[provider]),
                None,
                None,
            )
        elif provider == "okx":
            candles = self._fetch_rest_candle_page(
                provider,
                symbol,
                interval,
                REST_PROVIDER_PAGE_LIMITS[provider],
                None,
                None,
            )
        else:
            candles = self._fetch_rest_candle_range(
                provider, symbol, interval, from_ts_ms, to_ts_ms
            )
        candles = self._filter_time_range(candles, (from_ts_ms, to_ts_ms))
        if not candles:
            raise RuntimeError("Provider returned no candles for requested range")
        return CandleFetchResult(
            symbol=symbol,
            exchange_symbol=symbol.replace("-", "/"),
            provider=provider,
            candles=candles,
        )

    def _fetch_rest_candle_range(
        self,
        provider: str,
        symbol: str,
        interval: str,
        from_ts_ms: int,
        to_ts_ms: int,
    ) -> list[CryptoCandleData]:
        page_limit = REST_PROVIDER_PAGE_LIMITS[provider]
        interval_ms = INTERVAL_SECONDS[interval] * 1_000
        cursor = from_ts_ms
        candles: list[CryptoCandleData] = []
        while cursor <= to_ts_ms:
            page_end = min(to_ts_ms, cursor + interval_ms * (page_limit - 1))
            page = self._fetch_rest_candle_page(
                provider, symbol, interval, page_limit, cursor, page_end
            )
            candles.extend(page)
            if not page:
                break
            cursor = max(
                cursor + interval_ms,
                max(item.ts for item in page) + interval_ms,
            )
        unique_candles = {candle.ts: candle for candle in candles}
        return sorted(unique_candles.values(), key=lambda item: item.ts)

    def _fetch_rest_candle_page(
        self,
        provider: str,
        symbol: str,
        interval: str,
        limit: int,
        from_ts_ms: int | None,
        to_ts_ms: int | None,
    ) -> list[CryptoCandleData]:
        url = self._provider_candle_url(
            provider, symbol, interval, limit, from_ts_ms, to_ts_ms
        )
        with urlopen(url, timeout=FETCH_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return self._parse_rest_candles(provider, payload)

    @staticmethod
    def _provider_candle_url(
        provider: str,
        symbol: str,
        interval: str,
        limit: int,
        from_ts_ms: int | None,
        to_ts_ms: int | None,
    ) -> str:
        compact_symbol = symbol.replace("-", "")
        if provider == "binance":
            query: dict[str, str | int] = {
                "symbol": compact_symbol,
                "interval": REST_PROVIDER_INTERVALS[provider][interval],
                "limit": limit,
            }
            if from_ts_ms is not None:
                query["startTime"] = from_ts_ms
            if to_ts_ms is not None:
                query["endTime"] = to_ts_ms
            return f"https://api.binance.com/api/v3/klines?{urlencode(query)}"
        if provider == "mexc":
            query = {
                "symbol": compact_symbol,
                "interval": REST_PROVIDER_INTERVALS[provider][interval],
                "limit": limit,
            }
            if from_ts_ms is not None:
                query["startTime"] = from_ts_ms
            if to_ts_ms is not None:
                query["endTime"] = to_ts_ms
            return f"https://api.mexc.com/api/v3/klines?{urlencode(query)}"
        query = {
            "instId": symbol,
            "bar": REST_PROVIDER_INTERVALS[provider][interval],
            "limit": limit,
        }
        return f"https://www.okx.com/api/v5/market/candles?{urlencode(query)}"

    @staticmethod
    def _parse_rest_candles(
        provider: str, payload: object
    ) -> list[CryptoCandleData]:
        if provider == "okx":
            if not isinstance(payload, dict) or payload.get("code") != "0":
                raise RuntimeError("OKX returned an unsuccessful candle response")
            rows = payload.get("data")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise RuntimeError(f"{provider} returned malformed candle data")
        candles: list[CryptoCandleData] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            try:
                if provider == "okx":
                    candle = CryptoCandleData(
                        ts=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                else:
                    candle = CryptoCandleData(
                        ts=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
            except (TypeError, ValueError):
                continue
            candles.append(candle)
        return sorted(candles, key=lambda item: item.ts)

    def _fetch_gate_public_candles(
        self,
        symbol: str,
        interval: str,
        lookback: int,
        from_ts_ms: int | None = None,
        to_ts_ms: int | None = None,
    ) -> CandleFetchResult:
        if from_ts_ms is None or to_ts_ms is None:
            candles = self._fetch_gate_candle_page(
                symbol, interval, min(lookback, MAX_GATE_CANDLES_PER_REQUEST), None, None
            )
        else:
            interval_s = INTERVAL_SECONDS[interval]
            start_s = from_ts_ms // 1000
            end_s = to_ts_ms // 1000
            candles = []
            while start_s <= end_s:
                page_end_s = min(
                    end_s, start_s + interval_s * (MAX_GATE_CANDLES_PER_REQUEST - 1)
                )
                candles.extend(
                    self._fetch_gate_candle_page(
                        symbol,
                        interval,
                        MAX_GATE_CANDLES_PER_REQUEST,
                        start_s,
                        page_end_s,
                    )
                )
                start_s = page_end_s + interval_s
        candles = self._filter_time_range(candles, (from_ts_ms, to_ts_ms))
        if not candles:
            raise RuntimeError("Gate returned no candles for requested range")
        unique_candles = {candle.ts: candle for candle in candles}
        return CandleFetchResult(
            symbol,
            symbol.replace("-", "_"),
            "gate",
            sorted(unique_candles.values(), key=lambda item: item.ts),
        )

    @staticmethod
    def _fetch_gate_candle_page(
        symbol: str,
        interval: str,
        limit: int,
        from_s: int | None,
        to_s: int | None,
    ) -> list[CryptoCandleData]:
        query_data: dict[str, str | int] = {
            "currency_pair": symbol.replace("-", "_"),
            "interval": interval,
            "limit": limit,
        }
        if from_s is not None:
            query_data["from"] = from_s
        if to_s is not None:
            query_data["to"] = to_s
        url = f"https://api.gateio.ws/api/v4/spot/candlesticks?{urlencode(query_data)}"
        with urlopen(url, timeout=FETCH_TIMEOUT_S) as response:
            raw = json.loads(response.read().decode("utf-8"))
        return [
            CryptoCandleData(
                ts=int(row[0]) * 1000,
                volume=float(row[1]),
                close=float(row[2]),
                high=float(row[3]),
                low=float(row[4]),
                open=float(row[5]),
            )
            for row in raw
            if len(row) >= 6
        ]

    def _compute_indicators(
        self,
        candles: list[CryptoCandleData],
    ) -> list[CryptoIndicatorPointData]:
        if not candles:
            return []

        df = pd.DataFrame([item.model_dump() for item in candles])
        close = df["close"]
        for window in MA_WINDOWS:
            df[f"ma{window}"] = close.rolling(window=window).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=RSI_WINDOW).mean()
        loss = (-delta).clip(lower=0).rolling(window=RSI_WINDOW).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df.loc[(loss == 0) & (gain > 0), "rsi"] = 100.0
        df.loc[(loss == 0) & (gain == 0), "rsi"] = 50.0

        df["bb_middle"] = close.rolling(window=BOLLINGER_WINDOW).mean()
        bb_std = close.rolling(window=BOLLINGER_WINDOW).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * BOLLINGER_STD)
        df["bb_lower"] = df["bb_middle"] - (bb_std * BOLLINGER_STD)
        df["momentum"] = close - close.shift(MOMENTUM_WINDOW)
        df["ema12"] = close.ewm(span=12, adjust=False).mean()
        df["ema26"] = close.ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        points: list[CryptoIndicatorPointData] = []
        for row in df.to_dict("records"):
            points.append(
                CryptoIndicatorPointData(
                    ts=int(row["ts"]),
                    ma={f"ma{window}": self._finite(row.get(f"ma{window}")) for window in MA_WINDOWS},
                    rsi=self._finite(row.get("rsi")),
                    bollinger=BollingerBandData(
                        upper=self._finite(row.get("bb_upper")),
                        middle=self._finite(row.get("bb_middle")),
                        lower=self._finite(row.get("bb_lower")),
                    ),
                    momentum=self._finite(row.get("momentum")),
                    macd=self._finite(row.get("macd")),
                    macd_signal=self._finite(row.get("macd_signal")),
                    macd_histogram=self._finite(row.get("macd_histogram")),
                )
            )
        return points

    def _normalize_symbols(self, symbols: list[str]) -> list[str]:
        if not symbols:
            return ["BTC-USDT"]
        normalized: list[str] = []
        for raw_symbol in symbols:
            symbol = raw_symbol.strip().upper().replace("/", "-")
            if not symbol.endswith("-USDT"):
                raise ValueError(f"Only USDT crypto symbols are supported: {raw_symbol}")
            if symbol not in self._symbol_set:
                raise ValueError(f"Unsupported crypto symbol: {symbol}")
            if symbol not in normalized:
                normalized.append(symbol)
        return normalized

    def _normalize_interval(self, interval: str) -> str:
        raw_interval = interval.strip()
        if raw_interval in AGGREGATED_INTERVALS:
            return raw_interval
        normalized = raw_interval.lower()
        if normalized not in SOURCE_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")
        return normalized

    @staticmethod
    def _normalize_time_range(
        from_ts_ms: int | None, to_ts_ms: int | None
    ) -> tuple[int | None, int | None]:
        if from_ts_ms is not None and from_ts_ms <= 0:
            raise ValueError("from_ts_ms must be positive")
        if to_ts_ms is not None and to_ts_ms <= 0:
            raise ValueError("to_ts_ms must be positive")
        if from_ts_ms is not None and to_ts_ms is not None and from_ts_ms > to_ts_ms:
            raise ValueError("from_ts_ms must not be later than to_ts_ms")
        return from_ts_ms, to_ts_ms

    @staticmethod
    def _filter_time_range(
        candles: list[CryptoCandleData], time_range: tuple[int | None, int | None]
    ) -> list[CryptoCandleData]:
        from_ts_ms, to_ts_ms = time_range
        return [
            candle for candle in candles
            if (from_ts_ms is None or candle.ts >= from_ts_ms)
            and (to_ts_ms is None or candle.ts <= to_ts_ms)
        ]

    @staticmethod
    def _aggregate_candles(
        candles: list[CryptoCandleData], interval: str
    ) -> list[CryptoCandleData]:
        if interval not in AGGREGATED_INTERVALS:
            return candles
        buckets: dict[tuple[int, int], list[CryptoCandleData]] = {}
        for candle in candles:
            timestamp = datetime.fromtimestamp(candle.ts / 1000, tz=timezone.utc)
            if interval == "1w":
                iso_year, iso_week, _ = timestamp.isocalendar()
                bucket = (iso_year, iso_week)
            elif interval == "1M":
                bucket = (timestamp.year, timestamp.month)
            elif interval == "3M":
                bucket = (timestamp.year, (timestamp.month - 1) // 3 + 1)
            else:
                bucket = (timestamp.year, 1)
            buckets.setdefault(bucket, []).append(candle)
        return [
            CryptoCandleData(
                ts=items[0].ts,
                open=items[0].open,
                high=max(item.high for item in items),
                low=min(item.low for item in items),
                close=items[-1].close,
                volume=sum(item.volume for item in items),
            )
            for items in buckets.values()
        ]

    def _normalize_lookback(self, lookback: int) -> int:
        if lookback <= 0:
            return DEFAULT_LOOKBACK
        return min(lookback, MAX_LOOKBACK)

    def _normalize_providers(self, providers: list[str] | None) -> list[str]:
        raw_providers = providers or list(self.providers)
        normalized: list[str] = []
        for provider in raw_providers:
            item = provider.strip().lower()
            if item and item not in normalized:
                normalized.append(item)
        return normalized or list(DEFAULT_PROVIDERS)

    def _parse_ohlcv(self, raw: list[list[float]]) -> list[CryptoCandleData]:
        candles: list[CryptoCandleData] = []
        for row in raw or []:
            if len(row) < 6:
                continue
            ts, open_value, high, low, close, volume = row[:6]
            candles.append(
                CryptoCandleData(
                    ts=int(ts),
                    open=float(open_value),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
            )
        candles.sort(key=lambda item: item.ts)
        return candles

    def _to_exchange_symbol(self, symbol: str) -> str:
        return symbol.replace("-", "/")

    def _finite(self, value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        return number


_crypto_market_service: CryptoMarketService | None = None


def get_crypto_market_service() -> CryptoMarketService:
    global _crypto_market_service
    if _crypto_market_service is None:
        _crypto_market_service = CryptoMarketService()
    return _crypto_market_service
