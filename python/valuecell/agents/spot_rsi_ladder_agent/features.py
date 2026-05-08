"""Custom feature pipeline for spot RSI ladder strategies."""

from __future__ import annotations

import asyncio
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_INTERVAL_PREFIX,
    FEATURE_GROUP_BY_KEY,
)
from valuecell.agents.common.trading.data.interfaces import BaseMarketDataSource
from valuecell.agents.common.trading.features.interfaces import (
    BaseFeaturesPipeline,
    CandleBasedFeatureComputer,
)
from valuecell.agents.common.trading.features.market_snapshot import (
    MarketSnapshotFeatureComputer,
)
from valuecell.agents.common.trading.models import (
    Candle,
    FeaturesPipelineResult,
    FeatureVector,
    InstrumentRef,
    MarketSnapShotType,
    MarketType,
    UserRequest,
)
from valuecell.agents.common.trading.utils import get_exchange_cls

from .config import SpotRsiStrategyProfile


def _to_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


class SpotAwareMarketDataSource(BaseMarketDataSource):
    """Fetch OHLCV and ticker data with spot-aware symbol normalization."""

    def __init__(self, exchange_id: Optional[str], market_type: MarketType) -> None:
        self._exchange_id = exchange_id or "okx"
        self._market_type = market_type

    def _normalize_symbol(self, symbol: str) -> str:
        base_symbol = symbol.replace("-", "/")
        if self._market_type == MarketType.SPOT:
            return base_symbol
        if ":" not in base_symbol:
            parts = base_symbol.split("/")
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}:{parts[1]}"
        return base_symbol

    async def _create_exchange(self):
        exchange_cls = get_exchange_cls(self._exchange_id)
        return exchange_cls({"newUpdates": False})

    async def get_recent_candles(
        self,
        symbols: List[str],
        interval: str,
        lookback: int,
    ) -> List[Candle]:
        async def _fetch_symbol(symbol: str) -> List[Candle]:
            exchange = await self._create_exchange()
            normalized_symbol = self._normalize_symbol(symbol)
            try:
                raw = await exchange.fetch_ohlcv(
                    normalized_symbol,
                    timeframe=interval,
                    since=None,
                    limit=lookback,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch spot RSI candles for {symbol} ({normalized}) from {exchange_id} interval={interval}: {error}",
                    symbol=symbol,
                    normalized=normalized_symbol,
                    exchange_id=self._exchange_id,
                    interval=interval,
                    error=str(exc),
                )
                return []
            finally:
                try:
                    await exchange.close()
                except Exception:
                    logger.warning(
                        "Failed to close exchange connection for {exchange_id}",
                        exchange_id=self._exchange_id,
                    )

            candles: List[Candle] = []
            for ts, open_v, high_v, low_v, close_v, volume_v in raw:
                candles.append(
                    Candle(
                        ts=int(ts),
                        instrument=InstrumentRef(
                            symbol=symbol,
                            exchange_id=self._exchange_id,
                        ),
                        open=float(open_v),
                        high=float(high_v),
                        low=float(low_v),
                        close=float(close_v),
                        volume=float(volume_v),
                        interval=interval,
                    )
                )
            return candles

        results = await asyncio.gather(*[_fetch_symbol(symbol) for symbol in symbols])
        return list(itertools.chain.from_iterable(results))

    async def get_market_snapshot(self, symbols: List[str]) -> MarketSnapShotType:
        snapshot: dict[str, dict[str, object]] = defaultdict(dict)
        exchange = await self._create_exchange()
        try:
            for symbol in symbols:
                normalized_symbol = self._normalize_symbol(symbol)
                try:
                    ticker = await exchange.fetch_ticker(normalized_symbol)
                    snapshot[symbol]["price"] = ticker
                except Exception as exc:
                    logger.warning(
                        "Failed to fetch spot RSI market snapshot for {symbol} ({normalized}) from {exchange_id}: {error}",
                        symbol=symbol,
                        normalized=normalized_symbol,
                        exchange_id=self._exchange_id,
                        error=str(exc),
                    )
        finally:
            try:
                await exchange.close()
            except Exception:
                logger.warning(
                    "Failed to close exchange connection for {exchange_id}",
                    exchange_id=self._exchange_id,
                )
        return dict(snapshot)


class SpotRsiLadderCandleFeatureComputer(CandleBasedFeatureComputer):
    """Computes RSI/MA/Bollinger features required by the fixed rule-set."""

    def compute_features(
        self,
        candles: Optional[List[Candle]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[FeatureVector]:
        if not candles:
            return []

        grouped: Dict[str, List[Candle]] = defaultdict(list)
        for candle in candles:
            grouped[candle.instrument.symbol].append(candle)

        features: List[FeatureVector] = []
        for symbol, series in grouped.items():
            series.sort(key=lambda item: item.ts)
            rows = [
                {
                    "ts": candle.ts,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "interval": candle.interval,
                }
                for candle in series
            ]
            df = pd.DataFrame(rows)
            df["sma20"] = df["close"].rolling(window=20).mean()
            df["sma60"] = df["close"].rolling(window=60).mean()
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta).clip(lower=0).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.inf)
            df["rsi"] = 100 - (100 / (1 + rs))
            df["mtm14"] = df["close"] - df["close"].shift(14)
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
            df["bb_width_ratio"] = (
                (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)
            )
            df["bb_width_ratio_avg10"] = df["bb_width_ratio"].rolling(window=10).mean()

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            prev2 = df.iloc[-3] if len(df) > 2 else prev

            sma20 = _to_float(last.get("sma20"))
            sma60 = _to_float(last.get("sma60"))
            prev_sma20 = _to_float(prev.get("sma20"))
            prev_sma60 = _to_float(prev.get("sma60"))
            prev2_sma20 = _to_float(prev2.get("sma20"))
            prev2_sma60 = _to_float(prev2.get("sma60"))
            prev_bb_middle = _to_float(prev.get("bb_middle"))
            last_bb_middle = _to_float(last.get("bb_middle"))
            prev_rsi = _to_float(prev.get("rsi"))
            last_rsi = _to_float(last.get("rsi"))
            last_mtm = _to_float(last.get("mtm14"))
            prev_mtm = _to_float(prev.get("mtm14"))
            last_bb_lower = _to_float(last.get("bb_lower"))
            last_bb_upper = _to_float(last.get("bb_upper"))
            band_width_abs = None
            if last_bb_upper is not None and last_bb_lower is not None:
                band_width_abs = last_bb_upper - last_bb_lower
            last_bb_width_ratio = _to_float(last.get("bb_width_ratio"))
            avg_bb_width_ratio = _to_float(last.get("bb_width_ratio_avg10"))

            values = {
                "close": float(last.close),
                "prev_close": float(prev.close),
                "open": float(last.open),
                "high": float(last.high),
                "low": float(last.low),
                "volume": float(last.volume),
                "change_pct": (
                    (float(last.close) - float(prev.close)) / float(prev.close)
                    if float(prev.close)
                    else 0.0
                ),
                "sma20": sma20,
                "sma60": sma60,
                "sma20_slope": (
                    (sma20 - prev_sma20)
                    if sma20 is not None and prev_sma20 is not None
                    else None
                ),
                "sma60_slope": (
                    (sma60 - prev_sma60)
                    if sma60 is not None and prev_sma60 is not None
                    else None
                ),
                "sma20_prev_slope": (
                    (prev_sma20 - prev2_sma20)
                    if prev_sma20 is not None and prev2_sma20 is not None
                    else None
                ),
                "sma60_prev_slope": (
                    (prev_sma60 - prev2_sma60)
                    if prev_sma60 is not None and prev2_sma60 is not None
                    else None
                ),
                "rsi": last_rsi,
                "prev_rsi": prev_rsi,
                "rsi_turn_up": bool(
                    last_rsi is not None
                    and prev_rsi is not None
                    and last_rsi > prev_rsi
                ),
                "mtm14": last_mtm,
                "prev_mtm14": prev_mtm,
                "mtm_turn_up": bool(
                    last_mtm is not None
                    and prev_mtm is not None
                    and last_mtm > prev_mtm
                ),
                "mtm_below_zero": bool(
                    last_mtm is not None and last_mtm < 0
                ),
                "close_turn_up": bool(float(last.close) > float(prev.close)),
                "bb_upper": last_bb_upper,
                "bb_middle": last_bb_middle,
                "bb_lower": last_bb_lower,
                "bb_width_ratio": last_bb_width_ratio,
                "bb_width_ratio_avg10": avg_bb_width_ratio,
                "bb_squeeze": bool(
                    last_bb_width_ratio is not None
                    and avg_bb_width_ratio is not None
                    and last_bb_width_ratio <= avg_bb_width_ratio * 0.85
                ),
                "bb_near_lower": bool(
                    last_bb_lower is not None
                    and band_width_abs is not None
                    and band_width_abs > 0
                    and float(last.close) <= last_bb_lower + (band_width_abs * 0.25)
                ),
                "bb_mid_cross_up": bool(
                    prev_bb_middle is not None
                    and last_bb_middle is not None
                    and float(prev.close) <= prev_bb_middle
                    and float(last.close) > last_bb_middle
                ),
                "price_below_sma20": bool(
                    sma20 is not None and float(last.close) < sma20
                ),
                "price_below_sma60": bool(
                    sma60 is not None and float(last.close) < sma60
                ),
                "price_above_sma20": bool(
                    sma20 is not None and float(last.close) > sma20
                ),
                "price_above_sma60": bool(
                    sma60 is not None and float(last.close) > sma60
                ),
            }

            interval = series[-1].interval
            fv_meta = {
                FEATURE_GROUP_BY_KEY: f"{FEATURE_GROUP_BY_INTERVAL_PREFIX}{interval}",
                "interval": interval,
                "count": len(series),
                "window_start_ts": int(rows[0]["ts"]),
                "window_end_ts": int(last["ts"]),
            }
            if meta:
                for key, value in meta.items():
                    fv_meta.setdefault(key, value)

            features.append(
                FeatureVector(
                    ts=int(last["ts"]),
                    instrument=series[-1].instrument,
                    values=values,
                    meta=fv_meta,
                )
            )
        return features


class SpotRsiLadderFeaturesPipeline(BaseFeaturesPipeline):
    """Fetches multi-timeframe candles plus market snapshots for rule-based trading."""

    def __init__(
        self,
        request: UserRequest,
        market_data_source: BaseMarketDataSource,
        candle_feature_computer: CandleBasedFeatureComputer,
        market_snapshot_computer: MarketSnapshotFeatureComputer,
        profile: SpotRsiStrategyProfile,
    ) -> None:
        self._request = request
        self._market_data_source = market_data_source
        self._candle_feature_computer = candle_feature_computer
        self._market_snapshot_computer = market_snapshot_computer
        self._symbols = list(dict.fromkeys(request.trading_config.symbols))
        self._candle_configurations = list(profile.candle_configurations)

    async def build(self) -> FeaturesPipelineResult:
        async def _fetch_candles(interval: str, lookback: int) -> List[FeatureVector]:
            candles = await self._market_data_source.get_recent_candles(
                self._symbols,
                interval,
                lookback,
            )
            return self._candle_feature_computer.compute_features(candles=candles)

        async def _fetch_market_features() -> List[FeatureVector]:
            market_snapshot = await self._market_data_source.get_market_snapshot(
                self._symbols
            )
            return self._market_snapshot_computer.build(
                market_snapshot or {},
                self._request.exchange_config.exchange_id,
            )

        tasks = [
            _fetch_candles(config.interval, config.lookback)
            for config in self._candle_configurations
        ]
        tasks.append(_fetch_market_features())

        results = await asyncio.gather(*tasks)
        market_features = results.pop()
        candle_features = list(itertools.chain.from_iterable(results))
        candle_features.extend(market_features)
        return FeaturesPipelineResult(features=candle_features)

    @classmethod
    def from_request(
        cls,
        request: UserRequest,
        profile: SpotRsiStrategyProfile,
    ) -> "SpotRsiLadderFeaturesPipeline":
        market_data_source = SpotAwareMarketDataSource(
            exchange_id=request.exchange_config.exchange_id,
            market_type=request.exchange_config.market_type,
        )
        return cls(
            request=request,
            market_data_source=market_data_source,
            candle_feature_computer=SpotRsiLadderCandleFeatureComputer(),
            market_snapshot_computer=MarketSnapshotFeatureComputer(),
            profile=profile,
        )
