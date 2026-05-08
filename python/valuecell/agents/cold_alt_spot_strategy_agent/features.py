"""Feature pipeline for the cold alt spot strategy."""

from __future__ import annotations

import asyncio
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_INTERVAL_PREFIX,
    FEATURE_GROUP_BY_KEY,
)
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
    UserRequest,
)
from valuecell.agents.spot_rsi_ladder_agent.features import SpotAwareMarketDataSource

from .config import (
    ARBR_PERIOD,
    BOLLINGER_STD,
    BOLLINGER_WINDOW,
    ICE_RSI_MAX,
    ICE_RSI_MIN,
    MTM_PERIOD,
    RSI_PERIOD,
    ColdAltSpotStrategyProfile,
)


def _to_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _consecutive_true_from_tail(values: list[bool]) -> int:
    count = 0
    for flag in reversed(values):
        if not flag:
            break
        count += 1
    return count


class ColdAltSpotFeatureComputer(CandleBasedFeatureComputer):
    """Computes indicators and pattern flags for cold alt spot trading."""

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
            df["prev_close"] = df["close"].shift(1)
            df["sma20"] = df["close"].rolling(window=BOLLINGER_WINDOW).mean()
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(window=RSI_PERIOD).mean()
            loss = (-delta).clip(lower=0).rolling(window=RSI_PERIOD).mean()
            rs = gain / loss.replace(0, np.inf)
            df["rsi"] = 100 - (100 / (1 + rs))
            df["mtm14"] = df["close"] - df["close"].shift(MTM_PERIOD)
            df["bb_middle"] = df["close"].rolling(window=BOLLINGER_WINDOW).mean()
            bb_std = df["close"].rolling(window=BOLLINGER_WINDOW).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * BOLLINGER_STD)
            df["bb_lower"] = df["bb_middle"] - (bb_std * BOLLINGER_STD)
            df["bb_width_ratio"] = (
                (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)
            )

            ar_num = (df["high"] - df["open"]).rolling(window=ARBR_PERIOD).sum()
            ar_den = (df["open"] - df["low"]).rolling(window=ARBR_PERIOD).sum()
            df["ar26"] = 100 * ar_num / ar_den.replace(0, np.nan)

            br_num = (
                (df["high"] - df["prev_close"])
                .clip(lower=0)
                .rolling(window=ARBR_PERIOD)
                .sum()
            )
            br_den = (
                (df["prev_close"] - df["low"])
                .clip(lower=0)
                .rolling(window=ARBR_PERIOD)
                .sum()
            )
            df["br26"] = 100 * br_num / br_den.replace(0, np.nan)

            df["rolling_avg_volume_14"] = df["volume"].rolling(window=14).mean()
            df["rolling_peak_volume_14"] = (
                df["rolling_avg_volume_14"].rolling(window=90, min_periods=14).max()
            )

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            prev2 = df.iloc[-3] if len(df) > 2 else prev
            recent_14 = df.tail(14)
            recent_8 = df.tail(8)
            prior_7 = recent_8.iloc[:-1] if len(recent_8) > 1 else recent_8
            first_4 = df.head(min(4, len(df)))

            last_close = float(last["close"])
            prev_close = float(prev["close"])
            prev2_close = float(prev2["close"])
            last_rsi = _to_float(last.get("rsi"))
            prev_rsi = _to_float(prev.get("rsi"))
            prev2_rsi = _to_float(prev2.get("rsi"))
            last_mtm = _to_float(last.get("mtm14"))
            prev_mtm = _to_float(prev.get("mtm14"))
            prev2_mtm = _to_float(prev2.get("mtm14"))
            last_ar = _to_float(last.get("ar26"))
            prev_ar = _to_float(prev.get("ar26"))
            last_br = _to_float(last.get("br26"))
            prev_br = _to_float(prev.get("br26"))
            last_bb_middle = _to_float(last.get("bb_middle"))
            prev_bb_middle = _to_float(prev.get("bb_middle"))
            last_bb_upper = _to_float(last.get("bb_upper"))
            last_bb_lower = _to_float(last.get("bb_lower"))
            prev_bb_lower = _to_float(prev.get("bb_lower"))
            last_bb_width = _to_float(last.get("bb_width_ratio"))
            prev_bb_width = _to_float(prev.get("bb_width_ratio"))
            prev2_bb_width = _to_float(prev2.get("bb_width_ratio"))
            last_avg_volume = _to_float(last.get("rolling_avg_volume_14"))
            peak_avg_volume = _to_float(last.get("rolling_peak_volume_14"))

            avg_close_14 = _to_float(recent_14["close"].mean())
            sideways_range_ratio = None
            if avg_close_14 and avg_close_14 > 0:
                sideways_range_ratio = (
                    float(recent_14["high"].max()) - float(recent_14["low"].min())
                ) / avg_close_14

            volume_contraction_ratio = None
            if (
                last_avg_volume is not None
                and peak_avg_volume is not None
                and peak_avg_volume > 0
            ):
                volume_contraction_ratio = last_avg_volume / peak_avg_volume

            ice_zone_bars = _consecutive_true_from_tail(
                [
                    (
                        _to_float(value) is not None
                        and ICE_RSI_MIN <= float(value) <= ICE_RSI_MAX
                    )
                    for value in df["rsi"].tail(6).tolist()
                ]
            )

            prior_low = _to_float(prior_7["low"].min()) if not prior_7.empty else None
            prior_high = (
                _to_float(prior_7["high"].max()) if not prior_7.empty else None
            )
            prior_min_rsi = (
                _to_float(prior_7["rsi"].dropna().min()) if not prior_7.empty else None
            )
            prior_max_rsi = (
                _to_float(prior_7["rsi"].dropna().max()) if not prior_7.empty else None
            )

            bottom_divergence = bool(
                prior_low is not None
                and prior_min_rsi is not None
                and last_rsi is not None
                and float(last["low"]) <= prior_low
                and last_rsi > prior_min_rsi
                and last_rsi >= ICE_RSI_MIN
                and last_close >= prev_close
            )
            top_divergence = bool(
                prior_high is not None
                and prior_max_rsi is not None
                and last_rsi is not None
                and float(last["high"]) >= prior_high
                and last_rsi < prior_max_rsi
            )

            band_width_abs = None
            if last_bb_upper is not None and last_bb_lower is not None:
                band_width_abs = last_bb_upper - last_bb_lower

            upper_shadow_ratio = 0.0
            candle_range = max(float(last["high"]) - float(last["low"]), 1e-9)
            upper_shadow_ratio = (
                float(last["high"]) - max(float(last["open"]), float(last["close"]))
            ) / candle_range

            launch_rocket_15m = False
            early_upper_shadow_traps = 0
            if len(first_4) > 0:
                start_open = float(first_4.iloc[0]["open"])
                if start_open > 0:
                    launch_gain = (float(first_4["high"].max()) - start_open) / start_open
                    launch_rocket_15m = launch_gain >= 0.30
                for _, row in first_4.iterrows():
                    row_range = max(float(row["high"]) - float(row["low"]), 1e-9)
                    row_shadow = (
                        float(row["high"]) - max(float(row["open"]), float(row["close"]))
                    ) / row_range
                    if row_shadow >= 0.45:
                        early_upper_shadow_traps += 1

            values = {
                "close": last_close,
                "prev_close": prev_close,
                "prev2_close": prev2_close,
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "volume": float(last["volume"]),
                "change_pct": (
                    (last_close - prev_close) / prev_close if prev_close else 0.0
                ),
                "rsi": last_rsi,
                "prev_rsi": prev_rsi,
                "prev2_rsi": prev2_rsi,
                "rsi_turn_up": bool(
                    last_rsi is not None
                    and prev_rsi is not None
                    and last_rsi > prev_rsi
                ),
                "rsi_bottom_divergence": bottom_divergence,
                "rsi_top_divergence": top_divergence,
                "rsi_ice_zone_bars": ice_zone_bars,
                "mtm14": last_mtm,
                "prev_mtm14": prev_mtm,
                "prev2_mtm14": prev2_mtm,
                "mtm_turn_up": bool(
                    last_mtm is not None
                    and prev_mtm is not None
                    and last_mtm > prev_mtm
                ),
                "mtm_cross_up_zero": bool(
                    last_mtm is not None
                    and prev_mtm is not None
                    and last_mtm > 0
                    and prev_mtm <= 0
                ),
                "mtm_positive_bars": _consecutive_true_from_tail(
                    [
                        (_to_float(value) is not None and float(value) > 0)
                        for value in df["mtm14"].tail(6).tolist()
                    ]
                ),
                "mtm_weak_2bar": bool(
                    last_mtm is not None
                    and prev_mtm is not None
                    and prev2_mtm is not None
                    and last_mtm < prev_mtm < prev2_mtm
                ),
                "ar26": last_ar,
                "prev_ar26": prev_ar,
                "br26": last_br,
                "prev_br26": prev_br,
                "arbr_warm_up": bool(
                    last_ar is not None
                    and last_br is not None
                    and last_ar >= 40
                    and last_br >= 60
                    and (
                        (prev_ar is not None and prev_ar < 40)
                        or (prev_br is not None and prev_br < 60)
                        or (
                            prev_ar is not None
                            and prev_br is not None
                            and last_ar > prev_ar
                            and last_br > prev_br
                        )
                    )
                ),
                "arbr_cooling_off": bool(
                    last_ar is not None
                    and last_br is not None
                    and last_ar < 40
                    and last_br < 60
                ),
                "arbr_overheated": bool(
                    last_ar is not None
                    and last_br is not None
                    and last_ar > 110
                    and last_br > 130
                ),
                "bb_upper": last_bb_upper,
                "bb_middle": last_bb_middle,
                "bb_lower": last_bb_lower,
                "bb_middle_slope": (
                    (last_bb_middle - prev_bb_middle)
                    if last_bb_middle is not None and prev_bb_middle is not None
                    else None
                ),
                "bb_width_ratio": last_bb_width,
                "bb_squeeze_3bar": bool(
                    last_bb_width is not None
                    and prev_bb_width is not None
                    and prev2_bb_width is not None
                    and last_bb_width <= prev_bb_width <= prev2_bb_width
                ),
                "price_in_lower_half": bool(
                    last_bb_middle is not None
                    and last_bb_lower is not None
                    and last_close <= last_bb_middle
                    and last_close >= last_bb_lower
                ),
                "price_near_lower_band": bool(
                    band_width_abs is not None
                    and last_bb_lower is not None
                    and band_width_abs > 0
                    and last_close <= last_bb_lower + band_width_abs * 0.35
                ),
                "price_touch_upper_band": bool(
                    last_bb_upper is not None and float(last["high"]) >= last_bb_upper
                ),
                "upper_band_rejection": bool(
                    last_bb_upper is not None
                    and float(last["high"]) >= last_bb_upper
                    and last_close < float(last["high"])
                    and upper_shadow_ratio >= 0.25
                ),
                "below_lower_band_2bar": bool(
                    last_bb_lower is not None
                    and prev_bb_lower is not None
                    and last_close < last_bb_lower
                    and prev_close < prev_bb_lower
                    and last_close < float(last["open"])
                    and prev_close < float(prev["open"])
                ),
                "below_middle_band": bool(
                    last_bb_middle is not None and last_close < last_bb_middle
                ),
                "bb_opening_down": bool(
                    last_bb_width is not None
                    and prev_bb_width is not None
                    and prev2_bb_width is not None
                    and last_bb_width > prev_bb_width > prev2_bb_width
                    and last_bb_middle is not None
                    and prev_bb_middle is not None
                    and last_bb_middle < prev_bb_middle
                    and last_close < last_bb_middle
                ),
                "sideways_range_ratio_14": sideways_range_ratio,
                "volume_contraction_ratio_14v90": volume_contraction_ratio,
                "launch_rocket_15m": launch_rocket_15m,
                "early_upper_shadow_traps": early_upper_shadow_traps,
                "close_turn_up": bool(last_close > prev_close),
                "count": len(series),
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
                    instrument=InstrumentRef(symbol=symbol),
                    values=values,
                    meta=fv_meta,
                )
            )
        return features


class ColdAltSpotFeaturesPipeline(BaseFeaturesPipeline):
    """Builds candles and market snapshot features for cold alt spot trading."""

    def __init__(
        self,
        request: UserRequest,
        market_data_source: SpotAwareMarketDataSource,
        candle_feature_computer: CandleBasedFeatureComputer,
        market_snapshot_computer: MarketSnapshotFeatureComputer,
        profile: ColdAltSpotStrategyProfile,
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
        profile: ColdAltSpotStrategyProfile,
    ) -> "ColdAltSpotFeaturesPipeline":
        market_data_source = SpotAwareMarketDataSource(
            exchange_id=request.exchange_config.exchange_id,
            market_type=request.exchange_config.market_type,
        )
        return cls(
            request=request,
            market_data_source=market_data_source,
            candle_feature_computer=ColdAltSpotFeatureComputer(),
            market_snapshot_computer=MarketSnapshotFeatureComputer(),
            profile=profile,
        )
