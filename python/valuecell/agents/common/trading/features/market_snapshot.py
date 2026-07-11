from __future__ import annotations

from typing import Dict, List

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_KEY,
    FEATURE_GROUP_BY_MARKET_SNAPSHOT,
)
from valuecell.agents.common.trading.models import (
    FeatureVector,
    InstrumentRef,
    MarketSnapShotType,
)
from valuecell.utils.ts import get_current_timestamp_ms

# A REST ticker older than one minute is unsuitable for increasing exposure.
MARKET_SNAPSHOT_MAX_AGE_MS = 60_000


def market_snapshot_status_by_symbol(
    features: List[FeatureVector],
    *,
    now_ts_ms: int | None = None,
) -> dict[str, str]:
    """Return the best realtime snapshot status for each symbol.

    Producers created before freshness metadata existed retain ``unknown`` so
    historical replays remain readable. Current snapshot producers always emit
    an explicit ``fresh`` or ``stale`` status.
    """
    now_ts = now_ts_ms if now_ts_ms is not None else get_current_timestamp_ms()
    statuses: dict[str, str] = {}
    for feature in features:
        meta = feature.meta or {}
        if meta.get(FEATURE_GROUP_BY_KEY) != FEATURE_GROUP_BY_MARKET_SNAPSHOT:
            continue
        snapshot_ts = meta.get("snapshot_ts_ms")
        if snapshot_ts is not None:
            try:
                age_ms = max(0, now_ts - int(snapshot_ts))
                status = "fresh" if age_ms <= MARKET_SNAPSHOT_MAX_AGE_MS else "stale"
            except (TypeError, ValueError):
                status = "unknown"
        else:
            raw_status = meta.get("freshness_status")
            status = str(raw_status) if raw_status in {"fresh", "stale", "missing"} else "unknown"
        existing = statuses.get(feature.instrument.symbol)
        if existing != "fresh" or status == "fresh":
            statuses[feature.instrument.symbol] = status
    return statuses


class MarketSnapshotFeatureComputer:
    """Convert exchange market_snapshot structures into FeatureVector items.

    This class encapsulates the logic previously embedded in
    `DefaultFeaturesPipeline._build_market_features`. Keeping it separate
    makes the pipeline easier to test and replace.
    """

    def build(
        self, market_snapshot: MarketSnapShotType, exchange_id: str
    ) -> List[FeatureVector]:
        features: List[FeatureVector] = []
        now_ts = get_current_timestamp_ms()

        for symbol, data in (market_snapshot or {}).items():
            if not isinstance(data, dict):
                continue

            price_obj = data.get("price") if isinstance(data, dict) else None
            timestamp = None
            values: Dict[str, float] = {}

            if isinstance(price_obj, dict):
                timestamp = price_obj.get("timestamp") or price_obj.get("ts")
                for key in ("last", "close", "open", "high", "low", "bid", "ask"):
                    val = price_obj.get(key)
                    if val is not None:
                        try:
                            values[f"price.{key}"] = float(val)
                        except (TypeError, ValueError):
                            continue

                change = price_obj.get("percentage")
                if change is not None:
                    try:
                        values["price.change_pct"] = float(change)
                    except (TypeError, ValueError):
                        pass

                volume = price_obj.get("quoteVolume") or price_obj.get("baseVolume")
                if volume is not None:
                    try:
                        values["price.volume"] = float(volume)
                    except (TypeError, ValueError):
                        pass

            if isinstance(data.get("open_interest"), dict):
                oi = data["open_interest"]
                for field in ("openInterest", "openInterestAmount", "baseVolume"):
                    val = oi.get(field)
                    if val is not None:
                        try:
                            values["open_interest"] = float(val)
                        except (TypeError, ValueError):
                            pass
                        break

            if isinstance(data.get("funding_rate"), dict):
                fr = data["funding_rate"]
                rate = fr.get("fundingRate") or fr.get("funding_rate")
                if rate is not None:
                    try:
                        values["funding.rate"] = float(rate)
                    except (TypeError, ValueError):
                        pass
                mark_price = fr.get("markPrice") or fr.get("mark_price")
                if mark_price is not None:
                    try:
                        values["funding.mark_price"] = float(mark_price)
                    except (TypeError, ValueError):
                        pass

            if not values:
                continue

            try:
                snapshot_ts_ms = int(timestamp) if timestamp is not None else now_ts
            except (TypeError, ValueError):
                snapshot_ts_ms = now_ts
            freshness_age_ms = max(0, now_ts - snapshot_ts_ms)
            freshness_status = (
                "fresh"
                if freshness_age_ms <= MARKET_SNAPSHOT_MAX_AGE_MS
                else "stale"
            )

            fv_ts = snapshot_ts_ms
            feature = FeatureVector(
                ts=fv_ts,
                instrument=InstrumentRef(symbol=symbol, exchange_id=exchange_id),
                values=values,
                meta={
                    FEATURE_GROUP_BY_KEY: FEATURE_GROUP_BY_MARKET_SNAPSHOT,
                    "snapshot_ts_ms": snapshot_ts_ms,
                    "freshness_age_ms": freshness_age_ms,
                    "freshness_status": freshness_status,
                    "coverage_status": "complete",
                },
            )
            features.append(feature)

        return features
