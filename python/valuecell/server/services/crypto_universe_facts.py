"""Exchange facts and threshold decisions for the crypto symbol catalogue.

Everything here is derived from OKX public endpoints. A missing fact is never
estimated: the symbol is rejected with an explicit reason code and re-evaluated
on the next sync, so the catalogue can only ever contain instruments the venue
proved are listed and liquid.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from loguru import logger

OKX_BASE_URL = "https://www.okx.com"
USER_AGENT = "valuecell-market-data/1.0"

# Instruments whose base asset is a fiat-pegged unit are not strategy targets:
# their price cannot trend, so every trend/rotation rule is meaningless on them.
STABLE_BASE_ASSETS = frozenset(
    {
        "USDC",
        "USDT",
        "DAI",
        "TUSD",
        "FDUSD",
        "PYUSD",
        "USDD",
        "USDE",
        "USDG",
        "EURT",
        "EUR",
        "DAI",
    }
)
# OKX leveraged tokens carry a leverage suffix in the base asset name.
LEVERAGED_TOKEN_SUFFIXES = ("3L", "3S", "5L", "5S")

REASON_ADMITTED = "okx_liquidity_verified"
REASON_INSTRUMENT_NOT_LISTED = "okx_instrument_not_listed"
REASON_QUOTE_ASSET_MISMATCH = "quote_asset_not_supported"
REASON_STABLE_BASE = "stable_base_asset_excluded"
REASON_LEVERAGED_TOKEN = "leveraged_token_excluded"
REASON_LISTING_AGE = "listing_age_below_minimum"
REASON_VOLUME_BELOW_MINIMUM = "average_quote_volume_below_minimum"
REASON_VOLUME_UNAVAILABLE = "quote_volume_unavailable"


@dataclass(frozen=True, slots=True)
class OkxSpotInstrument:
    """One SPOT instrument exactly as OKX reports it."""

    instrument_id: str
    base_asset: str
    quote_asset: str
    state: str
    listed_at: datetime | None


@dataclass(frozen=True, slots=True)
class SymbolFacts:
    """Verified exchange facts for one candidate symbol."""

    symbol: str
    quote_volume_24h: float | None = None
    average_quote_volume_30d: float | None = None
    price_quote: float | None = None
    observed_days: int = 0


@dataclass(frozen=True, slots=True)
class SymbolDecision:
    """Threshold outcome for one candidate symbol."""

    symbol: str
    admitted: bool
    reason_code: str
    reason_detail: str
    permanent_exclusion: bool


def to_symbol(instrument_id: str) -> str:
    """``BTC-USDT`` style identifier used across the product."""

    return instrument_id.strip().upper().replace("/", "-")


def to_instrument_id(symbol: str) -> str:
    return symbol.strip().upper().replace("-", "-")


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _fetch_okx_json(path: str, query: dict[str, str | int], timeout_s: float) -> object:
    request = Request(
        f"{OKX_BASE_URL}{path}?{urlencode(query)}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _okx_rows(payload: object) -> list[object]:
    if not isinstance(payload, dict) or payload.get("code") != "0":
        raise ValueError("OKX returned an unsuccessful response")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("OKX response was malformed")
    return data


def fetch_okx_spot_instruments(
    quote_asset: str, *, timeout_s: float
) -> list[OkxSpotInstrument]:
    """Every SPOT instrument OKX currently reports for the quote asset."""

    rows = _okx_rows(
        _fetch_okx_json(
            "/api/v5/public/instruments",
            {"instType": "SPOT"},
            timeout_s,
        )
    )
    instruments: list[OkxSpotInstrument] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("quoteCcy", "")).upper() != quote_asset.upper():
            continue
        instrument_id = str(row.get("instId", "")).strip().upper()
        base_asset = str(row.get("baseCcy", "")).strip().upper()
        if not instrument_id or not base_asset:
            continue
        listed_at: datetime | None = None
        list_time = _finite(row.get("listTime"))
        if list_time is not None:
            listed_at = datetime.fromtimestamp(list_time / 1000.0, tz=timezone.utc)
        instruments.append(
            OkxSpotInstrument(
                instrument_id=instrument_id,
                base_asset=base_asset,
                quote_asset=str(row.get("quoteCcy", "")).upper(),
                state=str(row.get("state", "")).lower(),
                listed_at=listed_at,
            )
        )
    return instruments


def fetch_okx_spot_tickers(*, timeout_s: float) -> dict[str, float]:
    """24h quote volume per instrument, used only as a fetch prefilter."""

    rows = _okx_rows(
        _fetch_okx_json("/api/v5/market/tickers", {"instType": "SPOT"}, timeout_s)
    )
    volumes: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        instrument_id = str(row.get("instId", "")).strip().upper()
        if not instrument_id:
            continue
        volume = _finite(row.get("volCcy24h"))
        if volume is None:
            volume = _finite(row.get("vol24h"))
        if volume is None:
            continue
        volumes[instrument_id] = volume
    return volumes


def fetch_okx_daily_quote_volumes(
    instrument_id: str,
    *,
    window_days: int,
    timeout_s: float,
    observed_at: datetime,
) -> tuple[list[float], float | None]:
    """Completed daily quote volumes plus the latest close for one instrument.

    Only confirmed candles (``confirm == "1"``) that closed before the current
    UTC day count, so an in-progress day can neither inflate nor deflate the
    average. A row missing its quote volume is dropped rather than zero-filled.
    """

    rows = _okx_rows(
        _fetch_okx_json(
            "/api/v5/market/candles",
            {"instId": instrument_id, "bar": "1Dutc", "limit": min(window_days + 5, 300)},
            timeout_s,
        )
    )
    today = observed_at.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    volumes: list[tuple[int, float]] = []
    latest_close: float | None = None
    for row in rows:
        if not isinstance(row, list) or len(row) < 9:
            continue
        timestamp = _finite(row[0])
        if timestamp is None:
            continue
        if row[8] != "1":
            continue
        close_ts = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)
        if close_ts >= today:
            continue
        quote_volume = _finite(row[7])
        if quote_volume is None:
            quote_volume = _finite(row[6])
        if quote_volume is None:
            continue
        volumes.append((int(timestamp), quote_volume))
        if latest_close is None:
            close_price = _finite(row[4])
            if close_price is not None:
                latest_close = close_price
    volumes.sort(key=lambda item: item[0])
    return [item[1] for item in volumes[-window_days:]], latest_close


def select_volume_candidates(
    instruments: Iterable[OkxSpotInstrument],
    ticker_volumes: dict[str, float],
    *,
    minimum_average_quote_volume_30d: float,
    maximum_candidates: int,
) -> list[OkxSpotInstrument]:
    """Bound how many instruments need an exact 30-day volume verification.

    The 24h ticker is only a prefilter with a wide tolerance, because a single
    quiet day must not hide a symbol whose monthly turnover still qualifies.
    """

    tolerance = minimum_average_quote_volume_30d / 5.0
    scored = [
        (ticker_volumes.get(instrument.instrument_id, 0.0), instrument)
        for instrument in instruments
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        instrument
        for volume, instrument in scored[:maximum_candidates]
        if volume >= tolerance
    ]


def evaluate_symbol(
    *,
    symbol: str,
    facts: SymbolFacts,
    listed_at: datetime | None,
    observed_at: datetime,
    minimum_listing_age_days: int,
    minimum_average_quote_volume_30d: float,
) -> SymbolDecision:
    """Apply the admission thresholds to verified facts."""

    if listed_at is None:
        return SymbolDecision(
            symbol=symbol,
            admitted=False,
            reason_code=REASON_VOLUME_UNAVAILABLE,
            reason_detail="交易所未返回该标的的上市时间，无法证明上市时长。",
            permanent_exclusion=False,
        )
    listing_age_days = (observed_at - listed_at).days
    if listing_age_days < minimum_listing_age_days:
        return SymbolDecision(
            symbol=symbol,
            admitted=False,
            reason_code=REASON_LISTING_AGE,
            reason_detail=(
                f"上市 {listing_age_days} 天，低于要求 {minimum_listing_age_days} 天。"
            ),
            permanent_exclusion=False,
        )
    if facts.average_quote_volume_30d is None or facts.observed_days == 0:
        return SymbolDecision(
            symbol=symbol,
            admitted=False,
            reason_code=REASON_VOLUME_UNAVAILABLE,
            reason_detail="交易所未返回可用的日线成交额，无法证明流动性。",
            permanent_exclusion=False,
        )
    if facts.average_quote_volume_30d < minimum_average_quote_volume_30d:
        return SymbolDecision(
            symbol=symbol,
            admitted=False,
            reason_code=REASON_VOLUME_BELOW_MINIMUM,
            reason_detail=(
                f"近 {facts.observed_days} 天日均成交额 "
                f"{facts.average_quote_volume_30d:,.0f} USDT，低于门槛 "
                f"{minimum_average_quote_volume_30d:,.0f} USDT。"
            ),
            permanent_exclusion=False,
        )
    return SymbolDecision(
        symbol=symbol,
        admitted=True,
        reason_code=REASON_ADMITTED,
        reason_detail=(
            f"上市 {listing_age_days} 天；近 {facts.observed_days} 天日均成交额 "
            f"{facts.average_quote_volume_30d:,.0f} USDT。"
        ),
        permanent_exclusion=False,
    )


def policy_exclusion(base_asset: str) -> SymbolDecision | None:
    """Permanent exclusions that never depend on market conditions."""

    base = base_asset.strip().upper()
    if base in STABLE_BASE_ASSETS:
        return SymbolDecision(
            symbol="",
            admitted=False,
            reason_code=REASON_STABLE_BASE,
            reason_detail=f"{base} 是稳定币/法币锚定资产，趋势类策略无意义。",
            permanent_exclusion=True,
        )
    if base.endswith(LEVERAGED_TOKEN_SUFFIXES):
        return SymbolDecision(
            symbol="",
            admitted=False,
            reason_code=REASON_LEVERAGED_TOKEN,
            reason_detail=f"{base} 是杠杆代币，不适合作为策略标的。",
            permanent_exclusion=True,
        )
    return None


def gather_symbol_facts(
    candidates: list[OkxSpotInstrument],
    *,
    window_days: int,
    timeout_s: float,
    concurrency: int,
    observed_at: datetime,
) -> dict[str, SymbolFacts]:
    """Fetch exact 30-day quote volumes with bounded concurrency."""

    results: dict[str, SymbolFacts] = {}

    def _fetch(instrument: OkxSpotInstrument) -> tuple[str, SymbolFacts]:
        symbol = to_symbol(instrument.instrument_id)
        try:
            volumes, close_price = fetch_okx_daily_quote_volumes(
                instrument.instrument_id,
                window_days=window_days,
                timeout_s=timeout_s,
                observed_at=observed_at,
            )
        except Exception as exc:
            logger.warning(
                "Crypto universe volume fetch failed symbol={} err={}", symbol, exc
            )
            return symbol, SymbolFacts(symbol=symbol)
        if not volumes:
            return symbol, SymbolFacts(symbol=symbol)
        return symbol, SymbolFacts(
            symbol=symbol,
            average_quote_volume_30d=sum(volumes) / len(volumes),
            price_quote=close_price,
            observed_days=len(volumes),
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for symbol, facts in pool.map(_fetch, candidates):
            results[symbol] = facts
    return results


def utc_day_start(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def is_due(last_observed_at: datetime | None, now: datetime, interval_days: int) -> bool:
    """True when the catalogue has never been built or is older than a cycle."""

    if last_observed_at is None:
        return True
    if last_observed_at.tzinfo is None:
        last_observed_at = last_observed_at.replace(tzinfo=timezone.utc)
    return now - last_observed_at >= timedelta(days=interval_days)