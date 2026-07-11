"""Public Polymarket Gamma/CLOB observations for paper research only."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

import httpx

from valuecell.server.api.schemas.prediction_market import (
    PredictionMarketBookHealthData,
    PredictionMarketBookLevelData,
    PredictionMarketCatalogData,
    PredictionMarketOrderBookData,
    PredictionMarketOutcomeData,
    PredictionMarketSignalData,
    PredictionMarketSnapshotData,
    PredictionMarketSummaryData,
)

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
FETCH_TIMEOUT_S = 10.0
CACHE_TTL_S = 10.0
MAX_CATALOG_LIMIT = 100


class PredictionMarketTransport(Protocol):
    async def get(self, url: str, params: dict[str, Any] | None = None) -> Any: ...


class HttpxPredictionMarketTransport:
    """Short-lived public HTTP transport with an explicit deadline."""

    async def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_S) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()


@dataclass
class _CacheValue:
    value: Any
    observed_at_ms: int
    expires_at: float


class PredictionMarketService:
    """Fetch and normalize public market observations without live execution."""

    def __init__(self, transport: PredictionMarketTransport | None = None) -> None:
        self._transport = transport or HttpxPredictionMarketTransport()
        self._cache: dict[str, _CacheValue] = {}
        self._lock = asyncio.Lock()

    async def catalog(
        self, limit: int = 30, after_cursor: str | None = None
    ) -> PredictionMarketCatalogData:
        if not 1 <= limit <= MAX_CATALOG_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_CATALOG_LIMIT}")
        params: dict[str, Any] = {"closed": "false", "limit": limit}
        if after_cursor:
            params["after_cursor"] = after_cursor
        payload, observed_at_ms = await self._get_cached(
            f"catalog:{limit}:{after_cursor or ''}",
            f"{GAMMA_BASE_URL}/markets/keyset",
            params,
        )
        raw_markets = payload.get("markets", []) if isinstance(payload, dict) else []
        summaries = [_market_summary(item) for item in raw_markets]
        return PredictionMarketCatalogData(
            source_timestamp_ms=observed_at_ms,
            observed_at_ms=observed_at_ms,
            freshness_age_ms=0,
            freshness_status="fresh",
            markets=summaries,
            next_cursor=payload.get("next_cursor") if isinstance(payload, dict) else None,
        )

    async def snapshot(
        self, market_id: str, outcome: str
    ) -> PredictionMarketSnapshotData:
        market_payload, market_observed_at_ms = await self._get_cached(
            f"market:{market_id}", f"{GAMMA_BASE_URL}/markets/{market_id}", None
        )
        summary = _market_summary(market_payload)
        selected = next((item for item in summary.outcomes if item.outcome == outcome), None)
        if selected is None:
            raise ValueError("outcome must match a market outcome")
        book_payload, observed_at_ms = await self._get_cached(
            f"book:{selected.token_id}",
            f"{CLOB_BASE_URL}/book",
            {"token_id": selected.token_id},
        )
        book = _normalize_book(book_payload)
        source_ts = _timestamp_ms(book_payload.get("timestamp"), observed_at_ms)
        freshness_age_ms = max(0, observed_at_ms - source_ts)
        freshness_status = _freshness_status(freshness_age_ms)
        warnings: list[str] = []
        if market_observed_at_ms != observed_at_ms:
            warnings.append("Market metadata and order book were observed separately.")
        if book.health.status != "valid":
            warnings.append(f"Order book is {book.health.status}; paper exposure is not eligible.")
        return PredictionMarketSnapshotData(
            source_timestamp_ms=source_ts,
            observed_at_ms=observed_at_ms,
            freshness_age_ms=freshness_age_ms,
            freshness_status=freshness_status,
            market_id=summary.market_id,
            question=summary.question,
            outcome=selected.outcome,
            token_id=selected.token_id,
            book=book,
            warnings=warnings,
        )

    async def signal(
        self, market_id: str, outcome: str, history: list[str]
    ) -> PredictionMarketSnapshotData:
        snapshot = await self.snapshot(market_id, outcome)
        snapshot.signal = _signal(snapshot.book, history)
        return snapshot

    async def _get_cached(
        self, cache_key: str, url: str, params: dict[str, Any] | None
    ) -> tuple[dict[str, Any], int]:
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached and cached.expires_at > now:
            return cached.value, cached.observed_at_ms
        async with self._lock:
            cached = self._cache.get(cache_key)
            if cached and cached.expires_at > time.monotonic():
                return cached.value, cached.observed_at_ms
            payload = await self._transport.get(url, params)
            if not isinstance(payload, dict):
                raise ValueError("Public provider returned an invalid response")
            observed_at_ms = int(time.time() * 1000)
            self._cache[cache_key] = _CacheValue(
                value=payload,
                observed_at_ms=observed_at_ms,
                expires_at=time.monotonic() + CACHE_TTL_S,
            )
            return payload, observed_at_ms


def _market_summary(raw: Any) -> PredictionMarketSummaryData:
    if not isinstance(raw, dict):
        raise ValueError("market payload must be an object")
    market_id = str(raw.get("id") or "")
    question = str(raw.get("question") or "")
    if not market_id or not question:
        raise ValueError("market payload is missing id or question")
    outcomes = _json_list(raw.get("outcomes"), "outcomes")
    token_ids = _json_list(raw.get("clobTokenIds"), "clobTokenIds")
    if len(outcomes) != len(token_ids) or not outcomes:
        raise ValueError("market outcomes and CLOB token IDs must have equal non-zero length")
    prices = _json_list(raw.get("outcomePrices", "[]"), "outcomePrices")
    return PredictionMarketSummaryData(
        market_id=market_id,
        slug=str(raw.get("slug") or ""),
        question=question,
        active=bool(raw.get("active", False)),
        closed=bool(raw.get("closed", False)),
        outcomes=[
            PredictionMarketOutcomeData(
                outcome=str(name),
                token_id=str(token_ids[index]),
                price=str(prices[index]) if index < len(prices) else None,
            )
            for index, name in enumerate(outcomes)
        ],
    )


def _json_list(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError(f"market {field_name} must be a JSON array")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"market {field_name} is malformed") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"market {field_name} must be a JSON array")
    return parsed


def _normalize_book(raw: Any) -> PredictionMarketOrderBookData:
    if not isinstance(raw, dict):
        raise ValueError("order book payload must be an object")
    bids = _levels(raw.get("bids"), reverse=True)
    asks = _levels(raw.get("asks"), reverse=False)
    best_bid = _decimal(bids[0].price) if bids else None
    best_ask = _decimal(asks[0].price) if asks else None
    status = "valid"
    reason = None
    if not bids and not asks:
        status, reason = "empty", "No visible bid or ask levels."
    elif not bids or not asks:
        status, reason = "one_sided", "Both bid and ask liquidity are required."
    elif best_bid is not None and best_ask is not None and best_bid >= best_ask:
        status, reason = "crossed", "Best bid is not below best ask."
    midpoint = (best_bid + best_ask) / 2 if status == "valid" else None
    microprice = _microprice(bids, asks) if status == "valid" else None
    return PredictionMarketOrderBookData(
        bids=bids,
        asks=asks,
        best_bid=_decimal_string(best_bid),
        best_ask=_decimal_string(best_ask),
        midpoint=_decimal_string(midpoint),
        microprice=_decimal_string(microprice),
        health=PredictionMarketBookHealthData(
            status=status,
            reason=reason,
            crossed=status == "crossed",
            one_sided=status == "one_sided",
            bid_levels=len(bids),
            ask_levels=len(asks),
        ),
    )


def _levels(value: Any, reverse: bool) -> list[PredictionMarketBookLevelData]:
    if not isinstance(value, list):
        raise ValueError("order book levels must be lists")
    parsed: list[tuple[Decimal, Decimal]] = []
    for level in value:
        if not isinstance(level, dict):
            raise ValueError("order book level must be an object")
        price = _decimal(level.get("price"))
        size = _decimal(level.get("size"))
        if price is None or size is None or not Decimal("0") < price < Decimal("1") or size <= 0:
            raise ValueError("order book level has invalid probability price or size")
        parsed.append((price, size))
    if len({price for price, _ in parsed}) != len(parsed):
        raise ValueError("order book levels must have unique prices")
    parsed.sort(key=lambda item: item[0], reverse=reverse)
    return [PredictionMarketBookLevelData(price=str(price), size=str(size)) for price, size in parsed]


def _microprice(
    bids: list[PredictionMarketBookLevelData], asks: list[PredictionMarketBookLevelData]
) -> Decimal | None:
    bid_price, bid_size = _decimal(bids[0].price), _decimal(bids[0].size)
    ask_price, ask_size = _decimal(asks[0].price), _decimal(asks[0].size)
    if None in {bid_price, bid_size, ask_price, ask_size} or bid_size + ask_size <= 0:
        return None
    return (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)


def _signal(book: PredictionMarketOrderBookData, history: list[str]) -> PredictionMarketSignalData:
    reference = book.microprice or book.midpoint
    method = "microprice" if book.microprice else "midpoint" if book.midpoint else "unavailable"
    prices = [_decimal(item) for item in history]
    prices = [item for item in prices if item is not None and item > 0]
    if len(prices) < 3:
        return PredictionMarketSignalData(
            reference_price=reference,
            reference_method=method,
            observation_count=len(prices),
            volatility_status="insufficient_history",
        )
    returns = [math.log(float(right / left)) for left, right in zip(prices, prices[1:])]
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / len(returns)
    return PredictionMarketSignalData(
        reference_price=reference,
        reference_method=method,
        volatility=str(Decimal(str(math.sqrt(variance)))),
        observation_count=len(prices),
        volatility_status="available",
    )


def _decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _decimal_string(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _timestamp_ms(value: Any, fallback: int) -> int:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return fallback
    return timestamp if timestamp > 0 else fallback


def _freshness_status(age_ms: int) -> str:
    if age_ms <= 15_000:
        return "fresh"
    if age_ms <= 60_000:
        return "delayed"
    return "stale"


_prediction_market_service: PredictionMarketService | None = None


def get_prediction_market_service() -> PredictionMarketService:
    global _prediction_market_service
    if _prediction_market_service is None:
        _prediction_market_service = PredictionMarketService()
    return _prediction_market_service
