"""Helpers for filtering default spot symbols against exchange spot markets."""

from __future__ import annotations

from collections.abc import Iterable

from loguru import logger

from valuecell.agents.common.trading.models import MarketType
from valuecell.agents.common.trading.utils import get_exchange_cls


def _normalize_candidates(symbols: Iterable[str]) -> list[str]:
    return [str(symbol).upper() for symbol in symbols if str(symbol).strip()]


def _build_spot_symbol_aliases(markets: dict) -> set[str]:
    aliases: set[str] = set()

    for market in markets.values():
        if not isinstance(market, dict):
            continue
        if not market.get("spot", False):
            continue

        symbol = market.get("symbol")
        base = market.get("base")
        quote = market.get("quote")
        market_id = market.get("id")

        for raw in (symbol, market_id):
            if not raw:
                continue
            normalized = str(raw).upper().replace("/", "-").split(":")[0]
            aliases.add(normalized)

        if base and quote:
            aliases.add(f"{str(base).upper()}-{str(quote).upper()}")

    return aliases


async def filter_supported_spot_symbols(
    exchange_id: str | None,
    symbols: Iterable[str],
    market_type: MarketType,
) -> list[str]:
    """Return only symbols that exist as spot pairs on the selected exchange.

    If the market list cannot be loaded because of network or exchange errors,
    the function falls back to the original symbol list so strategy startup is
    not blocked by temporary connectivity issues.
    """

    candidate_symbols = _normalize_candidates(symbols)
    if not candidate_symbols:
        return []
    if market_type != MarketType.SPOT:
        return candidate_symbols
    if not exchange_id:
        return candidate_symbols

    try:
        exchange_cls = get_exchange_cls(exchange_id)
        exchange = exchange_cls({"newUpdates": False})
    except Exception as exc:
        logger.warning(
            "Failed to create exchange class for {exchange_id}; keeping original symbols. Error: {error}",
            exchange_id=exchange_id,
            error=str(exc),
        )
        return candidate_symbols

    try:
        markets = await exchange.load_markets()
        aliases = _build_spot_symbol_aliases(markets or {})
        filtered = [symbol for symbol in candidate_symbols if symbol in aliases]
        removed = [symbol for symbol in candidate_symbols if symbol not in aliases]

        if removed:
            logger.info(
                "Filtered unsupported spot symbols for {exchange_id}: {removed}",
                exchange_id=exchange_id,
                removed=removed,
            )

        if filtered:
            logger.info(
                "Kept {count}/{total} supported spot symbols for {exchange_id}",
                count=len(filtered),
                total=len(candidate_symbols),
                exchange_id=exchange_id,
            )
            return filtered

        logger.warning(
            "No candidate spot symbols matched {exchange_id}; keeping original list to avoid empty strategy universe",
            exchange_id=exchange_id,
        )
        return candidate_symbols
    except Exception as exc:
        logger.warning(
            "Failed to load spot markets for {exchange_id}; keeping original symbols. Error: {error}",
            exchange_id=exchange_id,
            error=str(exc),
        )
        return candidate_symbols
    finally:
        try:
            await exchange.close()
        except Exception:
            logger.warning(
                "Failed to close exchange connection while filtering symbols for {exchange_id}",
                exchange_id=exchange_id,
            )
