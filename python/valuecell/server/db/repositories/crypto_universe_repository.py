"""Persistence for the exchange-derived crypto symbol catalogue."""

from __future__ import annotations

import time
from datetime import datetime

from sqlalchemy.orm import Session

from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.crypto_universe import (
    CryptoSymbolExclusion,
    CryptoSymbolUniverse,
    CryptoSymbolUniverseEntry,
)

# The market-data service validates requested symbols on the request path, so the
# active catalogue is cached in-process instead of hitting the database per call.
ACTIVE_SYMBOL_CACHE_TTL_S = 60.0
_active_symbol_cache: tuple[float, tuple[str, ...]] | None = None


def invalidate_active_symbol_cache() -> None:
    """Drop the cached catalogue after a publish."""

    global _active_symbol_cache
    _active_symbol_cache = None


class CryptoSymbolUniverseRepository:
    """Read and publish immutable catalogue versions. No tenant scoping."""

    def __init__(self, db_session: Session | None = None) -> None:
        self.db_session = db_session

    def _get_session(self) -> Session:
        return self.db_session or get_database_manager().get_session()

    def active_universe(self) -> CryptoSymbolUniverse | None:
        session = self._get_session()
        try:
            universe = (
                session.query(CryptoSymbolUniverse)
                .filter(CryptoSymbolUniverse.status == "active")
                .order_by(CryptoSymbolUniverse.version.desc())
                .first()
            )
            if universe is not None:
                session.expunge(universe)
            return universe
        finally:
            if self.db_session is None:
                session.close()

    def latest_universe(self) -> CryptoSymbolUniverse | None:
        session = self._get_session()
        try:
            universe = (
                session.query(CryptoSymbolUniverse)
                .order_by(CryptoSymbolUniverse.version.desc())
                .first()
            )
            if universe is not None:
                session.expunge(universe)
            return universe
        finally:
            if self.db_session is None:
                session.close()

    def list_universes(self, limit: int = 12) -> list[CryptoSymbolUniverse]:
        session = self._get_session()
        try:
            rows = (
                session.query(CryptoSymbolUniverse)
                .order_by(CryptoSymbolUniverse.version.desc())
                .limit(limit)
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            if self.db_session is None:
                session.close()

    def entries(self, universe_id: int) -> list[CryptoSymbolUniverseEntry]:
        session = self._get_session()
        try:
            rows = (
                session.query(CryptoSymbolUniverseEntry)
                .filter(CryptoSymbolUniverseEntry.universe_id == universe_id)
                .order_by(
                    CryptoSymbolUniverseEntry.average_quote_volume_30d.desc(),
                    CryptoSymbolUniverseEntry.symbol.asc(),
                )
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            if self.db_session is None:
                session.close()

    def active_entries(self) -> list[CryptoSymbolUniverseEntry]:
        universe = self.active_universe()
        if universe is None:
            return []
        return self.entries(universe.id)

    def active_symbols(self) -> list[str]:
        return [
            entry.symbol
            for entry in self.active_entries()
            if entry.state == "admitted"
        ]

    def excluded_symbols(self) -> dict[str, CryptoSymbolExclusion]:
        session = self._get_session()
        try:
            rows = session.query(CryptoSymbolExclusion).all()
            for row in rows:
                session.expunge(row)
            return {row.symbol: row for row in rows}
        finally:
            if self.db_session is None:
                session.close()

    def publish(
        self,
        *,
        observed_at: datetime,
        source: str,
        quote_asset: str,
        min_listing_age_days: int,
        min_average_quote_volume_30d: float,
        reason_detail: str | None,
        entries: list[dict[str, object]],
    ) -> int:
        """Publish one immutable version and return its version number.

        The previous active row is superseded in the same transaction so a
        reader can never observe two active versions or none.
        """

        session = self._get_session()
        try:
            current = (
                session.query(CryptoSymbolUniverse)
                .order_by(CryptoSymbolUniverse.version.desc())
                .first()
            )
            next_version = 1 if current is None else int(current.version) + 1
            session.query(CryptoSymbolUniverse).filter(
                CryptoSymbolUniverse.status == "active"
            ).update({"status": "superseded"}, synchronize_session=False)

            admitted = [entry for entry in entries if entry["state"] == "admitted"]
            added = [entry for entry in entries if entry["decision"] == "added"]
            removed = [entry for entry in entries if entry["decision"] == "removed"]
            retained = [entry for entry in entries if entry["decision"] == "retained"]

            universe = CryptoSymbolUniverse(
                version=next_version,
                status="active",
                source=source,
                quote_asset=quote_asset,
                observed_at=observed_at,
                evaluated_count=len(entries),
                admitted_count=len(admitted),
                added_count=len(added),
                removed_count=len(removed),
                retained_count=len(retained),
                min_listing_age_days=min_listing_age_days,
                min_average_quote_volume_30d=min_average_quote_volume_30d,
                reason_detail=reason_detail,
            )
            session.add(universe)
            session.flush()
            session.add_all(
                [
                    CryptoSymbolUniverseEntry(
                        universe_id=universe.id,
                        symbol=str(entry["symbol"]),
                        state=str(entry["state"]),
                        decision=str(entry["decision"]),
                        reason_code=str(entry["reason_code"]),
                        reason_detail=entry.get("reason_detail"),
                        permanent_exclusion=bool(entry.get("permanent_exclusion", False)),
                        listed_at=entry.get("listed_at"),
                        listing_age_days=entry.get("listing_age_days"),
                        average_quote_volume_30d=entry.get("average_quote_volume_30d"),
                        quote_volume_24h=entry.get("quote_volume_24h"),
                        price_quote=entry.get("price_quote"),
                        observed_at=observed_at,
                    )
                    for entry in entries
                ]
            )
            session.commit()
            invalidate_active_symbol_cache()
            return next_version
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def record_exclusions(
        self,
        rows: list[dict[str, object]],
        *,
        observed_at: datetime,
    ) -> int:
        """Persist symbols the sync must never re-admit on its own."""

        if not rows:
            return 0
        session = self._get_session()
        try:
            existing = {
                row[0] for row in session.query(CryptoSymbolExclusion.symbol).all()
            }
            written = 0
            for entry in rows:
                symbol = str(entry["symbol"])
                if symbol in existing:
                    continue
                session.add(
                    CryptoSymbolExclusion(
                        symbol=symbol,
                        reason_code=str(entry["reason_code"]),
                        reason_detail=entry.get("reason_detail"),
                        observed_at=observed_at,
                    )
                )
                written += 1
            session.commit()
            return written
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()


def cached_active_symbols(repository: CryptoSymbolUniverseRepository | None = None) -> tuple[str, ...]:
    """Return the active admitted symbols, cached briefly for request paths."""

    global _active_symbol_cache
    now = time.monotonic()
    if _active_symbol_cache is not None:
        cached_at, symbols = _active_symbol_cache
        if now - cached_at < ACTIVE_SYMBOL_CACHE_TTL_S:
            return symbols
    try:
        symbols = tuple((repository or CryptoSymbolUniverseRepository()).active_symbols())
    except Exception:
        # A catalogue read must never break market-data validation; the caller
        # falls back to the seed list when the storage is unavailable.
        return ()
    _active_symbol_cache = (now, symbols)
    return symbols
