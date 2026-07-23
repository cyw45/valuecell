"""Import source-attributed WorldMonitor outputs into ValueCell research storage."""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger
from sqlalchemy.orm import Session

from valuecell.server.config.settings import get_settings
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.world_intelligence import WorldIntelligenceSnapshot


WORLD_MONITOR_FEEDS: dict[str, str] = {
    "risk_scores": "/api/intelligence/v1/get-risk-scores",
    "thermal_escalations": "/api/thermal/v1/list-thermal-escalations?max_items=25",
    "cross_source_signals": "/api/intelligence/v1/list-cross-source-signals",
    "market_implications": "/api/intelligence/v1/list-market-implications",
}


@dataclass(frozen=True)
class FetchedWorldMonitorFeed:
    """A successfully fetched WorldMonitor response."""

    feed: str
    payload: Any


@dataclass(frozen=True)
class WorldMonitorSyncReport:
    """Outcome of a single connector refresh cycle."""

    inserted_count: int
    unchanged_count: int
    errors: dict[str, str]


class WorldMonitorIntelligenceService:
    """Fetch, deduplicate, and store WorldMonitor evidence snapshots."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.WORLD_MONITOR_ENABLED
        self.base_url = settings.WORLD_MONITOR_API_URL.rstrip("/")
        self.api_token = settings.WORLD_MONITOR_API_TOKEN
        self.timeout_s = settings.WORLD_MONITOR_TIMEOUT_S

    async def sync(self) -> WorldMonitorSyncReport:
        """Import all configured feeds without failing the rest on one outage."""
        if not self.enabled:
            return WorldMonitorSyncReport(0, 0, {})

        fetched, errors = await self._fetch_all()
        if not fetched:
            return WorldMonitorSyncReport(0, 0, errors)

        session = get_database_manager().get_session()
        try:
            inserted_count, unchanged_count = self._persist(session, fetched)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return WorldMonitorSyncReport(inserted_count, unchanged_count, errors)

    async def _fetch_all(
        self,
    ) -> tuple[list[FetchedWorldMonitorFeed], dict[str, str]]:
        """Collect independent feeds concurrently while preserving feed-level errors."""
        timeout = httpx.Timeout(self.timeout_s)
        headers = {"User-Agent": "ValueCell-WorldMonitor-Connector/1.0"}
        if self.api_token:
            headers["X-WorldMonitor-Key"] = self.api_token
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        ) as client:
            results = await asyncio.gather(
                *(
                    self._fetch_feed(client, feed, path)
                    for feed, path in WORLD_MONITOR_FEEDS.items()
                ),
                return_exceptions=True,
            )

        fetched: list[FetchedWorldMonitorFeed] = []
        errors: dict[str, str] = {}
        for feed, result in zip(WORLD_MONITOR_FEEDS, results, strict=True):
            if isinstance(result, FetchedWorldMonitorFeed):
                fetched.append(result)
            else:
                message = str(result)
                errors[feed] = message
                logger.warning("WorldMonitor feed {} unavailable: {}", feed, message)
        return fetched, errors

    async def _fetch_feed(
        self,
        client: httpx.AsyncClient,
        feed: str,
        path: str,
    ) -> FetchedWorldMonitorFeed:
        """Fetch one endpoint and retain its response exactly as received."""
        response = await client.get(path)
        response.raise_for_status()
        return FetchedWorldMonitorFeed(feed=feed, payload=response.json())

    def _persist(
        self,
        session: Session,
        feeds: list[FetchedWorldMonitorFeed],
    ) -> tuple[int, int]:
        """Store only changed payloads so evidence history remains meaningful."""
        inserted_count = 0
        unchanged_count = 0
        for fetched in feeds:
            content_hash = _content_hash(fetched.payload)
            existing = (
                session.query(WorldIntelligenceSnapshot.id)
                .filter(
                    WorldIntelligenceSnapshot.feed == fetched.feed,
                    WorldIntelligenceSnapshot.content_hash == content_hash,
                )
                .first()
            )
            if existing is not None:
                unchanged_count += 1
                continue
            session.add(
                WorldIntelligenceSnapshot(
                    feed=fetched.feed,
                    content_hash=content_hash,
                    payload=fetched.payload,
                )
            )
            inserted_count += 1
        return inserted_count, unchanged_count

    def list_latest_snapshots(
        self,
        session: Session,
        feed: str | None,
        limit: int,
    ) -> list[WorldIntelligenceSnapshot]:
        """Return persisted evidence newest first for a research consumer."""
        query = session.query(WorldIntelligenceSnapshot)
        if feed is not None:
            query = query.filter(WorldIntelligenceSnapshot.feed == feed)
        return (
            query.order_by(WorldIntelligenceSnapshot.captured_at.desc())
            .limit(limit)
            .all()
        )

    def latest_snapshot_times(self, session: Session) -> dict[str, Any]:
        """Return the latest persisted timestamp for every configured feed."""
        latest: dict[str, Any] = {}
        for feed in WORLD_MONITOR_FEEDS:
            snapshot = (
                session.query(WorldIntelligenceSnapshot.captured_at)
                .filter(WorldIntelligenceSnapshot.feed == feed)
                .order_by(WorldIntelligenceSnapshot.captured_at.desc())
                .first()
            )
            latest[feed] = snapshot[0] if snapshot is not None else None
        return latest


def _content_hash(payload: Any) -> str:
    """Generate a stable content identity for JSON-compatible source payloads."""
    serialized = json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
