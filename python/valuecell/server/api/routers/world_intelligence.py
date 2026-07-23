"""Read-only API for WorldMonitor evidence imported into ValueCell."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.api.schemas.world_intelligence import (
    WorldIntelligenceFeedStatus,
    WorldIntelligenceSnapshotData,
    WorldIntelligenceSnapshotListData,
    WorldIntelligenceStatusData,
)
from valuecell.server.config.settings import get_settings
from valuecell.server.db.connection import get_db
from valuecell.server.services.world_intelligence_service import (
    WORLD_MONITOR_FEEDS,
    WorldMonitorIntelligenceService,
)


def create_world_intelligence_router() -> APIRouter:
    """Create research evidence endpoints backed by ValueCell's own database."""
    router = APIRouter(prefix="/world-intelligence", tags=["world-intelligence"])
    service = WorldMonitorIntelligenceService()

    @router.get(
        "/status",
        response_model=SuccessResponse[WorldIntelligenceStatusData],
    )
    async def world_intelligence_status(
        db: Session = Depends(get_db),
    ) -> SuccessResponse[WorldIntelligenceStatusData]:
        latest = service.latest_snapshot_times(db)
        feeds = [
            WorldIntelligenceFeedStatus(
                feed=feed,
                latest_snapshot_at=latest[feed],
            )
            for feed in WORLD_MONITOR_FEEDS
        ]
        return SuccessResponse.create(
            data=WorldIntelligenceStatusData(
                enabled=get_settings().WORLD_MONITOR_ENABLED,
                feeds=feeds,
            ),
            msg="WorldMonitor connector status retrieved",
        )

    @router.get(
        "/snapshots",
        response_model=SuccessResponse[WorldIntelligenceSnapshotListData],
    )
    async def list_world_intelligence_snapshots(
        feed: str | None = Query(None),
        limit: int = Query(20, ge=1, le=100),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[WorldIntelligenceSnapshotListData]:
        if feed is not None and feed not in WORLD_MONITOR_FEEDS:
            raise HTTPException(status_code=400, detail="Unknown WorldMonitor feed")
        snapshots = service.list_latest_snapshots(db, feed, limit)
        data = WorldIntelligenceSnapshotListData(
            snapshots=[
                WorldIntelligenceSnapshotData(
                    id=snapshot.id,
                    feed=snapshot.feed,
                    payload=snapshot.payload,
                    captured_at=snapshot.captured_at,
                )
                for snapshot in snapshots
            ]
        )
        return SuccessResponse.create(
            data=data,
            msg="WorldMonitor evidence snapshots retrieved",
        )

    return router
