"""Schemas for persisted WorldMonitor intelligence evidence."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WorldIntelligenceFeedStatus(BaseModel):
    """Freshness status for one imported WorldMonitor feed."""

    feed: str
    latest_snapshot_at: datetime | None = None


class WorldIntelligenceStatusData(BaseModel):
    """ValueCell's view of the WorldMonitor connector health."""

    enabled: bool
    feeds: list[WorldIntelligenceFeedStatus]


class WorldIntelligenceSummaryMetric(BaseModel):
    """A compact fact shown in the Chinese intelligence brief."""

    label: str
    value: str


class WorldIntelligenceSummaryData(BaseModel):
    """Deterministic Chinese summary derived from a source snapshot."""

    title: str
    level: str
    highlights: list[str] = Field(default_factory=list)
    metrics: list[WorldIntelligenceSummaryMetric] = Field(default_factory=list)
    data_notice: str | None = None


class WorldIntelligenceSnapshotData(BaseModel):
    """A stored source response retained for research traceability."""

    id: int
    feed: str
    payload: Any
    summary_zh: WorldIntelligenceSummaryData
    captured_at: datetime


class WorldIntelligenceSnapshotListData(BaseModel):
    """Latest WorldMonitor evidence snapshots."""

    snapshots: list[WorldIntelligenceSnapshotData] = Field(default_factory=list)
