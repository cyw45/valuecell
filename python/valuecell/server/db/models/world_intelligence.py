"""Persistent, source-attributed snapshots imported from WorldMonitor."""

from sqlalchemy import Column, DateTime, Integer, JSON, String, UniqueConstraint
from sqlalchemy.sql import func

from .base import Base


class WorldIntelligenceSnapshot(Base):
    """A deduplicated response from one WorldMonitor intelligence feed."""

    __tablename__ = "world_intelligence_snapshots"
    __table_args__ = (
        UniqueConstraint("feed", "content_hash", name="uq_world_intelligence_feed_hash"),
    )

    id = Column(Integer, primary_key=True, index=True)
    feed = Column(String(64), nullable=False, index=True)
    content_hash = Column(String(64), nullable=False)
    payload = Column(JSON, nullable=False)
    captured_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
