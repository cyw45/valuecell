"""Tests for WorldMonitor evidence persistence."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.db.models.base import Base
from valuecell.server.db.models.world_intelligence import WorldIntelligenceSnapshot
from valuecell.server.services.world_intelligence_service import (
    FetchedWorldMonitorFeed,
    WorldMonitorIntelligenceService,
)


def test_persist_keeps_only_changed_worldmonitor_payloads() -> None:
    """Repeated source payloads should not create misleading research history."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    service = WorldMonitorIntelligenceService()
    feed = FetchedWorldMonitorFeed(
        feed="risk_scores",
        payload={"scores": [{"country": "US", "score": 12}]},
    )

    assert service._persist(session, [feed]) == (1, 0)
    session.commit()
    assert service._persist(session, [feed]) == (0, 1)
    assert session.query(WorldIntelligenceSnapshot).count() == 1

    session.close()
