"""Tests for WorldMonitor evidence persistence."""

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.config.settings import get_settings
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


@pytest.mark.asyncio
async def test_fetch_all_sends_worldmonitor_api_key(monkeypatch) -> None:
    """The connector must authenticate directly against the sidecar API."""
    monkeypatch.setenv("WORLD_MONITOR_API_TOKEN", "test-sidecar-token")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-WorldMonitor-Key"] == "test-sidecar-token"
        return httpx.Response(200, json={"ok": True})

    service = WorldMonitorIntelligenceService()
    assert service.api_token == "test-sidecar-token"
    service.base_url = "http://worldmonitor.test"
    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        return original_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    try:
        fetched, errors = await service._fetch_all()

        assert errors == {}
        assert len(fetched) == 4
    finally:
        get_settings.cache_clear()
