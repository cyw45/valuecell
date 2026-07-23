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
    summarize_world_intelligence,
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


def test_summarize_risk_scores_in_chinese_and_discloses_degraded_data() -> None:
    summary = summarize_world_intelligence(
        "risk_scores",
        {
            "ciiScores": [
                {"region": "UA", "combinedScore": 60, "trend": "TREND_DIRECTION_STABLE"},
                {"region": "SY", "combinedScore": 55, "trend": "TREND_DIRECTION_RISING"},
            ],
            "strategicRisks": [
                {"region": "global", "level": "SEVERITY_LEVEL_MEDIUM", "score": 57}
            ],
            "degraded": True,
            "stale": False,
        },
    )

    assert summary.title == "全球风险概览"
    assert summary.level == "中等"
    assert summary.metrics[0].label == "全球风险分"
    assert summary.metrics[0].value == "57"
    assert "乌克兰 60" in summary.highlights[0]
    assert "叙利亚 55" in summary.highlights[0]
    assert summary.data_notice == "当前部分上游数据不可用，结果包含备用基线数据。"


def test_summarize_cross_source_signal_uses_chinese_context() -> None:
    summary = summarize_world_intelligence(
        "cross_source_signals",
        {
            "signals": [
                {
                    "theater": "Global Markets",
                    "summary": "SEC announces an enforcement leadership change",
                    "severity": "CROSS_SOURCE_SIGNAL_SEVERITY_HIGH",
                }
            ],
            "compositeCount": 1,
        },
    )

    assert summary.title == "跨源事件信号"
    assert summary.level == "高"
    assert summary.metrics[0].value == "1"
    assert summary.highlights == [
        "全球市场：美国证券交易委员会（SEC）宣布执法部门领导层发生人事变动。"
        "（原文：SEC announces an enforcement leadership change）"
    ]


def test_summarize_malformed_payloads_degrade_without_raising() -> None:
    risk = summarize_world_intelligence(
        "risk_scores",
        {
            "ciiScores": [
                {"region": "UA", "combinedScore": None},
                {"region": "SY", "combinedScore": "not-a-number"},
            ],
            "strategicRisks": None,
            "degraded": True,
            "stale": True,
        },
    )
    thermal = summarize_world_intelligence(
        "thermal_escalations", {"clusters": None}
    )
    signal = summarize_world_intelligence(
        "cross_source_signals", {"signals": {"unexpected": "object"}}
    )
    market = summarize_world_intelligence(
        "market_implications", {"cards": [{"summary": "Oil supply risk rises"}]}
    )

    assert risk.metrics[1].value == "2"
    assert "备用基线" in (risk.data_notice or "")
    assert "超过新鲜度窗口" in (risk.data_notice or "")
    assert thermal.metrics[0].value == "0"
    assert signal.metrics[0].value == "0"
    assert market.highlights == [
        "市场影响线索需结合原始仪表盘复核。（原文：Oil supply risk rises）"
    ]
