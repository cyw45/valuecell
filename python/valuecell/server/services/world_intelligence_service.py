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


@dataclass(frozen=True)
class WorldIntelligenceMetric:
    """One compact metric rendered in the Chinese intelligence brief."""

    label: str
    value: str


@dataclass(frozen=True)
class WorldIntelligenceSummary:
    """Deterministic Chinese-language interpretation of one source payload."""

    title: str
    level: str
    highlights: list[str]
    metrics: list[WorldIntelligenceMetric]
    data_notice: str | None = None


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
        *,
        latest_per_feed: bool = False,
    ) -> list[WorldIntelligenceSnapshot]:
        """Return persisted evidence newest first for a research consumer."""
        if latest_per_feed and feed is not None:
            snapshot = (
                session.query(WorldIntelligenceSnapshot)
                .filter(WorldIntelligenceSnapshot.feed == feed)
                .order_by(
                    WorldIntelligenceSnapshot.captured_at.desc(),
                    WorldIntelligenceSnapshot.id.desc(),
                )
                .first()
            )
            return [snapshot] if snapshot is not None else []
        if latest_per_feed and feed is None:
            snapshots = []
            for feed_name in WORLD_MONITOR_FEEDS:
                snapshot = (
                    session.query(WorldIntelligenceSnapshot)
                    .filter(WorldIntelligenceSnapshot.feed == feed_name)
                    .order_by(
                        WorldIntelligenceSnapshot.captured_at.desc(),
                        WorldIntelligenceSnapshot.id.desc(),
                    )
                    .first()
                )
                if snapshot is not None:
                    snapshots.append(snapshot)
            return sorted(
                snapshots, key=lambda item: item.captured_at, reverse=True
            )[:limit]

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


_REGION_NAMES = {
    "AF": "阿富汗",
    "BR": "巴西",
    "CN": "中国",
    "CU": "古巴",
    "EG": "埃及",
    "IL": "以色列",
    "IN": "印度",
    "IQ": "伊拉克",
    "IR": "伊朗",
    "KP": "朝鲜",
    "LB": "黎巴嫩",
    "MM": "缅甸",
    "MX": "墨西哥",
    "PK": "巴基斯坦",
    "RU": "俄罗斯",
    "SY": "叙利亚",
    "TR": "土耳其",
    "TW": "中国台湾",
    "UA": "乌克兰",
    "VE": "委内瑞拉",
    "YE": "也门",
}

_SEVERITY_NAMES = {
    "CRITICAL": "严重",
    "HIGH": "高",
    "MEDIUM": "中等",
    "LOW": "低",
}

_THEATER_NAMES = {
    "Global Markets": "全球市场",
    "global": "全球",
}


_SIGNAL_TRANSLATIONS = {
    "SEC announces an enforcement leadership change": (
        "美国证券交易委员会（SEC）宣布执法部门领导层发生人事变动。"
    ),
    "SEC: SEC Announces Departure of Principal Deputy Director of Enforcement Sam Waldon": (
        "美国证券交易委员会（SEC）宣布执法部门首席副主任 Sam Waldon 离任。"
    ),
}


def summarize_world_intelligence(
    feed: str, payload: Any
) -> WorldIntelligenceSummary:
    """Build a concise Chinese brief without inventing facts beyond the payload."""
    source = payload if isinstance(payload, dict) else {}
    if feed == "risk_scores":
        scores = sorted(
            _dict_items(source.get("ciiScores")),
            key=lambda item: _safe_float(item.get("combinedScore")),
            reverse=True,
        )
        strategic = next(
            (
                item
                for item in _dict_items(source.get("strategicRisks"))
                if item.get("region") == "global"
            ),
            {},
        )
        risk_score = strategic.get("score", "--")
        top_risks = "、".join(
            f"{_REGION_NAMES.get(str(item.get('region')), item.get('region', '未知地区'))} "
            f"{item.get('combinedScore', '--')}"
            for item in scores[:5]
        )
        notice_parts = []
        if source.get("degraded"):
            notice_parts.append("当前部分上游数据不可用，结果包含备用基线数据")
        if source.get("stale"):
            notice_parts.append("数据已超过新鲜度窗口")
        notice = "；".join(notice_parts) + "。" if notice_parts else None
        return WorldIntelligenceSummary(
            title="全球风险概览",
            level=_severity_name(strategic.get("level")),
            highlights=[f"当前风险评分最高的地区：{top_risks}。"] if top_risks else [],
            metrics=[
                WorldIntelligenceMetric("全球风险分", str(risk_score)),
                WorldIntelligenceMetric("覆盖地区", str(len(scores))),
            ],
            data_notice=notice,
        )

    if feed == "thermal_escalations":
        clusters = _dict_items(source.get("clusters"))
        countries: list[str] = []
        for item in clusters:
            name = _REGION_NAMES.get(
                str(item.get("countryCode")), str(item.get("countryName", "未知地区"))
            )
            if name not in countries:
                countries.append(name)
        high_relevance = sum(
            str(item.get("strategicRelevance", "")).endswith("HIGH")
            for item in clusters
        )
        return WorldIntelligenceSummary(
            title="热异常升级监测",
            level="高" if high_relevance else "观察",
            highlights=[f"热异常主要分布于：{'、'.join(countries[:5])}。"]
            if countries
            else [],
            metrics=[
                WorldIntelligenceMetric("异常簇", str(len(clusters))),
                WorldIntelligenceMetric("高战略相关", str(high_relevance)),
                WorldIntelligenceMetric(
                    "观察窗口", f"{source.get('observationWindowHours', '--')} 小时"
                ),
            ],
        )

    if feed == "cross_source_signals":
        signals = _dict_items(source.get("signals"))
        highlights = [
            _format_signal_highlight(item)
            for item in signals[:5]
        ]
        level = max(
            (_severity_name(item.get("severity")) for item in signals),
            key=_severity_rank,
            default="观察",
        )
        return WorldIntelligenceSummary(
            title="跨源事件信号",
            level=level,
            highlights=highlights,
            metrics=[WorldIntelligenceMetric("有效信号", str(len(signals)))],
        )

    if feed == "market_implications":
        cards = _dict_items(source.get("cards"))
        if not cards:
            cards = _dict_items(source.get("implications"))
        highlights = [_format_market_highlight(item) for item in cards[:5]]
        return WorldIntelligenceSummary(
            title="市场影响研判",
            level="观察",
            highlights=highlights,
            metrics=[WorldIntelligenceMetric("影响线索", str(len(cards)))],
            data_notice=None if cards else "当前周期未形成可展示的市场影响线索。",
        )

    return WorldIntelligenceSummary(
        title="全球情报快照",
        level="观察",
        highlights=[],
        metrics=[],
        data_notice="暂不支持该情报类型的中文摘要。",
    )


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_market_highlight(item: dict[str, Any]) -> str:
    original = str(
        item.get("summary")
        or item.get("title")
        or item.get("implication")
        or "未提供市场影响摘要"
    )
    if any("\u4e00" <= character <= "\u9fff" for character in original):
        return original
    return f"市场影响线索需结合原始仪表盘复核。（原文：{original}）"


def _format_signal_highlight(item: dict[str, Any]) -> str:
    theater = _THEATER_NAMES.get(
        str(item.get("theater")), str(item.get("theater", "全球"))
    )
    original = str(item.get("summary", "未提供事件摘要"))
    translated = _SIGNAL_TRANSLATIONS.get(original)
    if translated is None:
        return f"{theater}：事件来源提供英文摘要，请在原始仪表盘复核。（原文：{original}）"
    return f"{theater}：{translated}（原文：{original}）"


def _severity_name(value: Any) -> str:
    normalized = str(value or "").upper()
    return next(
        (label for key, label in _SEVERITY_NAMES.items() if key in normalized),
        "观察",
    )


def _severity_rank(value: str) -> int:
    return {"观察": 0, "低": 1, "中等": 2, "高": 3, "严重": 4}.get(value, 0)


def _content_hash(payload: Any) -> str:
    """Generate a stable content identity for JSON-compatible source payloads."""
    serialized = json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
