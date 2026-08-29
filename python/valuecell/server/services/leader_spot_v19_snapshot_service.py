"""Background-only V19 ranking and market snapshot collection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Sequence
from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.leader_spot_v19_snapshots import (
    LeaderSpotV19MarketInput,
    LeaderSpotV19MarketSnapshotProvider,
    LeaderSpotV19RankingProvider,
    LeaderSpotV19RankingSnapshot,
)
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19CandidateSnapshot,
    LeaderSpotV19MarketSnapshot,
)


@dataclass(frozen=True, slots=True)
class LeaderSpotV19CollectionResult:
    """Persisted collection outcome used by a future scheduler tick."""

    snapshot_group_id: str
    ranking_snapshot_id: str
    market_snapshot_count: int
    candidate_count: int
    accepted_candidate_count: int
    data_state: str
    ranking_fresh: bool


class LeaderSpotV19SnapshotCollector:
    """Collect and persist V19 source facts without deciding or executing trades."""

    _INTERVALS: tuple[str, ...] = ("1m", "5m", "15m")

    def __init__(self, session: Session) -> None:
        self._session = session

    async def collect(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        ranking_provider: LeaderSpotV19RankingProvider,
        market_provider: LeaderSpotV19MarketSnapshotProvider,
        now: datetime | None = None,
    ) -> LeaderSpotV19CollectionResult:
        """Fetch provider data, persist it, and return only durable collection facts.

        Provider calls are intentionally injected. The caller must run this method from
        a background worker; API routes never instantiate or invoke this collector.
        """

        observed_now = (now or datetime.now(UTC)).astimezone(UTC)
        ranking = await ranking_provider.fetch_ranking()
        ranking_fresh = self._is_fresh(
            observed_at=ranking.observed_at,
            expires_at=ranking.expires_at,
            now=observed_now,
        )
        data_state = "DATA_OK" if ranking_fresh and ranking.completeness == "complete" else "DATA_UNSAFE"
        snapshot_group_id = str(uuid4())
        self._persist_ranking(
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            batch_id=batch_id,
            snapshot_group_id=snapshot_group_id,
            ranking=ranking,
            now=observed_now,
        )

        market_inputs: list[LeaderSpotV19MarketInput] = []
        if ranking_fresh and ranking.completeness == "complete":
            market_inputs = await market_provider.fetch_market_inputs(
                [item.symbol for item in ranking.items if item.spot_tradable],
                self._INTERVALS,
            )
            self._persist_market_inputs(
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                snapshot_group_id=snapshot_group_id,
                inputs=market_inputs,
                now=observed_now,
            )

        accepted = self._persist_candidates(
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            batch_id=batch_id,
            snapshot_group_id=snapshot_group_id,
            ranking=ranking,
            ranking_fresh=ranking_fresh,
            data_state=data_state,
            now=observed_now,
        )
        self._session.commit()
        return LeaderSpotV19CollectionResult(
            snapshot_group_id=snapshot_group_id,
            ranking_snapshot_id=ranking.source_snapshot_id,
            market_snapshot_count=len(market_inputs),
            candidate_count=len(ranking.items),
            accepted_candidate_count=accepted,
            data_state=data_state,
            ranking_fresh=ranking_fresh,
        )

    def _persist_ranking(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        snapshot_group_id: str,
        ranking: LeaderSpotV19RankingSnapshot,
        now: datetime,
    ) -> None:
        self._session.add(
            LeaderSpotV19MarketSnapshot(
                snapshot_id=str(uuid4()),
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                source=ranking.source,
                snapshot_kind="ranking",
                symbol=None,
                payload={
                    "snapshot_group_id": snapshot_group_id,
                    "source_snapshot_id": ranking.source_snapshot_id,
                    "items": ranking.model_dump(mode="json")["items"],
                    "completeness": ranking.completeness,
                    "collected_at": now.isoformat(),
                },
                freshness="fresh"
                if self._is_fresh(
                    observed_at=ranking.observed_at,
                    expires_at=ranking.expires_at,
                    now=now,
                )
                else "unsafe",
                observed_at=ranking.observed_at,
                expires_at=ranking.expires_at,
            )
        )

    def _persist_market_inputs(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        snapshot_group_id: str,
        inputs: Sequence[LeaderSpotV19MarketInput],
        now: datetime,
    ) -> None:
        for item in inputs:
            self._session.add(
                LeaderSpotV19MarketSnapshot(
                    snapshot_id=str(uuid4()),
                    tenant_id=tenant_id,
                    strategy_id=strategy_id,
                    batch_id=batch_id,
                    source=item.source,
                    snapshot_kind=f"market_{item.interval}",
                    symbol=item.symbol,
                    payload={
                        "snapshot_group_id": snapshot_group_id,
                        **item.model_dump(mode="json"),
                    },
                    freshness="fresh"
                    if self._is_fresh(
                        observed_at=item.observed_at,
                        expires_at=item.expires_at,
                        now=now,
                    )
                    else "stale",
                    observed_at=item.observed_at,
                    expires_at=item.expires_at,
                )
            )

    def _persist_candidates(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        snapshot_group_id: str,
        ranking: LeaderSpotV19RankingSnapshot,
        ranking_fresh: bool,
        data_state: str,
        now: datetime,
    ) -> int:
        accepted_count = 0
        for item in ranking.items:
            accepted = ranking_fresh and ranking.completeness == "complete" and item.spot_tradable
            reason_code = None if accepted else self._candidate_rejection_reason(
                ranking_fresh=ranking_fresh,
                complete=ranking.completeness == "complete",
                spot_tradable=item.spot_tradable,
            )
            if accepted:
                accepted_count += 1
            self._session.add(
                LeaderSpotV19CandidateSnapshot(
                    candidate_id=str(uuid4()),
                    tenant_id=tenant_id,
                    strategy_id=strategy_id,
                    batch_id=batch_id,
                    snapshot_group_id=snapshot_group_id,
                    source=ranking.source,
                    symbol=item.symbol,
                    source_rank=item.rank,
                    market_state="M0" if data_state == "DATA_UNSAFE" else "M3",
                    data_state=data_state,
                    funnel_stage="ranking",
                    accepted=accepted,
                    score=None,
                    reason_code=reason_code,
                    facts={
                        "quote_volume_24h": item.quote_volume_24h,
                        "listing_at": item.listing_at.isoformat() if item.listing_at else None,
                        "spot_tradable": item.spot_tradable,
                    },
                    observed_at=ranking.observed_at,
                )
            )
        return accepted_count

    @staticmethod
    def _candidate_rejection_reason(
        *, ranking_fresh: bool, complete: bool, spot_tradable: bool
    ) -> str:
        if not ranking_fresh:
            return "ranking_snapshot_expired"
        if not complete:
            return "ranking_snapshot_incomplete"
        if not spot_tradable:
            return "symbol_not_spot_tradable"
        return "candidate_rejected"

    @staticmethod
    def _is_fresh(*, observed_at: datetime, expires_at: datetime, now: datetime) -> bool:
        return observed_at <= now <= expires_at
