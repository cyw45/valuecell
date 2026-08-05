"""Tenant-scoped persistence boundary for immutable strategy validation runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, or_, update
from sqlalchemy.orm import Session

from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.rule_strategy import RuleStrategy
from valuecell.server.db.models.rule_strategy_validation import (
    RuleStrategyValidationDataset,
    RuleStrategyValidationFill,
    RuleStrategyValidationPoint,
    RuleStrategyValidationRun,
)


class RuleStrategyValidationRepository:
    """Persist and lease validation work without reading another tenant's rows."""

    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory or get_database_manager().get_session

    def create(
        self,
        run: RuleStrategyValidationRun,
        datasets: Iterable[RuleStrategyValidationDataset],
    ) -> RuleStrategyValidationRun:
        """Create the immutable request and all materialized data atomically."""

        dataset_rows = list(datasets)
        if any(
            row.run_id != run.run_id
            or row.tenant_id != run.tenant_id
            or row.strategy_id != run.strategy_id
            for row in dataset_rows
        ):
            raise ValueError("validation datasets must belong to their run and tenant")
        session = self._session_factory()
        try:
            session.add(run)
            session.add_all(dataset_rows)
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def strategy(self, strategy_id: str, tenant_id: str) -> RuleStrategy | None:
        """Read exactly one tenant-owned strategy for an immutable config snapshot."""

        session = self._session_factory()
        try:
            strategy = (
                session.query(RuleStrategy)
                .filter(
                    RuleStrategy.strategy_id == strategy_id,
                    RuleStrategy.tenant_id == tenant_id,
                )
                .first()
            )
            if strategy is not None:
                session.expunge(strategy)
            return strategy
        finally:
            session.close()

    def get(self, run_id: str, tenant_id: str) -> RuleStrategyValidationRun | None:
        session = self._session_factory()
        try:
            run = (
                session.query(RuleStrategyValidationRun)
                .filter_by(run_id=run_id, tenant_id=tenant_id)
                .first()
            )
            if run is not None:
                session.expunge(run)
            return run
        finally:
            session.close()

    def list(
        self,
        strategy_id: str,
        tenant_id: str,
        *,
        limit: int = 100,
    ) -> list[RuleStrategyValidationRun]:
        session = self._session_factory()
        try:
            runs = (
                session.query(RuleStrategyValidationRun)
                .filter_by(strategy_id=strategy_id, tenant_id=tenant_id)
                .order_by(
                    RuleStrategyValidationRun.created_at.desc(),
                    RuleStrategyValidationRun.id.desc(),
                )
                .limit(limit)
                .all()
            )
            for run in runs:
                session.expunge(run)
            return runs
        finally:
            session.close()

    def datasets(
        self, run_id: str, tenant_id: str
    ) -> list[RuleStrategyValidationDataset]:
        session = self._session_factory()
        try:
            rows = (
                session.query(RuleStrategyValidationDataset)
                .filter_by(run_id=run_id, tenant_id=tenant_id)
                .order_by(
                    RuleStrategyValidationDataset.symbol.asc(),
                    RuleStrategyValidationDataset.interval.asc(),
                    RuleStrategyValidationDataset.id.asc(),
                )
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()

    def points(
        self, run_id: str, tenant_id: str
    ) -> list[RuleStrategyValidationPoint]:
        session = self._session_factory()
        try:
            rows = (
                session.query(RuleStrategyValidationPoint)
                .filter_by(run_id=run_id, tenant_id=tenant_id)
                .order_by(RuleStrategyValidationPoint.sequence.asc())
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()

    def fills(
        self, run_id: str, tenant_id: str
    ) -> list[RuleStrategyValidationFill]:
        session = self._session_factory()
        try:
            rows = (
                session.query(RuleStrategyValidationFill)
                .filter_by(run_id=run_id, tenant_id=tenant_id)
                .order_by(RuleStrategyValidationFill.sequence.asc())
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            session.close()

    def claim(
        self,
        run_id: str,
        tenant_id: str,
        worker_id: str,
        *,
        now: datetime | None = None,
        lease_duration: timedelta = timedelta(minutes=5),
    ) -> RuleStrategyValidationRun | None:
        """Atomically claim one queued or expired validation run."""

        timestamp = _utc(now or datetime.now(UTC))
        session = self._session_factory()
        try:
            claimed = session.execute(
                update(RuleStrategyValidationRun)
                .where(
                    RuleStrategyValidationRun.run_id == run_id,
                    RuleStrategyValidationRun.tenant_id == tenant_id,
                    or_(
                        RuleStrategyValidationRun.status == "pending",
                        and_(
                            RuleStrategyValidationRun.status == "running",
                            RuleStrategyValidationRun.lease_expires_at.is_not(None),
                            RuleStrategyValidationRun.lease_expires_at < timestamp,
                        ),
                    ),
                )
                .values(
                    status="running",
                    worker_id=worker_id,
                    lease_expires_at=timestamp + lease_duration,
                    started_at=case(
                        (RuleStrategyValidationRun.started_at.is_(None), timestamp),
                        else_=RuleStrategyValidationRun.started_at,
                    ),
                )
            ).rowcount
            if claimed != 1:
                session.rollback()
                return None
            run = (
                session.query(RuleStrategyValidationRun)
                .filter_by(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    worker_id=worker_id,
                    status="running",
                )
                .first()
            )
            if run is None:
                session.rollback()
                return None
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def renew_lease(
        self,
        run_id: str,
        tenant_id: str,
        worker_id: str,
        *,
        now: datetime | None = None,
        lease_duration: timedelta = timedelta(minutes=5),
    ) -> bool:
        """Atomically renew only a still-valid lease owned by the caller."""

        timestamp = _utc(now or datetime.now(UTC))
        session = self._session_factory()
        try:
            renewed = session.execute(
                update(RuleStrategyValidationRun)
                .where(
                    RuleStrategyValidationRun.run_id == run_id,
                    RuleStrategyValidationRun.tenant_id == tenant_id,
                    RuleStrategyValidationRun.worker_id == worker_id,
                    RuleStrategyValidationRun.status == "running",
                    RuleStrategyValidationRun.lease_expires_at.is_not(None),
                    RuleStrategyValidationRun.lease_expires_at >= timestamp,
                )
                .values(lease_expires_at=timestamp + lease_duration)
            ).rowcount
            if renewed != 1:
                session.rollback()
                return False
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def cancel(
        self,
        run_id: str,
        tenant_id: str,
        *,
        now: datetime | None = None,
    ) -> RuleStrategyValidationRun | None:
        """Cancel pending work or request cooperative cancellation from its owner."""

        timestamp = _utc(now or datetime.now(UTC))
        session = self._session_factory()
        try:
            run = (
                session.query(RuleStrategyValidationRun)
                .filter_by(run_id=run_id, tenant_id=tenant_id)
                .with_for_update()
                .first()
            )
            if run is None:
                session.rollback()
                return None
            if run.status == "completed":
                session.expunge(run)
                session.rollback()
                return run
            if run.status == "pending":
                run.status = "cancelled"
                run.completed_at = timestamp
                run.worker_id = None
                run.lease_expires_at = None
            elif run.status == "running":
                run.cancel_requested_at = timestamp
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def cancel_requested(
        self, run_id: str, tenant_id: str, worker_id: str
    ) -> bool:
        session = self._session_factory()
        try:
            run = (
                session.query(RuleStrategyValidationRun.cancel_requested_at)
                .filter_by(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    worker_id=worker_id,
                    status="running",
                )
                .first()
            )
            return run is None or run[0] is not None
        finally:
            session.close()

    def finish(
        self,
        run_id: str,
        tenant_id: str,
        worker_id: str,
        *,
        points: Iterable[RuleStrategyValidationPoint],
        fills: Iterable[RuleStrategyValidationFill],
        metrics: dict[str, Any],
        artifact_fingerprint: str,
        now: datetime | None = None,
    ) -> RuleStrategyValidationRun | None:
        """Atomically append replay evidence and make a run immutable."""

        timestamp = _utc(now or datetime.now(UTC))
        point_rows = list(points)
        fill_rows = list(fills)
        session = self._session_factory()
        try:
            run = (
                session.query(RuleStrategyValidationRun)
                .filter_by(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    worker_id=worker_id,
                    status="running",
                )
                .with_for_update()
                .first()
            )
            if run is None or run.lease_expires_at is None or _utc(run.lease_expires_at) < timestamp:
                session.rollback()
                return None
            if run.cancel_requested_at is not None:
                run.status = "cancelled"
                run.completed_at = timestamp
                run.worker_id = None
                run.lease_expires_at = None
                session.commit()
                session.refresh(run)
                session.expunge(run)
                return run
            _validate_evidence_rows(run, point_rows, fill_rows)
            session.add_all(point_rows)
            session.add_all(fill_rows)
            run.metrics = metrics
            run.artifact_fingerprint = artifact_fingerprint
            run.status = "completed"
            run.completed_at = timestamp
            run.worker_id = None
            run.lease_expires_at = None
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def fail(
        self,
        run_id: str,
        tenant_id: str,
        worker_id: str,
        error_code: str,
        error_detail: str,
        *,
        now: datetime | None = None,
    ) -> RuleStrategyValidationRun | None:
        """Make a claimed run terminal without altering its immutable inputs."""

        timestamp = _utc(now or datetime.now(UTC))
        session = self._session_factory()
        try:
            run = (
                session.query(RuleStrategyValidationRun)
                .filter_by(
                    run_id=run_id,
                    tenant_id=tenant_id,
                    worker_id=worker_id,
                    status="running",
                )
                .with_for_update()
                .first()
            )
            if run is None:
                session.rollback()
                return None
            if run.cancel_requested_at is not None:
                run.status = "cancelled"
                run.completed_at = timestamp
            else:
                run.status = "failed"
                run.error_code = error_code[:96]
                run.error_detail = error_detail[:2_000]
                run.completed_at = timestamp
            run.worker_id = None
            run.lease_expires_at = None
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()




def _validate_evidence_rows(
    run: RuleStrategyValidationRun,
    points: list[RuleStrategyValidationPoint],
    fills: list[RuleStrategyValidationFill],
) -> None:
    if any(
        row.run_id != run.run_id
        or row.tenant_id != run.tenant_id
        or row.strategy_id != run.strategy_id
        for row in [*points, *fills]
    ):
        raise ValueError("validation evidence must belong to its run and tenant")
    point_sequences = [row.sequence for row in points]
    fill_sequences = [row.sequence for row in fills]
    if len(point_sequences) != len(set(point_sequences)):
        raise ValueError("validation point sequences must be unique")
    if len(fill_sequences) != len(set(fill_sequences)):
        raise ValueError("validation fill sequences must be unique")


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


