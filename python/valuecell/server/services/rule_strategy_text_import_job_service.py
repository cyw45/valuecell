"""Durable background jobs for natural-language strategy compilation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import or_, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valuecell.server.api.schemas.rule_strategy import RuleStrategyTextImportProposal
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.rule_strategy_text_import_job import (
    RuleStrategyTextImportJobRecord,
)
from valuecell.server.services.rule_strategy_text_import_service import (
    RuleStrategyTextImportService,
    RuleStrategyTextImportUnavailableError,
)

_ACTIVE = ("pending", "running")
_TERMINAL = ("completed", "failed")
_LEASE_DURATION = timedelta(minutes=5)
_RECOVERY_INTERVAL_S = 15
_HEARTBEAT_INTERVAL_S = 15
_CAPACITY_LOCK_ID = 0x56434C4C


class RuleStrategyTextImportJobNotFoundError(Exception):
    """Raised when a job is absent or not owned by the current principal."""


class RuleStrategyTextImportJobCapacityError(Exception):
    """Raised when global or per-user active-job capacity is exhausted."""


class RuleStrategyTextImportJobConflictError(Exception):
    """Raised when an idempotency key is reused for different input."""


class RuleStrategyTextImportJob(BaseModel):
    """Public state of one background strategy compilation."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    proposal: RuleStrategyTextImportProposal | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class StrategyTextImporter(Protocol):
    async def parse(self, strategy_text: str) -> RuleStrategyTextImportProposal: ...


class RuleStrategyTextImportJobService:
    """Persist, lease, and execute AI jobs outside HTTP request lifetimes."""

    def __init__(
        self,
        importer: StrategyTextImporter | None = None,
        session_factory: Callable[[], Session] | None = None,
        global_active_limit: int = 4,
        owner_active_limit: int = 1,
        retention: timedelta = timedelta(hours=24),
        execution_timeout: timedelta = timedelta(hours=1),
    ) -> None:
        self._importer = importer or RuleStrategyTextImportService()
        self._session_factory = session_factory or get_database_manager().get_session
        self._global_active_limit = global_active_limit
        self._owner_active_limit = owner_active_limit
        self._retention = retention
        self._execution_timeout = execution_timeout
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._recovery_task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        if self._recovery_task is None or self._recovery_task.done():
            self._recovery_task = asyncio.create_task(self._recover_forever())

    async def stop(self) -> None:
        tasks = [*self._tasks.values()]
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            tasks.append(self._recovery_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._recovery_task = None

    def submit(
        self,
        strategy_text: str,
        tenant_id: str,
        user_id: str,
        request_id: str,
    ) -> RuleStrategyTextImportJob:
        job = self._submit_record(strategy_text, tenant_id, user_id, request_id)
        self._schedule(job.job_id)
        return job

    async def submit_async(
        self,
        strategy_text: str,
        tenant_id: str,
        user_id: str,
        request_id: str,
    ) -> RuleStrategyTextImportJob:
        job = await asyncio.to_thread(
            self._submit_record, strategy_text, tenant_id, user_id, request_id
        )
        self._schedule(job.job_id)
        return job

    def _submit_record(
        self,
        strategy_text: str,
        tenant_id: str,
        user_id: str,
        request_id: str,
    ) -> RuleStrategyTextImportJob:
        db = self._session_factory()
        try:
            self._lock_capacity(db)
            self._prune(db)
            existing = self._find_request(db, tenant_id, user_id, request_id)
            if existing is not None:
                if existing.strategy_text != strategy_text:
                    raise RuleStrategyTextImportJobConflictError(
                        "request_id has already been used with different text"
                    )
                job = self._public(existing)
            else:
                active_filter = RuleStrategyTextImportJobRecord.status.in_(_ACTIVE)
                global_count = (
                    db.query(RuleStrategyTextImportJobRecord)
                    .filter(active_filter)
                    .count()
                )
                owner_count = (
                    db.query(RuleStrategyTextImportJobRecord)
                    .filter(
                        RuleStrategyTextImportJobRecord.tenant_id == tenant_id,
                        RuleStrategyTextImportJobRecord.user_id == user_id,
                        active_filter,
                    )
                    .count()
                )
                if global_count >= self._global_active_limit:
                    raise RuleStrategyTextImportJobCapacityError(
                        "AI 策略分析任务繁忙，请稍后重试"
                    )
                if owner_count >= self._owner_active_limit:
                    raise RuleStrategyTextImportJobCapacityError(
                        "当前账号已有策略分析任务正在执行"
                    )
                record = RuleStrategyTextImportJobRecord(
                    job_id=str(uuid4()),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    request_id=request_id,
                    strategy_text=strategy_text,
                    status="pending",
                )
                db.add(record)
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    existing = self._find_request(db, tenant_id, user_id, request_id)
                    if existing is None:
                        raise
                    if existing.strategy_text != strategy_text:
                        raise RuleStrategyTextImportJobConflictError(
                            "request_id has already been used with different text"
                        )
                    record = existing
                db.refresh(record)
                job = self._public(record)
        finally:
            db.close()
        return job

    def get(
        self, job_id: str, tenant_id: str, user_id: str
    ) -> RuleStrategyTextImportJob:
        job = self._get_record(job_id, tenant_id, user_id)
        if job.status in _ACTIVE:
            self._schedule(job.job_id)
        return job

    async def get_async(
        self, job_id: str, tenant_id: str, user_id: str
    ) -> RuleStrategyTextImportJob:
        job = await asyncio.to_thread(self._get_record, job_id, tenant_id, user_id)
        if job.status in _ACTIVE:
            self._schedule(job.job_id)
        return job

    def _get_record(
        self, job_id: str, tenant_id: str, user_id: str
    ) -> RuleStrategyTextImportJob:
        db = self._session_factory()
        try:
            record = (
                db.query(RuleStrategyTextImportJobRecord)
                .filter_by(job_id=job_id, tenant_id=tenant_id, user_id=user_id)
                .first()
            )
            if record is None:
                raise RuleStrategyTextImportJobNotFoundError(
                    "Strategy text import job not found"
                )
            job = self._public(record)
        finally:
            db.close()
        return job

    def _schedule(self, job_id: str) -> None:
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            return
        task = asyncio.create_task(self._run(job_id))
        self._tasks[job_id] = task
        task.add_done_callback(lambda done: self._forget(job_id, done))

    async def _recover_forever(self) -> None:
        try:
            while True:
                job_ids = await asyncio.to_thread(self._recoverable_job_ids)
                for job_id in job_ids:
                    self._schedule(job_id)
                await asyncio.sleep(_RECOVERY_INTERVAL_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Strategy text import recovery loop failed; restarting")
            await asyncio.sleep(_RECOVERY_INTERVAL_S)
            self._recovery_task = None
            self.start()

    def _recoverable_job_ids(self) -> list[str]:
        db = self._session_factory()
        try:
            now = datetime.now(UTC)
            return [
                row[0]
                for row in db.query(RuleStrategyTextImportJobRecord.job_id)
                .filter(
                    or_(
                        RuleStrategyTextImportJobRecord.status == "pending",
                        (
                            (RuleStrategyTextImportJobRecord.status == "running")
                            & (RuleStrategyTextImportJobRecord.lease_expires_at <= now)
                        ),
                    )
                )
                .all()
            ]
        finally:
            db.close()

    async def _run(self, job_id: str) -> None:
        worker_id = str(uuid4())
        claim_task = asyncio.create_task(
            asyncio.to_thread(self._claim, job_id, worker_id)
        )
        try:
            strategy_text = await asyncio.shield(claim_task)
        except asyncio.CancelledError:
            strategy_text = await claim_task
            if strategy_text is not None:
                await asyncio.to_thread(self._release, job_id, worker_id)
            raise
        if strategy_text is None:
            return
        lease_lost = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(job_id, worker_id, lease_lost))
        parse_task = asyncio.create_task(self._importer.parse(strategy_text))
        lease_task = asyncio.create_task(lease_lost.wait())
        finalizing = False
        try:
            done, _ = await asyncio.wait(
                {parse_task, lease_task},
                timeout=self._execution_timeout.total_seconds(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if parse_task in done:
                proposal = parse_task.result()
                finalizing = True
                await self._finish_durable(
                    job_id,
                    worker_id,
                    "completed",
                    proposal.model_dump(mode="json"),
                    None,
                )
            elif lease_task in done:
                parse_task.cancel()
                await asyncio.gather(parse_task, return_exceptions=True)
                await asyncio.to_thread(self._release, job_id, worker_id)
            else:
                parse_task.cancel()
                await asyncio.gather(parse_task, return_exceptions=True)
                finalizing = True
                await self._finish_durable(
                    job_id,
                    worker_id,
                    "failed",
                    None,
                    "AI 策略分析超过 60 分钟仍未完成，请重试",
                )
        except RuleStrategyTextImportUnavailableError as exc:
            finalizing = True
            await self._finish_durable(job_id, worker_id, "failed", None, str(exc))
        except asyncio.CancelledError:
            parse_task.cancel()
            await asyncio.gather(parse_task, return_exceptions=True)
            if not finalizing:
                await asyncio.to_thread(self._release, job_id, worker_id)
            raise
        except Exception:
            parse_task.cancel()
            await asyncio.gather(parse_task, return_exceptions=True)
            logger.exception("Unexpected strategy text import job failure: {}", job_id)
            finalizing = True
            await self._finish_durable(
                job_id,
                worker_id,
                "failed",
                None,
                "AI 策略分析服务发生内部错误，请重试",
            )
        finally:
            heartbeat.cancel()
            lease_task.cancel()
            await asyncio.gather(heartbeat, lease_task, return_exceptions=True)

    async def _finish_durable(
        self,
        job_id: str,
        worker_id: str,
        status: str,
        proposal: dict[str, Any] | None,
        error: str | None,
    ) -> int:
        task = asyncio.create_task(
            asyncio.to_thread(self._finish, job_id, worker_id, status, proposal, error)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def _claim(self, job_id: str, worker_id: str) -> str | None:
        db = self._session_factory()
        try:
            now = datetime.now(UTC)
            statement = (
                update(RuleStrategyTextImportJobRecord)
                .where(
                    RuleStrategyTextImportJobRecord.job_id == job_id,
                    or_(
                        RuleStrategyTextImportJobRecord.status == "pending",
                        (
                            (RuleStrategyTextImportJobRecord.status == "running")
                            & (RuleStrategyTextImportJobRecord.lease_expires_at <= now)
                        ),
                    ),
                )
                .values(
                    status="running",
                    worker_id=worker_id,
                    lease_expires_at=now + _LEASE_DURATION,
                    updated_at=now,
                )
                .returning(RuleStrategyTextImportJobRecord.strategy_text)
            )
            strategy_text = db.execute(statement).scalar_one_or_none()
            db.commit()
            return strategy_text
        finally:
            db.close()

    async def _heartbeat(
        self, job_id: str, worker_id: str, lease_lost: asyncio.Event
    ) -> None:
        failures = 0
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            try:
                updated = await asyncio.to_thread(self._renew_lease, job_id, worker_id)
                if updated != 1:
                    lease_lost.set()
                    return
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                logger.warning(
                    "Strategy text import lease heartbeat failed ({}/2): {}",
                    failures,
                    job_id,
                )
                if failures >= 2:
                    lease_lost.set()
                    return

    def _renew_lease(self, job_id: str, worker_id: str) -> int:
        db = self._session_factory()
        try:
            now = datetime.now(UTC)
            updated = (
                db.query(RuleStrategyTextImportJobRecord)
                .filter_by(job_id=job_id, worker_id=worker_id, status="running")
                .update(
                    {
                        "lease_expires_at": now + _LEASE_DURATION,
                        "updated_at": now,
                    }
                )
            )
            db.commit()
            return updated
        finally:
            db.close()

    def _finish(
        self,
        job_id: str,
        worker_id: str,
        status: str,
        proposal: dict[str, Any] | None,
        error: str | None,
    ) -> int:
        db = self._session_factory()
        try:
            updated = (
                db.query(RuleStrategyTextImportJobRecord)
                .filter_by(job_id=job_id, worker_id=worker_id, status="running")
                .update(
                    {
                        "status": status,
                        "proposal": proposal,
                        "error": error,
                        "worker_id": None,
                        "lease_expires_at": None,
                        "updated_at": datetime.now(UTC),
                    }
                )
            )
            db.commit()
            return updated
        finally:
            db.close()

    def _release(self, job_id: str, worker_id: str) -> None:
        db = self._session_factory()
        try:
            db.query(RuleStrategyTextImportJobRecord).filter_by(
                job_id=job_id, worker_id=worker_id, status="running"
            ).update(
                {
                    "status": "pending",
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": datetime.now(UTC),
                }
            )
            db.commit()
        finally:
            db.close()

    def _lock_capacity(self, db: Session) -> None:
        if db.bind is not None and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT pg_advisory_xact_lock(:lock_id)"),
                {"lock_id": _CAPACITY_LOCK_ID},
            )

    def _prune(self, db: Session) -> None:
        cutoff = datetime.now(UTC) - self._retention
        db.query(RuleStrategyTextImportJobRecord).filter(
            RuleStrategyTextImportJobRecord.status.in_(_TERMINAL),
            RuleStrategyTextImportJobRecord.updated_at < cutoff,
        ).delete(synchronize_session=False)
        db.flush()

    @staticmethod
    def _find_request(
        db: Session, tenant_id: str, user_id: str, request_id: str
    ) -> RuleStrategyTextImportJobRecord | None:
        return (
            db.query(RuleStrategyTextImportJobRecord)
            .filter_by(tenant_id=tenant_id, user_id=user_id, request_id=request_id)
            .first()
        )

    def _forget(self, job_id: str, task: asyncio.Task[Any]) -> None:
        if self._tasks.get(job_id) is task:
            self._tasks.pop(job_id, None)

    @staticmethod
    def _public(record: RuleStrategyTextImportJobRecord) -> RuleStrategyTextImportJob:
        proposal = (
            RuleStrategyTextImportProposal.model_validate(record.proposal)
            if record.proposal is not None
            else None
        )
        return RuleStrategyTextImportJob(
            job_id=record.job_id,
            status=record.status,
            proposal=proposal,
            error=record.error,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


_job_service: RuleStrategyTextImportJobService | None = None


def get_rule_strategy_text_import_job_service() -> RuleStrategyTextImportJobService:
    global _job_service
    if _job_service is None:
        _job_service = RuleStrategyTextImportJobService()
    return _job_service
