"""Independent lifecycle scheduler for the isolated V19 leader spot module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from sqlalchemy.orm import Session

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_scheduler import (
    LeaderSpotV19BatchSummary,
    LeaderSpotV19SchedulerTickResult,
)
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19Account,
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19ExecutionLease,
    LeaderSpotV19RiskState,
    LeaderSpotV19Strategy,
)


_LEASE_TTL_S = 90
_SYNC_JOB_ID = "_leader_spot_v19_sync_running"


class LeaderSpotV19Scheduler:
    """Own V19 jobs, batches, and generation fences without touching StrategyScheduler."""

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()
        self._worker_id = f"leader-spot-v19-scheduler-{uuid4().hex}"

    async def start(self) -> None:
        self._scheduler.start()
        logger.info("LeaderSpotV19Scheduler started")

    async def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("LeaderSpotV19Scheduler stopped")

    def install_sync_job(self) -> None:
        """Install the V19-only reconciliation job once per scheduler instance."""

        self._scheduler.add_job(
            self.sync_running_strategies,
            trigger=IntervalTrigger(seconds=60),
            id=_SYNC_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    def start_strategy(
        self, session: Session, *, strategy_id: str, tenant_id: str
    ) -> LeaderSpotV19BatchSummary:
        """Atomically create an isolated V19 batch and its account/risk state."""

        strategy = session.query(LeaderSpotV19Strategy).filter_by(
            strategy_id=strategy_id, tenant_id=tenant_id
        ).with_for_update().first()
        if strategy is None:
            raise ValueError("leader_spot_v19_strategy_not_found")
        if strategy.status == "running":
            raise ValueError("leader_spot_v19_strategy_already_running")
        config_payload = dict(strategy.config)
        config_payload.pop("daily_loss_limit_quote", None)
        config = LeaderSpotV19Config.model_validate(config_payload)
        strategy.execution_generation += 1
        batch = LeaderSpotV19ExecutionBatch(
            batch_id=str(uuid4()), tenant_id=tenant_id, strategy_id=strategy_id,
            strategy_name_snapshot=strategy.name, execution_generation=strategy.execution_generation,
            status="running", config_snapshot=config.model_dump(mode="json"),
        )
        strategy.status = "running"
        strategy.current_batch_id = batch.batch_id
        account = LeaderSpotV19Account(
            account_id=str(uuid4()), tenant_id=tenant_id, strategy_id=strategy_id,
            batch_id=batch.batch_id, scope=config.environment,
            credential_id=strategy.credential_id, initial_equity_quote=config.position.order_amount_quote * config.position.max_positions,
            quote_balance=config.position.order_amount_quote * config.position.max_positions,
            equity_quote=config.position.order_amount_quote * config.position.max_positions,
        )
        now = datetime.now(UTC)
        risk = LeaderSpotV19RiskState(
            risk_state_id=str(uuid4()), account_id=account.account_id, tenant_id=tenant_id,
            strategy_id=strategy_id, batch_id=batch.batch_id,
            daily_loss_limit_quote=config.daily_loss_limit_quote,
            daily_loss_reset_at=self._next_utc_midnight(now),
            prior_close_equity_quote=account.initial_equity_quote,
        )
        session.add_all([batch, account, risk])
        session.commit()
        return LeaderSpotV19BatchSummary(
            batch_id=batch.batch_id, strategy_id=strategy_id,
            execution_generation=batch.execution_generation, status="running",
            started_at=batch.started_at.replace(tzinfo=UTC) if batch.started_at.tzinfo is None else batch.started_at,
        )

    def stop_strategy(
        self, session: Session, *, strategy_id: str, tenant_id: str
    ) -> LeaderSpotV19BatchSummary:
        """Fence future ticks, close only the current V19 batch, and retain history."""

        strategy = session.query(LeaderSpotV19Strategy).filter_by(
            strategy_id=strategy_id, tenant_id=tenant_id
        ).with_for_update().first()
        if strategy is None or strategy.current_batch_id is None:
            raise ValueError("leader_spot_v19_running_batch_not_found")
        batch = session.query(LeaderSpotV19ExecutionBatch).filter_by(
            batch_id=strategy.current_batch_id, tenant_id=tenant_id, strategy_id=strategy_id
        ).with_for_update().first()
        if batch is None or batch.status != "running":
            raise ValueError("leader_spot_v19_running_batch_not_found")
        now = datetime.now(UTC)
        batch.status = "stopped"
        batch.stopped_at = now
        strategy.status = "stopped"
        strategy.current_batch_id = None
        strategy.execution_generation += 1
        session.commit()
        return LeaderSpotV19BatchSummary(
            batch_id=batch.batch_id, strategy_id=strategy_id,
            execution_generation=batch.execution_generation, status="stopped",
            started_at=batch.started_at.replace(tzinfo=UTC) if batch.started_at.tzinfo is None else batch.started_at,
            stopped_at=now,
        )

    def sync_running_strategies(self) -> None:
        """Reconcile only V19 jobs; legacy jobs are stored in a different scheduler."""

        session = get_database_manager().get_session()
        try:
            strategies = session.query(LeaderSpotV19Strategy).filter_by(status="running").all()
            wanted = {self._job_id(item.strategy_id) for item in strategies}
            for job in list(self._scheduler.get_jobs()):
                if job.id.startswith("leader_spot_v19:") and job.id not in wanted:
                    self._scheduler.remove_job(job.id)
            for strategy in strategies:
                config_payload = dict(strategy.config)
                config_payload.pop("daily_loss_limit_quote", None)
                config = LeaderSpotV19Config.model_validate(config_payload)
                job_id = self._job_id(strategy.strategy_id)
                args = [strategy.strategy_id, strategy.tenant_id, strategy.execution_generation]
                existing = self._scheduler.get_job(job_id)
                if existing is not None and list(existing.args) == args:
                    continue
                self._scheduler.add_job(
                    self._tick, trigger=IntervalTrigger(seconds=config.data.check_interval_seconds),
                    id=job_id, args=args, replace_existing=True, coalesce=True,
                    max_instances=1, next_run_time=datetime.now(UTC),
                )
        finally:
            session.close()

    async def _tick(self, strategy_id: str, tenant_id: str, execution_generation: int) -> LeaderSpotV19SchedulerTickResult:
        """Claim a V19 lease then fail closed until all later phase inputs exist."""

        session = get_database_manager().get_session()
        try:
            strategy = session.query(LeaderSpotV19Strategy).filter_by(
                strategy_id=strategy_id, tenant_id=tenant_id, status="running"
            ).first()
            if strategy is None or strategy.execution_generation != execution_generation:
                return LeaderSpotV19SchedulerTickResult(strategy_id=strategy_id, batch_id=None, status="skipped", reason_code="stale_generation")
            if not self._claim_lease(session, strategy_id, execution_generation):
                return LeaderSpotV19SchedulerTickResult(strategy_id=strategy_id, batch_id=strategy.current_batch_id, status="skipped", reason_code="lease_unavailable")
            return LeaderSpotV19SchedulerTickResult(
                strategy_id=strategy_id, batch_id=strategy.current_batch_id,
                status="blocked", reason_code="v19_execution_pipeline_not_wired",
            )
        finally:
            session.close()

    def _claim_lease(self, session: Session, strategy_id: str, generation: int) -> bool:
        now = datetime.now(UTC)
        lease = session.query(LeaderSpotV19ExecutionLease).filter_by(
            strategy_id=strategy_id, execution_generation=generation
        ).with_for_update().first()
        if lease is not None:
            expires_at = lease.expires_at.replace(tzinfo=UTC) if lease.expires_at.tzinfo is None else lease.expires_at.astimezone(UTC)
            if expires_at > now and lease.owner_id != self._worker_id:
                session.rollback()
                return False
            lease.owner_id = self._worker_id
            lease.expires_at = now + timedelta(seconds=_LEASE_TTL_S)
        else:
            session.add(LeaderSpotV19ExecutionLease(
                strategy_id=strategy_id, execution_generation=generation,
                owner_id=self._worker_id, expires_at=now + timedelta(seconds=_LEASE_TTL_S),
            ))
        session.commit()
        return True

    @staticmethod
    def _job_id(strategy_id: str) -> str:
        return f"leader_spot_v19:{strategy_id}"

    @staticmethod
    def _next_utc_midnight(value: datetime) -> datetime:
        current = value.astimezone(UTC)
        return current.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
