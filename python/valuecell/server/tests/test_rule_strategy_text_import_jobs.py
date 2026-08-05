import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from valuecell.server.api.schemas.rule_strategy import RuleStrategyTextImportProposal
from valuecell.server.db.models.rule_strategy_text_import_job import (
    RuleStrategyTextImportJobRecord,
)
from valuecell.server.services.rule_strategy_text_import_job_service import (
    RuleStrategyTextImportJobCapacityError,
    RuleStrategyTextImportJobNotFoundError,
    RuleStrategyTextImportJobService,
)


def _proposal() -> RuleStrategyTextImportProposal:
    return RuleStrategyTextImportProposal.model_validate(
        {
            "strategy_name": "后台解析策略",
            "executable": False,
            "config": None,
            "summary": "解析完成",
            "unresolved_items": ["测试结果"],
            "corrections": [],
            "rejection_reasons": ["测试拒绝"],
        }
    )


async def _wait_for_job(jobs: RuleStrategyTextImportJobService, job_id: str) -> None:
    task = jobs._tasks.get(job_id)
    assert task is not None
    await asyncio.wait_for(asyncio.shield(task), timeout=1)


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    RuleStrategyTextImportJobRecord.__table__.create(engine)
    return sessionmaker(bind=engine)


@pytest.mark.asyncio
async def test_text_import_job_returns_immediately_and_persists_result(session_factory):
    release = asyncio.Event()
    started = asyncio.Event()

    class DelayedImporter:
        async def parse(self, strategy_text: str):
            assert strategy_text == "等待模型完成的复杂策略描述"
            started.set()
            await release.wait()
            return _proposal()

    jobs = RuleStrategyTextImportJobService(
        importer=DelayedImporter(), session_factory=session_factory
    )
    submitted = jobs.submit(
        "等待模型完成的复杂策略描述",
        tenant_id="tenant-a",
        user_id="user-a",
        request_id="00000000-0000-0000-0000-000000000001",
    )
    assert submitted.status == "pending"

    await asyncio.wait_for(started.wait(), timeout=1)
    assert jobs.get(submitted.job_id, "tenant-a", "user-a").status == "running"

    release.set()
    await _wait_for_job(jobs, submitted.job_id)
    completed = jobs.get(submitted.job_id, "tenant-a", "user-a")
    assert completed.status == "completed"
    assert completed.proposal == _proposal()
    assert completed.error is None


@pytest.mark.asyncio
async def test_text_import_job_resumes_pending_record_after_service_restart(
    session_factory,
):
    class Importer:
        async def parse(self, strategy_text: str):
            assert strategy_text == "服务重启后恢复的策略描述"
            return _proposal()

    db = session_factory()
    record = RuleStrategyTextImportJobRecord(
        job_id="00000000-0000-0000-0000-000000000002",
        tenant_id="tenant-a",
        user_id="user-a",
        request_id="00000000-0000-0000-0000-000000000003",
        strategy_text="服务重启后恢复的策略描述",
        status="pending",
    )
    db.add(record)
    db.commit()
    job_id = record.job_id
    db.close()

    restarted = RuleStrategyTextImportJobService(
        importer=Importer(), session_factory=session_factory
    )
    assert restarted.get(job_id, "tenant-a", "user-a").status == "pending"
    await _wait_for_job(restarted, job_id)
    assert restarted.get(job_id, "tenant-a", "user-a").status == "completed"


@pytest.mark.asyncio
async def test_text_import_submission_is_idempotent_and_owner_scoped(session_factory):
    release = asyncio.Event()

    class Importer:
        async def parse(self, _strategy_text: str):
            await release.wait()
            return _proposal()

    jobs = RuleStrategyTextImportJobService(
        importer=Importer(), session_factory=session_factory
    )
    request_id = "00000000-0000-0000-0000-000000000004"
    first = jobs.submit("一个足够长的策略描述", "tenant-a", "user-a", request_id)
    repeated = jobs.submit("一个足够长的策略描述", "tenant-a", "user-a", request_id)
    assert repeated.job_id == first.job_id

    with pytest.raises(RuleStrategyTextImportJobNotFoundError):
        jobs.get(first.job_id, "tenant-b", "user-a")
    with pytest.raises(RuleStrategyTextImportJobNotFoundError):
        jobs.get(first.job_id, "tenant-a", "user-b")

    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_text_import_job_enforces_owner_active_limit(session_factory):
    release = asyncio.Event()

    class Importer:
        async def parse(self, _strategy_text: str):
            await release.wait()
            return _proposal()

    jobs = RuleStrategyTextImportJobService(
        importer=Importer(), session_factory=session_factory, owner_active_limit=1
    )
    jobs.submit(
        "第一个足够长的策略描述",
        "tenant-a",
        "user-a",
        "00000000-0000-0000-0000-000000000005",
    )

    with pytest.raises(RuleStrategyTextImportJobCapacityError):
        jobs.submit(
            "第二个足够长的策略描述",
            "tenant-a",
            "user-a",
            "00000000-0000-0000-0000-000000000006",
        )

    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_text_import_job_cancels_provider_when_lease_is_lost(session_factory):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class Importer:
        async def parse(self, strategy_text: str) -> RuleStrategyTextImportProposal:
            assert strategy_text
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return _proposal()

    jobs = RuleStrategyTextImportJobService(
        importer=Importer(), session_factory=session_factory
    )

    async def lose_lease(
        job_id: str, worker_id: str, lease_lost: asyncio.Event
    ) -> None:
        assert job_id and worker_id
        await started.wait()
        lease_lost.set()

    jobs._heartbeat = lose_lease
    submitted = jobs.submit(
        "租约丢失时应取消上游模型调用",
        "tenant-a",
        "user-a",
        "00000000-0000-0000-0000-000000000007",
    )
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    await _wait_for_job(jobs, submitted.job_id)

    assert jobs.get(submitted.job_id, "tenant-a", "user-a").status == "pending"


@pytest.mark.asyncio
async def test_text_import_job_claim_does_not_block_event_loop(session_factory):
    thread_release = asyncio.Event()
    loop_progressed = asyncio.Event()

    class Importer:
        async def parse(self, strategy_text: str) -> RuleStrategyTextImportProposal:
            assert strategy_text
            return _proposal()

    jobs = RuleStrategyTextImportJobService(
        importer=Importer(), session_factory=session_factory
    )

    def slow_claim(job_id: str, worker_id: str) -> None:
        assert job_id and worker_id
        import time

        time.sleep(0.1)
        return None

    jobs._claim = slow_claim
    task = asyncio.create_task(jobs._run("job-id"))
    await asyncio.sleep(0)
    loop_progressed.set()

    assert loop_progressed.is_set()
    assert not task.done()
    await asyncio.wait_for(task, timeout=1)
    thread_release.set()


@pytest.mark.asyncio
async def test_async_submit_and_get_schedule_on_event_loop(session_factory):
    release = asyncio.Event()
    started = asyncio.Event()

    class Importer:
        async def parse(self, strategy_text: str) -> RuleStrategyTextImportProposal:
            assert strategy_text
            started.set()
            await release.wait()
            return _proposal()

    jobs = RuleStrategyTextImportJobService(
        importer=Importer(), session_factory=session_factory
    )
    submitted = await jobs.submit_async(
        "异步接口提交的复杂策略描述",
        "tenant-a",
        "user-a",
        "00000000-0000-0000-0000-000000000008",
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    fetched = await jobs.get_async(submitted.job_id, "tenant-a", "user-a")
    assert fetched.status == "running"
    release.set()
    await _wait_for_job(jobs, submitted.job_id)


@pytest.mark.asyncio
async def test_cancel_during_claim_releases_acquired_lease(session_factory):
    import threading

    claim_started = threading.Event()
    allow_claim = threading.Event()
    released = threading.Event()

    class Importer:
        async def parse(self, strategy_text: str) -> RuleStrategyTextImportProposal:
            assert strategy_text
            return _proposal()

    jobs = RuleStrategyTextImportJobService(
        importer=Importer(), session_factory=session_factory
    )

    def slow_claim(job_id: str, worker_id: str) -> str:
        assert job_id and worker_id
        claim_started.set()
        allow_claim.wait(timeout=1)
        return "已认领的复杂策略描述"

    def record_release(job_id: str, worker_id: str) -> None:
        assert job_id and worker_id
        released.set()

    jobs._claim = slow_claim
    jobs._release = record_release
    task = asyncio.create_task(jobs._run("job-id"))
    await asyncio.to_thread(claim_started.wait, 1)
    task.cancel()
    allow_claim.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()


@pytest.mark.asyncio
async def test_cancel_during_finish_does_not_release_completed_job(session_factory):
    import threading

    finish_started = threading.Event()
    allow_finish = threading.Event()
    release_called = threading.Event()

    class Importer:
        async def parse(self, strategy_text: str) -> RuleStrategyTextImportProposal:
            assert strategy_text
            return _proposal()

    jobs = RuleStrategyTextImportJobService(
        importer=Importer(), session_factory=session_factory
    )
    jobs._claim = lambda job_id, worker_id: "即将完成的复杂策略描述"

    def slow_finish(
        job_id: str,
        worker_id: str,
        status: str,
        proposal: dict | None,
        error: str | None,
    ) -> int:
        assert job_id and worker_id and status == "completed"
        assert proposal is not None and error is None
        finish_started.set()
        allow_finish.wait(timeout=1)
        return 1

    def record_release(job_id: str, worker_id: str) -> None:
        assert job_id and worker_id
        release_called.set()

    jobs._finish = slow_finish
    jobs._release = record_release
    task = asyncio.create_task(jobs._run("job-id"))
    await asyncio.to_thread(finish_started.wait, 1)
    task.cancel()
    allow_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not release_called.is_set()
