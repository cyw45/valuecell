"""Minimal asyncio scheduler compatible with the project's APScheduler calls."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class _Job:
    id: str
    func: Callable[..., Any]
    trigger: Any
    args: list[Any] = field(default_factory=list)
    coalesce: bool = True
    max_instances: int = 1
    task: asyncio.Task[None] | None = None
    active_instances: int = 0


class AsyncIOScheduler:
    """Small subset of APScheduler's AsyncIOScheduler used by ValueCell."""

    def __init__(self) -> None:
        self.running = False
        self._jobs: dict[str, _Job] = {}

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        for job in self._jobs.values():
            self._ensure_task(job, immediate=False)

    def shutdown(self, wait: bool = True) -> None:  # noqa: ARG002 - APScheduler API
        self.running = False
        for job in self._jobs.values():
            if job.task is not None:
                job.task.cancel()
                job.task = None

    def add_job(
        self,
        func: Callable[..., Any],
        *,
        trigger: Any,
        id: str,
        args: list[Any] | tuple[Any, ...] | None = None,
        replace_existing: bool = False,
        coalesce: bool = True,
        max_instances: int = 1,
        next_run_time: datetime | None = None,
        **_: object,
    ) -> _Job:
        if id in self._jobs:
            if not replace_existing:
                raise ValueError(f"Job {id!r} already exists")
            self.remove_job(id)
        job = _Job(
            id=id,
            func=func,
            trigger=trigger,
            args=list(args or []),
            coalesce=coalesce,
            max_instances=max_instances,
        )
        self._jobs[id] = job
        if self.running:
            immediate = next_run_time is not None and next_run_time <= datetime.now(timezone.utc)
            self._ensure_task(job, immediate=immediate)
        return job

    def get_jobs(self) -> list[_Job]:
        return list(self._jobs.values())

    def get_job(self, job_id: str) -> _Job | None:
        return self._jobs.get(job_id)

    def remove_job(self, job_id: str) -> None:
        job = self._jobs.pop(job_id)
        if job.task is not None:
            job.task.cancel()

    def pause(self) -> None:
        self.running = False
        for job in self._jobs.values():
            if job.task is not None:
                job.task.cancel()
                job.task = None

    def _ensure_task(self, job: _Job, *, immediate: bool) -> None:
        if job.task is None or job.task.done():
            job.task = asyncio.create_task(self._run_job_loop(job, immediate=immediate))

    async def _run_job_loop(self, job: _Job, *, immediate: bool) -> None:
        first = True
        while self.running and job.id in self._jobs:
            if first and immediate:
                first = False
            else:
                first = False
                await asyncio.sleep(float(job.trigger.seconds))
            await self._invoke(job)

    async def _invoke(self, job: _Job) -> None:
        if job.active_instances >= job.max_instances:
            return
        job.active_instances += 1
        try:
            result = job.func(*job.args)
            if inspect.isawaitable(result):
                await result
        finally:
            job.active_instances -= 1
