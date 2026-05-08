"""PostgreSQL-backed task store."""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, delete, select

from valuecell.server.db.connection import get_database_manager

from ..storage.postgres_runtime import RuntimeTaskRecord
from .models import ScheduleConfig, Task, TaskPattern, TaskStatus
from .task_store import TaskStore


def _record_to_task(record: RuntimeTaskRecord) -> Task:
    schedule_config = None
    if record.schedule_config:
        try:
            schedule_config = ScheduleConfig.model_validate_json(record.schedule_config)
        except Exception:
            schedule_config = None

    return Task(
        task_id=record.task_id,
        title=record.title or "",
        query=record.query,
        conversation_id=record.conversation_id,
        thread_id=record.thread_id,
        user_id=record.user_id,
        agent_name=record.agent_name,
        status=TaskStatus(record.status),
        pattern=TaskPattern(record.pattern),
        schedule_config=schedule_config,
        handoff_from_super_agent=bool(record.handoff_from_super_agent),
        created_at=datetime.fromisoformat(record.created_at),
        started_at=datetime.fromisoformat(record.started_at)
        if record.started_at
        else None,
        completed_at=datetime.fromisoformat(record.completed_at)
        if record.completed_at
        else None,
        updated_at=datetime.fromisoformat(record.updated_at),
        error_message=record.error_message,
    )


class PostgresTaskStore(TaskStore):
    """Task store backed by the project's main SQL database."""

    async def save_task(self, task: Task) -> None:
        db = get_database_manager().get_session()
        try:
            existing = db.get(RuntimeTaskRecord, task.task_id)
            schedule_json = (
                task.schedule_config.model_dump_json()
                if task.schedule_config is not None
                else None
            )
            if existing is None:
                db.add(
                    RuntimeTaskRecord(
                        task_id=task.task_id,
                        title=task.title,
                        query=task.query,
                        conversation_id=task.conversation_id,
                        thread_id=task.thread_id,
                        user_id=task.user_id,
                        agent_name=task.agent_name,
                        status=task.status.value,
                        pattern=task.pattern.value,
                        schedule_config=schedule_json,
                        handoff_from_super_agent=task.handoff_from_super_agent,
                        created_at=task.created_at.isoformat(),
                        started_at=task.started_at.isoformat()
                        if task.started_at
                        else None,
                        completed_at=task.completed_at.isoformat()
                        if task.completed_at
                        else None,
                        updated_at=task.updated_at.isoformat(),
                        error_message=task.error_message,
                    )
                )
            else:
                existing.title = task.title
                existing.query = task.query
                existing.conversation_id = task.conversation_id
                existing.thread_id = task.thread_id
                existing.user_id = task.user_id
                existing.agent_name = task.agent_name
                existing.status = task.status.value
                existing.pattern = task.pattern.value
                existing.schedule_config = schedule_json
                existing.handoff_from_super_agent = task.handoff_from_super_agent
                existing.created_at = task.created_at.isoformat()
                existing.started_at = (
                    task.started_at.isoformat() if task.started_at else None
                )
                existing.completed_at = (
                    task.completed_at.isoformat() if task.completed_at else None
                )
                existing.updated_at = task.updated_at.isoformat()
                existing.error_message = task.error_message
            db.commit()
        finally:
            db.close()

    async def load_task(self, task_id: str) -> Optional[Task]:
        db = get_database_manager().get_session()
        try:
            record = db.get(RuntimeTaskRecord, task_id)
            return _record_to_task(record) if record else None
        finally:
            db.close()

    async def delete_task(self, task_id: str) -> bool:
        db = get_database_manager().get_session()
        try:
            record = db.get(RuntimeTaskRecord, task_id)
            if record is None:
                return False
            db.delete(record)
            db.commit()
            return True
        finally:
            db.close()

    async def list_tasks(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Task]:
        db = get_database_manager().get_session()
        try:
            stmt = select(RuntimeTaskRecord)
            conditions = []
            if conversation_id is not None:
                conditions.append(RuntimeTaskRecord.conversation_id == conversation_id)
            if user_id is not None:
                conditions.append(RuntimeTaskRecord.user_id == user_id)
            if status is not None:
                conditions.append(RuntimeTaskRecord.status == status.value)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.order_by(RuntimeTaskRecord.created_at.desc())
            stmt = stmt.offset(offset).limit(limit)
            rows = db.execute(stmt).scalars().all()
            return [_record_to_task(row) for row in rows]
        finally:
            db.close()

    async def task_exists(self, task_id: str) -> bool:
        db = get_database_manager().get_session()
        try:
            stmt = select(RuntimeTaskRecord.task_id).where(
                RuntimeTaskRecord.task_id == task_id
            )
            return db.execute(stmt).first() is not None
        finally:
            db.close()
