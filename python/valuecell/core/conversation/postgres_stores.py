"""PostgreSQL-backed conversation and item stores."""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, delete, func, select

from valuecell.core.types import ConversationItem, ConversationItemEvent, Role
from valuecell.server.db.connection import get_database_manager

from .conversation_store import ConversationStore
from .item_store import ItemStore
from .models import Conversation, ConversationStatus
from ..storage.postgres_runtime import (
    RuntimeConversationItemRecord,
    RuntimeConversationRecord,
)


def _conversation_to_record(
    conversation: Conversation,
) -> RuntimeConversationRecord:
    return RuntimeConversationRecord(
        conversation_id=conversation.conversation_id,
        user_id=conversation.user_id,
        title=conversation.title,
        agent_name=conversation.agent_name,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
        status=conversation.status.value,
    )


def _record_to_conversation(record: RuntimeConversationRecord) -> Conversation:
    return Conversation(
        conversation_id=record.conversation_id,
        user_id=record.user_id,
        title=record.title,
        agent_name=record.agent_name,
        created_at=datetime.fromisoformat(record.created_at),
        updated_at=datetime.fromisoformat(record.updated_at),
        status=ConversationStatus(record.status),
    )


def _record_to_item(record: RuntimeConversationItemRecord) -> ConversationItem:
    return ConversationItem(
        item_id=record.item_id,
        role=record.role,
        event=record.event,
        conversation_id=record.conversation_id,
        thread_id=record.thread_id,
        task_id=record.task_id,
        payload=record.payload,
        agent_name=record.agent_name,
        metadata=record.item_metadata or "{}",
    )


class PostgresConversationStore(ConversationStore):
    """Conversation store backed by the project's main SQL database."""

    async def save_conversation(self, conversation: Conversation) -> None:
        db = get_database_manager().get_session()
        try:
            existing = db.get(
                RuntimeConversationRecord,
                conversation.conversation_id,
            )
            if existing is None:
                db.add(_conversation_to_record(conversation))
            else:
                existing.user_id = conversation.user_id
                existing.title = conversation.title
                existing.agent_name = conversation.agent_name
                existing.created_at = conversation.created_at.isoformat()
                existing.updated_at = conversation.updated_at.isoformat()
                existing.status = conversation.status.value
            db.commit()
        finally:
            db.close()

    async def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        db = get_database_manager().get_session()
        try:
            record = db.get(RuntimeConversationRecord, conversation_id)
            return _record_to_conversation(record) if record else None
        finally:
            db.close()

    async def delete_conversation(self, conversation_id: str) -> bool:
        db = get_database_manager().get_session()
        try:
            record = db.get(RuntimeConversationRecord, conversation_id)
            if record is None:
                return False
            db.delete(record)
            db.commit()
            return True
        finally:
            db.close()

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Conversation]:
        db = get_database_manager().get_session()
        try:
            stmt = select(RuntimeConversationRecord)
            if user_id is not None:
                stmt = stmt.where(RuntimeConversationRecord.user_id == user_id)
            stmt = stmt.order_by(RuntimeConversationRecord.created_at.desc())
            stmt = stmt.offset(offset).limit(limit)
            rows = db.execute(stmt).scalars().all()
            return [_record_to_conversation(row) for row in rows]
        finally:
            db.close()

    async def conversation_exists(self, conversation_id: str) -> bool:
        db = get_database_manager().get_session()
        try:
            stmt = select(RuntimeConversationRecord.conversation_id).where(
                RuntimeConversationRecord.conversation_id == conversation_id
            )
            return db.execute(stmt).first() is not None
        finally:
            db.close()


class PostgresItemStore(ItemStore):
    """Conversation item store backed by the project's main SQL database."""

    async def save_item(self, item: ConversationItem) -> None:
        db = get_database_manager().get_session()
        try:
            existing = db.get(RuntimeConversationItemRecord, item.item_id)
            created_at = datetime.now().isoformat()
            if existing is None:
                db.add(
                    RuntimeConversationItemRecord(
                        item_id=item.item_id,
                        role=getattr(item.role, "value", str(item.role)),
                        event=getattr(item.event, "value", str(item.event)),
                        conversation_id=item.conversation_id,
                        thread_id=item.thread_id,
                        task_id=item.task_id,
                        payload=item.payload,
                        agent_name=item.agent_name,
                        item_metadata=item.metadata,
                        created_at=created_at,
                    )
                )
            else:
                existing.role = getattr(item.role, "value", str(item.role))
                existing.event = getattr(item.event, "value", str(item.event))
                existing.conversation_id = item.conversation_id
                existing.thread_id = item.thread_id
                existing.task_id = item.task_id
                existing.payload = item.payload
                existing.agent_name = item.agent_name
                existing.item_metadata = item.metadata
            db.commit()
        finally:
            db.close()

    async def get_items(
        self,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        role: Optional[Role] = None,
        **kwargs,
    ) -> List[ConversationItem]:
        event: Optional[ConversationItemEvent] = kwargs.get("event")
        component_type: Optional[str] = kwargs.get("component_type")
        task_id: Optional[str] = kwargs.get("task_id")

        db = get_database_manager().get_session()
        try:
            stmt = select(RuntimeConversationItemRecord)
            conditions = []
            if conversation_id is not None:
                conditions.append(
                    RuntimeConversationItemRecord.conversation_id == conversation_id
                )
            if role is not None:
                conditions.append(
                    RuntimeConversationItemRecord.role
                    == getattr(role, "value", str(role))
                )
            if event is not None:
                conditions.append(
                    RuntimeConversationItemRecord.event
                    == getattr(event, "value", str(event))
                )
            if task_id is not None:
                conditions.append(RuntimeConversationItemRecord.task_id == task_id)
            if component_type is not None:
                stmt = stmt.where(
                    RuntimeConversationItemRecord.payload.contains(
                        json.dumps(component_type).strip('"')
                    )
                )
            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(RuntimeConversationItemRecord.created_at.asc())
            if offset:
                stmt = stmt.offset(offset)
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = db.execute(stmt).scalars().all()
            return [_record_to_item(row) for row in rows]
        finally:
            db.close()

    async def get_latest_item(
        self,
        conversation_id: str,
    ) -> Optional[ConversationItem]:
        db = get_database_manager().get_session()
        try:
            stmt = (
                select(RuntimeConversationItemRecord)
                .where(RuntimeConversationItemRecord.conversation_id == conversation_id)
                .order_by(RuntimeConversationItemRecord.created_at.desc())
                .limit(1)
            )
            row = db.execute(stmt).scalars().first()
            return _record_to_item(row) if row else None
        finally:
            db.close()

    async def get_item(self, item_id: str) -> Optional[ConversationItem]:
        db = get_database_manager().get_session()
        try:
            row = db.get(RuntimeConversationItemRecord, item_id)
            return _record_to_item(row) if row else None
        finally:
            db.close()

    async def get_item_count(self, conversation_id: str) -> int:
        db = get_database_manager().get_session()
        try:
            stmt = select(func.count(RuntimeConversationItemRecord.item_id)).where(
                RuntimeConversationItemRecord.conversation_id == conversation_id
            )
            result = db.execute(stmt).scalar_one()
            return int(result)
        finally:
            db.close()

    async def delete_conversation_items(self, conversation_id: str) -> None:
        db = get_database_manager().get_session()
        try:
            stmt = delete(RuntimeConversationItemRecord).where(
                RuntimeConversationItemRecord.conversation_id == conversation_id
            )
            db.execute(stmt)
            db.commit()
        finally:
            db.close()
