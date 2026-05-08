"""Shared SQLAlchemy runtime models and helpers for core persistence."""

from __future__ import annotations

from sqlalchemy import Boolean, Column, Index, Text

from valuecell.server.db.models.base import Base


class RuntimeConversationRecord(Base):
    """Conversation metadata persisted in the main SQL database."""

    __tablename__ = "runtime_conversations"

    conversation_id = Column(Text, primary_key=True)
    user_id = Column(Text, nullable=False, index=True)
    title = Column(Text, nullable=True)
    agent_name = Column(Text, nullable=True)
    created_at = Column(Text, nullable=False)
    updated_at = Column(Text, nullable=False, index=True)
    status = Column(Text, nullable=False, default="active")


class RuntimeConversationItemRecord(Base):
    """Conversation items persisted in the main SQL database."""

    __tablename__ = "runtime_conversation_items"

    item_id = Column(Text, primary_key=True)
    role = Column(Text, nullable=False)
    event = Column(Text, nullable=False)
    conversation_id = Column(Text, nullable=False, index=True)
    thread_id = Column(Text, nullable=True)
    task_id = Column(Text, nullable=True, index=True)
    payload = Column(Text, nullable=True)
    agent_name = Column(Text, nullable=True)
    item_metadata = Column("metadata", Text, nullable=True)
    created_at = Column(Text, nullable=False)

    __table_args__ = (
        Index(
            "idx_runtime_item_conv_created_at",
            "conversation_id",
            "created_at",
        ),
    )


class RuntimeTaskRecord(Base):
    """Task records persisted in the main SQL database."""

    __tablename__ = "runtime_tasks"

    task_id = Column(Text, primary_key=True)
    title = Column(Text, nullable=True)
    query = Column(Text, nullable=False)
    conversation_id = Column(Text, nullable=False, index=True)
    thread_id = Column(Text, nullable=False)
    user_id = Column(Text, nullable=False, index=True)
    agent_name = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="pending", index=True)
    pattern = Column(Text, nullable=False, default="once")
    schedule_config = Column(Text, nullable=True)
    handoff_from_super_agent = Column(Boolean, nullable=False, default=False)
    created_at = Column(Text, nullable=False)
    started_at = Column(Text, nullable=True)
    completed_at = Column(Text, nullable=True)
    updated_at = Column(Text, nullable=False)
    error_message = Column(Text, nullable=True)
