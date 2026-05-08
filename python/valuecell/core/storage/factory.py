"""Storage backend factory for runtime persistence."""

from __future__ import annotations

from valuecell.server.config.settings import get_settings

from valuecell.core.conversation.conversation_store import SQLiteConversationStore
from valuecell.core.conversation.item_store import SQLiteItemStore
from valuecell.core.conversation.postgres_stores import (
    PostgresConversationStore,
    PostgresItemStore,
)
from valuecell.core.task.postgres_task_store import PostgresTaskStore
from valuecell.core.task.task_store import SQLiteTaskStore
from valuecell.utils.db import resolve_db_path


def _is_sqlite_url(database_url: str) -> bool:
    return database_url.startswith("sqlite")


def create_conversation_store():
    """Create the default conversation store for the configured database."""
    database_url = get_settings().DATABASE_URL
    if _is_sqlite_url(database_url):
        return SQLiteConversationStore(resolve_db_path())
    return PostgresConversationStore()


def create_item_store():
    """Create the default item store for the configured database."""
    database_url = get_settings().DATABASE_URL
    if _is_sqlite_url(database_url):
        return SQLiteItemStore(resolve_db_path())
    return PostgresItemStore()


def create_task_store():
    """Create the default task store for the configured database."""
    database_url = get_settings().DATABASE_URL
    if _is_sqlite_url(database_url):
        return SQLiteTaskStore(resolve_db_path())
    return PostgresTaskStore()
