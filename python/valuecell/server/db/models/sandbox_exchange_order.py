"""Durable audit records for explicit sandbox exchange orders."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class SandboxExchangeOrder(Base):
    """Tenant-scoped, idempotent record of a sandbox exchange order."""

    __tablename__ = "sandbox_exchange_orders"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    credential_id = Column(
        String(36),
        ForeignKey("tenant_credentials.id", ondelete="RESTRICT"),
        nullable=False,
    )
    provider = Column(String(16), nullable=False)
    client_order_id = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    order_type = Column(String(8), nullable=False)
    requested_quote = Column(String(32), nullable=False)
    requested_quantity = Column(String(32), nullable=True)
    status = Column(String(32), nullable=False, default="pending")
    exchange_order_id = Column(String(128), nullable=True)
    sandbox = Column(Boolean, nullable=False, default=True)
    response_metadata = Column(JSON, nullable=False, default=dict)
    error_code = Column(String(64), nullable=True)
    # Nullable preserves legacy and manual orders. PostgreSQL migration triggers
    # make an attributed tuple immutable after insert.
    strategy_id = Column(
        String(100),
        ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"),
        nullable=True,
    )
    evaluation_id = Column(
        String(100),
        ForeignKey(
            "rule_strategy_evaluation_journal.evaluation_id", ondelete="RESTRICT"
        ),
        nullable=True,
    )
    execution_generation = Column(Integer, nullable=True)
    execution_source = Column(String(32), nullable=True)
    execution_intent_id = Column(
        String(36),
        ForeignKey("rule_strategy_execution_intents.id", ondelete="RESTRICT"),
        nullable=True,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "client_order_id", name="uq_sandbox_exchange_order_client_id"
        ),
        Index(
            "ix_sandbox_exchange_orders_tenant_client_order",
            "tenant_id",
            "client_order_id",
        ),
        Index(
            "ix_sandbox_exchange_orders_strategy_evaluation",
            "strategy_id",
            "evaluation_id",
        ),
        CheckConstraint(
            "(strategy_id IS NULL AND evaluation_id IS NULL "
            "AND execution_generation IS NULL AND execution_source IS NULL "
            "AND execution_intent_id IS NULL) OR "
            "(strategy_id IS NOT NULL AND evaluation_id IS NOT NULL "
            "AND execution_generation IS NOT NULL AND execution_generation >= 1 "
            "AND execution_source IS NOT NULL AND execution_intent_id IS NOT NULL)",
            name="ck_sandbox_exchange_orders_attribution_complete",
        ),
    )
