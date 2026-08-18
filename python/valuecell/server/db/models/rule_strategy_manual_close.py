"""Durable, user-confirmed manual close commands for OKX Demo strategies."""

from __future__ import annotations

import uuid
from sqlalchemy import Column, DateTime, ForeignKey, Index, JSON, String, UniqueConstraint
from sqlalchemy.sql import func
from .base import Base


class RuleStrategyManualCloseCommand(Base):
    """One idempotent emergency close request and its per-symbol observations."""

    __tablename__ = "rule_strategy_manual_close_commands"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"), nullable=False, index=True)
    requested_by = Column(String(36), nullable=False)
    scope = Column(String(16), nullable=False)
    symbol = Column(String(32), nullable=True)
    idempotency_key = Column(String(128), nullable=False)
    status = Column(String(24), nullable=False, default="pending")
    results = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_rule_strategy_manual_close_idempotency"),
        Index("ix_rule_strategy_manual_close_strategy_created", "strategy_id", "created_at"),
    )
