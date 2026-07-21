"""Persistence models for deterministic paper rule strategies."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean, CheckConstraint, Column, DateTime, ForeignKey, Index, Integer,
    JSON, String, UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class RuleStrategy(Base):
    """A standalone, paper-only deterministic strategy configuration."""

    __tablename__ = "rule_strategies"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(String(100), unique=True, nullable=False, index=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(String(1000), nullable=True)
    status = Column(String(20), nullable=False, default="stopped", index=True)
    paper_mode = Column(Boolean, nullable=False, default=True)
    execution_generation = Column(Integer, nullable=False, default=1, server_default="1")
    config = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("execution_generation >= 1", name="ck_rule_strategies_execution_generation"),
    )


class RuleStrategyEvaluationJournal(Base):
    """Durable explanation and paper-log record for a single evaluation."""

    __tablename__ = "rule_strategy_evaluation_journal"

    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(String(100), unique=True, nullable=False, index=True)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    result = Column(JSON, nullable=False)
    signals = Column(JSON, nullable=False, default=list)
    trades = Column(JSON, nullable=False, default=list)
    funding = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index(
            "ix_rule_strategy_journal_tenant_strategy_created",
            "tenant_id",
            "strategy_id",
            created_at.desc(),
        ),
    )


class RuleStrategyExecutionIntent(Base):
    """Durable, tenant-scoped request to execute one strategy evaluation."""

    __tablename__ = "rule_strategy_execution_intents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"), nullable=False)
    evaluation_id = Column(String(100), ForeignKey("rule_strategy_evaluation_journal.evaluation_id", ondelete="RESTRICT"), nullable=False)
    execution_generation = Column(Integer, nullable=False)
    execution_source = Column(String(32), nullable=False, default="rule_strategy")
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False)
    credential_id = Column(String(36), ForeignKey("tenant_credentials.id", ondelete="RESTRICT"), nullable=False)
    idempotency_key = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    order_type = Column(String(8), nullable=False)
    requested_quote = Column(String(32), nullable=False)
    requested_quantity = Column(String(32), nullable=True)
    status = Column(String(32), nullable=False, default="pending")
    attempt_count = Column(Integer, nullable=False, default=0, server_default="0")
    error_code = Column(String(64), nullable=True)
    error_message = Column(String(1000), nullable=True)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    terminal_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    request_payload = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("execution_generation >= 1", name="ck_rule_strategy_execution_intent_generation"),
        CheckConstraint("attempt_count >= 0", name="ck_rule_strategy_execution_intent_attempt_count"),
        UniqueConstraint("strategy_id", "evaluation_id", "execution_generation", name="uq_rule_strategy_execution_intent"),
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_rule_strategy_execution_intent_tenant_idempotency"),
        Index("ix_rule_strategy_execution_intents_strategy_generation", "strategy_id", "execution_generation"),
        Index("ix_rule_strategy_execution_intents_strategy_status", "strategy_id", "status"),
        Index("ix_rule_strategy_execution_intents_lifecycle", "status", "updated_at"),
    )
