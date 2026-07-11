"""Persistence models for deterministic paper rule strategies."""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String
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
    config = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class RuleStrategyEvaluationJournal(Base):
    """Durable explanation and paper-log record for a single evaluation."""

    __tablename__ = "rule_strategy_evaluation_journal"

    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(String(100), unique=True, nullable=False, index=True)
    strategy_id = Column(
        String(100),
        ForeignKey("rule_strategies.strategy_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id = Column(String(36), nullable=False, index=True)
    result = Column(JSON, nullable=False)
    signals = Column(JSON, nullable=False, default=list)
    trades = Column(JSON, nullable=False, default=list)
    funding = Column(JSON, nullable=False, default=list)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
