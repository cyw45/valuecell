"""Durable jobs for natural-language rule-strategy compilation."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from .base import Base


class RuleStrategyTextImportJobRecord(Base):
    """Tenant-scoped AI compilation request and its eventual result."""

    __tablename__ = "rule_strategy_text_import_jobs"

    job_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(36), nullable=False, index=True)
    request_id = Column(String(36), nullable=False)
    strategy_text = Column(Text, nullable=False)
    status = Column(String(16), nullable=False, default="pending", index=True)
    proposal = Column(JSON, nullable=True)
    error = Column(String(1000), nullable=True)
    worker_id = Column(String(36), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
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
            "tenant_id",
            "user_id",
            "request_id",
            name="uq_rule_strategy_text_import_request",
        ),
        Index(
            "ix_rule_strategy_text_import_owner_status",
            "tenant_id",
            "user_id",
            "status",
        ),
    )
