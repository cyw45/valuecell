"""Encrypted tenant credential metadata for future controlled integrations."""

from __future__ import annotations

import uuid
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class TenantCredential(Base):
    """Encrypted secret payload bound to one tenant; never returned verbatim."""

    __tablename__ = "tenant_credentials"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by_user_id = Column(
        String(36),
        ForeignKey("saas_users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    kind = Column(String(32), nullable=False)
    provider = Column(String(100), nullable=False)
    label = Column(String(200), nullable=False)
    encrypted_payload = Column(String, nullable=False)
    nonce = Column(String(64), nullable=False)
    key_version = Column(String(32), nullable=False, default="local-v1")
    metadata_json = Column(JSON, nullable=False, default=dict)
    revoked = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("tenant_id", "kind", "provider", "label", name="uq_tenant_credential_label"),
    )
