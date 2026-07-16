"""Tenant identity models for the SaaS rule-strategy domain."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.sql import func

from .base import Base


class SaaSUser(Base):
    """A locally authenticated SaaS user."""

    __tablename__ = "saas_users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(320), nullable=False, unique=True, index=True)
    password_hash = Column(String(512), nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Tenant(Base):
    """An isolated SaaS workspace."""

    __tablename__ = "tenants"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TenantMembership(Base):
    """Binds an authenticated user to a tenant and role."""

    __tablename__ = "tenant_memberships"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        String(36),
        ForeignKey("saas_users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(32), nullable=False, default="owner")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", name="uq_tenant_membership"),
    )


class TenantProfile(Base):
    """Commercial tenant classification created during registration."""

    __tablename__ = "tenant_profiles"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    tenant_type = Column(String(16), nullable=False, default="personal", index=True)
    organization_name = Column(String(200), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
