"""Strategy-owned paper accounts, positions, and append-only fills."""

from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, Column, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.sql import func

from .base import Base


class FixedPaperAccount(Base):
    """Batch-scoped paper capital state that supports long and short positions."""

    __tablename__ = "fixed_paper_accounts"

    account_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"), nullable=False)
    batch_id = Column(String(36), nullable=False)
    initial_capital_quote = Column(Float, nullable=False)
    quote_balance = Column(Float, nullable=False)
    reserved_quote = Column(Float, nullable=False, default=0.0)
    occupied_quote = Column(Float, nullable=False, default=0.0)
    realized_pnl_quote = Column(Float, nullable=False, default=0.0)
    unrealized_pnl_quote = Column(Float, nullable=False, default=0.0)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "strategy_id", "batch_id", name="uq_fixed_paper_account_batch"),
        CheckConstraint("initial_capital_quote > 0", name="ck_fixed_paper_account_initial"),
        CheckConstraint("reserved_quote >= 0", name="ck_fixed_paper_account_reserved"),
        CheckConstraint("occupied_quote >= 0", name="ck_fixed_paper_account_occupied"),
        CheckConstraint("version >= 1", name="ck_fixed_paper_account_version"),
        Index("ix_fixed_paper_account_tenant_strategy", "tenant_id", "strategy_id"),
    )


class FixedPaperPosition(Base):
    """One strategy-owned long or short paper position."""

    __tablename__ = "fixed_paper_positions"

    position_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id = Column(String(36), ForeignKey("fixed_paper_accounts.account_id", ondelete="CASCADE"), nullable=False)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"), nullable=False)
    batch_id = Column(String(36), nullable=False)
    symbol = Column(String(32), nullable=False)
    pair = Column(String(64), nullable=True)
    side = Column(String(8), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_quote = Column(Float, nullable=False)
    entry_timestamp_ms = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False, default="open", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "strategy_id", "batch_id", "symbol", name="uq_fixed_paper_position_symbol"),
        CheckConstraint("quantity > 0", name="ck_fixed_paper_position_quantity"),
        CheckConstraint("entry_price > 0", name="ck_fixed_paper_position_entry_price"),
        CheckConstraint("entry_quote > 0", name="ck_fixed_paper_position_entry_quote"),
        Index("ix_fixed_paper_position_strategy_status", "strategy_id", "status"),
    )


class FixedPaperFill(Base):
    """Append-only fill fact for one fixed-strategy paper action."""

    __tablename__ = "fixed_paper_fills"

    fill_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id = Column(String(36), ForeignKey("fixed_paper_accounts.account_id", ondelete="RESTRICT"), nullable=False)
    position_id = Column(String(36), ForeignKey("fixed_paper_positions.position_id", ondelete="SET NULL"), nullable=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"), nullable=False)
    batch_id = Column(String(36), nullable=False)
    evaluation_id = Column(String(100), nullable=False)
    idempotency_key = Column(String(160), nullable=False)
    symbol = Column(String(32), nullable=False)
    pair = Column(String(64), nullable=True)
    action = Column(String(16), nullable=False)
    side = Column(String(8), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    quote_amount = Column(Float, nullable=False)
    fee_quote = Column(Float, nullable=False, default=0.0)
    realized_pnl_quote = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_fixed_paper_fill_idempotency"),
        CheckConstraint("quantity > 0", name="ck_fixed_paper_fill_quantity"),
        CheckConstraint("price > 0", name="ck_fixed_paper_fill_price"),
        CheckConstraint("quote_amount > 0", name="ck_fixed_paper_fill_quote"),
        Index("ix_fixed_paper_fill_strategy_batch", "tenant_id", "strategy_id", "batch_id"),
    )
