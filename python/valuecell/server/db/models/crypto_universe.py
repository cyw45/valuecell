"""Persistence for the exchange-derived crypto symbol universe (catalogue).

The catalogue is a data authority, not a display preference: every strategy
evaluation iterates the symbols of its own config, and the catalogue decides
which symbols may be configured at all. Facts therefore come from the exchange
only, and a version is immutable once published so an audit can always replay
what the system believed at a given time.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class CryptoSymbolUniverse(Base):
    """One published, immutable version of the tradeable symbol catalogue."""

    __tablename__ = "crypto_symbol_universes"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(Integer, nullable=False, unique=True)
    status = Column(
        String(16), nullable=False, default="superseded", server_default="superseded"
    )
    source = Column(String(32), nullable=False, default="okx", server_default="okx")
    quote_asset = Column(String(16), nullable=False, default="USDT", server_default="USDT")
    observed_at = Column(DateTime(timezone=True), nullable=False)
    evaluated_count = Column(Integer, nullable=False, default=0)
    admitted_count = Column(Integer, nullable=False, default=0)
    added_count = Column(Integer, nullable=False, default=0)
    removed_count = Column(Integer, nullable=False, default=0)
    retained_count = Column(Integer, nullable=False, default=0)
    min_listing_age_days = Column(Integer, nullable=False, default=90)
    min_average_quote_volume_30d = Column(Float, nullable=False, default=5_000_000.0)
    reason_detail = Column(String(1000), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_crypto_symbol_universe_status", "status", "version"),
    )


class CryptoSymbolUniverseEntry(Base):
    """One symbol's decision plus the exchange facts behind it."""

    __tablename__ = "crypto_symbol_universe_entries"

    id = Column(Integer, primary_key=True, index=True)
    universe_id = Column(
        Integer,
        ForeignKey("crypto_symbol_universes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol = Column(String(32), nullable=False, index=True)
    state = Column(String(16), nullable=False)
    decision = Column(String(16), nullable=False)
    reason_code = Column(String(96), nullable=False)
    reason_detail = Column(String(1000), nullable=True)
    permanent_exclusion = Column(Boolean, nullable=False, default=False)
    listed_at = Column(DateTime(timezone=True), nullable=True)
    listing_age_days = Column(Integer, nullable=True)
    average_quote_volume_30d = Column(Float, nullable=True)
    quote_volume_24h = Column(Float, nullable=True)
    price_quote = Column(Float, nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "universe_id", "symbol", name="uq_crypto_symbol_universe_entry"
        ),
        Index("ix_crypto_symbol_universe_entry_symbol", "symbol", "universe_id"),
    )


class CryptoSymbolExclusion(Base):
    """A symbol the automated sync must never re-admit on its own.

    Delisted and policy-excluded instruments land here. Re-admitting one is a
    deliberate operator action, never a side effect of a later sync run.
    """

    __tablename__ = "crypto_symbol_exclusions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(32), nullable=False, unique=True)
    reason_code = Column(String(96), nullable=False)
    reason_detail = Column(String(1000), nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)