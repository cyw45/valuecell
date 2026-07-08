"""
ValueCell Server - Strategy Cycle Diagnostics Model

Stores compact, UI-safe per-cycle diagnostics so users can inspect what the
strategy saw and why it did or did not place orders.
"""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.sql import func

from .base import Base


class StrategyCycleDiagnostics(Base):
    """Compact diagnostics payload for one strategy compose cycle."""

    __tablename__ = "strategy_cycle_diagnostics"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(
        String(100),
        ForeignKey("strategies.strategy_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Runtime strategy identifier",
    )
    compose_id = Column(
        String(200), nullable=False, index=True, comment="Compose cycle identifier"
    )
    payload = Column(JSON, nullable=False, comment="Sanitized per-cycle diagnostics")
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
            "strategy_id", "compose_id", name="uq_strategy_cycle_diagnostics"
        ),
    )
