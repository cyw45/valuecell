"""Initial strategy capital envelopes for one shared OKX Demo account."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.db.models.multi_strategy import StrategySharedAccount
from valuecell.server.db.models.shared_demo_execution import (
    SharedDemoStrategyAllocationCap,
)


def ensure_initial_strategy_cap(
    session: Session,
    *,
    tenant_id: str,
    strategy_id: str,
    credential_id: str | None,
    initial_capital_quote: float,
) -> SharedDemoStrategyAllocationCap | None:
    """Seed one bounded cap after the shared Demo account identity exists."""
    if credential_id is None:
        return None
    account = (
        session.query(StrategySharedAccount)
        .filter_by(
            tenant_id=tenant_id,
            credential_id=credential_id,
            environment="okx_demo",
            active=True,
        )
        .first()
    )
    if account is None:
        return None
    existing = (
        session.query(SharedDemoStrategyAllocationCap)
        .filter_by(
            account_id=account.id,
            tenant_id=tenant_id,
            credential_id=credential_id,
            environment="okx_demo",
            strategy_id=strategy_id,
            active=1,
        )
        .first()
    )
    if existing is not None:
        return existing
    cap = SharedDemoStrategyAllocationCap(
        cap_id=str(uuid4()),
        account_id=account.id,
        tenant_id=tenant_id,
        credential_id=credential_id,
        environment="okx_demo",
        strategy_id=strategy_id,
        max_reserved_quote=initial_capital_quote,
        max_occupied_quote=initial_capital_quote,
        active=1,
        version=1,
        effective_at=datetime.now(timezone.utc),
    )
    session.add(cap)
    return cap
