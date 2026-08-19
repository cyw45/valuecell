"""Persistence and daily curve construction for OKX Demo wallet facts."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from valuecell.server.db.models.rule_strategy import (
    RuleStrategyDemoAccountSnapshot,
    RuleStrategyOfficialTestBaseline,
)
from valuecell.server.services.rule_strategy_pnl_service import (
    DailyPnlObservation,
    build_daily_pnl_points,
)


def record_demo_account_snapshot(
    session: Session,
    *,
    tenant_id: str,
    strategy_id: str,
    credential_id: str,
    account: dict[str, Any],
    positions: dict[str, Any],
) -> RuleStrategyDemoAccountSnapshot:
    """Persist one exchange response before it is exposed as historical evidence."""
    observed_at = _parse_timestamp(
        positions.get("checked_at") or account.get("checked_at")
    )
    snapshot = RuleStrategyDemoAccountSnapshot(
        tenant_id=tenant_id,
        strategy_id=strategy_id,
        credential_id=credential_id,
        observed_at=observed_at,
        source=str(account.get("source") or positions.get("source") or "okx_demo"),
        total_usdt_value=_finite(account.get("total_usdt_value")),
        balances=list(account.get("balances") or []),
        positions=list(positions.get("positions") or []),
    )
    session.add(snapshot)
    session.commit()
    session.refresh(snapshot)
    return snapshot


def list_demo_account_snapshots(
    session: Session,
    *,
    tenant_id: str,
    strategy_id: str,
    credential_id: str | None = None,
) -> list[RuleStrategyDemoAccountSnapshot]:
    """Return persisted snapshots, optionally restricted to the current credential."""
    query = session.query(RuleStrategyDemoAccountSnapshot).filter(
        RuleStrategyDemoAccountSnapshot.tenant_id == tenant_id,
        RuleStrategyDemoAccountSnapshot.strategy_id == strategy_id,
    )
    if credential_id is not None:
        query = query.filter(RuleStrategyDemoAccountSnapshot.credential_id == credential_id)
    return query.order_by(
        RuleStrategyDemoAccountSnapshot.observed_at.asc(),
        RuleStrategyDemoAccountSnapshot.id.asc(),
    ).all()


def get_latest_demo_account_snapshot(
    session: Session,
    *,
    tenant_id: str,
    strategy_id: str,
    credential_id: str,
) -> RuleStrategyDemoAccountSnapshot | None:
    """Return the newest snapshot for the strategy's current credential."""
    return (
        session.query(RuleStrategyDemoAccountSnapshot)
        .filter_by(
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            credential_id=credential_id,
        )
        .order_by(
            RuleStrategyDemoAccountSnapshot.observed_at.desc(),
            RuleStrategyDemoAccountSnapshot.id.desc(),
        )
        .first()
    )


def get_demo_account_sync_state(
    session: Session, *, tenant_id: str, strategy_id: str
):
    """Return background sync metadata without contacting the exchange."""
    from valuecell.server.db.models.rule_strategy import RuleStrategyDemoAccountSyncState

    return (
        session.query(RuleStrategyDemoAccountSyncState)
        .filter_by(tenant_id=tenant_id, strategy_id=strategy_id)
        .first()
    )

def get_official_test_baseline(
    session: Session, *, tenant_id: str, strategy_id: str
) -> RuleStrategyOfficialTestBaseline | None:
    return session.query(RuleStrategyOfficialTestBaseline).filter_by(
        tenant_id=tenant_id, strategy_id=strategy_id
    ).first()


def build_demo_daily_curve(
    snapshots: Iterable[RuleStrategyDemoAccountSnapshot],
    *,
    started_at: datetime | None = None,
) -> list[dict[str, float | str]]:
    """Build daily wallet equity from persisted, exchange-observed values only."""
    rows = [
        snapshot
        for snapshot in snapshots
        if snapshot.total_usdt_value is not None
        and (started_at is None or _aware(snapshot.observed_at) >= _aware(started_at))
    ]
    if not rows:
        return []
    baseline = float(rows[0].total_usdt_value)
    observations = [
        DailyPnlObservation(
            occurred_at=_aware(snapshot.observed_at),
            equity_quote=float(snapshot.total_usdt_value),
            action="wallet_snapshot",
        )
        for snapshot in rows
    ]
    return build_daily_pnl_points(
        baseline,
        _aware(rows[0].observed_at),
        observations,
    )


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return _aware(value)
    if isinstance(value, str) and value.strip():
        return _aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None
