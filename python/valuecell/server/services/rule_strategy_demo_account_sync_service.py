"""Background synchronization of OKX Demo wallet facts into local snapshots."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from valuecell.server.config.settings import get_settings
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyDemoAccountSyncState,
)
from valuecell.server.services.rule_strategy_demo_snapshot_service import (
    get_latest_demo_account_snapshot,
    record_demo_account_snapshot,
)
from valuecell.server.services.sandbox_exchange_trading_service import (
    SandboxExchangeTradingService,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _demo_connection(strategy: RuleStrategy) -> str | None:
    config = strategy.config or {}
    execution = config.get("execution") or {}
    if execution.get("environment") != "okx_demo":
        return None
    connection_id = execution.get("sandbox_connection_id")
    return connection_id if isinstance(connection_id, str) and connection_id else None
def _state(
    session: Session, tenant_id: str, strategy_id: str, credential_id: str
) -> RuleStrategyDemoAccountSyncState:
    value = (
        session.query(RuleStrategyDemoAccountSyncState)
        .filter_by(tenant_id=tenant_id, strategy_id=strategy_id)
        .first()
    )
    if value is None:
        value = RuleStrategyDemoAccountSyncState(
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            credential_id=credential_id,
            consecutive_failures=0,
        )
        session.add(value)
        session.flush()
    elif value.credential_id != credential_id:
        value.credential_id = credential_id
        value.latest_snapshot_id = None
    return value


def _strategies(session: Session) -> dict[tuple[str, str], list[RuleStrategy]]:
    grouped: dict[tuple[str, str], list[RuleStrategy]] = defaultdict(list)
    for strategy in (
        session.query(RuleStrategy)
        .filter(RuleStrategy.archived_at.is_(None))
        .all()
    ):
        connection_id = _demo_connection(strategy)
        if connection_id is not None:
            grouped[(strategy.tenant_id, connection_id)].append(strategy)
    return grouped


def _record_failure(
    session: Session,
    strategy: RuleStrategy,
    credential_id: str,
    exc: BaseException,
) -> None:
    state = _state(session, strategy.tenant_id, strategy.strategy_id, credential_id)
    now = _utc_now()
    state.last_attempt_at = now
    state.consecutive_failures += 1
    state.last_error_code = type(exc).__name__
    state.next_retry_at = now + timedelta(seconds=get_settings().DEMO_ACCOUNT_SYNC_INTERVAL_S)
    session.commit()


def _record_success(
    session: Session,
    strategy: RuleStrategy,
    credential_id: str,
    snapshot_id: int,
) -> None:
    state = _state(session, strategy.tenant_id, strategy.strategy_id, credential_id)
    now = _utc_now()
    state.latest_snapshot_id = snapshot_id
    state.last_attempt_at = now
    state.last_success_at = now
    state.consecutive_failures = 0
    state.last_error_code = None
    state.next_retry_at = None
    session.commit()


async def _fetch_account(
    service: SandboxExchangeTradingService,
    tenant_id: str,
    credential_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = get_settings()
    last_error: BaseException | None = None
    for attempt in range(settings.DEMO_ACCOUNT_SYNC_ATTEMPTS):
        try:
            account = await asyncio.wait_for(
                service.balance(tenant_id, credential_id),
                timeout=settings.DEMO_ACCOUNT_READ_TIMEOUT_S,
            )
            positions = await asyncio.wait_for(
                service.positions(tenant_id, credential_id, account=account),
                timeout=settings.DEMO_ACCOUNT_READ_TIMEOUT_S,
            )
            await asyncio.wait_for(
                service.refresh_open_orders(tenant_id, credential_id),
                timeout=settings.DEMO_ACCOUNT_READ_TIMEOUT_S,
            )
            return account, positions
        except Exception as exc:  # one provider failure must not stop other accounts
            last_error = exc
            if attempt + 1 < settings.DEMO_ACCOUNT_SYNC_ATTEMPTS:
                await asyncio.sleep(settings.DEMO_ACCOUNT_SYNC_RETRY_DELAY_S * (attempt + 1))
    assert last_error is not None
    raise last_error


async def sync_demo_account_snapshots(session: Session) -> dict[str, int]:
    """Fetch each bound Demo account once and persist facts for its strategies.

    This function is called by the scheduler, never by an HTTP request. A shared
    credential is fetched once per cycle even when multiple strategies use it.
    """
    grouped = _strategies(session)
    synced = 0
    failed = 0
    service = SandboxExchangeTradingService(session)
    for (tenant_id, credential_id), strategies in grouped.items():
        try:
            account, positions = await _fetch_account(service, tenant_id, credential_id)
            observed_at = positions.get("checked_at") or account.get("checked_at")
            for strategy in strategies:
                latest = get_latest_demo_account_snapshot(
                    session,
                    tenant_id=tenant_id,
                    strategy_id=strategy.strategy_id,
                    credential_id=credential_id,
                )
                if latest is not None and observed_at:
                    incoming_at = datetime.fromisoformat(
                        str(observed_at).replace("Z", "+00:00")
                    )
                    if _aware(latest.observed_at) == _aware(incoming_at):
                        continue
                snapshot = record_demo_account_snapshot(
                    session,
                    tenant_id=tenant_id,
                    strategy_id=strategy.strategy_id,
                    credential_id=credential_id,
                    account=account,
                    positions=positions,
                )
                _record_success(session, strategy, credential_id, snapshot.id)
                synced += 1
        except Exception as exc:
            session.rollback()
            for strategy in strategies:
                _record_failure(session, strategy, credential_id, exc)
            failed += 1
    return {"accounts": len(grouped), "synced": synced, "failed": failed}
