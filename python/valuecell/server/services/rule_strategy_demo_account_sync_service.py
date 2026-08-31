"""Background synchronization of OKX Demo wallet facts into local snapshots."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from valuecell.server.config.settings import get_settings
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyDemoAccountSyncState,
)
from valuecell.server.db.models.multi_strategy import StrategySharedAccount
from valuecell.server.db.models.shared_demo_execution import (
    SharedDemoAccountSnapshot,
    SharedDemoAccountSyncState,
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
def _shared_account(
    session: Session, tenant_id: str, credential_id: str
) -> StrategySharedAccount:
    value = (
        session.query(StrategySharedAccount)
        .filter_by(
            tenant_id=tenant_id,
            credential_id=credential_id,
            environment="okx_demo",
        )
        .first()
    )
    if value is None:
        value = StrategySharedAccount(
            tenant_id=tenant_id,
            credential_id=credential_id,
            environment="okx_demo",
            sync_status="unavailable",
            attribution_status="partial",
        )
        session.add(value)
        session.flush()
    return value


def _update_shared_account(
    account_row: StrategySharedAccount,
    account: dict[str, Any],
    observed_at: str | None,
) -> None:
    balances = account.get("balances")
    usdt = next(
        (
            item
            for item in balances
            if isinstance(item, dict) and item.get("currency") == "USDT"
        ),
        None,
    ) if isinstance(balances, list) else None
    account_row.wallet_equity_quote = account.get("total_usdt_value")
    account_row.available_quote = usdt.get("free") if usdt is not None else None
    account_row.sync_status = "healthy"
    account_row.attribution_status = "partial"
    account_row.observed_at = (
        datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
        if observed_at
        else None
    )
    denominator = account_row.wallet_equity_quote
    account_row.utilization_denominator_quote = (
        denominator if denominator and denominator > 0 else None
    )
    if account_row.reusable_quote is None:
        account_row.reusable_quote = account_row.available_quote



def _observed_at(value: str | None) -> datetime:
    if value is None:
        return _utc_now()
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _shared_sync_state(
    session: Session,
    account: StrategySharedAccount,
) -> SharedDemoAccountSyncState:
    state = session.get(SharedDemoAccountSyncState, account.id)
    if state is not None:
        return state
    state = SharedDemoAccountSyncState(
        account_id=account.id,
        tenant_id=account.tenant_id,
        credential_id=account.credential_id,
        environment=account.environment,
        sync_status="unavailable",
        reconciliation_status="pending",
        consecutive_failures=0,
        unresolved_submission_count=0,
    )
    session.add(state)
    session.flush()
    return state


def _record_shared_snapshot(
    session: Session,
    *,
    account: StrategySharedAccount,
    wallet: dict[str, Any],
    positions: dict[str, Any],
    observed_at: str | None,
) -> SharedDemoAccountSnapshot:
    """Store one immutable wallet observation for a shared Demo credential."""
    timestamp = _observed_at(observed_at)
    snapshot = (
        session.query(SharedDemoAccountSnapshot)
        .filter_by(account_id=account.id, observed_at=timestamp)
        .first()
    )
    if snapshot is None:
        snapshot = SharedDemoAccountSnapshot(
            account_id=account.id,
            tenant_id=account.tenant_id,
            credential_id=account.credential_id,
            environment=account.environment,
            source="okx_account_sync",
            observed_at=timestamp,
            wallet_equity_quote=wallet.get("total_usdt_value"),
            available_quote=account.available_quote,
            balances=list(wallet.get("balances") or []),
            positions=list(positions.get("positions") or []),
            open_orders=[],
        )
        session.add(snapshot)
        session.flush()
    state = _shared_sync_state(session, account)
    state.latest_snapshot_id = snapshot.snapshot_id
    state.sync_status = "healthy"
    state.reconciliation_status = "pending"
    state.last_attempt_at = _utc_now()
    state.last_success_at = _utc_now()
    state.stale_after = state.last_success_at + timedelta(
        seconds=get_settings().DEMO_ACCOUNT_SYNC_INTERVAL_S * 2
    )
    state.consecutive_failures = 0
    state.last_error_code = None
    return snapshot

def _mark_shared_account_failure(
    session: Session, tenant_id: str, credential_id: str
) -> None:
    account = _shared_account(session, tenant_id, credential_id)
    account.sync_status = "unavailable"
    account.attribution_status = "unavailable"
    account.observed_at = _utc_now()
    shared_state = session.get(SharedDemoAccountSyncState, account.id)
    if shared_state is not None:
        shared_state.sync_status = "failed"
        shared_state.reconciliation_status = "blocked"
        shared_state.last_attempt_at = _utc_now()
        shared_state.consecutive_failures += 1
        shared_state.last_error_code = "shared_account_sync_failed"
    session.commit()


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
            shared = _shared_account(session, tenant_id, credential_id)
            _update_shared_account(shared, account, observed_at)
            _record_shared_snapshot(
                session,
                account=shared,
                wallet=account,
                positions=positions,
                observed_at=observed_at,
            )
            session.commit()
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
            _mark_shared_account_failure(session, tenant_id, credential_id)
            for strategy in strategies:
                _record_failure(session, strategy, credential_id, exc)
            failed += 1
    return {"accounts": len(grouped), "synced": synced, "failed": failed}
