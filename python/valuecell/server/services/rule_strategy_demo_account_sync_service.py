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
    RuleStrategyExecutionIntent,
)
from valuecell.server.db.models.multi_strategy import StrategySharedAccount
from valuecell.server.db.models.shared_demo_execution import (
    SharedDemoAccountSnapshot,
    SharedDemoAccountSyncState,
    SharedDemoExecutionIntent,
    SharedDemoOrderProjection,
    SharedDemoVenueOrder,
)
from valuecell.server.services.shared_demo_allocation_cap_service import (
    ensure_initial_strategy_cap,
)
from valuecell.server.services.rule_strategy_demo_snapshot_service import (
    get_latest_demo_account_snapshot,
    record_demo_account_snapshot,
)
from valuecell.server.services.sandbox_exchange_trading_service import (
    IGNORED_DUST_STATUS,
    SandboxExchangeTradingService,
)


# Execution intents and venue order projections only leave the gate closed while
# they are still unsettled. Everything else is already a durable end state, so a
# snapshot cycle can declare the shared account fully attributed.
_TERMINAL_EXECUTION_STATUSES = frozenset(
    {
        "filled",
        "closed",
        "canceled",
        "cancelled",
        "failed",
        "rejected",
        "stale",
        IGNORED_DUST_STATUS,
    }
)

# A durable intent only holds the entry gate shut while the shared execution
# chain can still advance it. The chain proves that by mirroring the allocator
# reservation into a ``SharedDemoExecutionIntent`` binding and by persisting its
# venue order before any remote call. Rows written by the pre-shared-account
# code paths received neither, so no code path can ever settle them; counting
# them kept this account's gate blocked forever. They are retained as audit
# history and closed as ``stale`` instead of being deleted.
_EXECUTION_ACTIVITY_WINDOW_MULTIPLIER = 2
_UNBOUND_INTENT_TERMINAL_STATUS = "stale"
_UNBOUND_INTENT_ERROR_CODE = "stale_unbound_submission"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _execution_activity_cutoff() -> datetime:
    """Cutoff for a durable intent the shared chain has not bound yet."""
    window_s = (
        get_settings().DEMO_ACCOUNT_SYNC_INTERVAL_S * _EXECUTION_ACTIVITY_WINDOW_MULTIPLIER
    )
    return _utc_now() - timedelta(seconds=window_s)


def _is_recent_activity(value: datetime | None, *, cutoff: datetime) -> bool:
    """Treat an unproven timestamp as recent so it can never pass as settled."""
    observed = _aware(value)
    return observed is None or observed >= cutoff


def _close_unbound_intent(intent: RuleStrategyExecutionIntent) -> None:
    """Close a durable intent the shared Demo execution chain never accepted.

    It is kept for audit; only its state becomes terminal, so a submission that
    never reached the venue stops holding the shared account gate shut.
    """
    intent.status = _UNBOUND_INTENT_TERMINAL_STATUS
    intent.error_code = _UNBOUND_INTENT_ERROR_CODE
    intent.error_message = "execution intent never entered the shared Demo execution chain"
    intent.terminal_at = _utc_now()


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
    state.last_attempt_at = _utc_now()
    state.last_success_at = _utc_now()
    state.stale_after = state.last_success_at + timedelta(
        seconds=get_settings().DEMO_ACCOUNT_SYNC_INTERVAL_S * 2
    )
    state.consecutive_failures = 0
    state.last_error_code = None
    _reconcile_shared_account(session, account=account, state=state)
    return snapshot


def _reconcile_shared_account(
    session: Session,
    *,
    account: StrategySharedAccount,
    state: SharedDemoAccountSyncState,
) -> None:
    """Advance attribution from persisted execution facts only.

    A shared wallet is attributable when no execution fact can still change. The
    gate therefore counts only work the shared chain can advance: an intent
    whose allocator reservation is mirrored as a ``SharedDemoExecutionIntent``
    binding, or one still inside the activity window of a live tick. Intents the
    chain never accepted are closed as ``stale`` audit history, because no code
    path can resolve them and each one used to keep this gate shut forever.
    Nothing is inferred from balances, and open work never reports completeness.
    """
    cutoff = _execution_activity_cutoff()
    bound_intent_ids = {
        str(binding.intent_id)
        for binding in session.query(SharedDemoExecutionIntent)
        .filter_by(account_id=account.id)
        .all()
    }
    unresolved_intents: list[RuleStrategyExecutionIntent] = []
    for intent in (
        session.query(RuleStrategyExecutionIntent)
        .filter_by(
            tenant_id=account.tenant_id,
            credential_id=account.credential_id,
            execution_target="okx_demo",
        )
        .all()
    ):
        if str(intent.status or "pending") in _TERMINAL_EXECUTION_STATUSES:
            continue
        if str(intent.id) in bound_intent_ids or _is_recent_activity(
            intent.updated_at, cutoff=cutoff
        ):
            unresolved_intents.append(intent)
            continue
        _close_unbound_intent(intent)
    order_ids = [
        row.order_id
        for row in session.query(SharedDemoVenueOrder)
        .filter_by(account_id=account.id)
        .all()
    ]
    projections = (
        session.query(SharedDemoOrderProjection)
        .filter(SharedDemoOrderProjection.order_id.in_(order_ids))
        .all()
        if order_ids
        else []
    )
    unresolved_orders = [
        row
        for row in projections
        if str(row.status or "pending") not in _TERMINAL_EXECUTION_STATUSES
    ]
    state.unresolved_submission_count = len(unresolved_intents) + len(unresolved_orders)
    if state.unresolved_submission_count > 0:
        account.attribution_status = "partial"
        state.reconciliation_status = "reconciling"
        return
    account.attribution_status = "complete"
    state.reconciliation_status = "complete"
    state.last_reconciled_at = _utc_now()


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


def _ensure_initial_caps(
    session: Session,
    *,
    account: StrategySharedAccount,
    strategies: list[RuleStrategy],
) -> None:
    """Create initial caps for Demo strategies created before first sync."""
    for strategy in strategies:
        config = strategy.config if isinstance(strategy.config, dict) else {}
        initial = config.get("initial_capital_quote")
        if not isinstance(initial, (int, float)) or initial <= 0:
            continue
        ensure_initial_strategy_cap(
            session,
            tenant_id=account.tenant_id,
            strategy_id=strategy.strategy_id,
            credential_id=account.credential_id,
            initial_capital_quote=float(initial),
        )


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
            _ensure_initial_caps(session, account=shared, strategies=strategies)
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
