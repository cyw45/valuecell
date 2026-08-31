"""Read models for shared wallet and attributed strategy performance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.multi_strategy import (
    AccountStrategyOverview,
    CapitalAllocatorSummary,
    SharedWalletSummary,
    StrategyAllocation,
)
from valuecell.server.db.models.multi_strategy import (
    StrategyCapitalReservation,
    StrategySharedAccount,
)
from valuecell.server.db.models.rule_strategy import RuleStrategy


class SharedAccountSummaryUnavailable(RuntimeError):
    """Raised when an authoritative shared-wallet summary cannot be built."""


def _observed_at(account: StrategySharedAccount) -> datetime:
    return account.observed_at or datetime.now(timezone.utc)


def _active_reservations(
    session: Session,
    *,
    account_id: str,
    tenant_id: str,
) -> list[StrategyCapitalReservation]:
    return (
        session.query(StrategyCapitalReservation)
        .filter(
            StrategyCapitalReservation.account_id == account_id,
            StrategyCapitalReservation.tenant_id == tenant_id,
            StrategyCapitalReservation.status.in_(("reserved", "occupied", "partially_released")),
        )
        .all()
    )


def build_shared_account_overview(
    session: Session,
    *,
    tenant_id: str,
    credential_id: str,
    environment: str = "okx_demo",
) -> AccountStrategyOverview:
    """Build wallet and attributed allocation facts without assigning shared assets."""
    account = (
        session.query(StrategySharedAccount)
        .filter_by(
            tenant_id=tenant_id,
            credential_id=credential_id,
            environment=environment,
            active=True,
        )
        .first()
    )
    if account is None:
        raise SharedAccountSummaryUnavailable("shared account has no persisted snapshot")
    denominator = account.utilization_denominator_quote
    if denominator is None or denominator <= 0:
        raise SharedAccountSummaryUnavailable("shared account equity is unavailable")
    reservations = _active_reservations(
        session,
        account_id=account.id,
        tenant_id=tenant_id,
    )
    strategies = {
        strategy.strategy_id: strategy
        for strategy in session.query(RuleStrategy)
        .filter(
            RuleStrategy.tenant_id == tenant_id,
            RuleStrategy.archived_at.is_(None),
        )
        .all()
        if (
            isinstance(strategy.config, dict)
            and isinstance(strategy.config.get("execution"), dict)
            and strategy.config["execution"].get("environment") == environment
            and strategy.config["execution"].get("sandbox_connection_id") == credential_id
        )
    }
    grouped: dict[str, list[StrategyCapitalReservation]] = {}
    for reservation in reservations:
        grouped.setdefault(reservation.strategy_id, []).append(reservation)
    allocations: list[StrategyAllocation] = []
    for strategy_id, strategy in strategies.items():
        rows = grouped.get(strategy_id, [])
        reserved = sum(float(row.reserved_quote) for row in rows)
        occupied = sum(float(row.consumed_quote) for row in rows)
        released = sum(float(row.released_quote) for row in rows)
        state = "occupied" if occupied > 0 else "reserved" if reserved > 0 else "available"
        # Shared-wallet strategy PnL must be derived from attributed Demo fills.
        # Paper account rows are a separate ledger and cannot enter this read model.
        realized = None
        unrealized = None
        net = None
        allocations.append(
            StrategyAllocation(
                strategy_id=strategy_id,
                kind=getattr(strategy, "strategy_kind", "configurable_rule"),
                reserved_quote=reserved,
                occupied_quote=occupied,
                released_quote=released,
                realized_pnl_quote=realized,
                unrealized_pnl_quote=unrealized,
                net_pnl_quote=net,
                allocation_state=state,
                utilization_denominator_quote=denominator,
            )
        )
    total_strategy_pnl = None
    wallet = SharedWalletSummary(
        tenant_id=tenant_id,
        credential_id=credential_id,
        environment=environment,
        total_equity_quote=account.wallet_equity_quote,
        available_quote=account.available_quote,
        observed_at=_observed_at(account),
        sync_status=account.sync_status,
        attribution_status=account.attribution_status,
        unassigned_equity_quote=None,
    )
    allocator = CapitalAllocatorSummary(
        wallet_equity_quote=account.wallet_equity_quote,
        available_for_strategies_quote=account.available_quote,
        reserved_quote=account.reserved_quote,
        occupied_notional_quote=account.occupied_notional_quote,
        pending_settlement_quote=account.pending_settlement_quote,
        reusable_quote=account.reusable_quote,
        utilization_denominator_quote=denominator,
        account_utilization_ratio=(
            account.reserved_quote + account.occupied_notional_quote
        ) / denominator,
        allocations=allocations,
        observed_at=_observed_at(account),
    )
    return AccountStrategyOverview(
        wallet=wallet,
        allocator=allocator,
        strategy_pnl_total_quote=total_strategy_pnl,
        wallet_strategy_reconciliation_delta_quote=None,
        data_complete=account.attribution_status == "complete",
        incomplete_reason=(
            None
            if account.attribution_status == "complete"
            else "共享钱包已同步，但全部策略归属事实尚未完整。"
        ),
    )


def shared_account_summary_dict(
    session: Session,
    *,
    tenant_id: str,
    credential_id: str,
    environment: str = "okx_demo",
) -> dict[str, Any]:
    """Return a JSON-ready shared account summary for the API boundary."""
    return build_shared_account_overview(
        session,
        tenant_id=tenant_id,
        credential_id=credential_id,
        environment=environment,
    ).model_dump(mode="json")
