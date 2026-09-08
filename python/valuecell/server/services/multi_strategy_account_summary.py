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
from valuecell.server.db.models.shared_demo_execution import (
    SharedDemoAccountSnapshot,
    SharedDemoFill,
    SharedDemoOrderProjection,
    SharedDemoStrategyAllocationCap,
    SharedDemoVenueOrder,
)
from valuecell.server.services.rule_strategy_demo_execution_read_model import (
    _pnl_and_curve,
    _shared_evidence_orders,
)


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
            StrategyCapitalReservation.status.in_(
                (
                    "reserved",
                    "occupied",
                    "partially_released",
                    "submission_unknown",
                    "recovery_required",
                )
            ),
        )
        .all()
    )


def _strategy_demo_pnl(
    session: Session,
    *,
    tenant_id: str,
    credential_id: str,
    strategy_id: str,
    account_id: str,
) -> tuple[float | None, float | None, float | None]:
    """Replay only strategy-owned Demo fills and mark open lots with wallet snapshots."""
    venue_orders = session.query(SharedDemoVenueOrder).filter_by(
        tenant_id=tenant_id,
        credential_id=credential_id,
        strategy_id=strategy_id,
        account_id=account_id,
        environment="okx_demo",
    ).all()
    order_ids = [row.order_id for row in venue_orders]
    if not order_ids:
        return None, None, None
    projections = session.query(SharedDemoOrderProjection).filter(
        SharedDemoOrderProjection.order_id.in_(order_ids)
    ).all()
    fills = session.query(SharedDemoFill).filter(
        SharedDemoFill.order_id.in_(order_ids)
    ).order_by(SharedDemoFill.occurred_at.asc()).all()
    orders = _shared_evidence_orders(
        [], fills=fills, venue_orders=venue_orders, projections=projections
    )
    snapshots = session.query(SharedDemoAccountSnapshot).filter_by(
        account_id=account_id,
        tenant_id=tenant_id,
        credential_id=credential_id,
        environment="okx_demo",
    ).order_by(SharedDemoAccountSnapshot.observed_at.desc()).all()
    positions = {"positions": list(snapshots[0].positions or [])} if snapshots else {"positions": []}
    pnl, _ = _pnl_and_curve(orders, positions, (snapshots[0].observed_at.isoformat() if snapshots else datetime.now(timezone.utc).isoformat()))
    realized = float(pnl["realized"]) if pnl.get("realized") is not None else None
    unrealized = float(pnl["unrealized"]) if pnl.get("unrealized") is not None else None
    if realized is None and unrealized is None:
        return None, None, None
    fees = sum(float(row.fee_quote or 0) for row in fills)
    net = (realized or 0.0) + (unrealized or 0.0) - fees
    return realized, unrealized, net


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
        recovery_rows = [
            row for row in rows if row.status in {"submission_unknown", "recovery_required"}
        ]
        state = (
            "recovery_required"
            if any(row.status == "recovery_required" for row in recovery_rows)
            else "submission_unknown"
            if recovery_rows
            else "occupied"
            if occupied > 0
            else "reserved"
            if reserved > 0
            else "available"
        )
        lifecycle_reason = None
        if strategy.status != "running":
            lifecycle_reason = "策略当前未运行，尚未产生本批次执行事实。"
        elif not rows:
            lifecycle_reason = "策略已运行，但当前批次尚无资金预留或订单事实。"
        elif account.attribution_status != "complete":
            lifecycle_reason = "共享钱包已同步，但策略归属成交仍待完整对账。"
        realized, unrealized, net = _strategy_demo_pnl(
            session,
            tenant_id=tenant_id,
            credential_id=credential_id,
            strategy_id=strategy_id,
            account_id=account.id,
        )
        initial_capital = strategy.config.get("initial_capital_quote") if isinstance(strategy.config, dict) else None
        return_base = float(initial_capital) if initial_capital and float(initial_capital) > 0 else reserved or denominator
        return_rate = net / return_base if net is not None and return_base > 0 else None
        cap = (
            session.query(SharedDemoStrategyAllocationCap)
            .filter_by(
                account_id=account.id,
                tenant_id=tenant_id,
                credential_id=credential_id,
                environment=environment,
                strategy_id=strategy_id,
                active=1,
            )
            .order_by(SharedDemoStrategyAllocationCap.version.desc())
            .first()
        )
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
                return_rate_pct=return_rate,
                allocation_state=state,
                lifecycle_reason=lifecycle_reason,
                utilization_denominator_quote=denominator,
                max_reserved_quote=(float(cap.max_reserved_quote) if cap else None),
                max_occupied_quote=(float(cap.max_occupied_quote) if cap else None),
                status=str(strategy.status),
                current_batch_id=getattr(strategy, "current_batch_id", None),
                utilization_ratio=(reserved + occupied) / denominator,
            )
        )
    known_net_pnl = [allocation.net_pnl_quote for allocation in allocations]
    total_strategy_pnl = (
        sum(value for value in known_net_pnl if value is not None)
        if known_net_pnl and all(value is not None for value in known_net_pnl)
        else None
    )
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
