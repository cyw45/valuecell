"""Reconnect reconciliation for V19, with exit completion after venue sync."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.leader_spot_v19_quality import (
    LeaderSpotV19RecoveryVenue,
)
from valuecell.server.db.models.leader_spot_v19 import LeaderSpotV19Event


@dataclass(frozen=True, slots=True)
class LeaderSpotV19RecoveryResult:
    """Durable recovery outcome after reconciliation and due-exit completion."""

    observed_at: datetime
    positions_reconciled: int
    orders_reconciled: int
    exits_submitted: int
    reconciliation_event_id: str
    exit_event_ids: tuple[str, ...]


class LeaderSpotV19RecoveryCoordinator:
    """Reconcile first; only then submit exits proven absent from venue orders."""

    def __init__(self, session: Session) -> None:
        self._session = session

    async def recover(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        venue: LeaderSpotV19RecoveryVenue,
        now: datetime | None = None,
    ) -> LeaderSpotV19RecoveryResult:
        """Synchronize venue state before attempting any locally-triggered exit.

        ``venue.reconcile`` is the only authority for whether a prior order already
        exists. The coordinator never retries an unknown submission blindly.
        """

        observation = await venue.reconcile(tenant_id, strategy_id, batch_id)
        observed_at = observation.observed_at.astimezone(UTC)
        reconciliation_event_id = str(uuid4())
        self._session.add(
            LeaderSpotV19Event(
                event_id=reconciliation_event_id,
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                correlation_id=reconciliation_event_id,
                actor="system",
                reason_code="reconnect_reconciled",
                before_state={},
                after_state={
                    "positions": observation.positions,
                    "orders": observation.orders,
                    "observed_at": observed_at.isoformat(),
                },
            )
        )
        exit_event_ids: list[str] = []
        for exit_request in observation.due_exits:
            if exit_request.venue_order_id is not None:
                continue
            await venue.submit_recovery_exit(
                tenant_id,
                strategy_id,
                batch_id,
                exit_request,
            )
            event_id = str(uuid4())
            exit_event_ids.append(event_id)
            self._session.add(
                LeaderSpotV19Event(
                    event_id=event_id,
                    tenant_id=tenant_id,
                    strategy_id=strategy_id,
                    batch_id=batch_id,
                    correlation_id=reconciliation_event_id,
                    position_id=None,
                    intent_id=None,
                    attempt_id=None,
                    actor="system",
                    reason_code="reconnect_exit_submitted",
                    before_state={},
                    after_state={
                        "symbol": exit_request.symbol,
                        "quantity": exit_request.quantity,
                        "reason_code": exit_request.reason_code,
                    },
                )
            )
        self._session.commit()
        return LeaderSpotV19RecoveryResult(
            observed_at=observed_at,
            positions_reconciled=len(observation.positions),
            orders_reconciled=len(observation.orders),
            exits_submitted=len(exit_event_ids),
            reconciliation_event_id=reconciliation_event_id,
            exit_event_ids=tuple(exit_event_ids),
        )
