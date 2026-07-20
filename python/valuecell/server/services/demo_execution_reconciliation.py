"""Scheduled recovery helpers for ambiguous Demo exchange submissions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from valuecell.server.services.saas_access_service import TenantAccessService


async def reconcile_active_tenant_intents(
    session: Any,
    tenant_ids: list[str],
    service_factory: Callable[[Any], Any],
) -> None:
    """Reconcile ambiguous orders for active tenants without re-submitting them."""
    for tenant_id in tenant_ids:
        if TenantAccessService.access_for(session, tenant_id).active:
            await service_factory(session).reconcile_nonterminal_intents(tenant_id)
