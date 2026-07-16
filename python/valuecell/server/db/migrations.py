"""Small, idempotent data migrations required by SaaS cutovers."""

from __future__ import annotations


from loguru import logger
from sqlalchemy.orm import Session

from valuecell.server.db.models.rule_strategy import RuleStrategy
from valuecell.server.db.models.tenant import Tenant, TenantProfile


def migrate_fixed_order_amounts(session: Session) -> int:
    """Replace legacy dynamic sizing with the approved fixed-order contract.

    Existing fixed-quote strategies retain their amount. Legacy equal-split and
    equity-fraction strategies are intentionally reset to the safe 100 USDT
    default because their former values were ratios rather than quote amounts.
    """

    migrated = 0
    for strategy in session.query(RuleStrategy).all():
        config = dict(strategy.config or {})
        risk = dict(config.get("risk") or {})
        if "order_quote_amount" in risk:
            continue
        legacy_mode = risk.pop("size_mode", None)
        legacy_value = risk.pop("size_value", None)
        order_quote_amount = (
            legacy_value
            if legacy_mode == "fixed_quote"
            and isinstance(legacy_value, (int, float))
            and legacy_value > 0
            else 100.0
        )
        risk["order_quote_amount"] = order_quote_amount
        config["risk"] = risk
        strategy.config = config
        migrated += 1
    if migrated:
        session.commit()
        logger.info(
            "Migrated fixed order amounts for {count} rule strategies", count=migrated
        )
    return migrated


def migrate_tenant_profiles(session: Session) -> int:
    """Classify existing workspaces as personal until an admin changes them."""

    profiled_tenant_ids = {
        tenant_id for (tenant_id,) in session.query(TenantProfile.tenant_id).all()
    }
    profiles = [
        TenantProfile(tenant_id=tenant.id, tenant_type="personal")
        for tenant in session.query(Tenant).all()
        if tenant.id not in profiled_tenant_ids
    ]
    if profiles:
        session.add_all(profiles)
        session.commit()
        logger.info(
            "Created profiles for {count} existing tenants", count=len(profiles)
        )
    return len(profiles)
