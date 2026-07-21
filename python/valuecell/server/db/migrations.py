"""Small, idempotent data and schema migrations required by SaaS cutovers."""

from __future__ import annotations

from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session

from valuecell.server.db.models.rule_strategy import RuleStrategy
from valuecell.server.db.models.tenant import Tenant, TenantProfile

EXECUTION_ATTRIBUTION_MIGRATION_VERSION = "20260719_rule_strategy_execution_attribution_v2"
# Stable, namespaced advisory-lock key for the duration of the migration transaction.
EXECUTION_ATTRIBUTION_MIGRATION_LOCK_KEY = 7720250719
RULE_STRATEGY_JOURNAL_INDEX_NAME = "ix_rule_strategy_journal_tenant_strategy_created"


def ensure_rule_strategy_journal_read_index(session: Session) -> None:
    """Create the tenant/strategy/history index used by strategy read models."""
    session.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {RULE_STRATEGY_JOURNAL_INDEX_NAME} "
            "ON rule_strategy_evaluation_journal "
            "(tenant_id, strategy_id, created_at DESC)"
        )
    )
    session.commit()


def migrate_rule_strategy_execution_attribution(session: Session) -> bool:
    """Install execution-attribution DDL exactly once, failing closed on errors."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"execution attribution migration supports PostgreSQL and SQLite, got {dialect!r}")

    # PostgreSQL locks before reading the marker, preventing competing startup
    # processes from concurrently applying the same DDL and marker.
    if dialect == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": EXECUTION_ATTRIBUTION_MIGRATION_LOCK_KEY})

    session.execute(text(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
        "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    ))
    if session.execute(text("SELECT version FROM schema_migrations WHERE version = :version"), {"version": EXECUTION_ATTRIBUTION_MIGRATION_VERSION}).fetchall():
        return False

    if dialect == "postgresql":
        _migrate_execution_attribution_postgresql(session)
    else:
        _migrate_execution_attribution_sqlite(session)
    session.execute(text("INSERT INTO schema_migrations (version) VALUES (:version)"), {"version": EXECUTION_ATTRIBUTION_MIGRATION_VERSION})
    session.commit()
    logger.info("Applied schema migration {version}", version=EXECUTION_ATTRIBUTION_MIGRATION_VERSION)
    return True


def _intent_table_ddl(json_type: str, timestamp_type: str, payload_default: str) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS rule_strategy_execution_intents (
            id VARCHAR(36) PRIMARY KEY,
            strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE RESTRICT,
            evaluation_id VARCHAR(100) NOT NULL REFERENCES rule_strategy_evaluation_journal(evaluation_id) ON DELETE RESTRICT,
            execution_generation INTEGER NOT NULL CHECK (execution_generation >= 1),
            execution_source VARCHAR(32) NOT NULL DEFAULT 'rule_strategy',
            tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
            credential_id VARCHAR(36) NOT NULL REFERENCES tenant_credentials(id) ON DELETE RESTRICT,
            idempotency_key VARCHAR(128) NOT NULL,
            symbol VARCHAR(32) NOT NULL,
            side VARCHAR(8) NOT NULL,
            order_type VARCHAR(8) NOT NULL,
            requested_quote VARCHAR(32) NOT NULL,
            requested_quantity VARCHAR(32),
            status VARCHAR(32) NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            error_code VARCHAR(64),
            error_message VARCHAR(1000),
            submitted_at {timestamp_type},
            terminal_at {timestamp_type},
            updated_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP,
            request_payload {json_type} NOT NULL DEFAULT {payload_default},
            created_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_rule_strategy_execution_intent UNIQUE (strategy_id, evaluation_id, execution_generation),
            CONSTRAINT uq_rule_strategy_execution_intent_tenant_idempotency UNIQUE (tenant_id, idempotency_key)
        )
    """


def _migrate_execution_attribution_postgresql(session: Session) -> None:
    """Apply transactional PostgreSQL DDL, including legacy order-table upgrades."""
    statements = (
        "ALTER TABLE rule_strategies ADD COLUMN IF NOT EXISTS execution_generation INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE sandbox_exchange_orders ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(100)",
        "ALTER TABLE sandbox_exchange_orders ADD COLUMN IF NOT EXISTS evaluation_id VARCHAR(100)",
        "ALTER TABLE sandbox_exchange_orders ADD COLUMN IF NOT EXISTS execution_generation INTEGER",
        "ALTER TABLE sandbox_exchange_orders ADD COLUMN IF NOT EXISTS execution_source VARCHAR(32)",
        "ALTER TABLE sandbox_exchange_orders ADD COLUMN IF NOT EXISTS execution_intent_id VARCHAR(36)",
        _intent_table_ddl("JSON", "TIMESTAMP WITH TIME ZONE", "'{}'::json"),
        "CREATE INDEX IF NOT EXISTS ix_sandbox_exchange_orders_strategy_evaluation ON sandbox_exchange_orders (strategy_id, evaluation_id)",
        "CREATE INDEX IF NOT EXISTS ix_rule_strategy_execution_intents_strategy_generation ON rule_strategy_execution_intents (strategy_id, execution_generation)",
        "CREATE INDEX IF NOT EXISTS ix_rule_strategy_execution_intents_strategy_status ON rule_strategy_execution_intents (strategy_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_rule_strategy_execution_intents_lifecycle ON rule_strategy_execution_intents (status, updated_at)",
        """DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_rule_strategies_execution_generation') THEN ALTER TABLE rule_strategies ADD CONSTRAINT ck_rule_strategies_execution_generation CHECK (execution_generation >= 1); END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_sandbox_exchange_orders_strategy') THEN ALTER TABLE sandbox_exchange_orders ADD CONSTRAINT fk_sandbox_exchange_orders_strategy FOREIGN KEY (strategy_id) REFERENCES rule_strategies(strategy_id) ON DELETE RESTRICT; END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_sandbox_exchange_orders_evaluation') THEN ALTER TABLE sandbox_exchange_orders ADD CONSTRAINT fk_sandbox_exchange_orders_evaluation FOREIGN KEY (evaluation_id) REFERENCES rule_strategy_evaluation_journal(evaluation_id) ON DELETE RESTRICT; END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_sandbox_exchange_orders_execution_intent') THEN ALTER TABLE sandbox_exchange_orders ADD CONSTRAINT fk_sandbox_exchange_orders_execution_intent FOREIGN KEY (execution_intent_id) REFERENCES rule_strategy_execution_intents(id) ON DELETE RESTRICT; END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_sandbox_exchange_orders_attribution_complete') THEN ALTER TABLE sandbox_exchange_orders ADD CONSTRAINT ck_sandbox_exchange_orders_attribution_complete CHECK ((strategy_id IS NULL AND evaluation_id IS NULL AND execution_generation IS NULL AND execution_source IS NULL AND execution_intent_id IS NULL) OR (strategy_id IS NOT NULL AND evaluation_id IS NOT NULL AND execution_generation IS NOT NULL AND execution_generation >= 1 AND execution_source IS NOT NULL AND execution_intent_id IS NOT NULL)); END IF;
        END $$""",
        """CREATE OR REPLACE FUNCTION prevent_sandbox_order_attribution_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
            IF OLD.strategy_id IS DISTINCT FROM NEW.strategy_id OR OLD.evaluation_id IS DISTINCT FROM NEW.evaluation_id OR OLD.execution_generation IS DISTINCT FROM NEW.execution_generation OR OLD.execution_source IS DISTINCT FROM NEW.execution_source OR OLD.execution_intent_id IS DISTINCT FROM NEW.execution_intent_id THEN RAISE EXCEPTION 'sandbox order execution attribution is immutable'; END IF;
            RETURN NEW;
        END $$""",
        "DROP TRIGGER IF EXISTS trg_sandbox_order_attribution_immutable ON sandbox_exchange_orders",
        "CREATE TRIGGER trg_sandbox_order_attribution_immutable BEFORE UPDATE ON sandbox_exchange_orders FOR EACH ROW EXECUTE FUNCTION prevent_sandbox_order_attribution_mutation()",
    )
    for statement in statements:
        session.execute(text(statement))


def _migrate_execution_attribution_sqlite(session: Session) -> None:
    """SQLite-compatible test implementation, including legacy table additions."""
    columns = {row[1] for row in session.execute(text("PRAGMA table_info(sandbox_exchange_orders)")).fetchall()}
    for name, definition in (("strategy_id", "VARCHAR(100)"), ("evaluation_id", "VARCHAR(100)"), ("execution_generation", "INTEGER"), ("execution_source", "VARCHAR(32)"), ("execution_intent_id", "VARCHAR(36)")):
        if name not in columns:
            session.execute(text(f"ALTER TABLE sandbox_exchange_orders ADD COLUMN {name} {definition}"))
    strategy_columns = {row[1] for row in session.execute(text("PRAGMA table_info(rule_strategies)")).fetchall()}
    if "execution_generation" not in strategy_columns:
        session.execute(text("ALTER TABLE rule_strategies ADD COLUMN execution_generation INTEGER NOT NULL DEFAULT 1"))
    session.execute(text(_intent_table_ddl("JSON", "DATETIME", "'{}'")))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_sandbox_exchange_orders_strategy_evaluation ON sandbox_exchange_orders (strategy_id, evaluation_id)"))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_rule_strategy_execution_intents_strategy_generation ON rule_strategy_execution_intents (strategy_id, execution_generation)"))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_rule_strategy_execution_intents_strategy_status ON rule_strategy_execution_intents (strategy_id, status)"))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_rule_strategy_execution_intents_lifecycle ON rule_strategy_execution_intents (status, updated_at)"))


def migrate_fixed_order_amounts(session: Session) -> int:
    """Replace legacy dynamic sizing with the approved fixed-order contract."""
    migrated = 0
    for strategy in session.query(RuleStrategy).all():
        config = dict(strategy.config or {})
        risk = dict(config.get("risk") or {})
        if "order_quote_amount" in risk:
            continue
        legacy_mode = risk.pop("size_mode", None)
        legacy_value = risk.pop("size_value", None)
        risk["order_quote_amount"] = legacy_value if legacy_mode == "fixed_quote" and isinstance(legacy_value, (int, float)) and legacy_value > 0 else 100.0
        config["risk"] = risk
        strategy.config = config
        migrated += 1
    if migrated:
        session.commit()
        logger.info("Migrated fixed order amounts for {count} rule strategies", count=migrated)
    return migrated


def migrate_tenant_profiles(session: Session) -> int:
    """Classify existing workspaces as personal until an admin changes them."""
    profiled_tenant_ids = {tenant_id for (tenant_id,) in session.query(TenantProfile.tenant_id).all()}
    profiles = [TenantProfile(tenant_id=tenant.id, tenant_type="personal") for tenant in session.query(Tenant).all() if tenant.id not in profiled_tenant_ids]
    if profiles:
        session.add_all(profiles)
        session.commit()
        logger.info("Created profiles for {count} existing tenants", count=len(profiles))
    return len(profiles)
