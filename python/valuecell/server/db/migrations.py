"""Small, idempotent data and schema migrations required by SaaS cutovers."""

from __future__ import annotations

from loguru import logger
from sqlalchemy import Boolean, String, bindparam, text
from sqlalchemy.orm import Session

from valuecell.server.db.models.base import Base
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

RULE_STRATEGY_ARCHIVING_MIGRATION_VERSION = "20260801_rule_strategy_archiving_v1"
# Stable, namespaced advisory-lock key for the duration of the migration transaction.
RULE_STRATEGY_ARCHIVING_MIGRATION_LOCK_KEY = 7720250720

EXECUTION_BATCH_MIGRATION_VERSION = "20260820_rule_strategy_execution_batches_v1"
EXECUTION_BATCH_MIGRATION_LOCK_KEY = 7720250722

MULTI_STRATEGY_MIGRATION_VERSION = "20260828_multi_strategy_account_v1"
MULTI_STRATEGY_MIGRATION_LOCK_KEY = 7720250731
FIXED_PAPER_LEDGER_MIGRATION_VERSION = "20260829_fixed_strategy_paper_ledger_v1"
FIXED_PAPER_LEDGER_MIGRATION_LOCK_KEY = 7720250732
SHARED_DEMO_EXECUTION_MIGRATION_VERSION = "20260829_shared_demo_execution_v1"
SHARED_DEMO_EXECUTION_MIGRATION_LOCK_KEY = 7720250733


def migrate_fixed_strategy_paper_ledger(session: Session) -> bool:
    """Install side-aware fixed-strategy paper ledger tables idempotently."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"fixed paper ledger migration does not support {dialect!r}")
    if dialect == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": FIXED_PAPER_LEDGER_MIGRATION_LOCK_KEY})
    session.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
    if session.execute(text("SELECT 1 FROM schema_migrations WHERE version=:version"), {"version": FIXED_PAPER_LEDGER_MIGRATION_VERSION}).first():
        return False
    ts = "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "DATETIME"
    session.execute(text(f"""CREATE TABLE IF NOT EXISTS fixed_paper_accounts (
        account_id VARCHAR(36) PRIMARY KEY, tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
        strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE RESTRICT,
        batch_id VARCHAR(36) NOT NULL, initial_capital_quote FLOAT NOT NULL, quote_balance FLOAT NOT NULL,
        reserved_quote FLOAT NOT NULL DEFAULT 0, occupied_quote FLOAT NOT NULL DEFAULT 0,
        realized_pnl_quote FLOAT NOT NULL DEFAULT 0, unrealized_pnl_quote FLOAT NOT NULL DEFAULT 0,
        version INTEGER NOT NULL DEFAULT 1, created_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_fixed_paper_account_batch UNIQUE (tenant_id, strategy_id, batch_id),
        CONSTRAINT ck_fixed_paper_account_initial CHECK (initial_capital_quote > 0),
        CONSTRAINT ck_fixed_paper_account_reserved CHECK (reserved_quote >= 0),
        CONSTRAINT ck_fixed_paper_account_occupied CHECK (occupied_quote >= 0),
        CONSTRAINT ck_fixed_paper_account_version CHECK (version >= 1)
    )"""))
    session.execute(text(f"""CREATE TABLE IF NOT EXISTS fixed_paper_positions (
        position_id VARCHAR(36) PRIMARY KEY, account_id VARCHAR(36) NOT NULL REFERENCES fixed_paper_accounts(account_id) ON DELETE CASCADE,
        tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
        strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE RESTRICT,
        batch_id VARCHAR(36) NOT NULL, symbol VARCHAR(32) NOT NULL, pair VARCHAR(64), side VARCHAR(8) NOT NULL,
        quantity FLOAT NOT NULL, entry_price FLOAT NOT NULL, entry_quote FLOAT NOT NULL,
        entry_timestamp_ms BIGINT NOT NULL, status VARCHAR(16) NOT NULL DEFAULT 'open',
        created_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_fixed_paper_position_symbol UNIQUE (tenant_id, strategy_id, batch_id, symbol),
        CONSTRAINT ck_fixed_paper_position_quantity CHECK (quantity > 0),
        CONSTRAINT ck_fixed_paper_position_entry_price CHECK (entry_price > 0),
        CONSTRAINT ck_fixed_paper_position_entry_quote CHECK (entry_quote > 0)
    )"""))
    session.execute(text(f"""CREATE TABLE IF NOT EXISTS fixed_paper_fills (
        fill_id VARCHAR(36) PRIMARY KEY, account_id VARCHAR(36) NOT NULL REFERENCES fixed_paper_accounts(account_id) ON DELETE RESTRICT,
        position_id VARCHAR(36) REFERENCES fixed_paper_positions(position_id) ON DELETE SET NULL,
        tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
        strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE RESTRICT,
        batch_id VARCHAR(36) NOT NULL, evaluation_id VARCHAR(100) NOT NULL, idempotency_key VARCHAR(160) NOT NULL,
        symbol VARCHAR(32) NOT NULL, pair VARCHAR(64), action VARCHAR(16) NOT NULL, side VARCHAR(8) NOT NULL,
        quantity FLOAT NOT NULL, price FLOAT NOT NULL, quote_amount FLOAT NOT NULL, fee_quote FLOAT NOT NULL DEFAULT 0,
        realized_pnl_quote FLOAT NOT NULL DEFAULT 0, created_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_fixed_paper_fill_idempotency UNIQUE (tenant_id, idempotency_key),
        CONSTRAINT ck_fixed_paper_fill_quantity CHECK (quantity > 0),
        CONSTRAINT ck_fixed_paper_fill_price CHECK (price > 0),
        CONSTRAINT ck_fixed_paper_fill_quote CHECK (quote_amount > 0)
    )"""))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_fixed_paper_account_tenant_strategy ON fixed_paper_accounts (tenant_id, strategy_id)"))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_fixed_paper_position_strategy_status ON fixed_paper_positions (strategy_id, status)"))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_fixed_paper_fill_strategy_batch ON fixed_paper_fills (tenant_id, strategy_id, batch_id)"))
    session.execute(text("INSERT INTO schema_migrations (version) VALUES (:version)"), {"version": FIXED_PAPER_LEDGER_MIGRATION_VERSION})
    session.commit()
    return True



def migrate_multi_strategy_account(session: Session) -> bool:
    """Install additive strategy identity and shared-account allocator tables."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"multi-strategy migration does not support {dialect!r}")
    if dialect == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": MULTI_STRATEGY_MIGRATION_LOCK_KEY})
    session.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
    if session.execute(text("SELECT 1 FROM schema_migrations WHERE version=:version"), {"version": MULTI_STRATEGY_MIGRATION_VERSION}).first():
        return False
    ts = "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "DATETIME"
    session.execute(text(f"""CREATE TABLE IF NOT EXISTS strategy_shared_accounts (
        id VARCHAR(36) PRIMARY KEY, tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
        credential_id VARCHAR(36) NOT NULL REFERENCES tenant_credentials(id) ON DELETE RESTRICT,
        environment VARCHAR(16) NOT NULL DEFAULT 'okx_demo', wallet_equity_quote FLOAT,
        available_quote FLOAT, reserved_quote FLOAT NOT NULL DEFAULT 0, occupied_notional_quote FLOAT NOT NULL DEFAULT 0,
        pending_settlement_quote FLOAT NOT NULL DEFAULT 0, reusable_quote FLOAT,
        utilization_denominator_quote FLOAT, sync_status VARCHAR(16) NOT NULL DEFAULT 'unavailable',
        attribution_status VARCHAR(16) NOT NULL DEFAULT 'unavailable', observed_at {ts}, version INTEGER NOT NULL DEFAULT 1,
        active BOOLEAN NOT NULL DEFAULT TRUE, created_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_strategy_shared_account_scope UNIQUE (tenant_id, credential_id, environment),
        CONSTRAINT ck_strategy_shared_account_reserved CHECK (reserved_quote >= 0),
        CONSTRAINT ck_strategy_shared_account_occupied CHECK (occupied_notional_quote >= 0),
        CONSTRAINT ck_strategy_shared_account_pending CHECK (pending_settlement_quote >= 0),
        CONSTRAINT ck_strategy_shared_account_version CHECK (version >= 1)
    )"""))
    session.execute(text(f"""CREATE TABLE IF NOT EXISTS strategy_capital_reservations (
        reservation_id VARCHAR(36) PRIMARY KEY, account_id VARCHAR(36) NOT NULL REFERENCES strategy_shared_accounts(id) ON DELETE RESTRICT,
        tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
        strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE RESTRICT,
        batch_id VARCHAR(36), intent_id VARCHAR(36), idempotency_key VARCHAR(128) NOT NULL,
        symbol VARCHAR(32) NOT NULL, side VARCHAR(16) NOT NULL, requested_quote FLOAT NOT NULL,
        reserved_quote FLOAT NOT NULL, consumed_quote FLOAT NOT NULL DEFAULT 0, released_quote FLOAT NOT NULL DEFAULT 0,
        status VARCHAR(24) NOT NULL DEFAULT 'reserved', reason VARCHAR(1000),
        created_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_strategy_reservation_idempotency UNIQUE (tenant_id, idempotency_key),
        CONSTRAINT ck_strategy_reservation_requested CHECK (requested_quote >= 0),
        CONSTRAINT ck_strategy_reservation_reserved CHECK (reserved_quote >= 0),
        CONSTRAINT ck_strategy_reservation_consumed CHECK (consumed_quote >= 0),
        CONSTRAINT ck_strategy_reservation_released CHECK (released_quote >= 0),
        CONSTRAINT ck_strategy_reservation_settled CHECK (consumed_quote + released_quote <= requested_quote)
    )"""))
    additions = {
        "rule_strategies": {
            "strategy_kind": "VARCHAR(32) NOT NULL DEFAULT 'configurable_rule'",
            "strategy_version": "VARCHAR(64) NOT NULL DEFAULT 'existing'",
            "code_fingerprint": "VARCHAR(128) NOT NULL DEFAULT 'legacy-configurable'",
        },
        "rule_strategy_execution_batches": {
            "strategy_kind": "VARCHAR(32) NOT NULL DEFAULT 'configurable_rule'",
            "strategy_version": "VARCHAR(64) NOT NULL DEFAULT 'existing'",
            "code_fingerprint": "VARCHAR(128) NOT NULL DEFAULT 'legacy-configurable'",
        },
        "rule_strategy_execution_intents": {
            "reservation_id": "VARCHAR(36)",
        },
    }
    for table, columns in additions.items():
        existing = {row[1] for row in session.execute(text(f"PRAGMA table_info({table})")).fetchall()} if dialect == "sqlite" else set()
        for name, definition in columns.items():
            if dialect == "postgresql":
                session.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {definition}"))
            elif name not in existing:
                session.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_strategy_shared_account_tenant_active ON strategy_shared_accounts (tenant_id, active)"))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_strategy_reservation_account_status ON strategy_capital_reservations (account_id, status)"))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_strategy_reservation_strategy_status ON strategy_capital_reservations (strategy_id, status)"))
    session.execute(text("INSERT INTO schema_migrations (version) VALUES (:version)"), {"version": MULTI_STRATEGY_MIGRATION_VERSION})
    session.commit()
    return True



def migrate_shared_demo_execution_storage(session: Session) -> bool:
    """Install isolated shared-OKX-Demo evidence storage without ownership backfills."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "shared Demo execution migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": SHARED_DEMO_EXECUTION_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT 1 FROM schema_migrations WHERE version = :version"),
        {"version": SHARED_DEMO_EXECUTION_MIGRATION_VERSION},
    ).first():
        return False

    _migrate_strategy_capital_reservation_settlement_constraint(session, dialect)
    _create_shared_demo_execution_identity_indexes(session)
    # Composite Demo foreign keys resolve against this existing account metadata.
    from valuecell.server.db.models.multi_strategy import StrategySharedAccount  # noqa: F401

    from valuecell.server.db.models.shared_demo_execution import (
        SharedDemoAccountSnapshot,
        SharedDemoAccountSyncState,
        SharedDemoExecutionIntent,
        SharedDemoExecutionReservation,
        SharedDemoFill,
        SharedDemoOrderProjection,
        SharedDemoReservationRecoveryEvent,
        SharedDemoStrategyAllocationCap,
        SharedDemoVenueOrder,
    )

    Base.metadata.create_all(
        bind=session.bind,
        tables=[
            SharedDemoAccountSnapshot.__table__,
            SharedDemoAccountSyncState.__table__,
            SharedDemoExecutionReservation.__table__,
            SharedDemoExecutionIntent.__table__,
            SharedDemoVenueOrder.__table__,
            SharedDemoOrderProjection.__table__,
            SharedDemoFill.__table__,
            SharedDemoReservationRecoveryEvent.__table__,
            SharedDemoStrategyAllocationCap.__table__,
        ],
    )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": SHARED_DEMO_EXECUTION_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=SHARED_DEMO_EXECUTION_MIGRATION_VERSION,
    )
    return True


def _create_shared_demo_execution_identity_indexes(session: Session) -> None:
    """Make composite scope foreign keys valid for both supported databases."""
    statements = (
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_shared_account_scope_identity "
        "ON strategy_shared_accounts (id, tenant_id, credential_id, environment)",
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_rule_strategy_tenant_identity "
        "ON rule_strategies (tenant_id, strategy_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_rule_strategy_batch_scope_identity "
        "ON rule_strategy_execution_batches (tenant_id, strategy_id, batch_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_rule_strategy_intent_scope_identity "
        "ON rule_strategy_execution_intents (id, tenant_id, strategy_id, batch_id)",
        "CREATE INDEX IF NOT EXISTS ix_strategy_reservation_account_status "
        "ON strategy_capital_reservations (account_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_strategy_reservation_strategy_status "
        "ON strategy_capital_reservations (strategy_id, status)",
    )
    for statement in statements:
        session.execute(text(statement))


def provision_quant_tables_with_shared_demo_constraints(session: Session) -> None:
    """Provision quant tables in FK-safe order under one PostgreSQL DDL lock.

    Existing installations need composite parent identities before PostgreSQL
    accepts the shared-Demo foreign keys. Fresh installations first need those
    parent tables themselves. Create non-shared tables, then their identity
    indexes, then the shared-Demo tables while retaining one transaction-level
    advisory lock for the complete sequence.
    """
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "quant table provisioning supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": SHARED_DEMO_EXECUTION_MIGRATION_LOCK_KEY},
        )

    connection = session.connection()
    shared_tables = [
        table
        for table in Base.metadata.sorted_tables
        if table.name.startswith("shared_demo_")
    ]
    foundational_tables = [
        table for table in Base.metadata.sorted_tables if table not in shared_tables
    ]
    Base.metadata.create_all(bind=connection, tables=foundational_tables)
    _create_shared_demo_execution_identity_indexes(session)
    Base.metadata.create_all(bind=connection, tables=shared_tables)
    session.commit()


def _migrate_strategy_capital_reservation_settlement_constraint(
    session: Session,
    dialect: str,
) -> None:
    """Treat reserved quote as outstanding, not lifetime settled quote.

    No ownership fields are inferred or populated. SQLite rebuilds only this
    existing table definition while preserving its rows verbatim because it
    cannot replace a named CHECK constraint in place.
    """
    if dialect == "postgresql":
        session.execute(
            text(
                "ALTER TABLE strategy_capital_reservations "
                "DROP CONSTRAINT IF EXISTS ck_strategy_reservation_settled"
            )
        )
        session.execute(
            text(
                "ALTER TABLE strategy_capital_reservations "
                "ADD CONSTRAINT ck_strategy_reservation_settled "
                "CHECK (consumed_quote + released_quote <= requested_quote)"
            )
        )
        return

    existing_sql = session.execute(
        text(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'strategy_capital_reservations'"
        )
    ).scalar_one_or_none()
    if existing_sql is None:
        return
    normalized_sql = existing_sql.lower().replace(" ", "").replace("\n", "")
    if "consumed_quote+released_quote<=requested_quote" in normalized_sql:
        return
    session.execute(
        text(
            "CREATE TABLE strategy_capital_reservations__settlement_v2 ("
            "reservation_id VARCHAR(36) PRIMARY KEY, "
            "account_id VARCHAR(36) NOT NULL REFERENCES strategy_shared_accounts(id) "
            "ON DELETE RESTRICT, "
            "tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT, "
            "strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) "
            "ON DELETE RESTRICT, "
            "batch_id VARCHAR(36), intent_id VARCHAR(36), "
            "idempotency_key VARCHAR(128) NOT NULL, symbol VARCHAR(32) NOT NULL, "
            "side VARCHAR(16) NOT NULL, requested_quote FLOAT NOT NULL, "
            "reserved_quote FLOAT NOT NULL, consumed_quote FLOAT NOT NULL DEFAULT 0, "
            "released_quote FLOAT NOT NULL DEFAULT 0, status VARCHAR(24) NOT NULL "
            "DEFAULT 'reserved', reason VARCHAR(1000), "
            "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "CONSTRAINT uq_strategy_reservation_idempotency "
            "UNIQUE (tenant_id, idempotency_key), "
            "CONSTRAINT ck_strategy_reservation_requested CHECK (requested_quote >= 0), "
            "CONSTRAINT ck_strategy_reservation_reserved CHECK (reserved_quote >= 0), "
            "CONSTRAINT ck_strategy_reservation_consumed CHECK (consumed_quote >= 0), "
            "CONSTRAINT ck_strategy_reservation_released CHECK (released_quote >= 0), "
            "CONSTRAINT ck_strategy_reservation_settled "
            "CHECK (consumed_quote + released_quote <= requested_quote)"
            ")"
        )
    )
    session.execute(
        text(
            "INSERT INTO strategy_capital_reservations__settlement_v2 "
            "(reservation_id, account_id, tenant_id, strategy_id, batch_id, intent_id, "
            "idempotency_key, symbol, side, requested_quote, reserved_quote, "
            "consumed_quote, released_quote, status, reason, created_at, updated_at) "
            "SELECT reservation_id, account_id, tenant_id, strategy_id, batch_id, intent_id, "
            "idempotency_key, symbol, side, requested_quote, reserved_quote, "
            "consumed_quote, released_quote, status, reason, created_at, updated_at "
            "FROM strategy_capital_reservations"
        )
    )
    session.execute(text("DROP TABLE strategy_capital_reservations"))
    session.execute(
        text(
            "ALTER TABLE strategy_capital_reservations__settlement_v2 "
            "RENAME TO strategy_capital_reservations"
        )
    )


def migrate_rule_strategy_execution_batches(session: Session) -> bool:
    """Install batch lifecycle and nullable attribution columns idempotently."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"execution batch migration does not support {dialect!r}")
    if dialect == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": EXECUTION_BATCH_MIGRATION_LOCK_KEY})
    session.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
    if session.execute(text("SELECT 1 FROM schema_migrations WHERE version=:version"), {"version": EXECUTION_BATCH_MIGRATION_VERSION}).first():
        return False
    ts = "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "DATETIME"
    session.execute(text(f"""CREATE TABLE IF NOT EXISTS rule_strategy_execution_batches (
        batch_id VARCHAR(36) PRIMARY KEY, tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
        strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE CASCADE,
        strategy_name_snapshot VARCHAR(200) NOT NULL, execution_generation INTEGER NOT NULL,
        status VARCHAR(16) NOT NULL DEFAULT 'running', started_at {ts} NOT NULL DEFAULT CURRENT_TIMESTAMP,
        stopped_at {ts}, config_snapshot JSON NOT NULL)"""))
    session.execute(text("CREATE INDEX IF NOT EXISTS ix_rule_strategy_batches_tenant_strategy_started ON rule_strategy_execution_batches (tenant_id, strategy_id, started_at DESC)"))
    additions = {"rule_strategies": {"current_batch_id": "VARCHAR(36)"}, "rule_strategy_evaluation_journal": {"batch_id": "VARCHAR(36)"}, "rule_strategy_execution_intents": {"batch_id": "VARCHAR(36)"}, "sandbox_exchange_orders": {"batch_id": "VARCHAR(36)"}}
    for table, columns in additions.items():
        existing = {row[1] for row in session.execute(text(f"PRAGMA table_info({table})")).fetchall()} if dialect == "sqlite" else set()
        for name, definition in columns.items():
            if dialect == "postgresql":
                session.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {definition}"))
            elif name not in existing:
                session.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))
    session.execute(text("INSERT INTO schema_migrations (version) VALUES (:version)"), {"version": EXECUTION_BATCH_MIGRATION_VERSION})
    session.commit()
    return True


def migrate_rule_strategy_archiving(session: Session) -> bool:
    """Install archive-state DDL exactly once, failing closed on errors."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "rule strategy archiving migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )

    # PostgreSQL locks before reading the marker, preventing competing startup
    # processes from concurrently applying the same DDL and marker.
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": RULE_STRATEGY_ARCHIVING_MIGRATION_LOCK_KEY},
        )

    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": RULE_STRATEGY_ARCHIVING_MIGRATION_VERSION},
    ).fetchall():
        return False

    if dialect == "postgresql":
        _migrate_rule_strategy_archiving_postgresql(session)
    else:
        _migrate_rule_strategy_archiving_sqlite(session)
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": RULE_STRATEGY_ARCHIVING_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=RULE_STRATEGY_ARCHIVING_MIGRATION_VERSION,
    )
    return True


def _migrate_rule_strategy_archiving_postgresql(session: Session) -> None:
    """Apply archive-state DDL with PostgreSQL's idempotent column syntax."""
    session.execute(
        text(
            "ALTER TABLE rule_strategies ADD COLUMN IF NOT EXISTS archived_at "
            "TIMESTAMP WITH TIME ZONE NULL"
        )
    )


def _migrate_rule_strategy_archiving_sqlite(session: Session) -> None:
    """Apply archive-state DDL after inspecting SQLite's legacy table shape."""
    columns = {
        row[1]
        for row in session.execute(text("PRAGMA table_info(rule_strategies)")).fetchall()
    }
    if "archived_at" not in columns:
        session.execute(
            text(
                "ALTER TABLE rule_strategies ADD COLUMN archived_at "
                "TIMESTAMP WITH TIME ZONE NULL"
            )
        )



STRATEGY_PRODUCT_STATE_MIGRATION_VERSION = "20260805_strategy_product_state_v1"
STRATEGY_PRODUCT_STATE_MIGRATION_LOCK_KEY = 7720250721


def migrate_strategy_product_state(session: Session) -> bool:
    """Install current strategy account state before scheduler reconciliation."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "strategy product state migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": STRATEGY_PRODUCT_STATE_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    applied = session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": STRATEGY_PRODUCT_STATE_MIGRATION_VERSION},
    ).fetchall()
    if applied:
        return False
    _create_strategy_product_state_tables(session, dialect)
    _extend_execution_intent_state(session, dialect)
    session.execute(
        text("DROP INDEX IF EXISTS uq_rule_strategies_tenant_single_running")
    )
    _backfill_strategy_product_state(session)
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": STRATEGY_PRODUCT_STATE_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=STRATEGY_PRODUCT_STATE_MIGRATION_VERSION,
    )
    return True


def _create_strategy_product_state_tables(session: Session, dialect: str) -> None:
    timestamp_type = "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "DATETIME"
    primary_id = "BIGSERIAL PRIMARY KEY" if dialect == "postgresql" else "INTEGER PRIMARY KEY AUTOINCREMENT"
    json_type = "JSON"
    statements = (
        f"""CREATE TABLE IF NOT EXISTS rule_strategy_accounts (
            id {primary_id},
            tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
            strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE CASCADE,
            scope VARCHAR(32) NOT NULL, credential_id VARCHAR(36) REFERENCES tenant_credentials(id) ON DELETE RESTRICT,
            allocation_quote FLOAT NOT NULL, quote_balance FLOAT NOT NULL, positions {json_type} NOT NULL DEFAULT '{{}}',
            realized_pnl_quote FLOAT NOT NULL DEFAULT 0, unrealized_pnl_quote FLOAT NOT NULL DEFAULT 0,
            equity_quote FLOAT NOT NULL, version INTEGER NOT NULL DEFAULT 1, active BOOLEAN NOT NULL DEFAULT 1,
            created_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_rule_strategy_account_tenant_strategy UNIQUE (tenant_id, strategy_id)
        )""",
        f"""CREATE TABLE IF NOT EXISTS rule_strategy_risk_states (
            id {primary_id},
            account_id INTEGER NOT NULL UNIQUE REFERENCES rule_strategy_accounts(id) ON DELETE CASCADE,
            tenant_id VARCHAR(36) NOT NULL, strategy_id VARCHAR(100) NOT NULL, state VARCHAR(16) NOT NULL,
            daily_equity_baseline FLOAT NOT NULL, high_water_equity FLOAT NOT NULL,
            current_drawdown_pct FLOAT NOT NULL DEFAULT 0, cooldown_until {timestamp_type},
            reason_code VARCHAR(96), reason_detail VARCHAR(1000), version INTEGER NOT NULL DEFAULT 1,
            updated_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""",
        f"""CREATE TABLE IF NOT EXISTS rule_strategy_monitor_symbols (
            id {primary_id},
            tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
            strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE CASCADE,
            symbol VARCHAR(32) NOT NULL, state VARCHAR(16) NOT NULL DEFAULT 'candidate',
            reason_code VARCHAR(96), reason_detail VARCHAR(1000), evaluated_at {timestamp_type},
            next_check_at {timestamp_type}, protected_held BOOLEAN NOT NULL DEFAULT 0,
            consecutive_low_volume_days INTEGER NOT NULL DEFAULT 0, lease_owner VARCHAR(100), lease_until {timestamp_type},
            CONSTRAINT uq_rule_strategy_monitor_symbol UNIQUE (tenant_id, strategy_id, symbol)
        )""",
        f"""CREATE TABLE IF NOT EXISTS rule_strategy_order_attempts (
            id VARCHAR(36) PRIMARY KEY, intent_id VARCHAR(36) NOT NULL REFERENCES rule_strategy_execution_intents(id) ON DELETE CASCADE,
            tenant_id VARCHAR(36) NOT NULL, venue VARCHAR(32) NOT NULL, client_order_id VARCHAR(128) NOT NULL,
            venue_order_id VARCHAR(128), requested_price VARCHAR(32), requested_quantity VARCHAR(32) NOT NULL,
            status VARCHAR(32) NOT NULL, reconciliation_source VARCHAR(32), error_code VARCHAR(96),
            created_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_rule_strategy_order_attempt_venue_client UNIQUE (venue, client_order_id)
        )""",
        f"""CREATE TABLE IF NOT EXISTS rule_strategy_fills (
            id VARCHAR(36) PRIMARY KEY, intent_id VARCHAR(36) NOT NULL REFERENCES rule_strategy_execution_intents(id) ON DELETE CASCADE,
            attempt_id VARCHAR(36) REFERENCES rule_strategy_order_attempts(id) ON DELETE SET NULL,
            tenant_id VARCHAR(36) NOT NULL, average_price VARCHAR(32) NOT NULL, quantity VARCHAR(32) NOT NULL,
            fee_quote VARCHAR(32) NOT NULL DEFAULT '0', remaining_quantity VARCHAR(32) NOT NULL DEFAULT '0',
            observed_slippage_pct VARCHAR(32) NOT NULL DEFAULT '0', reconciliation_source VARCHAR(32) NOT NULL,
            created_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""",
        f"""CREATE TABLE IF NOT EXISTS rule_strategy_events (
            id VARCHAR(36) PRIMARY KEY, tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
            strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE CASCADE,
            account_id INTEGER REFERENCES rule_strategy_accounts(id) ON DELETE SET NULL, correlation_id VARCHAR(100) NOT NULL,
            evaluation_id VARCHAR(100), intent_id VARCHAR(36), order_attempt_id VARCHAR(36), monitor_symbol_id INTEGER,
            actor VARCHAR(16) NOT NULL, reason_code VARCHAR(96) NOT NULL, payload_version INTEGER NOT NULL DEFAULT 1,
            before_state {json_type} NOT NULL DEFAULT '{{}}', after_state {json_type} NOT NULL DEFAULT '{{}}',
            created_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""",
        f"""CREATE TABLE IF NOT EXISTS rule_strategy_execution_leases (
            strategy_id VARCHAR(100) NOT NULL REFERENCES rule_strategies(strategy_id) ON DELETE CASCADE,
            execution_generation INTEGER NOT NULL, owner_id VARCHAR(100) NOT NULL, expires_at {timestamp_type} NOT NULL,
            updated_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (strategy_id, execution_generation)
        )""",
        "CREATE INDEX IF NOT EXISTS ix_rule_strategy_risk_tenant_strategy ON rule_strategy_risk_states (tenant_id, strategy_id)",
        "CREATE INDEX IF NOT EXISTS ix_rule_strategy_monitor_due ON rule_strategy_monitor_symbols (state, next_check_at)",
        "CREATE INDEX IF NOT EXISTS ix_rule_strategy_events_tenant_strategy ON rule_strategy_events (tenant_id, strategy_id, created_at)",
    )
    for statement in statements:
        session.execute(text(statement))


def _extend_execution_intent_state(session: Session, dialect: str) -> None:
    additions = {
        "accepted_quote": "VARCHAR(32)",
        "accepted_quantity": "VARCHAR(32)",
        "decision_price": "VARCHAR(32)",
        "execution_target": "VARCHAR(32) NOT NULL DEFAULT 'paper'",
        "leg_kind": "VARCHAR(16) NOT NULL DEFAULT 'entry'",
        "lifecycle_state": "VARCHAR(32) NOT NULL DEFAULT 'pending'",
    }
    if dialect == "postgresql":
        for name, definition in additions.items():
            session.execute(
                text(
                    "ALTER TABLE rule_strategy_execution_intents "
                    f"ADD COLUMN IF NOT EXISTS {name} {definition}"
                )
            )
        return
    columns = {
        row[1]
        for row in session.execute(
            text("PRAGMA table_info(rule_strategy_execution_intents)")
        ).fetchall()
    }
    for name, definition in additions.items():
        if name not in columns:
            session.execute(
                text(
                    "ALTER TABLE rule_strategy_execution_intents "
                    f"ADD COLUMN {name} {definition}"
                )
            )


def _strategy_monitor_symbol_backfill_statement():
    return text(
        "INSERT INTO rule_strategy_monitor_symbols "
        "(tenant_id, strategy_id, symbol, state, protected_held, consecutive_low_volume_days) "
        "SELECT :tenant_id, :strategy_id, :symbol, 'candidate', :protected_held, 0 "
        "WHERE NOT EXISTS (SELECT 1 FROM rule_strategy_monitor_symbols "
        "WHERE tenant_id = :tenant_id AND strategy_id = :strategy_id AND symbol = :symbol)"
    ).bindparams(
        bindparam("tenant_id", type_=String(36)),
        bindparam("strategy_id", type_=String(100)),
        bindparam("symbol", type_=String(32)),
        bindparam("protected_held", type_=Boolean()),
    )


def _backfill_strategy_product_state(session: Session) -> None:
    monitor_symbol_statement = _strategy_monitor_symbol_backfill_statement()
    for strategy in session.query(RuleStrategy).all():
        config = dict(strategy.config or {})
        capital = float(config.get("initial_capital_quote", 10_000.0))
        execution = dict(config.get("execution") or {})
        environment = execution.get("environment", "paper")
        credential_id = execution.get("sandbox_connection_id")
        scope = "paper_virtual" if environment == "paper" else "dedicated_credential"
        existing_account = session.execute(
            text(
                "SELECT id FROM rule_strategy_accounts "
                "WHERE tenant_id = :tenant_id AND strategy_id = :strategy_id"
            ),
            {"tenant_id": strategy.tenant_id, "strategy_id": strategy.strategy_id},
        ).first()
        if existing_account is None:
            session.execute(
                text(
                    "INSERT INTO rule_strategy_accounts "
                    "(tenant_id, strategy_id, scope, credential_id, allocation_quote, quote_balance, positions, "
                    "realized_pnl_quote, unrealized_pnl_quote, equity_quote, version, active) "
                    "VALUES (:tenant_id, :strategy_id, :scope, :credential_id, :capital, :capital, :positions, "
                    "0, 0, :capital, 1, :active)"
                ),
                {
                    "tenant_id": strategy.tenant_id,
                    "strategy_id": strategy.strategy_id,
                    "scope": scope,
                    "credential_id": credential_id,
                    "capital": capital,
                    "positions": "{}",
                    "active": True,
                }
            )
        account_id = session.execute(
            text(
                "SELECT id FROM rule_strategy_accounts "
                "WHERE tenant_id = :tenant_id AND strategy_id = :strategy_id"
            ),
            {"tenant_id": strategy.tenant_id, "strategy_id": strategy.strategy_id},
        ).scalar_one()
        state = "normal" if environment == "paper" else "only_reduce"
        reason_code = None if environment == "paper" else "shared_exchange_account_requires_dedicated_scope"
        session.execute(
            text(
                "INSERT INTO rule_strategy_risk_states "
                "(account_id, tenant_id, strategy_id, state, daily_equity_baseline, high_water_equity, "
                "current_drawdown_pct, reason_code, reason_detail, version) "
                "SELECT :account_id, :tenant_id, :strategy_id, :state, :capital, :capital, "
                "0, :reason_code, :reason_detail, 1 "
                "WHERE NOT EXISTS (SELECT 1 FROM rule_strategy_risk_states WHERE account_id = :account_id)"
            ),
            {
                "account_id": account_id,
                "tenant_id": strategy.tenant_id,
                "strategy_id": strategy.strategy_id,
                "state": state,
                "capital": capital,
                "reason_code": reason_code,
                "reason_detail": "共享交易所账户未证明隔离，已仅允许减仓或平仓。" if reason_code else None,
            },
        )
        for symbol in config.get("symbols") or []:
            session.execute(
                monitor_symbol_statement,
                {
                    "tenant_id": strategy.tenant_id,
                    "strategy_id": strategy.strategy_id,
                    "symbol": str(symbol).upper().replace("/", "-"),
                    "protected_held": False,
                },
            )
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
            credential_id VARCHAR(36) REFERENCES tenant_credentials(id) ON DELETE RESTRICT,
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

RULE_STRATEGY_VALIDATION_MIGRATION_VERSION = "20260805_rule_strategy_validation_v1"
RULE_STRATEGY_VALIDATION_MIGRATION_LOCK_KEY = 7720250722


def migrate_rule_strategy_validation(session: Session) -> bool:
    """Create reproducible validation tables before any validation worker starts."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "rule strategy validation migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": RULE_STRATEGY_VALIDATION_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": RULE_STRATEGY_VALIDATION_MIGRATION_VERSION},
    ).first():
        return False
    from valuecell.server.db.models.rule_strategy_validation import (
        RuleStrategyValidationDataset,
        RuleStrategyValidationFill,
        RuleStrategyValidationPoint,
        RuleStrategyValidationRun,
    )
    Base.metadata.create_all(
        bind=session.bind,
        tables=[
            RuleStrategyValidationRun.__table__,
            RuleStrategyValidationDataset.__table__,
            RuleStrategyValidationPoint.__table__,
            RuleStrategyValidationFill.__table__,
        ],
    )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": RULE_STRATEGY_VALIDATION_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=RULE_STRATEGY_VALIDATION_MIGRATION_VERSION,
    )
    return True


STRATEGY_MONITOR_METADATA_MIGRATION_VERSION = "20260806_strategy_monitor_metadata_v1"
STRATEGY_MONITOR_METADATA_MIGRATION_LOCK_KEY = 7720250722


def migrate_strategy_monitor_metadata(session: Session) -> bool:
    """Install durable provider facts required by monitor admission."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "strategy monitor metadata migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": STRATEGY_MONITOR_METADATA_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": STRATEGY_MONITOR_METADATA_MIGRATION_VERSION},
    ).first():
        return False
    additions = {
        "metadata_provider": "VARCHAR(32)",
        "listing_first_tradable_at": (
            "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "DATETIME"
        ),
        "listing_age_days": "INTEGER",
        "average_quote_volume_30d": "FLOAT",
        "price_quote": "FLOAT",
        "price_observed_at": (
            "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "DATETIME"
        ),
    }
    if dialect == "postgresql":
        for name, definition in additions.items():
            session.execute(
                text(
                    "ALTER TABLE rule_strategy_monitor_symbols "
                    f"ADD COLUMN IF NOT EXISTS {name} {definition}"
                )
            )
    else:
        columns = {
            row[1]
            for row in session.execute(
                text("PRAGMA table_info(rule_strategy_monitor_symbols)")
            ).fetchall()
        }
        for name, definition in additions.items():
            if name not in columns:
                session.execute(
                    text(
                        "ALTER TABLE rule_strategy_monitor_symbols "
                        f"ADD COLUMN {name} {definition}"
                    )
                )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": STRATEGY_MONITOR_METADATA_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=STRATEGY_MONITOR_METADATA_MIGRATION_VERSION,
    )
    return True


DEMO_DAILY_LIMIT_MIGRATION_VERSION = "20260812_demo_daily_limit_v1"
DEMO_DAILY_LIMIT_MIGRATION_LOCK_KEY = 7720250723
DEMO_DAILY_LIMIT_USDT = 1_000_000.0


def migrate_demo_daily_execution_limit(session: Session) -> bool:
    """Raise every persisted OKX Demo strategy's daily safety throughput cap."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "Demo daily execution limit migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": DEMO_DAILY_LIMIT_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": DEMO_DAILY_LIMIT_MIGRATION_VERSION},
    ).first():
        return False
    for strategy in session.query(RuleStrategy).all():
        config = dict(strategy.config or {})
        execution = dict(config.get("execution") or {})
        if execution.get("environment") != "okx_demo":
            continue
        current_daily_limit = execution.get("max_daily_quote_amount")
        if (
            not isinstance(current_daily_limit, (int, float))
            or current_daily_limit < DEMO_DAILY_LIMIT_USDT
        ):
            execution["max_daily_quote_amount"] = DEMO_DAILY_LIMIT_USDT
        config["execution"] = execution
        strategy.config = config
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": DEMO_DAILY_LIMIT_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied Demo daily execution limit migration {version}",
        version=DEMO_DAILY_LIMIT_MIGRATION_VERSION,
    )
    return True


STRATEGY_DEMO_SNAPSHOT_MIGRATION_VERSION = "20260813_strategy_demo_account_snapshots_v1"
STRATEGY_DEMO_SNAPSHOT_MIGRATION_LOCK_KEY = 7720250724

STRATEGY_DEMO_SYNC_STATE_MIGRATION_VERSION = "20260819_strategy_demo_account_sync_state_v1"
STRATEGY_DEMO_SYNC_STATE_MIGRATION_LOCK_KEY = 7720250727


def migrate_strategy_demo_account_sync_state(session: Session) -> bool:
    """Create the current state table for background Demo synchronization."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "strategy Demo sync migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": STRATEGY_DEMO_SYNC_STATE_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": STRATEGY_DEMO_SYNC_STATE_MIGRATION_VERSION},
    ).first():
        return False
    from valuecell.server.db.models.rule_strategy import RuleStrategyDemoAccountSyncState

    Base.metadata.create_all(
        bind=session.bind,
        tables=[RuleStrategyDemoAccountSyncState.__table__],
    )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": STRATEGY_DEMO_SYNC_STATE_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=STRATEGY_DEMO_SYNC_STATE_MIGRATION_VERSION,
    )
    return True
STRATEGY_OFFICIAL_TEST_BASELINE_MIGRATION_VERSION = "20260817_strategy_official_test_baseline_v1"
STRATEGY_OFFICIAL_TEST_BASELINE_MIGRATION_LOCK_KEY = 7720250725


def migrate_strategy_demo_account_snapshots(session: Session) -> bool:
    """Create the durable exchange-wallet history table before Demo reads run."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "strategy Demo snapshot migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": STRATEGY_DEMO_SNAPSHOT_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": STRATEGY_DEMO_SNAPSHOT_MIGRATION_VERSION},
    ).first():
        return False
    from valuecell.server.db.models.rule_strategy import RuleStrategyDemoAccountSnapshot

    Base.metadata.create_all(
        bind=session.bind,
        tables=[RuleStrategyDemoAccountSnapshot.__table__],
    )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": STRATEGY_DEMO_SNAPSHOT_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=STRATEGY_DEMO_SNAPSHOT_MIGRATION_VERSION,
    )


    return True


MANUAL_CLOSE_MIGRATION_VERSION = "20260814_rule_strategy_manual_close_v1"
MANUAL_CLOSE_MIGRATION_LOCK_KEY = 7720250726


def migrate_rule_strategy_manual_close(session: Session) -> bool:
    """Create durable manual-close command storage before enabling the route."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "manual close migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": MANUAL_CLOSE_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": MANUAL_CLOSE_MIGRATION_VERSION},
    ).first():
        return False
    from valuecell.server.db.models.rule_strategy_manual_close import RuleStrategyManualCloseCommand

    Base.metadata.create_all(
        bind=session.bind,
        tables=[RuleStrategyManualCloseCommand.__table__],
    )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": MANUAL_CLOSE_MIGRATION_VERSION},
    )
    session.commit()
    logger.info("Applied schema migration {version}", version=MANUAL_CLOSE_MIGRATION_VERSION)
    return True


def migrate_strategy_official_test_baselines(session: Session) -> bool:
    """Create the immutable official-test boundary table before Demo scheduling."""
    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "strategy official-test baseline migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": STRATEGY_OFFICIAL_TEST_BASELINE_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": STRATEGY_OFFICIAL_TEST_BASELINE_MIGRATION_VERSION},
    ).first():
        return False
    from valuecell.server.db.models.rule_strategy import RuleStrategyOfficialTestBaseline

    Base.metadata.create_all(
        bind=session.bind,
        tables=[RuleStrategyOfficialTestBaseline.__table__],
    )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": STRATEGY_OFFICIAL_TEST_BASELINE_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=STRATEGY_OFFICIAL_TEST_BASELINE_MIGRATION_VERSION,
    )
    return True


LEADER_SPOT_V19_STORAGE_MIGRATION_VERSION = "20260824_leader_spot_v19_storage_v1"
LEADER_SPOT_V19_STORAGE_MIGRATION_LOCK_KEY = 7720250728


def migrate_leader_spot_v19_storage(session: Session) -> bool:
    """Install isolated V19 tables before any V19 route or scheduler is enabled."""

    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "leader spot V19 storage migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": LEADER_SPOT_V19_STORAGE_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": LEADER_SPOT_V19_STORAGE_MIGRATION_VERSION},
    ).first():
        return False

    from valuecell.server.db.models.leader_spot_v19 import (
        LeaderSpotV19Account,
        LeaderSpotV19CandidateSnapshot,
        LeaderSpotV19Event,
        LeaderSpotV19ExecutionBatch,
        LeaderSpotV19ExecutionIntent,
        LeaderSpotV19ExecutionLease,
        LeaderSpotV19Fill,
        LeaderSpotV19MarketSnapshot,
        LeaderSpotV19OrderAttempt,
        LeaderSpotV19Position,
        LeaderSpotV19RiskState,
        LeaderSpotV19Strategy,
    )

    Base.metadata.create_all(
        bind=session.bind,
        tables=[
            LeaderSpotV19Strategy.__table__,
            LeaderSpotV19ExecutionBatch.__table__,
            LeaderSpotV19Account.__table__,
            LeaderSpotV19RiskState.__table__,
            LeaderSpotV19CandidateSnapshot.__table__,
            LeaderSpotV19MarketSnapshot.__table__,
            LeaderSpotV19Position.__table__,
            LeaderSpotV19ExecutionIntent.__table__,
            LeaderSpotV19OrderAttempt.__table__,
            LeaderSpotV19Fill.__table__,
            LeaderSpotV19Event.__table__,
            LeaderSpotV19ExecutionLease.__table__,
        ],
    )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": LEADER_SPOT_V19_STORAGE_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=LEADER_SPOT_V19_STORAGE_MIGRATION_VERSION,
    )
    return True
LEADER_SPOT_V19_QUALITY_MIGRATION_VERSION = "20260824_leader_spot_v19_quality_v1"
LEADER_SPOT_V19_QUALITY_MIGRATION_LOCK_KEY = 7720250729


def migrate_leader_spot_v19_quality(session: Session) -> bool:
    """Install durable V19 quality reports without altering prior V19 tables."""

    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "leader spot V19 quality migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": LEADER_SPOT_V19_QUALITY_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": LEADER_SPOT_V19_QUALITY_MIGRATION_VERSION},
    ).first():
        return False
    from valuecell.server.db.models.leader_spot_v19 import LeaderSpotV19DataQualityReport

    Base.metadata.create_all(
        bind=session.bind,
        tables=[LeaderSpotV19DataQualityReport.__table__],
    )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": LEADER_SPOT_V19_QUALITY_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=LEADER_SPOT_V19_QUALITY_MIGRATION_VERSION,
    )
    return True


LEADER_SPOT_V19_MARKET_STATE_MIGRATION_VERSION = "20260824_leader_spot_v19_market_state_v1"
LEADER_SPOT_V19_MARKET_STATE_MIGRATION_LOCK_KEY = 7720250730


def migrate_leader_spot_v19_market_state(session: Session) -> bool:
    """Install append-only V19 market decisions before selection can run."""

    dialect = session.bind.dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(
            "leader spot V19 market-state migration supports PostgreSQL and SQLite, "
            f"got {dialect!r}"
        )
    if dialect == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": LEADER_SPOT_V19_MARKET_STATE_MIGRATION_LOCK_KEY},
        )
    session.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP WITH TIME ZONE "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    if session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": LEADER_SPOT_V19_MARKET_STATE_MIGRATION_VERSION},
    ).first():
        return False
    from valuecell.server.db.models.leader_spot_v19 import LeaderSpotV19MarketStateDecision

    Base.metadata.create_all(
        bind=session.bind,
        tables=[LeaderSpotV19MarketStateDecision.__table__],
    )
    session.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": LEADER_SPOT_V19_MARKET_STATE_MIGRATION_VERSION},
    )
    session.commit()
    logger.info(
        "Applied schema migration {version}",
        version=LEADER_SPOT_V19_MARKET_STATE_MIGRATION_VERSION,
    )
    return True
