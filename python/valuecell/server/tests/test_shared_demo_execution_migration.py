from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from valuecell.server.db import migrations
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.multi_strategy import (  # noqa: F401
    StrategyCapitalReservation,
    StrategySharedAccount,
)
from valuecell.server.db.models.rule_strategy import (  # noqa: F401
    RuleStrategy,
    RuleStrategyExecutionBatch,
    RuleStrategyExecutionIntent,
)
from valuecell.server.db.models.tenant import SaaSUser, Tenant  # noqa: F401
from valuecell.server.db.models.tenant_credential import TenantCredential  # noqa: F401


_SHARED_DEMO_TABLES = {
    "shared_demo_account_snapshots",
    "shared_demo_account_sync_states",
    "shared_demo_execution_reservations",
    "shared_demo_execution_intents",
    "shared_demo_venue_orders",
    "shared_demo_order_projections",
    "shared_demo_fills",
    "shared_demo_reservation_recovery_events",
    "shared_demo_strategy_allocation_caps",
}


def _column_names(inspector, table_name: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table_name)}


def _constraint_names(inspector, table_name: str) -> set[str]:
    return {
        constraint["name"]
        for constraint in (
            inspector.get_check_constraints(table_name)
            + inspector.get_unique_constraints(table_name)
        )
        if constraint["name"] is not None
    }


def test_shared_demo_execution_migration_is_idempotent_and_account_scoped() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    assert migrations.migrate_multi_strategy_account(session) is True
    assert migrations.migrate_shared_demo_execution_storage(session) is True
    assert migrations.migrate_shared_demo_execution_storage(session) is False

    inspector = inspect(engine)
    assert _SHARED_DEMO_TABLES <= set(inspector.get_table_names())

    assert {
        "account_id",
        "tenant_id",
        "credential_id",
        "environment",
        "wallet_equity_quote",
        "available_quote",
        "balances",
        "positions",
        "open_orders",
    } <= _column_names(inspector, "shared_demo_account_snapshots")
    assert {
        "account_id",
        "latest_snapshot_id",
        "sync_status",
        "reconciliation_status",
        "unresolved_submission_count",
    } <= _column_names(inspector, "shared_demo_account_sync_states")
    assert {
        "reservation_id",
        "account_id",
        "tenant_id",
        "credential_id",
        "strategy_id",
        "batch_id",
        "requested_quote",
        "reserved_quote",
    } <= _column_names(inspector, "shared_demo_execution_reservations")
    assert {"intent_id", "reservation_id", "account_id", "strategy_id", "batch_id"} <= _column_names(
        inspector, "shared_demo_execution_intents"
    )
    assert {"venue", "client_order_id", "venue_order_id", "requested_quantity"} <= _column_names(
        inspector, "shared_demo_venue_orders"
    )
    assert {"order_id", "status", "filled_quantity", "remaining_quantity"} <= _column_names(
        inspector, "shared_demo_order_projections"
    )
    assert {"venue", "venue_fill_id", "price", "quantity", "quote_amount"} <= _column_names(
        inspector, "shared_demo_fills"
    )
    assert {"reservation_id", "event_type", "outstanding_reserved_quote"} <= _column_names(
        inspector, "shared_demo_reservation_recovery_events"
    )
    assert {"account_id", "strategy_id", "max_reserved_quote", "max_occupied_quote"} <= _column_names(
        inspector, "shared_demo_strategy_allocation_caps"
    )

    numeric_columns = (
        ("shared_demo_account_snapshots", "wallet_equity_quote"),
        ("shared_demo_execution_reservations", "requested_quote"),
        ("shared_demo_venue_orders", "requested_quantity"),
        ("shared_demo_fills", "quantity"),
        ("shared_demo_order_projections", "filled_quantity"),
        ("shared_demo_strategy_allocation_caps", "max_reserved_quote"),
    )
    for table_name, column_name in numeric_columns:
        column = next(
            column
            for column in inspector.get_columns(table_name)
            if column["name"] == column_name
        )
        assert "NUMERIC" in str(column["type"]).upper()

    assert {
        "uq_shared_demo_intent_reservation",
        "uq_shared_demo_intent_client_order",
    } <= _constraint_names(inspector, "shared_demo_execution_intents")
    assert {
        "uq_shared_demo_order_venue_client",
        "uq_shared_demo_order_venue_order",
    } <= _constraint_names(inspector, "shared_demo_venue_orders")
    assert "uq_shared_demo_fill_venue_fill" in _constraint_names(
        inspector, "shared_demo_fills"
    )
    assert "ck_shared_demo_sync_state_reconciliation" in _constraint_names(
        inspector, "shared_demo_account_sync_states"
    )
    assert "ck_shared_demo_projection_status" in _constraint_names(
        inspector, "shared_demo_order_projections"
    )

    order_indexes = {index["name"] for index in inspector.get_indexes("shared_demo_venue_orders")}
    fill_indexes = {index["name"] for index in inspector.get_indexes("shared_demo_fills")}
    assert "ix_shared_demo_order_account_strategy_batch" in order_indexes
    assert "ix_shared_demo_fill_account_strategy_batch_time" in fill_indexes

    intent_foreign_tables = {
        foreign_key["referred_table"]
        for foreign_key in inspector.get_foreign_keys("shared_demo_execution_intents")
    }
    assert {
        "rule_strategy_execution_intents",
        "shared_demo_execution_reservations",
    } <= intent_foreign_tables


def test_shared_demo_models_register_and_startup_runs_storage_migration() -> None:
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

    models = (
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
    assert {model.__tablename__ for model in models} <= set(Base.metadata.tables)

    app_source = (
        __import__("pathlib").Path(__file__).parents[1] / "api" / "app.py"
    ).read_text(encoding="utf-8")
    assert "migrate_shared_demo_execution_storage(session)" in app_source
    assert app_source.index("migrate_multi_strategy_account(session)") < app_source.index(
        "migrate_shared_demo_execution_storage(session)"
    )
    assert "provision_quant_tables_with_shared_demo_constraints()" in app_source
    assert app_source.rindex(
        "provision_quant_tables_with_shared_demo_constraints()"
    ) < app_source.rindex("_run_required_rule_strategy_archiving_migration()")


def test_quant_table_provisioning_supports_fresh_database_and_is_idempotent() -> None:
    from valuecell.server.db.migrations import (
        provision_quant_tables_with_shared_demo_constraints,
    )

    engine = create_engine("sqlite://")
    session = sessionmaker(bind=engine)()

    provision_quant_tables_with_shared_demo_constraints(session)
    provision_quant_tables_with_shared_demo_constraints(session)

    inspector = inspect(engine)
    assert _SHARED_DEMO_TABLES <= set(inspector.get_table_names())
    assert "strategy_shared_accounts" in inspector.get_table_names()
