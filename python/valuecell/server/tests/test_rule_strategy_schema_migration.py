from types import SimpleNamespace

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from valuecell.server.db import migrations
from valuecell.server.db.models.base import Base

# Register the tables under test with SQLAlchemy metadata.
from valuecell.server.db.models.rule_strategy import (  # noqa: F401
    RuleStrategy,
    RuleStrategyEvaluationJournal,
    RuleStrategyExecutionIntent,
)
from valuecell.server.db.models.sandbox_exchange_order import SandboxExchangeOrder  # noqa: F401
from valuecell.server.db.models.tenant import Tenant  # noqa: F401
from valuecell.server.db.models.tenant_credential import TenantCredential  # noqa: F401


class FakeResult:
    def __init__(self, rows=()):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    def __init__(self):
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
        self.statements = []
        self.commits = 0

    def execute(self, statement, params=None):
        self.statements.append(str(statement))
        return FakeResult()

    def commit(self):
        self.commits += 1


class AppliedValidationMigrationSession(FakeSession):
    def execute(self, statement, params=None):
        self.statements.append(str(statement))
        if "SELECT version FROM schema_migrations" in str(statement):
            return FakeResult([("applied",)])
        return FakeResult()


def test_execution_attribution_migration_uses_idempotent_concurrent_postgres_ddl():
    session = FakeSession()

    changed = migrations.migrate_rule_strategy_execution_attribution(session)

    assert changed is True
    assert session.commits == 1
    statements = "\n".join(session.statements)
    assert "CREATE TABLE IF NOT EXISTS schema_migrations" in statements
    assert "SELECT pg_advisory_xact_lock" in statements
    assert "execution_generation INTEGER NOT NULL DEFAULT 1" in statements
    assert "ALTER TABLE sandbox_exchange_orders ADD COLUMN IF NOT EXISTS execution_intent_id" in statements
    assert "CREATE TABLE IF NOT EXISTS rule_strategy_execution_intents" in statements
    for column in (
        "tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id)",
        "credential_id VARCHAR(36) REFERENCES tenant_credentials(id) ON DELETE RESTRICT",
        "idempotency_key VARCHAR(128) NOT NULL",
        "symbol VARCHAR(32) NOT NULL",
        "side VARCHAR(8) NOT NULL",
        "order_type VARCHAR(8) NOT NULL",
        "requested_quote VARCHAR(32) NOT NULL",
        "requested_quantity VARCHAR(32)",
        "attempt_count INTEGER NOT NULL DEFAULT 0",
        "error_code VARCHAR(64)",
        "error_message",
        "submitted_at",
        "terminal_at",
        "updated_at",
    ):
        assert column in statements
    assert "UNIQUE (tenant_id, idempotency_key)" in statements
    assert "FOREIGN KEY (execution_intent_id) REFERENCES rule_strategy_execution_intents(id) ON DELETE RESTRICT" in statements
    assert "execution_intent_id IS NULL" in statements
    assert "CREATE OR REPLACE FUNCTION prevent_sandbox_order_attribution_mutation" in statements


def test_execution_attribution_migration_is_idempotent_on_sqlite():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    assert migrations.migrate_rule_strategy_execution_attribution(session) is True
    assert migrations.migrate_rule_strategy_execution_attribution(session) is False

    inspector = inspect(engine)
    order_columns = {
        column["name"] for column in inspector.get_columns("sandbox_exchange_orders")
    }
    assert {
        "strategy_id", "evaluation_id", "execution_generation", "execution_source",
        "execution_intent_id",
    } <= order_columns
    intent_columns = {
        column["name"]
        for column in inspector.get_columns("rule_strategy_execution_intents")
    }
    assert {
        "tenant_id", "credential_id", "idempotency_key", "symbol", "side",
        "order_type", "requested_quote", "requested_quantity", "status",
        "attempt_count", "error_code", "error_message", "submitted_at",
        "terminal_at", "updated_at", "request_payload",
    } <= intent_columns



def test_strategy_product_monitor_backfill_binds_postgresql_parameter_types():
    statement = migrations._strategy_monitor_symbol_backfill_statement()

    assert statement._bindparams["tenant_id"].type.length == 36
    assert statement._bindparams["strategy_id"].type.length == 100
    assert statement._bindparams["symbol"].type.length == 32
    assert statement._bindparams["protected_held"].type.python_type is bool


def test_strategy_product_state_migration_backfills_account_risk_and_monitor_rows():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Tenant(id="tenant-state", name="State tenant"))
    session.add(
        RuleStrategy(
            strategy_id="rule-state",
            tenant_id="tenant-state",
            name="State strategy",
            status="stopped",
            paper_mode=True,
            config={
                "initial_capital_quote": 1_000.0,
                "symbols": ["BTC-USDT"],
                "execution": {"environment": "paper"},
            },
        )
    )
    session.commit()

    assert migrations.migrate_strategy_product_state(session) is True
    assert migrations.migrate_strategy_product_state(session) is False

    account = session.execute(
        __import__("sqlalchemy").text(
            "SELECT scope, allocation_quote, quote_balance FROM rule_strategy_accounts"
        )
    ).one()
    assert account == ("paper_virtual", 1_000.0, 1_000.0)
    assert session.execute(
        __import__("sqlalchemy").text("SELECT state FROM rule_strategy_risk_states")
    ).scalar_one() == "normal"
    assert session.execute(
        __import__("sqlalchemy").text("SELECT state FROM rule_strategy_monitor_symbols")
    ).scalar_one() == "candidate"

def test_rule_strategy_validation_migration_creates_tables_idempotently():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    assert migrations.migrate_rule_strategy_validation(session) is True
    assert migrations.migrate_rule_strategy_validation(session) is False

    tables = set(inspect(engine).get_table_names())
    assert {
        "rule_strategy_validation_runs",
        "rule_strategy_validation_datasets",
        "rule_strategy_validation_points",
        "rule_strategy_validation_fills",
    } <= tables


def test_rule_strategy_validation_migration_locks_before_marker_read():
    session = AppliedValidationMigrationSession()

    assert migrations.migrate_rule_strategy_validation(session) is False
    assert "SELECT pg_advisory_xact_lock" in session.statements[0]


def test_execution_attribution_models_define_intent_contract_and_attribution():
    order = SandboxExchangeOrder.__table__
    intent = RuleStrategyExecutionIntent.__table__

    assert {foreign_key.target_fullname for foreign_key in order.foreign_keys} >= {
        "rule_strategies.strategy_id",
        "rule_strategy_evaluation_journal.evaluation_id",
        "rule_strategy_execution_intents.id",
    }
    assert "ix_sandbox_exchange_orders_strategy_evaluation" in {
        index.name for index in order.indexes
    }
    assert "uq_rule_strategy_execution_intent" in {
        constraint.name for constraint in intent.constraints
    }
    assert "uq_rule_strategy_execution_intent_tenant_idempotency" in {
        constraint.name for constraint in intent.constraints
    }
    assert {column.name for column in intent.columns} >= {
        "tenant_id", "credential_id", "idempotency_key", "symbol", "side",
        "order_type", "requested_quote", "requested_quantity", "status",
        "attempt_count", "error_code", "error_message", "submitted_at",
        "terminal_at", "updated_at", "request_payload",
    }
    assert "ix_rule_strategy_execution_intents_strategy_status" in {
        index.name for index in intent.indexes
    }
    assert "ix_rule_strategy_execution_intents_lifecycle" in {
        index.name for index in intent.indexes
    }
    assert "ck_sandbox_exchange_orders_attribution_complete" in {
        constraint.name for constraint in order.constraints
    }


def test_rule_strategy_journal_read_index_is_idempotent():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    migrations.ensure_rule_strategy_journal_read_index(session)
    migrations.ensure_rule_strategy_journal_read_index(session)

    indexes = {
        index["name"]
        for index in inspect(engine).get_indexes("rule_strategy_evaluation_journal")
    }
    assert migrations.RULE_STRATEGY_JOURNAL_INDEX_NAME in indexes


def test_app_lifespan_wires_required_migration_before_best_effort_migrations():
    from pathlib import Path

    source = Path(__file__).parents[1] / "api" / "app.py"
    app_source = source.read_text(encoding="utf-8")
    required_call = app_source.index("_run_required_execution_attribution_migration()")
    best_effort_block = app_source.index("migrate_fixed_order_amounts(session)")

    assert required_call < best_effort_block
    assert required_call < app_source.index("await _scheduler.start()")
    assert "migrate_rule_strategy_execution_attribution(session)" in app_source
    assert "_scheduler_review_strategy_monitors" in app_source
    assert "review_running_strategy_monitors(repository, service)" in app_source



def test_monitor_metadata_migration_installs_provider_fact_columns():
    session = FakeSession()

    assert migrations.migrate_strategy_monitor_metadata(session) is True

    statements = "\n".join(session.statements)
    assert "SELECT pg_advisory_xact_lock" in statements
    for column in (
        "metadata_provider VARCHAR(32)",
        "listing_first_tradable_at TIMESTAMP WITH TIME ZONE",
        "listing_age_days INTEGER",
        "average_quote_volume_30d FLOAT",
        "price_quote FLOAT",
        "price_observed_at TIMESTAMP WITH TIME ZONE",
    ):
        assert column in statements


def test_demo_daily_execution_limit_migration_updates_only_demo_strategies():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Tenant(id="tenant-demo-limit", name="Demo limit tenant"))
    session.add_all(
        [
            RuleStrategy(
                strategy_id="rule-demo-limit",
                tenant_id="tenant-demo-limit",
                name="Demo limit",
                status="stopped",
                paper_mode=False,
                config={
                    "execution": {
                        "environment": "okx_demo",
                        "max_daily_quote_amount": 500,
                    }
                },
            ),
            RuleStrategy(
                strategy_id="rule-paper-limit",
                tenant_id="tenant-demo-limit",
                name="Paper limit",
                status="stopped",
                paper_mode=True,
                config={"execution": {"environment": "paper", "max_daily_quote_amount": 500}},
            ),
        ]
    )
    session.commit()

    assert migrations.migrate_demo_daily_execution_limit(session) is True
    assert migrations.migrate_demo_daily_execution_limit(session) is False

    limits = dict(
        session.execute(
            __import__("sqlalchemy").text(
                "SELECT strategy_id, json_extract(config, '$.execution.max_daily_quote_amount') "
                "FROM rule_strategies"
            )
        ).all()
    )
    assert limits["rule-demo-limit"] == migrations.DEMO_DAILY_LIMIT_USDT
    assert limits["rule-paper-limit"] == 500


def test_demo_snapshot_migration_is_registered_before_scheduler_startup():
    from pathlib import Path

    source = Path(__file__).parents[1] / "api" / "app.py"
    app_source = source.read_text(encoding="utf-8")

    assert "migrate_strategy_demo_account_snapshots(session)" in app_source
    assert app_source.index("migrate_strategy_demo_account_snapshots(session)") < app_source.index(
        "await _scheduler.start()"
    )


def test_manual_close_migration_is_registered_before_scheduler_startup():
    from pathlib import Path

    source = Path(__file__).parents[1] / "api" / "app.py"
    app_source = source.read_text(encoding="utf-8")

    assert "migrate_rule_strategy_manual_close(session)" in app_source
    assert app_source.index("migrate_rule_strategy_manual_close(session)") < app_source.index(
        "await _scheduler.start()"
    )