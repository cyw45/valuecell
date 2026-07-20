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
        "credential_id VARCHAR(36) NOT NULL REFERENCES tenant_credentials(id) ON DELETE RESTRICT",
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


def test_app_lifespan_wires_required_migration_before_best_effort_migrations():
    from pathlib import Path

    source = Path(__file__).parents[1] / "api" / "app.py"
    app_source = source.read_text(encoding="utf-8")
    required_call = app_source.index("_run_required_execution_attribution_migration()")
    best_effort_block = app_source.index("migrate_fixed_order_amounts(session)")

    assert required_call < best_effort_block
    assert required_call < app_source.index("await _scheduler.start()")
    assert "migrate_rule_strategy_execution_attribution(session)" in app_source
