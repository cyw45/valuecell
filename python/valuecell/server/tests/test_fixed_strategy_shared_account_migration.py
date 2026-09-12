from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.db import migrations
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.multi_strategy import StrategySharedAccount  # noqa: F401
from valuecell.server.db.models.rule_strategy import (  # noqa: F401
    RuleStrategy,
    RuleStrategyAccount,
    RuleStrategyRiskState,
)
from valuecell.server.db.models.shared_demo_execution import (  # noqa: F401
    SharedDemoStrategyAllocationCap,
)
from valuecell.server.db.models.tenant import Tenant  # noqa: F401
from valuecell.server.db.models.tenant_credential import TenantCredential  # noqa: F401

_FIXED_KINDS = ("dual_ma_trend", "pair_rotation", "leader_breakout")


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.add(
        StrategySharedAccount(
            id="account-a",
            tenant_id="tenant-a",
            credential_id="credential-a",
            environment="okx_demo",
            sync_status="healthy",
            active=True,
        )
    )
    for index, kind in enumerate(_FIXED_KINDS, start=1):
        strategy_id = f"rule-fixed-{index}"
        session.add(
            RuleStrategy(
                strategy_id=strategy_id,
                tenant_id="tenant-a",
                name=f"Fixed {kind}",
                strategy_kind=kind,
                status="running",
                paper_mode=True,
                execution_generation=3,
                current_batch_id=f"batch-{index}",
                config={
                    "initial_capital_quote": 600.0,
                    "symbols": ["BTC-USDT"],
                    "execution": {"environment": "paper", "sandbox_connection_id": None},
                },
            )
        )
        session.add(
            RuleStrategyAccount(
                tenant_id="tenant-a",
                strategy_id=strategy_id,
                scope="paper_virtual",
                allocation_quote=600.0,
                quote_balance=600.0,
                equity_quote=600.0,
            )
        )
    session.add(
        RuleStrategy(
            strategy_id="rule-already-shared",
            tenant_id="tenant-a",
            name="Already shared",
            strategy_kind="dual_ma_trend",
            status="running",
            paper_mode=False,
            current_batch_id="batch-shared",
            config={
                "initial_capital_quote": 10_000.0,
                "execution": {
                    "environment": "okx_demo",
                    "sandbox_connection_id": "credential-a",
                },
            },
        )
    )
    session.add(
        RuleStrategy(
            strategy_id="rule-configurable",
            tenant_id="tenant-a",
            name="Configurable",
            strategy_kind="configurable_rule",
            status="running",
            paper_mode=True,
            config={
                "initial_capital_quote": 500.0,
                "execution": {"environment": "paper", "sandbox_connection_id": None},
            },
        )
    )
    session.commit()
    return session


def test_fixed_strategies_are_rebound_to_the_shared_demo_account_once() -> None:
    session = _session()

    assert migrations.migrate_fixed_strategies_to_shared_account(session) is True
    assert migrations.migrate_fixed_strategies_to_shared_account(session) is False

    for index in (1, 2, 3):
        strategy = (
            session.query(RuleStrategy)
            .filter_by(strategy_id=f"rule-fixed-{index}")
            .one()
        )
        assert strategy.config["execution"] == {
            "environment": "okx_demo",
            "sandbox_connection_id": "credential-a",
        }
        assert strategy.paper_mode is False
        assert strategy.status == "stopped"
        assert strategy.current_batch_id is None
        account = (
            session.query(RuleStrategyAccount)
            .filter_by(strategy_id=strategy.strategy_id)
            .one()
        )
        assert account.scope == "shared_exchange_account"
        assert account.credential_id == "credential-a"

    caps = session.query(SharedDemoStrategyAllocationCap).all()
    assert {cap.strategy_id for cap in caps} == {
        "rule-fixed-1",
        "rule-fixed-2",
        "rule-fixed-3",
    }
    assert {float(cap.max_reserved_quote) for cap in caps} == {600.0}

    already_shared = (
        session.query(RuleStrategy).filter_by(strategy_id="rule-already-shared").one()
    )
    assert already_shared.status == "running"
    assert already_shared.current_batch_id == "batch-shared"
    configurable = (
        session.query(RuleStrategy).filter_by(strategy_id="rule-configurable").one()
    )
    assert configurable.status == "running"
    assert configurable.config["execution"]["environment"] == "paper"


def test_fixed_strategy_without_a_shared_account_keeps_its_scope() -> None:
    session = _session()
    session.add(Tenant(id="tenant-b", name="Tenant B"))
    session.add(
        RuleStrategy(
            strategy_id="rule-orphan",
            tenant_id="tenant-b",
            name="Orphan",
            strategy_kind="pair_rotation",
            status="running",
            paper_mode=True,
            config={
                "initial_capital_quote": 600.0,
                "execution": {"environment": "paper", "sandbox_connection_id": None},
            },
        )
    )
    session.commit()

    assert migrations.migrate_fixed_strategies_to_shared_account(session) is True

    orphan = session.query(RuleStrategy).filter_by(strategy_id="rule-orphan").one()
    assert orphan.config["execution"]["environment"] == "paper"
    assert orphan.status == "running"
