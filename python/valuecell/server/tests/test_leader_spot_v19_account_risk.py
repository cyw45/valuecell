from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_account_risk import (
    LeaderSpotV19AccountRiskInput,
)
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19Account,
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19ExecutionIntent,
    LeaderSpotV19Position,
    LeaderSpotV19RiskState,
    LeaderSpotV19Strategy,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.leader_spot_v19_account_risk_service import (
    LeaderSpotV19AccountRiskEngine,
)


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _fixtures():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    tenant = Tenant(id="tenant-i", name="I")
    session.add(tenant)
    session.commit()
    strategy = LeaderSpotV19Strategy(
        strategy_id="leader-i", tenant_id=tenant.id, name="I", status="running",
        environment="paper", config={"module_id": "leader_spot_v19_0", "schema_version": 19},
    )
    session.add(strategy)
    session.commit()
    batch = LeaderSpotV19ExecutionBatch(
        batch_id="batch-i", tenant_id=tenant.id, strategy_id=strategy.strategy_id,
        strategy_name_snapshot=strategy.name, execution_generation=1,
        config_snapshot=strategy.config,
    )
    session.add(batch)
    session.commit()
    account = LeaderSpotV19Account(
        account_id="account-i", tenant_id=tenant.id, strategy_id=strategy.strategy_id,
        batch_id=batch.batch_id, scope="paper", initial_equity_quote=600,
        quote_balance=600, equity_quote=600,
    )
    session.add(account)
    session.commit()
    risk = LeaderSpotV19RiskState(
        risk_state_id="risk-i", account_id=account.account_id, tenant_id=tenant.id,
        strategy_id=strategy.strategy_id, batch_id=batch.batch_id, daily_loss_limit_quote=48,
        daily_loss_reset_at=NOW + timedelta(hours=12), prior_close_equity_quote=600,
    )
    session.add(risk)
    session.commit()
    return session, risk


def _input(**changes):
    values = {
        "account_id": "account-i", "daily_realized_pnl_quote": 0,
        "daily_loss_limit_quote": 48, "daily_loss_reset_at": NOW + timedelta(hours=12),
        "prior_close_equity_quote": 600, "equity_quote": 600, "observed_at": NOW,
    }
    values.update(changes)
    return LeaderSpotV19AccountRiskInput(**values)


def _entry_intent(risk, intent_id, status="pending"):
    return LeaderSpotV19ExecutionIntent(
        intent_id=intent_id, tenant_id=risk.tenant_id, strategy_id=risk.strategy_id,
        batch_id=risk.batch_id, execution_generation=1, idempotency_key=intent_id,
        symbol="BTC-USDT", side="buy", order_type="limit", leg_kind="entry",
        requested_quote="100", requested_quantity="1", requested_price="100",
        lifecycle_state=status, status=status,
    )


def test_risk_engine_resets_daily_loss_at_utc_boundary_and_halts_daily_losses():
    engine = LeaderSpotV19AccountRiskEngine()
    halted = engine.decide(LeaderSpotV19Config(), _input(daily_realized_pnl_quote=-48))
    assert halted.state == "daily_loss_halted"
    assert halted.can_open is False
    assert halted.cancel_pending_entries is True
    reset = engine.decide(
        LeaderSpotV19Config(),
        _input(daily_realized_pnl_quote=-80, daily_loss_reset_at=NOW - timedelta(seconds=1)),
    )
    assert reset.state == "normal"
    assert reset.daily_realized_pnl_quote == 0


def test_equity_halt_cancels_entries_and_stages_loss_ordered_market_exits():
    session, risk = _fixtures()
    session.add_all([
        _entry_intent(risk, "entry-pending"),
        LeaderSpotV19Position(
            position_id="position-worse", tenant_id=risk.tenant_id, strategy_id=risk.strategy_id,
            batch_id=risk.batch_id, symbol="BTC-USDT", entry_price="100", entry_quantity="1",
            entry_time=NOW, peak_price="100", moving_stop_price="92", loss_circuit_started_at=NOW,
            peak_profit_pct=0,
        ),
        LeaderSpotV19Position(
            position_id="position-better", tenant_id=risk.tenant_id, strategy_id=risk.strategy_id,
            batch_id=risk.batch_id, symbol="ETH-USDT", entry_price="100", entry_quantity="1",
            entry_time=NOW, peak_price="100", moving_stop_price="92", loss_circuit_started_at=NOW,
            peak_profit_pct=0.2,
        ),
    ])
    session.commit()
    decision, cancelled = LeaderSpotV19AccountRiskEngine().apply(
        session, config=LeaderSpotV19Config(), risk=risk,
        risk_input=_input(equity_quote=510), execution_generation=1,
    )
    assert decision.state == "equity_halted"
    assert decision.force_close_positions is True
    assert cancelled.cancelled_intent_ids == ["entry-pending"]
    exits = session.query(LeaderSpotV19ExecutionIntent).filter_by(side="sell").all()
    assert [item.symbol for item in exits] == ["BTC-USDT", "ETH-USDT"]
    assert all(item.order_type == "market" for item in exits)


def test_account_risk_halt_is_idempotent_and_never_cancels_unknown_submissions():
    session, risk = _fixtures()
    unknown = _entry_intent(risk, "entry-unknown", status="submission_unknown")
    session.add(unknown)
    session.commit()
    engine = LeaderSpotV19AccountRiskEngine()
    first, cancelled = engine.apply(
        session, config=LeaderSpotV19Config(), risk=risk,
        risk_input=_input(daily_realized_pnl_quote=-48), execution_generation=1,
    )
    second, repeat = engine.apply(
        session, config=LeaderSpotV19Config(), risk=risk,
        risk_input=_input(daily_realized_pnl_quote=-48), execution_generation=1,
    )
    assert first.state == second.state == "daily_loss_halted"
    assert cancelled.cancelled_intent_ids == []
    assert repeat.cancelled_intent_ids == []
    assert session.get(LeaderSpotV19ExecutionIntent, unknown.intent_id).status == "submission_unknown"
