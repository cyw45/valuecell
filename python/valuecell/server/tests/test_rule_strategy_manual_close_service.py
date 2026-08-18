import asyncio

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.db.models.base import Base
from valuecell.server.services import rule_strategy_manual_close_service as manual_close


class FakeExchange:
    submits = 0

    def __init__(self, _db):
        pass

    async def balance(self, *_args):
        return {"source": "okx_demo", "total_usdt_value": 100.0, "balances": [], "checked_at": "2026-08-01T00:00:00+00:00"}

    async def positions(self, *_args, **_kwargs):
        return {"source": "okx_demo", "checked_at": "2026-08-01T00:00:00+00:00", "positions": [{"symbol": "BTC/USDT", "available_quantity": 1.0, "mark_price": 100.0}]}

    def list_orders(self, *_args):
        return [{"strategy_id": "strategy-a", "execution_source": "rule_strategy", "symbol": "BTC/USDT", "side": "buy", "status": "filled", "filled_quantity": "1", "filled_quote": "100", "average_fill_price": "100"}]

    async def submit_order(self, *_args, **_kwargs):
        type(self).submits += 1
        return {"id": "order-close", "status": "filled", "exchange_order_id": "exchange-close"}


def test_manual_close_is_idempotent_and_targets_verified_strategy_inventory(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(manual_close, "SandboxExchangeTradingService", FakeExchange)
    monkeypatch.setattr(manual_close, "record_audit_event", lambda *_args, **_kwargs: None)
    strategy = {
        "strategy_id": "strategy-a",
        "config": {"execution": {"environment": "okx_demo", "sandbox_connection_id": "conn-a"}},
    }

    first = asyncio.run(manual_close.execute_manual_close(
        session,
        tenant_id="tenant-a",
        requested_by="user-a",
        strategy=strategy,
        scope="symbol",
        symbol="BTC/USDT",
        idempotency_key="manual-close-idempotency-1",
    ))
    replay = asyncio.run(manual_close.execute_manual_close(
        session,
        tenant_id="tenant-a",
        requested_by="user-a",
        strategy=strategy,
        scope="symbol",
        symbol="BTC/USDT",
        idempotency_key="manual-close-idempotency-1",
    ))

    assert first["status"] == "completed"
    assert replay == first
    assert FakeExchange.submits == 1
