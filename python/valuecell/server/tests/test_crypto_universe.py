"""Catalogue sync, admission thresholds, and strategy propagation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.db.models.base import Base
from valuecell.server.db.models.crypto_universe import CryptoSymbolUniverse
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyAccount,
    RuleStrategyEvent,
    RuleStrategyMonitorSymbol,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.db.repositories.crypto_universe_repository import (
    CryptoSymbolUniverseRepository,
    invalidate_active_symbol_cache,
)
from valuecell.server.services import crypto_universe_facts as facts_module
from valuecell.server.services import crypto_universe_service as universe_module
from valuecell.server.services.crypto_universe_facts import (
    REASON_LISTING_AGE,
    REASON_VOLUME_BELOW_MINIMUM,
    REASON_VOLUME_UNAVAILABLE,
    OkxSpotInstrument,
    SymbolFacts,
    evaluate_symbol,
    policy_exclusion,
    select_volume_candidates,
)
from valuecell.server.services.crypto_universe_service import (
    CryptoSymbolUniverseService,
)

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
MIN_AGE_DAYS = 90
MIN_VOLUME = 5_000_000.0


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _instrument(
    symbol: str, *, state: str = "live", listed_days_ago: int = 400
) -> OkxSpotInstrument:
    base = symbol.split("-")[0]
    return OkxSpotInstrument(
        instrument_id=symbol,
        base_asset=base,
        quote_asset="USDT",
        state=state,
        listed_at=NOW - timedelta(days=listed_days_ago),
    )


def _config_symbols(session, strategy_id: str) -> list[str]:
    strategy = (
        session.query(RuleStrategy)
        .filter(RuleStrategy.strategy_id == strategy_id)
        .one()
    )
    return list(strategy.config["symbols"])


def _add_strategy(session, *, strategy_id: str, symbols: list[str]) -> None:
    from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig

    config = RuleStrategyConfig(symbols=symbols)
    session.add(
        RuleStrategy(
            strategy_id=strategy_id,
            tenant_id="tenant-a",
            name=strategy_id,
            config=config.model_dump(mode="json"),
        )
    )
    session.add_all(
        [
            RuleStrategyMonitorSymbol(
                tenant_id="tenant-a",
                strategy_id=strategy_id,
                symbol=symbol,
                state="admitted",
            )
            for symbol in symbols
        ]
    )


def _publish_initial(session, symbols: list[str]) -> None:
    repository = CryptoSymbolUniverseRepository(db_session=session)
    repository.publish(
        observed_at=NOW,
        source="okx",
        quote_asset="USDT",
        min_listing_age_days=MIN_AGE_DAYS,
        min_average_quote_volume_30d=MIN_VOLUME,
        reason_detail=None,
        entries=[
            {
                "symbol": symbol,
                "state": "admitted",
                "decision": "added",
                "reason_code": "okx_liquidity_verified",
                "reason_detail": None,
            }
            for symbol in symbols
        ],
    )


def test_evaluate_symbol_rejects_short_listing_age():
    decision = evaluate_symbol(
        symbol="NEW-USDT",
        facts=SymbolFacts(
            symbol="NEW-USDT",
            average_quote_volume_30d=50_000_000.0,
            observed_days=30,
        ),
        listed_at=NOW - timedelta(days=10),
        observed_at=NOW,
        minimum_listing_age_days=MIN_AGE_DAYS,
        minimum_average_quote_volume_30d=MIN_VOLUME,
    )

    assert decision.admitted is False
    assert decision.reason_code == REASON_LISTING_AGE
    assert decision.permanent_exclusion is False


def test_evaluate_symbol_rejects_low_volume_but_keeps_it_recoverable():
    decision = evaluate_symbol(
        symbol="THIN-USDT",
        facts=SymbolFacts(
            symbol="THIN-USDT",
            average_quote_volume_30d=100_000.0,
            observed_days=30,
        ),
        listed_at=NOW - timedelta(days=800),
        observed_at=NOW,
        minimum_listing_age_days=MIN_AGE_DAYS,
        minimum_average_quote_volume_30d=MIN_VOLUME,
    )

    assert decision.admitted is False
    assert decision.reason_code == REASON_VOLUME_BELOW_MINIMUM
    assert "100,000 USDT" in decision.reason_detail
    assert decision.permanent_exclusion is False


def test_evaluate_symbol_fails_closed_without_volume_evidence():
    decision = evaluate_symbol(
        symbol="NODATA-USDT",
        facts=SymbolFacts(symbol="NODATA-USDT"),
        listed_at=NOW - timedelta(days=500),
        observed_at=NOW,
        minimum_listing_age_days=MIN_AGE_DAYS,
        minimum_average_quote_volume_30d=MIN_VOLUME,
    )

    assert decision.admitted is False
    assert decision.reason_code == REASON_VOLUME_UNAVAILABLE


def test_evaluate_symbol_admits_verified_liquid_symbol():
    decision = evaluate_symbol(
        symbol="BTC-USDT",
        facts=SymbolFacts(
            symbol="BTC-USDT",
            average_quote_volume_30d=9_000_000_000.0,
            observed_days=30,
        ),
        listed_at=NOW - timedelta(days=3000),
        observed_at=NOW,
        minimum_listing_age_days=MIN_AGE_DAYS,
        minimum_average_quote_volume_30d=MIN_VOLUME,
    )

    assert decision.admitted is True
    assert decision.reason_code == "okx_liquidity_verified"


def test_policy_exclusion_marks_stable_and_leveraged_assets_permanent():
    stable = policy_exclusion("USDC")
    leveraged = policy_exclusion("BTC3L")

    assert stable is not None and stable.permanent_exclusion is True
    assert leveraged is not None and leveraged.permanent_exclusion is True
    assert policy_exclusion("BTC") is None


def test_volume_prefilter_is_bounded_and_keeps_quiet_day_symbols():
    instruments = [_instrument(f"SYM{i}-USDT") for i in range(10)]
    ticker_volumes = {
        f"SYM{index}-USDT": float(9 - index) * 1_000_000.0 for index in range(6)
    }
    # A quiet day on an otherwise liquid symbol must survive the prefilter,
    # while symbols below the tolerance are never verified.
    ticker_volumes["SYM8-USDT"] = 1_500_000.0
    ticker_volumes["SYM9-USDT"] = 10_000.0

    bounded = select_volume_candidates(
        instruments,
        ticker_volumes,
        minimum_average_quote_volume_30d=MIN_VOLUME,
        maximum_candidates=4,
    )
    wide = select_volume_candidates(
        instruments,
        ticker_volumes,
        minimum_average_quote_volume_30d=MIN_VOLUME,
        maximum_candidates=100,
    )

    assert [item.instrument_id for item in bounded] == [
        "SYM0-USDT",
        "SYM1-USDT",
        "SYM2-USDT",
        "SYM3-USDT",
    ]
    assert [item.instrument_id for item in wide] == [
        "SYM0-USDT",
        "SYM1-USDT",
        "SYM2-USDT",
        "SYM3-USDT",
        "SYM4-USDT",
        "SYM5-USDT",
        "SYM8-USDT",
    ]


def test_daily_quote_volumes_ignore_unconfirmed_and_current_day(monkeypatch):
    today_ms = int(NOW.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    day_ms = 86_400_000
    rows = [
        [str(today_ms), "1", "1", "1", "12.5", "10", "125", "125", "0"],
        [str(today_ms - day_ms), "1", "1", "1", "11.0", "10", "110", "110", "1"],
        [str(today_ms - 2 * day_ms), "1", "1", "1", "10.0", "10", "100", "100", "1"],
        [str(today_ms - 3 * day_ms), "1", "1", "1", "9.0", "10", "", "", "1"],
    ]
    monkeypatch.setattr(
        facts_module,
        "_fetch_okx_json",
        lambda path, query, timeout_s: {"code": "0", "data": rows},
    )

    volumes, close_price = facts_module.fetch_okx_daily_quote_volumes(
        "BTC-USDT", window_days=30, timeout_s=1.0, observed_at=NOW
    )

    assert volumes == [100.0, 110.0]
    assert close_price == 11.0


def test_sync_aborts_without_touching_published_version(monkeypatch):
    session = _session()
    _publish_initial(session, ["BTC-USDT"])
    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("okx unreachable")

    monkeypatch.setattr(facts_module, "fetch_okx_spot_instruments", _boom)

    result = service.sync(force=True)

    assert result.status == "failed"
    assert result.reason == "okx_instruments_unavailable"
    active = session.query(CryptoSymbolUniverse).filter_by(status="active").one()
    assert active.version == 1
    assert service.repository.active_symbols() == ["BTC-USDT"]


def test_sync_prunes_delisted_symbol_and_appends_new_admitted_symbol(monkeypatch):
    session = _session()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    _publish_initial(session, ["BTC-USDT", "DEAD-USDT"])
    _add_strategy(session, strategy_id="strategy-a", symbols=["BTC-USDT", "DEAD-USDT"])
    session.commit()

    monkeypatch.setattr(
        facts_module,
        "fetch_okx_spot_instruments",
        lambda quote_asset, timeout_s: [
            _instrument("BTC-USDT"),
            _instrument("DEAD-USDT", state="suspend"),
            _instrument("NEW-USDT", listed_days_ago=200),
        ],
    )
    monkeypatch.setattr(
        facts_module,
        "fetch_okx_spot_tickers",
        lambda timeout_s: {
            "BTC-USDT": 900_000_000.0,
            "NEW-USDT": 40_000_000.0,
        },
    )
    monkeypatch.setattr(
        facts_module,
        "gather_symbol_facts",
        lambda candidates, **kwargs: {
            "BTC-USDT": SymbolFacts(
                symbol="BTC-USDT",
                average_quote_volume_30d=900_000_000.0,
                price_quote=60_000.0,
                observed_days=30,
            ),
            "NEW-USDT": SymbolFacts(
                symbol="NEW-USDT",
                average_quote_volume_30d=40_000_000.0,
                price_quote=3.5,
                observed_days=30,
            ),
        },
    )

    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )
    result = service.sync(force=True, now=NOW)

    assert result.status == "published"
    assert result.admitted == 2
    assert result.removed == 1
    assert result.added == 1
    assert result.strategies_adapted == 1
    assert sorted(service.repository.active_symbols()) == ["BTC-USDT", "NEW-USDT"]
    # The delisted symbol is gone from the strategy so no unsupported order can
    # be attempted, and the newly admitted one is electable.
    assert _config_symbols(session, "strategy-a") == ["BTC-USDT", "NEW-USDT"]
    events = session.query(RuleStrategyEvent).all()
    assert [event.reason_code for event in events] == ["crypto_universe_adapted"]
    assert events[0].before_state["symbols"] == ["BTC-USDT", "DEAD-USDT"]
    assert events[0].after_state["removed_symbols"] == ["DEAD-USDT"]
    monitor = (
        session.query(RuleStrategyMonitorSymbol)
        .filter_by(strategy_id="strategy-a", symbol="DEAD-USDT")
        .one()
    )
    assert monitor.state == "removed"
    assert monitor.reason_code == "crypto_universe_removed"


def test_sync_keeps_removed_symbol_while_position_is_open(monkeypatch):
    session = _session()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    _publish_initial(session, ["BTC-USDT", "GONE-USDT"])
    _add_strategy(session, strategy_id="strategy-b", symbols=["BTC-USDT", "GONE-USDT"])
    session.add(
        RuleStrategyAccount(
            tenant_id="tenant-a",
            strategy_id="strategy-b",
            scope="paper_virtual",
            allocation_quote=1000.0,
            quote_balance=900.0,
            equity_quote=1000.0,
            positions={"GONE-USDT": {"quantity": 5.0}},
        )
    )
    session.commit()

    monkeypatch.setattr(
        facts_module,
        "fetch_okx_spot_instruments",
        lambda quote_asset, timeout_s: [
            _instrument("BTC-USDT"),
            _instrument("GONE-USDT", state="suspend"),
        ],
    )
    monkeypatch.setattr(
        facts_module,
        "fetch_okx_spot_tickers",
        lambda timeout_s: {"BTC-USDT": 900_000_000.0},
    )
    monkeypatch.setattr(
        facts_module,
        "gather_symbol_facts",
        lambda candidates, **kwargs: {
            "BTC-USDT": SymbolFacts(
                symbol="BTC-USDT",
                average_quote_volume_30d=900_000_000.0,
                observed_days=30,
            )
        },
    )

    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )
    service.sync(force=True, now=NOW)

    # The exit must stay evaluable even though the venue dropped the symbol.
    assert _config_symbols(session, "strategy-b") == ["BTC-USDT", "GONE-USDT"]
    event = session.query(RuleStrategyEvent).one()
    assert event.reason_code == "crypto_universe_protected_symbol_retained"
    assert event.after_state["protected_symbols"] == ["GONE-USDT"]
    assert event.after_state["removed_symbols"] == []


def _demo_strategy(session, *, strategy_id: str, symbols: list[str]) -> None:
    _add_strategy(session, strategy_id=strategy_id, symbols=symbols)
    strategy = (
        session.query(RuleStrategy)
        .filter(RuleStrategy.strategy_id == strategy_id)
        .one()
    )
    strategy.config = {
        **strategy.config,
        "execution": {
            "environment": "okx_demo",
            "sandbox_connection_id": "credential-a",
        },
    }
    strategy.current_batch_id = "batch-demo"


def _stub_okx_dropping_gone_symbol(monkeypatch) -> None:
    monkeypatch.setattr(
        facts_module,
        "fetch_okx_spot_instruments",
        lambda quote_asset, timeout_s: [
            _instrument("BTC-USDT"),
            _instrument("GONE-USDT", state="suspend"),
        ],
    )
    monkeypatch.setattr(
        facts_module,
        "fetch_okx_spot_tickers",
        lambda timeout_s: {"BTC-USDT": 900_000_000.0},
    )
    monkeypatch.setattr(
        facts_module,
        "gather_symbol_facts",
        lambda candidates, **kwargs: {
            "BTC-USDT": SymbolFacts(
                symbol="BTC-USDT",
                average_quote_volume_30d=900_000_000.0,
                observed_days=30,
            )
        },
    )


def _demo_evidence_holding(symbol: str) -> dict[str, list[dict]]:
    """Evidence shaped like the shared Demo read model returns to the caller."""
    return {
        "venue_orders": [
            {
                "order_id": "order-demo-1",
                "strategy_id": "strategy-demo",
                "batch_id": "batch-demo",
                "symbol": symbol,
                "side": "buy",
                "created_at": NOW - timedelta(days=1),
            }
        ],
        "order_projections": [
            {
                "order_id": "order-demo-1",
                "status": "filled",
                "filled_quantity": "5",
                "filled_quote": "250",
            }
        ],
        "fills": [
            {
                "order_id": "order-demo-1",
                "strategy_id": "strategy-demo",
                "batch_id": "batch-demo",
                "symbol": symbol,
                "side": "buy",
                "quantity": "5",
                "quote_amount": "250",
                "occurred_at": NOW - timedelta(days=1),
            }
        ],
    }


def test_adaptation_keeps_a_symbol_with_attributed_demo_ownership(monkeypatch):
    session = _session()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    _publish_initial(session, ["BTC-USDT", "GONE-USDT"])
    _demo_strategy(
        session, strategy_id="strategy-demo", symbols=["BTC-USDT", "GONE-USDT"]
    )
    session.commit()

    _stub_okx_dropping_gone_symbol(monkeypatch)
    monkeypatch.setattr(
        universe_module,
        "shared_demo_evidence_for_strategy",
        lambda session, **kwargs: _demo_evidence_holding("GONE-USDT"),
    )

    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )
    service.sync(force=True, now=NOW)

    # Strategy-owned Demo inventory keeps the exit evaluable after the drop.
    assert _config_symbols(session, "strategy-demo") == ["BTC-USDT", "GONE-USDT"]
    event = session.query(RuleStrategyEvent).one()
    assert event.reason_code == "crypto_universe_protected_symbol_retained"
    assert event.after_state["protected_symbols"] == ["GONE-USDT"]


def test_adaptation_drops_a_symbol_without_attributed_demo_ownership(monkeypatch):
    session = _session()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    _publish_initial(session, ["BTC-USDT", "GONE-USDT"])
    _demo_strategy(
        session, strategy_id="strategy-demo", symbols=["BTC-USDT", "GONE-USDT"]
    )
    session.commit()

    _stub_okx_dropping_gone_symbol(monkeypatch)
    # Shared wallet balances are not strategy-owned evidence, so they protect
    # nothing: only attributed fills may keep a dropped symbol observable.
    monkeypatch.setattr(
        universe_module,
        "shared_demo_evidence_for_strategy",
        lambda session, **kwargs: {
            "venue_orders": [],
            "order_projections": [],
            "fills": [],
        },
    )

    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )
    service.sync(force=True, now=NOW)

    assert _config_symbols(session, "strategy-demo") == ["BTC-USDT"]
    monitor = (
        session.query(RuleStrategyMonitorSymbol)
        .filter(RuleStrategyMonitorSymbol.symbol == "GONE-USDT")
        .one()
    )
    assert monitor.state == "removed"


def test_catalogue_reads_back_with_facts_and_thresholds():
    session = _session()
    _publish_initial(session, ["BTC-USDT"])
    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )

    catalog = service.catalog()

    assert catalog is not None
    assert catalog["version"] == 1
    assert catalog["admitted_count"] == 1
    assert catalog["sync_interval_days"] == 90
    assert catalog["min_average_quote_volume_30d"] == MIN_VOLUME
    assert catalog["entries"][0]["symbol"] == "BTC-USDT"
    assert catalog["entries"][0]["average_quote_volume_30d"] is None


def test_catalogue_is_absent_before_the_first_seed():
    session = _session()
    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )

    assert service.catalog() is None


def test_seed_from_code_defaults_is_idempotent():
    session = _session()
    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )

    first = service.seed_from_code_defaults(now=NOW)
    second = service.seed_from_code_defaults(now=NOW)

    assert first == 1
    assert second is None
    assert session.query(CryptoSymbolUniverse).count() == 1


def test_supported_symbols_follow_published_catalogue(monkeypatch):
    from valuecell.server.db.repositories import crypto_universe_repository as repo_module
    from valuecell.server.services.crypto_market_service import CryptoMarketService

    invalidate_active_symbol_cache()
    monkeypatch.setattr(
        repo_module, "cached_active_symbols", lambda repository=None: ("WIF-USDT",)
    )
    service = CryptoMarketService(providers=("okx",))

    catalog = service.get_supported_symbols()

    assert catalog.symbols == ["WIF-USDT"]
    # A symbol outside the seed tuple is accepted once the catalogue admits it.
    assert service._normalize_symbols(["WIF-USDT"]) == ["WIF-USDT"]
    with pytest.raises(ValueError):
        service._normalize_symbols(["NOTLISTED-USDT"])


def test_universe_sync_job_is_idempotent_when_not_due(monkeypatch):
    session = _session()
    _publish_initial(session, ["BTC-USDT"])
    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )

    assert service.is_due(NOW) is False
    assert service.is_due(NOW + timedelta(days=91)) is True

    monkeypatch.setattr(
        facts_module,
        "fetch_okx_spot_instruments",
        lambda *args, **kwargs: pytest.fail("catalogue must not refetch when not due"),
    )
    result = service.sync(now=NOW)

    assert result.status == "skipped"
    assert result.reason == "not_due"


def test_adaptation_refuses_to_write_an_empty_symbol_list(monkeypatch):
    session = _session()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    _publish_initial(session, ["BTC-USDT", "DEAD-USDT"])
    _add_strategy(session, strategy_id="strategy-c", symbols=["BTC-USDT", "DEAD-USDT"])
    session.commit()

    service = CryptoSymbolUniverseService(
        repository=CryptoSymbolUniverseRepository(db_session=session),
        db_session=session,
    )
    adapted = service.adapt_strategies([], removed_symbols=["BTC-USDT"], entries=[], now=NOW)

    assert adapted == 0
    assert _config_symbols(session, "strategy-c") == ["BTC-USDT", "DEAD-USDT"]
