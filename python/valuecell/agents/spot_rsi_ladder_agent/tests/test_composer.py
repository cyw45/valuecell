import pytest

from valuecell.agents.common.trading.models import (
    ComposeContext,
    Constraints,
    ExchangeConfig,
    FeatureVector,
    InstrumentRef,
    LLMModelConfig,
    MarketType,
    PortfolioView,
    PositionSnapshot,
    TradeDigest,
    TradingConfig,
    TradingMode,
    UserRequest,
)
from valuecell.agents.spot_rsi_ladder_agent.composer import SpotRsiLadderComposer
from valuecell.agents.spot_rsi_ladder_agent.config import (
    LONG_TERM_PROFILE,
    SHORT_TERM_PROFILE,
)


def _make_request(profile_type: str) -> UserRequest:
    return UserRequest(
        llm_model_config=LLMModelConfig(provider="openrouter", model_id="test-model"),
        exchange_config=ExchangeConfig(
            exchange_id="okx",
            trading_mode=TradingMode.VIRTUAL,
            market_type=MarketType.SPOT,
        ),
        trading_config=TradingConfig(
            strategy_type=profile_type,
            symbols=["BTC-USDT"],
            initial_capital=1000.0,
            initial_free_cash=1000.0,
            max_leverage=1.0,
            max_positions=1,
        ),
    )


def _make_feature(
    symbol: str,
    interval: str,
    values: dict[str, object],
) -> FeatureVector:
    return FeatureVector(
        ts=1,
        instrument=InstrumentRef(symbol=symbol),
        values=values,
        meta={"interval": interval},
    )


@pytest.mark.asyncio
async def test_long_term_composer_emits_buy_instruction() -> None:
    request = _make_request(LONG_TERM_PROFILE.strategy_type)
    composer = SpotRsiLadderComposer(request, LONG_TERM_PROFILE)
    features = [
        _make_feature(
            "BTC-USDT",
            "30m",
            {
                "close": 95.0,
                "rsi": 24.0,
                "sma60": 100.0,
                "sma60_slope": 0.2,
                "close_turn_up": True,
                "rsi_turn_up": True,
                "bb_squeeze": True,
                "mtm_turn_up": True,
            },
        ),
        _make_feature(
            "BTC-USDT",
            "1m",
            {"close": 95.0, "rsi_turn_up": True, "close_turn_up": True},
        ),
        _make_feature(
            "BTC-USDT",
            "3m",
            {"close": 95.0, "rsi_turn_up": True, "close_turn_up": True},
        ),
        _make_feature(
            "BTC-USDT",
            "5m",
            {"close": 95.0, "rsi_turn_up": True, "close_turn_up": True},
        ),
        _make_feature(
            "BTC-USDT",
            "15m",
            {"close": 95.0, "rsi_turn_up": True, "close_turn_up": True},
        ),
        _make_feature(
            "BTC-USDT",
            "1h",
            {
                "close": 95.0,
                "rsi_turn_up": True,
                "sma60": 100.0,
                "sma60_slope": 0.1,
            },
        ),
        _make_feature(
            "BTC-USDT",
            "4h",
            {
                "close": 95.0,
                "rsi_turn_up": True,
                "sma60": 100.0,
                "sma60_slope": 0.1,
            },
        ),
        _make_feature("BTC-USDT", "1d", {"change_pct": 0.02}),
        FeatureVector(
            ts=1,
            instrument=InstrumentRef(symbol="BTC-USDT"),
            values={"price.last": 95.0},
            meta={"group_by_key": "market_snapshot"},
        ),
    ]
    portfolio = PortfolioView(
        ts=1,
        account_balance=600.0,
        positions={},
        constraints=Constraints(max_positions=1, max_leverage=1.0),
        total_value=600.0,
        free_cash=600.0,
    )
    context = ComposeContext(
        ts=1,
        compose_id="compose-1",
        strategy_id="strategy-1",
        features=features,
        portfolio=portfolio,
        digest=TradeDigest(ts=1),
    )

    result = await composer.compose(context)

    assert len(result.instructions) == 1
    assert result.instructions[0].action.value == "open_long"
    assert result.instructions[0].quantity == pytest.approx(240.0 / 95.0)


@pytest.mark.asyncio
async def test_short_term_composer_emits_buy_instruction_with_momentum_filters() -> None:
    request = _make_request(SHORT_TERM_PROFILE.strategy_type)
    composer = SpotRsiLadderComposer(request, SHORT_TERM_PROFILE)
    features = [
        _make_feature(
            "BTC-USDT",
            "15m",
            {
                "close": 95.0,
                "rsi": 18.0,
                "sma20": 100.0,
                "close_turn_up": True,
                "rsi_turn_up": True,
                "bb_squeeze": True,
                "bb_near_lower": True,
                "mtm_turn_up": True,
                "mtm_below_zero": True,
            },
        ),
        _make_feature(
            "BTC-USDT",
            "1m",
            {"close": 95.0, "rsi_turn_up": True, "close_turn_up": True},
        ),
        _make_feature(
            "BTC-USDT",
            "3m",
            {"close": 95.0, "rsi_turn_up": True, "close_turn_up": True},
        ),
        _make_feature(
            "BTC-USDT",
            "5m",
            {"close": 95.0, "rsi_turn_up": True, "close_turn_up": True},
        ),
        _make_feature(
            "BTC-USDT",
            "30m",
            {
                "close": 96.0,
                "rsi_turn_up": True,
                "close_turn_up": True,
                "sma20": 99.0,
                "sma20_slope": 0.1,
            },
        ),
        _make_feature(
            "BTC-USDT",
            "4h",
            {
                "close": 98.0,
                "sma20": 97.0,
                "sma20_slope": 0.1,
                "rsi": 48.0,
                "mtm14": 2.0,
            },
        ),
        _make_feature("BTC-USDT", "1d", {"change_pct": 0.02}),
        FeatureVector(
            ts=1,
            instrument=InstrumentRef(symbol="BTC-USDT"),
            values={"price.last": 95.0},
            meta={"group_by_key": "market_snapshot"},
        ),
    ]
    portfolio = PortfolioView(
        ts=1,
        account_balance=400.0,
        positions={},
        constraints=Constraints(max_positions=1, max_leverage=1.0),
        total_value=400.0,
        free_cash=400.0,
    )
    context = ComposeContext(
        ts=1,
        compose_id="compose-short-buy",
        strategy_id="strategy-short-buy",
        features=features,
        portfolio=portfolio,
        digest=TradeDigest(ts=1),
    )

    result = await composer.compose(context)

    assert len(result.instructions) == 1
    assert result.instructions[0].action.value == "open_long"
    assert result.instructions[0].quantity == pytest.approx(280.0 / 95.0)


@pytest.mark.asyncio
async def test_composer_is_silent_in_bear_market() -> None:
    request = _make_request(SHORT_TERM_PROFILE.strategy_type)
    composer = SpotRsiLadderComposer(request, SHORT_TERM_PROFILE)
    features = [
        _make_feature(
            "BTC-USDT",
            "15m",
            {
                "close": 105.0,
                "rsi": 66.0,
                "sma20": 100.0,
                "close_turn_up": True,
                "rsi_turn_up": True,
            },
        ),
        _make_feature("BTC-USDT", "30m", {"close": 105.0, "rsi_turn_up": True, "close_turn_up": True}),
        _make_feature(
            "BTC-USDT",
            "1d",
            {"change_pct": 0.03, "close": 90.0, "sma60": 100.0, "sma60_slope": -1.0},
        ),
        _make_feature("BTC-USDT", "4h", {"rsi": 40.0}),
        FeatureVector(
            ts=1,
            instrument=InstrumentRef(symbol="BTC-USDT"),
            values={"price.last": 105.0},
            meta={"group_by_key": "market_snapshot"},
        ),
    ]
    portfolio = PortfolioView(
        ts=1,
        account_balance=1000.0,
        positions={
            "BTC-USDT": PositionSnapshot(
                instrument=InstrumentRef(symbol="BTC-USDT"),
                quantity=1.0,
                avg_price=90.0,
                mark_price=105.0,
            )
        },
        constraints=Constraints(max_positions=1, max_leverage=1.0),
        total_value=1100.0,
    )
    context = ComposeContext(
        ts=1,
        compose_id="compose-2",
        strategy_id="strategy-2",
        features=features,
        portfolio=portfolio,
        digest=TradeDigest(ts=1),
    )

    result = await composer.compose(context)

    assert result.instructions == []


@pytest.mark.asyncio
async def test_short_term_add_uses_ten_percent_of_remaining_cash() -> None:
    request = _make_request(SHORT_TERM_PROFILE.strategy_type)
    composer = SpotRsiLadderComposer(request, SHORT_TERM_PROFILE)
    features = [
        _make_feature(
            "BTC-USDT",
            "15m",
            {
                "close": 100.0,
                "rsi": 50.0,
                "sma20": 95.0,
                "close_turn_up": True,
                "rsi_turn_up": True,
                "bb_mid_cross_up": True,
            },
        ),
        _make_feature(
            "BTC-USDT",
            "4h",
            {
                "close": 101.0,
                "sma20": 99.0,
                "sma20_slope": 0.2,
                "rsi": 50.0,
                "mtm14": 1.5,
            },
        ),
        _make_feature(
            "BTC-USDT",
            "30m",
            {
                "close": 100.0,
                "rsi_turn_up": True,
                "close_turn_up": True,
                "sma20": 99.0,
                "sma20_slope": 0.1,
            },
        ),
        _make_feature("BTC-USDT", "1d", {"change_pct": 0.02}),
        FeatureVector(
            ts=1,
            instrument=InstrumentRef(symbol="BTC-USDT"),
            values={"price.last": 100.0},
            meta={"group_by_key": "market_snapshot"},
        ),
    ]
    portfolio = PortfolioView(
        ts=1,
        account_balance=300.0,
        positions={
            "BTC-USDT": PositionSnapshot(
                instrument=InstrumentRef(symbol="BTC-USDT"),
                quantity=1.0,
                avg_price=90.0,
                mark_price=100.0,
            )
        },
        constraints=Constraints(max_positions=1, max_leverage=1.0),
        total_value=500.0,
        free_cash=300.0,
    )
    context = ComposeContext(
        ts=1,
        compose_id="compose-short-add",
        strategy_id="strategy-short-add",
        features=features,
        portfolio=portfolio,
        digest=TradeDigest(ts=1),
    )

    result = await composer.compose(context)

    assert len(result.instructions) == 1
    assert result.instructions[0].action.value == "open_long"
    assert result.instructions[0].quantity == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_long_term_tail_drawdown_closes_remaining_tail_before_final_rsi() -> None:
    request = _make_request(LONG_TERM_PROFILE.strategy_type)
    composer = SpotRsiLadderComposer(request, LONG_TERM_PROFILE)
    portfolio = PortfolioView(
        ts=1,
        account_balance=0.0,
        positions={
            "BTC-USDT": PositionSnapshot(
                instrument=InstrumentRef(symbol="BTC-USDT"),
                quantity=1.0,
                avg_price=90.0,
                mark_price=120.0,
            )
        },
        constraints=Constraints(max_positions=1, max_leverage=1.0),
        total_value=120.0,
        free_cash=0.0,
    )
    first_context = ComposeContext(
        ts=1,
        compose_id="compose-tail-1",
        strategy_id="strategy-tail",
        features=[
            _make_feature(
                "BTC-USDT",
                "30m",
                {
                    "close": 120.0,
                    "rsi": 80.0,
                    "sma60": 100.0,
                    "close_turn_up": True,
                    "rsi_turn_up": True,
                },
            ),
            _make_feature("BTC-USDT", "1d", {"change_pct": 0.02}),
            FeatureVector(
                ts=1,
                instrument=InstrumentRef(symbol="BTC-USDT"),
                values={"price.last": 120.0},
                meta={"group_by_key": "market_snapshot"},
            ),
        ],
        portfolio=portfolio,
        digest=TradeDigest(ts=1),
    )

    first_result = await composer.compose(first_context)

    assert len(first_result.instructions) == 1
    assert first_result.instructions[0].action.value == "close_long"
    assert first_result.instructions[0].quantity == pytest.approx(0.9)

    second_context = ComposeContext(
        ts=2,
        compose_id="compose-tail-2",
        strategy_id="strategy-tail",
        features=[
            _make_feature(
                "BTC-USDT",
                "30m",
                {
                    "close": 95.0,
                    "rsi": 83.0,
                    "sma60": 100.0,
                    "close_turn_up": False,
                    "rsi_turn_up": False,
                },
            ),
            _make_feature("BTC-USDT", "1d", {"change_pct": -0.10}),
            FeatureVector(
                ts=2,
                instrument=InstrumentRef(symbol="BTC-USDT"),
                values={"price.last": 95.0},
                meta={"group_by_key": "market_snapshot"},
            ),
        ],
        portfolio=PortfolioView(
            ts=2,
            account_balance=0.0,
            positions={
                "BTC-USDT": PositionSnapshot(
                    instrument=InstrumentRef(symbol="BTC-USDT"),
                    quantity=0.1,
                    avg_price=90.0,
                    mark_price=95.0,
                )
            },
            constraints=Constraints(max_positions=1, max_leverage=1.0),
            total_value=9.5,
            free_cash=0.0,
        ),
        digest=TradeDigest(ts=2),
    )

    second_result = await composer.compose(second_context)

    assert len(second_result.instructions) == 1
    assert second_result.instructions[0].action.value == "close_long"
    assert second_result.instructions[0].quantity == pytest.approx(0.1)
