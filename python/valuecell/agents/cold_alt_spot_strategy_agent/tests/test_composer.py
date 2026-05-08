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
from valuecell.agents.cold_alt_spot_strategy_agent.composer import ColdAltSpotComposer


def _make_request() -> UserRequest:
    return UserRequest(
        llm_model_config=LLMModelConfig(provider="openrouter", model_id="test-model"),
        exchange_config=ExchangeConfig(
            exchange_id="okx",
            trading_mode=TradingMode.VIRTUAL,
            market_type=MarketType.SPOT,
        ),
        trading_config=TradingConfig(
            strategy_type="ColdAltSpotStrategy",
            symbols=["ALT-USDT"],
            initial_capital=1000.0,
            initial_free_cash=1000.0,
            max_leverage=1.0,
            max_positions=5,
            decide_interval=3600,
        ),
    )


def _make_feature(symbol: str, interval: str, values: dict[str, object]) -> FeatureVector:
    return FeatureVector(
        ts=1,
        instrument=InstrumentRef(symbol=symbol),
        values=values,
        meta={"interval": interval},
    )


@pytest.mark.asyncio
async def test_composer_emits_old_coin_entry() -> None:
    composer = ColdAltSpotComposer(_make_request())
    features = [
        _make_feature(
            "ALT-USDT",
            "1d",
            {
                "count": 200,
                "sideways_range_ratio_14": 0.18,
                "volume_contraction_ratio_14v90": 0.35,
                "change_pct": 0.02,
            },
        ),
        _make_feature(
            "ALT-USDT",
            "4h",
            {
                "count": 220,
                "close": 1.0,
                "rsi": 18.0,
                "rsi_turn_up": True,
                "rsi_bottom_divergence": True,
                "rsi_ice_zone_bars": 2,
                "price_in_lower_half": True,
                "price_near_lower_band": True,
                "bb_squeeze_3bar": True,
                "mtm_cross_up_zero": True,
                "mtm_turn_up": True,
                "mtm14": 0.03,
                "mtm_positive_bars": 1,
                "ar26": 45.0,
                "br26": 70.0,
                "arbr_warm_up": True,
            },
        ),
        _make_feature(
            "ALT-USDT",
            "1h",
            {"rsi_turn_up": True, "mtm_turn_up": True, "close_turn_up": True},
        ),
        _make_feature("ALT-USDT", "15m", {"launch_rocket_15m": False}),
        FeatureVector(
            ts=1,
            instrument=InstrumentRef(symbol="ALT-USDT"),
            values={"price.last": 1.0},
            meta={"group_by_key": "market_snapshot"},
        ),
    ]
    portfolio = PortfolioView(
        ts=1,
        account_balance=1000.0,
        positions={},
        constraints=Constraints(max_positions=5, max_leverage=1.0),
        total_value=1000.0,
    )
    context = ComposeContext(
        ts=1,
        compose_id="compose-cold-1",
        strategy_id="strategy-cold-1",
        features=features,
        portfolio=portfolio,
        digest=TradeDigest(ts=1),
    )

    result = await composer.compose(context)

    assert len(result.instructions) == 1
    assert result.instructions[0].action.value == "open_long"
    assert result.instructions[0].quantity > 0


@pytest.mark.asyncio
async def test_composer_emits_stage_one_exit() -> None:
    composer = ColdAltSpotComposer(_make_request())
    features = [
        _make_feature(
            "ALT-USDT",
            "1d",
            {
                "count": 200,
                "sideways_range_ratio_14": 0.18,
                "volume_contraction_ratio_14v90": 0.35,
                "change_pct": 0.02,
            },
        ),
        _make_feature(
            "ALT-USDT",
            "4h",
            {
                "count": 220,
                "close": 1.3,
                "rsi": 61.0,
                "upper_band_rejection": True,
                "mtm_turn_up": True,
            },
        ),
        _make_feature(
            "ALT-USDT",
            "1h",
            {"rsi_turn_up": True, "mtm_turn_up": True, "close_turn_up": True},
        ),
        _make_feature("ALT-USDT", "15m", {"launch_rocket_15m": False}),
        FeatureVector(
            ts=1,
            instrument=InstrumentRef(symbol="ALT-USDT"),
            values={"price.last": 1.3},
            meta={"group_by_key": "market_snapshot"},
        ),
    ]
    portfolio = PortfolioView(
        ts=1,
        account_balance=1000.0,
        positions={
            "ALT-USDT": PositionSnapshot(
                instrument=InstrumentRef(symbol="ALT-USDT"),
                quantity=100.0,
                avg_price=1.0,
                mark_price=1.3,
            )
        },
        constraints=Constraints(max_positions=5, max_leverage=1.0),
        total_value=1130.0,
    )
    context = ComposeContext(
        ts=1,
        compose_id="compose-cold-2",
        strategy_id="strategy-cold-2",
        features=features,
        portfolio=portfolio,
        digest=TradeDigest(ts=1),
    )

    result = await composer.compose(context)

    assert len(result.instructions) == 1
    assert result.instructions[0].action.value == "close_long"
    assert result.instructions[0].quantity > 0
