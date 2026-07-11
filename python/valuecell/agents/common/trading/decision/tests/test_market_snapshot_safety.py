import pytest

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_KEY,
    FEATURE_GROUP_BY_MARKET_SNAPSHOT,
)
from valuecell.agents.common.trading.decision.interfaces import BaseComposer
from valuecell.agents.common.trading.models import (
    ComposeContext,
    ComposeResult,
    Constraints,
    ExchangeConfig,
    FeatureVector,
    InstrumentRef,
    LLMModelConfig,
    MarketType,
    PortfolioView,
    PositionSnapshot,
    TradeDecisionAction,
    TradeDecisionItem,
    TradeDigest,
    TradePlanProposal,
    TradeSide,
    TradingConfig,
    UserRequest,
)


SYMBOL = "BTC-USDT"


class StaticPlanComposer(BaseComposer):
    """Exercises BaseComposer guardrails with a deterministic proposed plan."""

    def __init__(self, request: UserRequest, plan: TradePlanProposal) -> None:
        super().__init__(request)
        self._plan = plan

    async def compose(self, context: ComposeContext) -> ComposeResult:
        return ComposeResult(
            instructions=self._normalize_plan(context, self._plan),
            rationale=self._plan.rationale,
        )


def _request() -> UserRequest:
    return UserRequest(
        llm_model_config=LLMModelConfig(provider="openrouter", model_id="test-model"),
        exchange_config=ExchangeConfig(exchange_id="okx", market_type=MarketType.SWAP),
        trading_config=TradingConfig(
            symbols=[SYMBOL],
            initial_capital=1_000.0,
            initial_free_cash=1_000.0,
            max_leverage=2.0,
            max_positions=1,
        ),
    )


def _plan(action: TradeDecisionAction, quantity: float) -> TradePlanProposal:
    return TradePlanProposal(
        ts=1,
        rationale="test plan",
        items=[
            TradeDecisionItem(
                instrument=InstrumentRef(symbol=SYMBOL, exchange_id="okx"),
                action=action,
                target_qty=quantity,
            )
        ],
    )


def _stale_snapshot() -> FeatureVector:
    return FeatureVector(
        ts=0,
        instrument=InstrumentRef(symbol=SYMBOL, exchange_id="okx"),
        values={"price.last": 100.0},
        meta={
            FEATURE_GROUP_BY_KEY: FEATURE_GROUP_BY_MARKET_SNAPSHOT,
            "snapshot_ts_ms": 0,
            "freshness_age_ms": 60_001,
            "freshness_status": "stale",
            "coverage_status": "complete",
        },
    )


def _context(
    *, features: list[FeatureVector], current_quantity: float = 0.0
) -> ComposeContext:
    positions = {}
    if current_quantity:
        positions[SYMBOL] = PositionSnapshot(
            instrument=InstrumentRef(symbol=SYMBOL, exchange_id="okx"),
            quantity=current_quantity,
            mark_price=100.0,
        )
    return ComposeContext(
        ts=1,
        compose_id="snapshot-safety",
        strategy_id="strategy-snapshot-safety",
        features=features,
        portfolio=PortfolioView(
            ts=1,
            account_balance=1_000.0,
            positions=positions,
            constraints=Constraints(max_positions=1, max_leverage=2.0),
            total_value=1_000.0,
        ),
        digest=TradeDigest(ts=1),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "features"),
    [
        ("missing snapshot", []),
        ("stale snapshot", [_stale_snapshot()]),
    ],
)
async def test_unusable_market_snapshot_blocks_exposure_increases(
    name: str, features: list[FeatureVector]
) -> None:
    result = await StaticPlanComposer(
        _request(), _plan(TradeDecisionAction.OPEN_LONG, 1.0)
    ).compose(_context(features=features))

    assert result.instructions == [], name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "features"),
    [
        ("missing snapshot", []),
        ("stale snapshot", [_stale_snapshot()]),
    ],
)
async def test_unusable_market_snapshot_preserves_reduce_only_long_exit(
    name: str, features: list[FeatureVector]
) -> None:
    result = await StaticPlanComposer(
        _request(), _plan(TradeDecisionAction.CLOSE_LONG, 1.0)
    ).compose(_context(features=features, current_quantity=2.0))

    assert len(result.instructions) == 1, name
    instruction = result.instructions[0]
    assert instruction.action is TradeDecisionAction.CLOSE_LONG
    assert instruction.side is TradeSide.SELL
    assert instruction.quantity == 1.0
    assert instruction.meta["reduceOnly"] is True
    assert instruction.meta["final_target_qty"] == 1.0
