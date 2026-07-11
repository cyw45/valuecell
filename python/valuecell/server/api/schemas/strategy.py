"""
Strategy API schemas for handling strategy-related requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from .base import SuccessResponse


class StrategyType(str, Enum):
    PROMPT = "PromptBasedStrategy"
    GRID = "GridStrategy"
    LONG_TERM_SPOT_RSI = "LongTermSpotRsiStrategy"
    SHORT_TERM_SPOT_RSI = "ShortTermSpotRsiStrategy"


class StrategySummaryData(BaseModel):
    """Summary data for a single strategy per product spec."""

    strategy_id: str = Field(
        ..., description="Runtime strategy identifier from StrategyAgent"
    )
    strategy_name: Optional[str] = Field(None, description="User-defined strategy name")
    strategy_type: Optional[StrategyType] = Field(
        None,
        description="Strategy type identifier",
    )
    status: Literal["running", "stopped"] = Field(..., description="Strategy status")
    stop_reason: Optional[str] = Field(None, description="Reason for strategy stop")
    trading_mode: Optional[Literal["live", "virtual"]] = Field(
        None, description="Trading mode: live or virtual"
    )
    total_pnl: Optional[float] = Field(None, description="Total PnL value")
    total_pnl_pct: Optional[float] = Field(None, description="Total PnL percentage")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    exchange_id: Optional[str] = Field(
        None, description="Associated exchange identifier"
    )
    model_id: Optional[str] = Field(None, description="Associated model identifier")


class StrategyListData(BaseModel):
    """Data model for strategy list."""

    strategies: List[StrategySummaryData] = Field(..., description="List of strategies")
    total: int = Field(..., description="Total number of strategies")
    running_count: int = Field(..., description="Number of running strategies")


StrategyListResponse = SuccessResponse[StrategyListData]


class PositionHoldingItem(BaseModel):
    symbol: str = Field(..., description="Instrument symbol")
    exchange_id: Optional[str] = Field(None, description="Exchange identifier")
    quantity: float = Field(..., description="Position quantity (+long, -short)")
    avg_price: Optional[float] = Field(None, description="Average entry price")
    mark_price: Optional[float] = Field(
        None, description="Current mark/reference price"
    )
    unrealized_pnl: Optional[float] = Field(None, description="Unrealized PnL value")
    unrealized_pnl_pct: Optional[float] = Field(
        None, description="Unrealized PnL percentage"
    )
    notional: Optional[float] = Field(
        None, description="Position notional in quote currency"
    )
    leverage: Optional[float] = Field(
        None, description="Leverage applied to the position"
    )
    entry_ts: Optional[int] = Field(None, description="Entry timestamp (ms)")
    trade_type: Optional[str] = Field(None, description="Trade type (LONG/SHORT)")


class StrategyHoldingData(BaseModel):
    strategy_id: str = Field(..., description="Strategy identifier")
    ts: int = Field(..., description="Snapshot timestamp in ms")
    cash: float = Field(..., description="Cash balance")
    positions: List[PositionHoldingItem] = Field(
        default_factory=list, description="List of position holdings"
    )
    total_value: Optional[float] = Field(
        None, description="Total portfolio value (cash + positions)"
    )
    total_unrealized_pnl: Optional[float] = Field(
        None, description="Sum of unrealized PnL across positions"
    )
    total_realized_pnl: Optional[float] = Field(
        None, description="Sum of realized PnL from closed positions"
    )
    gross_exposure: Optional[float] = Field(
        None, description="Aggregate gross exposure at snapshot"
    )
    net_exposure: Optional[float] = Field(
        None, description="Aggregate net exposure at snapshot"
    )
    available_cash: Optional[float] = Field(
        None, description="Cash available for new positions"
    )


StrategyHoldingResponse = SuccessResponse[StrategyHoldingData]


class StrategyPortfolioSummaryData(BaseModel):
    strategy_id: str = Field(..., description="Strategy identifier")
    ts: int = Field(..., description="Snapshot timestamp in ms")
    cash: Optional[float] = Field(None, description="Cash balance from snapshot")
    total_value: Optional[float] = Field(
        None, description="Total portfolio value (cash + positions)"
    )
    total_pnl: Optional[float] = Field(
        None,
        description="Combined realized and unrealized PnL for the snapshot",
    )
    total_pnl_pct: Optional[float] = Field(
        None, description="Total PnL percentage for the snapshot"
    )
    gross_exposure: Optional[float] = Field(
        None, description="Aggregate gross exposure at snapshot"
    )
    net_exposure: Optional[float] = Field(
        None, description="Aggregate net exposure at snapshot"
    )


StrategyPortfolioSummaryResponse = SuccessResponse[StrategyPortfolioSummaryData]


class StrategyActionCard(BaseModel):
    instruction_id: str = Field(..., description="Instruction identifier (NOT NULL)")
    symbol: str = Field(..., description="Instrument symbol")
    action: Optional[
        Literal["open_long", "open_short", "close_long", "close_short", "noop"]
    ] = Field(None, description="LLM action (includes noop)")
    action_display: Optional[str] = Field(
        None, description="Human-friendly action label for display, e.g. 'OPEN LONG'"
    )
    side: Optional[Literal["BUY", "SELL"]] = Field(
        None, description="Derived execution side"
    )
    quantity: Optional[float] = Field(None, description="Order quantity (units)")
    leverage: Optional[float] = Field(
        None, description="Leverage applied to the instruction (if any)"
    )
    avg_exec_price: Optional[float] = Field(
        None, description="Average execution price for fills"
    )
    entry_price: Optional[float] = Field(None, description="Entry price")
    exit_price: Optional[float] = Field(None, description="Exit price (if closed)")
    entry_at: Optional[datetime] = Field(None, description="Entry timestamp")
    exit_at: Optional[datetime] = Field(None, description="Exit timestamp")
    holding_time_ms: Optional[int] = Field(
        None, description="Holding time in milliseconds"
    )
    notional_entry: Optional[float] = Field(
        None, description="Entry notional in quote currency"
    )
    notional_exit: Optional[float] = Field(
        None, description="Exit notional in quote currency"
    )
    fee_cost: Optional[float] = Field(
        None, description="Total fees charged in quote currency"
    )
    realized_pnl: Optional[float] = Field(None, description="Realized PnL on close")
    realized_pnl_pct: Optional[float] = Field(
        None, description="Realized PnL percentage on close"
    )
    rationale: Optional[str] = Field(None, description="LLM rationale text")


class StrategyCycleDetail(BaseModel):
    compose_id: str = Field(..., description="Compose cycle identifier")
    cycle_index: int = Field(..., description="Cycle index (1-based)")
    created_at: datetime = Field(..., description="Compose datetime")
    rationale: Optional[str] = Field(None, description="LLM rationale text")
    actions: List[StrategyActionCard] = Field(
        default_factory=list, description="Instruction/action cards for this cycle"
    )


StrategyDetailResponse = SuccessResponse[List[StrategyCycleDetail]]


class StrategyHoldingFlatItem(BaseModel):
    symbol: str = Field(..., description="Instrument symbol")
    type: Literal["LONG", "SHORT"] = Field(
        ..., description="Trade type derived from position"
    )
    leverage: Optional[float] = Field(None, description="Leverage applied")
    entry_price: Optional[float] = Field(None, description="Average entry price")
    quantity: float = Field(..., description="Absolute position quantity")
    unrealized_pnl: Optional[float] = Field(None, description="Unrealized PnL value")
    unrealized_pnl_pct: Optional[float] = Field(
        None, description="Unrealized PnL percentage"
    )


# Response type for compact holdings array
StrategyHoldingFlatResponse = SuccessResponse[List[StrategyHoldingFlatItem]]


StrategyCurveResponse = SuccessResponse[List[List[str | float | None]]]


class StrategyStatusUpdateResponse(BaseModel):
    strategy_id: str = Field(..., description="Strategy identifier")
    status: Literal["running", "stopped"] = Field(
        ..., description="Updated strategy status"
    )
    message: str = Field(..., description="Status update message")


StrategyStatusSuccessResponse = SuccessResponse[StrategyStatusUpdateResponse]


# =====================
# Prompt Schemas (strategy namespace)
# =====================


class PromptItem(BaseModel):
    id: str = Field(..., description="Prompt UUID")
    name: str = Field(..., description="Prompt name")
    content: str = Field(..., description="Prompt content text")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Update timestamp")


class PromptCreateRequest(BaseModel):
    name: str = Field(..., description="Prompt name")
    content: str = Field(..., description="Prompt content text")


PromptListResponse = SuccessResponse[list[PromptItem]]
PromptCreateResponse = SuccessResponse[PromptItem]


class PromptDeleteResponse(BaseModel):
    deleted: bool = Field(
        ..., description="Whether the prompt was successfully deleted"
    )
    prompt_id: str = Field(..., description="ID of the deleted prompt")
    message: str = Field(..., description="Delete operation result message")


PromptDeleteSuccessResponse = SuccessResponse[PromptDeleteResponse]


class StrategyMarketDataHealth(BaseModel):
    ok: bool = Field(..., description="Whether the latest market data scan covered all expected symbols")
    provider: Optional[str] = Field(None, description="Market data provider/exchange")
    fetched_count: int = Field(0, description="Symbols with fresh market snapshot data")
    missing_count: int = Field(0, description="Expected symbols missing market snapshot data")
    missing_symbols: List[str] = Field(default_factory=list, description="Expected symbols missing data")

    status: str = Field("missing", description="Actionable health status: healthy, degraded, or missing")
    freshness_status: str = Field("missing", description="Snapshot freshness: fresh, stale, missing, or unknown")
    coverage_status: str = Field("missing", description="Symbol coverage: complete, partial, or missing")
    stale_count: int = Field(0, description="Expected symbols with stale realtime snapshots")
    stale_symbols: List[str] = Field(default_factory=list, description="Expected symbols with stale realtime snapshots")
    exposure_increase_allowed: bool = Field(False, description="Whether current data health permits opening or increasing exposure")

class StrategySymbolDecisionData(BaseModel):
    symbol: str = Field(..., description="Observed or expected symbol")
    intervals_seen: List[str] = Field(default_factory=list, description="Intervals observed in this scan")
    has_market_snapshot: bool = Field(False, description="Whether realtime market snapshot was fetched")
    latest_price: Optional[float] = Field(None, description="Latest observed price")
    snapshot_ts_ms: Optional[int] = Field(None, description="Source timestamp for the realtime market snapshot")
    freshness_age_ms: Optional[int] = Field(None, description="Snapshot age when the decision cycle was evaluated")
    freshness_status: str = Field("missing", description="Snapshot freshness: fresh, stale, missing, or unknown")
    coverage_status: str = Field("missing", description="Per-symbol snapshot coverage status")
    exposure_increase_allowed: bool = Field(False, description="Whether this symbol can open or increase exposure")
    action: Optional[str] = Field(None, description="Action emitted by the strategy or noop")
    quantity: Optional[float] = Field(None, description="Order quantity if an order was emitted")
    reason: Optional[str] = Field(None, description="Human-readable order/no-order reason")
    indicator_snapshot: dict[str, Any] = Field(default_factory=dict, description="Latest per-interval indicator values")
    conditions: List[dict[str, Any]] = Field(default_factory=list, description="Structured condition checks")
    decision_path: List[str] = Field(default_factory=list, description="Step-by-step decision path")
    fund_impact: dict[str, Any] = Field(default_factory=dict, description="Estimated and executed fund impact")


class StrategyDiagnosticsCycleData(BaseModel):
    compose_id: str = Field(..., description="Compose cycle identifier")
    cycle_index: Optional[int] = Field(None, description="Cycle index")
    created_at: Optional[datetime] = Field(None, description="Cycle creation time")
    rationale: Optional[str] = Field(None, description="Decision rationale")
    instruction_count: int = Field(0, description="Total instructions emitted")
    order_count: int = Field(0, description="Executable order count")
    no_order_count: int = Field(0, description="Symbols with no order")
    market_data_health: StrategyMarketDataHealth = Field(..., description="Market data fetch health")


class StrategyDiagnosticsData(BaseModel):
    strategy_id: str = Field(..., description="Strategy identifier")
    strategy_name: Optional[str] = Field(None, description="Strategy display name")
    status: Optional[str] = Field(None, description="Strategy runtime status")
    trading_mode: Optional[str] = Field(None, description="Trading mode")
    exchange_id: Optional[str] = Field(None, description="Exchange identifier")
    strategy_type: Optional[StrategyType] = Field(None, description="Strategy type")
    runtime_health: dict = Field(default_factory=dict, description="Runtime health summary")
    config: dict = Field(default_factory=dict, description="User-facing strategy configuration")
    explanation: dict[str, Any] = Field(default_factory=dict, description="Structured latest cycle explanation")
    observed_symbol_count: int = Field(0, description="Symbols with market snapshot data")
    expected_symbol_count: int = Field(0, description="Configured symbol count")
    latest_cycle: Optional[StrategyDiagnosticsCycleData] = Field(None, description="Latest scan diagnostics")
    symbol_decisions: List[StrategySymbolDecisionData] = Field(default_factory=list, description="Latest per-symbol decisions")
    recent_cycles: List[StrategyDiagnosticsCycleData] = Field(default_factory=list, description="Recent scan summaries")


StrategyDiagnosticsResponse = SuccessResponse[StrategyDiagnosticsData]


class StrategyPerformanceData(BaseModel):
    """Performance overview for a strategy including ROI and config."""

    strategy_id: str = Field(..., description="Strategy identifier")
    initial_capital: Optional[float] = Field(
        None, description="Initial capital used by the strategy"
    )
    return_rate_pct: Optional[float] = Field(
        None, description="Return rate percentage relative to initial capital"
    )
    # Flattened config fields (only the requested subset)
    llm_provider: Optional[str] = Field(
        None, description="Model provider (e.g., openrouter, google, openai)"
    )
    llm_model_id: Optional[str] = Field(
        None, description="Model identifier (e.g., deepseek-ai/deepseek-v3.1)"
    )
    exchange_id: Optional[str] = Field(None, description="Exchange identifier")
    strategy_type: Optional[StrategyType] = Field(
        None, description="Strategy type"
    )
    trading_mode: Optional[Literal["live", "virtual"]] = Field(
        None, description="Trading mode: live or virtual"
    )
    max_leverage: Optional[float] = Field(None, description="Maximum leverage")
    symbols: Optional[List[str]] = Field(None, description="Symbols universe")
    prompt_name: Optional[str] = Field(
        None, description="Prompt template name used by the strategy"
    )
    prompt: Optional[str] = Field(
        None, description="Final resolved prompt text used by the strategy"
    )


StrategyPerformanceResponse = SuccessResponse[StrategyPerformanceData]
