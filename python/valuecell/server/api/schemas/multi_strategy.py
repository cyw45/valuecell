"""Phase A contracts for concurrent strategy execution on one OKX wallet.

These contracts are additive. They do not alter the existing configurable rule
strategy engine or perform venue I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

StrategyKind = Literal[
    "configurable_rule",
    "dual_ma_trend",
    "pair_rotation",
    "leader_breakout",
]
StrategyParameterSource = Literal["configurable", "code"]
StrategyExecutionEnvironment = Literal["paper", "okx_demo"]
StrategyStatus = Literal["running", "stopped", "archived", "paused"]
AllocationState = Literal[
    "available",
    "reserved",
    "occupied",
    "partially_released",
    "released",
    "blocked",
]
TradeFactSide = Literal["buy", "sell", "short", "cover"]
TradeFactStatus = Literal[
    "signal",
    "blocked",
    "pending",
    "submitted",
    "partially_filled",
    "filled",
    "cancelled",
    "failed",
]


class MultiStrategyModel(BaseModel):
    """Strict wire model for shared strategy/account read models."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class StrategyDefinition(MultiStrategyModel):
    """Stable metadata for configurable and code-owned strategies."""

    kind: StrategyKind
    display_name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=2_000)
    rule_source: str = Field(min_length=1, max_length=255)
    strategy_version: str = Field(min_length=1, max_length=64)
    parameter_source: StrategyParameterSource
    editable: bool
    execution_environments: tuple[StrategyExecutionEnvironment, ...] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_editability(self) -> "StrategyDefinition":
        is_configurable = self.kind == "configurable_rule"
        if self.parameter_source == "configurable" and not is_configurable:
            raise ValueError("only configurable_rule may expose configurable parameters")
        if self.editable != is_configurable:
            raise ValueError("editable must match the parameter source")
        return self


STRATEGY_DEFINITIONS: tuple[StrategyDefinition, ...] = (
    StrategyDefinition(
        kind="configurable_rule",
        display_name="参数策略",
        description="现有可配置参数策略，保持当前执行行为。",
        rule_source="existing_rule_strategy",
        strategy_version="existing",
        parameter_source="configurable",
        editable=True,
        execution_environments=("paper", "okx_demo"),
    ),
    StrategyDefinition(
        kind="dual_ma_trend",
        display_name="双均线趋势",
        description="代码固定的双均线趋势策略。",
        rule_source="双均线趋势策略_工程可执行版.txt",
        strategy_version="v1",
        parameter_source="code",
        editable=False,
        execution_environments=("paper", "okx_demo"),
    ),
    StrategyDefinition(
        kind="pair_rotation",
        display_name="配对套利",
        description="代码固定的单腿轮换配对策略。",
        rule_source="配对套利策略_工程可执行版.txt",
        strategy_version="v1",
        parameter_source="code",
        editable=False,
        execution_environments=("paper", "okx_demo"),
    ),
    StrategyDefinition(
        kind="leader_breakout",
        display_name="现货龙头",
        description="代码固定的现货龙头突破策略。",
        rule_source="龙头策略_工程可执行版.txt",
        strategy_version="v1",
        parameter_source="code",
        editable=False,
        execution_environments=("paper", "okx_demo"),
    ),
)


class StrategyIdentity(MultiStrategyModel):
    """Identity carried by every strategy-attributed fact."""

    strategy_id: str = Field(min_length=1, max_length=100)
    tenant_id: str = Field(min_length=1, max_length=36)
    kind: StrategyKind
    strategy_version: str = Field(min_length=1, max_length=64)
    code_fingerprint: str = Field(min_length=1, max_length=128)


class StrategyAllocation(MultiStrategyModel):
    """One strategy's reservation and occupation in a shared wallet."""

    strategy_id: str = Field(min_length=1, max_length=100)
    kind: StrategyKind
    reserved_quote: float = Field(ge=0)
    occupied_quote: float = Field(ge=0)
    released_quote: float = Field(ge=0)
    realized_pnl_quote: float | None = None
    unrealized_pnl_quote: float | None = None
    net_pnl_quote: float | None = None
    allocation_state: AllocationState
    utilization_denominator_quote: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_occupation(self) -> "StrategyAllocation":
        if self.occupied_quote > self.reserved_quote:
            raise ValueError("occupied quote cannot exceed reserved quote")
        return self


class SharedWalletSummary(MultiStrategyModel):
    """OKX-authoritative wallet facts, not strategy attribution."""

    tenant_id: str = Field(min_length=1, max_length=36)
    credential_id: str = Field(min_length=1, max_length=36)
    environment: StrategyExecutionEnvironment
    total_equity_quote: float | None = None
    available_quote: float | None = None
    currency_balances: dict[str, float | None] = Field(default_factory=dict)
    observed_at: datetime
    sync_status: Literal["healthy", "stale", "unavailable"]
    attribution_status: Literal["complete", "partial", "unavailable"]
    unassigned_equity_quote: float | None = None


class CapitalAllocatorSummary(MultiStrategyModel):
    """Account-level funds available to concurrent strategies."""

    wallet_equity_quote: float | None = None
    available_for_strategies_quote: float | None = None
    reserved_quote: float = Field(ge=0)
    occupied_notional_quote: float = Field(ge=0)
    pending_settlement_quote: float = Field(ge=0)
    reusable_quote: float | None = None
    utilization_denominator_quote: float = Field(gt=0)
    account_utilization_ratio: float = Field(ge=0)
    allocations: list[StrategyAllocation] = Field(default_factory=list)
    observed_at: datetime


class AccountStrategyOverview(MultiStrategyModel):
    """Wallet facts plus strategy attribution and reconciliation."""

    wallet: SharedWalletSummary
    allocator: CapitalAllocatorSummary
    strategy_pnl_total_quote: float | None = None
    wallet_strategy_reconciliation_delta_quote: float | None = None
    data_complete: bool
    incomplete_reason: str | None = None

    @model_validator(mode="after")
    def validate_completeness_reason(self) -> "AccountStrategyOverview":
        if not self.data_complete and not self.incomplete_reason:
            raise ValueError("incomplete account overviews require a reason")
        return self


class ExplanationCondition(MultiStrategyModel):
    """Persisted condition facts displayed beside a trade decision."""

    code: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=255)
    state: Literal["triggered", "not_triggered", "blocked", "unavailable"]
    actual: float | str | bool | None = None
    threshold: float | str | bool | None = None
    operator: str | None = Field(default=None, max_length=16)
    detail: str = Field(min_length=1, max_length=1_000)
    data_at: datetime | None = None


class TradeExplanation(MultiStrategyModel):
    """Durable explanation for a signal, order, or blocked decision."""

    decision: str = Field(min_length=1, max_length=64)
    decision_reason: str = Field(min_length=1, max_length=2_000)
    conditions: list[ExplanationCondition] = Field(default_factory=list)
    execution_path: str | None = Field(default=None, max_length=255)
    risk_check: str | None = Field(default=None, max_length=1_000)
    block_reason: str | None = Field(default=None, max_length=1_000)
    final_result: str | None = Field(default=None, max_length=255)


class UnifiedTradeFact(MultiStrategyModel):
    """Cross-strategy trade fact consumed by Web and Mobile."""

    identity: StrategyIdentity
    batch_id: str | None = Field(default=None, max_length=36)
    evaluation_id: str | None = Field(default=None, max_length=100)
    intent_id: str | None = Field(default=None, max_length=36)
    order_id: str | None = Field(default=None, max_length=128)
    fill_id: str | None = Field(default=None, max_length=36)
    symbol: str = Field(min_length=1, max_length=32)
    pair: str | None = Field(default=None, max_length=64)
    side: TradeFactSide
    status: TradeFactStatus
    requested_quote: float | None = None
    filled_quote: float | None = None
    requested_quantity: float | None = None
    filled_quantity: float | None = None
    average_fill_price: float | None = None
    fee_quote: float | None = None
    execution_cost_quote: float | None = None
    borrow_cost_quote: float | None = None
    created_at: datetime
    filled_at: datetime | None = None
    failure_code: str | None = Field(default=None, max_length=96)
    failure_reason: str | None = Field(default=None, max_length=2_000)
    explanation: TradeExplanation
