from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from valuecell.server.api.schemas.multi_strategy import (
    AccountStrategyOverview,
    CapitalAllocatorSummary,
    SharedWalletSummary,
    STRATEGY_DEFINITIONS,
    StrategyAllocation,
    StrategyDefinition,
)


UTC_NOW = datetime(2026, 8, 28, tzinfo=timezone.utc)


def test_registry_contains_one_configurable_and_three_fixed_strategies() -> None:
    assert [definition.kind for definition in STRATEGY_DEFINITIONS] == [
        "configurable_rule",
        "dual_ma_trend",
        "pair_rotation",
        "leader_breakout",
    ]
    assert [definition.editable for definition in STRATEGY_DEFINITIONS] == [
        True,
        False,
        False,
        False,
    ]


def test_fixed_strategy_cannot_be_marked_editable() -> None:
    with pytest.raises(ValidationError):
        StrategyDefinition(
            kind="dual_ma_trend",
            display_name="双均线趋势",
            description="固定规则",
            rule_source="双均线趋势策略_工程可执行版.txt",
            strategy_version="v1",
            parameter_source="code",
            editable=True,
            execution_environments=("paper",),
        )


def test_strategy_allocation_rejects_over_occupation() -> None:
    with pytest.raises(ValidationError):
        StrategyAllocation(
            strategy_id="strategy-1",
            kind="dual_ma_trend",
            reserved_quote=100,
            occupied_quote=101,
            released_quote=0,
            allocation_state="occupied",
            utilization_denominator_quote=1_000,
        )


def test_incomplete_account_overview_requires_reason() -> None:
    wallet = SharedWalletSummary(
        tenant_id="tenant-1",
        credential_id="credential-1",
        environment="okx_demo",
        observed_at=UTC_NOW,
        sync_status="healthy",
        attribution_status="partial",
    )
    allocator = CapitalAllocatorSummary(
        reserved_quote=0,
        occupied_notional_quote=0,
        pending_settlement_quote=0,
        utilization_denominator_quote=1_000,
        account_utilization_ratio=0,
        observed_at=UTC_NOW,
    )
    with pytest.raises(ValidationError):
        AccountStrategyOverview(
            wallet=wallet,
            allocator=allocator,
            data_complete=False,
        )
