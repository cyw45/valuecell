"""Deterministic monitor-pool admission and removal decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping

from valuecell.server.db.models.rule_strategy import RuleStrategyMonitorSymbol
from valuecell.server.db.repositories.rule_strategy_repository import RuleStrategyRepository


@dataclass(frozen=True, slots=True)
class StrategyMarketMetadata:
    """Provider facts required to admit a symbol into a strategy monitor pool."""

    listing_age_days: int | None
    average_quote_volume_30d: float | None
    price_quote: float | None
    provider: str | None = None
    listing_first_tradable_at: datetime | None = None
    price_observed_at: datetime | None = None

@dataclass(frozen=True, slots=True)
class MonitorDecision:
    """Stable state and reason written to the monitor read model."""

    state: str
    reason_code: str
    reason_detail: str
    consecutive_low_volume_days: int
    protected_held: bool


ADMISSION_REASON = "monitor_admission_criteria_met"
REMOVAL_REASON = "monitor_volume_below_removal_threshold"
METADATA_UNAVAILABLE_REASON = "market_metadata_unavailable"


def decide_monitor_state(
    row: RuleStrategyMonitorSymbol,
    metadata: StrategyMarketMetadata | None,
    *,
    position_quantity: float,
    minimum_listing_age_days: int = 60,
    minimum_average_quote_volume_30d: float = 5_000_000.0,
    minimum_price_quote: float = 1.0,
    removal_average_quote_volume_30d: float = 2_000_000.0,
    removal_consecutive_daily_checks: int = 7,
) -> MonitorDecision:
    """Apply V2.1 admission/removal rules without inferring missing provider facts."""
    if metadata is None or any(
        value is None
        for value in (
            metadata.listing_age_days,
            metadata.average_quote_volume_30d,
            metadata.price_quote,
        )
    ):
        return MonitorDecision(
            state="held" if position_quantity > 0 else "candidate",
            reason_code=METADATA_UNAVAILABLE_REASON,
            reason_detail="市场未提供上市年龄、30日成交额或价格，暂不准入。",
            consecutive_low_volume_days=row.consecutive_low_volume_days,
            protected_held=position_quantity > 0,
        )

    if position_quantity > 0:
        return MonitorDecision(
            state="held",
            reason_code="monitor_position_protected",
            reason_detail="标的仍有持仓，移除条件已保护保留。",
            consecutive_low_volume_days=row.consecutive_low_volume_days,
            protected_held=True,
        )

    assert metadata.average_quote_volume_30d is not None
    admitted = (
        metadata.listing_age_days >= minimum_listing_age_days
        and metadata.average_quote_volume_30d >= minimum_average_quote_volume_30d
        and metadata.price_quote >= minimum_price_quote
    )
    if admitted:
        return MonitorDecision(
            state="admitted",
            reason_code=ADMISSION_REASON,
            reason_detail="上市时间、30日成交额和价格均达到准入标准。",
            consecutive_low_volume_days=0,
            protected_held=False,
        )

    low_volume_days = (
        row.consecutive_low_volume_days + 1
        if metadata.average_quote_volume_30d < removal_average_quote_volume_30d
        else 0
    )
    if (
        metadata.average_quote_volume_30d < removal_average_quote_volume_30d
        and low_volume_days >= removal_consecutive_daily_checks
    ):
        return MonitorDecision(
            state="removed",
            reason_code=REMOVAL_REASON,
            reason_detail="30日平均成交额连续七次低于 2,000,000 USDT。",
            consecutive_low_volume_days=low_volume_days,
            protected_held=False,
        )
    return MonitorDecision(
        state="candidate",
        reason_code="monitor_admission_criteria_not_met",
        reason_detail="当前市场数据尚未满足全部准入条件。",
        consecutive_low_volume_days=low_volume_days,
        protected_held=False,
    )


class RuleStrategyMonitorAdmissionWorker:
    """Persist scheduled monitor reviews from an injected metadata provider."""

    def __init__(self, repository: RuleStrategyRepository) -> None:
        self.repository = repository

    def review(
        self,
        strategy_id: str,
        tenant_id: str,
        metadata_by_symbol: Mapping[str, StrategyMarketMetadata | None],
        positions_by_symbol: Mapping[str, float] | None = None,
        *,
        now: datetime | None = None,
        force: bool = False,
    ) -> list[RuleStrategyMonitorSymbol]:
        timestamp = now or datetime.now(timezone.utc)
        positions = positions_by_symbol or {}
        claimed = self.repository.claim_monitor_lease(
            strategy_id,
            tenant_id,
            f"monitor-worker-{strategy_id}",
            now=timestamp,
            force=force,
        )
        updated: list[RuleStrategyMonitorSymbol] = []
        for row in claimed:
            symbol = row.symbol.upper().replace("/", "-")
            metadata = metadata_by_symbol.get(symbol)
            decision = decide_monitor_state(
                row,
                metadata,
                position_quantity=float(positions.get(symbol, 0.0)),
            )
            saved = self.repository.update_monitor_state(
                row.id,
                tenant_id,
                state=decision.state,
                reason_code=decision.reason_code,
                reason_detail=decision.reason_detail,
                evaluated_at=timestamp,
                next_check_at=timestamp + timedelta(days=1),
                protected_held=decision.protected_held,
                consecutive_low_volume_days=decision.consecutive_low_volume_days,
                metadata_provider=None if metadata is None else metadata.provider,
                listing_first_tradable_at=(
                    None if metadata is None else metadata.listing_first_tradable_at
                ),
                listing_age_days=None if metadata is None else metadata.listing_age_days,
                average_quote_volume_30d=(
                    None if metadata is None else metadata.average_quote_volume_30d
                ),
                price_quote=None if metadata is None else metadata.price_quote,
                price_observed_at=(
                    None if metadata is None else metadata.price_observed_at
                ),
            )
            if saved is not None:
                updated.append(saved)
        return updated
