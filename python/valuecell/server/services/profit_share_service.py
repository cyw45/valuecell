"""Deterministic high-water-mark calculations for revenue-share agreements."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ProfitShareCalculation:
    """Reconciled profit amounts for one account settlement period."""

    high_water_mark_before: Decimal
    adjusted_high_water_mark: Decimal
    high_water_mark_after: Decimal
    eligible_profit: Decimal
    amount_due: Decimal


def calculate_profit_share(
    *,
    high_water_mark_before: Decimal,
    ending_equity: Decimal,
    net_external_cash_flow: Decimal,
    revenue_share_rate: Decimal,
) -> ProfitShareCalculation:
    """Apply external cash-flow adjustment before high-water-mark fee sharing."""

    adjusted_high_water_mark = high_water_mark_before + net_external_cash_flow
    eligible_profit = max(Decimal("0"), ending_equity - adjusted_high_water_mark)
    high_water_mark_after = max(adjusted_high_water_mark, ending_equity)
    return ProfitShareCalculation(
        high_water_mark_before=high_water_mark_before,
        adjusted_high_water_mark=adjusted_high_water_mark,
        high_water_mark_after=high_water_mark_after,
        eligible_profit=eligible_profit,
        amount_due=eligible_profit * revenue_share_rate,
    )
