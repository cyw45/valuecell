from decimal import Decimal

from valuecell.server.services.profit_share_service import calculate_profit_share


def test_high_water_mark_excludes_cash_flows_and_shares_only_new_profit():
    calculation = calculate_profit_share(
        high_water_mark_before=Decimal("1000"),
        ending_equity=Decimal("1350"),
        net_external_cash_flow=Decimal("200"),
        revenue_share_rate=Decimal("0.10"),
    )

    assert calculation.adjusted_high_water_mark == Decimal("1200")
    assert calculation.eligible_profit == Decimal("150")
    assert calculation.amount_due == Decimal("15.00")
    assert calculation.high_water_mark_after == Decimal("1350")


def test_high_water_mark_prevents_fee_on_recovery_after_loss():
    calculation = calculate_profit_share(
        high_water_mark_before=Decimal("1000"),
        ending_equity=Decimal("950"),
        net_external_cash_flow=Decimal("0"),
        revenue_share_rate=Decimal("0.10"),
    )

    assert calculation.eligible_profit == Decimal("0")
    assert calculation.amount_due == Decimal("0.00")
    assert calculation.high_water_mark_after == Decimal("1000")
