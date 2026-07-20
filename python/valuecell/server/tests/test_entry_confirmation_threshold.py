import pytest
from pydantic import ValidationError

from valuecell.server.api.schemas.rule_strategy import RuleStrategyEvaluationRequest
from valuecell.server.services.rule_engine import RuleEngine


def candles(closes):
    return [{"timestamp_ms": 1_700_000_000_000 + i * 60_000, "open": x,
             "high": x + 1, "low": x - 1, "close": x, "volume": 1}
            for i, x in enumerate(closes)]


def evaluate(advanced_rules):
    request = RuleStrategyEvaluationRequest.model_validate({
        "config": {"interval": "15m", "advanced_rules": advanced_rules},
        "candles": candles([10.0, 11.0, 12.0]),
        "market": {"symbol": "BTC-USDT", "price": 12, "equity_quote": 1000,
                   "quote_balance": 1000},
    })
    return RuleEngine().evaluate(request)


@pytest.mark.parametrize(("mode", "count", "ratio", "required"), [
    ("any", None, None, 1), ("all", None, None, 3),
    ("at_least", 2, None, 2), ("ratio", None, 0.5, 2),
])
def test_entry_confirmation_modes_and_summary(mode, count, ratio, required):
    rules = {
        "enabled": True, "entry_confirmation_mode": mode,
        "moving_average": {"enabled": True, "interval": "15m", "period": 2},
        "rsi": {"enabled": True, "interval": "15m", "period": 2, "entry_comparator": "below",
                "entry_threshold": 100},
        "momentum": {"enabled": True, "interval": "15m", "period": 2, "entry_comparator": "above",
                     "entry_threshold": 999},
    }
    if count is not None:
        rules["entry_confirmation_count"] = count
    if ratio is not None:
        rules["entry_confirmation_ratio"] = ratio
    result = evaluate(rules)
    assert result.entry_confirmation.model_dump() == {
        "enabled": 3, "available": 3, "passed": 2,
        "required": required, "mode": mode,
    }
    assert result.action == ("buy" if required <= 2 else "no_op")


def test_ratio_uses_ceiling_for_six_conditions():
    result = evaluate({
        "enabled": True, "entry_confirmation_mode": "ratio",
        "entry_confirmation_ratio": 0.34,
        "moving_average": {"enabled": True, "period": 2},
        "macd": {"enabled": True, "fast_window": 1, "slow_window": 2,
                 "signal_window": 1},
        "bollinger": {"enabled": True, "period": 2},
        "rsi": {"enabled": True, "period": 2},
        "momentum": {"enabled": True, "period": 2},
        "brar": {"enabled": True, "period": 2},
    })
    assert result.entry_confirmation.enabled == 6
    assert result.entry_confirmation.required == 3


def test_unavailable_does_not_pass_and_insufficient_availability_is_reported():
    result = evaluate({
        "enabled": True, "entry_confirmation_mode": "at_least",
        "entry_confirmation_count": 2,
        "moving_average": {"enabled": True, "interval": "15m", "period": 2},
        "macd": {"enabled": True, "interval": "5m", "slow_window": 26},
    })
    assert result.reason_code == "insufficient_candle_history"
    assert result.entry_confirmation.model_dump() == {
        "enabled": 2, "available": 1, "passed": 1,
        "required": 2, "mode": "at_least",
    }


@pytest.mark.parametrize("ratio", [0, -0.1, 1.01])
def test_ratio_rejects_out_of_range_values(ratio):
    with pytest.raises(ValidationError):
        evaluate({"entry_confirmation_mode": "ratio",
                  "entry_confirmation_ratio": ratio})


def test_at_least_rejects_count_above_enabled_entry_conditions():
    with pytest.raises(ValidationError, match="cannot exceed enabled entry conditions"):
        evaluate({
            "enabled": True,
            "entry_confirmation_mode": "at_least",
            "entry_confirmation_count": 2,
            "rsi": {"enabled": True},
        })
