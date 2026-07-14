import pytest
from pydantic import ValidationError

from valuecell.server.api.schemas.rule_strategy import RuleStrategyEvaluationRequest
from valuecell.server.services.rule_engine import RuleEngine


BASE_TIMESTAMP_MS = 1_700_000_000_000


def _candles(closes: list[float]) -> list[dict[str, float | int]]:
    return [
        {
            "timestamp_ms": BASE_TIMESTAMP_MS + index * 60_000,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0 + index,
        }
        for index, close in enumerate(closes)
    ]


def _request(
    closes: list[float],
    *,
    config: dict | None = None,
    market: dict | None = None,
) -> RuleStrategyEvaluationRequest:
    market_data = {
        "symbol": "BTC-USDT",
        "price": closes[-1],
        "equity_quote": 1_000.0,
        "quote_balance": 1_000.0,
    }
    if market:
        market_data.update(market)
    return RuleStrategyEvaluationRequest.model_validate(
        {
            "config": config or {},
            "candles": _candles(closes),
            "market": market_data,
        }
    )


def _evaluate(
    closes: list[float],
    *,
    config: dict | None = None,
    market: dict | None = None,
):
    return RuleEngine().evaluate(_request(closes, config=config, market=market))


def _condition(result, code: str):
    return next(condition for condition in result.conditions if condition.code == code)


@pytest.mark.parametrize(
    ("closes", "position", "expected_action", "expected_reason"),
    [
        ([10.0, 10.0, 10.0, 11.0], None, "buy", "indicator_buy_confirmed"),
        (
            [10.0, 10.0, 10.0, 9.0],
            {"quantity": 2.0, "entry_price": 10.0},
            "sell",
            "indicator_sell_confirmed",
        ),
    ],
)
def test_moving_average_crossovers_produce_long_entry_and_exit(
    closes, position, expected_action, expected_reason
):
    market = {"position": position} if position else None
    result = _evaluate(
        closes,
        config={
            "moving_average": {"enabled": True, "short_window": 2, "long_window": 3}
        },
        market=market,
    )

    assert result.action == expected_action
    assert result.reason_code == expected_reason
    crossover = _condition(result, "ma_crossover")
    assert crossover.state == "triggered"
    assert crossover.values["previous_moving_average_short"] == pytest.approx(10.0)
    assert crossover.values["previous_moving_average_long"] == pytest.approx(10.0)
    if expected_action == "buy":
        assert result.indicators.moving_average_short > result.indicators.moving_average_long
    else:
        assert result.indicators.moving_average_short < result.indicators.moving_average_long


@pytest.mark.parametrize(
    ("name", "closes", "indicator_config", "expected_action", "expected_code"),
    [
        (
            "rsi_oversold",
            [10.0, 9.0, 8.0],
            {"rsi": {"enabled": True, "period": 2}},
            "buy",
            "rsi",
        ),
        (
            "bollinger_lower_band",
            [10.0, 10.0, 10.0, 8.0],
            {
                "bollinger": {
                    "enabled": True,
                    "period": 4,
                    "standard_deviations": 1.0,
                }
            },
            "buy",
            "bollinger",
        ),
        (
            "momentum_macd_upward_crossover",
            [10.0, 10.0, 9.0, 10.0],
            {
                "momentum_macd": {
                    "enabled": True,
                    "momentum_period": 1,
                    "macd_fast_window": 1,
                    "macd_slow_window": 2,
                    "macd_signal_window": 2,
                }
            },
            "buy",
            "momentum_macd",
        ),
        (
            "momentum_macd_downward_crossover",
            [10.0, 10.0, 11.0, 10.0],
            {
                "momentum_macd": {
                    "enabled": True,
                    "momentum_period": 1,
                    "macd_fast_window": 1,
                    "macd_slow_window": 2,
                    "macd_signal_window": 2,
                }
            },
            "sell",
            "momentum_macd",
        ),
    ],
)
def test_indicator_rules_evaluate_explicit_candles(
    name, closes, indicator_config, expected_action, expected_code
):
    market = (
        {"position": {"quantity": 1.0, "entry_price": 10.0}}
        if expected_action == "sell"
        else None
    )
    result = _evaluate(
        closes,
        config={"confirmation_mode": "any", **indicator_config},
        market=market,
    )

    assert result.action == expected_action, name
    assert _condition(result, expected_code).state == "triggered"
    if expected_code == "rsi":
        assert result.indicators.rsi == pytest.approx(0.0)
    if expected_code == "bollinger":
        assert result.indicators.bollinger_lower > closes[-1]
    if expected_code == "momentum_macd":
        assert result.indicators.momentum is not None
        assert result.indicators.macd is not None
        assert result.indicators.macd_signal is not None


def test_neutral_and_unavailable_indicators_return_explainable_no_ops():
    neutral = _evaluate(
        [10.0, 11.0, 10.0],
        config={"rsi": {"enabled": True, "period": 2}},
    )
    unavailable = _evaluate(
        [10.0, 10.0, 10.0],
        config={
            "moving_average": {"enabled": True, "short_window": 2, "long_window": 3}
        },
    )

    assert neutral.action == "no_op"
    assert neutral.reason_code == "indicators_not_confirmed"
    assert _condition(neutral, "rsi").state == "not_triggered"
    assert unavailable.action == "no_op"
    assert unavailable.reason_code == "insufficient_candle_history"
    history = _condition(unavailable, "ma_crossover")
    assert history.state == "unavailable"
    assert history.values == {"required_candles": 4, "supplied_candles": 3}


def test_flat_bollinger_window_returns_explainable_no_op():
    result = _evaluate(
        [10.0, 10.0, 10.0, 10.0],
        config={
            "bollinger": {
                "enabled": True,
                "period": 4,
                "standard_deviations": 1.0,
            }
        },
    )

    assert result.action == "no_op"
    assert result.reason_code == "indicators_not_confirmed"
    bollinger = _condition(result, "bollinger")
    assert bollinger.state == "not_triggered"
    assert bollinger.detail == "Bollinger bands have zero width"
    assert bollinger.values == {
        "bollinger_upper": 10.0,
        "bollinger_middle": 10.0,
        "bollinger_lower": 10.0,
    }


@pytest.mark.parametrize(
    ("price", "risk", "expected_reason", "expected_exit"),
    [
        (110.0, {"take_profit_pct": 0.05}, "take_profit_triggered", "take_profit"),
        (90.0, {"stop_loss_pct": 0.05}, "stop_loss_triggered", "stop_loss"),
    ],
)
def test_risk_exits_sell_open_positions_before_indicator_confirmation(
    price, risk, expected_reason, expected_exit
):
    result = _evaluate(
        [100.0, 100.0],
        config={"risk": risk},
        market={
            "price": price,
            "position": {"quantity": 2.0, "entry_price": 100.0},
        },
    )

    assert result.action == "sell"
    assert result.reason_code == expected_reason
    assert _condition(result, expected_exit).state == "triggered"
    other_exit = "stop_loss" if expected_exit == "take_profit" else "take_profit"
    assert _condition(result, other_exit).state == "not_triggered"


@pytest.mark.parametrize(
    ("market", "risk", "expected_block"),
    [
        ({"open_position_count": 1}, {"max_positions": 1}, "max_positions"),
        ({"quote_balance": 50.0}, {}, "available_collateral"),
        ({"equity_quote": 50.0}, {}, "leverage_limit"),
    ],
)
def test_buy_signals_are_blocked_by_position_collateral_and_leverage_limits(
    market, risk, expected_block
):
    result = _evaluate(
        [10.0, 9.0, 8.0],
        config={"rsi": {"enabled": True, "period": 2}, "risk": risk},
        market=market,
    )

    assert result.action == "no_op"
    assert result.reason_code == expected_block
    assert _condition(result, "rsi").state == "triggered"
    assert _condition(result, expected_block).state == "blocked"




def test_contract_rejects_equity_fraction_size_above_one():
    with pytest.raises(ValidationError):
        _request(
            [10.0],
            config={
                "risk": {"size_mode": "equity_fraction", "size_value": 1.01}
            },
        )

@pytest.mark.parametrize(
    "payload",
    [
        {
            "config": {},
            "candles": [
                {
                    "timestamp_ms": BASE_TIMESTAMP_MS,
                    "open": 10.0,
                    "high": 9.0,
                    "low": 8.0,
                    "close": 10.0,
                    "volume": 1.0,
                }
            ],
            "market": {
                "symbol": "BTC-USDT",
                "price": 10.0,
                "equity_quote": 1_000.0,
                "quote_balance": 1_000.0,
            },
        },
        {
            "config": {},
            "candles": [
                {
                    "timestamp_ms": BASE_TIMESTAMP_MS,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0,
                    "volume": 1.0,
                },
                {
                    "timestamp_ms": BASE_TIMESTAMP_MS,
                    "open": 11.0,
                    "high": 12.0,
                    "low": 10.0,
                    "close": 11.0,
                    "volume": 1.0,
                },
            ],
            "market": {
                "symbol": "BTC-USDT",
                "price": 11.0,
                "equity_quote": 1_000.0,
                "quote_balance": 1_000.0,
            },
        },
        {
            "config": {
                "moving_average": {
                    "enabled": True,
                    "short_window": 3,
                    "long_window": 3,
                }
            },
            "candles": _candles([10.0]),
            "market": {
                "symbol": "BTC-USDT",
                "price": 10.0,
                "equity_quote": 1_000.0,
                "quote_balance": 1_000.0,
            },
        },
    ],
)
def test_contract_rejects_malformed_candles_and_invalid_rule_config(payload):
    with pytest.raises(ValidationError):
        RuleStrategyEvaluationRequest.model_validate(payload)

def test_contract_rejects_non_finite_candle_close():
    payload = {
        "config": {},
        "candles": _candles([10.0]),
        "market": {
            "symbol": "BTC-USDT",
            "price": 10.0,
            "equity_quote": 1_000.0,
            "quote_balance": 1_000.0,
        },
    }
    payload["candles"][0]["close"] = float("nan")

    with pytest.raises(ValidationError):
        RuleStrategyEvaluationRequest.model_validate(payload)


def test_result_explains_conditions_sizing_and_projected_funding():
    result = _evaluate(
        [10.0, 9.0, 8.0],
        config={
            "rsi": {"enabled": True, "period": 2},
            "risk": {"size_mode": "fixed_quote", "size_value": 100.0},
        },
        market={"funding_rate": 0.01},
    )

    assert result.action == "buy"
    assert {condition.code for condition in result.conditions} == {
        "rsi",
        "take_profit",
        "stop_loss",
        "max_positions",
        "available_collateral",
        "leverage_limit",
    }
    assert result.sizing.requested_quote == pytest.approx(100.0)
    assert result.sizing.quantity == pytest.approx(12.5)
    assert result.funding.current_notional_quote == pytest.approx(0.0)
    assert result.funding.projected_notional_quote == pytest.approx(100.0)
    assert result.funding.estimated_payment_quote == pytest.approx(-1.0)
    assert result.funding.direction == "debit"


def test_advanced_rules_combine_multiple_intervals_and_brar_for_entry():
    daily = _candles([10.0] * 19 + [20.0])
    macd = _candles([10.0, 10.0, 9.0, 10.0])
    fifteen_minute = _candles([10.0] * 17 + [100.0, 90.0, 80.0])
    request = RuleStrategyEvaluationRequest.model_validate(
        {
            "config": {
                "advanced_rules": {
                    "enabled": True,
                    "entry_confirmation_mode": "all",
                    "exit_confirmation_mode": "any",
                    "moving_average": {
                        "enabled": True,
                        "interval": "1d",
                        "period": 20,
                        "entry_comparator": "above",
                    },
                    "macd": {
                        "enabled": True,
                        "interval": "5m",
                        "fast_window": 1,
                        "slow_window": 2,
                        "signal_window": 2,
                        "entry_cross": "golden",
                    },
                    "bollinger": {
                        "enabled": True,
                        "interval": "15m",
                        "period": 20,
                        "standard_deviations": 2,
                        "entry_reference": "middle",
                        "entry_comparator": "above",
                    },
                    "rsi": {
                        "enabled": True,
                        "interval": "15m",
                        "period": 2,
                        "entry_comparator": "below",
                        "entry_threshold": 20,
                        "exit_enabled": True,
                        "exit_comparator": "above",
                        "exit_threshold": 85,
                    },
                    "momentum": {
                        "enabled": True,
                        "interval": "15m",
                        "period": 14,
                        "entry_comparator": "below",
                        "entry_threshold": 100,
                        "exit_enabled": True,
                        "exit_comparator": "above",
                        "exit_threshold": 100,
                    },
                    "brar": {
                        "enabled": True,
                        "interval": "15m",
                        "period": 2,
                        "component": "br",
                        "entry_comparator": "below",
                        "entry_threshold": 1_000,
                        "exit_enabled": False,
                    },
                }
            },
            "candles": fifteen_minute,
            "candle_sets": {"1d": daily, "5m": macd, "15m": fifteen_minute},
            "market": {
                "symbol": "BTC-USDT",
                "price": 80.0,
                "equity_quote": 1_000.0,
                "quote_balance": 1_000.0,
            },
        }
    )

    result = RuleEngine().evaluate(request)

    assert result.action == "buy"
    assert result.reason_code == "advanced_entry_confirmed"
    assert {condition.code for condition in result.conditions} >= {
        "price_ma",
        "macd_cross",
        "bollinger_price",
        "rsi_entry",
        "momentum_entry",
        "brar_entry",
    }
    assert result.indicators.brar_br is not None
