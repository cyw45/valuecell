"""Shared configuration for spot RSI ladder strategies."""

from __future__ import annotations

from dataclasses import dataclass, replace

from valuecell.agents.common.trading.models import CandleConfig

DEFAULT_SPOT_SYMBOLS: tuple[str, ...] = (
    "BTC-USDT",
    "ETH-USDT",
    "BNB-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "ADA-USDT",
    "DOGE-USDT",
    "DOT-USDT",
    "USDC-USDT",
    "LTC-USDT",
    "BCH-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "MATIC-USDT",
    "UNI-USDT",
    "ATOM-USDT",
    "ETC-USDT",
    "FIL-USDT",
    "AAVE-USDT",
    "SAND-USDT",
    "MANA-USDT",
    "ALGO-USDT",
    "FTM-USDT",
    "NEAR-USDT",
    "GRT-USDT",
    "CAKE-USDT",
    "XLM-USDT",
    "EOS-USDT",
    "TRX-USDT",
    "WBTC-USDT",
    "ARB-USDT",
    "OP-USDT",
    "MKR-USDT",
    "SNX-USDT",
    "CRV-USDT",
    "1INCH-USDT",
    "KAVA-USDT",
    "ZRX-USDT",
    "BAT-USDT",
    "OMG-USDT",
    "QTUM-USDT",
    "ICX-USDT",
    "VET-USDT",
    "THETA-USDT",
    "NEO-USDT",
    "ONT-USDT",
    "ZIL-USDT",
    "RVN-USDT",
    "XZC-USDT",
    "DASH-USDT",
    "HBAR-USDT",
    "IOTA-USDT",
    "WAVES-USDT",
    "KSM-USDT",
    "RSR-USDT",
    "CELR-USDT",
    "FET-USDT",
    "OCEAN-USDT",
    "REQ-USDT",
    "BNT-USDT",
    "LRC-USDT",
    "GNO-USDT",
    "PAXG-USDT",
    "UMA-USDT",
    "BAL-USDT",
    "MIR-USDT",
    "SPELL-USDT",
    "AUDIO-USDT",
    "RAY-USDT",
    "SRM-USDT",
    "FIDA-USDT",
    "DEXE-USDT",
    "CELO-USDT",
    "LUNA-USDT",
    "MASK-USDT",
    "COTI-USDT",
    "CHZ-USDT",
    "ENJ-USDT",
    "FUN-USDT",
    "GAS-USDT",
    "HOT-USDT",
    "IOST-USDT",
    "KEY-USDT",
    "LOKA-USDT",
    "MBL-USDT",
    "NKN-USDT",
    "OAX-USDT",
    "PNT-USDT",
    "RIF-USDT",
    "SXP-USDT",
)

ADD_BUY_RATIO: float = 0.10
TAIL_DRAWDOWN_RATIO: float = 0.20
DAILY_CIRCUIT_BREAKER_RATIO: float = 0.15

RELAXED_ENTRY_RSI_THRESHOLDS: tuple[int, ...] = (32, 28, 24)
RELAXED_ENTRY_BUY_RATIOS: dict[int, float] = {
    32: 0.30,
    28: 0.40,
    24: 0.30,
}

LONG_ENTRY_RSI_THRESHOLDS: tuple[int, ...] = (28, 24, 20)
LONG_ENTRY_BUY_RATIOS: dict[int, float] = {
    28: 0.30,
    24: 0.40,
    20: 0.30,
}
LONG_SELL_RSI_THRESHOLDS: tuple[int, ...] = (70, 75, 80, 85)
LONG_SELL_CUMULATIVE_RATIOS: dict[int, float] = {
    70: 0.10,
    75: 0.50,
    80: 0.90,
    85: 1.00,
}

SHORT_ENTRY_RSI_THRESHOLDS: tuple[int, ...] = RELAXED_ENTRY_RSI_THRESHOLDS
SHORT_ENTRY_BUY_RATIOS: dict[int, float] = {
    **RELAXED_ENTRY_BUY_RATIOS,
}
SHORT_SELL_RSI_THRESHOLDS: tuple[int, ...] = (62, 68, 73, 78)
SHORT_SELL_CUMULATIVE_RATIOS: dict[int, float] = {
    62: 0.10,
    68: 0.50,
    73: 0.90,
    78: 1.00,
}

LONG_TERM_INTERVALS: tuple[CandleConfig, ...] = (
    CandleConfig(interval="1m", lookback=80),
    CandleConfig(interval="3m", lookback=80),
    CandleConfig(interval="5m", lookback=80),
    CandleConfig(interval="15m", lookback=80),
    CandleConfig(interval="30m", lookback=80),
    CandleConfig(interval="1h", lookback=80),
    CandleConfig(interval="4h", lookback=80),
    CandleConfig(interval="1d", lookback=90),
)

SHORT_TERM_INTERVALS: tuple[CandleConfig, ...] = (
    CandleConfig(interval="1m", lookback=80),
    CandleConfig(interval="3m", lookback=80),
    CandleConfig(interval="5m", lookback=80),
    CandleConfig(interval="15m", lookback=80),
    CandleConfig(interval="30m", lookback=80),
    CandleConfig(interval="1h", lookback=80),
    CandleConfig(interval="4h", lookback=80),
    CandleConfig(interval="1d", lookback=90),
)


@dataclass(frozen=True)
class SpotRsiStrategyProfile:
    """Static rule-set definition for one strategy variant."""

    strategy_type: str
    agent_name: str
    display_name: str
    capital_fraction: float
    primary_interval: str
    default_decide_interval: int
    ma_field: str
    max_additions: int
    entry_confirm_intervals: tuple[str, ...]
    trend_confirm_intervals: tuple[str, ...]
    candle_configurations: tuple[CandleConfig, ...]
    allow_entries_in_bear: bool
    bear_cap_ratio: float
    entry_rsi_thresholds: tuple[int, ...]
    entry_buy_ratios: dict[int, float]
    sell_rsi_thresholds: tuple[int, ...]
    sell_cumulative_ratios: dict[int, float]
    reset_exit_rsi: int
    require_bollinger_squeeze: bool
    require_bollinger_lower_touch: bool
    require_mtm_turn_up: bool
    require_mtm_below_zero: bool
    add_requires_trend_up: bool
    add_requires_no_bear_trend_4h: bool
    use_short_dual_mode: bool
    daily_overbought_rsi: float
    strong_entry_intervals: tuple[str, ...]
    strong_entry_min_confirmations: int


PROFILE_OVERRIDE_FIELDS = {
    "primary_interval",
    "bear_cap_ratio",
    "entry_rsi_thresholds",
    "sell_rsi_thresholds",
    "daily_overbought_rsi",
    "max_additions",
}


def profile_with_overrides(
    profile: SpotRsiStrategyProfile,
    params: dict[str, object] | None,
) -> SpotRsiStrategyProfile:
    """Return a strategy profile with schema-approved runtime overrides."""
    if not params:
        return profile

    updates: dict[str, object] = {}
    for key in PROFILE_OVERRIDE_FIELDS:
        if key not in params:
            continue
        value = params[key]
        if key in {"entry_rsi_thresholds", "sell_rsi_thresholds"}:
            if not isinstance(value, list):
                continue
            thresholds = tuple(int(item) for item in value if item is not None)
            if thresholds:
                updates[key] = thresholds
                if key == "entry_rsi_thresholds":
                    ratio = 1.0 / len(thresholds)
                    updates["entry_buy_ratios"] = {
                        threshold: ratio for threshold in thresholds
                    }
                else:
                    step = 1.0 / len(thresholds)
                    updates["sell_cumulative_ratios"] = {
                        threshold: min(1.0, step * (index + 1))
                        for index, threshold in enumerate(thresholds)
                    }
        elif key in {"bear_cap_ratio", "daily_overbought_rsi"}:
            updates[key] = float(value)
        elif key == "max_additions":
            updates[key] = int(value)
        elif key == "primary_interval":
            updates[key] = str(value)

    if not updates:
        return profile
    return replace(profile, **updates)


LONG_TERM_PROFILE = SpotRsiStrategyProfile(
    strategy_type="LongTermSpotRsiStrategy",
    agent_name="LongTermSpotRsiStrategyAgent",
    display_name="Long-Term Spot RSI Strategy",
    capital_fraction=0.60,
    primary_interval="30m",
    default_decide_interval=60,
    ma_field="sma60",
    max_additions=2,
    entry_confirm_intervals=("3m", "5m", "15m"),
    trend_confirm_intervals=("30m",),
    candle_configurations=LONG_TERM_INTERVALS,
    allow_entries_in_bear=True,
    bear_cap_ratio=1.0,
    entry_rsi_thresholds=LONG_ENTRY_RSI_THRESHOLDS,
    entry_buy_ratios=LONG_ENTRY_BUY_RATIOS,
    sell_rsi_thresholds=LONG_SELL_RSI_THRESHOLDS,
    sell_cumulative_ratios=LONG_SELL_CUMULATIVE_RATIOS,
    reset_exit_rsi=70,
    require_bollinger_squeeze=False,
    require_bollinger_lower_touch=False,
    require_mtm_turn_up=True,
    require_mtm_below_zero=False,
    add_requires_trend_up=True,
    add_requires_no_bear_trend_4h=False,
    use_short_dual_mode=False,
    daily_overbought_rsi=75.0,
    strong_entry_intervals=(),
    strong_entry_min_confirmations=0,
)

SHORT_TERM_PROFILE = SpotRsiStrategyProfile(
    strategy_type="ShortTermSpotRsiStrategy",
    agent_name="ShortTermSpotRsiStrategyAgent",
    display_name="Short-Term Spot RSI Strategy",
    capital_fraction=0.40,
    primary_interval="15m",
    default_decide_interval=60,
    ma_field="sma20",
    max_additions=1,
    entry_confirm_intervals=("3m", "5m", "15m"),
    trend_confirm_intervals=("30m",),
    candle_configurations=SHORT_TERM_INTERVALS,
    allow_entries_in_bear=True,
    bear_cap_ratio=1.0,
    entry_rsi_thresholds=SHORT_ENTRY_RSI_THRESHOLDS,
    entry_buy_ratios=SHORT_ENTRY_BUY_RATIOS,
    sell_rsi_thresholds=SHORT_SELL_RSI_THRESHOLDS,
    sell_cumulative_ratios=SHORT_SELL_CUMULATIVE_RATIOS,
    reset_exit_rsi=62,
    require_bollinger_squeeze=False,
    require_bollinger_lower_touch=False,
    require_mtm_turn_up=True,
    require_mtm_below_zero=False,
    add_requires_trend_up=False,
    add_requires_no_bear_trend_4h=True,
    use_short_dual_mode=True,
    daily_overbought_rsi=75.0,
    strong_entry_intervals=("3m", "5m", "15m"),
    strong_entry_min_confirmations=2,
)
