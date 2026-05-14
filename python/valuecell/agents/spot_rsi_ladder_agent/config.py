"""Shared configuration for spot RSI ladder strategies."""

from __future__ import annotations

from dataclasses import dataclass

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

LONG_ENTRY_RSI_THRESHOLDS: tuple[int, ...] = (30, 25, 20, 15, 10)
LONG_ENTRY_BUY_RATIOS: dict[int, float] = {
    30: 0.20,
    25: 0.20,
    20: 0.20,
    15: 0.20,
    10: 0.20,
}
LONG_SELL_RSI_THRESHOLDS: tuple[int, ...] = (70, 75, 80, 85)
LONG_SELL_CUMULATIVE_RATIOS: dict[int, float] = {
    70: 0.10,
    75: 0.50,
    80: 0.90,
    85: 1.00,
}

SHORT_ENTRY_RSI_THRESHOLDS: tuple[int, ...] = (22, 18, 14)
SHORT_ENTRY_BUY_RATIOS: dict[int, float] = {
    22: 0.30,
    18: 0.40,
    14: 0.30,
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


LONG_TERM_PROFILE = SpotRsiStrategyProfile(
    strategy_type="LongTermSpotRsiStrategy",
    agent_name="LongTermSpotRsiStrategyAgent",
    display_name="Long-Term Spot RSI Strategy",
    capital_fraction=0.60,
    primary_interval="30m",
    default_decide_interval=300,
    ma_field="sma60",
    max_additions=2,
    entry_confirm_intervals=("1m", "3m", "5m", "15m", "30m"),
    trend_confirm_intervals=("1h", "4h"),
    candle_configurations=LONG_TERM_INTERVALS,
    entry_rsi_thresholds=LONG_ENTRY_RSI_THRESHOLDS,
    entry_buy_ratios=LONG_ENTRY_BUY_RATIOS,
    sell_rsi_thresholds=LONG_SELL_RSI_THRESHOLDS,
    sell_cumulative_ratios=LONG_SELL_CUMULATIVE_RATIOS,
    reset_exit_rsi=70,
    require_bollinger_squeeze=True,
    require_bollinger_lower_touch=False,
    require_mtm_turn_up=True,
    require_mtm_below_zero=False,
    add_requires_trend_up=True,
    add_requires_no_bear_trend_4h=False,
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
    entry_confirm_intervals=("1m", "3m", "5m", "15m"),
    trend_confirm_intervals=("30m", "4h"),
    candle_configurations=SHORT_TERM_INTERVALS,
    entry_rsi_thresholds=SHORT_ENTRY_RSI_THRESHOLDS,
    entry_buy_ratios=SHORT_ENTRY_BUY_RATIOS,
    sell_rsi_thresholds=SHORT_SELL_RSI_THRESHOLDS,
    sell_cumulative_ratios=SHORT_SELL_CUMULATIVE_RATIOS,
    reset_exit_rsi=62,
    require_bollinger_squeeze=True,
    require_bollinger_lower_touch=True,
    require_mtm_turn_up=True,
    require_mtm_below_zero=True,
    add_requires_trend_up=False,
    add_requires_no_bear_trend_4h=True,
)
