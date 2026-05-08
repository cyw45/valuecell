"""Configuration for the cold alt spot strategy."""

from __future__ import annotations

from dataclasses import dataclass

from valuecell.agents.common.trading.models import CandleConfig

BOLLINGER_WINDOW: int = 20
BOLLINGER_STD: float = 2.0
RSI_PERIOD: int = 14
MTM_PERIOD: int = 14
ARBR_PERIOD: int = 26

ICE_RSI_MIN: int = 12
ICE_RSI_MAX: int = 22
TAKE_PROFIT_RSI: int = 60
AR_WARM_THRESHOLD: float = 40.0
BR_WARM_THRESHOLD: float = 60.0
AR_HOT_THRESHOLD: float = 110.0
BR_HOT_THRESHOLD: float = 130.0

OLD_COIN_MIN_DAYS: int = 180
NEW_COIN_MAX_DAYS: int = 90
NEW_COIN_MIN_4H_BARS: int = 3
TIME_STOP_DAYS: int = 3

OLD_INITIAL_ALLOC_RATIO: float = 0.125
OLD_ADD_ALLOC_RATIO: float = 0.09375
OLD_MAX_EXPOSURE_RATIO: float = 0.21875

NEW_INITIAL_ALLOC_RATIO: float = 0.0625
NEW_ADD_ALLOC_RATIO: float = 0.0625
NEW_MAX_EXPOSURE_RATIO: float = 0.125

STAGE1_CUMULATIVE_EXIT_RATIO: float = 0.50
STAGE2_CUMULATIVE_EXIT_RATIO: float = 0.80
FINAL_EXIT_CUMULATIVE_EXIT_RATIO: float = 1.00

MAX_CONCURRENT_POSITIONS: int = 5
CAPITAL_FRACTION: float = 0.32
DEFAULT_DECIDE_INTERVAL: int = 3600

PRIMARY_INTERVAL: str = "4h"
CONFIRM_INTERVAL: str = "1h"
RISK_FILTER_INTERVAL: str = "15m"
SCREEN_INTERVAL: str = "1d"

OLD_SIDEWAYS_RANGE_MAX_RATIO: float = 0.25
NEW_SIDEWAYS_RANGE_MAX_RATIO: float = 0.35
VOLUME_CONTRACTION_MAX_RATIO: float = 0.40
MAX_SINGLE_DAY_VOLATILITY_RATIO: float = 0.80


@dataclass(frozen=True)
class ColdAltSpotStrategyProfile:
    """Static rule-set definition for the cold alt spot strategy."""

    strategy_type: str
    agent_name: str
    display_name: str
    capital_fraction: float
    default_decide_interval: int
    candle_configurations: tuple[CandleConfig, ...]


COLD_ALT_SPOT_INTERVALS: tuple[CandleConfig, ...] = (
    CandleConfig(interval="15m", lookback=120),
    CandleConfig(interval="1h", lookback=180),
    CandleConfig(interval="4h", lookback=240),
    CandleConfig(interval="1d", lookback=220),
)


COLD_ALT_SPOT_PROFILE = ColdAltSpotStrategyProfile(
    strategy_type="ColdAltSpotStrategy",
    agent_name="ColdAltSpotStrategyAgent",
    display_name="Cold Alt Spot Strategy",
    capital_fraction=CAPITAL_FRACTION,
    default_decide_interval=DEFAULT_DECIDE_INTERVAL,
    candle_configurations=COLD_ALT_SPOT_INTERVALS,
)
