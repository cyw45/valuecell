"""Rule-based spot RSI ladder strategy agents."""

from __future__ import annotations

from valuecell.agents.common.trading.base_agent import BaseStrategyAgent
from valuecell.agents.common.trading.features.interfaces import BaseFeaturesPipeline
from valuecell.agents.common.trading.models import MarketType, UserRequest

from .composer import SpotRsiLadderComposer
from .config import (
    DEFAULT_SPOT_SYMBOLS,
    LONG_TERM_PROFILE,
    SHORT_TERM_PROFILE,
    SpotRsiStrategyProfile,
)
from .features import SpotRsiLadderFeaturesPipeline


class BaseSpotRsiStrategyAgent(BaseStrategyAgent):
    """Shared setup for the long-term and short-term spot RSI agents."""

    PROFILE: SpotRsiStrategyProfile

    async def _build_features_pipeline(
        self,
        request: UserRequest,
    ) -> BaseFeaturesPipeline | None:
        return SpotRsiLadderFeaturesPipeline.from_request(request, self.PROFILE)

    async def _create_decision_composer(self, request: UserRequest):
        return SpotRsiLadderComposer(request, self.PROFILE)

    async def _create_runtime(
        self,
        request: UserRequest,
        strategy_id_override: str | None = None,
    ):
        self._prepare_request(request)
        return await super()._create_runtime(
            request,
            strategy_id_override=strategy_id_override,
        )

    def _prepare_request(self, request: UserRequest) -> None:
        if not request.trading_config.symbols:
            request.trading_config.symbols = list(DEFAULT_SPOT_SYMBOLS)
        request.exchange_config.market_type = MarketType.SPOT
        request.trading_config.max_leverage = 1.0
        request.trading_config.max_positions = max(
            request.trading_config.max_positions,
            len(request.trading_config.symbols),
        )
        if request.trading_config.decide_interval == 60:
            request.trading_config.decide_interval = self.PROFILE.default_decide_interval
        if not request.trading_config.strategy_name:
            request.trading_config.strategy_name = self.PROFILE.display_name
        if request.exchange_config.trading_mode.value == "virtual":
            request.trading_config.initial_capital *= self.PROFILE.capital_fraction
            request.trading_config.initial_free_cash *= self.PROFILE.capital_fraction


class LongTermSpotRsiStrategyAgent(BaseSpotRsiStrategyAgent):
    """Long-term 60% capital module of the user's spot RSI ladder strategy."""

    PROFILE = LONG_TERM_PROFILE


class ShortTermSpotRsiStrategyAgent(BaseSpotRsiStrategyAgent):
    """Short-term 40% capital module of the user's spot RSI ladder strategy."""

    PROFILE = SHORT_TERM_PROFILE
