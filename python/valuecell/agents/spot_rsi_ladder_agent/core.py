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
    profile_with_overrides,
)
from .features import SpotRsiLadderFeaturesPipeline
from .symbol_filter import filter_supported_spot_symbols


class BaseSpotRsiStrategyAgent(BaseStrategyAgent):
    """Shared setup for the long-term and short-term spot RSI agents."""

    PROFILE: SpotRsiStrategyProfile

    def _effective_profile(self, request: UserRequest) -> SpotRsiStrategyProfile:
        return profile_with_overrides(
            self.PROFILE,
            request.trading_config.strategy_params,
        )

    async def _build_features_pipeline(
        self,
        request: UserRequest,
    ) -> BaseFeaturesPipeline | None:
        return SpotRsiLadderFeaturesPipeline.from_request(
            request,
            self._effective_profile(request),
        )

    async def _create_decision_composer(self, request: UserRequest):
        return SpotRsiLadderComposer(request, self._effective_profile(request))

    async def _create_runtime(
        self,
        request: UserRequest,
        strategy_id_override: str | None = None,
    ):
        await self._prepare_request(request)
        return await super()._create_runtime(
            request,
            strategy_id_override=strategy_id_override,
        )

    async def _prepare_request(self, request: UserRequest) -> None:
        if not request.trading_config.symbols:
            request.trading_config.symbols = list(DEFAULT_SPOT_SYMBOLS)
        request.exchange_config.market_type = MarketType.SPOT
        request.trading_config.symbols = await filter_supported_spot_symbols(
            request.exchange_config.exchange_id,
            request.trading_config.symbols,
            request.exchange_config.market_type,
        )
        request.trading_config.max_leverage = 1.0
        request.trading_config.max_positions = max(
            request.trading_config.max_positions,
            len(request.trading_config.symbols),
        )
        profile = self._effective_profile(request)
        if request.trading_config.decide_interval == 60:
            request.trading_config.decide_interval = profile.default_decide_interval
        if not request.trading_config.strategy_name:
            request.trading_config.strategy_name = profile.display_name
        if request.exchange_config.trading_mode.value == "virtual":
            request.trading_config.initial_capital *= profile.capital_fraction
            request.trading_config.initial_free_cash *= profile.capital_fraction


class LongTermSpotRsiStrategyAgent(BaseSpotRsiStrategyAgent):
    """Long-term 60% capital module of the user's spot RSI ladder strategy."""

    PROFILE = LONG_TERM_PROFILE


class ShortTermSpotRsiStrategyAgent(BaseSpotRsiStrategyAgent):
    """Short-term 40% capital module of the user's spot RSI ladder strategy."""

    PROFILE = SHORT_TERM_PROFILE
