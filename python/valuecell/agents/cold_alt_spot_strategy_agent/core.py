"""Cold alt spot strategy agent."""

from __future__ import annotations

from valuecell.agents.common.trading.base_agent import BaseStrategyAgent
from valuecell.agents.common.trading.features.interfaces import BaseFeaturesPipeline
from valuecell.agents.common.trading.models import MarketType, UserRequest

from .composer import ColdAltSpotComposer
from .config import CAPITAL_FRACTION, COLD_ALT_SPOT_PROFILE, MAX_CONCURRENT_POSITIONS
from .features import ColdAltSpotFeaturesPipeline


class ColdAltSpotStrategyAgent(BaseStrategyAgent):
    """Rule-based spot strategy for cold altcoins that already passed manual screening."""

    async def _build_features_pipeline(
        self,
        request: UserRequest,
    ) -> BaseFeaturesPipeline | None:
        return ColdAltSpotFeaturesPipeline.from_request(request, COLD_ALT_SPOT_PROFILE)

    async def _create_decision_composer(self, request: UserRequest):
        return ColdAltSpotComposer(request)

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
        request.exchange_config.market_type = MarketType.SPOT
        request.trading_config.max_leverage = 1.0
        request.trading_config.max_positions = min(
            max(request.trading_config.max_positions, 1),
            MAX_CONCURRENT_POSITIONS,
        )
        if request.trading_config.decide_interval == 60:
            request.trading_config.decide_interval = (
                COLD_ALT_SPOT_PROFILE.default_decide_interval
            )
        if not request.trading_config.strategy_name:
            request.trading_config.strategy_name = COLD_ALT_SPOT_PROFILE.display_name
        if request.exchange_config.trading_mode.value == "virtual":
            request.trading_config.initial_capital *= CAPITAL_FRACTION
            request.trading_config.initial_free_cash *= CAPITAL_FRACTION
