"""Strategy configuration schema service."""

from __future__ import annotations

from typing import Any

from valuecell.agents.common.trading.models import StrategyType
from valuecell.agents.spot_rsi_ladder_agent.config import (
    LONG_TERM_PROFILE,
    SHORT_TERM_PROFILE,
    SpotRsiStrategyProfile,
)
from valuecell.server.api.schemas.crypto_market import CryptoSymbolCatalogData
from valuecell.server.api.schemas.strategy_schema import (
    StrategyConfigField,
    StrategyConfigOption,
    StrategyConfigSchema,
    StrategyConfigSchemaCatalog,
)
from valuecell.server.services.crypto_market_service import get_crypto_market_service

INTERVAL_OPTIONS = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]


def _options(values: list[Any] | tuple[Any, ...]) -> list[StrategyConfigOption]:
    return [StrategyConfigOption(label=str(value), value=value) for value in values]


class StrategySchemaService:
    """Build UI-consumable schemas for supported strategy types."""

    def get_catalog(self) -> StrategyConfigSchemaCatalog:
        symbols = get_crypto_market_service().get_supported_symbols()
        return StrategyConfigSchemaCatalog(
            schemas=[
                self._prompt_schema(symbols),
                self._grid_schema(symbols),
                self._spot_rsi_schema(LONG_TERM_PROFILE, symbols),
                self._spot_rsi_schema(SHORT_TERM_PROFILE, symbols),
            ]
        )

    def _prompt_schema(self, symbols: CryptoSymbolCatalogData) -> StrategyConfigSchema:
        defaults = {
            "symbols": ["BTC-USDT", "ETH-USDT"],
            "decide_interval": 60,
            "max_leverage": 1.0,
            "cap_factor": 1.0,
            "template_id": "",
        }
        return StrategyConfigSchema(
            strategy_type=StrategyType.PROMPT.value,
            label="Prompt Based Strategy",
            description="LLM-driven strategy using a saved prompt and market features.",
            defaults=defaults,
            fields=self._base_fields(symbols, defaults)
            + [
                StrategyConfigField(
                    key="template_id",
                    label="Prompt template",
                    field_type="text",
                    default="",
                    required=True,
                    group="logic",
                    description="Saved prompt template id used by the LLM strategy.",
                    persistence_target="trading_config",
                )
            ],
        )

    def _grid_schema(self, symbols: CryptoSymbolCatalogData) -> StrategyConfigSchema:
        defaults = {
            "symbols": ["BTC-USDT"],
            "decide_interval": 60,
            "max_leverage": 1.0,
            "cap_factor": 1.0,
        }
        return StrategyConfigSchema(
            strategy_type=StrategyType.GRID.value,
            label="Grid Strategy",
            description="Rule-based grid strategy with model-advised grid parameters.",
            defaults=defaults,
            fields=self._base_fields(symbols, defaults, max_symbols=1),
        )

    def _spot_rsi_schema(
        self,
        profile: SpotRsiStrategyProfile,
        symbols: CryptoSymbolCatalogData,
    ) -> StrategyConfigSchema:
        defaults = {
            "symbols": list(symbols.symbols[:8]),
            "decide_interval": profile.default_decide_interval,
            "max_leverage": 1.0,
            "max_positions": 8,
            "cap_factor": 1.0,
            "primary_interval": profile.primary_interval,
            "entry_rsi_thresholds": list(profile.entry_rsi_thresholds),
            "sell_rsi_thresholds": list(profile.sell_rsi_thresholds),
            "bear_cap_ratio": profile.bear_cap_ratio,
            "daily_overbought_rsi": profile.daily_overbought_rsi,
            "max_additions": profile.max_additions,
        }
        fields = self._base_fields(symbols, defaults) + [
            StrategyConfigField(
                key="primary_interval",
                label="Primary interval",
                field_type="select",
                default=profile.primary_interval,
                options=_options(INTERVAL_OPTIONS),
                required=True,
                group="market",
            ),
            StrategyConfigField(
                key="entry_rsi_thresholds",
                label="Entry RSI ladder",
                field_type="number_list",
                default=list(profile.entry_rsi_thresholds),
                min=1,
                max=100,
                group="thresholds",
                description="RSI values that allocate staged buy ratios.",
            ),
            StrategyConfigField(
                key="sell_rsi_thresholds",
                label="Sell RSI ladder",
                field_type="number_list",
                default=list(profile.sell_rsi_thresholds),
                min=1,
                max=100,
                group="thresholds",
            ),
            StrategyConfigField(
                key="daily_overbought_rsi",
                label="Daily overbought RSI",
                field_type="number",
                default=profile.daily_overbought_rsi,
                min=1,
                max=100,
                step=1,
                group="thresholds",
            ),
            StrategyConfigField(
                key="bear_cap_ratio",
                label="Bear market cap ratio",
                field_type="number",
                default=profile.bear_cap_ratio,
                min=0.05,
                max=1,
                step=0.05,
                group="position",
            ),
            StrategyConfigField(
                key="max_additions",
                label="Max additions",
                field_type="number",
                default=profile.max_additions,
                min=0,
                max=10,
                step=1,
                group="position",
            ),
        ]
        return StrategyConfigSchema(
            strategy_type=profile.strategy_type,
            label=profile.display_name,
            description="Rule-based spot RSI ladder strategy with configurable symbols, intervals, thresholds, and position limits.",
            defaults=defaults,
            fields=fields,
        )

    def _base_fields(
        self,
        symbols: CryptoSymbolCatalogData,
        defaults: dict[str, Any],
        max_symbols: int | None = None,
    ) -> list[StrategyConfigField]:
        symbol_description = "Crypto USDT symbols to observe and trade."
        if max_symbols == 1:
            symbol_description = "Single crypto USDT symbol for this strategy."
        return [
            StrategyConfigField(
                key="symbols",
                label="Observed symbols",
                field_type="multi_select",
                default=defaults.get("symbols", []),
                options=_options(symbols.symbols),
                required=True,
                group="market",
                description=symbol_description,
                persistence_target="trading_config",
            ),
            StrategyConfigField(
                key="initial_capital",
                label="Initial capital",
                field_type="number",
                default=defaults.get("initial_capital", 1000),
                min=1,
                step=100,
                required=True,
                group="position",
                persistence_target="trading_config",
            ),
            StrategyConfigField(
                key="decide_interval",
                label="Decision interval seconds",
                field_type="number",
                default=defaults.get("decide_interval", 60),
                min=10,
                max=3600,
                step=5,
                required=True,
                group="market",
                persistence_target="trading_config",
            ),
            StrategyConfigField(
                key="max_leverage",
                label="Max leverage",
                field_type="number",
                default=defaults.get("max_leverage", 1.0),
                min=1,
                max=5,
                step=0.5,
                required=True,
                group="position",
                persistence_target="trading_config",
            ),
            StrategyConfigField(
                key="max_positions",
                label="Max concurrent positions",
                field_type="number",
                default=defaults.get("max_positions", max_symbols or 5),
                min=1,
                max=max_symbols or 20,
                step=1,
                required=True,
                group="position",
                persistence_target="trading_config",
            ),
            StrategyConfigField(
                key="cap_factor",
                label="Capital cap factor",
                field_type="number",
                default=defaults.get("cap_factor", 1.0),
                min=0.1,
                max=5,
                step=0.1,
                group="position",
                persistence_target="trading_config",
            ),
        ]


_strategy_schema_service: StrategySchemaService | None = None


def get_strategy_schema_service() -> StrategySchemaService:
    global _strategy_schema_service
    if _strategy_schema_service is None:
        _strategy_schema_service = StrategySchemaService()
    return _strategy_schema_service
