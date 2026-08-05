"""Code-owned immutable templates for tenant-owned rule strategies."""

from __future__ import annotations

from dataclasses import dataclass

from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyAddTrancheConfig,
    RuleStrategyConfig,
    RuleStrategyMonitorConfig,
    RuleStrategyRiskConfig,
    RuleStrategyTrancheConfig,
    TrendResonanceV21Config,
)

TREND_RESONANCE_V2_1_TEMPLATE_ID = "trend_resonance_v2_1"
TREND_RESONANCE_V2_1_TEMPLATE_VERSION = 2


@dataclass(frozen=True, slots=True)
class RuleStrategyTemplate:
    """Immutable metadata and canonical owned-clone configuration."""

    template_id: str
    template_version: int
    display_name: str
    execution_mode: str
    config: RuleStrategyConfig

    def clone_config(
        self,
        *,
        initial_capital_quote: float,
        symbol_candidates: list[str],
    ) -> RuleStrategyConfig:
        """Return a tenant-owned config copy; never expose the registry instance."""
        return self.config.model_copy(
            deep=True,
            update={
                "initial_capital_quote": initial_capital_quote,
                "symbols": symbol_candidates,
            },
        )


def _trend_resonance_v2_1_config() -> RuleStrategyConfig:
    return RuleStrategyConfig(
        template_id=TREND_RESONANCE_V2_1_TEMPLATE_ID,
        template_version=TREND_RESONANCE_V2_1_TEMPLATE_VERSION,
        indicator_formula_version=TREND_RESONANCE_V2_1_TEMPLATE_ID,
        interval="15m",
        decide_interval_s=900,
        trend_resonance=TrendResonanceV21Config(),
        monitor=RuleStrategyMonitorConfig(),
        tranches=RuleStrategyTrancheConfig(
            entry_fraction_of_symbol_target=0.20,
            add_tranches=(
                RuleStrategyAddTrancheConfig(
                    trigger_return_pct=5.0,
                    fraction_of_symbol_target=0.30,
                ),
                RuleStrategyAddTrancheConfig(
                    trigger_return_pct=10.0,
                    fraction_of_symbol_target=0.20,
                ),
                RuleStrategyAddTrancheConfig(
                    trigger_return_pct=18.0,
                    fraction_of_symbol_target=0.15,
                ),
            ),
            profit_tier_return_pcts=(8.0, 15.0),
        ),
        risk=RuleStrategyRiskConfig(
            max_symbol_position_pct=0.15,
            max_total_position_pct=0.70,
            stop_loss_pct=0.08,
            take_profit_pct=0.15,
            add_to_winners=True,
            max_additions=3,
            daily_loss_halt_pct=0.03,
            max_drawdown_only_reduce_pct=0.10,
            symbol_daily_drop_force_close_pct=0.15,
            reentry_cooldown_hours=6,
            min_add_price_move_pct=0.02,
            brar_extreme_br_threshold=180.0,
            brar_extreme_ar_threshold=200.0,
            brar_extreme_cooldown_hours=6,
            br_breakout_lower_bound=50.0,
            br_breakout_upper_bound=150.0,
            br_breakout_threshold=200.0,
            br_breakout_lookback_bars=3,
            br_breakout_cooldown_hours=24,
        ),
    )


_TEMPLATES: dict[str, RuleStrategyTemplate] = {
    TREND_RESONANCE_V2_1_TEMPLATE_ID: RuleStrategyTemplate(
        template_id=TREND_RESONANCE_V2_1_TEMPLATE_ID,
        template_version=TREND_RESONANCE_V2_1_TEMPLATE_VERSION,
        display_name="多币种趋势共振 V2.1",
        execution_mode="spot_only",
        config=_trend_resonance_v2_1_config(),
    )
}


def list_rule_strategy_templates() -> tuple[RuleStrategyTemplate, ...]:
    """Return code-owned immutable template definitions in stable order."""
    return tuple(_TEMPLATES.values())


def get_rule_strategy_template(template_id: str) -> RuleStrategyTemplate | None:
    """Resolve a registered immutable template without a tenant-side fallback."""
    return _TEMPLATES.get(template_id)
