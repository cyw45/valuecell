"""Account-level risk circuit-breakers for strategy order admission."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from valuecell.server.db.models.rule_strategy import RuleStrategyRiskState


@dataclass(frozen=True, slots=True)
class RiskObservation:
    """Risk inputs calculated from current account and supplied market bars."""

    daily_loss_pct: float = 0.0
    drawdown_pct: float = 0.0
    symbol_daily_drop_pct: float = 0.0
    br: float | None = None
    ar: float | None = None
    prior_br_values: tuple[float, ...] = ()




def utc_day_end(now: datetime) -> datetime:
    """Return the next UTC midnight used by the daily-loss halt."""
    current = now.astimezone(timezone.utc)
    return (current.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))


def apply_risk_observation(
    risk: RuleStrategyRiskState,
    observation: RiskObservation,
    *,
    now: datetime,
) -> tuple[RuleStrategyRiskState, str | None]:
    """Apply the highest-priority V2.1 circuit and return its durable reason code."""
    timestamp = now.astimezone(timezone.utc)
    risk.current_drawdown_pct = max(0.0, observation.drawdown_pct)
    if observation.daily_loss_pct >= 0.03:
        state, code, until = "halted", "daily_loss_at_least_3_percent", utc_day_end(timestamp)
        detail = "UTC 日损失达到 3%，当日停止新订单。"
    elif observation.drawdown_pct >= 0.10:
        state, code, until = "only_reduce", "drawdown_at_least_10_percent", None
        detail = "账户峰值回撤达到 10%，仅允许减仓或平仓。"
    elif observation.symbol_daily_drop_pct >= 0.15:
        state, code, until = "only_reduce", "symbol_daily_drop_at_least_15_percent", None
        detail = "标的日跌幅达到 15%，触发强制平仓检查。"
    elif (
        observation.br is not None
        and observation.ar is not None
        and observation.br > 180
        and observation.ar > 200
    ):
        state, code, until = "only_reduce", "brar_extreme_circuit", timestamp + timedelta(hours=6)
        detail = "BR>180 且 AR>200，六小时内仅允许减仓或平仓。"
    elif (
        len(observation.prior_br_values) >= 3
        and all(50 <= value <= 150 for value in observation.prior_br_values[-3:])
        and observation.br is not None
        and observation.br > 200
    ):
        state, code, until = "only_reduce", "br_breakout_circuit", timestamp + timedelta(hours=24)
        detail = "BR 从 50–150 区间在三根 15 分钟 bar 内突破 200，24 小时仅减仓。"
    else:
        if risk.state == "halted" and risk.cooldown_until and risk.cooldown_until > timestamp:
            return risk, risk.reason_code
        state, code, until = "normal", None, None
        detail = None
    risk.state = state
    risk.cooldown_until = until
    risk.reason_code = code
    risk.reason_detail = detail
    return risk, code


def entries_allowed(risk: RuleStrategyRiskState) -> bool:
    """Return whether a current risk state permits entry/add intent creation."""
    return risk.state == "normal" and (
        risk.cooldown_until is None
        or risk.cooldown_until <= datetime.now(timezone.utc)
    )
