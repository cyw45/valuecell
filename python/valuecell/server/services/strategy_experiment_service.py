"""Paper-only validation for candidate spot RSI strategy parameters."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from valuecell.agents.spot_rsi_ladder_agent.config import (
    LONG_TERM_PROFILE,
    SHORT_TERM_PROFILE,
)
from valuecell.server.api.schemas.strategy_experiment import (
    StrategyExperimentCandidateSummary,
    StrategyExperimentDiagnostic,
    StrategyExperimentPreviewData,
)

_SUPPORTED_PROFILES = {
    LONG_TERM_PROFILE.strategy_type: LONG_TERM_PROFILE,
    SHORT_TERM_PROFILE.strategy_type: SHORT_TERM_PROFILE,
}
_ALLOWED_PARAMETERS = {
    "primary_interval",
    "entry_rsi_thresholds",
    "sell_rsi_thresholds",
    "bear_cap_ratio",
    "daily_overbought_rsi",
    "max_additions",
}


class StrategyExperimentService:
    """Validate a candidate profile without simulating historical performance."""

    def preview(
        self,
        strategy_type: str,
        parameters: dict[str, Any],
    ) -> StrategyExperimentPreviewData:
        profile = _SUPPORTED_PROFILES.get(strategy_type)
        if profile is None:
            raise ValueError("Unsupported paper experiment strategy type")

        unsupported = set(parameters) - _ALLOWED_PARAMETERS
        if unsupported:
            raise ValueError(f"Unsupported strategy parameters: {sorted(unsupported)}")

        canonical = {
            "primary_interval": profile.primary_interval,
            "entry_rsi_thresholds": list(profile.entry_rsi_thresholds),
            "sell_rsi_thresholds": list(profile.sell_rsi_thresholds),
            "bear_cap_ratio": profile.bear_cap_ratio,
            "daily_overbought_rsi": profile.daily_overbought_rsi,
            "max_additions": profile.max_additions,
        }
        canonical.update(parameters)
        canonical["primary_interval"] = _canonical_interval(
            canonical["primary_interval"]
        )
        canonical["entry_rsi_thresholds"] = _canonical_entry_ladder(
            canonical["entry_rsi_thresholds"]
        )
        canonical["sell_rsi_thresholds"] = _canonical_exit_ladder(
            canonical["sell_rsi_thresholds"]
        )
        canonical["bear_cap_ratio"] = _bounded_float(
            canonical["bear_cap_ratio"], "bear_cap_ratio", 0.05, 1.0
        )
        canonical["daily_overbought_rsi"] = _bounded_float(
            canonical["daily_overbought_rsi"], "daily_overbought_rsi", 1.0, 100.0
        )
        canonical["max_additions"] = _non_negative_int(
            canonical["max_additions"], "max_additions"
        )

        diagnostics = _diagnostics(canonical)
        warnings = [item.message for item in diagnostics if item.severity == "warning"]
        fingerprint = _fingerprint(strategy_type, canonical)
        summary = StrategyExperimentCandidateSummary(
            entry_steps=len(canonical["entry_rsi_thresholds"]),
            exit_steps=len(canonical["sell_rsi_thresholds"]),
            total_entry_allocation=1.0,
            max_exposure_ratio=canonical["bear_cap_ratio"],
            risk_level=_risk_level(canonical),
        )
        return StrategyExperimentPreviewData(
            strategy_type=strategy_type,
            parameters=canonical,
            fingerprint=fingerprint,
            warnings=warnings,
            diagnostics=diagnostics,
            candidate_summary=summary,
        )


def _canonical_interval(value: Any) -> str:
    if not isinstance(value, str) or value not in {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "4h",
        "1d",
    }:
        raise ValueError("primary_interval must be a supported candle interval")
    return value


def _canonical_entry_ladder(value: Any) -> list[int]:
    ladder = _canonical_ladder(value, "entry_rsi_thresholds")
    if ladder != sorted(ladder, reverse=True):
        raise ValueError("entry_rsi_thresholds must be strictly descending")
    return ladder


def _canonical_exit_ladder(value: Any) -> list[int]:
    ladder = _canonical_ladder(value, "sell_rsi_thresholds")
    if ladder != sorted(ladder):
        raise ValueError("sell_rsi_thresholds must be strictly ascending")
    return ladder


def _canonical_ladder(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    ladder: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_name} must contain numeric RSI values")
        if not float(item).is_integer() or not 1 <= int(item) <= 100:
            raise ValueError(f"{field_name} values must be integers between 1 and 100")
        ladder.append(int(item))
    if len(ladder) != len(set(ladder)):
        raise ValueError(f"{field_name} values must be distinct")
    return ladder


def _bounded_float(value: Any, field_name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return number


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be an integer")
    if not float(value).is_integer() or int(value) < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _diagnostics(parameters: dict[str, Any]) -> list[StrategyExperimentDiagnostic]:
    diagnostics = [
        StrategyExperimentDiagnostic(
            severity="info",
            code="PAPER_ONLY",
            message="This preview validates parameters only; it does not run a backtest.",
        )
    ]
    if parameters["bear_cap_ratio"] >= 0.9:
        diagnostics.append(
            StrategyExperimentDiagnostic(
                severity="warning",
                code="HIGH_CAPITAL_ALLOCATION",
                field="bear_cap_ratio",
                message="The candidate may allocate most of its paper capital in bear conditions.",
            )
        )
    if parameters["max_additions"] >= 3:
        diagnostics.append(
            StrategyExperimentDiagnostic(
                severity="warning",
                code="MULTIPLE_ADDITIONS",
                field="max_additions",
                message="Multiple additions can increase drawdown during a prolonged decline.",
            )
        )
    return diagnostics


def _risk_level(parameters: dict[str, Any]) -> str:
    if parameters["bear_cap_ratio"] >= 0.9 or parameters["max_additions"] >= 3:
        return "elevated"
    if parameters["bear_cap_ratio"] >= 0.6 or parameters["max_additions"] >= 2:
        return "moderate"
    return "low"


def _fingerprint(strategy_type: str, parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        {"strategy_type": strategy_type, "parameters": parameters, "version": 1},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
