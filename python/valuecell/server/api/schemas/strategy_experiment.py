"""Paper-only strategy experiment API contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .base import SuccessResponse


class StrategyExperimentPreviewRequest(BaseModel):
    """Requested candidate overrides for a supported spot RSI strategy."""

    strategy_type: Literal["LongTermSpotRsiStrategy", "ShortTermSpotRsiStrategy"]
    parameters: dict[str, Any] = Field(default_factory=dict)


class StrategyExperimentDiagnostic(BaseModel):
    """A validation or paper-risk finding for the requested candidate."""

    severity: Literal["info", "warning"]
    code: str
    message: str
    field: str | None = None


class StrategyExperimentCandidateSummary(BaseModel):
    """Deterministic profile facts, deliberately independent of market history."""

    entry_steps: int
    exit_steps: int
    total_entry_allocation: float
    max_exposure_ratio: float
    risk_level: Literal["low", "moderate", "elevated"]


class StrategyExperimentPreviewData(BaseModel):
    """Validated paper-only configuration preview without backtest claims."""

    mode: Literal["paper"] = "paper"
    strategy_type: Literal["LongTermSpotRsiStrategy", "ShortTermSpotRsiStrategy"]
    parameters: dict[str, Any]
    fingerprint: str = Field(
        description="SHA-256 of the versioned canonical parameter representation."
    )
    warnings: list[str] = Field(default_factory=list)
    diagnostics: list[StrategyExperimentDiagnostic] = Field(default_factory=list)
    candidate_summary: StrategyExperimentCandidateSummary


StrategyExperimentPreviewResponse = SuccessResponse[StrategyExperimentPreviewData]
