"""Isolated V19 scheduler lifecycle contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from .leader_spot_v19_quality import LeaderSpotV19QualityModel


class LeaderSpotV19BatchSummary(LeaderSpotV19QualityModel):
    batch_id: str
    strategy_id: str
    execution_generation: int = Field(ge=1)
    status: Literal["running", "stopped", "archived"]
    started_at: datetime
    stopped_at: datetime | None = None

    @model_validator(mode="after")
    def normalize(self) -> "LeaderSpotV19BatchSummary":
        if self.started_at.tzinfo is None:
            raise ValueError("batch start must be timezone-aware")
        self.started_at = self.started_at.astimezone(UTC)
        if self.stopped_at is not None:
            if self.stopped_at.tzinfo is None:
                raise ValueError("batch stop must be timezone-aware")
            self.stopped_at = self.stopped_at.astimezone(UTC)
        return self


class LeaderSpotV19SchedulerTickResult(LeaderSpotV19QualityModel):
    strategy_id: str
    batch_id: str | None
    status: Literal["processed", "skipped", "blocked"]
    reason_code: str | None = None
