"""Minimal interval trigger compatible with the project's scheduler usage."""

from __future__ import annotations


class IntervalTrigger:
    """Store an interval in seconds for AsyncIOScheduler.add_job."""

    def __init__(self, *, seconds: int | float, **_: object) -> None:
        self.seconds = float(seconds)
        if self.seconds <= 0:
            raise ValueError("IntervalTrigger seconds must be positive")
