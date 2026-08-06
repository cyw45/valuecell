"""Background orchestration for rule-strategy monitor admission reviews."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from loguru import logger


class RunningStrategy(Protocol):
    strategy_id: str
    tenant_id: str


class MonitorRepository(Protocol):
    def list_running(self) -> Iterable[RunningStrategy]: ...


class MonitorReviewService(Protocol):
    def _refresh_monitor_admission(
        self, strategy_id: str, tenant_id: str, *, force: bool
    ) -> None: ...


def review_running_strategy_monitors(
    repository: MonitorRepository, service: MonitorReviewService
) -> None:
    """Review every running strategy without one provider failure starving others."""
    for strategy in repository.list_running():
        try:
            service._refresh_monitor_admission(
                strategy.strategy_id,
                strategy.tenant_id,
                force=False,
            )
        except Exception as exc:
            logger.warning(
                "Strategy monitor review deferred strategy_id={} tenant_id={}: {}",
                strategy.strategy_id,
                strategy.tenant_id,
                exc,
            )
