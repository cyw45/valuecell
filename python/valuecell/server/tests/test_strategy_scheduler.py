from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.services import strategy_scheduler


@pytest.mark.asyncio
async def test_sync_does_not_postpone_unchanged_running_strategy(monkeypatch):
    """A frequent sync must not reset a slower strategy job's next run time."""
    config = RuleStrategyConfig(interval="5m").model_dump(mode="json")
    strategy = SimpleNamespace(
        strategy_id="rule-demo",
        tenant_id="tenant-demo",
        config=config,
    )
    repository = SimpleNamespace(list_running=lambda: [strategy])
    monkeypatch.setattr(
        strategy_scheduler,
        "RuleStrategyRepository",
        lambda db_session: repository,
    )
    scheduler = strategy_scheduler.StrategyScheduler()
    await scheduler.start()
    scheduler._scheduler.pause()
    try:
        scheduler.sync_running_strategies(SimpleNamespace())
        first_job = scheduler._scheduler.get_job(strategy.strategy_id)
        assert first_job is not None
        assert first_job.next_run_time <= datetime.now(timezone.utc)

        first_next_run = first_job.next_run_time
        scheduler.sync_running_strategies(SimpleNamespace())
        synced_job = scheduler._scheduler.get_job(strategy.strategy_id)

        assert synced_job is not None
        assert synced_job.next_run_time == first_next_run
    finally:
        await scheduler.stop()
