from types import SimpleNamespace

from valuecell.server.services.rule_strategy_monitor_scheduler import (
    review_running_strategy_monitors,
)


class Repository:
    def list_running(self):
        return [
            SimpleNamespace(strategy_id="first", tenant_id="tenant-a"),
            SimpleNamespace(strategy_id="second", tenant_id="tenant-b"),
        ]


class Service:
    def __init__(self):
        self.calls: list[tuple[str, str, bool]] = []

    def _refresh_monitor_admission(
        self, strategy_id: str, tenant_id: str, *, force: bool
    ) -> None:
        self.calls.append((strategy_id, tenant_id, force))
        if strategy_id == "first":
            raise RuntimeError("provider unavailable")


def test_monitor_review_isolates_each_running_strategy_failure():
    service = Service()

    review_running_strategy_monitors(Repository(), service)

    assert service.calls == [
        ("first", "tenant-a", False),
        ("second", "tenant-b", False),
    ]
