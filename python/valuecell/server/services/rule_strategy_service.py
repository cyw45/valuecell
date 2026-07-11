"""Application service for persisted, paper-only rule strategy evaluations."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyConfig,
    RuleStrategyEvaluationRequest,
)
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyEvaluationJournal,
)
from valuecell.server.db.repositories.rule_strategy_repository import (
    RuleStrategyRepository,
)
from valuecell.server.services.rule_engine import RuleEngine


class RuleStrategyNotFoundError(Exception):
    """Raised when an isolated rule strategy does not exist."""


class RuleStrategyNotRunningError(Exception):
    """Raised when evaluation is requested for a stopped strategy."""


class RuleStrategyService:
    """Persist and evaluate standalone deterministic paper rule strategies."""

    def __init__(
        self,
        repository: RuleStrategyRepository | None = None,
        engine: RuleEngine | None = None,
    ) -> None:
        self.repository = repository or RuleStrategyRepository()
        self.engine = engine or RuleEngine()

    def create(
        self,
        tenant_id: str,
        name: str,
        description: str | None,
        config: RuleStrategyConfig,
    ) -> dict[str, Any]:
        strategy = self.repository.create(
            RuleStrategy(
                tenant_id=tenant_id,
                name=name,
                description=description,
                status="stopped",
                paper_mode=True,
                config=config.model_dump(mode="json"),
            )
        )
        return self._strategy_data(strategy)
    def list(self, tenant_id: str) -> list[dict[str, Any]]:
        return [
            self._strategy_data(strategy)
            for strategy in self.repository.list(tenant_id)
        ]

    def get(self, strategy_id: str, tenant_id: str) -> dict[str, Any]:
        return self._strategy_data(self._require_strategy(strategy_id, tenant_id))

    def update(
        self,
        strategy_id: str,
        tenant_id: str,
        name: str | None,
        description: str | None,
        config: RuleStrategyConfig | None,
    ) -> dict[str, Any]:
        strategy = self._require_strategy(strategy_id, tenant_id)
        if name is not None:
            strategy.name = name
        if description is not None:
            strategy.description = description
        if config is not None:
            strategy.config = config.model_dump(mode="json")
        return self._strategy_data(self.repository.update(strategy))

    def start(self, strategy_id: str, tenant_id: str) -> dict[str, Any]:
        return self._set_status(strategy_id, tenant_id, "running")

    def stop(self, strategy_id: str, tenant_id: str) -> dict[str, Any]:
        return self._set_status(strategy_id, tenant_id, "stopped")

    def evaluate(
        self,
        strategy_id: str,
        tenant_id: str,
        candles: list[Any],
        market: Any,
    ) -> dict[str, Any]:
        strategy = self._require_strategy(strategy_id, tenant_id)
        if strategy.status != "running":
            raise RuleStrategyNotRunningError(
                "Rule strategy must be running before evaluation"
            )

        request = RuleStrategyEvaluationRequest(
            config=RuleStrategyConfig.model_validate(strategy.config),
            candles=candles,
            market=market,
        )
        result = self.engine.evaluate(request)
        result_data = result.model_dump(mode="json")
        journal = self.repository.append_evaluation(
            RuleStrategyEvaluationJournal(
                evaluation_id=f"evaluation_{uuid4().hex}",
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                result=result_data,
                signals=[
                    condition.model_dump(mode="json") for condition in result.conditions
                ],
                trades=self._trade_entries(result_data),
                funding=[result.funding.model_dump(mode="json")],
            )
        )
        return {
            "strategy_id": strategy_id,
            "evaluation_id": journal.evaluation_id,
            "config": strategy.config,
            **result_data,
        }

    def logs(
        self, strategy_id: str, tenant_id: str, log_type: str, limit: int
    ) -> dict[str, Any]:
        self._require_strategy(strategy_id, tenant_id)
        entries: list[dict[str, Any]] = []
        for journal in self.repository.get_evaluations(
            strategy_id, tenant_id, limit=limit
        ):
            raw_entries = getattr(journal, log_type)
            entries.extend(
                {
                    "evaluation_id": journal.evaluation_id,
                    "evaluated_at": journal.created_at,
                    **entry,
                }
                for entry in raw_entries
            )
        return {"strategy_id": strategy_id, "mode": "paper", "entries": entries}

    def _set_status(
        self, strategy_id: str, tenant_id: str, status: str
    ) -> dict[str, Any]:
        strategy = self._require_strategy(strategy_id, tenant_id)
        strategy.status = status
        return self._strategy_data(self.repository.update(strategy))

    def _require_strategy(self, strategy_id: str, tenant_id: str) -> RuleStrategy:
        strategy = self.repository.get(strategy_id, tenant_id)
        if strategy is None:
            raise RuleStrategyNotFoundError(
                f"Rule strategy '{strategy_id}' was not found"
            )
        return strategy

    @staticmethod
    def _strategy_data(strategy: RuleStrategy) -> dict[str, Any]:
        return {
            "strategy_id": strategy.strategy_id,
            "name": strategy.name,
            "description": strategy.description,
            "status": strategy.status,
            "mode": "paper",
            "config": strategy.config,
            "created_at": strategy.created_at,
            "updated_at": strategy.updated_at,
        }

    @staticmethod
    def _trade_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
        if result["action"] == "no_op":
            return []
        return [
            {
                "action": result["action"],
                "reason_code": result["reason_code"],
                "reason": result["reason"],
                "sizing": result["sizing"],
                "execution": "not_submitted",
            }
        ]
