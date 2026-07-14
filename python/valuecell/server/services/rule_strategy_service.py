"""Application service for persisted, paper-only rule strategy evaluations."""

from __future__ import annotations

import math
from typing import Any
from uuid import uuid4

from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyConfig,
    RuleStrategyEngineMarketSnapshot,
    RuleStrategyEvaluationRequest,
    RuleStrategyMarketSnapshot,
    RuleStrategyPaperAccount,
    RuleStrategyPaperPosition,
    RuleStrategyPosition,
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
                strategy_id=f"rule_{uuid4().hex}",
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
        market: RuleStrategyMarketSnapshot,
    ) -> dict[str, Any]:
        strategy = self._require_strategy(strategy_id, tenant_id)
        if strategy.status != "running":
            raise RuleStrategyNotRunningError(
                "Rule strategy must be running before evaluation"
            )

        config = RuleStrategyConfig.model_validate(strategy.config)
        account_before = self._account_from_history(strategy, tenant_id, config)
        engine_market = self._engine_market(account_before, market, config.risk.leverage)
        request = RuleStrategyEvaluationRequest(
            config=config,
            candles=candles,
            market=engine_market,
        )
        result = self.engine.evaluate(request)
        account_after, fill = self._apply_result(account_before, market, result.model_dump())
        result_data = result.model_dump(mode="json")
        result_data["account"] = account_after.model_dump(mode="json")
        journal = self.repository.append_evaluation(
            RuleStrategyEvaluationJournal(
                evaluation_id=f"evaluation_{uuid4().hex}",
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                result=result_data,
                signals=[
                    condition.model_dump(mode="json") for condition in result.conditions
                ],
                trades=self._trade_entries(result_data, fill),
                funding=[result.funding.model_dump(mode="json")],
            )
        )
        return {
            "strategy_id": strategy_id,
            "evaluation_id": journal.evaluation_id,
            "config": strategy.config,
            **result_data,
        }

    def evaluate_batch(
        self,
        strategy_id: str,
        tenant_id: str,
        market_inputs: list[tuple[list[Any] | dict[str, list[Any]], RuleStrategyMarketSnapshot]],
    ) -> list[dict[str, Any]]:
        """Evaluate one market cycle and split paper capital across buy candidates.

        All markets are assessed against the same opening account state. This
        prevents a symbol's position in the configured watchlist from changing
        which other symbols qualify during the same scheduler cycle.
        """
        strategy = self._require_strategy(strategy_id, tenant_id)
        if strategy.status != "running":
            raise RuleStrategyNotRunningError(
                "Rule strategy must be running before evaluation"
            )

        config = RuleStrategyConfig.model_validate(strategy.config)
        account = self._account_from_history(strategy, tenant_id, config)
        evaluated: list[tuple[RuleStrategyMarketSnapshot, dict[str, Any]]] = []
        for candle_input, market in market_inputs:
            candle_sets = (
                candle_input if isinstance(candle_input, dict) else {config.interval: candle_input}
            )
            candles = candle_sets.get(config.interval) or next(iter(candle_sets.values()))
            engine_market = self._engine_market(account, market, config.risk.leverage)
            result = self.engine.evaluate(
                RuleStrategyEvaluationRequest(
                    config=config,
                    candles=candles,
                    candle_sets=candle_sets,
                    market=engine_market,
                )
            )
            evaluated.append((market, result.model_dump(mode="json")))

        sell_items = [item for item in evaluated if item[1]["action"] == "sell"]
        buy_items = [item for item in evaluated if item[1]["action"] == "buy"]
        other_items = [
            item for item in evaluated if item[1]["action"] not in {"buy", "sell"}
        ]
        output: list[dict[str, Any]] = []

        for market, result_data in sell_items:
            account, fill = self._apply_result(account, market, result_data)
            output.append(
                self._record_evaluation(
                    strategy_id, tenant_id, strategy.config, market, result_data, account, fill
                )
            )

        available_slots = max(config.risk.max_positions - len(account.positions), 0)
        selected_buys = buy_items[:available_slots]
        unselected_buys = buy_items[available_slots:]
        # Each qualifying symbol receives an equal whole-USDT share of the
        # available strategy capital. The total is never scaled by a per-trade
        # setting: the candidate count is the only divisor.
        equal_share = math.floor(account.quote_balance / len(selected_buys)) if selected_buys else 0

        for market, result_data in selected_buys:
            result_data["sizing"] = {
                **result_data["sizing"],
                "requested_quote": equal_share,
                "affordable_quote": account.quote_balance,
                "quantity": equal_share / market.price,
            }
            account, fill = self._apply_result(account, market, result_data)
            output.append(
                self._record_evaluation(
                    strategy_id, tenant_id, strategy.config, market, result_data, account, fill
                )
            )

        for market, result_data in unselected_buys:
            result_data.update(
                action="no_op",
                reason_code="max_positions",
                reason="qualified entry skipped because the position limit was reached",
            )
            other_items.append((market, result_data))

        for market, result_data in other_items:
            account, fill = self._apply_result(account, market, result_data)
            output.append(
                self._record_evaluation(
                    strategy_id, tenant_id, strategy.config, market, result_data, account, fill
                )
            )
        return output

    def evaluations(
        self, strategy_id: str, tenant_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Return complete, durable explanations for each paper evaluation."""
        self._require_strategy(strategy_id, tenant_id)
        entries: list[dict[str, Any]] = []
        for journal in self.repository.get_evaluations(strategy_id, tenant_id, limit=limit):
            result = journal.result or {}
            entries.append(
                {
                    "strategy_id": strategy_id,
                    "evaluation_id": journal.evaluation_id,
                    **(
                        {"symbol": result["symbol"]}
                        if "symbol" in result
                        else {}
                    ),
                    "evaluated_at": journal.created_at,
                    "action": result.get("action", "no_op"),
                    "reason_code": result.get("reason_code", "unknown"),
                    "reason": result.get("reason", "No explanation was recorded."),
                    "conditions": result.get("conditions", []),
                    "indicators": result.get("indicators", {}),
                    "sizing": result.get("sizing", {}),
                    "funding": result.get("funding", {}),
                    "account": result.get("account", {}),
                    "trades": journal.trades or [],
                }
            )
        return entries

    def account(self, strategy_id: str, tenant_id: str) -> dict[str, Any]:
        strategy = self._require_strategy(strategy_id, tenant_id)
        config = RuleStrategyConfig.model_validate(strategy.config)
        return self._account_from_history(strategy, tenant_id, config).model_dump(mode="json")

    def _account_from_history(
        self, strategy: RuleStrategy, tenant_id: str, config: RuleStrategyConfig
    ) -> RuleStrategyPaperAccount:
        journals = self.repository.get_evaluations(strategy.strategy_id, tenant_id, limit=1)
        if journals:
            raw_account = (journals[0].result or {}).get("account")
            if raw_account is not None:
                return RuleStrategyPaperAccount.model_validate(raw_account)
        return RuleStrategyPaperAccount(
            initial_capital_quote=config.initial_capital_quote,
            quote_balance=config.initial_capital_quote,
            equity_quote=config.initial_capital_quote,
        )

    def _record_evaluation(
        self,
        strategy_id: str,
        tenant_id: str,
        config: dict[str, Any],
        market: RuleStrategyMarketSnapshot,
        result_data: dict[str, Any],
        account: RuleStrategyPaperAccount,
        fill: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result_data["symbol"] = market.symbol
        result_data["account"] = account.model_dump(mode="json")
        journal = self.repository.append_evaluation(
            RuleStrategyEvaluationJournal(
                evaluation_id=f"evaluation_{uuid4().hex}",
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                result=result_data,
                signals=result_data["conditions"],
                trades=self._trade_entries(result_data, fill),
                funding=[result_data["funding"]],
            )
        )
        return {
            "strategy_id": strategy_id,
            "evaluation_id": journal.evaluation_id,
            "config": config,
            "symbol": market.symbol,
            **result_data,
        }

    @staticmethod
    def _engine_market(
        account: RuleStrategyPaperAccount,
        market: RuleStrategyMarketSnapshot,
        leverage: float,
    ) -> RuleStrategyEngineMarketSnapshot:
        position = account.positions.get(market.symbol)
        marked_positions = {
            symbol: item.model_copy(update={"mark_price": market.price})
            if symbol == market.symbol
            else item
            for symbol, item in account.positions.items()
        }
        unrealized = sum(
            item.quantity * (item.mark_price - item.entry_price)
            for item in marked_positions.values()
        )
        equity = account.quote_balance + sum(
            item.quantity * item.mark_price for item in marked_positions.values()
        )
        return RuleStrategyEngineMarketSnapshot(
            symbol=market.symbol,
            price=market.price,
            funding_rate=market.funding_rate,
            equity_quote=equity,
            # The spot paper ledger never borrows. This keeps a leverage setting
            # from creating an unaudited negative cash balance.
            quote_balance=account.quote_balance / leverage,
            open_position_count=len(marked_positions),
            position=RuleStrategyPosition(
                quantity=position.quantity if position else 0.0,
                entry_price=position.entry_price if position else None,
            ),
        )

    @staticmethod
    def _apply_result(
        account: RuleStrategyPaperAccount,
        market: RuleStrategyMarketSnapshot,
        result: dict[str, Any],
    ) -> tuple[RuleStrategyPaperAccount, dict[str, Any] | None]:
        positions = dict(account.positions)
        realized = account.realized_pnl_quote
        fill: dict[str, Any] | None = None
        existing = positions.get(market.symbol)

        if result["action"] == "buy":
            quote_amount = min(float(result["sizing"]["requested_quote"]), account.quote_balance)
            quantity = quote_amount / market.price
            if quote_amount > 0 and existing is None:
                positions[market.symbol] = RuleStrategyPaperPosition(
                    quantity=quantity, entry_price=market.price, mark_price=market.price
                )
                fill = {"symbol": market.symbol, "price": market.price, "quantity": quantity,
                        "quote_amount": quote_amount, "realized_pnl_quote": 0.0}
                quote_balance = account.quote_balance - quote_amount
            else:
                quote_balance = account.quote_balance
        elif result["action"] == "sell" and existing is not None:
            quote_amount = existing.quantity * market.price
            realized_trade = existing.quantity * (market.price - existing.entry_price)
            realized += realized_trade
            positions.pop(market.symbol)
            quote_balance = account.quote_balance + quote_amount
            fill = {"symbol": market.symbol, "price": market.price, "quantity": existing.quantity,
                    "quote_amount": quote_amount, "realized_pnl_quote": realized_trade}
        else:
            quote_balance = account.quote_balance

        positions = {
            symbol: item.model_copy(update={"mark_price": market.price})
            if symbol == market.symbol else item
            for symbol, item in positions.items()
        }
        unrealized = sum(item.quantity * (item.mark_price - item.entry_price) for item in positions.values())
        equity = quote_balance + sum(item.quantity * item.mark_price for item in positions.values())
        return RuleStrategyPaperAccount(
            initial_capital_quote=account.initial_capital_quote,
            quote_balance=quote_balance,
            positions=positions,
            realized_pnl_quote=realized,
            unrealized_pnl_quote=unrealized,
            equity_quote=equity,
        ), fill

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

    def _strategy_data(self, strategy: RuleStrategy) -> dict[str, Any]:
        config = RuleStrategyConfig.model_validate(strategy.config)
        return {
            "strategy_id": strategy.strategy_id,
            "name": strategy.name,
            "description": strategy.description,
            "status": strategy.status,
            "mode": "paper",
            "config": strategy.config,
            "account": self._account_from_history(strategy, strategy.tenant_id, config).model_dump(
                mode="json"
            ),
            "created_at": strategy.created_at,
            "updated_at": strategy.updated_at,
        }

    @staticmethod
    def _trade_entries(
        result: dict[str, Any], fill: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if fill is None:
            return []
        return [
            {
                "action": result["action"],
                "reason_code": result["reason_code"],
                "reason": result["reason"],
                "sizing": result["sizing"],
                "execution": "paper_filled",
                **fill,
            }
        ]
