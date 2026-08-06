"""Application service for persisted, paper-only rule strategy evaluations."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyConfig,
    RuleStrategyEngineMarketSnapshot,
    RuleStrategyEvaluationRequest,
    RuleStrategyMarketSnapshot,
    RuleStrategyPaperAccount,
    RuleStrategyPaperPosition,
    RuleStrategyPosition,
)
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyEvaluationJournal,
    RuleStrategyEvent,
)
from valuecell.server.db.repositories.rule_strategy_repository import (
    RuleStrategyRepository,
)
from valuecell.server.services.rule_engine import RuleEngine
from valuecell.server.services.rule_strategy_templates import (
    get_rule_strategy_template,
)
from valuecell.server.services.crypto_market_service import get_crypto_market_service
from valuecell.server.services.rule_strategy_monitor_service import (
    RuleStrategyMonitorAdmissionWorker,
    StrategyMarketMetadata,
)


class RuleStrategyNotFoundError(Exception):
    """Raised when an isolated rule strategy does not exist."""


class RuleStrategyNotRunningError(Exception):
    """Raised when evaluation is requested for a stopped strategy."""


class RuleStrategyRunningUpdateError(Exception):
    """Raised when configuration changes are requested for a running strategy."""


class RuleStrategyStartAdmissionError(Exception):
    """Raised when current account, risk, monitor, or scope facts block start."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


class RuleStrategyDeleteConflictError(Exception):
    """Raised when runtime state or execution evidence prevents deletion."""


class RuleStrategyUnsupportedEvaluationError(Exception):
    """Raised when manual evaluation cannot use authoritative account facts."""


class RuleStrategyService:
    """Persist and evaluate standalone deterministic paper rule strategies."""

    def __init__(
        self,
        repository: RuleStrategyRepository | None = None,
        engine: RuleEngine | None = None,
        market_service: Any | None = None,
    ) -> None:
        self.repository = repository or RuleStrategyRepository()
        self.engine = engine or RuleEngine()
        self.market_service = market_service or get_crypto_market_service()
    def create(
        self,
        tenant_id: str,
        name: str,
        description: str | None,
        config: RuleStrategyConfig,
    ) -> dict[str, Any]:
        strategy = RuleStrategy(
            strategy_id=f"rule_{uuid4().hex}",
            tenant_id=tenant_id,
            name=name,
            description=description,
            status="stopped",
            paper_mode=True,
            config=config.model_dump(mode="json"),
        )
        create_with_state = getattr(self.repository, "create_with_current_state", None)
        if create_with_state is None:
            strategy = self.repository.create(strategy)
        else:
            strategy = create_with_state(
                strategy,
                scope="paper_virtual",
                credential_id=None,
                symbol_candidates=config.symbols,
            )
        return self._strategy_data(strategy)

    def instantiate_template(
        self,
        tenant_id: str,
        *,
        template_id: str,
        name: str,
        initial_capital_quote: float,
        symbol_candidates: list[str],
        scope: str,
        credential_id: str | None,
    ) -> dict[str, Any]:
        """Clone a code-owned template into a stopped tenant strategy."""
        template = get_rule_strategy_template(template_id)
        if template is None:
            raise ValueError(f"Unknown rule strategy template '{template_id}'")
        if any(
            item.name == name
            for item in self.repository.list(tenant_id, include_archived=True)
        ):
            raise ValueError("Rule strategy name already exists for this tenant")
        config = template.clone_config(
            initial_capital_quote=initial_capital_quote,
            symbol_candidates=symbol_candidates,
        )
        if scope != "paper_virtual":
            config = config.model_copy(
                update={
                    "execution": config.execution.model_copy(
                        update={
                            "environment": "okx_demo",
                            "sandbox_connection_id": credential_id,
                        }
                    )
                }
            )
        strategy = RuleStrategy(
            strategy_id=f"rule_{uuid4().hex}",
            tenant_id=tenant_id,
            name=name,
            status="stopped",
            paper_mode=scope == "paper_virtual",
            config=config.model_dump(mode="json"),
        )
        creator = getattr(self.repository, "create_with_current_state", None)
        if creator is None:
            strategy = self.repository.create(strategy)
        else:
            strategy = creator(
                strategy,
                scope=scope,
                credential_id=credential_id,
                symbol_candidates=config.symbols,
            )
        return self._strategy_data(strategy)

    def list(
        self, tenant_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        strategies = (
            self.repository.list(tenant_id, include_archived=True)
            if include_archived
            else self.repository.list(tenant_id)
        )
        return [self._strategy_data(strategy) for strategy in strategies]

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
        return self._locked_mutate(
            strategy_id,
            tenant_id,
            lambda strategy: self._apply_stopped_update(
                strategy, name, description, config
            ),
        )

    def _refresh_monitor_admission(
        self, strategy_id: str, tenant_id: str, *, force: bool = True
    ) -> None:
        """Persist fresh exchange facts before a stopped strategy can be started."""
        monitors = self.repository.monitors(strategy_id, tenant_id)
        facts = self.market_service.get_monitor_metadata(
            [monitor.symbol for monitor in monitors]
        )
        metadata = {
            symbol: (
                None
                if fact is None
                else StrategyMarketMetadata(
                    listing_age_days=fact.listing_age_days,
                    average_quote_volume_30d=fact.average_quote_volume_30d,
                    price_quote=fact.price_quote,
                    provider=fact.provider,
                    listing_first_tradable_at=fact.listing_first_tradable_at,
                    price_observed_at=fact.price_observed_at,
                )
            )
            for symbol, fact in facts.items()
        }
        state = self.repository.get_account_state(strategy_id, tenant_id)
        positions: dict[str, float] = {}
        if state is not None:
            for symbol, position in (state[0].positions or {}).items():
                if isinstance(position, dict):
                    quantity = position.get("quantity")
                    if isinstance(quantity, (int, float)):
                        positions[symbol.upper().replace("/", "-")] = float(quantity)
        RuleStrategyMonitorAdmissionWorker(self.repository).review(
            strategy_id,
            tenant_id,
            metadata,
            positions,
            force=force,
        )

    def start(self, strategy_id: str, tenant_id: str) -> dict[str, Any]:
        """Start one strategy after a fresh persisted admission review."""
        self._require_strategy(strategy_id, tenant_id)
        self._refresh_monitor_admission(strategy_id, tenant_id)
        current_state = self.repository.get_account_state(strategy_id, tenant_id)
        if current_state is not None:
            _account, risk = current_state
            if risk.state == "halted":
                raise RuleStrategyStartAdmissionError(
                    risk.reason_code or "risk_state_halted",
                    risk.reason_detail or "策略账户风险状态已停止执行。",
                )
            if risk.reason_code == "shared_exchange_account_requires_dedicated_scope":
                raise RuleStrategyStartAdmissionError(
                    risk.reason_code,
                    risk.reason_detail or "共享交易所账户未证明隔离。",
                )
        monitors = self.repository.monitors(strategy_id, tenant_id)
        if not any(item.state in {"admitted", "held"} for item in monitors):
            raise RuleStrategyStartAdmissionError(
                "no_admitted_monitor_symbols",
                "策略至少需要一个已准入或持仓保留的监控标的。",
            )

        def apply(current: RuleStrategy) -> None:
            current.status = "running"
            current.execution_generation = (current.execution_generation or 1) + 1

        return self._locked_mutate(strategy_id, tenant_id, apply)

    def stop(self, strategy_id: str, tenant_id: str) -> dict[str, Any]:
        return self._set_status(strategy_id, tenant_id, "stopped")

    def delete(self, strategy_id: str, tenant_id: str) -> bool:
        """Delete unaudited strategies; archive audited strategies and return True."""
        delete_if_allowed = getattr(self.repository, "delete_if_allowed", None)
        if delete_if_allowed is None:
            strategy = self._require_strategy(strategy_id, tenant_id)
            if strategy.status == "running":
                raise RuleStrategyDeleteConflictError(
                    "Stop the strategy before deleting it"
                )
            getattr(self.repository, "delete")(strategy_id, tenant_id)
            return False
        outcome = delete_if_allowed(strategy_id, tenant_id)
        if outcome == "not_found":
            raise RuleStrategyNotFoundError(
                f"Rule strategy '{strategy_id}' was not found"
            )
        if outcome == "running":
            raise RuleStrategyDeleteConflictError(
                "Stop the strategy before deleting it"
            )
        return outcome == "archived"

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
        if config.execution.environment == "okx_demo":
            raise RuleStrategyUnsupportedEvaluationError(
                "Manual evaluation requires a synchronized OKX Demo account; "
                "use scheduled Demo evaluation instead"
            )
        account_before = self._current_paper_account(strategy, tenant_id, config)
        engine_market = self._engine_market(
            account_before, market, config.risk.leverage
        )
        request = RuleStrategyEvaluationRequest(
            config=config,
            candles=candles,
            market=engine_market,
        )
        result = self.engine.evaluate(request)
        result_data = result.model_dump(mode="json")
        # A direct/manual evaluation of an exchange-backed strategy is analysis
        # only. It must never mutate the local paper ledger or fabricate a fill.
        if config.execution.environment != "paper":
            account_after = account_before
            fill = None
            result_data["execution_ledger"] = "external"
            result_data["paper_fill"] = False
        else:
            account_after, fill = self._apply_result(
                account_before, market, result_data
            )
        result_data["account"] = account_after.model_dump(mode="json")
        journal = self._append_evaluation_statefully(
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
            ),
            account_after if config.execution.environment == "paper" else None,
            tenant_id,
            strategy_id,
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
        market_inputs: list[
            tuple[list[Any] | dict[str, list[Any]], RuleStrategyMarketSnapshot]
        ],
    ) -> list[dict[str, Any]]:
        """Evaluate one market cycle using the configured fixed amount per entry.

        Sells are applied before entries. Each later buy uses the account balance
        left by earlier fills and is blocked by the rule engine when the full
        configured amount is not affordable; partial paper fills are forbidden.
        """
        strategy = self._require_strategy(strategy_id, tenant_id)
        if strategy.status != "running":
            raise RuleStrategyNotRunningError(
                "Rule strategy must be running before evaluation"
            )

        config = RuleStrategyConfig.model_validate(strategy.config)
        is_demo = config.execution.environment == "okx_demo"
        account = (
            None
            if is_demo
            else self._mark_account_for_cycle(
                self._current_paper_account(strategy, tenant_id, config),
                [market for _, market in market_inputs],
            )
        )
        evaluated: list[tuple[RuleStrategyMarketSnapshot, dict[str, Any]]] = []
        if not is_demo:
            assert account is not None
        for candle_input, market in market_inputs:
            candle_sets = (
                candle_input
                if isinstance(candle_input, dict)
                else {config.interval: candle_input}
            )
            candles = candle_sets.get(config.interval) or next(
                iter(candle_sets.values())
            )
            engine_market = (
                market
                if is_demo and isinstance(market, RuleStrategyEngineMarketSnapshot)
                else self._engine_market(account, market, config.risk.leverage)
            )
            result = self.engine.evaluate(
                RuleStrategyEvaluationRequest(
                    config=config,
                    candles=candles,
                    candle_sets=candle_sets,
                    market=engine_market,
                )
            )
            result_data = result.model_dump(mode="json")
            if config.advanced_rules.enabled:
                result_data["exit_confirmation_mode"] = (
                    config.advanced_rules.exit_confirmation_mode
                )
            evaluated.append((market, result_data))

        sell_items = [
            item for item in evaluated if item[1]["action"] in {"sell", "reduce", "close"}
        ]
        remaining_items = [
            item for item in evaluated if item[1]["action"] not in {"sell", "reduce", "close"}
        ]
        output: list[dict[str, Any]] = []

        # An exchange-backed strategy uses its exchange account as the execution
        # ledger. It may journal signals here, but it must never create a local
        # paper fill before the exchange accepts and later confirms the order.
        if is_demo:
            for market, result_data in evaluated:
                if not isinstance(market, RuleStrategyEngineMarketSnapshot):
                    raise ValueError("OKX Demo evaluation requires synced account facts")
                result_data["execution_ledger"] = "external"
                result_data["paper_fill"] = False
                output.append(
                    self._record_evaluation(
                        strategy_id,
                        tenant_id,
                        strategy.config,
                        market,
                        result_data,
                        {
                            "quote_balance": market.quote_balance,
                            "equity_quote": market.equity_quote,
                            "open_position_count": market.open_position_count,
                            "position": market.position.model_dump(mode="json"),
                            "source": "okx_demo",
                        },
                        None,
                    )
                )
            return output

        assert account is not None
        for market, result_data in sell_items:
            account, fill = self._apply_result(account, market, result_data)
            output.append(
                self._record_evaluation(
                    strategy_id,
                    tenant_id,
                    strategy.config,
                    market,
                    result_data,
                    account,
                    fill,
                )
            )

        for market, _ in remaining_items:
            engine_market = self._engine_market(account, market, config.risk.leverage)
            candle_input = next(
                raw_candles
                for raw_candles, candidate_market in market_inputs
                if candidate_market.symbol == market.symbol
            )
            candle_sets = (
                candle_input
                if isinstance(candle_input, dict)
                else {config.interval: candle_input}
            )
            candles = candle_sets.get(config.interval) or next(
                iter(candle_sets.values())
            )
            result_data = self.engine.evaluate(
                RuleStrategyEvaluationRequest(
                    config=config,
                    candles=candles,
                    candle_sets=candle_sets,
                    market=engine_market,
                )
            ).model_dump(mode="json")
            account, fill = self._apply_result(account, market, result_data)
            output.append(
                self._record_evaluation(
                    strategy_id,
                    tenant_id,
                    strategy.config,
                    market,
                    result_data,
                    account,
                    fill,
                )
            )
        return output

    def evaluations(
        self, strategy_id: str, tenant_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Return complete, durable explanations for each paper evaluation."""
        self._require_strategy(strategy_id, tenant_id)
        entries: list[dict[str, Any]] = []
        for journal in self.repository.get_evaluations(
            strategy_id, tenant_id, limit=limit
        ):
            result = journal.result or {}
            funnel_data = self._evaluation_funnel(result, journal.trades or [])
            entries.append(
                {
                    "strategy_id": strategy_id,
                    "evaluation_id": journal.evaluation_id,
                    **({"symbol": result["symbol"]} if "symbol" in result else {}),
                    "evaluated_at": journal.created_at,
                    "action": result.get("action", "no_op"),
                    "reason_code": result.get("reason_code", "unknown"),
                    "reason": result.get("reason", "No explanation was recorded."),
                    "conditions": result.get("conditions", []),
                    "indicators": result.get("indicators", {}),
                    "sizing": result.get("sizing", {}),
                    "funding": result.get("funding", {}),
                    "account": result.get("account", {}),
                    "entry_confirmation": result.get("entry_confirmation"),
                    "execution": result.get("execution"),
                    "execution_ledger": result.get("execution_ledger"),
                    "paper_fill": result.get("paper_fill"),
                    "trades": journal.trades or [],
                    **funnel_data,
                    **{
                        key: result[key]
                        for key in (
                            "stage",
                            "status",
                            "checked_at",
                            "next_check_at",
                        )
                        if key in result
                    },
                }
            )
        return entries

    def update_execution(
        self,
        tenant_id: str,
        strategy_id: str,
        evaluation_id: str,
        execution: dict[str, Any],
    ) -> None:
        """Persist execution only when all ownership identifiers match."""
        journal = self.repository.update_evaluation_execution(
            tenant_id, strategy_id, evaluation_id, execution
        )
        if journal is None:
            raise RuleStrategyNotFoundError(
                f"Evaluation '{evaluation_id}' was not found for rule strategy '{strategy_id}'"
            )

    @staticmethod
    def _evaluation_funnel(
        result: dict[str, Any], trades: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Normalize current and historical journals into one six-stage read model."""
        conditions = result.get("conditions") or []
        condition_category = "exit" if result.get("action") == "sell" else "indicator"
        indicator_conditions = [
            item for item in conditions if item.get("category") == condition_category
        ]
        risk_conditions = [item for item in conditions if item.get("category") == "risk"]
        is_sell = result.get("action") == "sell"
        confirmation = {} if is_sell else (result.get("entry_confirmation") or {})
        total = int(confirmation.get("enabled", len(indicator_conditions)))
        available = int(
            confirmation.get(
                "available",
                sum(item.get("state") != "unavailable" for item in indicator_conditions),
            )
        )
        matched = int(
            confirmation.get(
                "passed",
                sum(item.get("state") == "triggered" for item in indicator_conditions),
            )
        )
        exit_mode = result.get("exit_confirmation_mode", "any")
        required = int(
            confirmation.get(
                "required", 1 if is_sell and exit_mode == "any" and total else total
            )
        )
        summary = {
            "matched": matched,
            "total": total,
            "required": required,
            "available": available,
        }
        labels = {
            "strategy_run": "策略运行",
            "market_ready": "行情就绪",
            "conditions": "条件满足",
            "risk": "风控检查",
            "order_submission": "订单提交",
            "fill": "订单成交",
        }
        stages = [
            {
                "code": code,
                "label": label,
                "status": "pending",
                "detail": "尚未到达此阶段。",
            }
            for code, label in labels.items()
        ]

        def set_stage(code: str, status: str, detail: str) -> None:
            stage = next(item for item in stages if item["code"] == code)
            stage.update(status=status, detail=detail)

        set_stage("strategy_run", "passed", "策略已运行并记录本次检查。")
        diagnostic_stage = result.get("stage")
        if result.get("status") == "blocked" and diagnostic_stage in {
            "market_data",
            "account_sync",
        }:
            blocker = "market_ready" if diagnostic_stage == "market_data" else "risk"
            if blocker == "risk":
                set_stage("conditions", "pending", "账户同步失败，未执行条件评估。")
            set_stage(
                blocker,
                "blocked",
                result.get("reason", "同步暂不可用，已安全跳过。"),
            )
            return {
                "funnel": stages,
                "blocked_stage": blocker,
                "condition_summary": summary,
            }

        set_stage("market_ready", "passed", "行情数据已就绪。")
        has_signal = result.get("action") in {"buy", "sell"}
        risk_blocked = any(item.get("state") == "blocked" for item in risk_conditions)
        if risk_blocked:
            set_stage(
                "conditions", "passed", f"条件满足 {matched}/{total}，需要 {required} 项。"
            )
            detail = next(
                (item.get("detail") for item in risk_conditions if item.get("state") == "blocked"),
                "风控未通过。",
            )
            set_stage("risk", "blocked", detail)
            return {
                "funnel": stages,
                "blocked_stage": "risk",
                "condition_summary": summary,
            }
        if not has_signal:
            set_stage(
                "conditions",
                "blocked",
                f"条件满足 {matched}/{total}，需要 {required} 项。",
            )
            return {
                "funnel": stages,
                "blocked_stage": "conditions",
                "condition_summary": summary,
            }

        set_stage(
            "conditions", "passed", f"条件满足 {matched}/{total}，需要 {required} 项。"
        )
        set_stage("risk", "passed", "风控检查通过。")

        execution = result.get("execution") or {}
        if not isinstance(execution, dict):
            execution = {}
        order_status = str(execution.get("status") or "").lower()
        trade_filled = any(item.get("execution") == "paper_filled" for item in trades)
        submission_rejected = order_status in {"rejected", "failed", "stale"} or (
            execution.get("execution") == "blocked" and not order_status
        )
        fill_rejected = order_status in {"canceled", "cancelled"}
        submitted_statuses = {
            "submitted",
            "open",
            "partially_filled",
            "partial",
            "filled",
            "closed",
            "canceled",
            "cancelled",
        }
        submitted = trade_filled or order_status in submitted_statuses
        if submission_rejected:
            set_stage("order_submission", "rejected", "订单提交被拒绝或失败。")
            set_stage("fill", "rejected", "订单未成交。")
            blocked_stage = "order_submission"
        elif submitted:
            set_stage("order_submission", "passed", "订单已提交。")
            if fill_rejected:
                set_stage("fill", "rejected", "订单已取消，未完全成交。")
                blocked_stage = "fill"
            else:
                blocked_stage = None
                if trade_filled or order_status in {"filled", "closed"}:
                    set_stage("fill", "filled", "订单已成交。")
                elif order_status in {"partially_filled", "partial"}:
                    set_stage("fill", "partial", "订单部分成交。")
                else:
                    set_stage("fill", "pending", "订单待成交。")
        else:
            set_stage("order_submission", "pending", "订单提交结果待确认。")
            blocked_stage = None
        return {
            "funnel": stages,
            "blocked_stage": blocked_stage,
            "condition_summary": summary,
        }

    def account(self, strategy_id: str, tenant_id: str) -> dict[str, Any]:
        strategy = self._require_strategy(strategy_id, tenant_id)
        config = RuleStrategyConfig.model_validate(strategy.config)
        return self._current_paper_account(strategy, tenant_id, config).model_dump(
            mode="json"
        )

    def _current_paper_account(
        self,
        strategy: RuleStrategy,
        tenant_id: str,
        config: RuleStrategyConfig,
    ) -> RuleStrategyPaperAccount:
        """Load the durable account row; journal replay remains recovery-only."""
        state_reader = getattr(self.repository, "get_account_state", None)
        if state_reader is not None:
            current_state = state_reader(strategy.strategy_id, tenant_id)
            if current_state is not None:
                account, _risk = current_state
                try:
                    positions = {
                        symbol: RuleStrategyPaperPosition.model_validate(position)
                        for symbol, position in (account.positions or {}).items()
                    }
                    return RuleStrategyPaperAccount(
                        initial_capital_quote=account.allocation_quote,
                        quote_balance=account.quote_balance,
                        positions=positions,
                        realized_pnl_quote=account.realized_pnl_quote,
                        unrealized_pnl_quote=account.unrealized_pnl_quote,
                        equity_quote=account.equity_quote,
                    )
                except ValidationError:
                    self._halt_corrupt_account_state(strategy, tenant_id)
                    raise RuleStrategyNotRunningError(
                        "Strategy account state is corrupt and has been halted"
                    )
        return self._account_from_history(strategy, tenant_id, config)

    def _halt_corrupt_account_state(
        self, strategy: RuleStrategy, tenant_id: str
    ) -> None:
        """Fail closed when durable state cannot safely reconstruct balances."""
        state_reader = getattr(self.repository, "get_account_state", None)
        if state_reader is None:
            return
        current_state = state_reader(strategy.strategy_id, tenant_id)
        if current_state is None:
            return
        _account, risk = current_state
        risk.state = "halted"
        risk.reason_code = "corrupt_current_account_state"
        risk.reason_detail = "当前策略账户状态损坏，已停止执行。"
        updater = getattr(self.repository, "update_risk_state", None)
        if updater is not None:
            updater(risk)

    def _append_evaluation_statefully(
        self,
        journal: RuleStrategyEvaluationJournal,
        account: RuleStrategyPaperAccount | None,
        tenant_id: str,
        strategy_id: str,
    ) -> RuleStrategyEvaluationJournal:
        """Write paper account state, evaluation evidence, and event in one commit."""
        state_reader = getattr(self.repository, "get_account_state", None)
        state_writer = getattr(self.repository, "append_evaluation_with_state", None)
        if account is None or state_reader is None or state_writer is None:
            return self.repository.append_evaluation(journal)
        current_state = state_reader(strategy_id, tenant_id)
        if current_state is None:
            return self.repository.append_evaluation(journal)
        account_row, risk = current_state
        before_state = {
            "quote_balance": account_row.quote_balance,
            "equity_quote": account_row.equity_quote,
            "positions": account_row.positions,
        }
        account_row.quote_balance = account.quote_balance
        account_row.positions = {
            symbol: position.model_dump(mode="json")
            for symbol, position in account.positions.items()
        }
        account_row.realized_pnl_quote = account.realized_pnl_quote
        account_row.unrealized_pnl_quote = account.unrealized_pnl_quote
        account_row.equity_quote = account.equity_quote
        event = RuleStrategyEvent(
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            correlation_id=journal.evaluation_id,
            evaluation_id=journal.evaluation_id,
            actor="system",
            reason_code="evaluation_committed",
            before_state=before_state,
            after_state=account.model_dump(mode="json"),
        )
        return state_writer(journal, account_row, risk, event)

    def _account_from_history(
        self, strategy: RuleStrategy, tenant_id: str, config: RuleStrategyConfig
    ) -> RuleStrategyPaperAccount:
        # Account recovery is deliberately independent from bounded diagnostic
        # history. More than 100 diagnostics must not reset a durable ledger.
        account_query = getattr(self.repository, "get_latest_account_evaluations", None)
        journals = (
            account_query(strategy.strategy_id, tenant_id)
            if account_query is not None
            else self.repository.get_evaluations(
                strategy.strategy_id, tenant_id, limit=100_000
            )
        )
        required_fields = {
            "initial_capital_quote",
            "quote_balance",
            "positions",
            "realized_pnl_quote",
            "unrealized_pnl_quote",
            "equity_quote",
        }
        for journal in journals:
            raw_account = (journal.result or {}).get("account")
            # Demo diagnostics also use `account`, but describe an external
            # position rather than the server-owned paper ledger.
            if (
                not isinstance(raw_account, dict)
                or raw_account.get("source") == "okx_demo"
                or not required_fields.issubset(raw_account)
            ):
                continue
            try:
                return RuleStrategyPaperAccount.model_validate(raw_account)
            except ValidationError:
                # Skip corrupt snapshots and continue to an older valid one.
                continue
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
        account: RuleStrategyPaperAccount | dict[str, Any],
        fill: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result_data["symbol"] = market.symbol
        result_data["account"] = (
            account.model_dump(mode="json")
            if isinstance(account, RuleStrategyPaperAccount)
            else account
        )
        journal_record = RuleStrategyEvaluationJournal(
            evaluation_id=f"evaluation_{uuid4().hex}",
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            result=result_data,
            signals=result_data["conditions"],
            trades=self._trade_entries(result_data, fill),
            funding=[result_data["funding"]],
        )
        journal = self._append_evaluation_statefully(
            journal_record,
            account if isinstance(account, RuleStrategyPaperAccount) else None,
            tenant_id,
            strategy_id,
        )
        return {
            "strategy_id": strategy_id,
            "evaluation_id": journal.evaluation_id,
            "config": config,
            "symbol": market.symbol,
            **result_data,
        }

    @staticmethod
    def _mark_account_for_cycle(
        account: RuleStrategyPaperAccount,
        markets: list[RuleStrategyMarketSnapshot],
    ) -> RuleStrategyPaperAccount:
        prices = {market.symbol: market.price for market in markets}
        positions = {
            symbol: item.model_copy(
                update={
                    "mark_price": prices.get(symbol, item.mark_price),
                    "highest_price": max(
                        item.highest_price or item.entry_price,
                        prices.get(symbol, item.mark_price),
                    ),
                }
            )
            for symbol, item in account.positions.items()
        }
        unrealized = sum(
            item.quantity * (item.mark_price - item.entry_price)
            for item in positions.values()
        )
        equity = account.quote_balance + sum(
            item.quantity * item.mark_price for item in positions.values()
        )
        return account.model_copy(
            update={
                "positions": positions,
                "unrealized_pnl_quote": unrealized,
                "equity_quote": equity,
            }
        )

    @staticmethod
    def _engine_market(
        account: RuleStrategyPaperAccount,
        market: RuleStrategyMarketSnapshot,
        leverage: float,
    ) -> RuleStrategyEngineMarketSnapshot:
        position = account.positions.get(market.symbol)
        marked_positions = {
            symbol: item.model_copy(
                update={
                    "mark_price": market.price,
                    "highest_price": max(
                        item.highest_price or item.entry_price,
                        market.price,
                    ),
                }
            )
            if symbol == market.symbol
            else item
            for symbol, item in account.positions.items()
        }
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
            total_position_quote=sum(
                item.quantity * item.mark_price for item in marked_positions.values()
            ),
            position=RuleStrategyPosition(
                quantity=position.quantity if position else 0.0,
                entry_price=position.entry_price if position else None,
                highest_price=(
                    max(position.highest_price or position.entry_price, market.price)
                    if position
                    else None
                ),
                addition_count=position.addition_count if position else 0,
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

        action = result["action"]
        if action in {"buy", "entry", "add"}:
            quote_amount = float(result["sizing"]["requested_quote"])
            if quote_amount > account.quote_balance:
                raise ValueError("paper buy must be blocked before account mutation")
            quantity = quote_amount / market.price
            if quote_amount > 0 and existing is None:
                positions[market.symbol] = RuleStrategyPaperPosition(
                    quantity=quantity,
                    entry_price=market.price,
                    mark_price=market.price,
                    highest_price=market.price,
                )
                fill = {
                    "symbol": market.symbol,
                    "price": market.price,
                    "quantity": quantity,
                    "quote_amount": quote_amount,
                    "realized_pnl_quote": 0.0,
                }
                quote_balance = account.quote_balance - quote_amount
            elif quote_amount > 0 and existing is not None:
                combined_quantity = existing.quantity + quantity
                combined_cost = existing.quantity * existing.entry_price + quote_amount
                positions[market.symbol] = RuleStrategyPaperPosition(
                    quantity=combined_quantity,
                    entry_price=combined_cost / combined_quantity,
                    mark_price=market.price,
                    highest_price=max(
                        existing.highest_price or existing.entry_price,
                        market.price,
                    ),
                    addition_count=existing.addition_count + 1,
                )
                fill = {
                    "symbol": market.symbol,
                    "price": market.price,
                    "quantity": quantity,
                    "quote_amount": quote_amount,
                    "realized_pnl_quote": 0.0,
                }
                quote_balance = account.quote_balance - quote_amount
            else:
                quote_balance = account.quote_balance
        elif action in {"sell", "reduce", "close"} and existing is not None:
            if action == "reduce":
                requested_quote = float(result["sizing"].get("requested_quote", 0.0))
                quantity = min(
                    existing.quantity,
                    requested_quote / market.price if requested_quote > 0 else existing.quantity,
                )
            else:
                quantity = existing.quantity
            quote_amount = quantity * market.price
            realized_trade = quantity * (market.price - existing.entry_price)
            realized += realized_trade
            remaining_quantity = existing.quantity - quantity
            if remaining_quantity <= 1e-12:
                positions.pop(market.symbol)
            else:
                positions[market.symbol] = existing.model_copy(
                    update={
                        "quantity": remaining_quantity,
                        "mark_price": market.price,
                        "highest_price": max(
                            existing.highest_price or existing.entry_price,
                            market.price,
                        ),
                    }
                )
            quote_balance = account.quote_balance + quote_amount
            fill = {
                "symbol": market.symbol,
                "price": market.price,
                "quantity": quantity,
                "quote_amount": quote_amount,
                "realized_pnl_quote": realized_trade,
            }
        else:
            quote_balance = account.quote_balance

        positions = {
            symbol: item.model_copy(
                update={
                    "mark_price": market.price,
                    "highest_price": max(
                        item.highest_price or item.entry_price,
                        market.price,
                    ),
                }
            )
            if symbol == market.symbol
            else item
            for symbol, item in positions.items()
        }
        unrealized = sum(
            item.quantity * (item.mark_price - item.entry_price)
            for item in positions.values()
        )
        equity = quote_balance + sum(
            item.quantity * item.mark_price for item in positions.values()
        )
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
        def apply(strategy: RuleStrategy) -> None:
            strategy.status = status
            strategy.execution_generation = (strategy.execution_generation or 1) + 1
        return self._locked_mutate(strategy_id, tenant_id, apply)

    @staticmethod
    def _apply_stopped_update(
        strategy: RuleStrategy,
        name: str | None,
        description: str | None,
        config: RuleStrategyConfig | None,
    ) -> None:
        if strategy.status == "running":
            raise RuleStrategyRunningUpdateError(
                "Stop the strategy before updating its configuration"
            )
        RuleStrategyService._apply_update(strategy, name, description, config)

    @staticmethod
    def _apply_update(
        strategy: RuleStrategy, name: str | None, description: str | None,
        config: RuleStrategyConfig | None,
    ) -> None:
        if name is not None:
            strategy.name = name
        if description is not None:
            strategy.description = description
        if config is not None:
            strategy.config = config.model_dump(mode="json")
            strategy.execution_generation = (strategy.execution_generation or 1) + 1

    def _locked_mutate(self, strategy_id: str, tenant_id: str, apply: Any) -> dict[str, Any]:
        """Control-plane changes share the RuleStrategy lock with dispatch.

        Production repositories use a database transaction and row lock.  The
        small in-memory repository used by deterministic unit tests deliberately
        has no database session; it cannot race another process, so retain its
        repository-owned mutation semantics without opening the production DB.
        """
        if not hasattr(self.repository, "db_session"):
            strategy = self._require_strategy(strategy_id, tenant_id)
            apply(strategy)
            return self._strategy_data(self.repository.update(strategy))

        configured_session = self.repository.db_session
        session = configured_session or get_database_manager().get_session()
        owns_session = configured_session is None
        try:
            strategy = session.query(RuleStrategy).filter_by(
                strategy_id=strategy_id, tenant_id=tenant_id
            ).with_for_update().first()
            if strategy is None:
                raise RuleStrategyNotFoundError(f"Rule strategy '{strategy_id}' was not found")
            apply(strategy)
            session.commit()
            session.refresh(strategy)
            if owns_session:
                session.expunge(strategy)
            return self._strategy_data(strategy)
        except Exception:
            session.rollback()
            raise
        finally:
            if owns_session:
                session.close()

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
            "mode": config.execution.environment,
            "config": config.model_dump(mode="json"),
            "execution_generation": strategy.execution_generation,
            "account": self._account_from_history(
                strategy, strategy.tenant_id, config
            ).model_dump(mode="json"),
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
