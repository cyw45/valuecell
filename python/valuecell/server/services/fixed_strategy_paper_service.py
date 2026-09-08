"""Persist fixed-strategy decisions and adapt Demo spot execution safely."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from valuecell.server.api.schemas.fixed_strategy import (
    FixedEngineInput,
    FixedStrategySignal,
)
from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.db.models.rule_strategy import RuleStrategyEvaluationJournal
from valuecell.server.db.models.fixed_strategy_paper import FixedPaperPosition
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.repositories.rule_strategy_repository import RuleStrategyRepository
from valuecell.server.services.fixed_strategy_dispatcher import evaluate_fixed_strategy
from valuecell.server.services.fixed_strategy_paper_ledger import FixedPaperLedger

_FIXED_KINDS = {"dual_ma_trend", "pair_rotation", "leader_breakout"}


class FixedDemoExecutionAdapter:
    """Translate executable fixed signals to the shared Demo spot boundary."""

    def __init__(self, execution_boundary: Callable[..., Awaitable[dict[str, Any]]]) -> None:
        self._execution_boundary = execution_boundary

    async def execute(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        config: RuleStrategyConfig,
        signal: FixedStrategySignal,
        price: Decimal,
        candle_timestamp_ms: int,
        evaluation_id: str,
    ) -> dict[str, Any] | None:
        """Route long entry/exit only; Demo spot must never synthesize a short."""
        if signal.action == "short_entry":
            return {
                "execution": "blocked_execution_environment",
                "execution_ledger": "okx_demo",
                "paper_fill": False,
                "sandbox": True,
                "reason": "OKX Demo spot execution does not support fixed-strategy short actions",
            }
        action = {"long_entry": "buy", "exit": "sell"}.get(signal.action)
        if action is None:
            return None
        arguments = (
            tenant_id,
            strategy_id,
            config,
            signal.symbol,
            action,
            Decimal(str(config.risk.order_quote_amount)),
            price,
            candle_timestamp_ms,
            evaluation_id,
        )
        if signal.action == "exit":
            return await self._execution_boundary(
                *arguments, close_all_attributed=True
            )
        return await self._execution_boundary(*arguments)


class FixedPaperEvaluationService:
    """Record fixed-strategy decisions as durable explainable evidence."""

    def __init__(self, repository: RuleStrategyRepository | None = None) -> None:
        self._repository = repository or RuleStrategyRepository()

    def evaluate_and_record(
        self,
        *,
        strategy_id: str,
        tenant_id: str,
        strategy_kind: str,
        batch_id: str | None,
        request: FixedEngineInput,
        btc_request: list[Any] | None = None,
        environment: str = "paper",
    ) -> tuple[FixedStrategySignal, str]:
        """Evaluate and persist one signal before any optional Demo submission."""
        if strategy_kind not in _FIXED_KINDS:
            raise ValueError(f"Unsupported fixed strategy kind: {strategy_kind}")
        if strategy_kind == "leader_breakout":
            if btc_request is None:
                raise ValueError("leader_breakout requires BTC candle facts")
            signal = evaluate_fixed_strategy(
                strategy_kind, request, btc_candles=btc_request
            )
        else:
            signal = evaluate_fixed_strategy(strategy_kind, request)
        evaluation_key = (
            f"fixed-paper:{tenant_id}:{strategy_id}:{batch_id or 'unbatched'}:"
            f"{strategy_kind}:{signal.symbol}:{signal.observed_at.isoformat()}"
        )
        evaluation_id = f"fixed_{uuid5(NAMESPACE_URL, evaluation_key).hex}"
        existing_reader = getattr(self._repository, "get_evaluation", None)
        if callable(existing_reader):
            existing = existing_reader(evaluation_id, strategy_id, tenant_id)
            if existing is not None:
                return signal, evaluation_id
        result = {
            "strategy_kind": signal.kind,
            "symbol": signal.symbol,
            "action": signal.action,
            "reason_code": signal.reason_code,
            "reason": signal.reason,
            "conditions": [
                condition.model_dump(mode="json") for condition in signal.conditions
            ],
            "indicators": signal.indicators,
            "pair": signal.pair,
            "execution_block_reason": signal.execution_block_reason,
            "execution_ledger": "paper_signal_only" if environment == "paper" else "okx_demo",
            "paper_fill": False,
        }
        journal = self._repository.append_evaluation(
            RuleStrategyEvaluationJournal(
            evaluation_id=evaluation_id,
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                result=result,
                signals=result["conditions"],
                trades=[],
                funding=[],
            )
        )
        return signal, journal.evaluation_id

    def update_execution(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        evaluation_id: str,
        execution: dict[str, Any],
    ) -> None:
        """Attach Demo execution facts without altering fixed decision evidence."""
        journal = self._repository.update_evaluation_execution(
            tenant_id, strategy_id, evaluation_id, execution
        )
        if journal is None:
            raise LookupError(f"Fixed evaluation '{evaluation_id}' was not found")

    def record_paper_fill(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str | None,
        signal: FixedStrategySignal,
        evaluation_id: str,
        initial_capital_quote: Decimal,
        price: Decimal,
        order_quote_amount: Decimal,
    ) -> dict[str, Any]:
        """Apply an executable Paper signal to the isolated fixed ledger."""
        if batch_id is None:
            return {
                "execution": "paper_blocked_missing_batch",
                "execution_ledger": "paper",
                "paper_fill": False,
            }
        if signal.action not in {"long_entry", "short_entry", "exit"}:
            return {
                "execution": "paper_signal_only",
                "execution_ledger": "paper",
                "paper_fill": False,
            }
        session = get_database_manager().get_session()
        try:
            ledger = FixedPaperLedger(session)
            account = ledger.account(
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                initial_capital_quote=initial_capital_quote,
            )
            position = (
                session.query(FixedPaperPosition)
                .filter_by(
                    tenant_id=tenant_id,
                    strategy_id=strategy_id,
                    batch_id=batch_id,
                    symbol=signal.symbol,
                    status="open",
                )
                .first()
            )
            quantity = (
                Decimal(str(position.quantity))
                if signal.action == "exit" and position is not None
                else order_quote_amount / price
            )
            fill = ledger.apply_signal(
                account=account,
                signal=signal,
                evaluation_id=evaluation_id,
                price=price,
                quantity=quantity,
            )
            session.commit()
            return {
                "execution": "paper_filled" if fill is not None else "paper_signal_only",
                "execution_ledger": "paper",
                "paper_fill": fill is not None,
                "fill_id": fill.fill_id if fill is not None else None,
                "filled_side": fill.side if fill is not None else None,
                "filled_quantity": float(fill.quantity) if fill is not None else None,
                "filled_price": float(fill.price) if fill is not None else None,
            }
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
