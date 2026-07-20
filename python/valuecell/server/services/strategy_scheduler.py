"""Auto-scheduler: drives paper or explicitly bound OKX Demo rule-strategy evaluations."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyCandle,
    RuleStrategyConfig,
    RuleStrategyMarketSnapshot,
)
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyEvaluationJournal,
    RuleStrategyExecutionIntent,
)
from valuecell.server.db.repositories.rule_strategy_repository import (
    RuleStrategyRepository,
)
from valuecell.server.services.crypto_market_service import get_crypto_market_service
from valuecell.server.services.rule_strategy_service import (
    RuleStrategyService,
)
from valuecell.server.services.saas_access_service import TenantAccessService
from valuecell.server.services.sandbox_exchange_trading_service import (
    SandboxExchangeTradingService,
)

_MIN_INTERVAL_S = 60
_DEMO_SUBMISSION_TIMEOUT_S = 15
_SYNC_JOB_ID = "_scheduler_sync_running"
_INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _required_intervals(config: RuleStrategyConfig) -> set[str]:
    """Return every candle interval required by the configured rule set."""
    intervals = {config.interval}
    rules = config.advanced_rules
    if not rules.enabled:
        return intervals
    for rule in (
        rules.moving_average,
        rules.macd,
        rules.bollinger,
        rules.rsi,
        rules.momentum,
        rules.brar,
    ):
        if rule.enabled:
            intervals.add(rule.interval)
    return intervals


def _required_lookback(config: RuleStrategyConfig) -> int:
    """Fetch sufficient history for the largest configured technical window."""
    rules = config.advanced_rules
    if not rules.enabled:
        return 100
    return max(
        100,
        rules.moving_average.period,
        rules.macd.slow_window + rules.macd.signal_window,
        rules.bollinger.period,
        rules.rsi.period + 1,
        rules.momentum.period + 1,
        rules.brar.period + 1,
    )


def _market_data_unavailable_reason(symbol_result: Any | None) -> str:
    """Return an actionable reason for a safety-related evaluation skip."""
    if symbol_result is None:
        return "primary candles unavailable"
    if getattr(symbol_result, "freshness_status", None) != "fresh":
        return "primary candles stale age_ms={}".format(
            getattr(symbol_result, "freshness_age_ms", "unknown")
        )
    return "primary candles unavailable"


class StrategyScheduler:
    """Wraps AsyncIOScheduler to drive paper or explicitly bound OKX Demo strategies."""

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._scheduler.start()
        logger.info("StrategyScheduler started")

    async def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("StrategyScheduler stopped")

    # ------------------------------------------------------------------
    # Sync running strategies
    # ------------------------------------------------------------------

    def sync_running_strategies(self, db_session: Session) -> None:
        """Add jobs for new running strategies; remove jobs for stopped ones.

        Called periodically (every 60 s) from a separate IntervalTrigger job
        that passes a fresh session each time.
        """
        repository = RuleStrategyRepository(db_session=db_session)
        running_strategies = [
            strategy
            for strategy in repository.list_running()
            if TenantAccessService.access_for(db_session, strategy.tenant_id).active
        ]

        wanted_ids: set[str] = {s.strategy_id for s in running_strategies}
        # Application maintenance jobs share this scheduler but must never be
        # interpreted as strategy IDs and removed by strategy synchronization.
        existing_job_ids: set[str] = {
            job.id
            for job in self._scheduler.get_jobs()
            if not job.id.startswith("_scheduler_")
        }

        # Remove jobs for strategies no longer running
        for job_id in existing_job_ids - wanted_ids:
            self._scheduler.remove_job(job_id)
            logger.info("StrategyScheduler removed job strategy_id={}", job_id)

        # Add new jobs and replace jobs whose stored configuration changed.
        # Replacing every job on each sync postpones its next run indefinitely
        # when the sync frequency is shorter than the evaluation interval.
        for strategy in running_strategies:
            try:
                config = RuleStrategyConfig.model_validate(strategy.config)
            except ValueError as exc:
                logger.warning(
                    "StrategyScheduler skipped invalid config strategy_id={} err={}",
                    strategy.strategy_id,
                    exc,
                )
                continue
            interval_s = max(
                _MIN_INTERVAL_S,
                config.decide_interval_s or _INTERVAL_SECONDS[config.interval],
            )
            job_args = [
                strategy.strategy_id,
                strategy.tenant_id,
                config.model_dump(mode="json"),
            ]
            existing_job = self._scheduler.get_job(strategy.strategy_id)
            if existing_job is not None and list(existing_job.args) == job_args:
                continue
            self._scheduler.add_job(
                self._tick,
                trigger=IntervalTrigger(seconds=interval_s),
                id=strategy.strategy_id,
                args=job_args,
                replace_existing=True,
                coalesce=True,
                max_instances=1,
                next_run_time=datetime.now(timezone.utc),
            )
            logger.info(
                "StrategyScheduler added job strategy_id={} interval={}s",
                strategy.strategy_id,
                interval_s,
            )

    # ------------------------------------------------------------------
    # Tick (async coroutine, called by APScheduler asyncio executor)
    # ------------------------------------------------------------------

    async def _tick(
        self,
        strategy_id: str,
        tenant_id: str,
        config_dict: dict[str, Any],
    ) -> None:
        """Fetch OHLCV, evaluate rules, journal results, and safely route the chosen execution target."""
        session = get_database_manager().get_session()
        try:
            if not TenantAccessService.access_for(session, tenant_id).active:
                logger.info(
                    "StrategyScheduler skipped inactive tenant strategy_id={}",
                    strategy_id,
                )
                return
        finally:
            session.close()
        config = RuleStrategyConfig.model_validate(config_dict)
        symbols = config.symbols
        intervals = sorted(_required_intervals(config))
        lookback = _required_lookback(config)

        if not symbols:
            logger.warning(
                "StrategyScheduler tick skipped: no symbols configured strategy_id={}",
                strategy_id,
            )
            return

        market_service = get_crypto_market_service()
        try:
            fetched_sets = await asyncio.gather(
                *[
                    market_service.get_indicators(
                        symbols=symbols,
                        interval=interval,
                        lookback=lookback,
                    )
                    for interval in intervals
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "StrategyScheduler OHLCV fetch failed strategy_id={} err={}",
                strategy_id,
                exc,
            )
            return
        market_data_by_interval = {
            interval: {item.symbol: item for item in data.symbols}
            for interval, data in zip(intervals, fetched_sets)
        }

        service = RuleStrategyService()

        market_inputs: list[
            tuple[dict[str, list[RuleStrategyCandle]], RuleStrategyMarketSnapshot]
        ] = []
        for symbol in symbols:
            try:
                candle_sets: dict[str, list[RuleStrategyCandle]] = {}
                latest_price: float | None = None
                for interval, by_symbol in market_data_by_interval.items():
                    symbol_result = by_symbol.get(symbol)
                    if symbol_result is None:
                        continue
                    candles = [
                        RuleStrategyCandle(
                            timestamp_ms=c.ts,
                            open=c.open,
                            high=c.high,
                            low=c.low,
                            close=c.close,
                            volume=c.volume,
                        )
                        for c in symbol_result.candles
                    ]
                    if candles:
                        candle_sets[interval] = candles
                    if (
                        interval == config.interval
                        and symbol_result.latest_price is not None
                    ):
                        latest_price = symbol_result.latest_price
                primary_result = market_data_by_interval[config.interval].get(symbol)
                primary_candles = candle_sets.get(config.interval)
                if (
                    not primary_candles
                    or primary_result is None
                    or primary_result.freshness_status != "fresh"
                ):
                    reason = _market_data_unavailable_reason(primary_result)
                    logger.warning(
                        "StrategyScheduler market-data unavailable strategy_id={} symbol={} "
                        "interval={} reason={} failed_symbols={}",
                        strategy_id,
                        symbol,
                        config.interval,
                        reason,
                        {
                            interval: data.failed_symbols.get(symbol)
                            for interval, data in zip(intervals, fetched_sets)
                            if symbol in data.failed_symbols
                        },
                    )
                    continue
                market_snapshot = RuleStrategyMarketSnapshot(
                    symbol=symbol,
                    price=latest_price or primary_candles[-1].close,
                    funding_rate=0.0,
                )
                market_inputs.append((candle_sets, market_snapshot))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "StrategyScheduler tick error strategy_id={} symbol={} err={}",
                    strategy_id,
                    symbol,
                    exc,
                )
                continue

        if not market_inputs:
            return
        try:
            results = service.evaluate_batch(strategy_id, tenant_id, market_inputs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "StrategyScheduler batch tick failed strategy_id={} err={}",
                strategy_id,
                exc,
            )
            return
        for result in results:
            logger.info(
                "StrategyScheduler tick strategy_id={} symbol={} action={} evaluation_id={}",
                strategy_id,
                result.get("symbol"),
                result.get("action"),
                result.get("evaluation_id"),
            )
            if result.get("action") in {"buy", "sell"}:
                market = next(
                    candidate_market
                    for _, candidate_market in market_inputs
                    if candidate_market.symbol == result["symbol"]
                )
                candle_input = next(
                    raw_candles
                    for raw_candles, candidate_market in market_inputs
                    if candidate_market.symbol == result["symbol"]
                )
                candle_sets = (
                    candle_input
                    if isinstance(candle_input, dict)
                    else {config.interval: candle_input}
                )
                candles = candle_sets.get(config.interval) or next(
                    iter(candle_sets.values())
                )
                execution = await self._execute_signal(
                    tenant_id,
                    strategy_id,
                    config,
                    result["symbol"],
                    result["action"],
                    Decimal(str(result["sizing"]["requested_quote"])),
                    Decimal(str(market.price)),
                    candles[-1].timestamp_ms,
                    result["evaluation_id"],
                )
                if config.execution.environment == "okx_demo":
                    # The durable intent/order is the execution audit trail; never
                    # mutate the paper journal's trade JSON for exchange routing.
                    result["execution"] = execution
                else:
                    self._record_execution(
                        tenant_id,
                        strategy_id,
                        result["evaluation_id"],
                        execution,
                    )

    @staticmethod
    def _record_execution(
        tenant_id: str, strategy_id: str, evaluation_id: str, execution: dict[str, Any]
    ) -> None:
        session = get_database_manager().get_session()
        try:
            journal = (
                session.query(RuleStrategyEvaluationJournal)
                .filter_by(
                    evaluation_id=evaluation_id,
                    tenant_id=tenant_id,
                    strategy_id=strategy_id,
                )
                .first()
            )
            if journal is None:
                raise RuntimeError(
                    "Evaluation journal was not found for execution recording"
                )
            if not journal.trades:
                raise RuntimeError(
                    "Evaluation journal has no target trade for execution recording"
                )
            entries = list(journal.trades)
            entries[-1] = {**entries[-1], **execution}
            journal.trades = entries
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    async def _execute_signal(
        tenant_id: str,
        strategy_id: str,
        config: RuleStrategyConfig,
        symbol: str,
        action: str,
        quote_amount: Decimal,
        price: Decimal,
        candle_timestamp_ms: int,
        evaluation_id: str | None = None,
    ) -> dict[str, Any]:
        if config.execution.environment == "paper":
            return {
                "execution": "paper_filled",
                "execution_ledger": "paper",
                "paper_fill": True,
                "sandbox": False,
            }
        execution = await StrategyScheduler._execute_okx_demo_signal(
            tenant_id,
            strategy_id,
            config,
            symbol,
            action,
            quote_amount,
            price,
            candle_timestamp_ms,
            evaluation_id,
        )
        return {
            **execution,
            "execution_ledger": "okx_demo",
            # Submission/acceptance is deliberately not a local fill. The
            # exchange order reconciliation path is the authority for fills.
            "paper_fill": False,
        }

    @staticmethod
    async def _execute_okx_demo_signal(
        tenant_id: str,
        strategy_id: str,
        config: RuleStrategyConfig,
        symbol: str,
        action: str,
        quote_amount: Decimal,
        price: Decimal,
        candle_timestamp_ms: int,
        evaluation_id: str | None = None,
    ) -> dict[str, Any]:
        """Fence, reserve, and route while the strategy row lock is held.

        The lock deliberately spans one bounded remote submission.  This makes a
        committed stop/configuration change mutually exclusive with dispatch;
        there is no post-commit reread window in which an old job can submit.
        """
        execution_config = config.execution
        if (
            execution_config.environment != "okx_demo"
            or not execution_config.sandbox_connection_id
        ):
            return {
                "execution": "blocked",
                "reason": "OKX Demo execution is not configured",
            }
        if action not in {"buy", "sell"}:
            return {"execution": "blocked", "reason": "Signal is not executable"}
        requested_quote = min(
            quote_amount, Decimal(str(execution_config.max_order_quote_amount))
        )
        # An exchange route is valid only when it can be attributed to the
        # durable evaluation which produced it.  Never retain a direct-call
        # compatibility path here: it would bypass the intent/fencing protocol.
        if evaluation_id is None:
            return {
                "execution": "blocked",
                "sandbox": True,
                "reason": "durable evaluation is required for strategy execution",
            }
        session = get_database_manager().get_session()
        try:
            # This lock is acquired only after all market/evaluation I/O. A stale
            # captured job config is never authoritative for execution.
            strategy = (
                session.query(RuleStrategy)
                .filter_by(strategy_id=strategy_id, tenant_id=tenant_id)
                .with_for_update()
                .first()
            )
            if strategy is None or strategy.status != "running":
                return {
                    "execution": "blocked",
                    "sandbox": True,
                    "reason": "strategy is no longer running",
                }
            fresh_config = RuleStrategyConfig.model_validate(strategy.config)
            fresh_execution = fresh_config.execution
            if (
                fresh_execution.environment != "okx_demo"
                or fresh_execution.sandbox_connection_id
                != execution_config.sandbox_connection_id
            ):
                return {
                    "execution": "blocked",
                    "sandbox": True,
                    "reason": "strategy execution configuration changed",
                }
            requested_quote = min(
                quote_amount, Decimal(str(fresh_execution.max_order_quote_amount))
            )
            intents = (
                session.query(RuleStrategyExecutionIntent)
                .filter_by(
                    tenant_id=tenant_id,
                    strategy_id=strategy_id,
                    credential_id=fresh_execution.sandbox_connection_id,
                )
                .all()
            )
            # Daily throughput reserves every non-rejected/non-stale strategy
            # intent, including filled/closed records.  Total is active exposure
            # only, but conservatively includes pending/submitting/unknown.
            active_statuses = {
                "pending",
                "submitting",
                "submission_unknown",
                "submitted",
                "open",
                "partially_filled",
            }
            existing_total = sum(
                Decimal(str(row.requested_quote))
                for row in intents
                if row.status in active_statuses
            )
            daily_cutoff = datetime.now(timezone.utc) - timedelta(days=1)
            daily_total = sum(
                Decimal(str(row.requested_quote))
                for row in intents
                if row.status not in {"rejected", "stale"}
                and (
                    getattr(row, "created_at", None) is None
                    or row.created_at.replace(tzinfo=timezone.utc) >= daily_cutoff
                )
            )
            if daily_total + requested_quote > Decimal(
                str(fresh_execution.max_daily_quote_amount)
            ):
                return {
                    "execution": "blocked",
                    "sandbox": True,
                    "reason": "OKX Demo strategy daily limit reached",
                }
            if existing_total + requested_quote > Decimal(
                str(fresh_execution.max_total_quote_amount)
            ):
                return {
                    "execution": "blocked",
                    "sandbox": True,
                    "reason": "OKX Demo strategy total limit reached",
                }
            material = f"{strategy_id}:{candle_timestamp_ms}:{symbol.upper()}:{action}"
            key = (
                "vc-demo-"
                + __import__("hashlib")
                .sha256(material.encode("utf-8"))
                .hexdigest()[:48]
            )
            intent = (
                session.query(RuleStrategyExecutionIntent)
                .filter_by(
                    strategy_id=strategy_id,
                    evaluation_id=evaluation_id,
                    execution_generation=strategy.execution_generation,
                )
                .first()
            )
            if intent is None:
                intent = RuleStrategyExecutionIntent(
                    strategy_id=strategy_id,
                    evaluation_id=evaluation_id,
                    execution_generation=strategy.execution_generation,
                    execution_source="rule_strategy",
                    tenant_id=tenant_id,
                    credential_id=fresh_execution.sandbox_connection_id,
                    idempotency_key=key,
                    symbol=symbol.replace("-", "/"),
                    side=action,
                    order_type="market",
                    requested_quote=str(requested_quote),
                    status="pending",
                    request_payload={"candle_timestamp_ms": candle_timestamp_ms},
                )
                session.add(intent)
            # The intent is an audit/outbox record, not merely a row staged in
            # the transaction that will make a remote request. Commit it before
            # progressing, so a process crash can always be reconciled by its
            # idempotency key. The strategy lock is intentionally reacquired
            # below for the final no-stale-submit critical section.
            session.commit()
            intent = (
                session.query(RuleStrategyExecutionIntent)
                .filter_by(
                    strategy_id=strategy_id,
                    evaluation_id=evaluation_id,
                    execution_generation=strategy.execution_generation,
                )
                .first()
            )
            if intent is None:
                return {
                    "execution": "blocked",
                    "sandbox": True,
                    "reason": "execution intent unavailable",
                }
            # Persist the conservative in-flight state before remote I/O. A
            # restart will reconcile it; it is never automatically re-submitted.
            if intent.status == "pending":
                intent.status = "submitting"
                intent.attempt_count = (intent.attempt_count or 0) + 1
                intent.submitted_at = datetime.now(timezone.utc)
                session.commit()
            strategy = (
                session.query(RuleStrategy)
                .filter_by(strategy_id=strategy_id, tenant_id=tenant_id)
                .with_for_update()
                .first()
            )
            if (
                strategy is None
                or strategy.status != "running"
                or strategy.execution_generation != intent.execution_generation
            ):
                intent.status = "stale"
                intent.error_code = "stale_generation"
                intent.terminal_at = datetime.now(timezone.utc)
                session.commit()
                return {
                    "execution": "blocked",
                    "sandbox": True,
                    "reason": "strategy execution generation changed",
                }
            fresh_config = RuleStrategyConfig.model_validate(strategy.config)
            fresh_execution = fresh_config.execution
            if (
                fresh_execution.environment != "okx_demo"
                or fresh_execution.sandbox_connection_id != intent.credential_id
            ):
                intent.status = "stale"
                intent.error_code = "stale_execution_configuration"
                intent.terminal_at = datetime.now(timezone.utc)
                session.commit()
                return {
                    "execution": "blocked",
                    "sandbox": True,
                    "reason": "strategy execution configuration changed",
                }
            # The service owns the deadline so it can commit submission_unknown
            # before returning. An outer wait_for here would cancel it and let
            # this session's rollback erase a request that may reach the venue.
            # Final fence and remote I/O share this transaction/row lock. This is
            # deliberately after the durable intent commits above: a stop/update
            # cannot interleave after this lock is acquired and before the bounded
            # exchange create call returns.
            order = await SandboxExchangeTradingService(session).submit_order(
                tenant_id,
                fresh_execution.sandbox_connection_id,
                key,
                symbol.replace("-", "/"),
                action,
                "market",
                requested_quote,
                None,
                intent=intent,
                fenced=True,
                submission_timeout_s=_DEMO_SUBMISSION_TIMEOUT_S,
            )
            return {
                "execution": "okx_demo_submitted"
                if order.get("status") not in {"failed", "rejected", "stale"}
                else "blocked",
                "sandbox": True,
                "execution_intent_id": intent.id,
                "order_id": order.get("id"),
                "status": order.get("status"),
                "error_code": order.get("error_code"),
            }
        except Exception:
            session.rollback()
            return {
                "execution": "blocked",
                "sandbox": True,
                "reason": "OKX Demo order was rejected",
            }
        finally:
            session.close()
