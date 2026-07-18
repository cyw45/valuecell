"""Auto-scheduler: drives paper or explicitly bound OKX Demo rule-strategy evaluations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import asyncio
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from sqlalchemy.orm import Session

from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyCandle,
    RuleStrategyConfig,
    RuleStrategyMarketSnapshot,
)
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.rule_strategy import RuleStrategyEvaluationJournal
from valuecell.server.db.models.sandbox_exchange_order import SandboxExchangeOrder
from valuecell.server.services.live_execution_service import LiveExecutionService
from valuecell.server.services.sandbox_exchange_trading_service import (
    SandboxExchangeTradingService,
)
from valuecell.server.db.repositories.rule_strategy_repository import (
    RuleStrategyRepository,
)
from valuecell.server.services.crypto_market_service import get_crypto_market_service
from valuecell.server.services.rule_strategy_service import (
    RuleStrategyService,
)
from valuecell.server.services.saas_access_service import TenantAccessService

_MIN_INTERVAL_S = 60
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
        existing_job_ids: set[str] = {
            job.id for job in self._scheduler.get_jobs() if job.id != _SYNC_JOB_ID
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
                primary_candles = candle_sets.get(config.interval)
                if not primary_candles:
                    logger.warning(
                        "StrategyScheduler skipping empty candles symbol={} strategy_id={}",
                        symbol,
                        strategy_id,
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
                    self._record_execution(tenant_id, strategy_id, result["evaluation_id"], execution)

    @staticmethod
    def _record_execution(
        tenant_id: str, strategy_id: str, evaluation_id: str, execution: dict[str, Any]
    ) -> None:
        session = get_database_manager().get_session()
        try:
            journal = session.query(RuleStrategyEvaluationJournal).filter_by(
                evaluation_id=evaluation_id, tenant_id=tenant_id, strategy_id=strategy_id
            ).first()
            if journal is not None and journal.trades:
                entries = list(journal.trades)
                entries[-1] = {**entries[-1], **execution}
                journal.trades = entries
                session.commit()
        except Exception:
            session.rollback()
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
        evaluation_id: str,
    ) -> dict[str, Any]:
        if config.execution.environment == "okx_demo":
            execution = await StrategyScheduler._execute_okx_demo_signal(
                tenant_id,
                strategy_id,
                config,
                symbol,
                action,
                quote_amount,
                price,
                candle_timestamp_ms,
            )
        else:
            execution = await StrategyScheduler._execute_live_signal(
                tenant_id,
                strategy_id,
                symbol,
                action,
                quote_amount,
                price,
                candle_timestamp_ms,
                evaluation_id,
            )
        return execution

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
    ) -> dict[str, Any]:
        """Submit an idempotent, bounded order through an encrypted OKX Demo connection."""
        execution_config = config.execution
        if execution_config.environment != "okx_demo" or not execution_config.sandbox_connection_id:
            return {"execution": "blocked", "reason": "OKX Demo execution is not configured"}
        if action not in {"buy", "sell"}:
            return {"execution": "blocked", "reason": "Signal is not executable"}
        requested_quote = min(quote_amount, Decimal(str(execution_config.max_order_quote_amount)))
        session = get_database_manager().get_session()
        try:
            orders = session.query(SandboxExchangeOrder).filter_by(
                tenant_id=tenant_id,
                credential_id=execution_config.sandbox_connection_id,
                sandbox=True,
            ).all()
            active_orders = [row for row in orders if row.status not in {"failed", "rejected"}]
            existing_total = sum(Decimal(str(row.requested_quote)) for row in active_orders)
            daily_cutoff = datetime.now(timezone.utc) - timedelta(days=1)
            daily_total = sum(
                Decimal(str(row.requested_quote))
                for row in active_orders
                if getattr(row, "created_at", None) is None
                or row.created_at.replace(tzinfo=timezone.utc) >= daily_cutoff
            )
            if daily_total + requested_quote > Decimal(str(execution_config.max_daily_quote_amount)):
                return {"execution": "blocked", "sandbox": True, "reason": "OKX Demo strategy daily limit reached"}
            if existing_total + requested_quote > Decimal(str(execution_config.max_total_quote_amount)):
                return {"execution": "blocked", "sandbox": True, "reason": "OKX Demo strategy total limit reached"}
            material = f"{strategy_id}:{candle_timestamp_ms}:{symbol.upper()}:{action}"
            client_order_id = "vc-demo-" + __import__("hashlib").sha256(material.encode("utf-8")).hexdigest()[:48]
            order = await SandboxExchangeTradingService(session).submit_order(
                tenant_id,
                execution_config.sandbox_connection_id,
                client_order_id,
                symbol.replace("-", "/"),
                action,
                "market",
                requested_quote,
                None,
            )
            return {
                "execution": "okx_demo_submitted"
                if order.get("status") not in {"failed", "rejected"}
                else "blocked",
                "sandbox": True,
                "order_id": order.get("id"),
                "status": order.get("status"),
                "error_code": order.get("error_code"),
            }
        except Exception:
            session.rollback()
            return {"execution": "blocked", "sandbox": True, "reason": "OKX Demo order was rejected"}
        finally:
            session.close()

    @staticmethod
    async def _execute_live_signal(
        tenant_id: str,
        strategy_id: str,
        symbol: str,
        action: str,
        quote_amount: Decimal,
        price: Decimal,
        candle_timestamp_ms: int,
        evaluation_id: str,
    ) -> dict[str, Any]:
        session = get_database_manager().get_session()
        execution: dict[str, Any]
        try:
            execution = await LiveExecutionService(session).execute_strategy_signal(
                tenant_id,
                strategy_id,
                symbol,
                action,
                quote_amount,
                price,
                candle_timestamp_ms,
            )
        except Exception:
            session.rollback()
            execution = {"execution": "blocked", "reason": "实盘执行服务不可用"}
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
            if journal is not None:
                entries = list(journal.trades or [])
                if entries:
                    entries[-1] = {**entries[-1], **execution}
                    journal.trades = entries
                    session.commit()
            return execution
        except Exception:
            session.rollback()
            return {"execution": "blocked", "reason": "实盘执行服务不可用"}
        finally:
            session.close()
