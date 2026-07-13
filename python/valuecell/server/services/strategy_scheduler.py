"""Auto-scheduler: drives paper rule strategy evaluations on a background clock."""

from __future__ import annotations

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
    RuleStrategyPosition,
)
from valuecell.server.db.models.rule_strategy import RuleStrategy
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.rule_strategy import RuleStrategyEvaluationJournal
from valuecell.server.services.live_execution_service import LiveExecutionService
from valuecell.server.db.repositories.rule_strategy_repository import (
    RuleStrategyRepository,
)
from valuecell.server.services.crypto_market_service import get_crypto_market_service
from valuecell.server.services.rule_strategy_service import (
    RuleStrategyService,
)

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


class StrategyScheduler:
    """Wraps AsyncIOScheduler to drive paper rule strategy evaluations."""

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
        running_strategies = repository.list_running()

        wanted_ids: set[str] = {s.strategy_id for s in running_strategies}
        existing_job_ids: set[str] = {
            job.id
            for job in self._scheduler.get_jobs()
            if job.id != _SYNC_JOB_ID
        }

        # Remove jobs for strategies no longer running
        for job_id in existing_job_ids - wanted_ids:
            self._scheduler.remove_job(job_id)
            logger.info("StrategyScheduler removed job strategy_id={}", job_id)

        # Add jobs for newly running strategies
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
            self._scheduler.add_job(
                self._tick,
                trigger=IntervalTrigger(seconds=interval_s),
                id=strategy.strategy_id,
                args=[strategy.strategy_id, strategy.tenant_id, config.model_dump()],
                replace_existing=True,
                coalesce=True,
                max_instances=1,
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
        """Fetch OHLCV, evaluate rule engine, journal result — paper only."""
        config = RuleStrategyConfig.model_validate(config_dict)
        symbols = config.symbols
        interval = config.interval
        lookback = 100
        account_balance = config.initial_capital_quote

        if not symbols:
            logger.warning(
                "StrategyScheduler tick skipped: no symbols configured strategy_id={}",
                strategy_id,
            )
            return

        market_service = get_crypto_market_service()
        snapshot = market_service.get_default_snapshot()
        if (
            snapshot is not None
            and snapshot.data.interval == interval
        ):
            by_symbol = {item.symbol: item for item in snapshot.data.symbols}
            market_data = type(snapshot.data)(
                interval=snapshot.data.interval,
                lookback=min(lookback, snapshot.data.lookback),
                providers=snapshot.data.providers,
                symbols=[
                    item.model_copy(update={"candles": item.candles[-lookback:]})
                    for symbol in symbols
                    if (item := by_symbol.get(symbol)) is not None
                ],
                failed_symbols={
                    symbol: "market snapshot unavailable"
                    for symbol in symbols
                    if symbol not in by_symbol
                },
                snapshot_fetched_at=snapshot.fetched_at.isoformat(),
            )
        else:
            try:
                market_data = await market_service.get_indicators(
                    symbols=symbols,
                    interval=interval,
                    lookback=lookback,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "StrategyScheduler OHLCV fetch failed strategy_id={} err={}",
                    strategy_id,
                    exc,
                )
                return

        service = RuleStrategyService()

        market_inputs: list[tuple[list[RuleStrategyCandle], RuleStrategyMarketSnapshot]] = []
        for symbol_result in market_data.symbols:
            try:
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
                if not candles:
                    logger.warning(
                        "StrategyScheduler skipping empty candles symbol={} strategy_id={}",
                        symbol_result.symbol,
                        strategy_id,
                    )
                    continue

                latest_price = symbol_result.latest_price or candles[-1].close
                market_snapshot = RuleStrategyMarketSnapshot(
                    symbol=symbol_result.symbol,
                    price=latest_price,
                    funding_rate=0.0,
                )

                market_inputs.append((candles, market_snapshot))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "StrategyScheduler tick error strategy_id={} symbol={} err={}",
                    strategy_id,
                    symbol_result.symbol,
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

    @staticmethod
    async def _execute_live_signal(
        tenant_id: str, strategy_id: str, symbol: str, action: str,
        quote_amount: Decimal, price: Decimal, candle_timestamp_ms: int,
        evaluation_id: str,
    ) -> dict[str, Any]:
        session = get_database_manager().get_session()
        execution: dict[str, Any]
        try:
            execution = await LiveExecutionService(session).execute_strategy_signal(
                tenant_id, strategy_id, symbol, action, quote_amount, price,
                candle_timestamp_ms,
            )
        except Exception:
            session.rollback()
            execution = {"execution": "blocked", "reason": "实盘执行服务不可用"}
        try:
            journal = session.query(RuleStrategyEvaluationJournal).filter_by(
                evaluation_id=evaluation_id, tenant_id=tenant_id, strategy_id=strategy_id
            ).first()
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
