"""Reproducible, injected-data backtesting for tenant-owned rule strategies.

This module deliberately has no exchange client or credential dependency. A caller
must materialize candles and pass them to :meth:`submit`; replay then operates
only on those immutable stored bars.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol
from uuid import uuid4

from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyConfig,
    RuleStrategyEngineMarketSnapshot,
    RuleStrategyEvaluationRequest,
    RuleStrategyPosition,
)
from valuecell.server.api.schemas.rule_strategy_validation import (
    RuleStrategyValidationCandle,
    RuleStrategyValidationCreateRequest,
    RuleStrategyValidationDatasetInput,
    RuleStrategyValidationDatasetSummary,
    RuleStrategyValidationFillView,
    RuleStrategyValidationPointView,
    RuleStrategyValidationRunDetail,
    RuleStrategyValidationRunSummary,
    RuleStrategyValidationWindow,
)
from valuecell.server.db.models.rule_strategy_validation import (
    RuleStrategyValidationDataset,
    RuleStrategyValidationFill,
    RuleStrategyValidationPoint,
    RuleStrategyValidationRun,
)
from valuecell.server.db.repositories.rule_strategy_validation_repository import (
    RuleStrategyValidationRepository,
)
from valuecell.server.services.rule_engine import RuleEngine


RULE_ENGINE_VERSION = "rule_engine_v1"
DEFAULT_FEE_RATE = 0.001
DEFAULT_SLIPPAGE_RATE = 0.001
_DEFAULT_LEASE_SECONDS = 15 * 60
_LEASE_RENEW_EVERY_EVENTS = 512
_EPSILON = 1e-12
_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


class RuleStrategyValidationError(Exception):
    """Base error for user-visible validation failures."""


class RuleStrategyValidationNotFoundError(RuleStrategyValidationError):
    """Raised when a tenant cannot read a validation run or strategy."""


class RuleStrategyValidationWindowError(RuleStrategyValidationError):
    """Raised when an OOS end date cannot define a fully closed UTC window."""


class RuleStrategyValidationCoverageError(RuleStrategyValidationError):
    """Raised before queueing when injected bars do not prove complete coverage."""

    def __init__(
        self,
        code: str,
        detail: str,
        report: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.report = report or {}


class RuleStrategyValidationNotCompletedError(RuleStrategyValidationError):
    """Raised when an export is requested before a run reaches completion."""


class RuleStrategyValidationLeaseLostError(RuleStrategyValidationError):
    """Raised when a worker no longer owns the durable validation lease."""


class RuleStrategyValidationDataMaterializer(Protocol):
    """Optional injected source boundary; implementations must not use credentials."""

    def materialize(
        self,
        *,
        strategy_id: str,
        tenant_id: str,
        request: RuleStrategyValidationCreateRequest,
        window: RuleStrategyValidationWindow,
        required_intervals: set[str],
    ) -> Sequence[RuleStrategyValidationDatasetInput]: ...


@dataclass
class _ReplayPosition:
    quantity: float
    entry_price: float
    mark_price: float
    highest_price: float
    addition_count: int = 0


@dataclass
class _ReplayAccount:
    initial_capital_quote: float
    quote_balance: float
    realized_pnl_quote: float = 0.0
    positions: dict[str, _ReplayPosition] = field(default_factory=dict)

    def mark(self, prices: Mapping[str, float]) -> None:
        for symbol, position in self.positions.items():
            mark_price = prices.get(symbol, position.mark_price)
            position.mark_price = mark_price
            position.highest_price = max(position.highest_price, mark_price)

    @property
    def position_quote(self) -> float:
        return math.fsum(
            position.quantity * position.mark_price
            for position in self.positions.values()
        )

    @property
    def unrealized_pnl_quote(self) -> float:
        return math.fsum(
            position.quantity * (position.mark_price - position.entry_price)
            for position in self.positions.values()
        )

    @property
    def equity_quote(self) -> float:
        return self.quote_balance + self.position_quote

    def snapshot(self) -> dict[str, Any]:
        return {
            "initial_capital_quote": _finite(self.initial_capital_quote),
            "quote_balance": _finite(self.quote_balance),
            "positions": {
                symbol: {
                    "quantity": _finite(position.quantity),
                    "entry_price": _finite(position.entry_price),
                    "mark_price": _finite(position.mark_price),
                    "highest_price": _finite(position.highest_price),
                    "addition_count": position.addition_count,
                }
                for symbol, position in sorted(self.positions.items())
            },
            "realized_pnl_quote": _finite(self.realized_pnl_quote),
            "unrealized_pnl_quote": _finite(self.unrealized_pnl_quote),
            "equity_quote": _finite(self.equity_quote),
        }


@dataclass(frozen=True)
class _PendingFill:
    symbol: str
    window: Literal["in_sample", "out_of_sample"]
    leg_kind: Literal["entry", "add", "reduce", "close"]
    side: Literal["buy", "sell"]
    decision_at_ms: int
    decision_price: float
    requested_quote: float
    requested_quantity: float | None
    reason_code: str
    fill_open: float


class RuleStrategyValidationService:
    """Materialize, lease, and replay validation without market-network access."""

    def __init__(
        self,
        repository: RuleStrategyValidationRepository | None = None,
        engine: RuleEngine | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        engine_version: str = RULE_ENGINE_VERSION,
    ) -> None:
        if not 0 <= fee_rate < 1:
            raise ValueError("fee_rate must be in [0, 1)")
        if not 0 <= slippage_rate < 1:
            raise ValueError("slippage_rate must be in [0, 1)")
        if not engine_version:
            raise ValueError("engine_version is required")
        self._repository = repository or RuleStrategyValidationRepository()
        self._engine = engine or RuleEngine()
        self._now = now or (lambda: datetime.now(UTC))
        self._fee_rate = fee_rate
        self._slippage_rate = slippage_rate
        self._engine_version = engine_version

    @property
    def repository(self) -> RuleStrategyValidationRepository:
        """Expose the narrow persistence boundary for scheduler wiring."""

        return self._repository

    def submit(
        self,
        strategy_id: str,
        tenant_id: str,
        request: RuleStrategyValidationCreateRequest,
        datasets: Sequence[RuleStrategyValidationDatasetInput],
    ) -> RuleStrategyValidationRunDetail:
        """Validate injected coverage and queue an immutable run atomically.

        This is intentionally synchronous only through data validation and local
        persistence. The expensive replay is performed later by ``execute``.
        """

        window = derive_validation_window(request.oos_end_date, now=self._now())
        strategy = self._repository.strategy(strategy_id, tenant_id)
        if strategy is None:
            raise RuleStrategyValidationNotFoundError(
                f"Rule strategy '{strategy_id}' was not found"
            )
        try:
            config = RuleStrategyConfig.model_validate(strategy.config)
        except Exception as exc:
            raise RuleStrategyValidationError(
                "strategy configuration cannot be frozen for validation"
            ) from exc
        config_snapshot = config.model_dump(mode="json")
        selected_symbols = list(request.selected_symbols)
        configured_symbols = set(config.symbols)
        unsupported = sorted(set(selected_symbols) - configured_symbols)
        if unsupported:
            raise RuleStrategyValidationCoverageError(
                "selected_symbol_not_configured",
                "selected symbols must be configured on the strategy",
                {"unsupported_symbols": unsupported},
            )
        required_intervals = required_intervals_for_config(config_snapshot)
        normalized = self._materialize_datasets(
            strategy_id,
            tenant_id,
            window,
            selected_symbols,
            required_intervals,
            datasets,
        )
        run_id = str(uuid4())
        assumptions = {
            "execution_model": "bar_close_decision_next_bar_open_fill",
            "fee_rate": self._fee_rate,
            "slippage_rate": self._slippage_rate,
            "risk_free_rate": 0.0,
            "position_model": "spot_long_only",
            "candle_timestamp_convention": "utc_bar_open",
        }
        config_fingerprint = fingerprint(config_snapshot)
        assumptions_fingerprint = fingerprint(assumptions)
        data_fingerprint = fingerprint(
            [
                {
                    "source_provider": row.source_provider,
                    "symbol": row.symbol,
                    "interval": row.interval,
                    "start_at": _utc_iso(window.in_sample_start_at),
                    "end_at_exclusive": _utc_iso(window.out_of_sample_end_at_exclusive),
                    "content_hash": row.content_hash,
                    "page_manifest": row.page_manifest,
                    "coverage_manifest": row.coverage_manifest,
                    "retrieved_at": _utc_iso(row.retrieved_at),
                }
                for row in sorted(
                    normalized,
                    key=lambda item: (item.symbol, item.interval, item.source_provider),
                )
            ]
        )
        run = RuleStrategyValidationRun(
            run_id=run_id,
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            status="pending",
            source_preference=request.source_preference,
            selected_symbols=selected_symbols,
            config_json=config_snapshot,
            config_fingerprint=config_fingerprint,
            assumptions=assumptions,
            assumptions_fingerprint=assumptions_fingerprint,
            data_fingerprint=data_fingerprint,
            initial_capital_quote=config.initial_capital_quote,
            template_id=_as_optional_string(config_snapshot.get("template_id")),
            template_version=_as_optional_int(config_snapshot.get("template_version")),
            indicator_formula_version=(
                _as_optional_string(config_snapshot.get("indicator_formula_version"))
                or "legacy_rule_engine"
            ),
            engine_version=self._engine_version,
            in_sample_start_at=window.in_sample_start_at,
            in_sample_end_at_exclusive=window.in_sample_end_at_exclusive,
            out_of_sample_start_at=window.out_of_sample_start_at,
            out_of_sample_end_at_exclusive=window.out_of_sample_end_at_exclusive,
        )
        rows = [
            RuleStrategyValidationDataset(
                run_id=run_id,
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                source_provider=item.source_provider,
                symbol=item.symbol,
                interval=item.interval,
                start_at=window.in_sample_start_at,
                end_at_exclusive=window.out_of_sample_end_at_exclusive,
                bar_count=len(item.bars),
                bars=[bar.model_dump(mode="json") for bar in item.bars],
                page_manifest=item.page_manifest,
                coverage_manifest=item.coverage_manifest,
                content_hash=item.content_hash,
                retrieved_at=item.retrieved_at,
            )
            for item in normalized
        ]
        persisted = self._repository.create(run, rows)
        return self._detail(persisted)

    def submit_materialized(
        self,
        strategy_id: str,
        tenant_id: str,
        request: RuleStrategyValidationCreateRequest,
        materializer: RuleStrategyValidationDataMaterializer,
    ) -> RuleStrategyValidationRunDetail:
        """Convenience boundary for a future server-owned materializer.

        The materializer is injected by the caller. This service remains unable
        to perform any production exchange or credential operation itself.
        """

        window = derive_validation_window(request.oos_end_date, now=self._now())
        strategy = self._repository.strategy(strategy_id, tenant_id)
        if strategy is None:
            raise RuleStrategyValidationNotFoundError(
                f"Rule strategy '{strategy_id}' was not found"
            )
        config = RuleStrategyConfig.model_validate(strategy.config)
        datasets = materializer.materialize(
            strategy_id=strategy_id,
            tenant_id=tenant_id,
            request=request,
            window=window,
            required_intervals=required_intervals_for_config(
                config.model_dump(mode="json")
            ),
        )
        return self.submit(strategy_id, tenant_id, request, datasets)

    def get(self, run_id: str, tenant_id: str) -> RuleStrategyValidationRunDetail:
        run = self._repository.get(run_id, tenant_id)
        if run is None:
            raise RuleStrategyValidationNotFoundError(
                f"Validation run '{run_id}' was not found"
            )
        return self._detail(run)

    def list(
        self, strategy_id: str, tenant_id: str, *, limit: int = 100
    ) -> list[RuleStrategyValidationRunSummary]:
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        # Check the parent strategy first so a foreign strategy cannot be probed.
        if self._repository.strategy(strategy_id, tenant_id) is None:
            raise RuleStrategyValidationNotFoundError(
                f"Rule strategy '{strategy_id}' was not found"
            )
        return [self._summary(run) for run in self._repository.list(strategy_id, tenant_id, limit=limit)]

    def datasets(
        self, run_id: str, tenant_id: str
    ) -> list[RuleStrategyValidationDatasetSummary]:
        self._require_run(run_id, tenant_id)
        return [
            RuleStrategyValidationDatasetSummary(
                dataset_id=row.dataset_id,
                run_id=row.run_id,
                source_provider=row.source_provider,
                symbol=row.symbol,
                interval=row.interval,
                start_at=_utc(row.start_at),
                end_at_exclusive=_utc(row.end_at_exclusive),
                bar_count=row.bar_count,
                content_hash=row.content_hash,
                coverage_manifest=dict(row.coverage_manifest or {}),
                page_manifest=list(row.page_manifest or []),
                retrieved_at=_utc(row.retrieved_at),
            )
            for row in self._repository.datasets(run_id, tenant_id)
        ]

    def points(
        self, run_id: str, tenant_id: str
    ) -> list[RuleStrategyValidationPointView]:
        self._require_run(run_id, tenant_id)
        return [
            RuleStrategyValidationPointView(
                sequence=row.sequence,
                window=row.window,
                observed_at=_utc(row.observed_at),
                equity_quote=row.equity_quote,
                cash_quote=row.cash_quote,
                position_quote=row.position_quote,
                drawdown_pct=row.drawdown_pct,
                account_snapshot=dict(row.account_snapshot or {}),
                decisions=dict(row.decisions or {}),
            )
            for row in self._repository.points(run_id, tenant_id)
        ]

    def fills(
        self, run_id: str, tenant_id: str
    ) -> list[RuleStrategyValidationFillView]:
        self._require_run(run_id, tenant_id)
        return [self._fill_view(row) for row in self._repository.fills(run_id, tenant_id)]

    def cancel(self, run_id: str, tenant_id: str) -> RuleStrategyValidationRunDetail:
        run = self._repository.cancel(run_id, tenant_id, now=self._now())
        if run is None:
            raise RuleStrategyValidationNotFoundError(
                f"Validation run '{run_id}' was not found"
            )
        return self._detail(run)

    def execute(
        self,
        run_id: str,
        tenant_id: str,
        *,
        worker_id: str | None = None,
    ) -> RuleStrategyValidationRunDetail:
        """Claim and replay one run; no request leaves the process during replay."""

        owner_id = worker_id or str(uuid4())
        claimed = self._repository.claim(
            run_id,
            tenant_id,
            owner_id,
            now=self._now(),
            lease_duration=_lease_duration(),
        )
        if claimed is None:
            return self.get(run_id, tenant_id)
        try:
            points, fills, metrics = self._replay(claimed, owner_id)
            artifact_fingerprint = fingerprint(
                {
                    "run_id": claimed.run_id,
                    "data_fingerprint": claimed.data_fingerprint,
                    "config_fingerprint": claimed.config_fingerprint,
                    "assumptions_fingerprint": claimed.assumptions_fingerprint,
                    "metrics": metrics,
                    "points": [_point_fingerprint_payload(point) for point in points],
                    "fills": [_fill_fingerprint_payload(fill) for fill in fills],
                }
            )
            finished = self._repository.finish(
                claimed.run_id,
                claimed.tenant_id,
                owner_id,
                points=points,
                fills=fills,
                metrics=metrics,
                artifact_fingerprint=artifact_fingerprint,
                now=self._now(),
            )
            if finished is None:
                raise RuleStrategyValidationLeaseLostError(
                    "validation lease was lost before completion"
                )
            return self._detail(finished)
        except RuleStrategyValidationLeaseLostError:
            raise
        except Exception as exc:
            failed = self._repository.fail(
                claimed.run_id,
                claimed.tenant_id,
                owner_id,
                _error_code(exc),
                str(exc),
                now=self._now(),
            )
            if failed is None:
                raise RuleStrategyValidationLeaseLostError(
                    "validation lease was lost while recording failure"
                ) from exc
            return self._detail(failed)

    def _materialize_datasets(
        self,
        strategy_id: str,
        tenant_id: str,
        window: RuleStrategyValidationWindow,
        selected_symbols: list[str],
        required_intervals: set[str],
        datasets: Sequence[RuleStrategyValidationDatasetInput],
    ) -> list[_NormalizedDataset]:
        expected_keys = {
            (symbol, interval)
            for symbol in selected_symbols
            for interval in required_intervals
        }
        actual_keys = [(dataset.symbol, dataset.interval) for dataset in datasets]
        duplicates = sorted(
            key for key in set(actual_keys) if actual_keys.count(key) > 1
        )
        unexpected = sorted(set(actual_keys) - expected_keys)
        missing = sorted(expected_keys - set(actual_keys))
        if duplicates or unexpected or missing:
            raise RuleStrategyValidationCoverageError(
                "dataset_matrix_incomplete",
                "every selected symbol requires exactly one complete dataset for every required interval",
                {
                    "strategy_id": strategy_id,
                    "tenant_id": tenant_id,
                    "required": [list(key) for key in sorted(expected_keys)],
                    "duplicates": [list(key) for key in duplicates],
                    "unexpected": [list(key) for key in unexpected],
                    "missing": [list(key) for key in missing],
                },
            )
        normalized: list[_NormalizedDataset] = []
        for dataset in datasets:
            coverage_manifest = validate_dataset_coverage(dataset, window)
            bars = tuple(dataset.bars)
            normalized.append(
                _NormalizedDataset(
                    source_provider=dataset.source_provider,
                    symbol=dataset.symbol,
                    interval=dataset.interval,
                    bars=bars,
                    page_manifest=[dict(item) for item in dataset.page_manifest],
                    coverage_manifest=coverage_manifest,
                    content_hash=fingerprint(
                        [bar.model_dump(mode="json") for bar in bars]
                    ),
                    retrieved_at=_utc(dataset.retrieved_at),
                )
            )
        return normalized

    def _replay(
        self,
        run: RuleStrategyValidationRun,
        worker_id: str,
    ) -> tuple[list[RuleStrategyValidationPoint], list[RuleStrategyValidationFill], dict[str, Any]]:
        config = RuleStrategyConfig.model_validate(run.config_json)
        assumptions = dict(run.assumptions or {})
        fee_rate = _rate(assumptions.get("fee_rate"), "fee_rate")
        slippage_rate = _rate(assumptions.get("slippage_rate"), "slippage_rate")
        primary_interval = config.interval
        primary_ms = _INTERVAL_MS[primary_interval]
        datasets = self._repository.datasets(run.run_id, run.tenant_id)
        series = _dataset_series(datasets)
        symbols = list(run.selected_symbols or [])
        if not symbols:
            raise RuleStrategyValidationError("validation run has no selected symbols")
        for symbol in symbols:
            if (symbol, primary_interval) not in series:
                raise RuleStrategyValidationCoverageError(
                    "primary_dataset_missing",
                    f"missing primary validation dataset for {symbol}",
                )
        primary = {symbol: series[(symbol, primary_interval)] for symbol in symbols}
        primary_lengths = {len(items) for items in primary.values()}
        if len(primary_lengths) != 1:
            raise RuleStrategyValidationCoverageError(
                "primary_dataset_length_mismatch",
                "selected symbols must share a complete primary bar clock",
            )
        primary_count = next(iter(primary_lengths))
        if primary_count < 2:
            raise RuleStrategyValidationCoverageError(
                "primary_dataset_too_short",
                "at least two primary bars are required for next-bar execution",
            )
        for index in range(primary_count):
            timestamps = {primary[symbol][index].timestamp_ms for symbol in symbols}
            if len(timestamps) != 1:
                raise RuleStrategyValidationCoverageError(
                    "primary_dataset_clock_mismatch",
                    "selected symbols must share primary bar timestamps",
                )
        history_limit = engine_history_limit(config.model_dump(mode="json"))
        secondary_close_times = {
            key: [bar.timestamp_ms + _INTERVAL_MS[key[1]] for bar in bars]
            for key, bars in series.items()
        }
        account = _ReplayAccount(
            initial_capital_quote=run.initial_capital_quote,
            quote_balance=run.initial_capital_quote,
        )
        points: list[RuleStrategyValidationPoint] = []
        fills: list[RuleStrategyValidationFill] = []
        pending: dict[int, list[_PendingFill]] = defaultdict(list)
        point_sequence = 0
        fill_sequence = 0
        high_water = {
            "in_sample": account.equity_quote,
            "out_of_sample": account.equity_quote,
        }
        points.append(
            self._point(
                run,
                point_sequence,
                "in_sample",
                run.in_sample_start_at,
                account,
                high_water["in_sample"],
                {"event": "in_sample_start"},
            )
        )
        point_sequence += 1
        oos_start_ms = _datetime_to_ms(run.out_of_sample_start_at)
        for index in range(primary_count):
            first_bar = primary[symbols[0]][index]
            bar_open_ms = first_bar.timestamp_ms
            decision_at_ms = bar_open_ms + primary_ms
            window = _window_for_bar_open(run, bar_open_ms)
            if bar_open_ms == oos_start_ms:
                account.mark({symbol: primary[symbol][index].open for symbol in symbols})
                high_water["out_of_sample"] = account.equity_quote
                points.append(
                    self._point(
                        run,
                        point_sequence,
                        "out_of_sample",
                        _datetime_from_ms(bar_open_ms),
                        account,
                        high_water["out_of_sample"],
                        {"event": "out_of_sample_start"},
                    )
                )
                point_sequence += 1
            for scheduled in sorted(pending.pop(bar_open_ms, []), key=lambda item: item.symbol):
                replay_fill = _apply_pending_fill(
                    account,
                    scheduled,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                )
                if replay_fill is None:
                    continue
                fills.append(
                    RuleStrategyValidationFill(
                        run_id=run.run_id,
                        tenant_id=run.tenant_id,
                        strategy_id=run.strategy_id,
                        sequence=fill_sequence,
                        window=scheduled.window,
                        symbol=scheduled.symbol,
                        leg_kind=scheduled.leg_kind,
                        side=scheduled.side,
                        decision_at=_datetime_from_ms(scheduled.decision_at_ms),
                        filled_at=_datetime_from_ms(bar_open_ms),
                        decision_price=replay_fill.decision_price,
                        fill_price=replay_fill.fill_price,
                        quantity=replay_fill.quantity,
                        quote_amount=replay_fill.quote_amount,
                        fee_quote=replay_fill.fee_quote,
                        slippage_pct=slippage_rate * 100,
                        realized_pnl_quote=replay_fill.realized_pnl_quote,
                        reason_code=scheduled.reason_code,
                        account_before=replay_fill.account_before,
                        account_after=replay_fill.account_after,
                    )
                )
                fill_sequence += 1
            account.mark({symbol: primary[symbol][index].close for symbol in symbols})
            decisions: dict[str, Any] = {}
            for symbol in sorted(symbols):
                result = self._evaluate_symbol(
                    config,
                    account,
                    symbol,
                    primary[symbol][index],
                    series,
                    secondary_close_times,
                    decision_at_ms,
                    history_limit,
                )
                result_data = result.model_dump(mode="json")
                decisions[symbol] = _decision_evidence(result_data)
                if index + 1 >= primary_count:
                    decisions[symbol]["execution"] = "not_scheduled_no_next_bar"
                    continue
                next_bar = primary[symbol][index + 1]
                if _window_for_bar_open(run, next_bar.timestamp_ms) != window:
                    decisions[symbol]["execution"] = "not_scheduled_window_boundary"
                    continue
                planned = _pending_fill_from_result(
                    symbol=symbol,
                    window=window,
                    decision_at_ms=decision_at_ms,
                    decision_price=primary[symbol][index].close,
                    next_bar_open=next_bar.open,
                    result=result_data,
                    position_open=symbol in account.positions,
                )
                if planned is None:
                    decisions[symbol]["execution"] = "not_scheduled_no_action"
                else:
                    pending[next_bar.timestamp_ms].append(planned)
                    decisions[symbol]["execution"] = "scheduled_next_bar"
            high_water[window] = max(high_water[window], account.equity_quote)
            points.append(
                self._point(
                    run,
                    point_sequence,
                    window,
                    _datetime_from_ms(decision_at_ms),
                    account,
                    high_water[window],
                    {"symbols": decisions},
                )
            )
            point_sequence += 1
            if index and index % _LEASE_RENEW_EVERY_EVENTS == 0:
                if self._repository.cancel_requested(run.run_id, run.tenant_id, worker_id):
                    raise RuleStrategyValidationError("validation cancellation was requested")
                if not self._repository.renew_lease(
                    run.run_id,
                    run.tenant_id,
                    worker_id,
                    now=self._now(),
                    lease_duration=_lease_duration(),
                ):
                    raise RuleStrategyValidationLeaseLostError(
                        "validation lease was lost during replay"
                    )
        metrics = {
            "in_sample": calculate_window_metrics(
                [row for row in points if row.window == "in_sample"],
                [row for row in fills if row.window == "in_sample"],
            ),
            "out_of_sample": calculate_window_metrics(
                [row for row in points if row.window == "out_of_sample"],
                [row for row in fills if row.window == "out_of_sample"],
            ),
        }
        return points, fills, metrics

    def _evaluate_symbol(
        self,
        config: RuleStrategyConfig,
        account: _ReplayAccount,
        symbol: str,
        current: RuleStrategyValidationCandle,
        series: Mapping[tuple[str, str], tuple[RuleStrategyValidationCandle, ...]],
        close_times: Mapping[tuple[str, str], list[int]],
        decision_at_ms: int,
        history_limit: int,
    ):
        primary_key = (symbol, config.interval)
        primary_items = series[primary_key]
        primary_end = bisect_right(close_times[primary_key], decision_at_ms)
        primary_history = list(primary_items[max(0, primary_end - history_limit) : primary_end])
        candle_sets: dict[str, list[RuleStrategyValidationCandle]] = {}
        for (dataset_symbol, interval), items in series.items():
            if dataset_symbol != symbol or interval == config.interval:
                continue
            end = bisect_right(close_times[(dataset_symbol, interval)], decision_at_ms)
            if end:
                candle_sets[interval] = list(items[max(0, end - history_limit) : end])
        position = account.positions.get(symbol)
        leverage = config.risk.leverage
        market = RuleStrategyEngineMarketSnapshot(
            symbol=symbol,
            price=current.close,
            funding_rate=0.0,
            equity_quote=max(0.0, account.equity_quote),
            quote_balance=max(0.0, account.quote_balance / leverage),
            open_position_count=len(account.positions),
            total_position_quote=max(0.0, account.position_quote),
            position=RuleStrategyPosition(
                quantity=position.quantity if position is not None else 0.0,
                entry_price=position.entry_price if position is not None else None,
                highest_price=position.highest_price if position is not None else None,
                addition_count=position.addition_count if position is not None else 0,
            ),
        )
        return self._engine.evaluate(
            RuleStrategyEvaluationRequest(
                config=config,
                candles=primary_history,
                candle_sets=candle_sets,
                market=market,
            )
        )

    @staticmethod
    def _point(
        run: RuleStrategyValidationRun,
        sequence: int,
        window: Literal["in_sample", "out_of_sample"],
        observed_at: datetime,
        account: _ReplayAccount,
        high_water: float,
        decisions: dict[str, Any],
    ) -> RuleStrategyValidationPoint:
        equity = max(0.0, account.equity_quote)
        drawdown = (
            max(0.0, (high_water - equity) / high_water * 100)
            if high_water > _EPSILON
            else 0.0
        )
        return RuleStrategyValidationPoint(
            run_id=run.run_id,
            tenant_id=run.tenant_id,
            strategy_id=run.strategy_id,
            sequence=sequence,
            window=window,
            observed_at=_utc(observed_at),
            equity_quote=_finite(equity),
            cash_quote=_finite(max(0.0, account.quote_balance)),
            position_quote=_finite(max(0.0, account.position_quote)),
            drawdown_pct=_finite(drawdown),
            account_snapshot=account.snapshot(),
            decisions=decisions,
        )

    def _require_run(self, run_id: str, tenant_id: str) -> RuleStrategyValidationRun:
        run = self._repository.get(run_id, tenant_id)
        if run is None:
            raise RuleStrategyValidationNotFoundError(
                f"Validation run '{run_id}' was not found"
            )
        return run

    @staticmethod
    def _summary(run: RuleStrategyValidationRun) -> RuleStrategyValidationRunSummary:
        return RuleStrategyValidationRunSummary(
            run_id=run.run_id,
            strategy_id=run.strategy_id,
            status=run.status,
            source_preference=run.source_preference,
            selected_symbols=list(run.selected_symbols or []),
            window=_window_from_run(run),
            initial_capital_quote=run.initial_capital_quote,
            data_fingerprint=run.data_fingerprint,
            config_fingerprint=run.config_fingerprint,
            assumptions_fingerprint=run.assumptions_fingerprint,
            artifact_fingerprint=run.artifact_fingerprint,
            metrics=dict(run.metrics) if isinstance(run.metrics, Mapping) else None,
            error_code=run.error_code,
            error_detail=run.error_detail,
            created_at=_utc(run.created_at),
            started_at=_utc_or_none(run.started_at),
            completed_at=_utc_or_none(run.completed_at),
        )

    @classmethod
    def _detail(cls, run: RuleStrategyValidationRun) -> RuleStrategyValidationRunDetail:
        summary = cls._summary(run)
        return RuleStrategyValidationRunDetail(
            **summary.model_dump(),
            config_snapshot=dict(run.config_json or {}),
            assumptions=dict(run.assumptions or {}),
            template_id=run.template_id,
            template_version=run.template_version,
            indicator_formula_version=run.indicator_formula_version,
            engine_version=run.engine_version,
        )

    @staticmethod
    def _fill_view(row: RuleStrategyValidationFill) -> RuleStrategyValidationFillView:
        return RuleStrategyValidationFillView(
            sequence=row.sequence,
            window=row.window,
            symbol=row.symbol,
            leg_kind=row.leg_kind,
            side=row.side,
            decision_at=_utc(row.decision_at),
            filled_at=_utc(row.filled_at),
            decision_price=row.decision_price,
            fill_price=row.fill_price,
            quantity=row.quantity,
            quote_amount=row.quote_amount,
            fee_quote=row.fee_quote,
            slippage_pct=row.slippage_pct,
            realized_pnl_quote=row.realized_pnl_quote,
            reason_code=row.reason_code,
            account_before=dict(row.account_before or {}),
            account_after=dict(row.account_after or {}),
        )


@dataclass(frozen=True)
class _NormalizedDataset:
    source_provider: str
    symbol: str
    interval: str
    bars: tuple[RuleStrategyValidationCandle, ...]
    page_manifest: list[dict[str, Any]]
    coverage_manifest: dict[str, Any]
    content_hash: str
    retrieved_at: datetime


@dataclass(frozen=True)
class _AppliedFill:
    decision_price: float
    fill_price: float
    quantity: float
    quote_amount: float
    fee_quote: float
    realized_pnl_quote: float
    account_before: dict[str, Any]
    account_after: dict[str, Any]


def derive_validation_window(
    oos_end_date: date,
    *,
    now: datetime | None = None,
) -> RuleStrategyValidationWindow:
    """Derive exact contiguous 24-calendar-month IS plus 3-month OOS UTC spans.

    ``oos_end_date`` is inclusive and must precede the current UTC date, so no
    partial UTC day can enter a run. Month shifting is calendar-aware rather than
    an approximation based on days.
    """

    timestamp = _utc(now or datetime.now(UTC))
    if oos_end_date >= timestamp.date():
        raise RuleStrategyValidationWindowError(
            "oos_end_date must be a fully closed UTC day"
        )
    oos_end_exclusive = datetime(
        oos_end_date.year,
        oos_end_date.month,
        oos_end_date.day,
        tzinfo=UTC,
    )
    from datetime import timedelta

    oos_end_exclusive += timedelta(days=1)
    oos_start = _shift_months(oos_end_exclusive, -3)
    is_start = _shift_months(oos_start, -24)
    return RuleStrategyValidationWindow(
        in_sample_start_at=is_start,
        in_sample_end_at_exclusive=oos_start,
        out_of_sample_start_at=oos_start,
        out_of_sample_end_at_exclusive=oos_end_exclusive,
    )


def required_intervals_for_config(config: Mapping[str, Any]) -> set[str]:
    """Collect active RuleEngine intervals from an immutable config JSON."""

    intervals: set[str] = set()

    def visit(value: Any, key: str | None = None, enabled: bool = True) -> None:
        if isinstance(value, Mapping):
            if value.get("enabled") is False:
                return
            for child_key, child_value in value.items():
                visit(child_value, str(child_key), enabled)
        elif isinstance(value, list):
            for child in value:
                visit(child, key, enabled)
        elif enabled and isinstance(value, str) and key is not None:
            lowered = key.lower()
            if (
                lowered == "interval"
                or lowered.endswith("_interval")
                or lowered.endswith("_timeframe")
            ) and value in _INTERVAL_MS:
                intervals.add(value)

    visit(config)
    primary = config.get("interval")
    if not isinstance(primary, str) or primary not in _INTERVAL_MS:
        raise RuleStrategyValidationError("strategy has no supported primary interval")
    intervals.add(primary)
    return intervals


def engine_history_limit(config: Mapping[str, Any]) -> int:
    """Bound per-decision input without losing the largest declared lookback."""

    maximum = 2

    def visit(value: Any, parent_key: str | None = None) -> None:
        nonlocal maximum
        if isinstance(value, Mapping):
            slow = value.get("slow_window", value.get("slow_period"))
            signal = value.get("signal_window", value.get("signal_period"))
            if all(isinstance(item, int) and item > 0 for item in (slow, signal)):
                maximum = max(maximum, int(slow) + int(signal) + 2)
            period = value.get("period")
            lookback = value.get("lookback", 0)
            if isinstance(period, int) and period > 0:
                maximum = max(maximum, period + (lookback if isinstance(lookback, int) else 0) + 2)
            for key, child in value.items():
                visit(child, str(key))
        elif isinstance(value, list):
            for child in value:
                visit(child, parent_key)
        elif isinstance(value, int) and value > 0 and parent_key is not None:
            lowered = parent_key.lower()
            if "window" in lowered or "period" in lowered or "lookback" in lowered:
                maximum = max(maximum, value + 2)

    visit(config)
    return min(5_000, maximum)


def validate_dataset_coverage(
    dataset: RuleStrategyValidationDatasetInput,
    window: RuleStrategyValidationWindow,
) -> dict[str, Any]:
    """Reject gaps, duplicates, bounds violations, and off-grid bar timestamps."""

    interval_ms = _INTERVAL_MS[dataset.interval]
    start_ms = _datetime_to_ms(window.in_sample_start_at)
    end_ms = _datetime_to_ms(window.out_of_sample_end_at_exclusive)
    if (end_ms - start_ms) % interval_ms:
        raise RuleStrategyValidationCoverageError(
            "window_interval_misaligned",
            "validation window cannot be represented by the requested interval",
            {"interval": dataset.interval},
        )
    expected_count = (end_ms - start_ms) // interval_ms
    timestamps = [bar.timestamp_ms for bar in dataset.bars]
    off_grid = [
        timestamp
        for timestamp in timestamps
        if timestamp < start_ms
        or timestamp >= end_ms
        or (timestamp - start_ms) % interval_ms != 0
    ]
    present = set(timestamps)
    missing_ranges = _missing_ranges(start_ms, end_ms, interval_ms, present)
    duplicate_count = len(timestamps) - len(present)
    manifest = {
        "requested_start_at": _utc_iso(window.in_sample_start_at),
        "requested_end_at_exclusive": _utc_iso(window.out_of_sample_end_at_exclusive),
        "interval": dataset.interval,
        "expected_bar_count": expected_count,
        "received_bar_count": len(timestamps),
        "duplicate_count": duplicate_count,
        "off_grid_count": len(off_grid),
        "off_grid_timestamps": off_grid[:50],
        "gap_count": len(missing_ranges),
        "missing_ranges": missing_ranges[:50],
        "contiguous": not off_grid and duplicate_count == 0 and not missing_ranges,
    }
    if not manifest["contiguous"]:
        raise RuleStrategyValidationCoverageError(
            "dataset_coverage_incomplete",
            "injected dataset is incomplete or gappy for the required validation window",
            {
                "symbol": dataset.symbol,
                "source_provider": dataset.source_provider,
                **manifest,
            },
        )
    return manifest


def calculate_window_metrics(
    points: Sequence[RuleStrategyValidationPoint],
    fills: Sequence[RuleStrategyValidationFill],
) -> dict[str, Any]:
    """Return transparent, zero-risk-free metrics from one saved equity curve."""

    ordered_points = sorted(points, key=lambda row: (row.observed_at, row.sequence))
    ordered_fills = sorted(fills, key=lambda row: row.sequence)
    if not ordered_points:
        return {
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": None,
            "win_rate_pct": None,
            "profit_factor": None,
            "exposure_pct": 0.0,
            "turnover_quote": 0.0,
            "turnover_pct": 0.0,
            "fees_quote": 0.0,
            "slippage_quote": 0.0,
            "fill_count": 0,
        }
    start_equity = ordered_points[0].equity_quote
    end_equity = ordered_points[-1].equity_quote
    total_return = (
        (end_equity / start_equity - 1) * 100 if start_equity > _EPSILON else 0.0
    )
    elapsed_seconds = max(
        0.0,
        (ordered_points[-1].observed_at - ordered_points[0].observed_at).total_seconds(),
    )
    elapsed_years = elapsed_seconds / (365.25 * 24 * 60 * 60)
    annualized = (
        ((end_equity / start_equity) ** (1 / elapsed_years) - 1) * 100
        if start_equity > _EPSILON and end_equity > 0 and elapsed_years > 0
        else -100.0 if start_equity > _EPSILON and end_equity <= 0 else 0.0
    )
    returns = [
        current.equity_quote / previous.equity_quote - 1
        for previous, current in zip(ordered_points, ordered_points[1:])
        if previous.equity_quote > _EPSILON
    ]
    periods_per_year = (
        len(returns) / elapsed_years if elapsed_years > 0 else 0.0
    )
    if len(returns) >= 2:
        deviation = statistics.stdev(returns)
        sharpe = (
            statistics.fmean(returns) / deviation * math.sqrt(periods_per_year)
            if deviation > _EPSILON and periods_per_year > 0
            else None
        )
    else:
        sharpe = None
    closing_fills = [item for item in ordered_fills if item.side == "sell"]
    wins = [item for item in closing_fills if item.realized_pnl_quote > 0]
    gross_profit = math.fsum(
        item.realized_pnl_quote
        for item in closing_fills
        if item.realized_pnl_quote > 0
    )
    gross_loss = math.fsum(
        -item.realized_pnl_quote
        for item in closing_fills
        if item.realized_pnl_quote < 0
    )
    exposure_points = ordered_points[1:] or ordered_points
    turnover_quote = math.fsum(item.quote_amount for item in ordered_fills)
    fees_quote = math.fsum(item.fee_quote for item in ordered_fills)
    slippage_quote = math.fsum(
        item.quote_amount * item.slippage_pct / 100 for item in ordered_fills
    )
    return {
        "total_return_pct": _finite(total_return),
        "annualized_return_pct": _finite(annualized),
        "max_drawdown_pct": _finite(
            max((item.drawdown_pct for item in ordered_points), default=0.0)
        ),
        "sharpe": _finite_or_none(sharpe),
        "win_rate_pct": _finite_or_none(
            len(wins) / len(closing_fills) * 100 if closing_fills else None
        ),
        "profit_factor": _finite_or_none(
            gross_profit / gross_loss if gross_loss > _EPSILON else None
        ),
        "exposure_pct": _finite(
            sum(item.position_quote > _EPSILON for item in exposure_points)
            / len(exposure_points)
            * 100
        ),
        "turnover_quote": _finite(turnover_quote),
        "turnover_pct": _finite(
            turnover_quote / start_equity * 100 if start_equity > _EPSILON else 0.0
        ),
        "fees_quote": _finite(fees_quote),
        "slippage_quote": _finite(slippage_quote),
        "fill_count": len(ordered_fills),
    }


def fingerprint(value: Any) -> str:
    """SHA-256 over canonical JSON, shared by data/config/assumption artifacts."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return _utc_iso(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported canonical JSON value: {type(value)!r}")


def _dataset_series(
    rows: Iterable[RuleStrategyValidationDataset],
) -> dict[tuple[str, str], tuple[RuleStrategyValidationCandle, ...]]:
    series: dict[tuple[str, str], tuple[RuleStrategyValidationCandle, ...]] = {}
    for row in rows:
        key = (row.symbol, row.interval)
        if key in series:
            raise RuleStrategyValidationCoverageError(
                "dataset_duplicate_persisted",
                "validation run has duplicate persisted datasets",
                {"symbol": row.symbol, "interval": row.interval},
            )
        series[key] = tuple(
            RuleStrategyValidationCandle.model_validate(item)
            for item in (row.bars or [])
        )
    return series


def _pending_fill_from_result(
    *,
    symbol: str,
    window: Literal["in_sample", "out_of_sample"],
    decision_at_ms: int,
    decision_price: float,
    next_bar_open: float,
    result: Mapping[str, Any],
    position_open: bool,
) -> _PendingFill | None:
    action = str(result.get("action", "no_op"))
    sizing = result.get("sizing")
    sizing_mapping = sizing if isinstance(sizing, Mapping) else {}
    requested_quote = _positive_float(sizing_mapping.get("requested_quote"))
    requested_quantity = _positive_float(sizing_mapping.get("quantity"))
    reason_code = str(result.get("reason_code", "validation_action"))[:128]
    if action in {"buy", "entry", "add"}:
        leg_kind: Literal["entry", "add", "reduce", "close"]
        if action == "entry":
            leg_kind = "entry"
        elif action == "add":
            leg_kind = "add"
        else:
            leg_kind = "add" if position_open else "entry"
        if requested_quote <= _EPSILON:
            return None
        return _PendingFill(
            symbol=symbol,
            window=window,
            leg_kind=leg_kind,
            side="buy",
            decision_at_ms=decision_at_ms,
            decision_price=decision_price,
            requested_quote=requested_quote,
            requested_quantity=None,
            reason_code=reason_code,
            fill_open=next_bar_open,
        )
    if action in {"sell", "reduce", "close"}:
        leg_kind = "reduce" if action == "reduce" else "close"
        if not position_open:
            return None
        return _PendingFill(
            symbol=symbol,
            window=window,
            leg_kind=leg_kind,
            side="sell",
            decision_at_ms=decision_at_ms,
            decision_price=decision_price,
            requested_quote=requested_quote,
            requested_quantity=requested_quantity if requested_quantity > _EPSILON else None,
            reason_code=reason_code,
            fill_open=next_bar_open,
        )
    return None


def _apply_pending_fill(
    account: _ReplayAccount,
    pending: _PendingFill,
    *,
    fee_rate: float,
    slippage_rate: float,
) -> _AppliedFill | None:
    before = account.snapshot()
    existing = account.positions.get(pending.symbol)
    if pending.side == "buy":
        if pending.leg_kind == "entry" and existing is not None:
            return None
        if pending.leg_kind == "add" and existing is None:
            return None
        fill_price = pending.fill_open * (1 + slippage_rate)
        requested = min(pending.requested_quote, account.quote_balance / (1 + fee_rate))
        if requested <= _EPSILON or fill_price <= _EPSILON:
            return None
        quantity = requested / fill_price
        fee = requested * fee_rate
        if existing is None:
            account.positions[pending.symbol] = _ReplayPosition(
                quantity=quantity,
                entry_price=(requested + fee) / quantity,
                mark_price=fill_price,
                highest_price=fill_price,
                addition_count=0,
            )
        else:
            combined_quantity = existing.quantity + quantity
            existing.entry_price = (
                existing.quantity * existing.entry_price + requested + fee
            ) / combined_quantity
            existing.quantity = combined_quantity
            existing.mark_price = fill_price
            existing.highest_price = max(existing.highest_price, fill_price)
            existing.addition_count += 1
        account.quote_balance = max(0.0, account.quote_balance - requested - fee)
        return _AppliedFill(
            decision_price=pending.decision_price,
            fill_price=fill_price,
            quantity=quantity,
            quote_amount=requested,
            fee_quote=fee,
            realized_pnl_quote=0.0,
            account_before=before,
            account_after=account.snapshot(),
        )
    if existing is None:
        return None
    fill_price = pending.fill_open * (1 - slippage_rate)
    if fill_price <= _EPSILON:
        return None
    if pending.leg_kind == "close":
        quantity = existing.quantity
    elif pending.requested_quantity is not None:
        quantity = min(existing.quantity, pending.requested_quantity)
    elif pending.requested_quote > _EPSILON:
        quantity = min(existing.quantity, pending.requested_quote / fill_price)
    else:
        return None
    if quantity <= _EPSILON:
        return None
    quote_amount = quantity * fill_price
    fee = quote_amount * fee_rate
    realized = quote_amount - fee - quantity * existing.entry_price
    account.quote_balance += quote_amount - fee
    account.realized_pnl_quote += realized
    remaining = existing.quantity - quantity
    if remaining <= _EPSILON:
        account.positions.pop(pending.symbol, None)
    else:
        existing.quantity = remaining
        existing.mark_price = fill_price
        existing.highest_price = max(existing.highest_price, fill_price)
    return _AppliedFill(
        decision_price=pending.decision_price,
        fill_price=fill_price,
        quantity=quantity,
        quote_amount=quote_amount,
        fee_quote=fee,
        realized_pnl_quote=realized,
        account_before=before,
        account_after=account.snapshot(),
    )


def _decision_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": result.get("action"),
        "reason_code": result.get("reason_code"),
        "reason": result.get("reason"),
        "sizing": dict(result.get("sizing") or {}),
        "indicators": dict(result.get("indicators") or {}),
        "conditions": list(result.get("conditions") or []),
    }


def _window_for_bar_open(
    run: RuleStrategyValidationRun, timestamp_ms: int
) -> Literal["in_sample", "out_of_sample"]:
    timestamp = _datetime_from_ms(timestamp_ms)
    if _utc(run.in_sample_start_at) <= timestamp < _utc(run.out_of_sample_start_at):
        return "in_sample"
    if _utc(run.out_of_sample_start_at) <= timestamp < _utc(run.out_of_sample_end_at_exclusive):
        return "out_of_sample"
    raise RuleStrategyValidationCoverageError(
        "bar_outside_validation_window",
        "persisted bar is outside the validation window",
        {"timestamp": _utc_iso(timestamp)},
    )


def _window_from_run(run: RuleStrategyValidationRun) -> RuleStrategyValidationWindow:
    return RuleStrategyValidationWindow(
        in_sample_start_at=_utc(run.in_sample_start_at),
        in_sample_end_at_exclusive=_utc(run.in_sample_end_at_exclusive),
        out_of_sample_start_at=_utc(run.out_of_sample_start_at),
        out_of_sample_end_at_exclusive=_utc(run.out_of_sample_end_at_exclusive),
    )


def _missing_ranges(
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    present: set[int],
) -> list[dict[str, str]]:
    ranges: list[dict[str, str]] = []
    missing_start: int | None = None
    timestamp = start_ms
    while timestamp < end_ms:
        if timestamp not in present:
            if missing_start is None:
                missing_start = timestamp
        elif missing_start is not None:
            ranges.append(
                {
                    "start_at": _utc_iso(_datetime_from_ms(missing_start)),
                    "end_at_exclusive": _utc_iso(_datetime_from_ms(timestamp)),
                }
            )
            missing_start = None
        timestamp += interval_ms
    if missing_start is not None:
        ranges.append(
            {
                "start_at": _utc_iso(_datetime_from_ms(missing_start)),
                "end_at_exclusive": _utc_iso(_datetime_from_ms(end_ms)),
            }
        )
    return ranges


def _shift_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + months
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    days = _days_in_month(year, month)
    return value.replace(year=year, month=month, day=min(value.day, days))


def _days_in_month(year: int, month: int) -> int:
    from calendar import monthrange

    return monthrange(year, month)[1]


def _lease_duration():
    from datetime import timedelta

    return timedelta(seconds=_DEFAULT_LEASE_SECONDS)


def _datetime_from_ms(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1_000, UTC)


def _datetime_to_ms(value: datetime) -> int:
    return int(_utc(value).timestamp() * 1_000)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _utc_or_none(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None


def _utc_iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _rate(value: Any, name: str) -> float:
    number = _positive_or_zero(value)
    if number >= 1:
        raise RuleStrategyValidationError(f"immutable {name} must be below 1")
    return number


def _positive_or_zero(value: Any) -> float:
    if isinstance(value, bool):
        raise RuleStrategyValidationError("boolean is not a numeric assumption")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuleStrategyValidationError("immutable assumption is not numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise RuleStrategyValidationError("immutable assumption must be finite and non-negative")
    return number


def _positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise RuleStrategyValidationError("replay produced a non-finite value")
    return float(value)


def _finite_or_none(value: float | None) -> float | None:
    return _finite(value) if value is not None else None


def _as_optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _error_code(error: Exception) -> str:
    if isinstance(error, RuleStrategyValidationCoverageError):
        return error.code
    if isinstance(error, RuleStrategyValidationWindowError):
        return "invalid_validation_window"
    if isinstance(error, RuleStrategyValidationLeaseLostError):
        return "validation_lease_lost"
    return "validation_replay_failed"


def _point_fingerprint_payload(point: RuleStrategyValidationPoint) -> dict[str, Any]:
    return {
        "sequence": point.sequence,
        "window": point.window,
        "observed_at": _utc_iso(point.observed_at),
        "equity_quote": point.equity_quote,
        "cash_quote": point.cash_quote,
        "position_quote": point.position_quote,
        "drawdown_pct": point.drawdown_pct,
        "account_snapshot": point.account_snapshot,
        "decisions": point.decisions,
    }


def _fill_fingerprint_payload(fill: RuleStrategyValidationFill) -> dict[str, Any]:
    return {
        "sequence": fill.sequence,
        "window": fill.window,
        "symbol": fill.symbol,
        "leg_kind": fill.leg_kind,
        "side": fill.side,
        "decision_at": _utc_iso(fill.decision_at),
        "filled_at": _utc_iso(fill.filled_at),
        "decision_price": fill.decision_price,
        "fill_price": fill.fill_price,
        "quantity": fill.quantity,
        "quote_amount": fill.quote_amount,
        "fee_quote": fill.fee_quote,
        "slippage_pct": fill.slippage_pct,
        "realized_pnl_quote": fill.realized_pnl_quote,
        "reason_code": fill.reason_code,
        "account_before": fill.account_before,
        "account_after": fill.account_after,
    }
