"""Focused pure-contract checks for deterministic validation primitives."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from valuecell.server.api.schemas.rule_strategy_validation import (
    RuleStrategyValidationCandle,
    RuleStrategyValidationDatasetInput,
    RuleStrategyValidationPointView,
    RuleStrategyValidationRunDetail,
)
from valuecell.server.services.rule_strategy_validation_export_service import (
    RuleStrategyValidationExportService,
)
from valuecell.server.services.rule_strategy_validation_service import (
    RuleStrategyValidationCoverageError,
    RuleStrategyValidationWindowError,
    calculate_window_metrics,
    derive_validation_window,
    fingerprint,
    validate_dataset_coverage,
)


def _complete_daily_dataset(window, *, missing_index: int | None = None):
    start = window.in_sample_start_at
    end = window.out_of_sample_end_at_exclusive
    days = int((end - start).total_seconds() // 86_400)
    bars = []
    for index in range(days):
        if index == missing_index:
            continue
        price = 100.0 + index / 10
        bars.append(
            RuleStrategyValidationCandle(
                timestamp_ms=int((start + timedelta(days=index)).timestamp() * 1_000),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=10_000_000,
            )
        )
    return RuleStrategyValidationDatasetInput(
        source_provider="synthetic",
        symbol="BTC-USDT",
        interval="1d",
        bars=bars,
    )


def test_validation_window_is_contiguous_and_requires_closed_utc_day():
    window = derive_validation_window(
        date(2026, 6, 30),
        now=datetime(2026, 8, 5, 12, tzinfo=UTC),
    )

    assert window.in_sample_start_at == datetime(2024, 4, 1, tzinfo=UTC)
    assert window.in_sample_end_at_exclusive == datetime(2026, 4, 1, tzinfo=UTC)
    assert window.out_of_sample_start_at == window.in_sample_end_at_exclusive
    assert window.out_of_sample_end_at_exclusive == datetime(2026, 7, 1, tzinfo=UTC)
    with pytest.raises(RuleStrategyValidationWindowError):
        derive_validation_window(date(2026, 8, 5), now=datetime(2026, 8, 5, tzinfo=UTC))


def test_validation_coverage_rejects_one_gapped_bar_with_report():
    window = derive_validation_window(
        date(2026, 6, 30),
        now=datetime(2026, 8, 5, tzinfo=UTC),
    )
    dataset = _complete_daily_dataset(window, missing_index=17)

    with pytest.raises(RuleStrategyValidationCoverageError) as captured:
        validate_dataset_coverage(dataset, window)

    assert captured.value.code == "dataset_coverage_incomplete"
    assert captured.value.report["gap_count"] == 1
    assert captured.value.report["contiguous"] is False


def test_validation_fingerprint_is_canonical_and_metrics_are_window_local():
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})

    point_a = SimpleNamespace(
        sequence=0,
        observed_at=datetime(2024, 4, 1, tzinfo=UTC),
        equity_quote=100.0,
        cash_quote=100.0,
        position_quote=0.0,
        drawdown_pct=0.0,
    )
    point_b = SimpleNamespace(
        sequence=1,
        observed_at=datetime(2024, 4, 2, tzinfo=UTC),
        equity_quote=110.0,
        cash_quote=0.0,
        position_quote=110.0,
        drawdown_pct=0.0,
    )
    metrics = calculate_window_metrics([point_a, point_b], [])
    assert metrics["total_return_pct"] == pytest.approx(10.0)
    assert metrics["exposure_pct"] == pytest.approx(100.0)
    assert metrics["fill_count"] == 0


def test_validation_export_has_distinct_sheets_and_redacts_sensitive_snapshots():
    window = derive_validation_window(
        date(2026, 6, 30),
        now=datetime(2026, 8, 5, tzinfo=UTC),
    )
    run = RuleStrategyValidationRunDetail(
        run_id="run-a",
        strategy_id="strategy-a",
        status="completed",
        source_preference="injected",
        selected_symbols=["BTC-USDT"],
        window=window,
        initial_capital_quote=1_000,
        data_fingerprint="d" * 64,
        config_fingerprint="c" * 64,
        assumptions_fingerprint="a" * 64,
        artifact_fingerprint="f" * 64,
        metrics={"in_sample": {}, "out_of_sample": {}},
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        completed_at=datetime(2026, 8, 1, tzinfo=UTC),
        config_snapshot={"api_secret": "must-not-export"},
        assumptions={"access_token": "must-not-export"},
        engine_version="rule_engine_v1",
    )
    point = RuleStrategyValidationPointView(
        sequence=0,
        window="in_sample",
        observed_at=window.in_sample_start_at,
        equity_quote=1_000,
        cash_quote=1_000,
        position_quote=0,
        drawdown_pct=0,
        account_snapshot={},
        decisions={},
    )
    fake_service = SimpleNamespace(
        get=lambda _run_id, _tenant_id: run,
        datasets=lambda _run_id, _tenant_id: [],
        points=lambda _run_id, _tenant_id: [point],
        fills=lambda _run_id, _tenant_id: [],
    )

    workbook, filename = RuleStrategyValidationExportService(fake_service).build(
        "run-a", "tenant-a"
    )

    assert filename == "strategy-validation-run-a.xlsx"
    assert ZipFile(BytesIO(workbook)).testzip() is None
    assert b"must-not-export" not in workbook
    sheet_names_xml = ZipFile(BytesIO(workbook)).read("xl/workbook.xml")
    assert "样本内权益曲线".encode("utf-8") in sheet_names_xml
    assert "样本外权益曲线".encode("utf-8") in sheet_names_xml
