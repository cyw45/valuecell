"""Build redacted XLSX workbooks for immutable strategy validation results."""

from __future__ import annotations

import re
from typing import Any, Mapping

from valuecell.server.services.rule_strategy_export_service import (
    _build_xlsx,
    _flatten,
    _format_timestamp,
    _json_cell,
    _redact,
)
from valuecell.server.services.rule_strategy_validation_service import (
    RuleStrategyValidationNotCompletedError,
    RuleStrategyValidationService,
)


class RuleStrategyValidationExportService:
    """Render a completed tenant-scoped validation into distinct audit sheets."""

    def __init__(self, validation_service: RuleStrategyValidationService) -> None:
        self._validation_service = validation_service

    def build(self, run_id: str, tenant_id: str) -> tuple[bytes, str]:
        """Return a redacted, immutable-result workbook and a safe filename."""

        run = self._validation_service.get(run_id, tenant_id)
        if run.status != "completed":
            raise RuleStrategyValidationNotCompletedError(
                "validation export is available only after the run completes"
            )
        datasets = self._validation_service.datasets(run_id, tenant_id)
        points = self._validation_service.points(run_id, tenant_id)
        fills = self._validation_service.fills(run_id, tenant_id)
        sheets = [
            ("验证说明", self._description_rows(run)),
            ("数据清单", self._dataset_rows(datasets)),
            ("配置与假设", self._configuration_rows(run)),
            ("样本内指标", self._metric_rows(run.metrics, "in_sample")),
            ("样本外指标", self._metric_rows(run.metrics, "out_of_sample")),
            ("样本内权益曲线", self._equity_rows(points, "in_sample")),
            ("样本外权益曲线", self._equity_rows(points, "out_of_sample")),
            ("成交账本", self._fill_rows(fills)),
        ]
        return _build_xlsx(sheets), _attachment_filename(run_id)

    @staticmethod
    def _description_rows(run) -> list[list[Any]]:
        return [
            ["项目", "内容"],
            ["验证运行 ID", run.run_id],
            ["策略 ID", run.strategy_id],
            ["状态", run.status],
            ["数据来源偏好", run.source_preference],
            ["选定标的", ", ".join(run.selected_symbols)],
            ["样本内开始 (UTC)", _format_timestamp(run.window.in_sample_start_at)],
            [
                "样本内结束 (UTC，不含)",
                _format_timestamp(run.window.in_sample_end_at_exclusive),
            ],
            ["样本外开始 (UTC)", _format_timestamp(run.window.out_of_sample_start_at)],
            [
                "样本外结束 (UTC，不含)",
                _format_timestamp(run.window.out_of_sample_end_at_exclusive),
            ],
            ["初始资金 (计价币)", run.initial_capital_quote],
            ["数据指纹 (SHA-256)", run.data_fingerprint],
            ["配置指纹 (SHA-256)", run.config_fingerprint],
            ["假设指纹 (SHA-256)", run.assumptions_fingerprint],
            ["产物指纹 (SHA-256)", run.artifact_fingerprint],
            ["引擎版本", run.engine_version],
            ["指标公式版本", run.indicator_formula_version],
            ["模板 ID", run.template_id],
            ["模板版本", run.template_version],
            ["创建时间 (UTC)", _format_timestamp(run.created_at)],
            ["完成时间 (UTC)", _format_timestamp(run.completed_at)],
            [
                "重放说明",
                "仅使用本工作簿数据清单对应的已物化 K 线；每次决策在收盘后执行，成交使用下一根 K 线开盘价并应用保存的手续费与滑点假设。",
            ],
        ]

    @staticmethod
    def _dataset_rows(datasets) -> list[list[Any]]:
        rows: list[list[Any]] = [
            [
                "数据集 ID",
                "数据提供方",
                "标的",
                "周期",
                "开始 (UTC)",
                "结束 (UTC，不含)",
                "K 线数量",
                "内容哈希 (SHA-256)",
                "获取时间 (UTC)",
                "覆盖清单",
                "分页清单",
            ]
        ]
        for dataset in datasets:
            rows.append(
                [
                    dataset.dataset_id,
                    dataset.source_provider,
                    dataset.symbol,
                    dataset.interval,
                    _format_timestamp(dataset.start_at),
                    _format_timestamp(dataset.end_at_exclusive),
                    dataset.bar_count,
                    dataset.content_hash,
                    _format_timestamp(dataset.retrieved_at),
                    _json_cell(dataset.coverage_manifest),
                    _json_cell(dataset.page_manifest),
                ]
            )
        return rows

    @staticmethod
    def _configuration_rows(run) -> list[list[Any]]:
        rows: list[list[Any]] = [["分类", "路径", "值"]]
        for path, value in _flatten("config", _redact(run.config_snapshot)):
            rows.append(["策略配置快照", path, _json_cell(value) if isinstance(value, (dict, list)) else value])
        for path, value in _flatten("assumptions", _redact(run.assumptions)):
            rows.append(["不可变假设", path, _json_cell(value) if isinstance(value, (dict, list)) else value])
        return rows

    @staticmethod
    def _metric_rows(
        metrics: Mapping[str, Any] | None,
        window: str,
    ) -> list[list[Any]]:
        values = metrics.get(window, {}) if isinstance(metrics, Mapping) else {}
        rows: list[list[Any]] = [["指标", "值"]]
        for path, value in _flatten("metrics", _redact(values)):
            rows.append([path, _json_cell(value) if isinstance(value, (dict, list)) else value])
        return rows

    @staticmethod
    def _equity_rows(points, window: str) -> list[list[Any]]:
        rows: list[list[Any]] = [
            [
                "序号",
                "观测时间 (UTC)",
                "权益",
                "现金",
                "持仓市值",
                "回撤 (%)",
                "账户快照",
                "决策证据",
            ]
        ]
        for point in points:
            if point.window != window:
                continue
            rows.append(
                [
                    point.sequence,
                    _format_timestamp(point.observed_at),
                    point.equity_quote,
                    point.cash_quote,
                    point.position_quote,
                    point.drawdown_pct,
                    _json_cell(point.account_snapshot),
                    _json_cell(point.decisions),
                ]
            )
        return rows

    @staticmethod
    def _fill_rows(fills) -> list[list[Any]]:
        rows: list[list[Any]] = [
            [
                "序号",
                "窗口",
                "标的",
                "腿类型",
                "方向",
                "决策时间 (UTC)",
                "成交时间 (UTC)",
                "决策价格",
                "成交价格",
                "数量",
                "成交额",
                "手续费",
                "滑点 (%)",
                "已实现盈亏",
                "原因代码",
                "成交前账户",
                "成交后账户",
            ]
        ]
        for fill in fills:
            rows.append(
                [
                    fill.sequence,
                    fill.window,
                    fill.symbol,
                    fill.leg_kind,
                    fill.side,
                    _format_timestamp(fill.decision_at),
                    _format_timestamp(fill.filled_at),
                    fill.decision_price,
                    fill.fill_price,
                    fill.quantity,
                    fill.quote_amount,
                    fill.fee_quote,
                    fill.slippage_pct,
                    fill.realized_pnl_quote,
                    fill.reason_code,
                    _json_cell(fill.account_before),
                    _json_cell(fill.account_after),
                ]
            )
        return rows


def _attachment_filename(run_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    return f"strategy-validation-{(safe_id or 'run')[:80]}.xlsx"
