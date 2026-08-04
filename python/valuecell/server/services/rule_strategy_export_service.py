"""Build tenant-scoped XLSX exports for persisted rule-strategy history."""
from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Mapping
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from valuecell.server.services.rule_strategy_service import RuleStrategyService

XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

_REDACTED_VALUE = "[已脱敏]"
_SENSITIVE_KEY_PARTS = (
    "apikey",
    "apisecret",
    "secret",
    "passphrase",
    "authorization",
    "credentials",
    "token",
    "password",
    "privatekey",
    "seedphrase",
    "signature",
    "walletaddress",
)
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|api[_-]?secret|secret|passphrase|authorization|"
    r"(?:access|refresh)[_-]?token|token|password|private[_-]?key|"
    r"seed[_-]?phrase)\s*([:=])\s*([^\s,;]+)"
)


class RuleStrategyExportService:
    """Render persisted strategy facts into a dependency-free XLSX workbook."""

    def __init__(self, strategy_service: RuleStrategyService) -> None:
        self._strategy_service = strategy_service
        self._repository = strategy_service.repository

    def build(
        self,
        strategy_id: str,
        tenant_id: str,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> tuple[bytes, str]:
        """Return a sanitized workbook and safe attachment filename.

        Strategy lookup deliberately precedes all history reads so tenant-scoped
        existence remains the same as every other rule-strategy endpoint.
        """
        if from_date is not None and to_date is not None and from_date > to_date:
            raise ValueError("from_date must be on or before to_date")

        strategy = self._strategy_service.get(strategy_id, tenant_id)
        start_at, end_at_exclusive = _utc_bounds(from_date, to_date)
        journals = self._journals(
            strategy_id,
            tenant_id,
            start_at,
            end_at_exclusive,
        )
        evaluation_ids = {
            str(_field(journal, "evaluation_id"))
            for journal in journals
            if _field(journal, "evaluation_id") is not None
        }
        intents, orders = self._execution_records(
            strategy_id,
            tenant_id,
            evaluation_ids,
        )

        sheets = [
            ("导出说明", self._description_rows(strategy, from_date, to_date)),
            ("策略参数", self._parameter_rows(strategy)),
            ("成交明细", self._trade_rows(journals)),
            ("资金变化", self._fund_rows(journals)),
            ("执行明细", self._execution_rows(journals, intents, orders)),
            ("资金费", self._funding_rows(journals)),
        ]
        return _build_xlsx(sheets), _attachment_filename(strategy_id)

    def _journals(
        self,
        strategy_id: str,
        tenant_id: str,
        start_at: datetime | None,
        end_at_exclusive: datetime | None,
    ) -> list[Any]:
        reader = getattr(self._repository, "get_evaluations_for_export", None)
        if callable(reader):
            raw_journals = reader(
                strategy_id,
                tenant_id,
                start_at,
                end_at_exclusive,
            )
        else:
            # Non-database repositories are used only in focused API tests. The
            # production repository implements the unbounded export reader.
            raw_journals = self._repository.get_evaluations(
                strategy_id,
                tenant_id,
                limit=100_000,
            )

        journals: list[Any] = []
        for journal in raw_journals or []:
            if (
                _field(journal, "tenant_id") != tenant_id
                or _field(journal, "strategy_id") != strategy_id
            ):
                continue
            recorded_at = _as_utc(_field(journal, "created_at"))
            if start_at is not None and (recorded_at is None or recorded_at < start_at):
                continue
            if end_at_exclusive is not None and (
                recorded_at is None or recorded_at >= end_at_exclusive
            ):
                continue
            journals.append(journal)
        return sorted(
            journals,
            key=lambda item: (
                _as_utc(_field(item, "created_at"))
                or datetime.min.replace(tzinfo=timezone.utc),
                str(_field(item, "evaluation_id") or ""),
            ),
        )

    def _execution_records(
        self,
        strategy_id: str,
        tenant_id: str,
        evaluation_ids: set[str],
    ) -> tuple[list[Any], list[Any]]:
        reader = getattr(self._repository, "get_execution_records_for_export", None)
        if not evaluation_ids or not callable(reader):
            return [], []
        records = reader(strategy_id, tenant_id, sorted(evaluation_ids))
        if not isinstance(records, tuple) or len(records) != 2:
            return [], []
        intents, orders = records
        safe_intents = [
            item
            for item in intents or []
            if _field(item, "tenant_id") == tenant_id
            and _field(item, "strategy_id") == strategy_id
            and str(_field(item, "evaluation_id")) in evaluation_ids
        ]
        safe_orders = [
            item
            for item in orders or []
            if _field(item, "tenant_id") == tenant_id
            and _field(item, "strategy_id") == strategy_id
            and str(_field(item, "evaluation_id")) in evaluation_ids
        ]
        return (
            sorted(safe_intents, key=_record_sort_key),
            sorted(safe_orders, key=_record_sort_key),
        )

    @staticmethod
    def _description_rows(
        strategy: Mapping[str, Any],
        from_date: date | None,
        to_date: date | None,
    ) -> list[list[Any]]:
        return [
            ["项目", "内容"],
            ["策略 ID", strategy.get("strategy_id")],
            ["策略名称", strategy.get("name")],
            ["导出日期范围 (UTC)", _range_label(from_date, to_date)],
            ["日期解释", "筛选按 UTC 日历日执行，起始日和结束日均包含在内。"],
            ["数据来源", "策略参数来自服务端配置；其他工作表来自持久化评估日志和执行记录。"],
            ["敏感信息", "不导出 API 密钥、密钥、口令、令牌或凭据机密。"],
        ]

    @staticmethod
    def _parameter_rows(strategy: Mapping[str, Any]) -> list[list[Any]]:
        rows: list[list[Any]] = [["参数路径", "参数值"]]
        strategy_fields = (
            ("strategy_id", strategy.get("strategy_id")),
            ("name", strategy.get("name")),
            ("description", strategy.get("description")),
            ("status", strategy.get("status")),
            ("mode", strategy.get("mode")),
            ("execution_generation", strategy.get("execution_generation")),
            ("created_at", strategy.get("created_at")),
            ("updated_at", strategy.get("updated_at")),
        )
        for path, value in strategy_fields:
            rows.append([path, _spreadsheet_value(_redact(value, path))])
        config = _redact(strategy.get("config") or {}, "config")
        for path, value in _flatten("config", config):
            rows.append([path, _spreadsheet_value(value)])
        return rows

    @staticmethod
    def _trade_rows(journals: list[Any]) -> list[list[Any]]:
        rows: list[list[Any]] = [
            [
                "评估时间 (UTC)",
                "评估 ID",
                "序号",
                "动作",
                "标的",
                "执行状态",
                "价格",
                "数量",
                "金额 (计价币)",
                "已实现盈亏 (计价币)",
                "原因代码",
                "原因",
                "仓位配置 (JSON)",
                "成交记录 (JSON)",
            ]
        ]
        for journal in journals:
            result = _mapping(_field(journal, "result"))
            for index, trade in enumerate(_list(_field(journal, "trades")), start=1):
                record = _mapping(trade)
                rows.append(
                    [
                        _format_timestamp(_field(journal, "created_at")),
                        _field(journal, "evaluation_id"),
                        index,
                        _first(record, result, "action"),
                        _first(record, result, "symbol"),
                        record.get("execution"),
                        record.get("price"),
                        record.get("quantity"),
                        record.get("quote_amount"),
                        record.get("realized_pnl_quote"),
                        _first(record, result, "reason_code"),
                        _first(record, result, "reason"),
                        _json_cell(record.get("sizing")),
                        _json_cell(trade),
                    ]
                )
        return rows

    @staticmethod
    def _fund_rows(journals: list[Any]) -> list[list[Any]]:
        rows: list[list[Any]] = [
            [
                "评估时间 (UTC)",
                "评估 ID",
                "动作",
                "标的",
                "账户来源",
                "初始资金 (计价币)",
                "可用资金 (计价币)",
                "持仓 (JSON)",
                "已实现盈亏 (计价币)",
                "未实现盈亏 (计价币)",
                "权益 (计价币)",
                "账户快照 (JSON)",
            ]
        ]
        for journal in journals:
            result = _mapping(_field(journal, "result"))
            account = _mapping(result.get("account"))
            if not account:
                continue
            rows.append(
                [
                    _format_timestamp(_field(journal, "created_at")),
                    _field(journal, "evaluation_id"),
                    result.get("action"),
                    result.get("symbol"),
                    account.get("source"),
                    account.get("initial_capital_quote"),
                    account.get("quote_balance"),
                    _json_cell(account.get("positions")),
                    account.get("realized_pnl_quote"),
                    account.get("unrealized_pnl_quote"),
                    account.get("equity_quote"),
                    _json_cell(account),
                ]
            )
        return rows

    @staticmethod
    def _execution_rows(
        journals: list[Any],
        intents: list[Any],
        orders: list[Any],
    ) -> list[list[Any]]:
        rows: list[list[Any]] = [
            [
                "来源",
                "评估时间 (UTC)",
                "创建时间 (UTC)",
                "更新时间 (UTC)",
                "评估 ID",
                "执行意图 ID",
                "订单 ID",
                "状态",
                "标的",
                "方向",
                "订单类型",
                "请求金额 (计价币)",
                "请求数量",
                "交易所订单 ID",
                "错误代码",
                "错误信息",
                "执行代次",
                "执行来源",
                "凭据 ID",
                "记录详情 (JSON)",
            ]
        ]
        journal_times = {
            str(_field(journal, "evaluation_id")): _format_timestamp(
                _field(journal, "created_at")
            )
            for journal in journals
            if _field(journal, "evaluation_id") is not None
        }
        for journal in journals:
            result = _mapping(_field(journal, "result"))
            execution = result.get("execution")
            if execution is None:
                continue
            detail = _mapping(execution)
            rows.append(
                [
                    "评估执行记录",
                    _format_timestamp(_field(journal, "created_at")),
                    "",
                    "",
                    _field(journal, "evaluation_id"),
                    detail.get("execution_intent_id"),
                    detail.get("id"),
                    detail.get("status") or detail.get("execution"),
                    result.get("symbol"),
                    "",
                    "",
                    "",
                    "",
                    detail.get("exchange_order_id"),
                    detail.get("error_code"),
                    detail.get("error_message") or detail.get("reason"),
                    detail.get("execution_generation"),
                    result.get("execution_ledger"),
                    detail.get("credential_id"),
                    _json_cell(execution),
                ]
            )
        for intent in intents:
            detail = {
                "execution_intent_id": _field(intent, "id"),
                "evaluation_id": _field(intent, "evaluation_id"),
                "execution_generation": _field(intent, "execution_generation"),
                "execution_source": _field(intent, "execution_source"),
                "credential_id": _field(intent, "credential_id"),
                "idempotency_key": _field(intent, "idempotency_key"),
                "symbol": _field(intent, "symbol"),
                "side": _field(intent, "side"),
                "order_type": _field(intent, "order_type"),
                "requested_quote": _field(intent, "requested_quote"),
                "requested_quantity": _field(intent, "requested_quantity"),
                "status": _field(intent, "status"),
                "attempt_count": _field(intent, "attempt_count"),
                "error_code": _field(intent, "error_code"),
                "error_message": _field(intent, "error_message"),
                "submitted_at": _field(intent, "submitted_at"),
                "terminal_at": _field(intent, "terminal_at"),
                "request_payload": _field(intent, "request_payload"),
                "created_at": _field(intent, "created_at"),
                "updated_at": _field(intent, "updated_at"),
            }
            rows.append(
                [
                    "执行意图",
                    journal_times.get(str(_field(intent, "evaluation_id")), ""),
                    _format_timestamp(_field(intent, "created_at")),
                    _format_timestamp(_field(intent, "updated_at")),
                    _field(intent, "evaluation_id"),
                    _field(intent, "id"),
                    "",
                    _field(intent, "status"),
                    _field(intent, "symbol"),
                    _field(intent, "side"),
                    _field(intent, "order_type"),
                    _field(intent, "requested_quote"),
                    _field(intent, "requested_quantity"),
                    "",
                    _field(intent, "error_code"),
                    _field(intent, "error_message"),
                    _field(intent, "execution_generation"),
                    _field(intent, "execution_source"),
                    _field(intent, "credential_id"),
                    _json_cell(detail),
                ]
            )
        for order in orders:
            detail = {
                "order_id": _field(order, "id"),
                "evaluation_id": _field(order, "evaluation_id"),
                "execution_intent_id": _field(order, "execution_intent_id"),
                "execution_generation": _field(order, "execution_generation"),
                "execution_source": _field(order, "execution_source"),
                "credential_id": _field(order, "credential_id"),
                "provider": _field(order, "provider"),
                "client_order_id": _field(order, "client_order_id"),
                "symbol": _field(order, "symbol"),
                "side": _field(order, "side"),
                "order_type": _field(order, "order_type"),
                "requested_quote": _field(order, "requested_quote"),
                "requested_quantity": _field(order, "requested_quantity"),
                "status": _field(order, "status"),
                "exchange_order_id": _field(order, "exchange_order_id"),
                "sandbox": _field(order, "sandbox"),
                "response_metadata": _field(order, "response_metadata"),
                "error_code": _field(order, "error_code"),
                "created_at": _field(order, "created_at"),
                "updated_at": _field(order, "updated_at"),
            }
            rows.append(
                [
                    "沙盒订单",
                    journal_times.get(str(_field(order, "evaluation_id")), ""),
                    _format_timestamp(_field(order, "created_at")),
                    _format_timestamp(_field(order, "updated_at")),
                    _field(order, "evaluation_id"),
                    _field(order, "execution_intent_id"),
                    _field(order, "id"),
                    _field(order, "status"),
                    _field(order, "symbol"),
                    _field(order, "side"),
                    _field(order, "order_type"),
                    _field(order, "requested_quote"),
                    _field(order, "requested_quantity"),
                    _field(order, "exchange_order_id"),
                    _field(order, "error_code"),
                    "",
                    _field(order, "execution_generation"),
                    _field(order, "execution_source"),
                    _field(order, "credential_id"),
                    _json_cell(detail),
                ]
            )
        return rows

    @staticmethod
    def _funding_rows(journals: list[Any]) -> list[list[Any]]:
        rows: list[list[Any]] = [
            [
                "评估时间 (UTC)",
                "评估 ID",
                "序号",
                "资金费率",
                "当前名义金额 (计价币)",
                "预计名义金额 (计价币)",
                "预计资金费 (计价币)",
                "方向",
                "资金费记录 (JSON)",
            ]
        ]
        for journal in journals:
            result = _mapping(_field(journal, "result"))
            funding_entries = _list(_field(journal, "funding"))
            if not funding_entries and result.get("funding") is not None:
                funding_entries = [result["funding"]]
            for index, funding in enumerate(funding_entries, start=1):
                record = _mapping(funding)
                rows.append(
                    [
                        _format_timestamp(_field(journal, "created_at")),
                        _field(journal, "evaluation_id"),
                        index,
                        record.get("funding_rate"),
                        record.get("current_notional_quote"),
                        record.get("projected_notional_quote"),
                        record.get("estimated_payment_quote"),
                        record.get("direction"),
                        _json_cell(funding),
                    ]
                )
        return rows


def _utc_bounds(
    from_date: date | None,
    to_date: date | None,
) -> tuple[datetime | None, datetime | None]:
    start_at = (
        datetime(
            from_date.year,
            from_date.month,
            from_date.day,
            tzinfo=timezone.utc,
        )
        if from_date is not None
        else None
    )
    end_at_exclusive = (
        datetime(
            to_date.year,
            to_date.month,
            to_date.day,
            tzinfo=timezone.utc,
        )
        + timedelta(days=1)
        if to_date is not None
        else None
    )
    return start_at, end_at_exclusive


def _range_label(from_date: date | None, to_date: date | None) -> str:
    if from_date is None and to_date is None:
        return "全部历史记录"
    if from_date is not None and to_date is not None:
        return f"{from_date.isoformat()} 至 {to_date.isoformat()}（含）"
    if from_date is not None:
        return f"自 {from_date.isoformat()} 起（UTC，含当日）"
    assert to_date is not None
    return f"截至 {to_date.isoformat()}（UTC，含当日）"


def _field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _first(record: Mapping[str, Any], result: Mapping[str, Any], name: str) -> Any:
    return record[name] if name in record else result.get(name)


def _record_sort_key(record: Any) -> tuple[datetime, str]:
    return (
        _as_utc(_field(record, "created_at"))
        or datetime.min.replace(tzinfo=timezone.utc),
        str(_field(record, "id") or ""),
    )


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_timestamp(value: Any) -> str:
    timestamp = _as_utc(value)
    return timestamp.isoformat().replace("+00:00", "Z") if timestamp else ""


def _normalised_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _is_sensitive_key(key: str) -> bool:
    normalised = _normalised_key(key)
    return any(part in normalised for part in _SENSITIVE_KEY_PARTS)


def _redact(value: Any, key: str | None = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return _REDACTED_VALUE
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return _SENSITIVE_VALUE_PATTERN.sub(
            lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED_VALUE}",
            value,
        )
    return value


def _flatten(path: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        if not value:
            return [(path, "{}")]
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            rows.extend(_flatten(f"{path}.{key}", item))
        return rows
    if isinstance(value, list):
        if not value:
            return [(path, "[]")]
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten(f"{path}[{index}]", item))
        return rows
    return [(path, value)]


def _json_cell(value: Any) -> str:
    return json.dumps(
        _redact(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _spreadsheet_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _json_cell(value)
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return _redact(value)
    return value


def _attachment_filename(strategy_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", strategy_id).strip("._-")
    return f"strategy-export-{(safe_id or 'strategy')[:80]}.xlsx"


def _build_xlsx(sheets: list[tuple[str, list[list[Any]]]]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml(len(sheets)))
        archive.writestr("_rels/.rels", _root_relationships_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml(sheets))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_relationships_xml(sheets))
        for index, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(rows))
    return buffer.getvalue()


def _content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]
    overrides.extend(
        '<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(
            index=index
        )
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{''.join(overrides)}"
        "</Types>"
    )


def _root_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _workbook_xml(sheets: list[tuple[str, list[list[Any]]]]) -> str:
    entries = "".join(
        '<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'.format(
            name=_xml_attribute(name),
            index=index,
        )
        for index, (name, _) in enumerate(sheets, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{entries}</sheets>"
        "</workbook>"
    )


def _workbook_relationships_xml(sheets: list[tuple[str, list[list[Any]]]]) -> str:
    entries = "".join(
        '<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet{index}.xml"/>'.format(index=index)
        for index in range(1, len(sheets) + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{entries}"
        "</Relationships>"
    )


def _worksheet_xml(rows: list[list[Any]]) -> str:
    max_columns = max((len(row) for row in rows), default=1)
    max_rows = max(len(rows), 1)
    dimension = f"A1:{_excel_column(max_columns)}{max_rows}"
    body = "".join(
        '<row r="{row_index}">{cells}</row>'.format(
            row_index=row_index,
            cells="".join(
                _xlsx_cell(
                    f"{_excel_column(column_index)}{row_index}",
                    value,
                )
                for column_index, value in enumerate(row, start=1)
            ),
        )
        for row_index, row in enumerate(rows, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        f"<sheetData>{body}</sheetData>"
        "</worksheet>"
    )


def _xlsx_cell(reference: str, value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        return f'<c r="{reference}"><v>{value}</v></c>'
    elif isinstance(value, float) and math.isfinite(value):
        return f'<c r="{reference}"><v>{value!r}</v></c>'
    else:
        text = _xml_text(_spreadsheet_value(value))
    return (
        f'<c r="{reference}" t="inlineStr">'
        f'<is><t xml:space="preserve">{escape(text)}</t></is>'
        "</c>"
    )


def _excel_column(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_attribute(value: str) -> str:
    return escape(_xml_text(value), {'"': "&quot;"})


def _xml_text(value: Any) -> str:
    text = str(value)
    return "".join(
        character
        for character in text
        if character in "\t\n\r"
        or (ord(character) >= 32 and not 0xD800 <= ord(character) <= 0xDFFF)
    )
