from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace
from xml.etree import ElementTree
from zipfile import ZipFile

from fastapi import FastAPI
from fastapi.testclient import TestClient

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.services.rule_strategy_service import RuleStrategyService

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


class ExportRepository:
    """Small tenant-aware persistence boundary for XLSX API tests."""

    def __init__(self) -> None:
        self.strategies = {}
        self.journals = []
        self.intents = []
        self.orders = []

    def get(self, strategy_id: str, tenant_id: str):
        return self.strategies.get((tenant_id, strategy_id))

    def get_evaluations(self, strategy_id: str, tenant_id: str, limit: int = 100):
        matching = [
            journal
            for journal in self.journals
            if journal.strategy_id == strategy_id and journal.tenant_id == tenant_id
        ]
        return list(reversed(matching[-limit:]))

    def get_evaluations_for_export(
        self,
        strategy_id: str,
        tenant_id: str,
        _start_at,
        _end_at_exclusive,
    ):
        # The service must still apply its UTC interval and tenant defense.
        return list(self.journals)

    def get_execution_records_for_export(
        self,
        _strategy_id: str,
        _tenant_id: str,
        _evaluation_ids: list[str],
    ):
        return list(self.intents), list(self.orders)


def _config() -> dict:
    return RuleStrategyConfig().model_dump(mode="json")


def _strategy(tenant_id: str, strategy_id: str, name: str = "导出策略"):
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SimpleNamespace(
        strategy_id=strategy_id,
        tenant_id=tenant_id,
        name=name,
        description="服务端持久化策略",
        status="stopped",
        execution_generation=2,
        config=_config(),
        created_at=created_at,
        updated_at=created_at,
    )


def _journal(
    evaluation_id: str,
    recorded_at: datetime,
    *,
    tenant_id: str = "tenant-a",
    strategy_id: str = "strategy-a",
):
    return SimpleNamespace(
        evaluation_id=evaluation_id,
        tenant_id=tenant_id,
        strategy_id=strategy_id,
        created_at=recorded_at,
        result={
            "action": "buy",
            "reason_code": "indicator_buy_confirmed",
            "reason": "服务器已确认买入条件。",
            "symbol": "BTC-USDT",
            "account": {
                "initial_capital_quote": 1_000.0,
                "quote_balance": 900.0,
                "positions": {"BTC-USDT": {"quantity": 1.25, "entry_price": 80}},
                "realized_pnl_quote": 12.0,
                "unrealized_pnl_quote": 3.0,
                "equity_quote": 1_015.0,
            },
            "execution": {
                "execution_intent_id": f"intent-{evaluation_id}",
                "status": "filled",
                "api_secret": "journal-secret-must-not-export",
            },
        },
        trades=[
            {
                "action": "buy",
                "symbol": "BTC-USDT",
                "execution": "paper_filled",
                "price": 80.0,
                "quantity": 1.25,
                "quote_amount": 100.0,
                "realized_pnl_quote": 12.0,
                "sizing": {"requested_quote": 100.0},
            }
        ],
        funding=[
            {
                "funding_rate": 0.0001,
                "current_notional_quote": 0.0,
                "projected_notional_quote": 100.0,
                "estimated_payment_quote": -0.01,
                "direction": "debit",
            }
        ],
    )


def _client() -> tuple[TestClient, list[CurrentPrincipal], ExportRepository]:
    repository = ExportRepository()
    repository.strategies[("tenant-a", "strategy-a")] = _strategy(
        "tenant-a", "strategy-a"
    )
    service = RuleStrategyService(repository=repository)
    app = FastAPI()
    app.include_router(create_rule_strategy_router(service=service))
    principal = [CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")]
    app.dependency_overrides[get_current_principal] = lambda: principal[0]
    return TestClient(app), principal, repository


def _sheet_names(payload: bytes) -> list[str]:
    with ZipFile(BytesIO(payload)) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    return [
        sheet.attrib["name"]
        for sheet in workbook.findall(f"{{{_MAIN_NS}}}sheets/{{{_MAIN_NS}}}sheet")
    ]


def _sheet_rows(payload: bytes, sheet_name: str) -> list[list[str]]:
    with ZipFile(BytesIO(payload)) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheet = next(
            item
            for item in workbook.findall(
                f"{{{_MAIN_NS}}}sheets/{{{_MAIN_NS}}}sheet"
            )
            if item.attrib["name"] == sheet_name
        )
        relationship_id = sheet.attrib[f"{{{_REL_NS}}}id"]
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        target = next(
            item.attrib["Target"]
            for item in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
            if item.attrib["Id"] == relationship_id
        )
        worksheet = ElementTree.fromstring(archive.read(f"xl/{target}"))

    rows = []
    for row in worksheet.findall(
        f"{{{_MAIN_NS}}}sheetData/{{{_MAIN_NS}}}row"
    ):
        values = []
        for cell in row.findall(f"{{{_MAIN_NS}}}c"):
            if cell.attrib.get("t") == "inlineStr":
                text = cell.find(f"{{{_MAIN_NS}}}is/{{{_MAIN_NS}}}t")
                values.append(text.text if text is not None and text.text else "")
            else:
                value = cell.find(f"{{{_MAIN_NS}}}v")
                values.append(value.text if value is not None and value.text else "")
        rows.append(values)
    return rows


def _evaluation_ids(rows: list[list[str]]) -> set[str]:
    evaluation_id_column = rows[0].index("评估 ID")
    return {
        row[evaluation_id_column]
        for row in rows[1:]
        if len(row) > evaluation_id_column and row[evaluation_id_column]
    }


def test_strategy_export_returns_valid_xlsx_with_all_requested_data() -> None:
    client, _, repository = _client()
    repository.journals.append(
        _journal("evaluation-a", datetime(2026, 1, 10, 8, tzinfo=timezone.utc))
    )
    repository.intents.append(
        SimpleNamespace(
            id="intent-evaluation-a",
            tenant_id="tenant-a",
            strategy_id="strategy-a",
            evaluation_id="evaluation-a",
            execution_generation=2,
            execution_source="rule_strategy",
            credential_id="credential-a",
            idempotency_key="idempotency-a",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            requested_quote="100",
            requested_quantity="1.25",
            status="filled",
            attempt_count=1,
            error_code=None,
            error_message=None,
            submitted_at=datetime(2026, 1, 10, 8, tzinfo=timezone.utc),
            terminal_at=datetime(2026, 1, 10, 8, tzinfo=timezone.utc),
            request_payload={"api_key": "intent-secret-must-not-export"},
            created_at=datetime(2026, 1, 10, 8, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 10, 8, tzinfo=timezone.utc),
        )
    )
    repository.orders.append(
        SimpleNamespace(
            id="order-a",
            tenant_id="tenant-a",
            strategy_id="strategy-a",
            evaluation_id="evaluation-a",
            execution_intent_id="intent-evaluation-a",
            execution_generation=2,
            execution_source="rule_strategy",
            credential_id="credential-a",
            provider="okx",
            client_order_id="client-order-a",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            requested_quote="100",
            requested_quantity="1.25",
            status="filled",
            exchange_order_id="venue-order-a",
            sandbox=True,
            response_metadata={"api_secret": "order-secret-must-not-export"},
            error_code=None,
            created_at=datetime(2026, 1, 10, 8, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 10, 8, tzinfo=timezone.utc),
        )
    )

    response = client.get("/rule-strategies/strategy-a/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert response.headers["content-disposition"] == (
        'attachment; filename="strategy-export-strategy-a.xlsx"'
    )
    assert ZipFile(BytesIO(response.content)).testzip() is None
    assert _sheet_names(response.content) == [
        "导出说明",
        "策略参数",
        "成交明细",
        "资金变化",
        "执行明细",
        "资金费",
    ]

    parameter_rows = _sheet_rows(response.content, "策略参数")
    assert ["config.initial_capital_quote", "10000.0"] in parameter_rows
    assert any("BTC-USDT" in row for row in _sheet_rows(response.content, "成交明细"))
    assert any("1015.0" in row for row in _sheet_rows(response.content, "资金变化"))
    assert any("intent-evaluation-a" in row for row in _sheet_rows(response.content, "执行明细"))
    assert any("0.0001" in row for row in _sheet_rows(response.content, "资金费"))
    assert b"journal-secret-must-not-export" not in response.content
    assert b"intent-secret-must-not-export" not in response.content
    assert b"order-secret-must-not-export" not in response.content


def test_strategy_export_filters_every_journal_sheet_to_inclusive_utc_dates() -> None:
    client, _, repository = _client()
    repository.journals.extend(
        [
            _journal("before", datetime(2026, 1, 9, 23, 59, 59, tzinfo=timezone.utc)),
            _journal("start", datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)),
            _journal(
                "end",
                datetime(2026, 1, 10, 23, 59, 59, 999999, tzinfo=timezone.utc),
            ),
            _journal("after", datetime(2026, 1, 11, 0, 0, tzinfo=timezone.utc)),
        ]
    )

    response = client.get(
        "/rule-strategies/strategy-a/export",
        params={"from_date": "2026-01-10", "to_date": "2026-01-10"},
    )

    assert response.status_code == 200
    for sheet_name in ("成交明细", "资金变化", "执行明细", "资金费"):
        assert _evaluation_ids(_sheet_rows(response.content, sheet_name)) == {
            "start",
            "end",
        }
    note_rows = _sheet_rows(response.content, "导出说明")
    assert ["导出日期范围 (UTC)", "2026-01-10 至 2026-01-10（含）"] in note_rows


def test_strategy_export_is_tenant_scoped() -> None:
    client, principal, repository = _client()
    repository.journals.extend(
        [
            _journal("tenant-a-evaluation", datetime(2026, 1, 10, tzinfo=timezone.utc)),
            _journal(
                "tenant-b-evaluation",
                datetime(2026, 1, 10, tzinfo=timezone.utc),
                tenant_id="tenant-b",
            ),
        ]
    )

    principal[0] = CurrentPrincipal(user_id="user-b", tenant_id="tenant-b")
    denied = client.get("/rule-strategies/strategy-a/export")

    assert denied.status_code == 404

    principal[0] = CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")
    allowed = client.get("/rule-strategies/strategy-a/export")

    assert allowed.status_code == 200
    execution_ids = _evaluation_ids(_sheet_rows(allowed.content, "执行明细"))
    assert "tenant-a-evaluation" in execution_ids
    assert "tenant-b-evaluation" not in execution_ids


def test_strategy_export_rejects_descending_dates() -> None:
    client, _, _ = _client()

    response = client.get(
        "/rule-strategies/strategy-a/export",
        params={"from_date": "2026-01-11", "to_date": "2026-01-10"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "from_date must be on or before to_date"
