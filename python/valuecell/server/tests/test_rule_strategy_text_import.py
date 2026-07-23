import httpx
import pytest
from pydantic import ValidationError

from valuecell.server.api.schemas.rule_strategy import RuleStrategyTextImportProposal
from valuecell.server.services.rule_strategy_text_import_service import (
    RuleStrategyTextImportService,
)


@pytest.mark.asyncio
async def test_text_import_retries_one_transient_read_timeout(monkeypatch):
    payload = {
        "strategy_name": "均线策略",
        "executable": True,
        "config": {
            "interval": "4h",
            "program": {
                "schema_version": 2,
                "entry": {
                    "op": "compare",
                    "left": {"kind": "price", "interval": "4h", "source": "close"},
                    "comparator": "gt",
                    "right": {
                        "kind": "indicator",
                        "name": "ma",
                        "interval": "4h",
                        "period": 25,
                    },
                },
            },
            "risk": {"order_quote_amount": 100},
        },
        "summary": "四小时收盘价高于二十五期均线时买入。",
        "unresolved_items": [],
        "corrections": [],
        "rejection_reasons": [],
    }
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("provider stalled", request=request)
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": __import__("json").dumps(payload)}}]},
        )

    class ProviderConfig:
        api_key = "test-key"
        base_url = "https://provider.invalid/v1"
        default_model = "test-model"

    class ConfigManager:
        primary_provider = "openai-compatible"

        @staticmethod
        def get_provider_config(provider: str):
            assert provider == "openai-compatible"
            return ProviderConfig()

    monkeypatch.setattr(
        "valuecell.config.manager.get_config_manager", lambda: ConfigManager()
    )
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            assert kwargs["timeout"].read == 60.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, **kwargs):
            request = httpx.Request("POST", url)
            return await handler(request)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    proposal = await RuleStrategyTextImportService().parse("四小时收盘价高于MA25时买入")

    assert proposal.executable is True
    assert attempts == 2


def test_text_import_json_is_validated_as_a_reviewable_strategy_draft():
    raw = """
    ```json
    {
      "strategy_name": "多周期回调策略",
      "config": {
        "interval": "15m",
        "advanced_rules": {
          "enabled": true,
          "entry_confirmation_mode": "all",
          "exit_confirmation_mode": "any",
          "moving_average": {
            "enabled": true,
            "interval": "1d",
            "period": 20,
            "entry_comparator": "above"
          }
        },
        "risk": {
          "order_quote_amount": 100,
          "take_profit_pct": null,
          "stop_loss_pct": null,
          "max_positions": 100,
          "leverage": 1
        }
      },
      "summary": "日线价格高于二十日均线后，等待其他多周期条件确认。",
      "unresolved_items": ["未指定交易币种，保留用户当前选择。"]
    }
    ```
    """

    proposal = RuleStrategyTextImportProposal.model_validate(
        RuleStrategyTextImportService._parse_json_content(raw)
    )

    assert proposal.strategy_name == "多周期回调策略"
    assert proposal.config.advanced_rules.moving_average.interval == "1d"
    assert proposal.config.advanced_rules.moving_average.period == 20
    assert proposal.config.risk.take_profit_pct is None


def test_text_import_normalizes_human_percentages_before_validation():
    payload = {
        "strategy_name": "止损策略",
        "config": {
            "interval": "4h",
            "program": {
                "schema_version": 2,
                "entry": {
                    "op": "compare",
                    "left": {"kind": "price", "interval": "4h", "source": "close"},
                    "comparator": "gt",
                    "right": {"kind": "indicator", "name": "ma", "interval": "4h", "period": 25},
                },
            },
            "risk": {
                "order_quote_amount": 100,
                "stop_loss_pct": 8,
                "trailing_take_profit_pct": 8,
                "max_total_position_pct": 60,
                "max_symbol_position_pct": 10,
                "max_positions": 3,
                "leverage": 1,
            },
        },
        "summary": "价格高于均线时开仓，固定止损百分之八。",
        "unresolved_items": [],
        "corrections": [],
    }

    proposal = RuleStrategyTextImportService._validate_proposal(payload)

    assert proposal.config is not None
    assert proposal.config.risk.stop_loss_pct == pytest.approx(0.08)
    assert proposal.config.risk.trailing_take_profit_pct == pytest.approx(0.08)
    assert proposal.config.risk.max_total_position_pct == pytest.approx(0.6)
    assert proposal.config.risk.max_symbol_position_pct == pytest.approx(0.1)
    assert any("8" in correction for correction in proposal.corrections)


def test_text_import_rejected_proposal_has_real_reason_and_no_config():
    proposal = RuleStrategyTextImportProposal.model_validate(
        {
            "strategy_name": "模糊策略",
            "executable": False,
            "config": None,
            "summary": "缺少可执行阈值。",
            "unresolved_items": ["放量没有定义成交量窗口和倍数"],
            "rejection_reasons": ["成交量条件缺少窗口和倍数，无法确定性执行"],
        }
    )

    assert proposal.executable is False
    assert proposal.config is None


def test_import_prompt_advertises_stateful_paper_risk_capabilities():
    prompt = RuleStrategyTextImportService._system_prompt()

    assert "trailing_take_profit_pct" in prompt
    assert "max_total_position_pct" in prompt
    assert "max_symbol_position_pct" in prompt
    assert "add_to_winners" in prompt
    assert "默认 3" in prompt
    assert "从 program 中省略" in prompt
    assert "暂不可执行" not in prompt


def test_executable_text_import_requires_a_validated_config():
    with pytest.raises(ValidationError, match="config is required"):
        RuleStrategyTextImportProposal.model_validate(
            {
                "executable": True,
                "config": None,
                "summary": "错误响应",
                "unresolved_items": [],
            }
        )
