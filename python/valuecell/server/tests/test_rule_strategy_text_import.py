from valuecell.server.api.schemas.rule_strategy import RuleStrategyTextImportProposal
from valuecell.server.services.rule_strategy_text_import_service import (
    RuleStrategyTextImportService,
)


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
