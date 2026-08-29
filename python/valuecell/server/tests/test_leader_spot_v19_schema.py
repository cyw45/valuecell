import pytest
from pydantic import ValidationError

from valuecell.server.api.schemas.leader_spot_v19 import (
    DEFAULT_LAYERED_EXIT_TIERS,
    DEFAULT_MOVING_STOP_TIERS,
    LEADER_SPOT_V19_MODULE_ID,
    LeaderSpotV19Config,
    LeaderSpotV19CreateRequest,
)


def test_v19_defaults_freeze_money_sizing_and_loss_contract() -> None:
    config = LeaderSpotV19Config()

    assert config.module_id == LEADER_SPOT_V19_MODULE_ID
    assert config.schema_version == 19
    assert config.position.order_amount_quote == 100
    assert config.position.max_positions == 6
    assert config.position.leverage_enabled is False
    assert config.position.contract_enabled is False
    assert config.loss.stop_loss_pct == 0.08
    assert config.loss.loss_circuit_hours == 168
    assert config.daily_loss_limit_quote == 48
    assert config.profit.moving_stop_tiers == DEFAULT_MOVING_STOP_TIERS
    assert config.profit.layered_exit_tiers == DEFAULT_LAYERED_EXIT_TIERS


def test_v19_rejects_old_strategy_fields_and_mutable_profit_ladders() -> None:
    with pytest.raises(ValidationError):
        LeaderSpotV19Config.model_validate({"max_positions": 3})

    config = LeaderSpotV19Config()
    changed_tiers = list(config.profit.moving_stop_tiers)
    changed_tiers[0] = changed_tiers[0].model_copy(update={"stop_multiplier": 1.01})
    payload = config.model_dump(mode="json")
    payload["profit"]["moving_stop_tiers"] = [tier.model_dump() for tier in changed_tiers]
    with pytest.raises(ValidationError):
        LeaderSpotV19Config.model_validate(payload)


def test_v19_execution_target_is_explicit() -> None:
    with pytest.raises(ValidationError):
        LeaderSpotV19Config(
            environment="okx_demo",
        )

    demo_config = LeaderSpotV19Config(
        environment="okx_demo",
        sandbox_connection_id="credential-v19",
    )
    assert demo_config.environment == "okx_demo"

    with pytest.raises(ValidationError):
        LeaderSpotV19CreateRequest(
            name="V19",
            config=LeaderSpotV19Config(
                environment="paper",
                sandbox_connection_id="must-not-be-present",
            ),
        )
