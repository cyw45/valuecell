from datetime import datetime, timezone

from valuecell.server.services.rule_strategy_service import RuleStrategyService
from valuecell.server.api.routers.rule_strategy import FixedStrategyCreateRequest
from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig


class Repository:
    def __init__(self) -> None:
        self.items = []
        self.created = None
        self.state = None

    def list(self, tenant_id: str, include_archived: bool = False):
        return self.items

    def create_with_current_state(self, strategy, *, scope, credential_id, symbol_candidates):
        strategy.created_at = datetime.now(timezone.utc)
        strategy.updated_at = strategy.created_at
        self.created = strategy
        self.state = (scope, credential_id, tuple(symbol_candidates))
        self.items.append(strategy)
        return strategy

    def get_evaluations(self, strategy_id: str, tenant_id: str, limit: int = 100_000, batch_id=None):
        return []


def test_fixed_demo_strategy_persists_shared_demo_environment_and_scope() -> None:
    repository = Repository()
    result = RuleStrategyService(repository=repository).create_fixed(
        "tenant-a",
        kind="dual_ma_trend",
        name="双均线 Demo",
        initial_capital_quote=1_000,
        environment="okx_demo",
        credential_id="credential-a",
    )

    assert result["mode"] == "okx_demo"
    assert result["config"]["execution"]["environment"] == "okx_demo"
    assert repository.state[0] == "shared_exchange_account"
    assert repository.state[1] == "credential-a"


def test_fixed_strategy_request_defaults_to_shared_okx_demo() -> None:
    request = FixedStrategyCreateRequest(
        kind="dual_ma_trend",
        name="双均线默认 Demo",
        initial_capital_quote=1_000,
    )

    assert request.environment == "okx_demo"


def test_configurable_demo_strategy_uses_shared_account_scope() -> None:
    repository = Repository()
    config = RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "risk": {"order_quote_amount": 100},
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "credential-a",
            },
        }
    )

    result = RuleStrategyService(repository=repository).create(
        "tenant-a", "可配置 Demo", None, config
    )

    assert result["mode"] == "okx_demo"
    assert repository.state[0] == "shared_exchange_account"
    assert repository.state[1] == "credential-a"
