"""Aggregated Strategy API router.

Unifies strategy-related endpoints under a single registration point,
while keeping logical sub-routers separated for clarity.
"""

from fastapi import APIRouter

from .strategy import create_strategy_router
from .strategy_schema import create_strategy_schema_router
from .strategy_experiment import create_strategy_experiment_router


def create_strategy_api_router(quant_only_mode: bool = False) -> APIRouter:
    router = APIRouter()

    # Include core strategy endpoints (prefix: /strategies)
    router.include_router(create_strategy_router())

    if not quant_only_mode:
        from .strategy_agent import create_strategy_agent_router
        from .strategy_prompts import create_strategy_prompts_router

        # Include StrategyAgent endpoints (prefix: /strategies)
        router.include_router(create_strategy_agent_router())

        # Include strategy prompts endpoints (prefix: /strategies/prompts)
        router.include_router(create_strategy_prompts_router())

    # Include dynamic strategy schema endpoints (prefix: /strategies/schemas)
    router.include_router(create_strategy_schema_router())
    router.include_router(create_strategy_experiment_router())

    return router
