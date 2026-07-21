"""FastAPI application factory for ValueCell Server."""

import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from ...utils.env import ensure_system_env_dir, get_system_env_path
from ..config.settings import get_settings
from .exceptions import (
    APIException,
    api_exception_handler,
    general_exception_handler,
    validation_exception_handler,
)
from .routers.crypto_market import create_crypto_market_router
from .routers.prediction_market import create_prediction_market_router
from .routers.i18n import create_i18n_router
from .routers.saas_auth import create_saas_auth_router
from .routers.saas_admin import create_saas_admin_router
from .routers.tenant_credential import create_tenant_credential_router
from .routers.sandbox_exchange import create_sandbox_exchange_router
from .routers.live_execution import create_live_execution_router
from .routers.rule_strategy import create_rule_strategy_router
from .routers.strategy_api import create_strategy_api_router
from ..services.crypto_market_service import get_crypto_market_service
from ..services.sandbox_exchange_trading_service import SandboxExchangeTradingService
from ..services.demo_execution_reconciliation import (
    reconcile_active_tenant_intents as _reconcile_active_tenant_intents,
)
from ..db.connection import get_database_manager
from ..db.models.tenant import Tenant
from .routers.system import create_system_router
from .schemas.base import AppInfoData, SuccessResponse


async def reconcile_active_tenant_intents() -> None:
    """Reconcile ambiguous Demo submissions for active tenants only.

    This is deliberately a scheduled recovery loop rather than an automatic
    re-submit path: the exchange is queried by client order ID and unresolved
    requests remain explicitly marked for reconciliation.
    """
    session = get_database_manager().get_session()
    try:
        await _reconcile_active_tenant_intents(
            session,
            _active_tenant_ids(session),
            SandboxExchangeTradingService,
        )
    except Exception as exc:
        logger.warning("Demo execution reconciliation deferred: {}", exc)
    finally:
        session.close()


def _active_tenant_ids(session) -> list[str]:
    return [tenant_id for (tenant_id,) in session.query(Tenant.id).all()]


def _ensure_system_env_and_load() -> None:
    """Ensure the system `.env` exists and is loaded; use only the system path.

    Behavior:
    - If the system `.env` exists, load it with `override=True`.
    - If not, and the repository has `.env.example`, copy it to the system path and then load.
    - Do not create or load the repository root `.env`.
    """
    try:
        repo_root = Path(__file__).resolve().parents[4]
        sys_env = get_system_env_path()
        example_file = repo_root / ".env.example"

        try:
            import shutil

            if not sys_env.exists() and example_file.exists():
                ensure_system_env_dir()
                shutil.copy(example_file, sys_env)
        except Exception:
            pass

        # Docker/service-unit environment is authoritative. The user-level file
        # supplies only absent local-development values.
        if sys_env.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(sys_env, override=False)
            except Exception:
                try:
                    with open(sys_env, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                key, value = line.split("=", 1)
                                key = key.strip()
                                value = value.strip()
                                if (value.startswith('"') and value.endswith('"')) or (
                                    value.startswith("'") and value.endswith("'")
                                ):
                                    value = value[1:-1]
                                os.environ.setdefault(key, value)
                except Exception:
                    pass
    except Exception:
        # Do not block app creation if any step fails
        pass


def _run_required_execution_attribution_migration() -> None:
    """Run the fail-closed schema migration after table creation at startup."""
    from ..db.connection import get_database_manager
    from ..db.migrations import (
        ensure_single_running_rule_strategy_index,
        migrate_rule_strategy_execution_attribution,
    )

    session = get_database_manager().get_session()
    try:
        migrate_rule_strategy_execution_attribution(session)
        ensure_single_running_rule_strategy_index(session)
    finally:
        session.close()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    # Ensure .env exists and is loaded before reading settings
    _ensure_system_env_and_load()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        logger.info(
            f"ValueCell Server starting up on {settings.API_HOST}:{settings.API_PORT}..."
        )

        # Quant-only deployments must not initialize legacy Agent dependencies.
        if settings.QUANT_ONLY_MODE:
            from ..db.connection import create_tables

            logger.info("Provisioning SaaS database tables in quant-only mode")
            create_tables()
        else:
            try:
                from ...adapters.assets import get_adapter_manager
                from ..db import init_database

                logger.info("Initializing database tables...")
                init_database(force=False)
                logger.info("Configuring legacy data adapters...")
                manager = get_adapter_manager()
                manager.configure_yfinance()
                manager.configure_akshare()
                manager.configure_baostock()
            except Exception as exc:
                logger.warning("Legacy initialization unavailable: {}", exc)

        # This migration is required for every execution-capable deployment. Keep
        # it outside the best-effort legacy data migration below: failure must
        # prevent scheduler startup and therefore fail closed.
        _run_required_execution_attribution_migration()

        try:
            from ..db.connection import get_database_manager
            from ..db.migrations import (
                ensure_rule_strategy_journal_read_index,
                migrate_fixed_order_amounts,
                migrate_tenant_profiles,
            )
            from ..services.platform_bootstrap_service import (
                bootstrap_platform_administrator,
            )

            session = get_database_manager().get_session()
            try:
                ensure_rule_strategy_journal_read_index(session)
                migrate_fixed_order_amounts(session)
                migrate_tenant_profiles(session)
                result = bootstrap_platform_administrator(
                    session,
                    settings.BOOTSTRAP_PLATFORM_ADMIN_EMAIL,
                    settings.BOOTSTRAP_PLATFORM_ADMIN_PASSWORD,
                )
                if result.created:
                    logger.info(
                        "Bootstrapped platform administrator email={}", result.email
                    )
            finally:
                session.close()
        except Exception as exc:
            logger.warning("SaaS data migration deferred: {}", exc)

        # Market data is a backend-owned dependency. Begin prewarming at startup
        # without blocking the API while an upstream exchange is unavailable.
        market_service = get_crypto_market_service()

        async def _refresh_default_market_snapshot() -> None:
            while True:
                await market_service.refresh_default_snapshot()
                await asyncio.sleep(settings.MARKET_REFRESH_S)

        market_refresh_task = asyncio.create_task(_refresh_default_market_snapshot())
        _scheduler = None
        try:
            from apscheduler.triggers.interval import IntervalTrigger
            from ..db.connection import get_database_manager
            from ..services.strategy_scheduler import StrategyScheduler

            _scheduler = StrategyScheduler()
            await _scheduler.start()

            def _sync_job() -> None:
                try:
                    db = get_database_manager().get_session()
                    try:
                        _scheduler.sync_running_strategies(db)
                    finally:
                        db.close()
                except Exception as exc:
                    logger.warning(
                        "Strategy scheduler database sync deferred; retrying next cycle: {}",
                        exc,
                    )

            _scheduler._scheduler.add_job(
                _sync_job,
                trigger=IntervalTrigger(seconds=60),
                id="_scheduler_sync_running",
                replace_existing=True,
                coalesce=True,
            )
            _scheduler._scheduler.add_job(
                reconcile_active_tenant_intents,
                trigger=IntervalTrigger(seconds=60),
                id="_scheduler_reconcile_demo_execution",
                replace_existing=True,
                coalesce=True,
                max_instances=1,
            )
            # Attempt immediately, but database outages must not disable the
            # recurring job or prevent the API and market snapshot from starting.
            _sync_job()
            logger.info("Strategy scheduler started")
        except Exception as exc:
            logger.warning("Strategy scheduler initialization deferred: {}", exc)

        yield
        # Shutdown
        logger.info("ValueCell Server shutting down...")
        if _scheduler is not None:
            await _scheduler.stop()
        market_refresh_task.cancel()
        try:
            await market_refresh_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(
        title="ValueCell Server API",
        description="A community-driven, multi-agent platform for financial applications",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.API_DEBUG else None,
        redoc_url="/redoc" if settings.API_DEBUG else None,
    )

    # Add exception handlers
    _add_exception_handlers(app)

    # Add middleware
    _add_middleware(app, settings)

    # Add routes
    _add_routes(app, settings)

    return app


def _add_middleware(app: FastAPI, settings) -> None:
    """Add middleware to the application."""
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom logging middleware removed


def _add_exception_handlers(app: FastAPI) -> None:
    """Add exception handlers to the application."""
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)


API_PREFIX = "/api/v1"


def _add_routes(app: FastAPI, settings) -> None:
    """Add routes to the application."""

    # Root endpoint
    @app.get("/", response_model=SuccessResponse[AppInfoData])
    async def home_page():
        return SuccessResponse.create(
            data=AppInfoData(
                name=settings.APP_NAME,
                version=settings.APP_VERSION,
                environment=settings.APP_ENVIRONMENT,
            ),
            msg="Welcome to ValueCell Server API",
        )

    @app.get(f"{API_PREFIX}/healthz", response_model=SuccessResponse)
    async def health_check():
        return SuccessResponse.create(msg="Welcome to ValueCell!")

    # Include i18n router
    app.include_router(create_i18n_router(), prefix=API_PREFIX)

    # Include system router
    app.include_router(create_system_router(), prefix=API_PREFIX)

    app.include_router(create_saas_auth_router(), prefix=API_PREFIX)
    app.include_router(create_saas_admin_router(), prefix=API_PREFIX)
    app.include_router(create_tenant_credential_router(), prefix=API_PREFIX)
    app.include_router(create_sandbox_exchange_router(), prefix=API_PREFIX)
    app.include_router(create_live_execution_router(), prefix=API_PREFIX)

    # Quant-only deployments expose only deterministic SaaS and market APIs.
    # Legacy Agent, conversation, profile and watchlist APIs stay available in
    # non-quant deployments without forcing their dependencies at startup.
    if not settings.QUANT_ONLY_MODE:
        from .routers.agent import create_agent_router
        from .routers.agent_stream import create_agent_stream_router
        from .routers.conversation import create_conversation_router
        from .routers.models import create_models_router
        from .routers.task import create_task_router
        from .routers.user_profile import create_user_profile_router
        from .routers.watchlist import create_watchlist_router

        app.include_router(create_models_router(), prefix=API_PREFIX)
        app.include_router(create_conversation_router(), prefix=API_PREFIX)
        app.include_router(create_agent_stream_router(), prefix=API_PREFIX)
        app.include_router(create_agent_router(), prefix=API_PREFIX)
        app.include_router(create_task_router(), prefix=API_PREFIX)
        app.include_router(create_watchlist_router(), prefix=API_PREFIX)
        app.include_router(create_user_profile_router(), prefix=API_PREFIX)

    # Public quant market data and deterministic strategy APIs.
    app.include_router(create_crypto_market_router(), prefix=API_PREFIX)
    app.include_router(create_prediction_market_router(), prefix=API_PREFIX)

    # Deterministic strategy surface; agent routes are excluded in quant mode.
    # StrategyAgent/prompt endpoints and leaves deterministic quant APIs only.
    app.include_router(
        create_strategy_api_router(quant_only_mode=settings.QUANT_ONLY_MODE),
        prefix=API_PREFIX,
    )
    # Include standalone deterministic paper rule strategy API.
    app.include_router(create_rule_strategy_router(), prefix=API_PREFIX)


# For uvicorn
app = create_app()
