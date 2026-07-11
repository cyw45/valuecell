"""Crypto market data API for strategy dashboards."""

from fastapi import APIRouter, HTTPException, Query, Response
from valuecell.server.config.settings import get_settings

from valuecell.server.api.schemas import SuccessResponse
from valuecell.server.api.schemas.crypto_market import (
    CryptoMarketIndicatorsData,
    CryptoSymbolCatalogData,
)
from valuecell.server.services.crypto_market_service import get_crypto_market_service


def create_crypto_market_router() -> APIRouter:
    settings = get_settings()
    router = APIRouter(prefix="/crypto-market", tags=["Crypto Market"])

    @router.get(
        "/symbols",
        response_model=SuccessResponse[CryptoSymbolCatalogData],
        summary="List supported crypto symbols",
    )
    async def get_crypto_symbols():
        service = get_crypto_market_service()
        return SuccessResponse.create(
            data=service.get_supported_symbols(),
            msg="Crypto symbols retrieved successfully",
        )

    @router.get(
        "/health",
        response_model=SuccessResponse[dict],
        summary="Get public market-data provider health",
    )
    async def get_crypto_market_health():
        """Expose cache, cooldown, and provider state without any secrets."""
        return SuccessResponse.create(
            data=get_crypto_market_service().get_health(),
            msg="Crypto market-data provider health retrieved",
        )

    @router.get(
        "/indicators",
        response_model=SuccessResponse[CryptoMarketIndicatorsData],
        summary="Get crypto candles and indicators",
    )
    async def get_crypto_indicators(
        response: Response,
        symbols: str = Query(
            ",".join(settings.MARKET_DEFAULT_SYMBOLS),
            description="Comma-separated USDT crypto symbols, e.g. BTC-USDT,ETH-USDT",
        ),
        interval: str = Query(settings.MARKET_DEFAULT_INTERVAL, description="Candle interval"),
        lookback: int = Query(
            settings.MARKET_DEFAULT_LOOKBACK,
            ge=1,
            le=500,
            description="Number of candles",
        ),
        providers: str | None = Query(
            None,
            description="Optional comma-separated provider fallback order: okx,binance,gate,mexc",
        ),
    ):
        symbol_list = [item.strip() for item in symbols.split(",") if item.strip()]
        provider_list = (
            [item.strip() for item in providers.split(",") if item.strip()]
            if providers
            else None
        )
        service = get_crypto_market_service()
        if (
            providers is None
            and tuple(symbol_list) == settings.MARKET_DEFAULT_SYMBOLS
            and interval.strip().lower() == settings.MARKET_DEFAULT_INTERVAL
            and lookback == settings.MARKET_DEFAULT_LOOKBACK
        ):
            snapshot = service.get_default_snapshot()
            if snapshot is not None:
                response.headers["X-ValueCell-Market-Cache"] = "default-snapshot"
                return SuccessResponse.create(
                    data=snapshot.data,
                    msg="Crypto indicators retrieved from default market snapshot",
                )
        try:
            data = await service.get_indicators(
                symbols=symbol_list,
                interval=interval,
                lookback=lookback,
                providers=provider_list,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Crypto market data unavailable: {exc}",
            ) from exc
        return SuccessResponse.create(
            data=data,
            msg="Crypto indicators retrieved successfully",
        )

    return router
