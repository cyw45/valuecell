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
        from_ts_ms: int | None = Query(
            None, gt=0, description="Inclusive UTC range start in milliseconds"
        ),
        to_ts_ms: int | None = Query(
            None, gt=0, description="Inclusive UTC range end in milliseconds"
        ),
    ):
        symbol_list = [item.strip() for item in symbols.split(",") if item.strip()]
        provider_list = (
            [item.strip() for item in providers.split(",") if item.strip()]
            if providers
            else None
        )
        service = get_crypto_market_service()
        is_default_snapshot_query = (
            providers is None
            and interval.strip().lower() == settings.MARKET_DEFAULT_INTERVAL
            and lookback <= settings.MARKET_DEFAULT_LOOKBACK
            and set(symbol_list).issubset(set(settings.MARKET_DEFAULT_SYMBOLS))
            and from_ts_ms is None
            and to_ts_ms is None
        )
        if is_default_snapshot_query:
            snapshot = service.get_default_snapshot()
            if snapshot is not None:
                snapshot_symbols = {item.symbol: item for item in snapshot.data.symbols}
                selected = [
                    snapshot_symbols[symbol].model_copy(
                        update={"candles": snapshot_symbols[symbol].candles[-lookback:]}
                    )
                    for symbol in symbol_list
                    if symbol in snapshot_symbols
                ]
                if selected:
                    response.headers["X-ValueCell-Market-Cache"] = "default-snapshot"
                    return SuccessResponse.create(
                        data=CryptoMarketIndicatorsData(
                            interval=settings.MARKET_DEFAULT_INTERVAL,
                            lookback=lookback,
                            providers=snapshot.data.providers,
                            symbols=selected,
                            failed_symbols={
                                symbol: "market snapshot unavailable"
                                for symbol in symbol_list
                                if symbol not in snapshot_symbols
                            },
                            snapshot_fetched_at=snapshot.fetched_at.isoformat(),
                        ),
                        msg="Crypto indicators retrieved from shared market snapshot",
                    )
        if is_default_snapshot_query:
            raise HTTPException(
                status_code=503,
                detail="Shared market snapshot is warming; retry shortly",
            )
        try:
            request_kwargs = {
                "symbols": symbol_list,
                "interval": interval,
                "lookback": lookback,
                "providers": provider_list,
            }
            if from_ts_ms is not None:
                request_kwargs["from_ts_ms"] = from_ts_ms
            if to_ts_ms is not None:
                request_kwargs["to_ts_ms"] = to_ts_ms
            data = await service.get_indicators(**request_kwargs)
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
