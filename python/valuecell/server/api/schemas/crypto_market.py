"""Crypto market indicator API schemas."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .base import SuccessResponse


class CryptoCandleData(BaseModel):
    ts: int = Field(..., description="Candle timestamp in milliseconds")
    open: float
    high: float
    low: float
    close: float
    volume: float


class BollingerBandData(BaseModel):
    upper: Optional[float] = None
    middle: Optional[float] = None
    lower: Optional[float] = None


class CryptoIndicatorPointData(BaseModel):
    ts: int = Field(..., description="Indicator timestamp in milliseconds")
    ma: Dict[str, Optional[float]] = Field(default_factory=dict)
    rsi: Optional[float] = None
    bollinger: BollingerBandData = Field(default_factory=BollingerBandData)
    momentum: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None


class CryptoSymbolIndicatorsData(BaseModel):
    symbol: str
    exchange_symbol: str
    provider: str
    interval: str
    candles: List[CryptoCandleData]
    indicators: List[CryptoIndicatorPointData]
    latest_price: Optional[float] = None
    warning: Optional[str] = None
    snapshot_ts_ms: Optional[int] = Field(None, description="Source timestamp of the latest returned candle")
    freshness_age_ms: Optional[int] = Field(None, description="Age of the latest returned candle at response construction")
    freshness_status: str = Field("unknown", description="Candle freshness: fresh, stale, or unknown")
    coverage_status: str = Field("complete", description="Symbol response coverage status")


class CryptoMarketIndicatorsData(BaseModel):
    interval: str
    lookback: int
    providers: List[str]
    symbols: List[CryptoSymbolIndicatorsData]
    snapshot_fetched_at: Optional[str] = Field(
        None, description="Backend snapshot refresh time in UTC"
    )
    failed_symbols: Dict[str, str] = Field(default_factory=dict)


class CryptoSymbolCatalogData(BaseModel):
    quote_asset: str = "USDT"
    symbols: List[str]


CryptoMarketIndicatorsResponse = SuccessResponse[CryptoMarketIndicatorsData]
CryptoSymbolCatalogResponse = SuccessResponse[CryptoSymbolCatalogData]
