export interface CryptoCandle {
  ts: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface CryptoBollingerBand {
  upper?: number | null;
  middle?: number | null;
  lower?: number | null;
}

export interface CryptoIndicatorPoint {
  ts: number;
  ma: Record<string, number | null | undefined>;
  rsi?: number | null;
  bollinger: CryptoBollingerBand;
  momentum?: number | null;
  macd?: number | null;
  macd_signal?: number | null;
  macd_histogram?: number | null;
}

export interface CryptoSymbolIndicators {
  symbol: string;
  exchange_symbol: string;
  provider: string;
  interval: string;
  candles: CryptoCandle[];
  indicators: CryptoIndicatorPoint[];
  latest_price?: number | null;
  warning?: string | null;
  snapshot_ts_ms?: number | null;
  freshness_age_ms?: number | null;
  freshness_status: "fresh" | "stale" | "unknown";
  coverage_status: "complete" | "partial";
}

export interface CryptoMarketIndicators {
  interval: string;
  lookback: number;
  providers: string[];
  symbols: CryptoSymbolIndicators[];
  failed_symbols: Record<string, string>;
}

export interface CryptoSymbolCatalog {
  quote_asset: string;
  symbols: string[];
}
