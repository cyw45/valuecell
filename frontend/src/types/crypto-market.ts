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

export type CryptoSymbolUniverseState = "admitted" | "rejected" | string;

export type CryptoSymbolUniverseDecision =
  | "added"
  | "retained"
  | "removed"
  | "rejected"
  | string;

export interface CryptoSymbolUniverseEntry {
  symbol: string;
  state: CryptoSymbolUniverseState;
  decision: CryptoSymbolUniverseDecision;
  reason_code: string;
  reason_detail?: string | null;
  permanent_exclusion: boolean;
  listed_at?: string | null;
  listing_age_days?: number | null;
  average_quote_volume_30d?: number | null;
  quote_volume_24h?: number | null;
  price_quote?: number | null;
  observed_at?: string | null;
}

export interface CryptoSymbolUniverse {
  version: number;
  status: string;
  source: string;
  quote_asset: string;
  observed_at: string;
  next_sync_due_at?: string | null;
  sync_interval_days: number;
  min_listing_age_days: number;
  min_average_quote_volume_30d: number;
  evaluated_count: number;
  admitted_count: number;
  added_count: number;
  removed_count: number;
  retained_count: number;
  reason_detail?: string | null;
  entries: CryptoSymbolUniverseEntry[];
}
