export type PredictionMarketMode = "paper";
export type PredictionMarketFreshnessStatus =
  | "fresh"
  | "delayed"
  | "stale"
  | "unavailable";

export interface PredictionMarketOutcome {
  outcome: string;
  token_id: string;
  price?: string;
}

export interface PredictionMarketSummary {
  market_id: string;
  slug: string;
  question: string;
  active: boolean;
  closed: boolean;
  outcomes: PredictionMarketOutcome[];
}

export interface PredictionMarketCatalog {
  source: "polymarket-public";
  mode: PredictionMarketMode;
  source_timestamp_ms: number;
  observed_at_ms: number;
  freshness_age_ms: number;
  freshness_status: PredictionMarketFreshnessStatus;
  markets: PredictionMarketSummary[];
}

export interface PredictionMarketBookLevel {
  price: string;
  size: string;
}

export interface PredictionMarketBookHealth {
  status?: string;
  reason?: string;
  crossed?: boolean;
  one_sided?: boolean;
  bid_levels?: number;
  ask_levels?: number;
}

export interface PredictionMarketOrderBook {
  bids: PredictionMarketBookLevel[];
  asks: PredictionMarketBookLevel[];
  best_bid?: string;
  best_ask?: string;
  midpoint?: string;
  microprice?: string;
  health?: PredictionMarketBookHealth;
}

export interface PredictionMarketSignal {
  reference_price?: string;
  reference_method?: string;
  volatility?: string;
  observation_count?: number;
  volatility_status?: string;
}

export interface PredictionMarketSnapshot {
  source: "polymarket-public";
  mode: PredictionMarketMode;
  source_timestamp_ms: number;
  observed_at_ms: number;
  freshness_age_ms: number;
  freshness_status: PredictionMarketFreshnessStatus;
  market_id: string;
  question: string;
  outcome: string;
  token_id: string;
  book: PredictionMarketOrderBook;
  signal?: PredictionMarketSignal;
}

export interface PredictionReplayRequest {
  decision_time_ms: number;
  latency_ms: number;
  order: {
    side: "buy" | "sell";
    size: number;
    max_levels: number;
    extra_slippage_bps: number;
  };
  snapshots: Array<{
    source_timestamp_ms: number;
    observed_at_ms: number;
    bids: PredictionMarketBookLevel[];
    asks: PredictionMarketBookLevel[];
  }>;
}

export interface PredictionReplayResult {
  source: "polymarket-public";
  mode: "paper";
  simulation_mode: "simulated";
  source_timestamp_ms: number | null;
  observed_at_ms: number | null;
  freshness_age_ms: number | null;
  freshness_status: PredictionMarketFreshnessStatus;
  fingerprint: string;
  assumptions: {
    eligible_time_ms: number;
    execution_snapshot_timestamp_ms: number | null;
    max_levels: number;
    extra_slippage_bps: number;
    remainder_policy: "cancel";
    liquidity_scope: "visible_frozen_levels";
  };
  fill: {
    requested_size: number;
    filled_size: number;
    unfilled_size: number;
    vwap: number | null;
    levels_consumed: number;
  };
  mark_to_book: {
    mark_price: number | null;
    pnl: number;
    currency: "quote";
  };
}
