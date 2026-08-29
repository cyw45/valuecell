export interface LeaderSpotV19BacktestFill {
  symbol: string;
  side: "buy" | "sell";
  decision_timestamp_ms: number;
  fill_timestamp_ms: number;
  decision_price: number;
  fill_price: number;
  quantity: number;
  quote_amount: number;
  fee_quote: number;
  slippage_pct: number;
  realized_pnl_quote: number;
  reason_code: string;
}

export interface LeaderSpotV19WalkForwardWindow {
  train_start_ms: number;
  train_end_ms: number;
  test_start_ms: number;
  test_end_ms: number;
  test_metrics: Record<string, number>;
}

export interface LeaderSpotV19BacktestResult {
  data_fingerprint: string;
  config_fingerprint: string;
  assumptions_fingerprint: string;
  fills: LeaderSpotV19BacktestFill[];
  metrics: Record<string, number>;
  walk_forward: LeaderSpotV19WalkForwardWindow[];
}
