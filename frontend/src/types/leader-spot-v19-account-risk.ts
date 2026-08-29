export interface LeaderSpotV19AccountRiskDecision {
  state: "normal" | "daily_loss_halted" | "equity_halted";
  can_open: boolean;
  daily_realized_pnl_quote: number;
  daily_loss_reset_at: string;
  equity_drawdown_pct: number;
  halt_until: string | null;
  reason_code: string | null;
  cancel_pending_entries: boolean;
  force_close_positions: boolean;
  observed_at: string;
}

export interface LeaderSpotV19RiskCancellationResult {
  cancelled_intent_ids: string[];
  preserved_intent_ids: string[];
}
