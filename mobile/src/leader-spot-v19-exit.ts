export type LeaderSpotV19ExitReason =
  | "STOP_LOSS_8PCT"
  | "LOSS_CIRCUIT_7D"
  | "MOVING_STOP"
  | "LAYERED_RETRACEMENT"
  | "TREND_EXIT";

export interface LeaderSpotV19ExitDecision {
  position_id: string;
  protection_status: "PROTECTION_NONE" | "PROTECTION_PENDING" | "PROTECTION_ACTIVE";
  peak_price: number;
  peak_profit_pct: number;
  moving_stop_price: number;
  layered_exit_price: number | null;
  loss_circuit_active: boolean;
  trend_break_count: number;
  exit_reason_code: LeaderSpotV19ExitReason | null;
  order_type: "market" | "limit" | null;
  limit_price: number | null;
  observed_at: string;
}
