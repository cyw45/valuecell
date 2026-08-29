export interface LeaderSpotV19EntryTier {
  tier: 1 | 2 | 3;
  offset_pct: number;
  wait_seconds: number;
}

export interface LeaderSpotV19EntryOrderResult {
  client_order_id: string;
  venue_order_id: string | null;
  status: "filled" | "open" | "cancelled" | "rejected" | "submission_unknown";
  filled_quantity: number;
  average_price: number | null;
  fee_quote: number;
}

export interface LeaderSpotV19EntryDecision {
  accepted: boolean;
  reason_code: string | null;
  symbol: string;
  order_amount_quote: number;
  tier_results: LeaderSpotV19EntryOrderResult[];
  position_id: string | null;
  observed_at: string;
}
