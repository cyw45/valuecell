export type LeaderSpotV19EntryProfile =
  | "halt"
  | "degraded"
  | "standard"
  | "strong_trend";

export interface LeaderSpotV19MarketCondition {
  code: string;
  passed: boolean;
  actual: number | boolean | string;
  threshold: number | boolean | string;
}

export interface LeaderSpotV19SignalStarvationPolicy {
  elapsed_hours: number;
  recovered: boolean;
  relative_strength_rank_pct: number;
  liquidity_quote: number;
  score_threshold: number;
}

export interface LeaderSpotV19MarketStateDecision {
  market_state: "M0" | "M1" | "M2" | "M3" | "M4";
  entry_profile: LeaderSpotV19EntryProfile;
  can_open: boolean;
  reason_codes: string[];
  conditions: LeaderSpotV19MarketCondition[];
  starvation: LeaderSpotV19SignalStarvationPolicy;
  observed_at: string;
}
