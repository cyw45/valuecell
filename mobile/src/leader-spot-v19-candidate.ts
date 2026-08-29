export interface LeaderSpotV19BoxBreakoutEvidence {
  parameter_source: "V16.1";
  parameter_fingerprint: string;
  upper_bound: number;
  fifteen_minute_close_confirmed: boolean;
  five_minute_close_confirmations: number;
  second_five_minute_volume_confirmed: boolean;
  volume_multiplier: number;
  passed: boolean;
}

export interface LeaderSpotV19ScoreEvidence {
  formula_source: string;
  formula_fingerprint: string;
  total_score: number;
  factors: Record<string, number>;
}

export interface LeaderSpotV19CandidateStep {
  stage:
    | "entry_state"
    | "liquidity"
    | "new_coin"
    | "relative_strength"
    | "anomaly"
    | "box_breakout"
    | "score"
    | "order_book";
  passed: boolean;
  reason_code: string | null;
  facts: Record<string, number | boolean | string | null>;
}

export interface LeaderSpotV19CandidateDecision {
  symbol: string;
  source_rank: number;
  accepted: boolean;
  score: number | null;
  reason_code: string | null;
  steps: LeaderSpotV19CandidateStep[];
  observed_at: string;
}
