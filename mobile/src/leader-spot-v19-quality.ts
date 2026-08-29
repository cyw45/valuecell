export type LeaderSpotV19QualityState = "DATA_OK" | "DATA_DEGRADED" | "DATA_UNSAFE";
export type LeaderSpotV19IssueSeverity = "degraded" | "unsafe";

export interface LeaderSpotV19QualityIssue {
  code: string;
  severity: LeaderSpotV19IssueSeverity;
  detail: string;
  symbol: string | null;
}

export interface LeaderSpotV19DataQualityReport {
  data_state: LeaderSpotV19QualityState;
  observed_at: string;
  issues: LeaderSpotV19QualityIssue[];
  checked_symbols: string[];
  fresh_input_count: number;
  required_input_count: number;
  accepted_for_entry: boolean;
}

export interface LeaderSpotV19RecoveryExit {
  symbol: string;
  quantity: number;
  reason_code: string;
  local_triggered_at: string;
  venue_order_id: string | null;
}

export interface LeaderSpotV19RecoveryObservation {
  positions: Array<Record<string, unknown>>;
  orders: Array<Record<string, unknown>>;
  due_exits: LeaderSpotV19RecoveryExit[];
  observed_at: string;
}
