export type LiveExchangeProvider = "binance" | "okx";
export type LiveMarketType = "spot" | "swap";
export type LiveOrderSide = "buy" | "sell";
export type LiveOrderType = "market" | "limit";

export interface LiveExecutionStatus {
  live_trading_enabled: boolean;
  authorization_active: boolean;
  authorization_expires_at?: string | null;
  gate_reasons: string[];
}

export interface LiveConnection {
  id: string;
  label: string;
  provider: LiveExchangeProvider;
  market_type: LiveMarketType;
  active: boolean;
  created_at: string;
}

export interface CreateLiveConnectionRequest {
  label: string;
  provider: LiveExchangeProvider;
  market_type: LiveMarketType;
  api_key: string;
  api_secret: string;
  passphrase?: string;
  withdrawal_disabled_confirmed: boolean;
  ip_allowlist_confirmed: boolean;
}

export interface LiveRiskPolicy {
  id?: string;
  max_order_notional: number;
  max_open_positions: number;
  max_leverage: number;
  allowed_symbols: string[];
  active?: boolean;
}

export interface LiveStrategyBinding {
  id: string;
  strategy_id: string;
  connection_id: string;
  active: boolean;
  revoked_at?: string | null;
  created_at: string;
}

export interface CreateLiveStrategyBindingRequest {
  strategy_id: string;
  connection_id: string;
}

export interface StartupAuthorizationChallenge {
  challenge_code: string;
  expires_at: string;
}

export interface ConfirmStartupAuthorizationRequest {
  challenge_code: string;
}

export interface CreateLiveOrderRequest {
  connection_id: string;
  symbol: string;
  side: LiveOrderSide;
  type: LiveOrderType;
  quote_amount: number;
  price?: number;
  idempotency_key: string;
}

export interface LiveOrder {
  id: string;
  status: string;
  exchange_order_id?: string | null;
  created_at: string;
}
