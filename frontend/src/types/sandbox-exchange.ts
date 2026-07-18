export type SandboxExchangeProvider = "binance" | "okx";
export type SandboxOrderSide = "buy" | "sell";
export type SandboxOrderType = "market" | "limit";

export interface SandboxConnectionMetadata {
  sandbox: true;
  provider: SandboxExchangeProvider;
  market_type: "spot";
  validated_at: string;
}

export interface SandboxConnection {
  id: string;
  label: string;
  provider: SandboxExchangeProvider;
  metadata: SandboxConnectionMetadata;
  created_at: string;
}

export interface CreateSandboxConnectionRequest {
  provider: SandboxExchangeProvider;
  label: string;
  api_key: string;
  api_secret: string;
  passphrase?: string;
}

export interface SandboxBalance {
  currency: string;
  free: string | number;
  used: string | number;
  frozen: string | number;
  total: string | number;
  mark_price_usdt: number | null;
  usdt_value: number | null;
  valuation_status: "priced" | "unpriced";
}

export interface SandboxConnectionBalance {
  source: "okx_demo" | "binance_demo";
  balances: SandboxBalance[];
  total_usdt_value: number;
  checked_at: string;
}

export interface SandboxPosition {
  symbol: string;
  base_currency: string;
  quantity: number;
  available_quantity: number;
  frozen_quantity: number;
  mark_price: number | null;
  notional_usdt: number | null;
  unrealized_pnl_usdt: null;
}

export interface SandboxPositions {
  source: "okx_demo" | "binance_demo";
  positions: SandboxPosition[];
  checked_at: string;
}

export interface CreateSandboxOrderRequest {
  credential_id: string;
  symbol: string;
  side: SandboxOrderSide;
  type: SandboxOrderType;
  quote_amount: number;
  price?: number;
  idempotency_key: string;
}

export interface SandboxOrder {
  id: string;
  credential_id: string;
  provider: SandboxExchangeProvider;
  client_order_id: string;
  symbol: string;
  side: SandboxOrderSide;
  type: SandboxOrderType;
  requested_quote: string | number;
  requested_quantity?: string | number | null;
  status: string;
  exchange_order_id?: string | null;
  sandbox: true;
  error_code?: string | null;
  created_at: string;
  updated_at: string;
}
