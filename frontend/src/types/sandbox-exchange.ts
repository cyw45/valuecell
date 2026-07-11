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
  total: string | number;
}

export interface SandboxConnectionBalance {
  balances: SandboxBalance[];
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
