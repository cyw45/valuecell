export interface LeaderSpotV19RankingItem {
  symbol: string;
  rank: number;
  quote_volume_24h: number;
  listing_at: string | null;
  spot_tradable: boolean;
  quote_asset: "USDT";
  provider_payload: Record<string, unknown>;
}

export interface LeaderSpotV19RankingSnapshot {
  source: "okx";
  observed_at: string;
  expires_at: string;
  items: LeaderSpotV19RankingItem[];
  source_snapshot_id: string;
  completeness: "complete" | "partial" | "unsafe";
}

export interface LeaderSpotV19BookLevel {
  price: number;
  quantity: number;
}

export interface LeaderSpotV19OrderBookSnapshot {
  symbol: string;
  bids: LeaderSpotV19BookLevel[];
  asks: LeaderSpotV19BookLevel[];
  observed_at: string;
  source: "okx";
}

export interface LeaderSpotV19MarketInput {
  symbol: string;
  interval: "1m" | "5m" | "15m";
  source: "okx" | "market_service";
  candles: Array<Record<string, number>>;
  latest_price: number;
  order_book: LeaderSpotV19OrderBookSnapshot | null;
  observed_at: string;
  expires_at: string;
}
