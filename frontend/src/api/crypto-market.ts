import { useQuery } from "@tanstack/react-query";
import { API_QUERY_KEYS } from "@/constants/api";
import { type ApiResponse, apiClient } from "@/lib/api-client";
import {
  getMarketDataRefreshIntervalMs,
  useMarketDataRefreshMode,
} from "@/store/settings-store";
import type {
  CryptoMarketIndicators,
  CryptoSymbolCatalog,
} from "@/types/crypto-market";

const DEFAULT_SNAPSHOT_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"];

interface CryptoMarketIndicatorQueryParams {
  symbols: string[];
  interval: string;
  lookback?: number;
  providers?: string[];
  enabled?: boolean;
  fromTsMs?: number;
  toTsMs?: number;
  refreshIntervalSeconds?: number;
}

export const useGetCryptoSymbols = () =>
  useQuery({
    queryKey: API_QUERY_KEYS.CRYPTO_MARKET.symbols,
    queryFn: () =>
      apiClient.get<ApiResponse<CryptoSymbolCatalog>>("crypto-market/symbols"),
    select: (data) => data.data,
    staleTime: 10 * 60 * 1000,
  });

export const useGetCryptoMarketIndicators = (
  params: CryptoMarketIndicatorQueryParams,
) => {
  const marketDataRefreshMode = useMarketDataRefreshMode();
  const symbols = params.symbols.filter(Boolean);
  const providers = params.providers?.filter(Boolean) ?? [];
  const usesDefaultSnapshot =
    providers.length === 0
    && params.interval === "1h"
    && (params.lookback ?? 240) <= 240
    && symbols.every((symbol) => DEFAULT_SNAPSHOT_SYMBOLS.includes(symbol));
  const querySymbols = usesDefaultSnapshot ? DEFAULT_SNAPSHOT_SYMBOLS : symbols;
  const refetchInterval = getMarketDataRefreshIntervalMs(
    marketDataRefreshMode,
    params.refreshIntervalSeconds,
  );

  return useQuery({
    queryKey: API_QUERY_KEYS.CRYPTO_MARKET.indicators([
      querySymbols.join(","),
      params.interval,
      params.lookback ?? 240,
      providers.join(","),
      params.fromTsMs ?? "",
      params.toTsMs ?? "",
    ]),
    queryFn: () => {
      const searchParams = new URLSearchParams({
        symbols: querySymbols.join(","),
        interval: params.interval,
        lookback: String(params.lookback ?? 240),
      });
      if (providers.length > 0) {
        searchParams.set("providers", providers.join(","));
      }
      if (params.fromTsMs !== undefined) {
        searchParams.set("from_ts_ms", String(params.fromTsMs));
      }
      if (params.toTsMs !== undefined) {
        searchParams.set("to_ts_ms", String(params.toTsMs));
      }
      return apiClient.get<ApiResponse<CryptoMarketIndicators>>(
        `crypto-market/indicators?${searchParams.toString()}`,
      );
    },
    select: (data) => data.data,
    enabled: (params.enabled ?? true) && symbols.length > 0,
    refetchInterval,
    staleTime: 10_000,
  });
};

