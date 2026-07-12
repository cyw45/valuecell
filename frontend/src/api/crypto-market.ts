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

interface CryptoMarketIndicatorQueryParams {
  symbols: string[];
  interval: string;
  lookback?: number;
  providers?: string[];
  enabled?: boolean;
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
  const refetchInterval = getMarketDataRefreshIntervalMs(
    marketDataRefreshMode,
    params.refreshIntervalSeconds,
  );

  return useQuery({
    queryKey: API_QUERY_KEYS.CRYPTO_MARKET.indicators([
      symbols.join(","),
      params.interval,
      params.lookback ?? 240,
      providers.join(","),
    ]),
    queryFn: () => {
      const searchParams = new URLSearchParams({
        symbols: symbols.join(","),
        interval: params.interval,
        lookback: String(params.lookback ?? 240),
      });
      if (providers.length > 0) {
        searchParams.set("providers", providers.join(","));
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
