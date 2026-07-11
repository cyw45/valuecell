import { useQuery } from "@tanstack/react-query";
import { API_QUERY_KEYS } from "@/constants/api";
import { type ApiResponse, apiClient } from "@/lib/api-client";
import type {
  CryptoMarketIndicators,
  CryptoSymbolCatalog,
} from "@/types/crypto-market";

export const useGetCryptoSymbols = () =>
  useQuery({
    queryKey: API_QUERY_KEYS.CRYPTO_MARKET.symbols,
    queryFn: () =>
      apiClient.get<ApiResponse<CryptoSymbolCatalog>>("crypto-market/symbols"),
    select: (data) => data.data,
    staleTime: 10 * 60 * 1000,
  });

export const useGetCryptoMarketIndicators = (params: {
  symbols: string[];
  interval: string;
  lookback?: number;
  providers?: string[];
  enabled?: boolean;
}) => {
  const symbols = params.symbols.filter(Boolean);
  const providers = params.providers?.filter(Boolean) ?? [];
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
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
};
