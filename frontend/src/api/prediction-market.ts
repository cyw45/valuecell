import { useMutation, useQuery } from "@tanstack/react-query";
import { API_QUERY_KEYS } from "@/constants/api";
import { type ApiResponse, apiClient } from "@/lib/api-client";
import type {
  PredictionMarketCatalog,
  PredictionMarketSnapshot,
  PredictionReplayRequest,
  PredictionReplayResult,
} from "@/types/prediction-market";

export const usePredictionMarketCatalog = (limit = 50) =>
  useQuery({
    queryKey: API_QUERY_KEYS.PREDICTION_MARKET.catalog([limit]),
    queryFn: () =>
      apiClient.get<ApiResponse<PredictionMarketCatalog>>(
        `prediction-markets/catalog?limit=${limit}`,
      ),
    select: (response) => response.data,
    staleTime: 15_000,
    refetchInterval: 30_000,
  });

export const usePredictionMarketSnapshot = (
  marketId?: string,
  outcome?: string,
) =>
  useQuery({
    queryKey: API_QUERY_KEYS.PREDICTION_MARKET.snapshot([
      marketId ?? "",
      outcome ?? "",
    ]),
    queryFn: () => {
      const params = new URLSearchParams({ outcome: outcome ?? "" });
      return apiClient.get<ApiResponse<PredictionMarketSnapshot>>(
        `prediction-markets/markets/${encodeURIComponent(marketId ?? "")}?${params.toString()}`,
      );
    },
    select: (response) => response.data,
    enabled: Boolean(marketId && outcome),
    staleTime: 5_000,
    refetchInterval: 15_000,
  });

export const usePredictionMarketSignal = (
  marketId?: string,
  outcome?: string,
  history: string[] = [],
) =>
  useQuery({
    queryKey: API_QUERY_KEYS.PREDICTION_MARKET.signal([
      marketId ?? "",
      outcome ?? "",
      history.join(","),
    ]),
    queryFn: () => {
      const params = new URLSearchParams({
        outcome: outcome ?? "",
        history: history.join(","),
      });
      return apiClient.get<ApiResponse<PredictionMarketSnapshot>>(
        `prediction-markets/markets/${encodeURIComponent(marketId ?? "")}/signal?${params.toString()}`,
      );
    },
    select: (response) => response.data,
    enabled: Boolean(marketId && outcome),
    staleTime: 10_000,
    refetchInterval: 30_000,
  });

export const usePredictionReplayPreview = () =>
  useMutation({
    mutationFn: async (request: PredictionReplayRequest) => {
      const response = await apiClient.post<
        ApiResponse<PredictionReplayResult>
      >("prediction-markets/replay/preview", request);
      return response.data;
    },
  });
