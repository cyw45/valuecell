import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ApiResponse, apiClient } from "@/lib/api-client";
import type { LiveOrder } from "@/types/live-execution";

export interface LivePosition {
  symbol: string | null;
  contracts: number | string | null;
  notional: number | string | null;
  entry_price: number | string | null;
  mark_price: number | string | null;
  side: string | null;
}

const basePath = "/saas/live-execution";
const orderKey = ["live-execution", "orders"] as const;

export function useLiveOrders(connectionId?: string) {
  return useQuery({
    queryKey: [...orderKey, connectionId ?? ""] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<LiveOrder[]>>(
        `${basePath}/orders${connectionId ? `?connection_id=${connectionId}` : ""}`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
  });
}

export function useRefreshLiveOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (orderId: string) =>
      apiClient.post<ApiResponse<LiveOrder>>(
        `${basePath}/orders/${orderId}/refresh`,
        undefined,
        { requiresAuth: true },
      ),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: orderKey }),
  });
}

export function useLivePositions(connectionId?: string) {
  return useQuery({
    queryKey: ["live-execution", "positions", connectionId ?? ""] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<LivePosition[]>>(
        `${basePath}/connections/${connectionId}/positions`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(connectionId),
  });
}
