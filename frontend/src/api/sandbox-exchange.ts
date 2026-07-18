import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ApiResponse, apiClient } from "@/lib/api-client";
import type {
  CreateSandboxConnectionRequest,
  CreateSandboxOrderRequest,
  SandboxConnection,
  SandboxConnectionBalance,
  SandboxOrder,
  SandboxPositions,
} from "@/types/sandbox-exchange";

const sandboxExchangeKeys = {
  connections: ["sandbox-exchanges", "connections"] as const,
  balance: (connectionId: string) =>
    ["sandbox-exchanges", "connections", connectionId, "balance"] as const,
  positions: (connectionId: string) =>
    ["sandbox-exchanges", "connections", connectionId, "positions"] as const,
  orders: (connectionId?: string) =>
    ["sandbox-exchanges", "orders", connectionId ?? "all"] as const,
};

const basePath = "/saas/sandbox-exchanges";

export const useSandboxConnections = () =>
  useQuery({
    queryKey: sandboxExchangeKeys.connections,
    queryFn: () =>
      apiClient.get<ApiResponse<SandboxConnection[]>>(
        `${basePath}/connections`,
        {
          requiresAuth: true,
        },
      ),
    select: (response) => response.data,
  });

export const useCreateSandboxConnection = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateSandboxConnectionRequest) =>
      apiClient.post<ApiResponse<SandboxConnection>>(
        `${basePath}/connections`,
        request,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: sandboxExchangeKeys.connections,
      }),
  });
};

export const useSandboxBalance = (connectionId?: string, enabled = false) =>
  useQuery({
    queryKey: sandboxExchangeKeys.balance(connectionId ?? ""),
    queryFn: () =>
      apiClient.get<ApiResponse<SandboxConnectionBalance>>(
        `${basePath}/connections/${connectionId}/balance`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(connectionId) && enabled,
    refetchInterval: enabled ? 30_000 : false,
  });

export const useSandboxPositions = (connectionId?: string, enabled = false) =>
  useQuery({
    queryKey: sandboxExchangeKeys.positions(connectionId ?? ""),
    queryFn: () =>
      apiClient.get<ApiResponse<SandboxPositions>>(
        `${basePath}/connections/${connectionId}/positions`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(connectionId) && enabled,
    refetchInterval: enabled ? 30_000 : false,
  });

export const useSandboxOrders = (connectionId?: string) =>
  useQuery({
    queryKey: sandboxExchangeKeys.orders(connectionId),
    queryFn: () => {
      const query = connectionId
        ? `?credential_id=${encodeURIComponent(connectionId)}`
        : "";
      return apiClient.get<ApiResponse<SandboxOrder[]>>(
        `${basePath}/orders${query}`,
        {
          requiresAuth: true,
        },
      );
    },
    select: (response) => response.data,
    enabled: Boolean(connectionId),
  });

export const useCreateSandboxOrder = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateSandboxOrderRequest) =>
      apiClient.post<ApiResponse<SandboxOrder>>(`${basePath}/orders`, request, {
        requiresAuth: true,
        headers: { "Idempotency-Key": request.idempotency_key },
      }),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: sandboxExchangeKeys.orders() });
      queryClient.invalidateQueries({
        queryKey: sandboxExchangeKeys.balance(response.data.credential_id),
      });
    },
  });
};
