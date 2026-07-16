import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ApiResponse, apiClient } from "@/lib/api-client";
import type {
  ConfirmStartupAuthorizationRequest,
  CreateLiveConnectionRequest,
  CreateLiveOrderRequest,
  CreateLiveStrategyBindingRequest,
  LiveConnection,
  LiveExecutionStatus,
  LiveOrder,
  LiveRiskPolicy,
  LiveStrategyBinding,
  StartupAuthorizationChallenge,
} from "@/types/live-execution";

const basePath = "/saas/live-execution";
const liveExecutionKeys = {
  status: ["live-execution", "status"] as const,
  connections: ["live-execution", "connections"] as const,
  riskPolicy: ["live-execution", "risk-policy"] as const,
  bindings: ["live-execution", "bindings"] as const,
};

export const useLiveExecutionStatus = () =>
  useQuery({
    queryKey: liveExecutionKeys.status,
    queryFn: () =>
      apiClient.get<ApiResponse<LiveExecutionStatus>>(`${basePath}/status`, {
        requiresAuth: true,
      }),
    select: (response) => response.data,
    refetchInterval: 15_000,
  });

export const useLiveConnections = () =>
  useQuery({
    queryKey: liveExecutionKeys.connections,
    queryFn: () =>
      apiClient.get<ApiResponse<LiveConnection[]>>(`${basePath}/connections`, {
        requiresAuth: true,
      }),
    select: (response) => response.data,
  });

export const useCreateLiveConnection = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateLiveConnectionRequest) =>
      apiClient.post<ApiResponse<LiveConnection>>(
        `${basePath}/connections`,
        request,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: liveExecutionKeys.connections,
      }),
  });
};

export const useLiveRiskPolicy = () =>
  useQuery({
    queryKey: liveExecutionKeys.riskPolicy,
    queryFn: () =>
      apiClient.get<ApiResponse<LiveRiskPolicy | null>>(
        `${basePath}/risk-policies`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
  });

export const useSaveLiveRiskPolicy = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: LiveRiskPolicy) =>
      apiClient.post<ApiResponse<LiveRiskPolicy>>(
        `${basePath}/risk-policies`,
        request,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: liveExecutionKeys.riskPolicy }),
  });
};

export const useLiveStrategyBindings = () =>
  useQuery({
    queryKey: liveExecutionKeys.bindings,
    queryFn: () =>
      apiClient.get<ApiResponse<LiveStrategyBinding[]>>(
        `${basePath}/bindings`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
  });

export const useCreateLiveStrategyBinding = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateLiveStrategyBindingRequest) =>
      apiClient.post<ApiResponse<LiveStrategyBinding>>(
        `${basePath}/bindings`,
        request,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: liveExecutionKeys.bindings }),
  });
};

export const useRevokeLiveStrategyBinding = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (bindingId: string) =>
      apiClient.post<ApiResponse<LiveStrategyBinding>>(
        `${basePath}/bindings/${bindingId}/revoke`,
        undefined,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: liveExecutionKeys.bindings }),
  });
};

export const useRequestStartupAuthorizationChallenge = () =>
  useMutation({
    mutationFn: () =>
      apiClient.post<ApiResponse<StartupAuthorizationChallenge>>(
        `${basePath}/startup-authorization/challenge`,
        {},
        { requiresAuth: true },
      ),
  });

export const useConfirmStartupAuthorization = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: ConfirmStartupAuthorizationRequest) =>
      apiClient.post<ApiResponse<LiveExecutionStatus>>(
        `${basePath}/startup-authorization/confirm`,
        request,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: liveExecutionKeys.status }),
  });
};

export const useRevokeStartupAuthorization = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiClient.post<ApiResponse<LiveExecutionStatus>>(
        `${basePath}/startup-authorization/revoke`,
        undefined,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: liveExecutionKeys.status }),
  });
};

export const useCreateLiveOrder = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateLiveOrderRequest) =>
      apiClient.post<ApiResponse<LiveOrder>>(`${basePath}/orders`, request, {
        requiresAuth: true,
        headers: { "Idempotency-Key": request.idempotency_key },
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: liveExecutionKeys.status }),
  });
};
