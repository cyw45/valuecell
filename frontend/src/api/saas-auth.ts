import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import type { ApiResponse } from "@/lib/api-client";
import type {
  SaaSAuthResponse,
  SaaSLoginRequest,
  SaaSMeResponse,
  SaaSRegisterRequest,
  SaaSWorkspace,
} from "@/types/saas-auth";

export function useRegister() {
  return useMutation({
    mutationFn: (request: SaaSRegisterRequest) =>
      apiClient.post<ApiResponse<SaaSAuthResponse>>(
        "/saas/auth/register",
        request,
      ),
  });
}

export function useLogin() {
  return useMutation({
    mutationFn: (request: SaaSLoginRequest) =>
      apiClient.post<ApiResponse<SaaSAuthResponse>>(
        "/saas/auth/login",
        request,
      ),
  });
}

export function useMe() {
  return useMutation({
    mutationFn: () =>
      apiClient.get<ApiResponse<SaaSMeResponse>>("/saas/auth/me", {
        requiresAuth: true,
      }),
  });
}

export function useWorkspaces(enabled: boolean) {
  return useQuery({
    queryKey: ["saas", "workspaces"],
    enabled,
    queryFn: async () =>
      (
        await apiClient.get<ApiResponse<SaaSWorkspace[]>>(
          "/saas/auth/workspaces",
          { requiresAuth: true },
        )
      ).data,
  });
}

export function useSwitchWorkspace() {
  return useMutation({
    mutationFn: (tenantId: string) =>
      apiClient.post<ApiResponse<SaaSAuthResponse>>(
        "/saas/auth/switch",
        { tenant_id: tenantId },
        { requiresAuth: true },
      ),
  });
}
