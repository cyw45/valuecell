import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, type ApiResponse } from "@/lib/api-client";
import { useSaaSSession } from "@/store/system-store";
import type {
  AdminTenant,
  AuditEvent,
  SaaSAccess,
  ServicePlan,
  TenantBilling,
  TenantRole,
  WorkspaceMember,
} from "@/types/saas-control";

const ACCESS_KEY = ["saas", "access"] as const;
const MEMBERS_KEY = ["saas", "workspace-members"] as const;
const BILLING_KEY = ["saas", "billing"] as const;
const AUDIT_KEY = ["saas", "audit"] as const;
const ADMIN_TENANTS_KEY = ["saas", "admin", "tenants"] as const;
const ADMIN_PLANS_KEY = ["saas", "admin", "plans"] as const;
const ADMIN_AUDIT_KEY = ["saas", "admin", "audit"] as const;

export function useSaaSAccess() {
  const { isLoggedIn, tenantId } = useSaaSSession();
  return useQuery({
    queryKey: [...ACCESS_KEY, tenantId],
    enabled: isLoggedIn,
    queryFn: async () =>
      (
        await apiClient.get<ApiResponse<SaaSAccess>>("/saas/access", {
          requiresAuth: true,
        })
      ).data,
  });
}

export function useWorkspaceMembers(enabled = true) {
  return useQuery({
    queryKey: MEMBERS_KEY,
    enabled,
    queryFn: async () =>
      (
        await apiClient.get<ApiResponse<WorkspaceMember[]>>(
          "/saas/workspace/members",
          { requiresAuth: true },
        )
      ).data,
  });
}

export function useSaveWorkspaceMember() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: { email: string; role: TenantRole }) =>
      apiClient.post<ApiResponse<WorkspaceMember>>(
        "/saas/workspace/members",
        request,
        { requiresAuth: true },
      ),
    onSuccess: async () =>
      queryClient.invalidateQueries({ queryKey: MEMBERS_KEY }),
  });
}

export function useTenantBilling() {
  return useQuery({
    queryKey: BILLING_KEY,
    queryFn: async () =>
      (
        await apiClient.get<ApiResponse<TenantBilling>>("/saas/billing", {
          requiresAuth: true,
        })
      ).data,
  });
}

export function useTenantAudit(enabled = true) {
  return useQuery({
    queryKey: AUDIT_KEY,
    enabled,
    queryFn: async () =>
      (
        await apiClient.get<ApiResponse<AuditEvent[]>>("/saas/audit", {
          requiresAuth: true,
        })
      ).data,
  });
}

export function useAdminTenants(enabled = true) {
  return useQuery({
    queryKey: ADMIN_TENANTS_KEY,
    enabled,
    queryFn: async () =>
      (
        await apiClient.get<ApiResponse<AdminTenant[]>>("/saas/admin/tenants", {
          requiresAuth: true,
        })
      ).data,
  });
}

export function useAdminPlans(enabled = true) {
  return useQuery({
    queryKey: ADMIN_PLANS_KEY,
    enabled,
    queryFn: async () =>
      (
        await apiClient.get<ApiResponse<ServicePlan[]>>("/saas/admin/plans", {
          requiresAuth: true,
        })
      ).data,
  });
}

export function useCreatePlan() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: {
      code: string;
      name: string;
      duration_days: number;
      price_cents: number;
      currency?: string;
    }) =>
      apiClient.post<ApiResponse<ServicePlan>>("/saas/admin/plans", request, {
        requiresAuth: true,
      }),
    onSuccess: async () =>
      queryClient.invalidateQueries({ queryKey: ADMIN_PLANS_KEY }),
  });
}

export function useGrantSubscription() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: {
      tenant_id: string;
      plan_id: string;
      ends_at: string;
      note?: string;
    }) =>
      apiClient.post<ApiResponse<unknown>>(
        "/saas/admin/subscriptions",
        request,
        { requiresAuth: true },
      ),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ADMIN_TENANTS_KEY }),
        queryClient.invalidateQueries({ queryKey: ACCESS_KEY }),
      ]);
    },
  });
}

export function useCreateEnterpriseAgreement() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: {
      tenant_id: string;
      agreement_number: string;
      revenue_share_rate: string;
      settlement_cycle_days: number;
      starts_at: string;
    }) =>
      apiClient.post<ApiResponse<unknown>>("/saas/admin/agreements", request, {
        requiresAuth: true,
      }),
    onSuccess: async () =>
      queryClient.invalidateQueries({ queryKey: ADMIN_TENANTS_KEY }),
  });
}

export function useUpdateTenantProfile() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: {
      tenant_id: string;
      tenant_type: "personal" | "enterprise";
      organization_name?: string;
    }) =>
      apiClient.patch<ApiResponse<unknown>>(
        `/saas/admin/tenants/${request.tenant_id}/profile`,
        {
          tenant_type: request.tenant_type,
          organization_name: request.organization_name,
        },
        { requiresAuth: true },
      ),
    onSuccess: async () =>
      queryClient.invalidateQueries({ queryKey: ADMIN_TENANTS_KEY }),
  });
}
export function useAdminAudit(enabled = true) {
  return useQuery({
    queryKey: ADMIN_AUDIT_KEY,
    enabled,
    queryFn: async () =>
      (
        await apiClient.get<ApiResponse<AuditEvent[]>>("/saas/admin/audit", {
          requiresAuth: true,
        })
      ).data,
  });
}
