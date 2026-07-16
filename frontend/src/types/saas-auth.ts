export type TenantType = "personal" | "enterprise";

export interface SaaSRegisterRequest {
  email: string;
  password: string;
  tenant_type: TenantType;
  workspace_name: string;
  organization_name?: string;
}

export interface SaaSLoginRequest {
  email: string;
  password: string;
}

export interface SaaSAuthResponse {
  access_token: string;
  user_id: string;
  tenant_id: string;
  email: string;
  tenant_type: TenantType;
  organization_name: string | null;
}

export interface SaaSWorkspace {
  tenant_id: string;
  name: string;
  tenant_type: TenantType;
  organization_name: string | null;
  role: string;
  selected: boolean;
}

export interface SaaSMeResponse {
  user_id: string;
  tenant_id: string;
  role: string;
  is_platform_admin: boolean;
  access_status: "active" | "pending_activation";
  commercial_model: "subscription" | "revenue_share" | null;
  access_expires_at: string | null;
}
