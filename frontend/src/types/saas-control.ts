export type TenantRole =
  | "owner"
  | "admin"
  | "strategist"
  | "trader"
  | "viewer"
  | "billing_manager";

export interface SaaSAccess {
  role: TenantRole;
  tenant_type: "personal" | "enterprise";
  organization_name: string | null;
  is_platform_admin: boolean;
  status: "active" | "pending_activation";
  commercial_model: "subscription" | "revenue_share" | null;
  expires_at: string | null;
}

export interface WorkspaceMember {
  user_id: string;
  email: string;
  role: TenantRole;
  created_at: string;
}

export interface AuditEvent {
  id: string;
  tenant_id: string | null;
  actor_user_id: string | null;
  action: string;
  target_type: string;
  target_id: string;
  outcome: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface Subscription {
  id: string;
  tenant_id: string;
  plan_id: string;
  status: string;
  starts_at: string;
  ends_at: string;
  note: string | null;
}

export interface EnterpriseAgreement {
  id: string;
  tenant_id: string;
  agreement_number: string;
  status: string;
  revenue_share_rate: string;
  settlement_cycle_days: number;
  high_water_mark_quote: string;
  starts_at: string;
  ends_at: string | null;
}

export interface ProfitSettlement {
  id: string;
  connection_id: string;
  period_started_at: string;
  period_ended_at: string;
  ending_equity_quote: string;
  eligible_profit_quote: string;
  revenue_share_rate: string;
  amount_due_quote: string;
  status: string;
}

export interface TenantBilling {
  access: SaaSAccess & { entitlements: Record<string, unknown> };
  subscriptions: Subscription[];
  agreement: EnterpriseAgreement | null;
  settlements: ProfitSettlement[];
}

export interface ServicePlan {
  id: string;
  code: string;
  name: string;
  duration_days: number;
  price_cents: number;
  currency: string;
  entitlements: Record<string, number | boolean>;
  active: string;
}

export interface AdminTenant {
  id: string;
  name: string;
  tenant_type: "personal" | "enterprise";
  organization_name: string | null;
  created_at: string;
  access: SaaSAccess & { entitlements: Record<string, unknown> };
}
