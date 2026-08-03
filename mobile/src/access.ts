import type { SaaSAccess, TenantRole } from "./types";

export type MobilePermission =
  | "read"
  | "strategy.manage"
  | "connection.manage"
  | "trade.execute"
  | "member.manage"
  | "billing.manage";

export type AccessGate = {
  readable: boolean;
  roleAllowed: boolean;
  mutationAllowed: boolean;
  message: string | null;
};

const MUTATION_ROLES: Readonly<Record<Exclude<MobilePermission, "read">, readonly TenantRole[]>> = {
  "strategy.manage": ["owner", "admin", "strategist"],
  "connection.manage": ["owner", "admin", "trader"],
  "trade.execute": ["owner", "admin", "trader"],
  "member.manage": ["owner", "admin"],
  "billing.manage": ["owner", "admin", "billing_manager"],
};

/** This is the server's current active-tenant denial detail. */
export const INACTIVE_ACCESS_MESSAGE = "工作区尚未开通或服务已到期";

type AccessWithMessage = SaaSAccess & {
  access_message?: string | null;
  message?: string | null;
};

function serverAccessMessage(access: SaaSAccess): string {
  const candidate = access as AccessWithMessage;
  return candidate.access_message ?? candidate.message ?? INACTIVE_ACCESS_MESSAGE;
}

export function hasDisplayPermission(
  access: SaaSAccess | null | undefined,
  permission: MobilePermission,
): boolean {
  if (!access) return false;
  if (permission === "read") return true;
  return MUTATION_ROLES[permission].includes(access.role);
}

export function canMutate(
  access: SaaSAccess | null | undefined,
  permission: Exclude<MobilePermission, "read">,
): boolean {
  return access?.status === "active" && hasDisplayPermission(access, permission);
}

export function accessGate(
  access: SaaSAccess | null | undefined,
  permission: MobilePermission,
): AccessGate {
  const readable = Boolean(access);
  const roleAllowed = hasDisplayPermission(access, permission);
  const mutationAllowed = permission === "read"
    ? false
    : canMutate(access, permission);
  const message = access && access.status !== "active" ? serverAccessMessage(access) : null;
  return { readable, roleAllowed, mutationAllowed, message };
}

export function isPlatformAdmin(access: SaaSAccess | null | undefined): boolean {
  return access?.is_platform_admin === true;
}

export function platformAdminGate(access: SaaSAccess | null | undefined): AccessGate {
  const readable = isPlatformAdmin(access);
  const mutationAllowed = readable && access?.status === "active";
  const message = access && access.status !== "active" ? serverAccessMessage(access) : null;
  return { readable, roleAllowed: readable, mutationAllowed, message };
}
