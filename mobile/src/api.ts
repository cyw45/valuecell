import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import type {
  ApiEnvelope,
  DemoConnection,
  MarketResponse,
  SaaSAccess,
  Session,
  Strategy,
  Workspace,
} from "./types";

const ACCESS_TOKEN_KEY = "valuecell.mobile.access-token";
const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL ?? "https://vc.zhiweionline.com/api/v1";

export class MobileApiError extends Error {
  constructor(
    message: string,
    readonly endpoint: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "MobileApiError";
  }
}

async function readStoredSession(): Promise<string | null> {
  if (Platform.OS === "web") {
    return globalThis.localStorage?.getItem(ACCESS_TOKEN_KEY) ?? null;
  }
  return SecureStore.getItemAsync(ACCESS_TOKEN_KEY);
}

async function saveStoredSession(value: string): Promise<void> {
  if (Platform.OS === "web") {
    globalThis.localStorage?.setItem(ACCESS_TOKEN_KEY, value);
    return;
  }
  await SecureStore.setItemAsync(ACCESS_TOKEN_KEY, value);
}

async function removeStoredSession(): Promise<void> {
  if (Platform.OS === "web") {
    globalThis.localStorage?.removeItem(ACCESS_TOKEN_KEY);
    return;
  }
  await SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY);
}

function errorDetail(body: unknown): string {
  if (!body || typeof body !== "object" || !("detail" in body)) {
    return "服务返回了无法识别的错误。";
  }
  const detail = body.detail;
  if (typeof detail === "string") return detail;
  return "服务拒绝了本次请求，请检查账户权限或请求参数。";
}

class MobileApiClient {
  private accessToken = "";

  setAccessToken(accessToken: string) {
    this.accessToken = accessToken;
  }

  baseUrl() {
    return API_BASE_URL;
  }

  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    headers.set("Content-Type", "application/json");
    if (this.accessToken) headers.set("Authorization", `Bearer ${this.accessToken}`);

    let response: Response;
    try {
      response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
    } catch {
      throw new MobileApiError(
        `无法连接服务：${API_BASE_URL}。请确认手机可访问该 HTTPS 地址。`,
        path,
      );
    }

    const body = (await response.json().catch(() => null)) as
      | ApiEnvelope<T>
      | { detail?: unknown }
      | null;
    if (!response.ok) {
      throw new MobileApiError(
        `${response.status}：${errorDetail(body)}`,
        path,
        response.status,
      );
    }
    if (!body || !("data" in body)) {
      throw new MobileApiError("服务返回格式无效。", path, response.status);
    }
    return body.data;
  }

  login(email: string, password: string) {
    return this.request<{
      access_token: string;
      user_id: string;
      tenant_id: string;
      email: string;
    }>("/saas/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  }

  access() {
    return this.request<SaaSAccess>("/saas/access");
  }

  workspaces() {
    return this.request<Workspace[]>("/saas/auth/workspaces");
  }

  switchWorkspace(tenantId: string) {
    return this.request<{
      access_token: string;
      user_id: string;
      tenant_id: string;
      email: string;
    }>("/saas/auth/switch", {
      method: "POST",
      body: JSON.stringify({ tenant_id: tenantId }),
    });
  }

  strategies() {
    return this.request<Strategy[]>("/rule-strategies");
  }

  strategy(strategyId: string) {
    return this.request<Strategy>(`/rule-strategies/${strategyId}`);
  }

  startStrategy(strategyId: string) {
    return this.request<Strategy>(`/rule-strategies/${strategyId}/start`, {
      method: "POST",
    });
  }

  stopStrategy(strategyId: string) {
    return this.request<Strategy>(`/rule-strategies/${strategyId}/stop`, {
      method: "POST",
    });
  }

  updateStrategy(
    strategyId: string,
    request: Pick<Strategy, "name" | "description" | "config">,
  ) {
    return this.request<Strategy>(`/rule-strategies/${strategyId}`, {
      method: "PATCH",
      body: JSON.stringify(request),
    });
  }

  createStrategy(request: {
    name: string;
    description?: string;
    initial_capital_quote: number;
    config: Record<string, unknown>;
  }) {
    return this.request<Strategy>("/rule-strategies", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  demoConnections() {
    return this.request<DemoConnection[]>("/saas/sandbox-exchanges/connections");
  }

  market(symbol: string, interval: string, lookback: number) {
    const params = new URLSearchParams({
      symbols: symbol,
      interval,
      lookback: String(lookback),
    });
    return this.request<MarketResponse>(
      `/crypto-market/indicators?${params.toString()}`,
    );
  }
}

export const api = new MobileApiClient();

export async function loadSession(): Promise<Session | null> {
  const raw = await readStoredSession();
  if (!raw) return null;
  try {
    const session = JSON.parse(raw) as Session;
    api.setAccessToken(session.accessToken);
    return session;
  } catch {
    await removeStoredSession();
    return null;
  }
}

export async function persistSession(session: Session): Promise<void> {
  api.setAccessToken(session.accessToken);
  await saveStoredSession(JSON.stringify(session));
}

export async function clearSession(): Promise<void> {
  api.setAccessToken("");
  await removeStoredSession();
}
