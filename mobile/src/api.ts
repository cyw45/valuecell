import * as SecureStore from "expo-secure-store";
import { Platform } from "react-native";
import type {
  ApiEnvelope,
  MarketResponse,
  SaaSAccess,
  DemoConnection,
  Session,
  Strategy,
  Workspace,
} from "./types";

const ACCESS_TOKEN_KEY = "valuecell.mobile.access-token";
const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL ?? "https://vc.zhiweionline.com/api/v1";

async function readStoredSession(): Promise<string | null> {
  if (Platform.OS === "web") return globalThis.localStorage?.getItem(ACCESS_TOKEN_KEY) ?? null;
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

class MobileApiClient {
  private accessToken = "";

  setAccessToken(accessToken: string) {
    this.accessToken = accessToken;
  }

  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Content-Type", "application/json");
    if (this.accessToken) headers.set("Authorization", `Bearer ${this.accessToken}`);

    const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
    const body = (await response.json().catch(() => null)) as ApiEnvelope<T> | { detail?: unknown } | null;
    if (!response.ok) {
      const detail = body && "detail" in body ? body.detail : response.statusText;
      throw new Error(typeof detail === "string" ? detail : "请求失败，请稍后重试。");
    }
    if (!body || !("data" in body)) throw new Error("服务端返回格式无效。");
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
    return this.request<Strategy>(`/rule-strategies/${strategyId}/start`, { method: "POST" });
  }

  stopStrategy(strategyId: string) {
    return this.request<Strategy>(`/rule-strategies/${strategyId}/stop`, { method: "POST" });
  }

  updateStrategy(strategyId: string, request: Pick<Strategy, "name" | "description" | "config">) {
    return this.request<Strategy>(`/rule-strategies/${strategyId}`, {
      method: "PATCH",
      body: JSON.stringify(request),
    });
  }

  market(symbol: string, interval: string, lookback: number) {
    const params = new URLSearchParams({ symbols: symbol, interval, lookback: String(lookback) });
    return this.request<MarketResponse>(`/crypto-market/indicators?${params.toString()}`);
  }

  createStrategy(request: { name: string; description?: string; initial_capital_quote: number; config: Record<string, unknown> }) {
    return this.request<Strategy>("/rule-strategies", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  demoConnections() {
    return this.request<DemoConnection[]>("/saas/sandbox-exchanges/connections");
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
