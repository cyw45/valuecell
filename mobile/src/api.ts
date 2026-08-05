import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import { fetch as expoFetch } from "expo/fetch";
import type * as Types from "./types";

const ACCESS_TOKEN_KEY = "valuecell.mobile.access-token";
const REMEMBERED_EMAIL_KEY = "valuecell.mobile.remembered-email";
const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL ?? "https://vc.zhiweionline.com/api/v1";

type RequestOptions = {
  authenticated?: boolean;
};

export type StrategyExportFile = Readonly<{
  bytes: Uint8Array<ArrayBuffer>;
  filename: string;
  mimeType: string;
}>;

export type StrategyExportDateRange = Readonly<{
  fromDate?: string;
  toDate?: string;
}>;


type UnauthorizedHandler = () => Promise<void>;
type StrategyLogType = "signals" | "trades" | "funding";
type StrategyLogEntry<T extends StrategyLogType> = T extends "signals"
  ? Types.RuleStrategyLogEntry
  : T extends "trades"
    ? Types.RuleStrategyTradeLogEntry
    : Types.RuleStrategyFundingLogEntry;

type ParsedApiError = {
  message: string;
  code?: string;
};

export class MobileApiError extends Error {
  constructor(
    message: string,
    readonly endpoint: string,
    readonly status?: number,
    readonly code?: string,
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function stableErrorCode(detail: Record<string, unknown>): string | undefined {
  const code = detail.code;
  if (typeof code === "string" && code) return code;

  const errorCode = detail.error_code;
  if (typeof errorCode === "string" && errorCode) return errorCode;

  return undefined;
}

function errorDetail(body: unknown): ParsedApiError {
  if (!isRecord(body) || !("detail" in body)) {
    return { message: "服务返回了无法识别的错误。" };
  }

  const detail = body.detail;
  if (typeof detail === "string") return { message: detail };
  if (!isRecord(detail)) {
    return { message: "服务拒绝了本次请求，请检查账户权限或请求参数。" };
  }

  return {
    message:
      typeof detail.detail === "string"
        ? detail.detail
        : "服务拒绝了本次请求，请检查账户权限或请求参数。",
    code: stableErrorCode(detail),
  };
}

function withQuery(path: string, params: URLSearchParams): string {
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

function pathSegment(value: string): string {
  return encodeURIComponent(value);
}

function attachmentFilename(contentDisposition: string | null): string {
  const header = contentDisposition ?? "";
  const encoded = /filename\*=UTF-8''([^;]+)/i.exec(header)?.[1];
  const quoted = /filename="([^"]+)"/i.exec(header)?.[1];
  const unquoted = /filename=([^;\s]+)/i.exec(header)?.[1];
  const rawFilename = encoded ?? quoted ?? unquoted;
  if (!rawFilename) return "策略历史导出.xlsx";

  let decodedFilename = rawFilename;
  try {
    decodedFilename = decodeURIComponent(rawFilename);
  } catch {
    // Use the server-supplied filename verbatim when it is not URI encoded.
  }
  const safeFilename = decodedFilename
    .replace(/[\\/:*?"<>|\u0000-\u001F]/g, "_")
    .replace(/\s+/g, " ")
    .trim();
  if (!safeFilename) return "策略历史导出.xlsx";
  return /\.xlsx$/i.test(safeFilename) ? safeFilename : `${safeFilename}.xlsx`;
}

class MobileApiClient {
  private accessToken = "";
  private unauthorizedHandler: UnauthorizedHandler | null = null;

  setUnauthorizedHandler = (handler: UnauthorizedHandler | null): void => {
    this.unauthorizedHandler = handler;
  };

  setAccessToken = (accessToken: string): void => {
    this.accessToken = accessToken;
  };

  baseUrl = (): string => API_BASE_URL;

  private async request<T>(
    path: string,
    init: RequestInit = {},
    options: RequestOptions = {},
  ): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    headers.set("Content-Type", "application/json");
    if (options.authenticated && this.accessToken) {
      headers.set("Authorization", `Bearer ${this.accessToken}`);
    }

    let response: Response;
    try {
      response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
    } catch {
      throw new MobileApiError(
        `无法连接服务：${API_BASE_URL}。请确认手机可访问该 HTTPS 地址。`,
        path,
      );
    }

    const body = (await response.json().catch(() => null)) as unknown;
    if (!response.ok) {
      const detail = errorDetail(body);
      if (options.authenticated && response.status === 401) {
        await this.unauthorizedHandler?.();
      }
      throw new MobileApiError(detail.message, path, response.status, detail.code);
    }
    if (!isRecord(body) || !("data" in body)) {
      throw new MobileApiError("服务返回格式无效。", path, response.status);
    }

    return (body as unknown as Types.ApiEnvelope<T>).data;
  }

  private async authenticatedBinaryRequest(path: string): Promise<StrategyExportFile> {
    const headers = new Headers();
    headers.set("Accept", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
    if (this.accessToken) {
      headers.set("Authorization", `Bearer ${this.accessToken}`);
    }

    const response = await expoFetch(`${API_BASE_URL}${path}`, {
      credentials: "omit",
      headers,
      method: "GET",
    }).catch(() => {
      throw new MobileApiError(
        `无法连接服务：${API_BASE_URL}。请确认手机可访问该 HTTPS 地址。`,
        path,
      );
    });

    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as unknown;
      const detail = errorDetail(body);
      if (response.status === 401) {
        await this.unauthorizedHandler?.();
      }
      throw new MobileApiError(detail.message, path, response.status, detail.code);
    }

    return {
      bytes: new Uint8Array(await response.arrayBuffer()),
      filename: attachmentFilename(response.headers.get("content-disposition")),
      mimeType:
        response.headers.get("content-type") ??
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    };
  }

  private authenticatedRequest<T>(
    path: string,
    init: RequestInit = {},
  ): Promise<T> {
    return this.request<T>(path, init, { authenticated: true });
  }

  private publicRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
    return this.request<T>(path, init);
  }

  login = (email: string, password: string): Promise<Types.SaaSAuthResponse> =>
    this.publicRequest<Types.SaaSAuthResponse>("/saas/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password } satisfies Types.SaaSLoginRequest),
    });

  register = (
    request: Types.SaaSRegisterRequest,
  ): Promise<Types.SaaSAuthResponse> =>
    this.publicRequest<Types.SaaSAuthResponse>("/saas/auth/register", {
      method: "POST",
      body: JSON.stringify(request),
    });

  access = (): Promise<Types.SaaSAccess> =>
    this.authenticatedRequest<Types.SaaSAccess>("/saas/access");

  me = (): Promise<Types.SaaSMeResponse> =>
    this.authenticatedRequest<Types.SaaSMeResponse>("/saas/auth/me");

  workspaces = (): Promise<Types.Workspace[]> =>
    this.authenticatedRequest<Types.Workspace[]>("/saas/auth/workspaces");

  switchWorkspace = (tenantId: string): Promise<Types.SaaSAuthResponse> =>
    this.authenticatedRequest<Types.SaaSAuthResponse>("/saas/auth/switch", {
      method: "POST",
      body: JSON.stringify({ tenant_id: tenantId }),
    });

  workspaceMembers = (): Promise<Types.WorkspaceMember[]> =>
    this.authenticatedRequest<Types.WorkspaceMember[]>("/saas/workspace/members");

  saveWorkspaceMember = (
    request: Types.SaveWorkspaceMemberRequest,
  ): Promise<Types.WorkspaceMember> =>
    this.authenticatedRequest<Types.WorkspaceMember>("/saas/workspace/members", {
      method: "POST",
      body: JSON.stringify(request),
    });

  tenantBilling = (): Promise<Types.TenantBilling> =>
    this.authenticatedRequest<Types.TenantBilling>("/saas/billing");

  tenantAudit = (limit = 100): Promise<Types.AuditEvent[]> =>
    this.authenticatedRequest<Types.AuditEvent[]>(
      withQuery("/saas/audit", new URLSearchParams({ limit: String(limit) })),
    );

  strategies = (includeArchived?: unknown): Promise<Types.Strategy[]> => {
    const path =
      includeArchived === true
        ? "/rule-strategies?include_archived=true"
        : "/rule-strategies";
    return this.authenticatedRequest<Types.Strategy[]>(path);
  };

  strategy = (strategyId: string): Promise<Types.Strategy> =>
    this.authenticatedRequest<Types.Strategy>(
      `/rule-strategies/${pathSegment(strategyId)}`,
    );

  strategyExport = (
    strategyId: string,
    { fromDate, toDate }: StrategyExportDateRange = {},
  ): Promise<StrategyExportFile> => {
    const params = new URLSearchParams();
    if (fromDate) params.set("from_date", fromDate);
    if (toDate) params.set("to_date", toDate);
    return this.authenticatedBinaryRequest(
      withQuery(
        `/rule-strategies/${pathSegment(strategyId)}/export`,
        params,
      ),
    );
  };

  createStrategy = (
    request: Types.CreateRuleStrategyRequest,
  ): Promise<Types.Strategy> =>
    this.authenticatedRequest<Types.Strategy>("/rule-strategies", {
      method: "POST",
      body: JSON.stringify(request),
    });

  updateStrategy = (
    strategyId: string,
    request: Types.UpdateRuleStrategyRequest,
  ): Promise<Types.Strategy> =>
    this.authenticatedRequest<Types.Strategy>(
      `/rule-strategies/${pathSegment(strategyId)}`,
      {
        method: "PATCH",
        body: JSON.stringify(request),
      },
    );

  archiveStrategy = (strategyId: string): Promise<Types.Strategy> =>
    this.authenticatedRequest<Types.Strategy>(
      `/rule-strategies/${pathSegment(strategyId)}`,
      { method: "DELETE" },
    );

  startStrategy = (strategyId: string): Promise<Types.Strategy> =>
    this.authenticatedRequest<Types.Strategy>(
      `/rule-strategies/${pathSegment(strategyId)}/start`,
      { method: "POST" },
    );

  stopStrategy = (strategyId: string): Promise<Types.Strategy> =>
    this.authenticatedRequest<Types.Strategy>(
      `/rule-strategies/${pathSegment(strategyId)}/stop`,
      { method: "POST" },
    );

  evaluateStrategy = (
    strategyId: string,
    request: Types.EvaluateRuleStrategyRequest,
  ): Promise<Types.RuleStrategyEvaluation> =>
    this.authenticatedRequest<Types.RuleStrategyEvaluation>(
      `/rule-strategies/${pathSegment(strategyId)}/evaluate`,
      {
        method: "POST",
        body: JSON.stringify(request),
      },
    );

  strategyAccount = (
    strategyId: string,
  ): Promise<Types.RuleStrategyPaperAccount> =>
    this.authenticatedRequest<Types.RuleStrategyPaperAccount>(
      `/rule-strategies/${pathSegment(strategyId)}/account`,
    );

  strategyPnlCurve = (strategyId: string): Promise<Types.RuleStrategyPnlPoint[]> =>
    this.authenticatedRequest<Types.RuleStrategyPnlPoint[]>(
      `/rule-strategies/${pathSegment(strategyId)}/pnl-curve`,
    );

  strategyEvaluations = (
    strategyId: string,
    limit = 100,
  ): Promise<Types.RuleStrategyEvaluationHistoryEntry[]> =>
    this.authenticatedRequest<Types.RuleStrategyEvaluationHistoryEntry[]>(
      withQuery(
        `/rule-strategies/${pathSegment(strategyId)}/evaluations`,
        new URLSearchParams({ limit: String(limit) }),
      ),
    );

  strategyLog = <T extends StrategyLogType>(
    strategyId: string,
    logType: T,
    limit = 100,
  ): Promise<Types.RuleStrategyLog<StrategyLogEntry<T>>> =>
    this.authenticatedRequest<Types.RuleStrategyLog<StrategyLogEntry<T>>>(
      withQuery(
        `/rule-strategies/${pathSegment(strategyId)}/${logType}`,
        new URLSearchParams({ limit: String(limit) }),
      ),
    );

  strategyDemoExecution = (
    strategyId: string,
  ): Promise<Types.RuleStrategyDemoExecution> =>
    this.authenticatedRequest<Types.RuleStrategyDemoExecution>(
      `/rule-strategies/${pathSegment(strategyId)}/demo-execution`,
    );

  strategyAdvisory = (
    strategyId: string,
  ): Promise<Types.RuleStrategyAdvisory> =>
    this.authenticatedRequest<Types.RuleStrategyAdvisory>(
      `/rule-strategies/${pathSegment(strategyId)}/advisory-analysis`,
      { method: "POST" },
    );

  strategyMonitorState = (
    strategyId: string,
  ): Promise<Types.RuleStrategyMonitorState[]> =>
    this.authenticatedRequest<Types.RuleStrategyMonitorState[]>(
      `/rule-strategies/${pathSegment(strategyId)}/monitor-state`,
    );

  strategyRiskState = (
    strategyId: string,
  ): Promise<Types.RuleStrategyRiskState> =>
    this.authenticatedRequest<Types.RuleStrategyRiskState>(
      `/rule-strategies/${pathSegment(strategyId)}/risk-state`,
    );

  parseStrategyText = (
    strategyText: string,
  ): Promise<Types.RuleStrategyTextImportProposal> =>
    this.authenticatedRequest<Types.RuleStrategyTextImportProposal>(
      "/rule-strategies/parse-strategy-text",
      {
        method: "POST",
        body: JSON.stringify({ strategy_text: strategyText }),
      },
    );

  cryptoSymbols = (): Promise<Types.CryptoSymbolCatalog> =>
    this.publicRequest<Types.CryptoSymbolCatalog>("/crypto-market/symbols");

  market = (
    symbol: string,
    interval: string,
    lookback: number,
    options: Types.CryptoMarketQueryOptions = {},
  ): Promise<Types.CryptoMarketIndicators> => {
    const params = new URLSearchParams({
      symbols: symbol,
      interval,
      lookback: String(lookback),
    });
    if (options.providers?.length) {
      params.set("providers", options.providers.join(","));
    }
    if (options.from_ts_ms !== undefined) {
      params.set("from_ts_ms", String(options.from_ts_ms));
    }
    if (options.to_ts_ms !== undefined) {
      params.set("to_ts_ms", String(options.to_ts_ms));
    }
    return this.publicRequest<Types.CryptoMarketIndicators>(
      withQuery("/crypto-market/indicators", params),
    );
  };

  sandboxConnections = (): Promise<Types.SandboxConnection[]> =>
    this.authenticatedRequest<Types.SandboxConnection[]>(
      "/saas/sandbox-exchanges/connections",
    );

  demoConnections = (): Promise<Types.DemoConnection[]> =>
    this.authenticatedRequest<Types.DemoConnection[]>(
      "/saas/sandbox-exchanges/connections",
    );

  createSandboxConnection = (
    request: Types.CreateSandboxConnectionRequest,
  ): Promise<Types.SavedSandboxConnection> =>
    this.authenticatedRequest<Types.SavedSandboxConnection>(
      "/saas/sandbox-exchanges/connections",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
    );

  sandboxBalance = (
    connectionId: string,
  ): Promise<Types.SandboxConnectionBalance> =>
    this.authenticatedRequest<Types.SandboxConnectionBalance>(
      `/saas/sandbox-exchanges/connections/${pathSegment(connectionId)}/balance`,
    );

  sandboxPositions = (connectionId: string): Promise<Types.SandboxPositions> =>
    this.authenticatedRequest<Types.SandboxPositions>(
      `/saas/sandbox-exchanges/connections/${pathSegment(connectionId)}/positions`,
    );

  sandboxSymbols = (connectionId: string): Promise<Types.SandboxSymbol[]> =>
    this.authenticatedRequest<Types.SandboxSymbol[]>(
      `/saas/sandbox-exchanges/connections/${pathSegment(connectionId)}/symbols`,
    );

  sandboxOrders = (
    connectionId?: string,
    refresh = false,
  ): Promise<Types.SandboxOrder[]> => {
    const params = new URLSearchParams();
    if (connectionId) params.set("credential_id", connectionId);
    if (refresh) params.set("refresh", "true");
    return this.authenticatedRequest<Types.SandboxOrder[]>(
      withQuery("/saas/sandbox-exchanges/orders", params),
    );
  };

  sandboxOrderStatus = (orderId: string): Promise<Types.SandboxOrder> =>
    this.authenticatedRequest<Types.SandboxOrder>(
      `/saas/sandbox-exchanges/orders/${pathSegment(orderId)}/status`,
    );

  createSandboxOrder = (
    request: Types.CreateSandboxOrderSubmission,
    idempotencyKey: string,
  ): Promise<Types.SandboxOrder> => {
    const body: Types.CreateSandboxOrderRequest = {
      ...request,
      idempotency_key: idempotencyKey,
    };
    return this.authenticatedRequest<Types.SandboxOrder>(
      "/saas/sandbox-exchanges/orders",
      {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKey },
        body: JSON.stringify(body),
      },
    );
  };

  liveStatus = (): Promise<Types.LiveExecutionStatus> =>
    this.authenticatedRequest<Types.LiveExecutionStatus>(
      "/saas/live-execution/status",
    );

  liveConnections = (): Promise<Types.LiveConnection[]> =>
    this.authenticatedRequest<Types.LiveConnection[]>(
      "/saas/live-execution/connections",
    );

  createLiveConnection = (
    request: Types.CreateLiveConnectionRequest,
  ): Promise<Types.LiveConnection> =>
    this.authenticatedRequest<Types.LiveConnection>(
      "/saas/live-execution/connections",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
    );

  liveRiskPolicy = (): Promise<Types.LiveRiskPolicy | null> =>
    this.authenticatedRequest<Types.LiveRiskPolicy | null>(
      "/saas/live-execution/risk-policies",
    );

  saveLiveRiskPolicy = (
    request: Types.SaveLiveRiskPolicyRequest,
  ): Promise<Types.LiveRiskPolicy> =>
    this.authenticatedRequest<Types.LiveRiskPolicy>(
      "/saas/live-execution/risk-policies",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
    );

  liveBindings = (): Promise<Types.LiveStrategyBinding[]> =>
    this.authenticatedRequest<Types.LiveStrategyBinding[]>(
      "/saas/live-execution/bindings",
    );

  createLiveBinding = (
    request: Types.CreateLiveStrategyBindingRequest,
  ): Promise<Types.LiveStrategyBinding> =>
    this.authenticatedRequest<Types.LiveStrategyBinding>(
      "/saas/live-execution/bindings",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
    );

  revokeLiveBinding = (bindingId: string): Promise<Types.LiveStrategyBinding> =>
    this.authenticatedRequest<Types.LiveStrategyBinding>(
      `/saas/live-execution/bindings/${pathSegment(bindingId)}/revoke`,
      { method: "POST" },
    );

  requestStartupAuthorizationChallenge = (): Promise<Types.StartupAuthorizationChallenge> =>
    this.authenticatedRequest<Types.StartupAuthorizationChallenge>(
      "/saas/live-execution/startup-authorization/challenge",
      { method: "POST", body: JSON.stringify({}) },
    );

  confirmStartupAuthorization = (
    request: Types.ConfirmStartupAuthorizationRequest,
  ): Promise<Types.StartupAuthorizationConfirmation> =>
    this.authenticatedRequest<Types.StartupAuthorizationConfirmation>(
      "/saas/live-execution/startup-authorization/confirm",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
    );

  revokeStartupAuthorization = (): Promise<Types.StartupAuthorizationRevocation> =>
    this.authenticatedRequest<Types.StartupAuthorizationRevocation>(
      "/saas/live-execution/startup-authorization/revoke",
      { method: "POST" },
    );

  liveConnectionPositions = (
    connectionId: string,
  ): Promise<Types.LivePosition[]> =>
    this.authenticatedRequest<Types.LivePosition[]>(
      `/saas/live-execution/connections/${pathSegment(connectionId)}/positions`,
    );

  liveOrders = (connectionId?: string): Promise<Types.LiveOrder[]> => {
    const params = new URLSearchParams();
    if (connectionId) params.set("connection_id", connectionId);
    return this.authenticatedRequest<Types.LiveOrder[]>(
      withQuery("/saas/live-execution/orders", params),
    );
  };

  refreshLiveOrder = (orderId: string): Promise<Types.LiveOrder> =>
    this.authenticatedRequest<Types.LiveOrder>(
      `/saas/live-execution/orders/${pathSegment(orderId)}/refresh`,
      { method: "POST" },
    );

  createLiveOrder = (
    request: Types.CreateLiveOrderSubmission,
    idempotencyKey: string,
  ): Promise<Types.LiveOrder> => {
    const body: Types.CreateLiveOrderRequest = {
      ...request,
      idempotency_key: idempotencyKey,
    };
    return this.authenticatedRequest<Types.LiveOrder>(
      "/saas/live-execution/orders",
      {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKey },
        body: JSON.stringify(body),
      },
    );
  };

  worldIntelligenceStatus = (): Promise<Types.WorldIntelligenceStatus> =>
    this.publicRequest<Types.WorldIntelligenceStatus>(
      "/world-intelligence/status",
    );

  worldIntelligenceSnapshots = (
    request: Types.WorldIntelligenceSnapshotsRequest = {},
  ): Promise<Types.WorldIntelligenceSnapshotList> => {
    const params = new URLSearchParams();
    if (request.feed) params.set("feed", request.feed);
    if (request.limit !== undefined) params.set("limit", String(request.limit));
    return this.publicRequest<Types.WorldIntelligenceSnapshotList>(
      withQuery("/world-intelligence/snapshots", params),
    );
  };

  predictionMarketCatalog = (
    limit = 50,
  ): Promise<Types.PredictionMarketCatalog> =>
    this.publicRequest<Types.PredictionMarketCatalog>(
      withQuery(
        "/prediction-markets/catalog",
        new URLSearchParams({ limit: String(limit) }),
      ),
    );

  predictionMarketSnapshot = (
    marketId: string,
    outcome: string,
  ): Promise<Types.PredictionMarketSnapshot> =>
    this.publicRequest<Types.PredictionMarketSnapshot>(
      withQuery(
        `/prediction-markets/markets/${pathSegment(marketId)}`,
        new URLSearchParams({ outcome }),
      ),
    );

  predictionMarketSignal = (
    marketId: string,
    outcome: string,
    history: string[] = [],
  ): Promise<Types.PredictionMarketSnapshot> =>
    this.publicRequest<Types.PredictionMarketSnapshot>(
      withQuery(
        `/prediction-markets/markets/${pathSegment(marketId)}/signal`,
        new URLSearchParams({ outcome, history: history.join(",") }),
      ),
    );

  predictionReplayPreview = (
    request: Types.PredictionReplayRequest,
  ): Promise<Types.PredictionReplayResult> =>
    this.publicRequest<Types.PredictionReplayResult>(
      "/prediction-markets/replay/preview",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
    );

  adminTenants = (): Promise<Types.AdminTenant[]> =>
    this.authenticatedRequest<Types.AdminTenant[]>("/saas/admin/tenants");

  adminPlans = (): Promise<Types.ServicePlan[]> =>
    this.authenticatedRequest<Types.ServicePlan[]>("/saas/admin/plans");

  createPlan = (request: Types.CreatePlanRequest): Promise<Types.ServicePlan> =>
    this.authenticatedRequest<Types.ServicePlan>("/saas/admin/plans", {
      method: "POST",
      body: JSON.stringify(request),
    });

  grantSubscription = (
    request: Types.GrantSubscriptionRequest,
  ): Promise<Types.Subscription> =>
    this.authenticatedRequest<Types.Subscription>("/saas/admin/subscriptions", {
      method: "POST",
      body: JSON.stringify(request),
    });

  updateTenantProfile = (
    request: Types.UpdateTenantProfileRequest,
  ): Promise<Types.TenantProfile> => {
    const { tenant_id, ...profile } = request;
    return this.authenticatedRequest<Types.TenantProfile>(
      `/saas/admin/tenants/${pathSegment(tenant_id)}/profile`,
      {
        method: "PATCH",
        body: JSON.stringify(profile),
      },
    );
  };

  createEnterpriseAgreement = (
    request: Types.CreateEnterpriseAgreementRequest,
  ): Promise<Types.EnterpriseAgreement> =>
    this.authenticatedRequest<Types.EnterpriseAgreement>(
      "/saas/admin/agreements",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
    );

  adminAudit = (limit = 100): Promise<Types.AuditEvent[]> =>
    this.authenticatedRequest<Types.AuditEvent[]>(
      withQuery(
        "/saas/admin/audit",
        new URLSearchParams({ limit: String(limit) }),
      ),
    );
}

export const api = new MobileApiClient();

export async function loadSession(): Promise<Types.Session | null> {
  const raw = await readStoredSession();
  if (!raw) return null;
  try {
    const session = JSON.parse(raw) as Types.Session;
    api.setAccessToken(session.accessToken);
    return session;
  } catch {
    await removeStoredSession();
    return null;
  }
}

export async function persistSession(session: Types.Session): Promise<void> {
  api.setAccessToken(session.accessToken);
  await saveStoredSession(JSON.stringify(session));
}

export async function clearSession(): Promise<void> {
  api.setAccessToken("");
  await removeStoredSession();
}

export async function loadRememberedEmail(): Promise<string | null> {
  if (Platform.OS === "web") {
    return globalThis.localStorage?.getItem(REMEMBERED_EMAIL_KEY) ?? null;
  }
  return SecureStore.getItemAsync(REMEMBERED_EMAIL_KEY);
}

export async function persistRememberedEmail(email: string): Promise<void> {
  if (Platform.OS === "web") {
    globalThis.localStorage?.setItem(REMEMBERED_EMAIL_KEY, email);
    return;
  }
  await SecureStore.setItemAsync(REMEMBERED_EMAIL_KEY, email);
}
