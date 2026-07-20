import { toast } from "sonner";
import { getUserInfo } from "@/api/system";
import { VALUECELL_BACKEND_URL } from "@/constants/api";
import { useSystemStore } from "@/store/system-store";
import type { SystemInfo } from "@/types/system";

export class ApiError extends Error {
  public status: number;
  public details?: unknown;
  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}
export interface ApiResponse<T> {
  code: number;
  data: T;
  msg: string;
}
export interface RequestConfig {
  requiresAuth?: boolean;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  keepalive?: boolean;
  wrapError?: boolean;
}
export const getServerUrl = (endpoint: string) =>
  endpoint.startsWith("http")
    ? endpoint
    : `${import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1"}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;

type Session = Pick<
  SystemInfo,
  "access_token" | "refresh_token" | "id" | "tenant_id"
>;
type ApiClientDependencies = {
  fetch?: typeof fetch;
  getSession?: () => Session;
  refreshLegacySession?: (
    session: Session,
  ) => Promise<Partial<SystemInfo> | void>;
  setSession?: (session: Partial<SystemInfo>) => void;
  clearSession?: () => void;
  notifyError?: (message: string) => void;
};

export class ApiClient {
  private fetcher: typeof fetch;
  private getSession: () => Session;
  private refreshLegacySession: (
    session: Session,
  ) => Promise<Partial<SystemInfo> | void>;
  private legacyRefreshFlights = new Map<string, Promise<void>>();
  private setSession: (session: Partial<SystemInfo>) => void;
  private clearSession: () => void;
  private notifyError: (message: string) => void;

  constructor(dependencies: ApiClientDependencies = {}) {
    this.fetcher = dependencies.fetch ?? fetch;
    this.getSession =
      dependencies.getSession ?? (() => useSystemStore.getState());
    this.refreshLegacySession =
      dependencies.refreshLegacySession ??
      ((session) => this.refreshLegacy(session));
    this.setSession =
      dependencies.setSession ??
      ((session) => useSystemStore.getState().setSystemInfo(session));
    this.clearSession =
      dependencies.clearSession ??
      (() => useSystemStore.getState().clearSystemInfo());
    this.notifyError =
      dependencies.notifyError ?? ((message) => toast.error(message));
  }

  private async refreshLegacy(session: Session) {
    const { refresh_token } = session;
    const {
      data: { access_token, refresh_token: nextRefreshToken },
    } = await this.post<
      ApiResponse<Pick<SystemInfo, "access_token" | "refresh_token">>
    >(
      `${VALUECELL_BACKEND_URL}/refresh`,
      { refreshToken: refresh_token },
      { wrapError: false },
    );
    if (!access_token || !nextRefreshToken) return;
    const userInfo = await getUserInfo(access_token);
    if (userInfo)
      return {
        access_token,
        refresh_token: nextRefreshToken,
        ...userInfo,
      };
  }

  private async handleResponse<T>(
    response: Response,
    wrapError: boolean,
    requestSession?: Session,
  ): Promise<T> {
    if (wrapError && !response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const message = JSON.stringify(
        errorData.message ||
          errorData.detail ||
          response.statusText ||
          `HTTP ${response.status}`,
      );
      if (response.status === 401) {
        if (!requestSession || !this.isCurrentSession(requestSession)) {
          throw new ApiError(message, response.status, errorData);
        }
        // SaaS tokens are issued and refreshed by the SaaS auth boundary, not /refresh.
        if (requestSession.refresh_token) {
          try {
            const flightKey = this.sessionKey(requestSession);
            let flight = this.legacyRefreshFlights.get(flightKey);
            if (!flight) {
              flight = this.refreshLegacySession(requestSession)
                .then((refreshedSession) => {
                  if (refreshedSession && this.isCurrentSession(requestSession))
                    this.setSession(refreshedSession);
                })
                .finally(() => {
                  this.legacyRefreshFlights.delete(flightKey);
                });
              this.legacyRefreshFlights.set(flightKey, flight);
            }
            await flight;
          } catch (error) {
            this.notifyError(JSON.stringify(error));
            if (this.isCurrentSession(requestSession)) this.clearSession();
          }
        } else if (requestSession.access_token) {
          this.clearSession();
        }
      } else this.notifyError(message);
      throw new ApiError(message, response.status, errorData);
    }
    return response.headers.get("content-type")?.includes("application/json")
      ? response.json()
      : (response.text() as unknown as T);
  }

  private isCurrentSession(requestSession: Session) {
    const current = this.getSession();
    return (
      current.access_token === requestSession.access_token &&
      current.refresh_token === requestSession.refresh_token &&
      current.id === requestSession.id &&
      current.tenant_id === requestSession.tenant_id
    );
  }

  private sessionKey(session: Session) {
    return JSON.stringify([
      session.id,
      session.tenant_id,
      session.access_token,
      session.refresh_token,
    ]);
  }

  private async request<T>(
    method: string,
    endpoint: string,
    data?: unknown,
    config: RequestConfig = {},
  ): Promise<T> {
    // Never mutate defaults or caller headers: each request gets an isolated header object.
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...config.headers,
    };
    const requestSession = config.requiresAuth
      ? { ...this.getSession() }
      : undefined;
    if (requestSession?.access_token)
      headers.Authorization = `Bearer ${requestSession.access_token}`;
    const requestConfig: RequestInit = {
      method,
      headers,
      signal: config.signal,
      keepalive: config.keepalive,
    };
    if (data && ["POST", "PUT", "PATCH"].includes(method)) {
      if (data instanceof FormData) {
        delete headers["Content-Type"];
        requestConfig.body = data;
      } else requestConfig.body = JSON.stringify(data);
    }
    // Browser native methods require the Window receiver. Saving `fetch` as a
    // bare function and calling it as `this.fetcher(...)` invokes it with the
    // ApiClient instance instead, which throws `Illegal invocation` in some
    // browsers (notably after a fresh SPA deployment).
    const fetcher = this.fetcher;
    // Test doubles are ordinary functions; native browser fetch must retain the
    // Window receiver. `globalThis` is safe in both browser and test runtimes.
    const receiver = typeof window === "undefined" ? globalThis : window;
    return this.handleResponse<T>(
      await fetcher.call(receiver, getServerUrl(endpoint), requestConfig),
      config.wrapError ?? true,
      requestSession,
    );
  }
  get<T>(endpoint: string, config?: RequestConfig) {
    return this.request<T>("GET", endpoint, undefined, config);
  }
  post<T>(endpoint: string, data?: unknown, config?: RequestConfig) {
    return this.request<T>("POST", endpoint, data, config);
  }
  put<T>(endpoint: string, data?: unknown, config?: RequestConfig) {
    return this.request<T>("PUT", endpoint, data, config);
  }
  patch<T>(endpoint: string, data?: unknown, config?: RequestConfig) {
    return this.request<T>("PATCH", endpoint, data, config);
  }
  delete<T>(endpoint: string, config?: RequestConfig) {
    return this.request<T>("DELETE", endpoint, undefined, config);
  }
  upload<T>(endpoint: string, formData: FormData, config?: RequestConfig) {
    return this.request<T>("POST", endpoint, formData, config);
  }
}
export const apiClient = new ApiClient();
