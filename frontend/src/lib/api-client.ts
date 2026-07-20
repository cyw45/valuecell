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
  refreshLegacySession?: () => Promise<void>;
  notifyError?: (message: string) => void;
};

export class ApiClient {
  private fetcher: typeof fetch;
  private getSession: () => Session;
  private refreshLegacySession: () => Promise<void>;
  private notifyError: (message: string) => void;

  constructor(dependencies: ApiClientDependencies = {}) {
    this.fetcher = dependencies.fetch ?? fetch;
    this.getSession =
      dependencies.getSession ?? (() => useSystemStore.getState());
    this.refreshLegacySession =
      dependencies.refreshLegacySession ?? (() => this.refreshLegacy());
    this.notifyError =
      dependencies.notifyError ?? ((message) => toast.error(message));
  }

  private async refreshLegacy() {
    const { refresh_token } = this.getSession();
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
      useSystemStore.getState().setSystemInfo({
        access_token,
        refresh_token: nextRefreshToken,
        ...userInfo,
      });
  }

  private async handleResponse<T>(
    response: Response,
    wrapError: boolean,
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
        // SaaS tokens are issued and refreshed by the SaaS auth boundary, not /refresh.
        if (this.getSession().refresh_token) {
          try {
            await this.refreshLegacySession();
          } catch (error) {
            this.notifyError(JSON.stringify(error));
            useSystemStore.getState().clearSystemInfo();
          }
        }
      } else this.notifyError(message);
      throw new ApiError(message, response.status, errorData);
    }
    return response.headers.get("content-type")?.includes("application/json")
      ? response.json()
      : (response.text() as unknown as T);
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
    if (config.requiresAuth) {
      const token = this.getSession().access_token;
      if (token) headers.Authorization = `Bearer ${token}`;
    }
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
    return this.handleResponse<T>(
      await this.fetcher(getServerUrl(endpoint), requestConfig),
      config.wrapError ?? true,
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
