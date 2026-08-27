import {
  type QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { type ApiResponse, apiClient } from "@/lib/api-client";
import { useSaaSSession } from "@/store/system-store";
import type {
  CreateRuleStrategyRequest,
  EvaluateRuleStrategyRequest,
  RuleStrategy,
  RuleStrategyAdvisory,
  RuleStrategyEvaluation,
  RuleStrategyEvaluationHistoryEntry,
  RuleStrategyExecutionBatchPage,
  RuleStrategyFundingLogEntry,
  RuleStrategyLog,
  RuleStrategyLogEntry,
  RuleStrategyPnlPoint,
  RuleStrategyTextImportJob,
  RuleStrategyTextImportProposal,
  RuleStrategyTradeLogEntry,
  UpdateRuleStrategyRequest,
} from "@/types/rule-strategy";
import type { RuleStrategyDemoExecution } from "@/types/rule-strategy-demo-execution";

const ruleStrategiesKey = (tenantId: string) =>
  ["rule-strategies", tenantId] as const;
const ruleStrategyListKey = (tenantId: string, includeArchived: boolean) =>
  [...ruleStrategiesKey(tenantId), "list", includeArchived] as const;
const ruleStrategyKey = (tenantId: string, strategyId: string) =>
  [...ruleStrategiesKey(tenantId), strategyId] as const;
const ruleStrategyLogKey = (
  tenantId: string,
  strategyId: string,
  logType: "signals" | "trades" | "funding",
  batchId: string | null = null,
) => [...ruleStrategyKey(tenantId, strategyId), logType, batchId ?? "current"] as const;

export function useRuleStrategyBatches(strategyId?: string) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: [...ruleStrategyKey(tenantId, strategyId ?? ""), "batches"],
    queryFn: () => apiClient.get<ApiResponse<RuleStrategyExecutionBatchPage>>(
      `/rule-strategies/${strategyId}/batches?page=1&page_size=100`,
      { requiresAuth: true },
    ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId),
  });
};
const ruleStrategyDemoExecutionKey = (
  tenantId: string,
  strategyId: string,
  page: number,
  pageSize: number,
  batchId: string | null,
  allHistory: boolean,
) => [
    ...ruleStrategyKey(tenantId, strategyId),
    "demo-execution",
    allHistory ? "all-history" : batchId ?? "current",
    page,
    pageSize,
  ] as const;
function invalidateRuleStrategy(
  queryClient: QueryClient,
  tenantId: string,
  strategyId: string,
) {
  return Promise.all([
    queryClient.invalidateQueries({
      queryKey: ruleStrategyKey(tenantId, strategyId),
    }),
    queryClient.invalidateQueries({ queryKey: ruleStrategiesKey(tenantId) }),
  ]);
}

export function useRuleStrategies(tenantId?: string, includeArchived = false) {
  return useQuery({
    queryKey: ruleStrategyListKey(tenantId ?? "", includeArchived),
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategy[]>>(
        includeArchived
          ? "/rule-strategies?include_archived=true"
          : "/rule-strategies",
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(tenantId),
  });
}

export function useRuleStrategy(strategyId?: string) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: ruleStrategyKey(tenantId, strategyId ?? ""),
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategy>>(
        `/rule-strategies/${strategyId}`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId),
  });
}
export type RuleStrategyExportRequest = {
  strategyId: string;
  batchId?: string;
  fromDate?: string;
  toDate?: string;
};

/** Downloads a tenant-authorized strategy workbook as a browser attachment. */
export function useExportRuleStrategy() {
  return useMutation({
    mutationFn: ({ strategyId, batchId, fromDate, toDate }: RuleStrategyExportRequest) => {
      const query = new URLSearchParams();
      if (batchId) query.set("batch_id", batchId);
      if (fromDate) query.set("from_date", fromDate);
      if (toDate) query.set("to_date", toDate);
      const suffix = query.size > 0 ? `?${query}` : "";
      return apiClient.download(
        `/rule-strategies/${encodeURIComponent(strategyId)}/export${suffix}`,
        { requiresAuth: true },
      );
    },
  });
}

/** Reads only the exchange-authoritative Demo execution model for one strategy. */
export function useRuleStrategyDemoExecution(
  strategyId?: string,
  enabled = true,
  page = 1,
  pageSize = 10,
  batchId: string | null = null,
  allHistory = false,
) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: ruleStrategyDemoExecutionKey(
      tenantId,
      strategyId ?? "",
      page,
      pageSize,
      batchId,
      allHistory,
    ),
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyDemoExecution>>(
        `/rule-strategies/${strategyId}/demo-execution?page=${page}&page_size=${pageSize}${allHistory ? "&all_history=true" : batchId ? `&batch_id=${encodeURIComponent(batchId)}` : ""}`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId && enabled),
    refetchInterval: enabled ? 30_000 : false,
  });
}

export function useCreateRuleStrategy() {
  const queryClient = useQueryClient();
  const tenantId = useSaaSSession().tenantId;
  return useMutation({
    mutationFn: (request: CreateRuleStrategyRequest) =>
      apiClient.post<ApiResponse<RuleStrategy>>("/rule-strategies", request, {
        requiresAuth: true,
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ruleStrategiesKey(tenantId) }),
  });
}

export function useUpdateRuleStrategy(strategyId?: string) {
  const queryClient = useQueryClient();
  const tenantId = useSaaSSession().tenantId;
  return useMutation({
    mutationFn: (request: UpdateRuleStrategyRequest) =>
      apiClient.patch<ApiResponse<RuleStrategy>>(
        `/rule-strategies/${strategyId}`,
        request,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      strategyId && invalidateRuleStrategy(queryClient, tenantId, strategyId),
  });
}

export function useDeleteRuleStrategy(strategyId?: string) {
  const queryClient = useQueryClient();
  const tenantId = useSaaSSession().tenantId;
  return useMutation({
    mutationFn: () =>
      apiClient.delete<ApiResponse<{ strategy_id: string; archived: boolean }>>(
        `/rule-strategies/${strategyId}`,
        { requiresAuth: true },
      ),
    onSuccess: () => {
      if (strategyId) {
        queryClient.removeQueries({
          queryKey: ruleStrategyKey(tenantId, strategyId),
        });
      }
      return queryClient.invalidateQueries({
        queryKey: ruleStrategiesKey(tenantId),
      });
    },
  });
}

function useRuleStrategyStatusMutation(
  strategyId: string | undefined,
  status: "start" | "stop",
) {
  const queryClient = useQueryClient();
  const tenantId = useSaaSSession().tenantId;
  return useMutation({
    mutationFn: () =>
      apiClient.post<ApiResponse<RuleStrategy>>(
        `/rule-strategies/${strategyId}/${status}`,
        undefined,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      strategyId && invalidateRuleStrategy(queryClient, tenantId, strategyId),
  });
}
export function useStartRuleStrategy(strategyId?: string) {
  return useRuleStrategyStatusMutation(strategyId, "start");
}
export function useStopRuleStrategy(strategyId?: string) {
  return useRuleStrategyStatusMutation(strategyId, "stop");
}

export type RuleStrategyLifecycleAction = "start" | "stop" | "delete";

export function useRuleStrategyLifecycleAction() {
  const queryClient = useQueryClient();
  const tenantId = useSaaSSession().tenantId;
  return useMutation({
    mutationFn: async ({
      strategyId,
      action,
    }: {
      strategyId: string;
      action: RuleStrategyLifecycleAction;
    }) => {
      if (action === "delete") {
        return apiClient.delete<
          ApiResponse<{ strategy_id: string; archived: boolean }>
        >(`/rule-strategies/${strategyId}`, { requiresAuth: true });
      }
      return apiClient.post<ApiResponse<RuleStrategy>>(
        `/rule-strategies/${strategyId}/${action}`,
        undefined,
        { requiresAuth: true },
      );
    },
    onSuccess: async (_, { strategyId, action }) => {
      if (action === "delete") {
        queryClient.removeQueries({
          queryKey: ruleStrategyKey(tenantId, strategyId),
        });
      }
      await invalidateRuleStrategy(queryClient, tenantId, strategyId);
    },
  });
}

export function useArchiveRuleStrategy(strategyId?: string) {
  const queryClient = useQueryClient();
  const tenantId = useSaaSSession().tenantId;
  return useMutation({
    mutationFn: () =>
      apiClient.delete<ApiResponse<RuleStrategy>>(
        `/rule-strategies/${strategyId}`,
        { requiresAuth: true },
      ),
    onSuccess: () =>
      strategyId && invalidateRuleStrategy(queryClient, tenantId, strategyId),
  });
}

export function useEvaluateRuleStrategy(strategyId?: string) {
  const queryClient = useQueryClient();
  const tenantId = useSaaSSession().tenantId;
  return useMutation({
    mutationFn: (request: EvaluateRuleStrategyRequest) =>
      apiClient.post<ApiResponse<RuleStrategyEvaluation>>(
        `/rule-strategies/${strategyId}/evaluate`,
        request,
        { requiresAuth: true },
      ),
    onSuccess: () => {
      if (!strategyId) return;
      queryClient.invalidateQueries({
        queryKey: ruleStrategyKey(tenantId, strategyId),
      });
      for (const logType of ["signals", "trades", "funding"] as const)
        queryClient.invalidateQueries({
          queryKey: ruleStrategyLogKey(tenantId, strategyId, logType),
        });
      queryClient.invalidateQueries({
        queryKey: [...ruleStrategyKey(tenantId, strategyId), "pnl-curve"],
      });
    },
  });
}

export function useRuleStrategyEvaluations(
  strategyId?: string,
  batchId: string | null = null,
) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: [
      ...ruleStrategyKey(tenantId, strategyId ?? ""),
      "evaluations",
      batchId,
    ] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyEvaluationHistoryEntry[]>>(
        `/rule-strategies/${strategyId}/evaluations?limit=100${batchId ? `&batch_id=${encodeURIComponent(batchId)}` : ""}`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId),
    refetchInterval: 60_000,
  });
}
export function useRuleStrategyAdvisory(strategyId?: string) {
  return useMutation({
    mutationFn: () =>
      apiClient.post<ApiResponse<RuleStrategyAdvisory>>(
        `/rule-strategies/${strategyId}/advisory-analysis`,
        undefined,
        { requiresAuth: true },
      ),
  });
}
export function useParseRuleStrategyText() {
  return useMutation({
    mutationFn: async (strategyText: string) => {
      const requestId = crypto.randomUUID();
      const submitted = await apiClient.post<
        ApiResponse<RuleStrategyTextImportJob>
      >(
        "/rule-strategies/parse-strategy-text/jobs",
        { strategy_text: strategyText, request_id: requestId },
        { requiresAuth: true },
      );
      let job = submitted.data;
      let consecutivePollingFailures = 0;
      while (job.status === "pending" || job.status === "running") {
        const delay = Math.min(
          2_000 * 2 ** consecutivePollingFailures,
          16_000,
        );
        await new Promise((resolve) => setTimeout(resolve, delay));
        try {
          const response = await apiClient.get<
            ApiResponse<RuleStrategyTextImportJob>
          >(`/rule-strategies/parse-strategy-text/jobs/${job.job_id}`, {
            requiresAuth: true,
          });
          job = response.data;
          consecutivePollingFailures = 0;
        } catch (error) {
          consecutivePollingFailures += 1;
          if (consecutivePollingFailures >= 5) throw error;
        }
      }
      if (job.status === "failed") {
        throw new Error(job.error ?? "策略文本解析失败。");
      }
      if (!job.proposal) {
        throw new Error("策略文本解析完成，但未返回解析结果。");
      }
      return {
        ...submitted,
        data: job.proposal,
      } as ApiResponse<RuleStrategyTextImportProposal>;
    },
  });
}
function useRuleStrategyLog<T>(
  strategyId: string | undefined,
  logType: "signals" | "trades" | "funding",
  enabled = true,
  batchId: string | null = null,
) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: ruleStrategyLogKey(tenantId, strategyId ?? "", logType, batchId),
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyLog<T>>>(
        `/rule-strategies/${strategyId}/${logType}?limit=100${batchId ? `&batch_id=${encodeURIComponent(batchId)}` : ""}`,
        { requiresAuth: true },
      ),
    select: (response) => response.data.entries,
    enabled: Boolean(strategyId && tenantId && enabled),
  });
}
export function useRuleStrategySignals(strategyId?: string) {
  return useRuleStrategyLog<RuleStrategyLogEntry>(strategyId, "signals");
}
export function useRuleStrategyTrades(strategyId?: string, enabled = true, batchId: string | null = null) {
  return useRuleStrategyLog<RuleStrategyTradeLogEntry>(
    strategyId,
    "trades",
    enabled,
    batchId,
  );
}
export function useRuleStrategyFunding(
  strategyId?: string,
  batchId: string | null = null,
) {
  return useRuleStrategyLog<RuleStrategyFundingLogEntry>(
    strategyId,
    "funding",
    true,
    batchId,
  );
}
export function useRuleStrategyPnlCurve(
  strategyId?: string,
  batchId: string | null = null,
) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: [
      ...ruleStrategyKey(tenantId, strategyId ?? ""),
      "pnl-curve",
      batchId,
    ] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyPnlPoint[]>>(
        `/rule-strategies/${strategyId}/pnl-curve${batchId ? `?batch_id=${encodeURIComponent(batchId)}` : ""}`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId),
  });
}
export function useRuleStrategyAccount(
  strategyId?: string,
  batchId: string | null = null,
) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: [
      ...ruleStrategyKey(tenantId, strategyId ?? ""),
      "account",
      batchId,
    ] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategy["account"]>>(
        `/rule-strategies/${strategyId}/account${batchId ? `?batch_id=${encodeURIComponent(batchId)}` : ""}`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId),
  });
}

export interface RuleStrategyMonitorState {
  symbol: string;
  state: "candidate" | "admitted" | "held" | "removed";
  reason_code: string | null;
  reason_detail: string | null;
  evaluated_at: string | null;
  next_check_at: string | null;
  protected_held: boolean;
}

export interface RuleStrategyRiskState {
  state: "normal" | "warn" | "only_reduce" | "halted";
  daily_equity_baseline: number;
  high_water_equity: number;
  current_drawdown_pct: number;
  cooldown_until: string | null;
  reason_code: string | null;
  reason_detail: string | null;
}

export function useRuleStrategyMonitorState(strategyId?: string) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: [...ruleStrategyKey(tenantId, strategyId ?? ""), "monitor-state"],
    queryFn: () => apiClient.get<ApiResponse<RuleStrategyMonitorState[]>>(
      `/rule-strategies/${strategyId}/monitor-state`, { requiresAuth: true }
    ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId),
  });
}

export function useRuleStrategyRiskState(strategyId?: string) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: [...ruleStrategyKey(tenantId, strategyId ?? ""), "risk-state"],
    queryFn: () => apiClient.get<ApiResponse<RuleStrategyRiskState>>(
      `/rule-strategies/${strategyId}/risk-state`, { requiresAuth: true }
    ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId),
  });
}
