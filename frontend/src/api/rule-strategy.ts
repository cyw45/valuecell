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
  RuleStrategyFundingLogEntry,
  RuleStrategyLog,
  RuleStrategyLogEntry,
  RuleStrategyPnlPoint,
  RuleStrategyTextImportProposal,
  RuleStrategyTradeLogEntry,
  UpdateRuleStrategyRequest,
} from "@/types/rule-strategy";
import type { RuleStrategyDemoExecution } from "@/types/rule-strategy-demo-execution";

const ruleStrategiesKey = (tenantId: string) =>
  ["rule-strategies", tenantId] as const;
const ruleStrategyKey = (tenantId: string, strategyId: string) =>
  [...ruleStrategiesKey(tenantId), strategyId] as const;
const ruleStrategyLogKey = (
  tenantId: string,
  strategyId: string,
  logType: "signals" | "trades" | "funding",
) => [...ruleStrategyKey(tenantId, strategyId), logType] as const;
const ruleStrategyDemoExecutionKey = (tenantId: string, strategyId: string) =>
  [...ruleStrategyKey(tenantId, strategyId), "demo-execution"] as const;
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

export function useRuleStrategies(tenantId?: string) {
  return useQuery({
    queryKey: ruleStrategiesKey(tenantId ?? ""),
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategy[]>>("/rule-strategies", {
        requiresAuth: true,
      }),
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

/** Reads only the exchange-authoritative Demo execution model for one strategy. */
export function useRuleStrategyDemoExecution(
  strategyId?: string,
  enabled = true,
) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: ruleStrategyDemoExecutionKey(tenantId, strategyId ?? ""),
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyDemoExecution>>(
        `/rule-strategies/${strategyId}/demo-execution`,
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
      apiClient.delete<ApiResponse<{ strategy_id: string }>>(
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

export function useRuleStrategyEvaluations(strategyId?: string) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: [
      ...ruleStrategyKey(tenantId, strategyId ?? ""),
      "evaluations",
    ] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyEvaluationHistoryEntry[]>>(
        `/rule-strategies/${strategyId}/evaluations?limit=100`,
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
    mutationFn: (strategyText: string) =>
      apiClient.post<ApiResponse<RuleStrategyTextImportProposal>>(
        "/rule-strategies/parse-strategy-text",
        { strategy_text: strategyText },
        { requiresAuth: true },
      ),
  });
}
function useRuleStrategyLog<T>(
  strategyId: string | undefined,
  logType: "signals" | "trades" | "funding",
  enabled = true,
) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: ruleStrategyLogKey(tenantId, strategyId ?? "", logType),
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyLog<T>>>(
        `/rule-strategies/${strategyId}/${logType}?limit=100`,
        { requiresAuth: true },
      ),
    select: (response) => response.data.entries,
    enabled: Boolean(strategyId && tenantId && enabled),
  });
}
export function useRuleStrategySignals(strategyId?: string) {
  return useRuleStrategyLog<RuleStrategyLogEntry>(strategyId, "signals");
}
export function useRuleStrategyTrades(strategyId?: string, enabled = true) {
  return useRuleStrategyLog<RuleStrategyTradeLogEntry>(
    strategyId,
    "trades",
    enabled,
  );
}
export function useRuleStrategyFunding(strategyId?: string) {
  return useRuleStrategyLog<RuleStrategyFundingLogEntry>(strategyId, "funding");
}
export function useRuleStrategyPnlCurve(strategyId?: string) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: [
      ...ruleStrategyKey(tenantId, strategyId ?? ""),
      "pnl-curve",
    ] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyPnlPoint[]>>(
        `/rule-strategies/${strategyId}/pnl-curve`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId),
  });
}
export function useRuleStrategyAccount(strategyId?: string) {
  const tenantId = useSaaSSession().tenantId;
  return useQuery({
    queryKey: [
      ...ruleStrategyKey(tenantId, strategyId ?? ""),
      "account",
    ] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategy["account"]>>(
        `/rule-strategies/${strategyId}/account`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId && tenantId),
  });
}
