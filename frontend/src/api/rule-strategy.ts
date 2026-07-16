import {
  type QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { type ApiResponse, apiClient } from "@/lib/api-client";
import type {
  CreateRuleStrategyRequest,
  EvaluateRuleStrategyRequest,
  RuleStrategy,
  RuleStrategyEvaluation,
  RuleStrategyFundingLogEntry,
  RuleStrategyLog,
  RuleStrategyLogEntry,
  RuleStrategyPnlPoint,
  RuleStrategyTradeLogEntry,
  UpdateRuleStrategyRequest,
  RuleStrategyAdvisory,
  RuleStrategyEvaluationHistoryEntry,
  RuleStrategyTextImportProposal,
} from "@/types/rule-strategy";

const ruleStrategyKey = (strategyId: string) =>
  ["rule-strategies", strategyId] as const;
const ruleStrategyLogKey = (
  strategyId: string,
  logType: "signals" | "trades" | "funding",
) => [...ruleStrategyKey(strategyId), logType] as const;
function invalidateRuleStrategy(queryClient: QueryClient, strategyId: string) {
  return queryClient.invalidateQueries({
    queryKey: ruleStrategyKey(strategyId),
  });
}

export function useRuleStrategy(strategyId?: string) {
  return useQuery({
    queryKey: ruleStrategyKey(strategyId ?? ""),
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategy>>(
        `/rule-strategies/${strategyId}`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId),
  });
}

export function useCreateRuleStrategy() {
  return useMutation({
    mutationFn: (request: CreateRuleStrategyRequest) =>
      apiClient.post<ApiResponse<RuleStrategy>>("/rule-strategies", request, {
        requiresAuth: true,
      }),
  });
}

export function useUpdateRuleStrategy(strategyId?: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: UpdateRuleStrategyRequest) =>
      apiClient.patch<ApiResponse<RuleStrategy>>(
        `/rule-strategies/${strategyId}`,
        request,
        { requiresAuth: true },
      ),
    onSuccess: () => {
      if (strategyId) return invalidateRuleStrategy(queryClient, strategyId);
    },
  });
}

function useRuleStrategyStatusMutation(
  strategyId: string | undefined,
  status: "start" | "stop",
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () =>
      apiClient.post<ApiResponse<RuleStrategy>>(
        `/rule-strategies/${strategyId}/${status}`,
        undefined,
        { requiresAuth: true },
      ),
    onSuccess: () => {
      if (strategyId) return invalidateRuleStrategy(queryClient, strategyId);
    },
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

  return useMutation({
    mutationFn: (request: EvaluateRuleStrategyRequest) =>
      apiClient.post<ApiResponse<RuleStrategyEvaluation>>(
        `/rule-strategies/${strategyId}/evaluate`,
        request,
        { requiresAuth: true },
      ),
    onSuccess: () => {
      if (!strategyId) return;
      queryClient.invalidateQueries({ queryKey: ruleStrategyKey(strategyId) });
      for (const logType of ["signals", "trades", "funding"] as const) {
        queryClient.invalidateQueries({
          queryKey: ruleStrategyLogKey(strategyId, logType),
        });
      }
      queryClient.invalidateQueries({
        queryKey: [...ruleStrategyKey(strategyId), "pnl-curve"],
      });
    },
  });
}

export function useRuleStrategyEvaluations(strategyId?: string) {
  return useQuery({
    queryKey: [...ruleStrategyKey(strategyId ?? ""), "evaluations"] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyEvaluationHistoryEntry[]>>(
        `/rule-strategies/${strategyId}/evaluations?limit=100`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId),
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
) {
  return useQuery({
    queryKey: ruleStrategyLogKey(strategyId ?? "", logType),
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyLog<T>>>(
        `/rule-strategies/${strategyId}/${logType}?limit=100`,
        { requiresAuth: true },
      ),
    select: (response) => response.data.entries,
    enabled: Boolean(strategyId),
  });
}

export function useRuleStrategySignals(strategyId?: string) {
  return useRuleStrategyLog<RuleStrategyLogEntry>(strategyId, "signals");
}

export function useRuleStrategyTrades(strategyId?: string) {
  return useRuleStrategyLog<RuleStrategyTradeLogEntry>(strategyId, "trades");
}

export function useRuleStrategyFunding(strategyId?: string) {
  return useRuleStrategyLog<RuleStrategyFundingLogEntry>(strategyId, "funding");
}

export function useRuleStrategyPnlCurve(strategyId?: string) {
  return useQuery({
    queryKey: [...ruleStrategyKey(strategyId ?? ""), "pnl-curve"] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategyPnlPoint[]>>(
        `/rule-strategies/${strategyId}/pnl-curve`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId),
  });
}

export function useRuleStrategyAccount(strategyId?: string) {
  return useQuery({
    queryKey: [...ruleStrategyKey(strategyId ?? ""), "account"] as const,
    queryFn: () =>
      apiClient.get<ApiResponse<RuleStrategy["account"]>>(
        `/rule-strategies/${strategyId}/account`,
        { requiresAuth: true },
      ),
    select: (response) => response.data,
    enabled: Boolean(strategyId),
  });
}
