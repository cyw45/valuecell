import type {
  RuleStrategyEvaluationHistoryEntry,
  RuleStrategyFunnelCode,
  RuleStrategyFunnelStage,
  RuleStrategyFunnelStatus,
} from "@/types/rule-strategy";

export const DASHBOARD_FUNNEL_STAGES: ReadonlyArray<{
  code: RuleStrategyFunnelCode;
  label: string;
}> = [
  { code: "strategy_run", label: "策略运行" },
  { code: "market_ready", label: "行情是否就绪" },
  { code: "conditions", label: "条件满足几项" },
  { code: "risk", label: "风控是否通过" },
  { code: "order_submission", label: "是否已提交订单" },
  { code: "fill", label: "是否成交" },
];

const blockedStatuses = new Set<RuleStrategyFunnelStatus>([
  "blocked",
  "rejected",
]);

function fallbackFunnel(
  strategyRunning: boolean,
  evaluation?: RuleStrategyEvaluationHistoryEntry,
): RuleStrategyFunnelStage[] {
  const confirmation = evaluation?.entry_confirmation;
  const relevantConditions =
    evaluation?.conditions.filter(
      (condition) => condition.category === "indicator",
    ) ?? [];
  const total = confirmation?.enabled ?? relevantConditions.length;
  const available =
    confirmation?.available ??
    relevantConditions.filter((condition) => condition.state !== "unavailable")
      .length;
  const passed =
    confirmation?.passed ??
    relevantConditions.filter((condition) => condition.state === "triggered")
      .length;
  const required = confirmation?.required ?? total;
  const hasSignal =
    evaluation?.action === "buy" || evaluation?.action === "sell";
  const marketBlocked =
    evaluation?.status === "blocked" &&
    (evaluation.stage === "market_data" || evaluation.stage === "account_sync");
  const riskBlocked =
    evaluation?.stage === "risk" ||
    evaluation?.conditions.some(
      (condition) =>
        condition.category === "risk" && condition.state === "blocked",
    );
  const execution = evaluation?.execution;
  const executionStatus =
    execution && typeof execution.status === "string"
      ? execution.status.toLowerCase()
      : "";
  const executionResult =
    execution && typeof execution.execution === "string"
      ? execution.execution.toLowerCase()
      : "";
  const submitted =
    [
      "submitted",
      "open",
      "partially_filled",
      "partial",
      "filled",
      "closed",
      "canceled",
      "cancelled",
    ].includes(executionStatus) ||
    Boolean(evaluation?.paper_fill || evaluation?.trades.length);
  const submissionRejected =
    ["rejected", "failed", "stale"].includes(executionStatus) ||
    executionResult === "blocked";
  const filled = Boolean(
    evaluation?.paper_fill ||
      evaluation?.trades.length ||
      ["filled", "closed"].includes(executionStatus),
  );
  const conditionDetail = evaluation
    ? `通过 ${passed}/${total}，要求 ${required} 项（${available} 项数据可用）`
    : "等待策略评估";
  const stage = (
    code: RuleStrategyFunnelCode,
    status: RuleStrategyFunnelStatus,
    detail: string,
  ): RuleStrategyFunnelStage => ({
    code,
    label:
      DASHBOARD_FUNNEL_STAGES.find((item) => item.code === code)?.label ?? code,
    status,
    detail,
  });

  return [
    stage(
      "strategy_run",
      strategyRunning ? "passed" : "blocked",
      strategyRunning ? "调度器正在扫描" : "策略未运行",
    ),
    stage(
      "market_ready",
      marketBlocked ? "blocked" : evaluation ? "passed" : "pending",
      marketBlocked
        ? (evaluation?.reason ?? "行情数据不可用")
        : evaluation
          ? "行情已进入本轮评估"
          : "等待首轮行情",
    ),
    stage(
      "conditions",
      marketBlocked
        ? "pending"
        : hasSignal
          ? "passed"
          : evaluation
            ? "blocked"
            : "pending",
      marketBlocked ? "行情未就绪，尚未评估条件" : conditionDetail,
    ),
    stage(
      "risk",
      marketBlocked || !hasSignal
        ? "pending"
        : riskBlocked
          ? "blocked"
          : "passed",
      marketBlocked || !hasSignal
        ? "尚未进入风控"
        : riskBlocked
          ? (evaluation?.reason ?? "风控未通过")
          : "风控检查通过",
    ),
    stage(
      "order_submission",
      submissionRejected ? "rejected" : submitted ? "passed" : "pending",
      submissionRejected
        ? (evaluation?.reason ?? "订单提交失败")
        : submitted
          ? "订单已提交"
          : "尚未提交订单",
    ),
    stage(
      "fill",
      filled
        ? "filled"
        : ["partial", "partially_filled"].includes(executionStatus)
          ? "partial"
          : "pending",
      filled
        ? "订单已成交"
        : ["partial", "partially_filled"].includes(executionStatus)
          ? "订单部分成交"
          : "等待成交或本轮无订单",
    ),
  ];
}

export function buildDashboardFunnel({
  strategyRunning,
  evaluation,
}: {
  strategyRunning: boolean;
  evaluation?: RuleStrategyEvaluationHistoryEntry;
}): { steps: RuleStrategyFunnelStage[]; firstBlocker: string | null } {
  const fallback = fallbackFunnel(strategyRunning, evaluation);
  const backendByCode = new Map(
    (evaluation?.funnel ?? []).map((stage) => [stage.code, stage]),
  );
  const steps = evaluation?.funnel?.length
    ? fallback.map((safeStage) => {
        const backendStage = backendByCode.get(safeStage.code);
        return backendStage
          ? { ...backendStage, label: safeStage.label }
          : safeStage;
      })
    : fallback;
  const firstBlocked = steps.find((step) => blockedStatuses.has(step.status));
  return {
    steps,
    firstBlocker: firstBlocked
      ? `${firstBlocked.label}：${firstBlocked.detail}`
      : null,
  };
}

const CONDITION_NAMES: Record<string, string> = {
  moving_average: "均线条件",
  macd: "MACD 条件",
  bollinger: "布林带条件",
  rsi: "RSI 条件",
  momentum: "动量条件",
  brar: "BRAR 条件",
};

export function conditionDisplayName(code: string): string {
  const normalized = code.toLowerCase();
  const base = Object.entries(CONDITION_NAMES).find(([key]) =>
    normalized.includes(key),
  )?.[1];
  const suffix = normalized.includes("entry")
    ? "入场条件"
    : normalized.includes("exit")
      ? "离场条件"
      : undefined;
  if (base && suffix) return `${base.replace("条件", "").trim()} ${suffix}`;
  if (base) return base;
  return code.split("_").join(" ");
}

function formatValue(value: number | string | boolean | null): string {
  if (value === null) return "不可用";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number" && !Number.isInteger(value))
    return Number(value.toFixed(4)).toString();
  return String(value);
}

export function formatConditionValues(
  values: Record<string, number | string | boolean | null>,
): string {
  const entries = Object.entries(values);
  if (!entries.length) return "无实际值";
  return entries
    .map(([key, value]) => `${key}=${formatValue(value)}`)
    .join(" · ");
}
