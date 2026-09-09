import type { SandboxOrder } from "@/types/sandbox-exchange";
import type { UnifiedTradeFact } from "@/types/multi-strategy";

type DecisionCondition = NonNullable<SandboxOrder["decision_conditions"]>[number];

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(Number(value.toPrecision(8))) : String(value);
  }
  if (typeof value === "string" || typeof value === "boolean" || value === null) {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/** Return every durable condition emitted by the strategy evaluation. */
export function decisionConditions(order: SandboxOrder): DecisionCondition[] {
  const conditions = order.decision_conditions ?? [];
  const prefix = order.side === "buy" ? "program.entry." : "program.exit.";
  const directionalLeaves = conditions.filter((condition) =>
    condition.code?.startsWith(prefix),
  );
  return directionalLeaves.length > 0 ? directionalLeaves : conditions;
}

function summaryConditions(order: SandboxOrder): DecisionCondition[] {
  const conditions = decisionConditions(order);
  const prefix = order.side === "buy" ? "program.entry." : "program.exit.";
  const preferred = conditions.filter((condition) => condition.code?.startsWith(prefix));
  return preferred.length > 0 ? preferred : conditions;
}

export function formatConditionValues(values?: Record<string, unknown>): string {
  if (!values || Object.keys(values).length === 0) return "";
  const comparator = values.comparator;
  if (
    "left" in values &&
    "right" in values &&
    typeof comparator === "string"
  ) {
    const symbols: Record<string, string> = {
      gt: ">",
      gte: "≥",
      lt: "<",
      lte: "≤",
      eq: "=",
      neq: "≠",
    };
    return `（实际值 ${formatValue(values.left)} ${symbols[comparator] ?? comparator} 目标值 ${formatValue(values.right)}）`;
  }
  return `（${Object.entries(values)
    .map(([key, value]) => `${key}=${formatValue(value)}`)
    .join("，")}）`;
}

export function decisionLabel(order: SandboxOrder): string {
  const conditions = summaryConditions(order);
  const triggered = conditions.filter((condition) => condition.state === "triggered");
  if (triggered.length > 0) {
    return `${order.side === "buy" ? "买入" : "卖出"}：${triggered
      .map((condition) => condition.label || condition.code || "策略条件")
      .join("；")}`;
  }
  return order.decision_reason || order.decision_reason_code || "未记录策略原因";
}

export function formatTradeFactIdentifiers(fact: UnifiedTradeFact): string[] {
  return [
    ["批次", fact.batch_id],
    ["预留", fact.reservation_id],
    ["意图", fact.intent_id],
    ["订单", fact.order_id],
    ["成交", fact.fill_id],
  ].flatMap(([label, value]) => (value ? [`${label} ${value}`] : []));
}

export function formatTradeFactExecution(fact: UnifiedTradeFact): string[] {
  return [
    ["执行路径", fact.explanation.execution_path],
    ["风控", fact.explanation.risk_check],
    ["最终结果", fact.explanation.final_result],
  ].flatMap(([label, value]) => (value ? [`${label} ${value}`] : []));
}

export function tradeFactStatusDescription(status: string): string {
  if (status === "submission_unknown") return "提交结果未确认，待远端对账（不可重提）";
  if (status === "recovery_required") return "需要人工恢复或对账后才能继续";
  if (status === "partially_filled") return "部分成交，剩余数量仍待处理";
  return status;
}
