import type { SandboxOrder } from "@/types/sandbox-exchange";

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

/**
 * Preserve the preferred entry/exit grouping when present, but never hide
 * conditions merely because a future strategy uses a new code/category.
 */
export function decisionConditions(order: SandboxOrder): DecisionCondition[] {
  const conditions = order.decision_conditions ?? [];
  const prefix = order.side === "buy" ? "program.entry." : "program.exit.";
  const preferred = conditions.filter((condition) => condition.code?.startsWith(prefix));
  return preferred.length > 0 ? preferred : conditions;
}

export function formatConditionValues(values?: Record<string, unknown>): string {
  if (!values || Object.keys(values).length === 0) return "";
  return `（${Object.entries(values)
    .map(([key, value]) => `${key}=${formatValue(value)}`)
    .join("，")}）`;
}

export function decisionLabel(order: SandboxOrder): string {
  const conditions = decisionConditions(order);
  const triggered = conditions.filter((condition) => condition.state === "triggered");
  if (triggered.length > 0) {
    return `${order.side === "buy" ? "买入" : "卖出"}：${triggered
      .map((condition) => condition.label || condition.code || "策略条件")
      .join("；")}`;
  }
  return order.decision_reason || order.decision_reason_code || "未记录策略原因";
}
