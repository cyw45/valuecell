import type { RuleStrategyExecutionConfig } from "@/types/rule-strategy";
import type { SandboxOrder } from "@/types/sandbox-exchange";

export type TradesSource = "pending" | "paper" | "okx_demo";

export function selectTradesSource(
  hasStrategy: boolean,
  environment: RuleStrategyExecutionConfig["environment"] | undefined,
): TradesSource {
  if (!hasStrategy) return "pending";
  return environment ?? "paper";
}

// The authoritative Demo order response currently exposes requested quantity,
// but no exchange-filled quantity or average fill price. Keep these explicitly
// unavailable rather than substituting paper fills or requested values.
export function demoOrderFilledQuantityLabel(_order: SandboxOrder): string {
  return "不可用";
}

export function demoOrderAveragePriceLabel(_order: SandboxOrder): string {
  return "不可用";
}
