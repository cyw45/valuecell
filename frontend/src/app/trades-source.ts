import type { RuleStrategyExecutionConfig } from "@/types/rule-strategy";
import type { SandboxOrder } from "@/types/sandbox-exchange";

export type TradesSource = "pending" | "paper" | "okx_demo";

export function selectTradesSource(
  environment: RuleStrategyExecutionConfig["environment"] | undefined,
): TradesSource {
  return environment ?? "pending";
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
