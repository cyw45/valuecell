import type { RuleStrategy } from "@/types/rule-strategy";

type StrategySelection = Pick<
  RuleStrategy,
  "strategy_id" | "status" | "created_at"
>;
type StrategyPickerItemSource = Pick<
  RuleStrategy,
  "strategy_id" | "name" | "status"
> & {
  config: { execution: { environment: "paper" | "okx_demo" } };
};

const ACTIVE_RULE_STRATEGY_STORAGE_PREFIX = "valuecell.rule-strategy-id";

export function activeRuleStrategyStorageKey(userId: string, tenantId: string) {
  return `${ACTIVE_RULE_STRATEGY_STORAGE_PREFIX}:${userId}:${tenantId}`;
}

export function selectActiveRuleStrategyId(
  strategies: StrategySelection[],
  selectedStrategyId: string,
) {
  if (
    strategies.some((strategy) => strategy.strategy_id === selectedStrategyId)
  ) {
    return selectedStrategyId;
  }

  return (
    strategies
      .filter((strategy) => strategy.status === "running")
      .sort(
        (left, right) =>
          new Date(right.created_at ?? 0).getTime() -
          new Date(left.created_at ?? 0).getTime(),
      )[0]?.strategy_id ?? ""
  );
}

export function strategyPickerItems(
  strategies: StrategyPickerItemSource[],
  selectedStrategyId: string,
) {
  return strategies.map((strategy) => ({
    strategyId: strategy.strategy_id,
    name: strategy.name,
    status: strategy.status,
    executionEnvironment: strategy.config.execution.environment,
    selected: strategy.strategy_id === selectedStrategyId,
  }));
}
