interface StrategySummary {
  strategy_id: string;
}

interface StrategyDetail {
  config: {
    execution?: {
      environment?: string;
    };
  };
}

interface DashboardLoadingState {
  strategies: StrategySummary[] | undefined;
  strategyId: string;
  ruleStrategy: StrategyDetail | undefined;
  demoExecution: unknown;
  hasError?: boolean;
}

export function shouldShowDashboardPageLoading({
  strategies,
  strategyId,
  ruleStrategy,
  demoExecution,
  hasError = false,
}: DashboardLoadingState): boolean {
  if (hasError) return false;
  if (strategies === undefined) return true;
  if (strategies.length === 0) return false;
  if (!strategyId || ruleStrategy === undefined) return true;
  return (
    ruleStrategy.config.execution?.environment === "okx_demo" &&
    demoExecution === undefined
  );
}

export function shouldShowCandlestickLoading(
  isFetching: boolean,
  marketData: unknown,
): boolean {
  return isFetching && marketData === undefined;
}
