export type DashboardRefreshTarget =
  | "strategy"
  | "monitor-state"
  | "risk-state"
  | "evaluations"
  | "demo-execution"
  | "pnl-curve"
  | "trades";

export const dashboardRefreshTargets = (
  strategyId: string | undefined,
  isOkxDemo: boolean | undefined,
): readonly DashboardRefreshTarget[] => {
  const baseTargets: DashboardRefreshTarget[] = [
    "strategy",
    "monitor-state",
    "risk-state",
    "evaluations",
  ];
  if (!strategyId) return [];
  if (isOkxDemo === undefined) return baseTargets;
  return isOkxDemo
    ? [...baseTargets, "demo-execution"]
    : [...baseTargets, "pnl-curve", "trades"];
};
