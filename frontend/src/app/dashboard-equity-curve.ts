import type { RuleStrategyPnlPoint } from "@/types/rule-strategy";

export type DashboardEquityPoint = RuleStrategyPnlPoint & {
  equity_quote: number;
};

type LiveEquityCurveInput = {
  initialCapital: number;
  currentEquity: number;
  serverPoints: RuleStrategyPnlPoint[];
  nowMs: number;
};

const EMPTY_CURVE_LOOKBACK_MS = 15 * 60 * 1_000;

/**
 * Ensures the account's current mark is always visible, including before the
 * first trade creates a server-side journal point.
 */
export function buildLiveEquityCurve({
  initialCapital,
  currentEquity,
  serverPoints,
  nowMs,
}: LiveEquityCurveInput): DashboardEquityPoint[] {
  const baseline = Number.isFinite(initialCapital) ? initialCapital : 0;
  const equity = Number.isFinite(currentEquity) ? currentEquity : baseline;
  const points = serverPoints.map((point) => ({
    ...point,
    equity_quote: point.equity_quote ?? baseline + point.cumulative_pnl,
  }));
  const currentPoint: DashboardEquityPoint = {
    ts: new Date(nowMs).toISOString(),
    cumulative_pnl: equity - baseline,
    action: "mark_to_market",
    equity_quote: equity,
  };

  if (points.length === 0) {
    return [
      {
        ts: new Date(nowMs - EMPTY_CURVE_LOOKBACK_MS).toISOString(),
        cumulative_pnl: 0,
        action: "initial",
        equity_quote: baseline,
      },
      currentPoint,
    ];
  }

  return [...points, currentPoint];
}
