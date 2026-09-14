import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import { RuleStrategyEvaluationPath } from "./rule-strategy-evaluation-path.tsx";
import type {
  RuleStrategyCondition,
  RuleStrategyEvaluationHistoryEntry,
} from "@/types/rule-strategy";

/**
 * The leader engine records an availability fact with no value map at all, and
 * the live 2026-09-14 deployment crashed the dashboard with
 * "Cannot convert undefined or null to object" while rendering it. The
 * conditions below mirror that payload, so the reader must keep tolerating a
 * missing or null ``values`` map.
 */
function buildEvaluation(conditions: RuleStrategyCondition[]) {
  return {
    strategy_id: "rule_test",
    evaluation_id: "eval_test",
    mode: "paper",
    action: "no_op",
    reason_code: "quote_volume_unavailable",
    reason: "One or more of the latest six 4h quote-volume facts is unavailable.",
    conditions,
    indicators: {},
    sizing: {},
    funding: {},
    account: {},
    evaluated_at: "2026-09-14T03:00:00+00:00",
    trades: [],
  } as unknown as RuleStrategyEvaluationHistoryEntry;
}

test("renders an availability condition that persists no value map", () => {
  const html = renderToStaticMarkup(
    <RuleStrategyEvaluationPath
      evaluation={buildEvaluation([
        {
          code: "liquidity_quote_volume_available",
          label: "24h quote-volume availability",
          state: "unavailable",
          actual: null,
          threshold: null,
          operator: null,
          detail:
            "All six final 4h quote-volume values are required to calculate 24h liquidity.",
        },
      ])}
    />,
  );

  assert.match(html, /24h quote-volume availability/);
  assert.match(html, /服务端记录/);
});

test("renders a condition whose value map is explicitly null", () => {
  const html = renderToStaticMarkup(
    <RuleStrategyEvaluationPath
      evaluation={buildEvaluation([
        {
          code: "liquidity_quote_volume_available",
          label: "24h quote-volume availability",
          state: "blocked",
          detail: "24 小时成交额偏低，未达到策略流动性门槛。",
          values: null,
        },
      ])}
    />,
  );

  assert.match(html, /24 小时成交额偏低/);
});

test("renders an unavailable condition with neither value map nor detail", () => {
  const html = renderToStaticMarkup(
    <RuleStrategyEvaluationPath
      evaluation={buildEvaluation([
        { code: "price_ma", state: "unavailable", detail: "" },
      ])}
    />,
  );

  assert.match(html, /不可用/);
});
