import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { RuleStrategyEvaluationHistoryEntry } from "@/types/rule-strategy";
import {
  buildDashboardFunnel,
  conditionDisplayName,
  formatConditionValues,
} from "./dashboard-funnel";

const evaluation = (
  overrides: Partial<RuleStrategyEvaluationHistoryEntry> = {},
) =>
  ({
    strategy_id: "strategy-a",
    evaluation_id: "evaluation-a",
    mode: "paper",
    action: "no_op",
    reason_code: "indicator_conditions_not_met",
    reason: "条件不足",
    conditions: [],
    indicators: {} as RuleStrategyEvaluationHistoryEntry["indicators"],
    sizing: {} as RuleStrategyEvaluationHistoryEntry["sizing"],
    funding: {} as RuleStrategyEvaluationHistoryEntry["funding"],
    account: {},
    evaluated_at: "2026-07-20T00:00:00Z",
    trades: [],
    ...overrides,
  }) satisfies RuleStrategyEvaluationHistoryEntry;

describe("dashboard evaluation funnel", () => {
  test("prefers the backend fixed funnel and identifies its first blocker", () => {
    const result = buildDashboardFunnel({
      strategyRunning: true,
      evaluation: evaluation({
        funnel: [
          {
            code: "strategy_run",
            label: "后端策略",
            status: "passed",
            detail: "已运行",
          },
          {
            code: "market_ready",
            label: "后端行情",
            status: "blocked",
            detail: "K 线不足",
          },
          {
            code: "conditions",
            label: "后端条件",
            status: "pending",
            detail: "尚未到达",
          },
        ],
      }),
    });

    assert.deepEqual(
      result.steps.map((step) => step.label),
      [
        "策略运行",
        "行情是否就绪",
        "条件满足几项",
        "风控是否通过",
        "是否已提交订单",
        "是否成交",
      ],
    );
    assert.equal(result.steps[1]?.detail, "K 线不足");
    assert.equal(result.firstBlocker, "行情是否就绪：K 线不足");
  });

  test("safely derives the six stages when an older backend has no funnel", () => {
    const result = buildDashboardFunnel({
      strategyRunning: true,
      evaluation: evaluation({
        entry_confirmation: {
          enabled: 4,
          available: 3,
          passed: 2,
          required: 3,
          mode: "at_least",
        },
      }),
    });

    assert.equal(result.steps.length, 6);
    assert.equal(result.steps[2]?.status, "blocked");
    assert.equal(
      result.steps[2]?.detail,
      "通过 2/4，要求 3 项（3 项数据可用）",
    );
    assert.match(result.firstBlocker ?? "", /条件满足几项/);
  });

  test("renders friendly condition names and actual values without hiding null", () => {
    assert.equal(conditionDisplayName("advanced_macd_entry"), "MACD 入场条件");
    assert.equal(
      formatConditionValues({
        macd: 1.234567,
        signal: -0.5,
        previous: null,
        ready: true,
      }),
      "macd=1.2346 · signal=-0.5 · previous=不可用 · ready=是",
    );
  });
});
