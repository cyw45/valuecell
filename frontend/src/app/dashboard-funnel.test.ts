import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { RuleStrategyEvaluationHistoryEntry } from "@/types/rule-strategy";
import {
  buildDashboardFunnel,
  conditionDisplayName,
  conditionSatisfactionPercent,
  dashboardConditionSummary,
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

  test("renders a legacy condition with null values without throwing", () => {
    assert.equal(
      formatConditionValues(null),
      "无实际值",
    );
  });

  test("explains a fixed engine round whose entry conditions are not met", () => {
    const result = buildDashboardFunnel({
      strategyRunning: true,
      evaluation: evaluation({
        action: "no_signal",
        conditions: [
          {
            code: "trend.sma10_vs_sma20",
            category: "indicator",
            state: "triggered",
            detail: "bearish",
            values: {},
          },
          {
            code: "entry.price_cross_up",
            category: "indicator",
            state: "not_triggered",
            detail: "no cross",
            values: {},
          },
          {
            code: "entry.price_cross_down",
            category: "indicator",
            state: "not_triggered",
            detail: "no cross",
            values: {},
          },
        ],
      }),
    });

    assert.equal(result.steps[2]?.status, "blocked");
    assert.equal(result.steps[2]?.detail, "通过 1/3，要求 3 项（3 项数据可用）");
  });

  test("explains a held fixed position with its exit rules, not the entry rules", () => {
    const result = buildDashboardFunnel({
      strategyRunning: true,
      evaluation: evaluation({
        action: "hold",
        conditions: [
          {
            code: "trend.sma10_vs_sma20",
            category: "indicator",
            state: "triggered",
            detail: "bull",
            values: {},
          },
          {
            code: "exit.stop_loss",
            category: "exit",
            state: "not_triggered",
            detail: "above stop",
            values: {},
          },
          {
            code: "exit.timeout",
            category: "exit",
            state: "not_triggered",
            detail: "young",
            values: {},
          },
        ],
      }),
    });

    assert.equal(result.steps[2]?.detail, "通过 0/2，要求 1 项（2 项数据可用）");
  });

  test("never prints a meaningless zero-of-zero condition ratio", () => {
    const result = buildDashboardFunnel({
      strategyRunning: true,
      evaluation: evaluation({ action: "no_signal", conditions: [] }),
    });

    assert.equal(result.steps[2]?.detail, "本轮未记录条件明细");
  });
});

describe("dashboard condition satisfaction", () => {
  test("reads the recorded condition summary when the journal has one", () => {
    const summary = dashboardConditionSummary(
      evaluation({
        condition_summary: {
          matched: 2,
          total: 4,
          required: 3,
          available: 4,
        },
        entry_confirmation: {
          enabled: 9,
          available: 9,
          passed: 9,
          required: 9,
          mode: "at_least",
        },
      }),
    );

    assert.deepEqual(summary, {
      matched: 2,
      total: 4,
      required: 3,
      available: 4,
    });
    assert.equal(conditionSatisfactionPercent(summary), 50);
  });

  test("falls back to the entry confirmation counters for older journals", () => {
    const summary = dashboardConditionSummary(
      evaluation({
        entry_confirmation: {
          enabled: 5,
          available: 4,
          passed: 1,
          required: 5,
          mode: "at_least",
        },
      }),
    );

    assert.deepEqual(summary, {
      matched: 1,
      total: 5,
      required: 5,
      available: 4,
    });
    assert.equal(conditionSatisfactionPercent(summary), 20);
  });

  test("reports no ratio instead of a fake 0% when nothing was evaluated", () => {
    assert.equal(dashboardConditionSummary(undefined), null);
    assert.equal(dashboardConditionSummary(evaluation()), null);
    assert.equal(
      conditionSatisfactionPercent({
        matched: 0,
        total: 0,
        required: 0,
        available: 0,
      }),
      null,
    );
  });
});
