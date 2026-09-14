import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { RuleStrategyEvaluationHistoryEntry } from "@/types/rule-strategy";
import {
  evaluationActionDistribution,
  evaluationSatisfactionTrend,
  evaluationTimelineSegments,
} from "./dashboard-evaluation-rhythm";

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

describe("dashboard evaluation rhythm", () => {
  test("orders the satisfaction trend oldest to newest and skips rows with no recorded conditions", () => {
    const trend = evaluationSatisfactionTrend([
      evaluation({
        evaluation_id: "newest",
        evaluated_at: "2026-07-20T03:00:00Z",
        condition_summary: {
          matched: 3,
          total: 3,
          required: 3,
          available: 3,
        },
      }),
      evaluation({
        evaluation_id: "no-summary",
        evaluated_at: "2026-07-20T02:00:00Z",
      }),
      evaluation({
        evaluation_id: "oldest",
        evaluated_at: "2026-07-20T01:00:00Z",
        condition_summary: {
          matched: 1,
          total: 4,
          required: 2,
          available: 4,
        },
      }),
    ]);

    assert.deepEqual(
      trend.map((point) => point.percent),
      [25, 100],
    );
    assert.equal(trend[0]?.evaluatedAt, "2026-07-20T01:00:00Z");
    assert.equal(trend[1]?.evaluatedAt, "2026-07-20T03:00:00Z");
  });

  test("falls back to the recorded entry confirmation counters", () => {
    const trend = evaluationSatisfactionTrend([
      evaluation({
        entry_confirmation: {
          enabled: 4,
          available: 4,
          passed: 2,
          required: 3,
          mode: "at_least",
        },
      }),
    ]);

    assert.equal(trend.length, 1);
    assert.equal(trend[0]?.percent, 50);
  });

  test("splits recorded decisions into buckets that add up to the window", () => {
    const buckets = evaluationActionDistribution([
      evaluation({ evaluation_id: "a", action: "long_entry" }),
      evaluation({ evaluation_id: "b", action: "exit" }),
      evaluation({ evaluation_id: "c", action: "hold" }),
      evaluation({ evaluation_id: "d", action: "blocked" }),
      evaluation({ evaluation_id: "e", action: "no_signal" }),
    ]);

    const counts = Object.fromEntries(
      buckets.map((bucket) => [bucket.code, bucket.count]),
    );
    assert.deepEqual(counts, { open: 1, close: 1, hold: 2, blocked: 1 });
    const share = buckets.reduce((sum, bucket) => sum + bucket.share, 0);
    assert.ok(Math.abs(share - 100) < 1e-9);
  });

  test("reports zero shares instead of dividing by an empty journal", () => {
    const buckets = evaluationActionDistribution([]);

    assert.equal(buckets.length, 4);
    assert.ok(buckets.every((bucket) => bucket.count === 0 && bucket.share === 0));
  });
  test("falls back to the persisted condition states when no counters were recorded", () => {
    const trend = evaluationSatisfactionTrend([
      evaluation({
        evaluation_id: "fixed-engine",
        condition_summary: { matched: 0, total: 0, required: 0, available: 0 },
        conditions: [
          { code: "a", state: "triggered", detail: "", values: {} },
          { code: "b", state: "not_triggered", detail: "", values: {} },
          { code: "c", state: "not_triggered", detail: "", values: {} },
          { code: "d", state: "unavailable", detail: "", values: {} },
        ],
      }),
    ]);

    assert.equal(trend.length, 1);
    assert.ok(Math.abs((trend[0]?.percent ?? 0) - 100 / 3) < 1e-9);
  });

  test("orders the decision strip oldest to newest and buckets each action", () => {
    const segments = evaluationTimelineSegments([
      evaluation({ evaluation_id: "newest", action: "blocked" }),
      evaluation({ evaluation_id: "middle", action: "exit" }),
      evaluation({ evaluation_id: "oldest", action: "long_entry" }),
    ]);

    assert.deepEqual(
      segments.map((segment) => [segment.key, segment.code]),
      [
        ["oldest", "open"],
        ["middle", "close"],
        ["newest", "blocked"],
      ],
    );
  });
});
