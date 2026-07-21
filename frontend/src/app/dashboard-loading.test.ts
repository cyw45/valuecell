import assert from "node:assert/strict";
import test from "node:test";
import {
  shouldShowCandlestickLoading,
  shouldShowDashboardPageLoading,
} from "./dashboard-loading.ts";

test("dashboard stays loading until the strategy list and active detail are ready", () => {
  assert.equal(
    shouldShowDashboardPageLoading({
      strategies: undefined,
      strategyId: "",
      ruleStrategy: undefined,
      demoExecution: undefined,
    }),
    true,
  );
  assert.equal(
    shouldShowDashboardPageLoading({
      strategies: [{ strategy_id: "strategy-1" }],
      strategyId: "strategy-1",
      ruleStrategy: undefined,
      demoExecution: undefined,
    }),
    true,
  );
});

test("dashboard does not return to loading while ready data refreshes", () => {
  assert.equal(
    shouldShowDashboardPageLoading({
      strategies: [{ strategy_id: "strategy-1" }],
      strategyId: "strategy-1",
      ruleStrategy: { config: { execution: { environment: "paper" } } },
      demoExecution: undefined,
    }),
    false,
  );
});

test("OKX Demo dashboard waits for authoritative execution data", () => {
  const ruleStrategy = {
    config: { execution: { environment: "okx_demo" } },
  } as const;
  assert.equal(
    shouldShowDashboardPageLoading({
      strategies: [{ strategy_id: "strategy-1" }],
      strategyId: "strategy-1",
      ruleStrategy,
      demoExecution: undefined,
    }),
    true,
  );
  assert.equal(
    shouldShowDashboardPageLoading({
      strategies: [{ strategy_id: "strategy-1" }],
      strategyId: "strategy-1",
      ruleStrategy,
      demoExecution: {},
    }),
    false,
  );
});

test("dashboard exits loading when a critical request fails", () => {
  assert.equal(
    shouldShowDashboardPageLoading({
      strategies: undefined,
      strategyId: "",
      ruleStrategy: undefined,
      demoExecution: undefined,
      hasError: true,
    }),
    false,
  );
});

test("candlestick loading is limited to the first fetch without data", () => {
  assert.equal(shouldShowCandlestickLoading(true, undefined), true);
  assert.equal(shouldShowCandlestickLoading(true, { symbols: [] }), false);
  assert.equal(shouldShowCandlestickLoading(false, undefined), false);
});
