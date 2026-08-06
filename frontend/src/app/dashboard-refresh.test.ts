import assert from "node:assert/strict";
import test from "node:test";
import { dashboardRefreshTargets } from "./dashboard-refresh";

test("OKX Demo refresh excludes paper PnL endpoints", () => {
  assert.deepEqual(dashboardRefreshTargets("rule_1", true), [
    "strategy",
    "monitor-state",
    "risk-state",
    "evaluations",
    "demo-execution",
  ]);
});

test("paper refresh excludes Demo execution", () => {
  assert.deepEqual(dashboardRefreshTargets("rule_1", false), [
    "strategy",
    "monitor-state",
    "risk-state",
    "evaluations",
    "pnl-curve",
    "trades",
  ]);
});

test("missing strategy identity refreshes no strategy-specific endpoints", () => {
  assert.deepEqual(dashboardRefreshTargets(undefined, undefined), []);
});
