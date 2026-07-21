import assert from "node:assert/strict";
import test from "node:test";
import {
  activeRuleStrategyStorageKey,
  selectActiveRuleStrategyId,
  strategyPickerItems,
} from "./active-rule-strategy-selection.ts";

test("scopes persisted strategy selection by user and tenant", () => {
  assert.equal(
    activeRuleStrategyStorageKey("user-1", "tenant-a"),
    "valuecell.rule-strategy-id:user-1:tenant-a",
  );
  assert.notEqual(
    activeRuleStrategyStorageKey("user-1", "tenant-a"),
    activeRuleStrategyStorageKey("user-1", "tenant-b"),
  );
});

test("keeps a local selection only when it belongs to the tenant strategy list", () => {
  const strategies = [
    {
      strategy_id: "tenant-a-running",
      status: "running" as const,
      created_at: "2026-07-18T10:00:00Z",
    },
  ];

  assert.equal(
    selectActiveRuleStrategyId(strategies, "tenant-b-stale"),
    "tenant-a-running",
  );
});

test("a running strategy overrides a persisted stopped selection", () => {
  const strategies = [
    {
      strategy_id: "active-running",
      status: "running" as const,
      created_at: "2026-07-18T10:00:00Z",
    },
    {
      strategy_id: "persisted-stopped",
      status: "stopped" as const,
      created_at: "2026-07-18T12:00:00Z",
    },
  ];

  assert.equal(
    selectActiveRuleStrategyId(strategies, "persisted-stopped"),
    "active-running",
  );
});

test("keeps an explicitly selected running strategy when several are running", () => {
  const strategies = [
    { strategy_id: "older-running", status: "running" as const, created_at: "2026-07-18T10:00:00Z" },
    { strategy_id: "newer-running", status: "running" as const, created_at: "2026-07-18T12:00:00Z" },
  ];
  assert.equal(selectActiveRuleStrategyId(strategies, "older-running"), "older-running");
});

test("selects the newest stopped strategy when none is running or persisted", () => {
  const strategies = [
    { strategy_id: "older-stopped", status: "stopped" as const, created_at: "2026-07-18T10:00:00Z" },
    { strategy_id: "newer-stopped", status: "stopped" as const, created_at: "2026-07-18T12:00:00Z" },
  ];
  assert.equal(selectActiveRuleStrategyId(strategies, ""), "newer-stopped");
});

test("fresh browser selects the newest running strategy", () => {
  const strategies = [
    {
      strategy_id: "older",
      status: "running" as const,
      created_at: "2026-07-18T08:00:00Z",
    },
    {
      strategy_id: "stopped-newest",
      status: "stopped" as const,
      created_at: "2026-07-18T12:00:00Z",
    },
    {
      strategy_id: "newest-running",
      status: "running" as const,
      created_at: "2026-07-18T11:00:00Z",
    },
  ];

  assert.equal(selectActiveRuleStrategyId(strategies, ""), "newest-running");
});

test("exposes every tenant strategy for the picker while retaining the active running selection", () => {
  const strategies = [
    {
      strategy_id: "paper-running",
      name: "Paper momentum",
      status: "running" as const,
      config: { execution: { environment: "paper" as const } },
    },
    {
      strategy_id: "okx-stopped",
      name: "Demo mean reversion",
      status: "stopped" as const,
      config: { execution: { environment: "okx_demo" as const } },
    },
  ];

  assert.deepEqual(strategyPickerItems(strategies, "paper-running"), [
    {
      strategyId: "paper-running",
      name: "Paper momentum",
      status: "running",
      executionEnvironment: "paper",
      selected: true,
    },
    {
      strategyId: "okx-stopped",
      name: "Demo mean reversion",
      status: "stopped",
      executionEnvironment: "okx_demo",
      selected: false,
    },
  ]);
});

test("treats a legacy strategy without execution config as paper instead of crashing", () => {
  const strategies = [
    {
      strategy_id: "legacy",
      name: "Legacy strategy",
      status: "running" as const,
      config: {},
    },
  ];

  assert.deepEqual(strategyPickerItems(strategies, "legacy"), [
    {
      strategyId: "legacy",
      name: "Legacy strategy",
      status: "running",
      executionEnvironment: "paper",
      selected: true,
    },
  ]);
});
