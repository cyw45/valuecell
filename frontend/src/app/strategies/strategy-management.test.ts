import assert from "node:assert/strict";
import test from "node:test";
import { strategyManagementActions } from "./strategy-management";

test("running strategy only exposes stop and blocks destructive actions", () => {
  assert.deepEqual(
    strategyManagementActions({
      selectedStatus: "running",
      anotherRunning: false,
    }),
    { canSave: false, canStart: false, canStop: true, canDelete: false },
  );
});

test("stopped strategy supports update, start, and delete when no strategy runs", () => {
  assert.deepEqual(
    strategyManagementActions({
      selectedStatus: "stopped",
      anotherRunning: false,
    }),
    { canSave: true, canStart: true, canStop: false, canDelete: true },
  );
});

test("another running strategy blocks start but not viewing or editing a stopped strategy", () => {
  assert.deepEqual(
    strategyManagementActions({
      selectedStatus: "stopped",
      anotherRunning: true,
    }),
    { canSave: true, canStart: false, canStop: false, canDelete: true },
  );
});

test("new draft can be saved but cannot start, stop, or delete", () => {
  assert.deepEqual(
    strategyManagementActions({
      selectedStatus: undefined,
      anotherRunning: true,
    }),
    { canSave: true, canStart: false, canStop: false, canDelete: false },
  );
});
