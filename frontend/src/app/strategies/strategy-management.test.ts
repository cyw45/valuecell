import assert from "node:assert/strict";
import test from "node:test";
import { strategyManagementActions } from "./strategy-management";

test("running strategy only exposes stop and blocks destructive actions", () => {
  assert.deepEqual(
    strategyManagementActions({ selectedStatus: "running" }),
    { canSave: false, canStart: false, canStop: true, canDelete: false },
  );
});

test("stopped strategies can start concurrently", () => {
  assert.deepEqual(
    strategyManagementActions({ selectedStatus: "stopped" }),
    { canSave: true, canStart: true, canStop: false, canDelete: true },
  );
});

test("new draft can be saved but cannot start, stop, or delete", () => {
  assert.deepEqual(
    strategyManagementActions({ selectedStatus: undefined }),
    { canSave: true, canStart: false, canStop: false, canDelete: false },
  );
});
