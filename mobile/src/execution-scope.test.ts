import assert from "node:assert/strict";
import test from "node:test";
import { attributedDecisionReason, executionQueryScope } from "./execution-scope.ts";

test("stopped demo waits for the current batch instead of requesting an empty snapshot", () => {
  assert.deepEqual(
    executionQueryScope({ environment: "okx_demo", status: "stopped", currentBatchId: undefined }),
    { ready: false, batchId: null, allHistory: false, unavailableReason: "batch_pending" },
  );
  assert.deepEqual(
    executionQueryScope({ environment: "okx_demo", status: "stopped", currentBatchId: "batch-1" }),
    { ready: true, batchId: "batch-1", allHistory: false, unavailableReason: null },
  );
});

test("a stopped demo with no current batch does not fall back to older batches", () => {
  assert.equal(
    executionQueryScope({ environment: "okx_demo", status: "stopped", currentBatchId: null }).unavailableReason,
    "no_current_batch",
  );
});

test("running demo and explicit history keep the web query shape", () => {
  assert.equal(
    executionQueryScope({ environment: "okx_demo", status: "running", currentBatchId: undefined }).ready,
    true,
  );
  assert.equal(
    executionQueryScope({ environment: "okx_demo", status: "stopped", currentBatchId: null, selectedBatchId: "__all__" }).allHistory,
    true,
  );
});

test("order rows use the unified trade fact when the order payload has no reason", () => {
  assert.equal(
    attributedDecisionReason(
      { id: "order-1", decision_reason: null, decision_reason_code: "entry" },
      [{ order_id: "order-1", explanation: { decision_reason: "收盘价上穿均线" } }],
    ),
    "收盘价上穿均线",
  );
});
