import assert from "node:assert/strict";
import test from "node:test";
import {
  demoExecutionCheckedAtLabel,
  demoExecutionUnvaluedAssetCount,
} from "./rule-strategy-demo-execution.ts";

test("reports the strategy-scoped Demo snapshot timestamp and unvalued asset count", () => {
  const snapshot = {
    checked_at: "2026-07-19T12:34:56Z",
    account: {
      scope: "exchange_connection_shared_account" as const,
      data: {
        source: "okx_demo" as const,
        checked_at: "2026-07-19T12:34:50Z",
        balances: [
          { currency: "USDT", valuation_status: "priced" as const },
          { currency: "XYZ", valuation_status: "unpriced" as const },
        ],
        total_usdt_value: 100,
      },
    },
  };

  assert.equal(demoExecutionCheckedAtLabel(snapshot), "2026-07-19T12:34:56Z");
  assert.equal(demoExecutionUnvaluedAssetCount(snapshot), 1);
});

test("uses the account snapshot timestamp when the read-model timestamp is absent", () => {
  const snapshot = {
    account: {
      scope: "exchange_connection_shared_account" as const,
      data: {
        source: "okx_demo" as const,
        checked_at: "2026-07-19T12:34:50Z",
        balances: [],
        total_usdt_value: 0,
      },
    },
  };

  assert.equal(demoExecutionCheckedAtLabel(snapshot), "2026-07-19T12:34:50Z");
  assert.equal(demoExecutionUnvaluedAssetCount(snapshot), 0);
});
