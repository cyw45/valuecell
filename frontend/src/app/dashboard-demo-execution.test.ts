import assert from "node:assert/strict";
import test from "node:test";
import {
  buildDemoEquityCurve,
  demoOrderStatusLabel,
  demoPnlPresentation,
  demoPurchaseStatePresentation,
  formatOptionalAmount,
} from "./dashboard-demo-execution";

test("purchase state is explicit and readable", () => {
  assert.equal(demoPurchaseStatePresentation("bought").label, "已买入");
  assert.equal(demoPurchaseStatePresentation("not_bought").label, "尚未买入");
  assert.equal(demoPurchaseStatePresentation("pending").label, "待确认");
  assert.equal(demoPurchaseStatePresentation("partially_filled").label, "待确认（部分成交）");
  assert.equal(demoPurchaseStatePresentation("failed").label, "失败");
  assert.equal(demoPurchaseStatePresentation(undefined).label, "待确认");
});

test("order statuses are localized and unknown values remain readable", () => {
  assert.equal(demoOrderStatusLabel("filled"), "已成交");
  assert.equal(demoOrderStatusLabel("partially_filled"), "部分成交");
  assert.equal(demoOrderStatusLabel("submission_unknown"), "待远端对账");
  assert.equal(demoOrderStatusLabel("canceled"), "已取消");
  assert.equal(demoOrderStatusLabel("mystery"), "未知状态（mystery）");
});

test("null amounts are never rendered as zero", () => {
  assert.equal(formatOptionalAmount(null), "—");
  assert.equal(formatOptionalAmount(undefined), "—");
  assert.equal(formatOptionalAmount(0), "0.00");
  assert.equal(formatOptionalAmount("12.5"), "12.50");
});

test("available and partial pnl values are displayed", () => {
  const available = demoPnlPresentation({
    status: "available",
    total_pnl: 12.5,
    realized_pnl: 2,
    unrealized_pnl: 10.5,
    return_pct: 0.0125,
  });
  assert.equal(available.available, true);
  assert.equal(available.totalPnl, 12.5);
  assert.match(available.detail, /已实现.*2\.00.*未实现.*10\.50.*1\.25%/);

  const partial = demoPnlPresentation({
    status: "partial",
    reason_code: "incomplete_fill_metadata",
    total: null,
    value: -3,
  });
  assert.equal(partial.available, false);
  assert.equal(partial.totalPnl, null);
  assert.match(partial.detail, /缺少数量、均价或成本明细/);
  assert.match(partial.detail, /恢复条件/);
});

test("unavailable pnl explains reason and recovery condition in Chinese", () => {
  const result = demoPnlPresentation({
    status: "unavailable",
    reason_code: "missing_fill_history",
  });
  assert.equal(result.available, false);
  assert.match(result.detail, /缺少成交历史/);
  assert.match(result.detail, /恢复条件/);
});

test("demo curve points preserve persisted snapshot actions", () => {
  assert.deepEqual(
    buildDemoEquityCurve({
      points: [
        { ts: "2026-08-11T00:00:00Z", value: "0" },
        { timestamp: "2026-08-11T01:00:00Z", value: "12" },
      ],
    }),
    [
      { ts: "2026-08-11T00:00:00Z", cumulative_pnl: 0, daily_pnl_quote: undefined, equity_quote: undefined, action: "wallet_snapshot" },
      { ts: "2026-08-11T01:00:00Z", cumulative_pnl: 12, daily_pnl_quote: undefined, equity_quote: undefined, action: "wallet_snapshot" },
    ],
  );
});

test("legacy or malformed curve payloads do not crash", () => {
  assert.deepEqual(buildDemoEquityCurve(undefined), []);
  assert.deepEqual(buildDemoEquityCurve({ points: [{ ts: "bad", equity: null }] }), []);
});
