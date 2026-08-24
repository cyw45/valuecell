import assert from "node:assert/strict";
import test from "node:test";
import {
  buildDemoEquityCurve,
  buildStrategyHoldingRows,
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

test("strategy holdings exclude unrelated shared-account assets", () => {
  const rows = buildStrategyHoldingRows([], [
    {
      symbol: "BTC/USDT",
      base_currency: "BTC",
      quantity: 9,
      available_quantity: 9,
      frozen_quantity: 0,
      mark_price: 110,
      notional_usdt: 990,
      unrealized_pnl_usdt: 0,
    },
  ]);
  assert.deepEqual(rows, []);
});

test("strategy holdings use confirmed net fills and shared marks only", () => {
  const baseOrder = {
    id: "order-1",
    credential_id: "connection-1",
    provider: "okx" as const,
    client_order_id: "client-1",
    symbol: "BTC/USDT",
    side: "buy" as const,
    type: "market" as const,
    requested_quote: 100,
    filled_quantity: 1,
    average_fill_price: 100,
    status: "filled",
    sandbox: true as const,
    created_at: "2026-08-24T00:00:00Z",
    updated_at: "2026-08-24T00:00:00Z",
  };
  const rows = buildStrategyHoldingRows(
    [
      baseOrder,
      {
        ...baseOrder,
        id: "order-2",
        side: "sell",
        filled_quantity: 0.25,
        average_fill_price: 105,
        created_at: "2026-08-24T01:00:00Z",
      },
    ],
    [
      {
        symbol: "BTC/USDT",
        base_currency: "BTC",
        quantity: 9,
        available_quantity: 9,
        frozen_quantity: 0,
        mark_price: 110,
        notional_usdt: 990,
        unrealized_pnl_usdt: 0,
      },
    ],
  );
  assert.equal(rows.length, 1);
  assert.equal(rows[0].position.quantity, 0.75);
  assert.equal(rows[0].position.entry_price, 100);
  assert.equal(rows[0].value, 82.5);
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
