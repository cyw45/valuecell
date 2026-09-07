import assert from "node:assert/strict";
import test from "node:test";
import type { SandboxOrder } from "@/types/sandbox-exchange";
import type { UnifiedTradeFact } from "@/types/multi-strategy";
import {
  decisionConditions,
  decisionLabel,
  formatConditionValues,
  formatTradeFactIdentifiers,
  tradeFactStatusDescription,
} from "./trades-strategy-conditions";

const order: SandboxOrder = {
  id: "order-1",
  credential_id: "connection-1",
  provider: "okx",
  client_order_id: "client-1",
  symbol: "BTC/USDT",
  side: "buy",
  type: "market",
  requested_quote: 100,
  status: "filled",
  sandbox: true,
  decision_conditions: [
    {
      code: "custom.alpha.signal",
      label: "动态 Alpha 条件",
      category: "custom_alpha",
      state: "triggered",
      values: {
        score: 88.5,
        thresholds: { minimum: 80, regime: "bull" },
        confirmations: ["volume", "trend"],
      },
    },
    {
      code: "risk.position_capacity",
      label: "仓位容量",
      category: "risk",
      state: "not_triggered",
      values: { remaining_quote: 250 },
    },
  ],
  created_at: "2026-08-24T00:00:00Z",
  updated_at: "2026-08-24T00:00:00Z",
};

test("all runtime strategy condition categories remain displayable", () => {
  assert.deepEqual(
    decisionConditions(order).map((condition) => condition.code),
    ["custom.alpha.signal", "risk.position_capacity"],
  );
  assert.match(decisionLabel(order), /动态 Alpha 条件/);
});

test("all eight runtime conditions remain visible when strategy adds a new category", () => {
  const eightConditionOrder: SandboxOrder = {
    ...order,
    decision_conditions: [
      ...(order.decision_conditions ?? []),
      ...Array.from({ length: 6 }, (_, index) => ({
        code: `custom.condition.${index + 1}`,
        label: `动态条件 ${index + 1}`,
        category: "future_strategy",
        state: "not_triggered",
        values: { threshold: index + 1, nested: { enabled: true } },
      })),
    ],
  };
  assert.equal(decisionConditions(eightConditionOrder).length, 8);
  assert.deepEqual(
    decisionConditions(eightConditionOrder).slice(-1)[0]?.values,
    { threshold: 6, nested: { enabled: true } },
  );
});

test("program orders show only their directional leaf conditions", () => {
  const sellOrder: SandboxOrder = {
    ...order,
    side: "sell",
    decision_conditions: [
      { code: "program.entry.1", label: "15m收盘价 > 15mMA20", state: "triggered", values: {} },
      { code: "program.exit.1", label: "1h收盘价 < 1hMA20", state: "triggered", values: {} },
      { code: "program.exit.2", label: "15mRSI14 > 80", state: "triggered", values: {} },
      { code: "program.exit", label: "exit", state: "triggered", values: { passed: 2 } },
    ],
  };

  assert.deepEqual(
    decisionConditions(sellOrder).map((condition) => condition.code),
    ["program.exit.1", "program.exit.2"],
  );
});

test("comparison values render as an explicit actual-value judgment", () => {
  assert.equal(
    formatConditionValues({ left: 84.125, comparator: "gt", right: 80 }),
    "（实际值 84.125 > 目标值 80）",
  );
  assert.equal(
    formatConditionValues({ left: 11.544, comparator: "lt", right: 11.54495 }),
    "（实际值 11.544 < 目标值 11.54495）",
  );
});

test("trade facts expose every durable lifecycle identifier", () => {
  const fact = {
    batch_id: "batch-1",
    intent_id: "intent-1",
    order_id: "order-1",
    fill_id: "fill-1",
    reservation_id: "reservation-1",
  } as UnifiedTradeFact & { reservation_id: string };
  assert.deepEqual(formatTradeFactIdentifiers(fact), [
    "批次 batch-1",
    "预留 reservation-1",
    "意图 intent-1",
    "订单 order-1",
    "成交 fill-1",
  ]);
});

test("unknown submission copy states that resubmission is forbidden", () => {
  assert.match(tradeFactStatusDescription("submission_unknown"), /不可重提/);
  assert.match(tradeFactStatusDescription("recovery_required"), /恢复/);
});
