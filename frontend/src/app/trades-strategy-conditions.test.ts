import assert from "node:assert/strict";
import test from "node:test";
import type { SandboxOrder } from "@/types/sandbox-exchange";
import {
  decisionConditions,
  decisionLabel,
  formatConditionValues,
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

test("condition values render unknown scalar, nested, and array fields", () => {
  const rendered = formatConditionValues(order.decision_conditions?.[0].values);
  assert.match(rendered, /score=88\.5/);
  assert.match(rendered, /thresholds=/);
  assert.match(rendered, /minimum/);
  assert.match(rendered, /confirmations=/);
  assert.match(rendered, /volume/);
});
