import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { SandboxOrder } from "@/types/sandbox-exchange";
import {
  demoOrderAveragePriceLabel,
  demoOrderFilledQuantityLabel,
  selectTradesSource,
} from "./trades-source";

describe("selectTradesSource", () => {
  test("waits only while the strategy detail is absent", () => {
    assert.equal(selectTradesSource(false, undefined), "pending");
  });

  test("treats a loaded legacy strategy without execution as paper", () => {
    assert.equal(selectTradesSource(true, undefined), "paper");
  });

  test("uses paper trades only for paper execution", () => {
    assert.equal(selectTradesSource(true, "paper"), "paper");
  });

  test("uses exchange-authoritative Demo orders only for OKX Demo", () => {
    assert.equal(selectTradesSource(true, "okx_demo"), "okx_demo");
  });
});

describe("Demo order execution fields", () => {
  const order: SandboxOrder = {
    id: "order-1",
    credential_id: "connection-1",
    provider: "okx",
    client_order_id: "client-1",
    symbol: "BTC/USDT",
    side: "buy",
    type: "market",
    requested_quote: "100",
    requested_quantity: "0.0015",
    status: "filled",
    exchange_order_id: "exchange-1",
    sandbox: true,
    created_at: "2026-07-20T00:00:00Z",
    updated_at: "2026-07-20T00:00:01Z",
  };

  test("does not mislabel requested quantity as filled quantity", () => {
    assert.equal(demoOrderFilledQuantityLabel(order), "不可用");
  });

  test("does not invent an average price absent from the API type", () => {
    assert.equal(demoOrderAveragePriceLabel(order), "不可用");
  });
});
