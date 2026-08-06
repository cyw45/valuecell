import assert from "node:assert/strict";
import test from "node:test";
import { buildLiveEquityCurve } from "./dashboard-equity-curve";

test("keeps an equity curve visible before the first trade", () => {
  const nowMs = Date.parse("2026-08-06T12:00:00Z");
  const curve = buildLiveEquityCurve({
    initialCapital: 10_000,
    currentEquity: 10_000,
    serverPoints: [],
    nowMs,
  });

  assert.deepEqual(curve.map((point) => point.equity_quote), [10_000, 10_000]);
  assert.equal(curve[0]?.action, "initial");
  assert.equal(curve[1]?.ts, "2026-08-06T12:00:00.000Z");
});

test("adds the latest account mark after persisted journal points", () => {
  const curve = buildLiveEquityCurve({
    initialCapital: 10_000,
    currentEquity: 10_250,
    serverPoints: [
      {
        ts: "2026-08-06T11:00:00Z",
        cumulative_pnl: 100,
        action: "buy",
      },
    ],
    nowMs: Date.parse("2026-08-06T12:00:00Z"),
  });

  assert.deepEqual(curve.map((point) => point.equity_quote), [10_100, 10_250]);
  assert.equal(curve[1]?.cumulative_pnl, 250);
  assert.equal(curve[1]?.action, "mark_to_market");
});
