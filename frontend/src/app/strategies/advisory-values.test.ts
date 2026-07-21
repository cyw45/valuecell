import assert from "node:assert/strict";
import test from "node:test";
import { initialCapitalLabel } from "./advisory-values";

test("legacy strategy without initial capital renders a stable unavailable label", () => {
  assert.equal(initialCapitalLabel(undefined), "—");
});

test("initial capital label formats a persisted amount", () => {
  assert.equal(initialCapitalLabel(10_000), "10,000 USDT");
});
