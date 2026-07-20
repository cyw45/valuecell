import { describe, test } from "node:test";
import assert from "node:assert/strict";
import type { AdvancedRuleSetConfig, RuleStrategyEntryConfirmation } from "./rule-strategy";

describe("entry confirmation contract", () => {
  test("supports count and ratio modes and evaluation summary", () => {
    const countRule = { entry_confirmation_mode: "at_least", entry_confirmation_count: 2 } satisfies Partial<AdvancedRuleSetConfig>;
    const ratioRule = { entry_confirmation_mode: "ratio", entry_confirmation_ratio: 0.5 } satisfies Partial<AdvancedRuleSetConfig>;
    const summary: RuleStrategyEntryConfirmation = { enabled: 6, available: 6, passed: 2, required: 2, mode: "at_least" };
    assert.equal(countRule.entry_confirmation_count, 2);
    assert.equal(ratioRule.entry_confirmation_ratio, 0.5);
    assert.equal(summary.required, 2);
  });
});
