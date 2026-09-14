import type {
  RuleStrategyAction,
  RuleStrategyCondition,
  RuleStrategyConditionCategory,
} from "@/types/rule-strategy";

// The configurable engine persists `buy`/`sell` while the code-owned fixed
// engines persist `long_entry`/`short_entry`/`exit`. Both name the same order
// decision, so every read surface normalizes the vocabulary in one place
// instead of each component guessing its own list.
const ENTRY_ACTIONS = new Set(["buy", "long_entry", "short_entry", "entry", "add"]);
const EXIT_ACTIONS = new Set(["sell", "exit", "reduce", "close"]);

export function isEntryAction(action?: string | null): boolean {
  return ENTRY_ACTIONS.has(action ?? "");
}

export function isExitAction(action?: string | null): boolean {
  return EXIT_ACTIONS.has(action ?? "");
}

/** Report whether the recorded action produced (or attempted) a real order. */
export function isActionableAction(action?: string | null): boolean {
  return isEntryAction(action) || isExitAction(action);
}

/** Older journals were persisted before the engines labeled their facts. */
export function conditionCategory(
  condition: Pick<RuleStrategyCondition, "category">,
): RuleStrategyConditionCategory {
  return condition.category ?? "indicator";
}

/** Return the condition bucket whose numbers explain the current decision. */
export function conditionBucket(
  action: RuleStrategyAction | string | null | undefined,
  conditions: ReadonlyArray<RuleStrategyCondition>,
): RuleStrategyConditionCategory {
  if (isExitAction(action)) {
    return "exit";
  }
  // A held position is waiting on its exit rules, so those numbers are the ones
  // that explain why no order was produced in this round.
  if (action === "hold" && conditions.some((item) => conditionCategory(item) === "exit")) {
    return "exit";
  }
  return "indicator";
}
