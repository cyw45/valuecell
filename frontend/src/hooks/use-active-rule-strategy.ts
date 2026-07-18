import { useCallback, useEffect, useState } from "react";

const ACTIVE_RULE_STRATEGY_STORAGE_KEY = "valuecell.rule-strategy-id";
const ACTIVE_RULE_STRATEGY_CHANGED_EVENT =
  "valuecell:active-rule-strategy-changed";

function readActiveRuleStrategyId(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(ACTIVE_RULE_STRATEGY_STORAGE_KEY) ?? "";
}

/** Keeps every strategy entry point focused on the same persisted strategy. */
export function useActiveRuleStrategyId(): readonly [
  string,
  (strategyId: string) => void,
] {
  const [strategyId, setStrategyId] = useState(readActiveRuleStrategyId);

  useEffect(() => {
    const synchronize = () => setStrategyId(readActiveRuleStrategyId());
    window.addEventListener(ACTIVE_RULE_STRATEGY_CHANGED_EVENT, synchronize);
    window.addEventListener("storage", synchronize);
    return () => {
      window.removeEventListener(
        ACTIVE_RULE_STRATEGY_CHANGED_EVENT,
        synchronize,
      );
      window.removeEventListener("storage", synchronize);
    };
  }, []);

  const setActiveRuleStrategyId = useCallback((nextStrategyId: string) => {
    window.localStorage.setItem(
      ACTIVE_RULE_STRATEGY_STORAGE_KEY,
      nextStrategyId,
    );
    window.dispatchEvent(new Event(ACTIVE_RULE_STRATEGY_CHANGED_EVENT));
  }, []);

  return [strategyId, setActiveRuleStrategyId] as const;
}
