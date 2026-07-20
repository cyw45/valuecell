import { useCallback, useEffect, useState } from "react";
import { useRuleStrategies } from "@/api/rule-strategy";
import { useSaaSSession } from "@/store/system-store";
import {
  activeRuleStrategyStorageKey,
  selectActiveRuleStrategyId,
} from "./active-rule-strategy-selection";

const ACTIVE_RULE_STRATEGY_CHANGED_EVENT =
  "valuecell:active-rule-strategy-changed";

function readActiveRuleStrategyId(userId: string, tenantId: string): string {
  if (typeof window === "undefined" || !userId || !tenantId) return "";
  return (
    window.localStorage.getItem(
      activeRuleStrategyStorageKey(userId, tenantId),
    ) ?? ""
  );
}

/** Keeps every strategy entry point focused on a user- and tenant-scoped strategy. */
export function useActiveRuleStrategyId(): readonly [
  string,
  (strategyId: string) => void,
] {
  const { userId, tenantId } = useSaaSSession();
  const strategiesQuery = useRuleStrategies(tenantId);
  const [strategyId, setStrategyId] = useState(() =>
    readActiveRuleStrategyId(userId, tenantId),
  );

  useEffect(() => {
    const synchronize = () =>
      setStrategyId(readActiveRuleStrategyId(userId, tenantId));
    synchronize();
    window.addEventListener(ACTIVE_RULE_STRATEGY_CHANGED_EVENT, synchronize);
    window.addEventListener("storage", synchronize);
    return () => {
      window.removeEventListener(
        ACTIVE_RULE_STRATEGY_CHANGED_EVENT,
        synchronize,
      );
      window.removeEventListener("storage", synchronize);
    };
  }, [userId, tenantId]);

  useEffect(() => {
    if (!userId || !tenantId || !strategiesQuery.data) return;
    const selectedId = selectActiveRuleStrategyId(
      strategiesQuery.data,
      strategyId,
    );
    if (selectedId !== strategyId) {
      window.localStorage.setItem(
        activeRuleStrategyStorageKey(userId, tenantId),
        selectedId,
      );
      setStrategyId(selectedId);
    }
  }, [strategiesQuery.data, strategyId, userId, tenantId]);

  const setActiveRuleStrategyId = useCallback(
    (nextStrategyId: string) => {
      if (!userId || !tenantId || typeof window === "undefined") return;
      window.localStorage.setItem(
        activeRuleStrategyStorageKey(userId, tenantId),
        nextStrategyId,
      );
      window.dispatchEvent(new Event(ACTIVE_RULE_STRATEGY_CHANGED_EVENT));
    },
    [userId, tenantId],
  );

  return [strategyId, setActiveRuleStrategyId] as const;
}
