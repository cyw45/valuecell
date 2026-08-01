import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

const ACTIVE_STRATEGY_PREFIX = "valuecell.rule-strategy-id";

export function activeStrategyStorageKey(userId: string, tenantId: string): string {
  return `${ACTIVE_STRATEGY_PREFIX}:${userId}:${tenantId}`;
}

export async function readActiveStrategyId(
  userId: string,
  tenantId: string,
): Promise<string> {
  const key = activeStrategyStorageKey(userId, tenantId);
  if (Platform.OS === "web") {
    return globalThis.localStorage?.getItem(key) ?? "";
  }
  return (await SecureStore.getItemAsync(key)) ?? "";
}

export async function saveActiveStrategyId(
  userId: string,
  tenantId: string,
  strategyId: string,
): Promise<void> {
  const key = activeStrategyStorageKey(userId, tenantId);
  if (Platform.OS === "web") {
    globalThis.localStorage?.setItem(key, strategyId);
    return;
  }
  await SecureStore.setItemAsync(key, strategyId);
}

export function selectActiveStrategyId(
  strategies: Array<{ strategy_id: string; status: string; created_at?: string }>,
  selectedStrategyId: string,
): string {
  if (strategies.some((strategy) => strategy.strategy_id === selectedStrategyId)) {
    return selectedStrategyId;
  }
  const running = strategies
    .filter((strategy) => strategy.status === "running")
    .sort(
      (left, right) =>
        new Date(right.created_at ?? 0).getTime() -
        new Date(left.created_at ?? 0).getTime(),
    );
  return running[0]?.strategy_id ?? strategies[0]?.strategy_id ?? "";
}

export function formatQuote(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} USDT`;
}

export function formatTimestamp(value: unknown): string {
  if (typeof value !== "string" || !value) return "—";
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime()) ? value : timestamp.toLocaleString();
}
