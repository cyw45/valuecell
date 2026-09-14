import type {
  CryptoSymbolUniverse,
  CryptoSymbolUniverseEntry,
} from "@/types/crypto-market";

export type UniverseScope = "admitted" | "rejected" | "all";
export type UniverseSort = "volume" | "age" | "symbol";

/**
 * Admitted symbols of the persisted catalogue version. The persisted version is
 * the only authority on what the venue currently lists, so callers must not
 * substitute guessed symbols when it is missing.
 */
export function universeAdmittedSymbols(
  universe?: CryptoSymbolUniverse | null,
): string[] {
  if (!universe) return [];
  return universe.entries
    .filter((entry) => entry.state === "admitted")
    .map((entry) => entry.symbol);
}

/** Ordered de-duplicated symbol options; primary keeps its original priority. */
export function mergeChartSymbols(
  primary: readonly string[],
  secondary: readonly string[],
): string[] {
  const merged: string[] = [];
  const seen = new Set<string>();
  for (const symbol of [...primary, ...secondary]) {
    if (!symbol || seen.has(symbol)) continue;
    seen.add(symbol);
    merged.push(symbol);
  }
  return merged;
}

/** Maps every symbol to the strategies that currently observe it. */
export function buildWatchedBySymbol(
  strategies?: ReadonlyArray<{
    name: string;
    config: { symbols: readonly string[] };
  }> | null,
): Record<string, string[]> {
  const watched: Record<string, string[]> = {};
  for (const strategy of strategies ?? []) {
    for (const symbol of strategy.config.symbols) {
      const names = watched[symbol] ?? [];
      if (!names.includes(strategy.name)) names.push(strategy.name);
      watched[symbol] = names;
    }
  }
  return watched;
}

/** Missing facts sort last; they are never rendered as a numeric value. */
const MISSING_SORT_SENTINEL = Number.NEGATIVE_INFINITY;

export function filterUniverseEntries(
  entries: readonly CryptoSymbolUniverseEntry[],
  scope: UniverseScope,
  query: string,
  sort: UniverseSort,
): CryptoSymbolUniverseEntry[] {
  const needle = query.trim().toUpperCase();
  const filtered = entries.filter((entry) => {
    if (scope !== "all" && entry.state !== scope) return false;
    if (!needle) return true;
    return entry.symbol.includes(needle);
  });
  const ranked = [...filtered];
  ranked.sort((left, right) => {
    if (sort === "symbol") return left.symbol.localeCompare(right.symbol);
    if (sort === "age") {
      return (
        (right.listing_age_days ?? MISSING_SORT_SENTINEL) -
        (left.listing_age_days ?? MISSING_SORT_SENTINEL)
      );
    }
    return (
      (right.average_quote_volume_30d ?? MISSING_SORT_SENTINEL) -
      (left.average_quote_volume_30d ?? MISSING_SORT_SENTINEL)
    );
  });
  return ranked;
}
