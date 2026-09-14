import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { CryptoSymbolUniverseEntry } from "@/types/crypto-market";
import {
  buildWatchedBySymbol,
  filterUniverseEntries,
  mergeChartSymbols,
  universeAdmittedSymbols,
} from "./symbol-universe";

const entry = (
  symbol: string,
  overrides: Partial<CryptoSymbolUniverseEntry> = {},
): CryptoSymbolUniverseEntry => ({
  symbol,
  state: "admitted",
  decision: "retained",
  reason_code: "okx_liquidity_verified",
  reason_detail: null,
  permanent_exclusion: false,
  listed_at: "2020-01-01T00:00:00+00:00",
  listing_age_days: 2000,
  average_quote_volume_30d: 10_000_000,
  quote_volume_24h: 12_000_000,
  price_quote: 30_000,
  observed_at: "2026-09-14T00:00:00+00:00",
  ...overrides,
});

const universeOf = (entries: CryptoSymbolUniverseEntry[]) => ({
  version: 3,
  status: "active",
  source: "okx",
  quote_asset: "USDT",
  observed_at: "2026-09-14T00:00:00+00:00",
  next_sync_due_at: null,
  sync_interval_days: 90,
  min_listing_age_days: 90,
  min_average_quote_volume_30d: 5_000_000,
  evaluated_count: entries.length,
  admitted_count: entries.filter((item) => item.state === "admitted").length,
  added_count: 0,
  removed_count: 0,
  retained_count: 0,
  reason_detail: null,
  entries,
});

describe("universeAdmittedSymbols", () => {
  test("keeps only admitted symbols in catalogue order", () => {
    const universe = universeOf([
      entry("BTC-USDT"),
      entry("DEAD-USDT", { state: "rejected", decision: "removed" }),
      entry("ETH-USDT"),
    ]);
    assert.deepEqual(universeAdmittedSymbols(universe), [
      "BTC-USDT",
      "ETH-USDT",
    ]);
  });

  test("returns nothing when the catalogue has not been synced yet", () => {
    assert.deepEqual(universeAdmittedSymbols(undefined), []);
    assert.deepEqual(universeAdmittedSymbols(null), []);
  });
});

describe("mergeChartSymbols", () => {
  test("keeps primary priority and drops duplicates", () => {
    assert.deepEqual(
      mergeChartSymbols(["ETH-USDT", "BTC-USDT"], ["BTC-USDT", "SOL-USDT"]),
      ["ETH-USDT", "BTC-USDT", "SOL-USDT"],
    );
  });

  test("ignores empty entries", () => {
    assert.deepEqual(mergeChartSymbols([""], ["SOL-USDT"]), ["SOL-USDT"]);
  });
});

describe("buildWatchedBySymbol", () => {
  test("maps each symbol to the strategies that observe it", () => {
    const watched = buildWatchedBySymbol([
      { name: "双均线", config: { symbols: ["BTC-USDT", "ETH-USDT"] } },
      { name: "配对套利", config: { symbols: ["ETH-USDT"] } },
    ]);
    assert.deepEqual(watched["BTC-USDT"], ["双均线"]);
    assert.deepEqual(watched["ETH-USDT"], ["双均线", "配对套利"]);
    assert.equal(watched["SOL-USDT"], undefined);
  });

  test("tolerates a missing strategy list", () => {
    assert.deepEqual(buildWatchedBySymbol(undefined), {});
  });
});

describe("filterUniverseEntries", () => {
  const entries = [
    entry("BTC-USDT", {
      average_quote_volume_30d: 900,
      listing_age_days: 3000,
    }),
    entry("ETH-USDT", {
      average_quote_volume_30d: 5_000,
      listing_age_days: 100,
    }),
    entry("SOL-USDT", {
      average_quote_volume_30d: null,
      listing_age_days: null,
      state: "rejected",
      decision: "removed",
      reason_code: "average_quote_volume_below_minimum",
    }),
  ];

  test("filters by scope", () => {
    assert.deepEqual(
      filterUniverseEntries(entries, "rejected", "", "volume").map(
        (item) => item.symbol,
      ),
      ["SOL-USDT"],
    );
    assert.equal(filterUniverseEntries(entries, "all", "", "volume").length, 3);
  });

  test("matches a partial, case-insensitive query", () => {
    assert.deepEqual(
      filterUniverseEntries(entries, "all", "eth", "volume").map(
        (item) => item.symbol,
      ),
      ["ETH-USDT"],
    );
    assert.deepEqual(
      filterUniverseEntries(entries, "all", "USDT", "symbol").map(
        (item) => item.symbol,
      ),
      ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
    );
  });

  test("sorts by volume descending and keeps unknown facts last", () => {
    assert.deepEqual(
      filterUniverseEntries(entries, "all", "", "volume").map(
        (item) => item.symbol,
      ),
      ["ETH-USDT", "BTC-USDT", "SOL-USDT"],
    );
  });

  test("sorts by listing age descending", () => {
    assert.deepEqual(
      filterUniverseEntries(entries, "all", "", "age").map(
        (item) => item.symbol,
      ),
      ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
    );
  });

  test("does not mutate the input entries", () => {
    const before = entries.map((item) => item.symbol);
    filterUniverseEntries(entries, "all", "", "symbol");
    assert.deepEqual(
      entries.map((item) => item.symbol),
      before,
    );
  });
});
