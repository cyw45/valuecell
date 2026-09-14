import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import type {
  CryptoSymbolUniverse,
  CryptoSymbolUniverseEntry,
} from "@/types/crypto-market";
import { SymbolUniverseBoardView } from "./symbol-universe-board.tsx";

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
  listed_at: "2019-01-01T00:00:00+00:00",
  listing_age_days: 2500,
  average_quote_volume_30d: 12_500_000,
  quote_volume_24h: 13_000_000,
  price_quote: 62_000,
  observed_at: "2026-09-14T00:00:00+00:00",
  ...overrides,
});

const universe: CryptoSymbolUniverse = {
  version: 7,
  status: "active",
  source: "okx",
  quote_asset: "USDT",
  observed_at: "2026-09-14T00:00:00+00:00",
  next_sync_due_at: "2026-12-13T00:00:00+00:00",
  sync_interval_days: 90,
  min_listing_age_days: 90,
  min_average_quote_volume_30d: 5_000_000,
  evaluated_count: 3,
  admitted_count: 3,
  added_count: 1,
  removed_count: 1,
  retained_count: 2,
  reason_detail: null,
  entries: [
    entry("BTC-USDT"),
    entry("NEW-USDT", { decision: "added" }),
    entry("THIN-USDT", {
      average_quote_volume_30d: null,
      listing_age_days: null,
      price_quote: null,
    }),
    entry("GONE-USDT", {
      state: "rejected",
      decision: "removed",
      reason_code: "okx_instrument_not_listed",
    }),
  ],
};

/**
 * The board is the operator-facing catalogue view, so it must render the
 * persisted version facts without inventing numbers or crashing on an entry
 * whose liquidity facts were never recorded.
 */
test("renders the published catalogue, its thresholds and each strategy scope", () => {
  const html = renderToStaticMarkup(
    <SymbolUniverseBoardView
      onSelectSymbol={() => {}}
      selectedSymbol="BTC-USDT"
      universe={universe}
      watchedBySymbol={{ "BTC-USDT": ["双均线", "配对套利"] }}
    />,
  );
  assert.match(html, /版本 v7/);
  assert.match(html, /纳入 3 个标的/);
  assert.match(html, /每 90 天自动同步 OKX/);
  assert.match(html, /上市满 90 天/);
  assert.match(html, /BTC\/USDT/);
  assert.match(html, /—/);
  assert.match(html, /2 个策略/);
});

test("never shows a dropped symbol as tradable nor fabricates a missing fact", () => {
  const html = renderToStaticMarkup(
    <SymbolUniverseBoardView
      onSelectSymbol={() => {}}
      selectedSymbol="GONE-USDT"
      universe={universe}
    />,
  );
  assert.doesNotMatch(html, /GONE\/USDT/);
  assert.doesNotMatch(html, /NaN|undefined|Infinity/);
});
