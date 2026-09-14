import type { CryptoSymbolUniverse, CryptoSymbolUniverseEntry } from "./types";

const UNIVERSE_REASON_LABELS: Record<string, string> = {
  okx_liquidity_verified: "达标纳入",
  okx_instrument_not_listed: "OKX 已下架",
  quote_asset_not_supported: "非 USDT 计价",
  stable_base_asset_excluded: "稳定币不计入",
  leveraged_token_excluded: "杠杆代币不计入",
  listing_age_below_minimum: "上市时间不足",
  average_quote_volume_below_minimum: "成交额不足",
  quote_volume_unavailable: "成交额缺失",
  crypto_universe_liquidity_lost: "流动性跌破门槛",
  universe_seeded_from_code_defaults: "初始播种目录",
};

/** Unknown codes fall back to the raw code instead of an invented label. */
export function universeReasonLabel(reasonCode: string): string {
  return UNIVERSE_REASON_LABELS[reasonCode] ?? reasonCode;
}

export function universeAdmittedSymbols(
  universe?: CryptoSymbolUniverse | null,
): string[] {
  if (!universe) return [];
  return universe.entries
    .filter((entry) => entry.state === "admitted")
    .map((entry) => entry.symbol);
}

/** Symbols dropped from the venue this round; permanent exclusions stay out. */
export function universeRemovedEntries(
  universe?: CryptoSymbolUniverse | null,
): CryptoSymbolUniverseEntry[] {
  if (!universe) return [];
  return universe.entries.filter(
    (entry) => entry.state === "rejected" && !entry.permanent_exclusion,
  );
}

export function universeEntryForSymbol(
  universe: CryptoSymbolUniverse | undefined,
  symbol: string,
): CryptoSymbolUniverseEntry | undefined {
  return universe?.entries.find((entry) => entry.symbol === symbol);
}

export function formatUniverseUsd(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude >= 100_000_000) return `${(value / 100_000_000).toFixed(2)} 亿`;
  if (magnitude >= 10_000) return `${(value / 10_000).toFixed(2)} 万`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function formatUniversePrice(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 8 });
}

export function universeObservedLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function universeNextSyncLabel(value?: string | null): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

export function universeVersionSummary(universe: CryptoSymbolUniverse): string {
  const parts = [
    `目录 v${universe.version}`,
    `纳入 ${universe.admitted_count} 个`,
    `本轮评估 ${universe.evaluated_count} 个`,
  ];
  if (universe.added_count > 0) parts.push(`新增 ${universe.added_count}`);
  if (universe.removed_count > 0) parts.push(`剔除 ${universe.removed_count}`);
  return parts.join(" · ");
}

export function universeThresholdSummary(universe: CryptoSymbolUniverse): string {
  const minVolume = formatUniverseUsd(universe.min_average_quote_volume_30d);
  return `入池门槛：USDT 现货 · 上市满 ${universe.min_listing_age_days} 天 · 近 30 天日均成交额 ≥ ${minVolume} USDT`;
}

export function universeSyncSummary(universe: CryptoSymbolUniverse): string {
  const nextSync = universeNextSyncLabel(universe.next_sync_due_at);
  const parts = [`每 ${universe.sync_interval_days} 天自动同步 OKX`];
  if (nextSync) parts.push(`下次 ${nextSync}`);
  parts.push(`数据时间 ${universeObservedLabel(universe.observed_at)}`);
  return parts.join(" · ");
}

/** Every fact is rendered as persisted; missing facts show as "—". */
export function universeEntrySummary(entry: CryptoSymbolUniverseEntry): string {
  const parts = [
    `结论 ${universeReasonLabel(entry.reason_code)}`,
    `最新价 ${formatUniversePrice(entry.price_quote)}`,
    `近30日日均 ${formatUniverseUsd(entry.average_quote_volume_30d)} USDT`,
    `上市 ${entry.listing_age_days == null ? "—" : `${entry.listing_age_days} 天`}`,
  ];
  if (entry.permanent_exclusion) parts.push("永久剔除");
  return parts.join(" · ");
}
