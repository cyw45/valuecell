import { useEffect, useMemo, useState } from "react";
import { Coins, Filter, Search, ShieldCheck } from "lucide-react";
import { useGetCryptoSymbolUniverse } from "@/api/crypto-market";
import {
  filterUniverseEntries,
  type UniverseScope,
  type UniverseSort,
} from "@/app/symbol-universe";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import type {
  CryptoSymbolUniverse,
  CryptoSymbolUniverseEntry,
} from "@/types/crypto-market";

const compactAmount = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
  notation: "compact",
});

const priceFormat = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 8,
});

const SCOPE_LABELS: Record<UniverseScope, string> = {
  admitted: "已纳入",
  rejected: "已剔除",
  all: "全部",
};

const SORT_LABELS: Record<UniverseSort, string> = {
  volume: "按30日成交额",
  age: "按上市天数",
  symbol: "按名称",
};

const MAX_VISIBLE_ROWS = 150;

interface SymbolUniverseBoardProps {
  className?: string;
  selectedSymbol: string;
  onSelectSymbol: (symbol: string) => void;
  watchedBySymbol?: Record<string, string[]>;
}

export interface SymbolUniverseBoardViewProps extends SymbolUniverseBoardProps {
  universe?: CryptoSymbolUniverse;
  isUniverseError?: boolean;
}

function decisionLabel(entry: CryptoSymbolUniverseEntry): string {
  if (entry.state === "admitted") {
    return entry.decision === "added" ? "本轮新增" : "持续纳入";
  }
  if (entry.permanent_exclusion) {
    return "永久剔除";
  }
  return entry.decision === "removed" ? "本轮剔除" : "未达标";
}

function decisionTone(entry: CryptoSymbolUniverseEntry): string {
  if (entry.state === "admitted") {
    return "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300";
  }
  if (entry.permanent_exclusion) {
    return "border-rose-500/40 bg-rose-500/10 text-rose-600 dark:text-rose-300";
  }
  return "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-300";
}

function formatObserved(value?: string | null): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "—";
  return parsed.toLocaleString();
}

/**
 * The symbol universe board is the catalogue view for every strategy: it shows
 * what OKX currently lists, why each symbol was admitted or rejected, and which
 * strategies observe it. It never derives facts of its own — every number here
 * comes from the persisted sync version.
 */
export function SymbolUniverseBoardView({
  className,
  universe,
  isUniverseError,
  selectedSymbol,
  onSelectSymbol,
  watchedBySymbol,
}: SymbolUniverseBoardViewProps) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<UniverseScope>("admitted");
  const [sort, setSort] = useState<UniverseSort>("volume");

  useEffect(() => {
    if (!universe) return;
    const admittedSymbols = new Set(
      universe.entries
        .filter((entry) => entry.state === "admitted")
        .map((entry) => entry.symbol),
    );
    if (selectedSymbol && !admittedSymbols.has(selectedSymbol)) {
      // Keep a strategy symbol that the venue dropped visible instead of
      // silently switching the chart to another instrument.
      setScope("all");
    }
  }, [selectedSymbol, universe]);

  const rows = useMemo(
    () =>
      universe
        ? filterUniverseEntries(universe.entries, scope, query, sort)
        : [],
    [query, scope, sort, universe],
  );

  return (
    <Card className={className}>
      <CardHeader className="gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Coins className="size-5" /> 币种清单
            </CardTitle>
            <CardDescription className="mt-1">
              {universe
                ? `版本 v${universe.version} · 纳入 ${universe.admitted_count} 个标的 · 本轮评估 ${universe.evaluated_count} 个 · 数据时间 ${formatObserved(universe.observed_at)}`
                : isUniverseError
                  ? "币种目录暂时不可用，将在下一次自动同步后恢复。"
                  : "正在读取交易所符号目录。"}
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge className="font-normal" variant="outline">
              <ShieldCheck className="mr-1 size-3.5" />
              {universe
                ? `每 ${universe.sync_interval_days} 天自动同步 OKX`
                : "同步周期读取中"}
            </Badge>
            {universe?.next_sync_due_at ? (
              <Badge className="font-normal" variant="secondary">
                下次同步 {formatObserved(universe.next_sync_due_at)}
              </Badge>
            ) : null}
          </div>
        </div>
        <p className="text-muted-foreground text-xs">
          {universe
            ? `入池门槛：USDT 现货 · 上市满 ${universe.min_listing_age_days} 天 · 近 30 天日均成交额 ≥ ${compactAmount.format(universe.min_average_quote_volume_30d)} USDT。退市与不再支持的标的自动剔除，新达标的活跃标的自动纳入，策略观察范围随之自动调整。`
            : "目录门槛将在同步完成后显示。"}
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <div className="relative flex-1">
            <Search className="absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              aria-label="搜索币种"
              className="pl-9"
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索，例如 BTC 或 BTC-USDT"
              value={query}
            />
          </div>
          <Select
            onValueChange={(value) => setScope(value as UniverseScope)}
            value={scope}
          >
            <SelectTrigger aria-label="按纳入状态筛选" className="sm:w-36">
              <Filter className="size-3.5" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(SCOPE_LABELS) as UniverseScope[]).map((key) => (
                <SelectItem key={key} value={key}>
                  {SCOPE_LABELS[key]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select
            onValueChange={(value) => setSort(value as UniverseSort)}
            value={sort}
          >
            <SelectTrigger aria-label="排序方式" className="sm:w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(SORT_LABELS) as UniverseSort[]).map((key) => (
                <SelectItem key={key} value={key}>
                  {SORT_LABELS[key]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="max-h-96 overflow-y-auto rounded-md border">
          <table className="w-full border-collapse text-sm">
            <thead className="sticky top-0 bg-muted/80 text-muted-foreground backdrop-blur">
              <tr>
                <th className="px-3 py-2 text-left font-medium">币种</th>
                <th className="px-3 py-2 text-right font-medium">最新价</th>
                <th className="px-3 py-2 text-right font-medium">
                  近30日日均成交额
                </th>
                <th className="px-3 py-2 text-right font-medium">上市天数</th>
                <th className="px-3 py-2 text-left font-medium">目录结论</th>
                <th className="px-3 py-2 text-left font-medium">策略关注</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, MAX_VISIBLE_ROWS).map((entry) => {
                const watchers = watchedBySymbol?.[entry.symbol] ?? [];
                return (
                  <tr
                    className={cn(
                      "cursor-pointer border-t transition-colors hover:bg-muted/60",
                      entry.symbol === selectedSymbol && "bg-primary/10",
                    )}
                    key={entry.symbol}
                    onClick={() => onSelectSymbol(entry.symbol)}
                  >
                    <td className="px-3 py-2 font-medium">
                      {entry.symbol.replace("-", "/")}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {entry.price_quote == null
                        ? "—"
                        : priceFormat.format(entry.price_quote)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {entry.average_quote_volume_30d == null
                        ? "—"
                        : compactAmount.format(entry.average_quote_volume_30d)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {entry.listing_age_days ?? "—"}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={cn(
                          "rounded-full border px-2 py-0.5 text-xs",
                          decisionTone(entry),
                        )}
                        title={entry.reason_detail ?? undefined}
                      >
                        {decisionLabel(entry)}
                      </span>
                    </td>
                      <td className="px-3 py-2 text-muted-foreground text-xs">
                      {watchers.length > 0 ? `${watchers.length} 个策略` : "—"}
                    </td>
                  </tr>
                );
              })}
              {rows.length === 0 ? (
                <tr>
                  <td
                    className="px-3 py-6 text-center text-muted-foreground"
                    colSpan={6}
                  >
                    {universe
                      ? "没有符合当前筛选条件的币种。"
                      : "目录同步完成后即可浏览。"}
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <p className="text-muted-foreground text-xs">
          {rows.length > MAX_VISIBLE_ROWS
            ? `共 ${rows.length} 条，显示前 ${MAX_VISIBLE_ROWS} 条；可用搜索或筛选收窄。点击任意币种可切换上方的走势与技术指标。`
            : `共 ${rows.length} 条；点击任意币种可切换上方的走势与技术指标。`}
          {universe?.reason_detail ? ` ${universe.reason_detail}` : ""}
        </p>
      </CardContent>
    </Card>
  );
}

/** Data half: reads the published catalogue and hands it to the view. */
export function SymbolUniverseBoard({
  className,
  selectedSymbol,
  onSelectSymbol,
  watchedBySymbol,
}: SymbolUniverseBoardProps) {
  const universeQuery = useGetCryptoSymbolUniverse();
  return (
    <SymbolUniverseBoardView
      className={className}
      isUniverseError={universeQuery.isError}
      onSelectSymbol={onSelectSymbol}
      selectedSymbol={selectedSymbol}
      universe={universeQuery.data}
      watchedBySymbol={watchedBySymbol}
    />
  );
}

export default SymbolUniverseBoard;
