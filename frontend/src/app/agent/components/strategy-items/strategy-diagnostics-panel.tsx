import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  CircleSlash,
  Clock3,
  Gauge,
  RadioTower,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { type FC, type ReactNode, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TIME_FORMATS, TimeUtils } from "@/lib/time";
import { formatChange, getChangeType, numberFixed } from "@/lib/utils";
import { useStockColors } from "@/store/settings-store";
import type {
  PortfolioSummary,
  Position,
  Strategy,
  StrategyDiagnostics,
  StrategySymbolDecision,
} from "@/types/strategy";

interface StrategyDiagnosticsPanelProps {
  diagnostics?: StrategyDiagnostics;
  summary?: PortfolioSummary;
  positions?: Position[];
  priceCurve?: Array<Array<number | string>>;
  strategy?: Strategy;
  view?: "overview" | "execution" | "configuration";
}

interface RuntimeMetricCardProps {
  label: string;
  value: ReactNode;
  caption?: ReactNode;
  icon: ReactNode;
  tone?: "neutral" | "positive" | "warning" | "negative";
}

interface SymbolRuntimeRow {
  symbol: string;
  intervals_seen: string[];
  has_market_snapshot: boolean;
  latest_price?: number;
  snapshot_ts_ms?: number;
  freshness_age_ms?: number;
  freshness_status?: "fresh" | "stale" | "unavailable";
  coverage_status?: "complete" | "partial" | "unavailable";
  exposure_increase_allowed?: boolean;
  action?: string;
  quantity?: number;
  reason?: string;
  position?: Position;
}

interface ConditionCheck {
  label: string;
  status: string;
  detail: string;
}

const EMPTY_VALUE = "--";

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined || value === "") return EMPTY_VALUE;
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "number") {
    return Number.isInteger(value)
      ? `${value}`
      : value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
};

const conditionChecks = (
  value: Array<Record<string, unknown>> | undefined,
): ConditionCheck[] =>
  (value ?? []).map((condition) => ({
    label: String(condition.label ?? EMPTY_VALUE),
    status: String(condition.status ?? EMPTY_VALUE),
    detail: String(condition.detail ?? EMPTY_VALUE),
  }));

const impactEntries = (impact: Record<string, unknown> | undefined) =>
  Object.entries(impact ?? {}).filter(
    ([, value]) => value !== undefined && value !== null,
  );

const actionVariant = (action?: string) => {
  if (!action || action === "noop") return "outline" as const;
  if (action.includes("open")) return "default" as const;
  if (action.includes("close")) return "secondary" as const;
  return "outline" as const;
};

const metricToneClassName = (
  tone: RuntimeMetricCardProps["tone"] = "neutral",
) => {
  if (tone === "positive") {
    return "border-emerald-500/20 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300";
  }
  if (tone === "warning") {
    return "border-amber-500/20 bg-amber-500/10 text-amber-600 dark:text-amber-300";
  }
  if (tone === "negative") {
    return "border-rose-500/20 bg-rose-500/10 text-rose-600 dark:text-rose-300";
  }
  return "border-border bg-muted/40 text-muted-foreground";
};


const buildSymbolRuntimeRows = (
  decisions: StrategySymbolDecision[],
  positions: Position[],
): SymbolRuntimeRow[] => {
  const positionsBySymbol = new Map(
    positions.map((position) => [position.symbol, position]),
  );
  const seenSymbols = new Set<string>();
  const rows = decisions.map((decision) => {
    seenSymbols.add(decision.symbol);
    return {
      ...decision,
      position: positionsBySymbol.get(decision.symbol),
    };
  });

  for (const position of positions) {
    if (seenSymbols.has(position.symbol)) continue;
    rows.push({
      symbol: position.symbol,
      intervals_seen: [],
      has_market_snapshot: false,
      latest_price: position.entry_price,
      action: "position",
      quantity: position.quantity,
      reason: undefined,
      position,
    });
  }

  return rows;
};

const RuntimeMetricCard: FC<RuntimeMetricCardProps> = ({
  label,
  value,
  caption,
  icon,
  tone = "neutral",
}) => (
  <Card className="gap-3 overflow-hidden py-4">
    <CardHeader className="flex flex-row items-start justify-between gap-3 px-4">
      <div className="min-w-0">
        <CardDescription className="truncate">{label}</CardDescription>
        <CardTitle className="mt-2 truncate text-base">{value}</CardTitle>
        {caption && (
          <p className="mt-1 truncate text-muted-foreground text-xs">{caption}</p>
        )}
      </div>
      <div
        className={`flex size-9 shrink-0 items-center justify-center rounded-full border ${metricToneClassName(
          tone,
        )}`}
      >
        {icon}
      </div>
    </CardHeader>
  </Card>
);

const StrategyDiagnosticsPanel: FC<StrategyDiagnosticsPanelProps> = ({
  diagnostics,
  summary,
  positions = [],
  priceCurve = [],
  strategy,
  view = "overview",
}) => {
  const { t } = useTranslation();
  const stockColors = useStockColors();
  const latest = diagnostics?.latest_cycle;
  const health = latest?.market_data_health;
  const status = diagnostics?.status ?? strategy?.status;
  const isRunning = status === "running";
  const pnlValue = summary?.total_pnl ?? strategy?.total_pnl;
  const pnlPct = summary?.total_pnl_pct ?? strategy?.total_pnl_pct;
  const pnlType = getChangeType(pnlValue);
  const latestCurvePoint = priceCurve.length > 1 ? priceCurve[priceCurve.length - 1] : undefined;
  const latestCurveValue = latestCurvePoint?.[1];
  const latestEquity =
    typeof latestCurveValue === "number" ? latestCurveValue : summary?.total_value;
  const latestCurveTimeValue = latestCurvePoint?.[0];
  const latestCurveTime =
    typeof latestCurveTimeValue === "string" ? latestCurveTimeValue : undefined;
  const configEntries = Object.entries(diagnostics?.config ?? {}).filter(
    ([, value]) => value !== undefined && value !== null && value !== "",
  );
  const symbolRows = useMemo(
    () => buildSymbolRuntimeRows(diagnostics?.symbol_decisions ?? [], positions),
    [diagnostics?.symbol_decisions, positions],
  );
  const statusLabel = isRunning
    ? t("strategy.status.running")
    : status === "stopped"
      ? t("strategy.status.stopped")
      : EMPTY_VALUE;
  const observedCount = diagnostics?.observed_symbol_count ?? 0;
  const expectedCount = diagnostics?.expected_symbol_count ?? 0;
  const marketIsHealthy = health?.ok ?? (expectedCount > 0 && observedCount >= expectedCount);
  const pnlTone = pnlType === "positive" ? "positive" : pnlType === "negative" ? "negative" : "neutral";
  const explanation = diagnostics?.explanation ?? {};
  const triggeredConditions = conditionChecks(
    Array.isArray(explanation.triggered_conditions)
      ? (explanation.triggered_conditions as Array<Record<string, unknown>>)
      : undefined,
  );
  const blockedConditions = conditionChecks(
    Array.isArray(explanation.blocked_conditions)
      ? (explanation.blocked_conditions as Array<Record<string, unknown>>)
      : undefined,
  );
  const cycleImpact =
    explanation.fund_impact && typeof explanation.fund_impact === "object"
      ? (explanation.fund_impact as Record<string, unknown>)
      : undefined;
  const showOverview = view === "overview";
  const showExecution = view === "execution";
  const showConfiguration = view === "configuration";

  return (
    <div className="flex flex-col gap-5">
      {showOverview && (
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="font-semibold text-base text-foreground">
              {t("strategy.diagnostics.dashboard", { defaultValue: "实时策略驾驶舱" })}
            </h3>
            <p className="mt-1 text-muted-foreground text-sm">
              {t("strategy.diagnostics.dashboardDesc", {
                defaultValue: "实时监控运行状态、当前决策与资金摘要。",
              })}
            </p>
          </div>
          <Badge variant={isRunning ? "default" : "outline"} className="gap-2">
            {isRunning && <span className="size-2 rounded-full bg-emerald-400" />}
            {t("strategy.diagnostics.liveRefresh", { defaultValue: "5 秒刷新" })}
          </Badge>
        </div>
      )}

      {showOverview && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-4">
          <RuntimeMetricCard
            label={t("strategy.diagnostics.runtime", { defaultValue: "运行状态" })}
            value={statusLabel}
            caption={diagnostics?.exchange_id ?? strategy?.exchange_id ?? EMPTY_VALUE}
            icon={isRunning ? <CheckCircle2 className="size-4" /> : <CircleSlash className="size-4" />}
            tone={isRunning ? "positive" : "neutral"}
          />
          <RuntimeMetricCard
            label={t("strategy.portfolio.totalPnl")}
            value={numberFixed(pnlValue, 4)}
            caption={formatChange(pnlPct, "%", 2)}
            icon={pnlType === "negative" ? <TrendingDown className="size-4" /> : <TrendingUp className="size-4" />}
            tone={pnlTone}
          />
          <RuntimeMetricCard
            label={t("strategy.portfolio.totalEquity")}
            value={numberFixed(latestEquity, 4)}
            caption={latestCurveTime ? TimeUtils.formatUTC(latestCurveTime, TIME_FORMATS.MODAL_TRADE_TIME) : t("strategy.diagnostics.waitingCurve", { defaultValue: "等待资产曲线" })}
            icon={<Gauge className="size-4" />}
          />
          <RuntimeMetricCard
            label={t("strategy.diagnostics.marketData", { defaultValue: "OKX 行情抓取" })}
            value={`${health?.fetched_count ?? observedCount}/${expectedCount}`}
            caption={health?.missing_count ? `${t("strategy.diagnostics.missingSymbols", { defaultValue: "缺失行情" })}: ${health.missing_symbols.join(", ")}` : t("strategy.diagnostics.marketReady", { defaultValue: "行情覆盖正常" })}
            icon={marketIsHealthy ? <RadioTower className="size-4" /> : <AlertTriangle className="size-4" />}
            tone={marketIsHealthy ? "positive" : "warning"}
          />
        </div>
      )}

      {(showOverview || showExecution) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {showExecution
                ? t("strategy.diagnostics.latestDecision", { defaultValue: "最近决策" })
                : t("strategy.diagnostics.liveTable", { defaultValue: "动态运行明细" })}
            </CardTitle>
            <CardDescription>
              {latest?.created_at
                ? TimeUtils.formatUTC(latest.created_at, TIME_FORMATS.DATETIME)
                : t("strategy.diagnostics.empty", {
                    defaultValue: "暂无诊断数据。策略完成下一轮扫描后会显示参数、行情抓取和下单原因。",
                  })}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-lg border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("strategy.diagnostics.symbol", { defaultValue: "币种" })}</TableHead>
                    <TableHead>{t("strategy.diagnostics.snapshot", { defaultValue: "实时行情" })}</TableHead>
                    <TableHead>{t("strategy.diagnostics.price", { defaultValue: "价格" })}</TableHead>
                    <TableHead>{t("strategy.diagnostics.action", { defaultValue: "动作" })}</TableHead>
                    <TableHead>{t("strategy.positions.quantity")}</TableHead>
                    <TableHead>{t("strategy.positions.pnl")}</TableHead>
                    <TableHead className="min-w-72">{t("strategy.diagnostics.reason", { defaultValue: "原因" })}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {symbolRows.length > 0 ? symbolRows.map((item) => {
                    const positionPnlType = getChangeType(item.position?.unrealized_pnl);
                    return (
                      <TableRow key={item.symbol}>
                        <TableCell className="font-medium"><div className="flex flex-col gap-1"><span>{item.symbol}</span><span className="text-muted-foreground text-xs">{item.intervals_seen.join(", ") || EMPTY_VALUE}</span></div></TableCell>
                        <TableCell><Badge variant={item.has_market_snapshot ? "secondary" : "outline"}>{item.freshness_status === "stale" ? "Stale" : item.freshness_status === "unavailable" ? "Unavailable" : item.has_market_snapshot ? "Fresh" : "Missing"}</Badge></TableCell>
                        <TableCell>{formatValue(item.latest_price)}</TableCell>
                        <TableCell><Badge variant={actionVariant(item.action)}>{item.action || "noop"}</Badge></TableCell>
                        <TableCell>{formatValue(item.quantity)}</TableCell>
                        <TableCell><span style={{ color: stockColors[positionPnlType] }}>{item.position ? `${formatChange(item.position.unrealized_pnl, "", 2)} (${formatChange(item.position.unrealized_pnl_pct, "", 2)}%)` : EMPTY_VALUE}</span></TableCell>
                        <TableCell className="whitespace-normal text-muted-foreground">{item.reason || EMPTY_VALUE}</TableCell>
                      </TableRow>
                    );
                  }) : <TableRow><TableCell colSpan={7} className="h-28 text-center text-muted-foreground">{t("strategy.diagnostics.empty", { defaultValue: "暂无诊断数据。" })}</TableCell></TableRow>}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
      {showExecution && (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            {t("strategy.diagnostics.executionExplanation", { defaultValue: "本轮执行解释" })}
          </CardTitle>
          <CardDescription>
            {t("strategy.diagnostics.executionExplanationDesc", {
              defaultValue: "记录触发与未触发条件、动作理由，以及本轮资金影响。",
            })}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(16rem,0.8fr)]">
          <div className="space-y-2">
            <p className="font-medium text-sm">{t("strategy.diagnostics.triggered", { defaultValue: "触发条件" })}</p>
            {triggeredConditions.length > 0 ? triggeredConditions.map((condition, index) => (
              <div key={`${condition.label}-${index}`} className="rounded-md border border-emerald-500/20 bg-emerald-500/5 p-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium text-sm">{condition.label}</span>
                  <span className="text-emerald-600 text-xs dark:text-emerald-300">{condition.status}</span>
                </div>
                <p className="mt-1 text-muted-foreground text-xs">{condition.detail}</p>
              </div>
            )) : <p className="text-muted-foreground text-sm">{EMPTY_VALUE}</p>}
          </div>
          <div className="space-y-2">
            <p className="font-medium text-sm">{t("strategy.diagnostics.notTriggered", { defaultValue: "未触发条件" })}</p>
            {blockedConditions.length > 0 ? blockedConditions.map((condition, index) => (
              <div key={`${condition.label}-${index}`} className="rounded-md border border-amber-500/20 bg-amber-500/5 p-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium text-sm">{condition.label}</span>
                  <span className="text-amber-600 text-xs dark:text-amber-300">{condition.status}</span>
                </div>
                <p className="mt-1 text-muted-foreground text-xs">{condition.detail}</p>
              </div>
            )) : <p className="text-muted-foreground text-sm">{EMPTY_VALUE}</p>}
          </div>
          <div className="space-y-3">
            <div>
              <p className="font-medium text-sm">{t("strategy.diagnostics.actionReason", { defaultValue: "动作理由" })}</p>
              <p className="mt-1 text-muted-foreground text-sm">{formatValue(explanation.action_reason)}</p>
            </div>
            <div>
              <p className="font-medium text-sm">{t("strategy.diagnostics.fundImpact", { defaultValue: "资金影响" })}</p>
              <dl className="mt-2 space-y-1 text-sm">
                {impactEntries(cycleImpact).map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-3">
                    <dt className="text-muted-foreground">{key}</dt>
                    <dd className="text-right font-medium">{formatValue(value)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </CardContent>
      </Card>
      )}
      {showExecution && (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            {t("strategy.diagnostics.symbolExplanation", { defaultValue: "币种决策明细" })}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {symbolRows.map((item) => {
            const decision = diagnostics?.symbol_decisions.find(
              (candidate) => candidate.symbol === item.symbol,
            );
            const conditions = conditionChecks(decision?.conditions);
            const impact = impactEntries(decision?.fund_impact);
            return (
              <details key={item.symbol} className="rounded-md border">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2 hover:bg-muted/40">
                  <span className="font-medium text-sm">{item.symbol}</span>
                  <span className="flex items-center gap-2 text-muted-foreground text-xs">
                    {item.action ?? "noop"}
                    <ChevronDown className="size-4" />
                  </span>
                </summary>
                <div className="border-t px-3 py-3">
                  <p className="text-muted-foreground text-sm">{item.reason ?? EMPTY_VALUE}</p>
                  <div className="mt-3 grid gap-4 lg:grid-cols-2">
                    <div className="space-y-2">
                      {conditions.map((condition, index) => (
                        <div key={`${condition.label}-${index}`} className="flex gap-2 text-sm">
                          <span className={condition.status === "triggered" || condition.status === "passed" ? "shrink-0 font-medium text-emerald-600 dark:text-emerald-300" : "shrink-0 font-medium text-amber-600 dark:text-amber-300"}>
                            {condition.status}
                          </span>
                          <span>{condition.label}: {condition.detail}</span>
                        </div>
                      ))}
                    </div>
                    <dl className="space-y-1 text-sm">
                      {impact.map(([key, value]) => (
                        <div key={key} className="flex justify-between gap-3">
                          <dt className="text-muted-foreground">{key}</dt>
                          <dd className="text-right font-medium">{formatValue(value)}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>
                </div>
              </details>
            );
          })}
        </CardContent>
      </Card>
      )}

      {showConfiguration && (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(20rem,0.75fr)]">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t("strategy.diagnostics.config", { defaultValue: "实际生效配置" })}</CardTitle>
              <CardDescription>{t("strategy.diagnostics.configDesc", { defaultValue: "后端已保存且用于执行的策略、金额和仓位参数。" })}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                {configEntries.length > 0 ? configEntries.map(([key, value]) => (
                  <div key={key} className="rounded-md border bg-muted/30 p-3">
                    <p className="text-muted-foreground text-xs">{key}</p>
                    <p className="mt-1 break-words font-medium text-sm">{formatValue(value)}</p>
                  </div>
                )) : <div className="rounded-md border border-dashed bg-muted/30 p-4 text-muted-foreground text-sm md:col-span-2">{t("strategy.diagnostics.noConfig", { defaultValue: "等待后端返回策略执行参数。" })}</div>}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t("strategy.diagnostics.latestCycle", { defaultValue: "最近扫描" })}</CardTitle>
              <CardDescription className="flex items-center gap-2"><Clock3 className="size-3.5" />{latest?.created_at ? TimeUtils.formatUTC(latest.created_at, TIME_FORMATS.DATETIME) : EMPTY_VALUE}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="rounded-md border bg-muted/30 p-3 text-sm leading-relaxed">{latest?.rationale || t("strategy.diagnostics.noRationale", { defaultValue: "暂无本轮理由。" })}</div>
              <div className="grid grid-cols-3 gap-2 text-center text-sm">
                <div className="rounded-md bg-muted/40 p-3"><p className="text-muted-foreground text-xs">orders</p><p className="mt-1 font-semibold">{latest?.order_count ?? 0}</p></div>
                <div className="rounded-md bg-muted/40 p-3"><p className="text-muted-foreground text-xs">no-op</p><p className="mt-1 font-semibold">{latest?.no_order_count ?? 0}</p></div>
                <div className="rounded-md bg-muted/40 p-3"><p className="text-muted-foreground text-xs">cycles</p><p className="mt-1 font-semibold">{diagnostics?.recent_cycles.length ?? 0}</p></div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default StrategyDiagnosticsPanel;
