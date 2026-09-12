import {
  AlertTriangle,
  BarChart3,
  CandlestickChart,
  CircleDollarSign,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Cpu,
  Crosshair,
  FileDown,
  Layers3,
  Moon,
  RadioTower,
  RefreshCw,
  Sun,
  TrendingDown,
  TrendingUp,
  WalletCards,
} from "lucide-react";
import { useTheme } from "next-themes";
import { toast } from "sonner";
import {
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useSearchParams } from "react-router";
import { useGetCryptoMarketIndicators } from "@/api/crypto-market";
import {
  useRuleStrategy,
  useRuleStrategyDemoExecution,
  useRuleStrategyEvaluations,
  useExportRuleStrategy,
  useRuleStrategyMonitorState,
  useRuleStrategyPnlCurve,
  useRuleStrategyRiskState,
  useRuleStrategyTrades,
  useSharedAccountSummary,
  useUpdateStrategyAllocationCap,
} from "@/api/rule-strategy";
import {
  buildDashboardFunnel,
  conditionDisplayName,
  formatConditionValues,
} from "@/app/dashboard-funnel";
import {
  buildDemoEquityCurve,
  buildStrategyHoldingRows,
  allocationPnlPresentation,
  demoOrderStatusLabel,
  demoPnlPresentation,
  demoPurchaseStatePresentation,
  formatDemoTime,
  formatOptionalAmount,
  formatOptionalPercent,
} from "@/app/dashboard-demo-execution";
import { dashboardRefreshTargets } from "@/app/dashboard-refresh";
import { DashboardStrategyManagement } from "@/app/dashboard-strategy-management";
import {
  shouldShowCandlestickLoading,
  shouldShowDashboardPageLoading,
} from "@/app/dashboard-loading";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import CandlestickChartComponent, {
  type CandlestickData,
  type CandlestickMovingAverage,
  type CandlestickTradeMarker,
} from "@/components/valuecell/charts/candlestick-chart";
import {
  MarketIndicatorPanelChart,
  type RsiBollingerMode,
} from "@/components/valuecell/charts/market-indicator-panel";
import { PnlLineChart } from "@/components/valuecell/charts/pnl-line-chart";
import { ThresholdGauge } from "@/components/valuecell/charts/threshold-gauge";
import { useActiveRuleStrategyId } from "@/hooks/use-active-rule-strategy";
import { cn } from "@/lib/utils";
import {
  demoExecutionCheckedAtLabel,
  demoExecutionUnvaluedAssetCount,
} from "@/types/rule-strategy-demo-execution";
import type { AccountStrategyOverview } from "@/types/multi-strategy";

const currency = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const compactCurrency = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 2,
});

const MARKET_INTERVALS = [
  "1m",
  "3m",
  "5m",
  "15m",
  "30m",
  "1h",
  "4h",
  "1d",
  "1w",
  "1M",
] as const;
const MARKET_HISTORY_RANGES = [
  { value: "1d", label: "日", days: 1 },
  { value: "5d", label: "5日", days: 5 },
  { value: "1w", label: "周", days: 7 },
  { value: "1m", label: "月", days: 31 },
  { value: "1y", label: "年", days: 365 },
] as const;
const EQUITY_RANGES = [
  { value: "1d", label: "日" },
  { value: "5d", label: "5日" },
  { value: "1w", label: "周" },
  { value: "1m", label: "月" },
  { value: "1y", label: "年" },
  { value: "all", label: "全部" },
] as const;
const MARKET_INTERVAL_SECONDS: Record<
  (typeof MARKET_INTERVALS)[number],
  number
> = {
  "1m": 60,
  "3m": 180,
  "5m": 300,
  "15m": 900,
  "30m": 1_800,
  "1h": 3_600,
  "4h": 14_400,
  "1d": 86_400,
  "1w": 604_800,
  "1M": 2_592_000,
};

function toDashboardSymbol(symbol: string) {
  return symbol.replace("-", "/");
}

const MONITOR_STATE_LABELS: Record<string, string> = {
  candidate: "待准入",
  admitted: "已准入",
  held: "持仓保留",
  removed: "已移除",
};
const RISK_STATE_LABELS: Record<string, string> = {
  normal: "正常",
  only_reduce: "仅允许减仓",
  blocked: "已阻断",
  cooldown: "冷静期",
  warn: "预警",
  halted: "已暂停",
};
const REASON_CODE_LABELS: Record<string, string> = {
  shared_exchange_account_requires_dedicated_scope:
    "共享交易所账户未证明隔离，仅允许减仓或平仓。",
  program_entry_confirmed: "已确认策略入场条件。",
  program_entry_not_confirmed: "尚未确认策略入场条件。",
  program_exit_confirmed: "已确认策略离场条件。",
  program_exit_not_confirmed: "尚未确认策略离场条件。",
  insufficient_candle_history: "可用 K 线历史不足。",
  advanced_entry_confirmed: "已确认多周期入场规则。",
  advanced_entry_not_confirmed: "尚未确认多周期入场规则。",
  advanced_exit_confirmed: "已确认多周期离场规则。",
  no_exit_signal: "尚未确认离场信号。",
};

function displayReason(
  reasonCode?: string | null,
  reasonDetail?: string | null,
) {
  if (reasonDetail && /[\u4e00-\u9fff]/.test(reasonDetail)) return reasonDetail;
  if (reasonCode && REASON_CODE_LABELS[reasonCode]) {
    return REASON_CODE_LABELS[reasonCode];
  }
  return reasonDetail ? "系统暂未提供中文说明" : "暂无说明";
}
const ALLOCATION_STATE_LABELS: Record<string, string> = {
  available: "可分配",
  reserved: "已预留",
  occupied: "已占用",
  partially_released: "部分释放",
  released: "已释放",
  submission_unknown: "提交结果待对账",
  recovery_required: "等待对账恢复",
  blocked: "已阻断",
};

function formatQuote(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "—" : currency.format(value);
}

function StrategyAllocationCapEditor({
  allocation,
  credentialId,
}: {
  allocation: AccountStrategyOverview["allocator"]["allocations"][number];
  credentialId: string;
}) {
  const updateCap = useUpdateStrategyAllocationCap();
  const [reserved, setReserved] = useState(
    allocation.max_reserved_quote == null ? "" : String(allocation.max_reserved_quote),
  );
  const [occupied, setOccupied] = useState(
    allocation.max_occupied_quote == null ? "" : String(allocation.max_occupied_quote),
  );
  const save = async () => {
    const maxReservedQuote = Number(reserved);
    const maxOccupiedQuote = Number(occupied);
    if (!Number.isFinite(maxReservedQuote) || !Number.isFinite(maxOccupiedQuote) || maxReservedQuote < 0 || maxOccupiedQuote < 0 || maxOccupiedQuote > maxReservedQuote) {
      toast.error("请输入有效上限，已占用上限不能大于预留上限。");
      return;
    }
    try {
      await updateCap.mutateAsync({
        strategyId: allocation.strategy_id,
        credentialId,
        maxReservedQuote,
        maxOccupiedQuote,
      });
      toast.success("策略资金上限已更新。");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "资金上限更新失败。");
    }
  };
  return (
    <div className="flex min-w-52 flex-col gap-1.5">
      <div className="flex items-center gap-1">
        <Input aria-label="最大预留资金" className="h-7 w-24 text-xs" min={0} onChange={(event) => setReserved(event.target.value)} placeholder="预留上限" step="0.01" type="number" value={reserved} />
        <Input aria-label="最大占用资金" className="h-7 w-24 text-xs" min={0} onChange={(event) => setOccupied(event.target.value)} placeholder="占用上限" step="0.01" type="number" value={occupied} />
        <Button aria-label="保存策略资金上限" disabled={updateCap.isPending} onClick={() => void save()} size="icon" type="button" variant="outline">
          <CircleDollarSign className="size-3.5" />
        </Button>
      </div>
      <span className="text-[10px] text-muted-foreground">预留 / 占用上限（USDT）</span>
    </div>
  );
}

function TerminalValue({
  value,
  suffix = "",
  signed = false,
  compact = false,
  className,
}: {
  value: number;
  suffix?: string;
  signed?: boolean;
  compact?: boolean;
  className?: string;
}) {
  const initialValue = Number.isFinite(value) ? value : 0;
  const [displayValue, setDisplayValue] = useState(initialValue);
  const [showFlash, setShowFlash] = useState(false);
  const previousValueRef = useRef(initialValue);

  useEffect(() => {
    const target = Number.isFinite(value) ? value : 0;
    const start = previousValueRef.current;
    if (start === target) return;

    previousValueRef.current = target;
    setShowFlash(true);
    const startTime = performance.now();
    const animationDurationMs = 650;
    let frameId = 0;
    const animateValue = (time: number) => {
      const progress = Math.min((time - startTime) / animationDurationMs, 1);
      const easedProgress = 1 - (1 - progress) ** 3;
      setDisplayValue(start + (target - start) * easedProgress);
      if (progress < 1) frameId = requestAnimationFrame(animateValue);
    };
    frameId = requestAnimationFrame(animateValue);
    const flashTimer = window.setTimeout(() => setShowFlash(false), 3_000);

    return () => {
      cancelAnimationFrame(frameId);
      window.clearTimeout(flashTimer);
    };
  }, [value]);

  const formatter = compact ? compactCurrency : currency;
  const visibleValue = `${signed && displayValue >= 0 ? "+" : ""}${formatter.format(displayValue)}${suffix}`;

  return (
    <span
      className={cn(
        "terminal-number relative inline-block tabular-nums",
        className,
      )}
    >
      <span>{visibleValue}</span>
      {showFlash ? (
        <span aria-hidden className="terminal-value-flash">
          {visibleValue}
        </span>
      ) : null}
    </span>
  );
}

function KpiCard({
  icon: Icon,
  label,
  value,
  detail,
  trend,
}: {
  icon: typeof WalletCards;
  label: string;
  value: ReactNode;
  detail: string;
  trend?: "positive" | "negative" | "neutral";
}) {
  const isPositive = trend === "positive";
  const isNegative = trend === "negative";

  return (
    <Card className="dashboard-kpi overflow-hidden rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
      <CardContent className="relative p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="font-medium text-[11px] text-muted-foreground uppercase tracking-[0.08em]">
              {label}
            </p>
            <div
              className={cn(
                "dashboard-amount mt-2 truncate font-semibold text-2xl tabular-nums",
                isPositive && "text-emerald-500 dark:text-emerald-400",
                isNegative && "text-rose-500 dark:text-rose-400",
              )}
            >
              {value}
            </div>
          </div>
          <span
            className={cn(
              "grid size-9 shrink-0 place-items-center rounded-md border",
              isPositive
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-500"
                : isNegative
                  ? "border-rose-500/30 bg-rose-500/10 text-rose-500"
                  : "border-sky-500/30 bg-sky-500/10 text-sky-500",
            )}
          >
            <Icon className="size-4" />
          </span>
        </div>
        <p
          className="mt-3 whitespace-normal break-words text-muted-foreground text-xs"
          title={detail}
        >
          {detail}
        </p>
      </CardContent>
    </Card>
  );
}

export default function DashboardPage() {
  const { resolvedTheme, setTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const [strategyId, setActiveStrategyId, strategiesQuery] =
    useActiveRuleStrategyId();
  const [searchParams] = useSearchParams();
  const selectedBatchId = searchParams.get("batch_id");
  const strategyQuery = useRuleStrategy(strategyId);
  const { data: ruleStrategy, isError: ruleStrategyError } = strategyQuery;
  const monitorStateQuery = useRuleStrategyMonitorState(strategyId || undefined);
  const riskStateQuery = useRuleStrategyRiskState(strategyId || undefined);
  const monitorRows = monitorStateQuery.data ?? [];
  const monitorCounts = monitorRows.reduce<Record<string, number>>((counts, row) => {
    counts[row.state] = (counts[row.state] ?? 0) + 1;
    return counts;
  }, {});
  const execution = ruleStrategy?.config.execution;
  const isOkxDemo = execution?.environment === "okx_demo";
  const sharedCredentialId =
    (isOkxDemo ? execution?.sandbox_connection_id : undefined)
    ?? strategiesQuery.data?.find(
      (item) => item.config.execution.environment === "okx_demo" && item.config.execution.sandbox_connection_id,
    )?.config.execution.sandbox_connection_id;
  const sharedAccountQuery = useSharedAccountSummary(sharedCredentialId);
  const sharedAccountSummary = sharedAccountQuery.data;
  const strategyExecutionModeIsDemo = ruleStrategy ? isOkxDemo : undefined;
  const [demoOrdersPage, setDemoOrdersPage] = useState(1);
  const exportStrategy = useExportRuleStrategy();
  const demoExecutionQuery = useRuleStrategyDemoExecution(
    strategyId || undefined,
    isOkxDemo,
    demoOrdersPage,
    10,
    selectedBatchId,
  );
  const {
    data: demoExecution,
    isError: demoExecutionError,
    isFetching: demoExecutionLoading,
  } = demoExecutionQuery;
  const demoBalance = demoExecution?.account.data;
  const demoPositions = demoExecution?.positions.data.positions ?? [];
  const strategyPositions = demoExecution?.strategy_positions ?? [];
  const demoOrders = demoExecution?.orders ?? [];
  const demoOrdersPagination = demoExecution?.pagination;
  useEffect(() => {
    setDemoOrdersPage(1);
  }, [strategyId]);
  useEffect(() => {
    if (
      demoOrdersPagination &&
      demoOrdersPage > demoOrdersPagination.total_pages
    ) {
      setDemoOrdersPage(demoOrdersPagination.total_pages);
    }
  }, [demoOrdersPage, demoOrdersPagination]);
  const downloadAllDemoOrders = async () => {
    if (!strategyId || exportStrategy.isPending) return;
    try {
      const workbook = await exportStrategy.mutateAsync({ strategyId, batchId: selectedBatchId ?? undefined });
      const objectUrl = URL.createObjectURL(workbook.blob);
      const filename = workbook.filename?.toLowerCase().endsWith(".xlsx")
        ? workbook.filename
        : `${workbook.filename || "策略订单导出"}.xlsx`;
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
      toast.success("全部订单记录已开始下载。");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "订单导出失败，请稍后重试。");
    }
  };
  const demoPurchaseState = demoPurchaseStatePresentation(
    demoExecution?.trade_summary?.purchase_state,
  );
  const demoPnl = demoPnlPresentation(demoExecution?.pnl);
  const demoEquityCurve = buildDemoEquityCurve(demoExecution?.equity_curve);
  const demoCheckedAt = demoExecution
    ? demoExecutionCheckedAtLabel(demoExecution)
    : undefined;
  const demoCurveMode = "pnl" as const;
  const demoCheckedAtTime = demoCheckedAt
    ? new Date(demoCheckedAt).toLocaleTimeString("zh-CN")
    : "等待策略账户同步";
  const demoUnvaluedAssetCount = demoExecution
    ? demoExecutionUnvaluedAssetCount(demoExecution)
    : 0;
  const pnlCurveQuery = useRuleStrategyPnlCurve(
    strategyExecutionModeIsDemo === false ? strategyId || undefined : undefined,
    selectedBatchId,
  );
  const pnlCurve = pnlCurveQuery.data ?? [];
  const tradesQuery = useRuleStrategyTrades(
    strategyExecutionModeIsDemo === false ? strategyId || undefined : undefined,
    true,
    selectedBatchId,
  );
  const trades = tradesQuery.data ?? [];
  const evaluationsQuery = useRuleStrategyEvaluations(
    strategyId || undefined,
    selectedBatchId,
  );
  const evaluations = evaluationsQuery.data ?? [];
  const trackedSymbols = ruleStrategy?.config.symbols ?? [];
  const activeSymbols = isOkxDemo
    ? demoPositions.map((position) => position.symbol.replace("/", "-"))
    : Object.keys(ruleStrategy?.account.positions ?? {});
  const marketSymbols = useMemo(
    () => Array.from(new Set([...trackedSymbols, ...activeSymbols])),
    [activeSymbols, trackedSymbols],
  );
  const [selectedSymbol, setSelectedSymbol] = useState("BTC-USDT");
  const [selectedOrderId, setSelectedOrderId] = useState<string | null>(null);
  const [selectedEvaluationId, setSelectedEvaluationId] = useState<string | null>(null);
  const [marketInterval, setMarketInterval] =
    useState<(typeof MARKET_INTERVALS)[number]>("1h");
  const [historyRange, setHistoryRange] =
    useState<(typeof MARKET_HISTORY_RANGES)[number]["value"]>("5d");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [requestNowMs, setRequestNowMs] = useState(() => Date.now());
  const [equityRange, setEquityRange] = useState<
    (typeof EQUITY_RANGES)[number]["value"]
  >("1m");
  const [rsiMode, setRsiMode] = useState<RsiBollingerMode>("both");
  useEffect(() => {
    const timer = window.setInterval(() => {
      const refreshTargets = dashboardRefreshTargets(
        strategyId || undefined,
        strategyExecutionModeIsDemo,
      );
      const refreshes: Array<() => Promise<unknown>> = [];
      if (refreshTargets.length > 0) {
        refreshes.push(
          () => strategyQuery.refetch(),
          () => monitorStateQuery.refetch(),
          () => riskStateQuery.refetch(),
          () => evaluationsQuery.refetch(),
        );
        if (refreshTargets.includes("demo-execution")) {
          refreshes.push(() => demoExecutionQuery.refetch());
        } else if (refreshTargets.includes("pnl-curve")) {
          refreshes.push(
            () => pnlCurveQuery.refetch(),
            () => tradesQuery.refetch(),
          );
        }
      }
      if (sharedCredentialId) {
        refreshes.push(() => sharedAccountQuery.refetch());
      }
      void Promise.all(refreshes.map((refetch) => refetch()));
    }, 15_000);
    return () => window.clearInterval(timer);
  }, [
    demoExecutionQuery,
    evaluationsQuery,
    monitorStateQuery,
    pnlCurveQuery,
    riskStateQuery,
    sharedAccountQuery,
    sharedCredentialId,
    strategyExecutionModeIsDemo,
    strategyId,
    strategyQuery,
    tradesQuery,
  ]);
  const selectedIsAvailable = marketSymbols.includes(selectedSymbol);
  const effectiveSymbol =
    selectedIsAvailable || marketSymbols.length === 0
      ? selectedSymbol
      : marketSymbols[0];
  const fromTsMs = useMemo(() => {
    if (fromDate) return new Date(`${fromDate}T00:00:00Z`).getTime();
    const days =
      MARKET_HISTORY_RANGES.find((range) => range.value === historyRange)
        ?.days ?? 10;
    return requestNowMs - days * 24 * 60 * 60 * 1000;
  }, [fromDate, historyRange, requestNowMs]);
  const toTsMs = useMemo(
    () =>
      toDate ? new Date(`${toDate}T23:59:59.999Z`).getTime() : requestNowMs,
    [requestNowMs, toDate],
  );
  const invalidDateRange = fromTsMs > toTsMs;
  const lookback = useMemo(
    () =>
      Math.min(
        5_000,
        Math.max(
          1,
          Math.ceil(
            (toTsMs - fromTsMs) /
              (MARKET_INTERVAL_SECONDS[marketInterval] * 1000),
          ) + 2,
        ),
      ),
    [fromTsMs, marketInterval, toTsMs],
  );
  useEffect(() => {
    if (ruleStrategy?.config.interval)
      setMarketInterval(ruleStrategy.config.interval);
  }, [ruleStrategy?.config.interval]);
  const {
    data: marketData,
    isFetching: marketLoading,
    isPlaceholderData: marketDataIsPrevious,
    isError: marketError,
  } = useGetCryptoMarketIndicators({
    symbols: [effectiveSymbol],
    interval: marketInterval,
    lookback,
    fromTsMs,
    toTsMs,
    enabled: !invalidDateRange,
  });
  const market = marketData?.symbols[0];
  const marketFailure = marketData?.failed_symbols[effectiveSymbol];
  const pageLoading = shouldShowDashboardPageLoading({
    strategies: strategiesQuery.data,
    strategyId,
    ruleStrategy,
    demoExecution,
    hasError:
      strategiesQuery.isError || ruleStrategyError || demoExecutionError,
  });
  const candlestickLoading = shouldShowCandlestickLoading(
    marketLoading,
    marketData,
  );

  const candles: CandlestickData[] =
    market?.candles.map((candle) => ({
      time: new Date(candle.ts).toISOString(),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
    })) ?? [];
  const movingAverages: CandlestickMovingAverage[] = market?.indicators.length
    ? ["ma5", "ma20", "ma60"].map((key, index) => ({
        name: key.toUpperCase(),
        values: market.indicators.map((indicator) => indicator.ma[key] ?? null),
        color: ["#fbbf24", "#38bdf8", "#c084fc"][index],
      }))
    : [];
  const selectedOrder = demoOrders.find((order) => order.id === selectedOrderId);
  const selectedEvaluation = evaluations.find((evaluation) =>
    evaluation.evaluation_id === selectedEvaluationId ||
    evaluation.evaluation_id === selectedOrder?.evaluation_id,
  );
  const tradeMarkers = useMemo<CandlestickTradeMarker[]>(() => {
    const demoMarkers = selectedOrder ? [selectedOrder] : demoOrders;
    if (isOkxDemo) {
      return demoMarkers.flatMap((order) => {
        const price = Number(order.average_fill_price);
        const time = order.filled_at ?? order.updated_at ?? order.created_at;
        return Number.isFinite(price) && price > 0 && time
          ? [{ time, price, side: order.side, label: order.side === "buy" ? "买入" : "卖出" }]
          : [];
      });
    }
    return trades
      .filter((trade) => !selectedEvaluationId || trade.evaluation_id === selectedEvaluationId)
      .map((trade) => ({ time: trade.evaluated_at, price: trade.price, side: trade.action === "sell" || trade.action === "close" ? "sell" : "buy", label: trade.action === "sell" || trade.action === "close" ? "卖出" : "买入" }));
  }, [demoOrders, isOkxDemo, selectedEvaluationId, selectedOrder, trades]);
  const account = ruleStrategy?.account;
  const demoUsdt = Number(
    demoBalance?.balances.find((balance) => balance.currency === "USDT")
      ?.free ?? 0,
  );
  const displayEquity = isOkxDemo
    ? (demoBalance?.total_usdt_value ?? 0)
    : (account?.equity_quote ?? 0);
  const displayCash = isOkxDemo ? demoUsdt : (account?.quote_balance ?? 0);
  const liveEquityCurve = pnlCurve;
  const holdingRows = useMemo(
    () =>
      isOkxDemo
        ? strategyPositions.map((position) => ({
            symbol: position.symbol.replace("/", "-"),
            position: {
              quantity: Number(position.quantity),
              entry_price:
                position.entry_price == null ? null : Number(position.entry_price),
              mark_price:
                position.mark_price == null ? null : Number(position.mark_price),
            },
            value:
              position.notional_usdt == null
                ? null
                : Number(position.notional_usdt),
            profit:
              position.unrealized_pnl_usdt == null
                ? null
                : Number(position.unrealized_pnl_usdt),
          }))
        : Object.entries(account?.positions ?? {}).map(([symbol, position]) => {
            const value = position.mark_price == null
              ? null
              : position.quantity * position.mark_price;
            const profit = position.mark_price == null
              ? null
              : position.quantity * (position.mark_price - position.entry_price);
            return { symbol, position, value, profit };
          }),
    [account?.positions, isOkxDemo, strategyPositions],
  );
  const pnl =
    (account?.realized_pnl_quote ?? 0) + (account?.unrealized_pnl_quote ?? 0);
  const pnlPercent =
    account && account.initial_capital_quote > 0
      ? (pnl / account.initial_capital_quote) * 100
      : 0;
  const invested = isOkxDemo
    ? Math.max(displayEquity - displayCash, 0)
    : account
      ? account.equity_quote - account.quote_balance
      : 0;
  const latestRsi =
    market?.indicators[market.indicators.length - 1]?.rsi ?? null;
  const rsiDescription =
    latestRsi === null
      ? "等待可用行情"
      : latestRsi <= 30
        ? "超卖区：低于 30"
        : latestRsi >= 70
          ? "超买区：高于 70"
          : "中性区：30–70";
  const capitalUtilization =
    displayEquity > 0
      ? Math.min(
          Math.max((Math.max(invested, 0) / displayEquity) * 100, 0),
          100,
        )
      : null;
  const utilizationDescription = isOkxDemo
    ? "OKX Demo 已估值资产中非可用 USDT 的比例"
    : capitalUtilization === null
      ? "等待策略账户"
      : "已投入资金 ÷ 当前组合权益";
  const recentSignals = evaluations
    .filter((item) => item.action !== "no_op")
    .slice(0, 5);
  const latestEvaluation = evaluations[0];
  const enabledIndicators = [
    ruleStrategy?.config.moving_average.enabled
      ? `均线 ${ruleStrategy.config.moving_average.short_window}/${ruleStrategy.config.moving_average.long_window}`
      : null,
    ruleStrategy?.config.rsi.enabled
      ? `RSI ${ruleStrategy.config.rsi.period}`
      : null,
    ruleStrategy?.config.bollinger.enabled
      ? `布林带 ${ruleStrategy.config.bollinger.period}`
      : null,
    ruleStrategy?.config.momentum_macd.enabled
      ? `MACD ${ruleStrategy.config.momentum_macd.macd_fast_window}/${ruleStrategy.config.momentum_macd.macd_slow_window}/${ruleStrategy.config.momentum_macd.macd_signal_window}`
      : null,
  ].filter((item): item is string => item !== null);
  const recentlyScanned = evaluations.slice(0, 8);
  const requestedCapital = latestEvaluation?.sizing?.requested_quote ?? 0;
  const latestConfirmation = latestEvaluation?.entry_confirmation;
  const latestConditionSummary =
    latestEvaluation?.condition_summary ??
    (latestConfirmation
      ? {
          matched: latestConfirmation.passed,
          total: latestConfirmation.enabled,
          required: latestConfirmation.required,
          available: latestConfirmation.available,
        }
      : null);
  const latestConditions = latestEvaluation?.conditions ?? [];
  const { steps: funnelSteps, firstBlocker } = buildDashboardFunnel({
    strategyRunning: ruleStrategy?.status === "running",
    evaluation: latestEvaluation,
  });


  if (pageLoading) {
    return (
      <div
        aria-busy="true"
        className="grid size-full place-items-center text-muted-foreground"
      >
        <div className="flex items-center gap-2 text-sm">
          <RefreshCw className="size-4 animate-spin" />
          正在加载仪表盘…
        </div>
      </div>
    );
  }

  if (strategiesQuery.isError || ruleStrategyError || demoExecutionError) {
    return (
      <div className="grid size-full place-items-center p-8">
        <Alert className="max-w-xl border-destructive/40" variant="destructive">
          <AlertTriangle />
          <AlertTitle>仪表盘数据加载失败</AlertTitle>
          <AlertDescription>
            无法确认当前策略或模拟账户的权威状态。请稍后重试，页面不会先显示 0 作为真实账户数据。
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="scroll-container dashboard-shell flex size-full flex-col">
      <div className="mx-auto flex w-full max-w-[1800px] flex-col gap-4 p-4 lg:p-5">
        <header className="dashboard-header flex flex-col gap-4 rounded-lg border border-sky-400/15 px-4 py-4 md:flex-row md:items-center md:justify-between md:px-5">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="live-pulse size-2 rounded-full bg-emerald-400" />
              <span className="font-semibold text-[11px] text-sky-500 tracking-[0.14em] dark:text-sky-300">
                {isOkxDemo ? "OKX DEMO 模拟盘终端" : "纸面交易终端"}
              </span>
              <Badge
                variant="outline"
                className="border-sky-500/30 bg-sky-500/10 text-sky-600 dark:text-sky-300"
              >
                {isOkxDemo
                  ? demoExecutionError
                    ? "OKX Demo 数据读取失败"
                    : demoExecutionLoading
                      ? "同步 OKX Demo 策略账户中"
                      : "OKX Demo · 共享交易所账户"
                  : ruleStrategy?.status === "running"
                    ? "策略扫描中"
                    : "策略待命"}
              </Badge>
            </div>
            <h1 className="dashboard-title mt-2 font-semibold text-2xl tracking-[0.02em]">
              市场指挥中心
            </h1>
            <p className="mt-1 text-muted-foreground text-sm">
              {ruleStrategy
                ? `${ruleStrategy.name}，正在监测 ${trackedSymbols.length} 个市场`
                : "BTC 市场数据已就绪。配置策略后即可开启模拟执行。"}
            </p>
            {isOkxDemo ? (
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <span className="text-muted-foreground text-xs">真实交易状态</span>
                <Badge
                  variant="outline"
                  className={cn(
                    demoPurchaseState.tone === "positive" && "border-emerald-500/30 bg-emerald-500/10 text-emerald-600",
                    demoPurchaseState.tone === "warning" && "border-amber-500/30 bg-amber-500/10 text-amber-700",
                    demoPurchaseState.tone === "negative" && "border-rose-500/30 bg-rose-500/10 text-rose-600",
                  )}
                >
                  {demoPurchaseState.label}
                </Badge>
                <span className="text-muted-foreground text-xs">
                  订单 {demoExecution?.trade_summary?.order_count ?? demoOrders.length} 笔 · 已成交 {demoExecution?.trade_summary?.filled_order_count ?? "—"} · 部分成交 {demoExecution?.trade_summary?.partially_filled_order_count ?? "—"} · 待远端对账 {demoExecution?.trade_summary?.submission_unknown_orders ?? demoExecution?.trade_summary?.unknown_order_count ?? "—"} · 失败 {demoExecution?.trade_summary?.failed_order_count ?? "—"}
                </span>
              </div>
            ) : null}
          </div>
          <div className="flex items-center gap-2 self-start md:self-auto">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  aria-label={isDark ? "切换为浅色主题" : "切换为深色主题"}
                  onClick={() => setTheme(isDark ? "light" : "dark")}
                  size="icon"
                  type="button"
                  variant="outline"
                >
                  {isDark ? <Sun /> : <Moon />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {isDark ? "切换为浅色主题" : "切换为深色主题"}
              </TooltipContent>
            </Tooltip>
          </div>
        </header>
        <DashboardStrategyManagement />
        {sharedCredentialId ? (
          <section aria-label="共享账户概览">
            <Card className="dashboard-panel overflow-hidden rounded-lg border-sky-500/20 bg-card/90 py-0 shadow-none">
              <CardHeader className="gap-1 border-border/70 border-b px-5 py-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <WalletCards className="size-4 text-sky-500" />
                      共享账户概览
                    </CardTitle>
                    <CardDescription>
                      钱包权威总额与策略归属分配分开呈现，不将当前策略视为整个账户
                    </CardDescription>
                  </div>
                  <Badge
                    className={cn(
                      "shrink-0",
                      sharedAccountQuery.isError &&
                        "border-rose-500/30 bg-rose-500/10 text-rose-600 dark:text-rose-300",
                      !sharedAccountQuery.isError &&
                        sharedAccountSummary?.data_complete !== false &&
                        sharedAccountSummary?.wallet.sync_status === "healthy" &&
                        sharedAccountSummary?.wallet.attribution_status === "complete" &&
                        "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300",
                      sharedAccountSummary != null &&
                        (sharedAccountSummary.data_complete === false ||
                          sharedAccountSummary.wallet.sync_status !== "healthy" ||
                          sharedAccountSummary.wallet.attribution_status !== "complete") &&
                        "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300",
                    )}
                    variant="outline"
                  >
                    {sharedAccountQuery.isError
                      ? "账户数据不可用"
                      : sharedAccountQuery.isFetching && !sharedAccountSummary
                        ? "正在同步"
                        : !sharedAccountSummary
                          ? "等待账户数据"
                          : sharedAccountSummary.wallet.sync_status === "unavailable"
                            ? "钱包不可用"
                            : sharedAccountSummary.wallet.sync_status === "stale"
                              ? "钱包数据延迟"
                              : sharedAccountSummary.wallet.attribution_status !== "complete"
                                ? "归因不完整"
                              : sharedAccountSummary?.data_complete === false
                                ? "数据不完整"
                                : "已同步"}
                  </Badge>
                  {sharedAccountSummary ? (
                    <Badge
                      className={cn(
                        "shrink-0",
                        sharedAccountSummary.execution_gate.status === "ready" &&
                          "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300",
                        sharedAccountSummary.execution_gate.status === "protected" &&
                          "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300",
                        sharedAccountSummary.execution_gate.status === "blocked" &&
                          "border-rose-500/30 bg-rose-500/10 text-rose-600 dark:text-rose-300",
                      )}
                      variant="outline"
                      title={sharedAccountSummary.execution_gate.reasons.join("；") || "可进行新的 Demo 开仓"}
                    >
                      {sharedAccountSummary.execution_gate.status === "ready"
                        ? "可开仓"
                        : sharedAccountSummary.execution_gate.status === "protected"
                          ? "只读保护"
                          : "开仓已阻断"}
                    </Badge>
                  ) : null}
                </div>
              </CardHeader>
              <CardContent className="p-4 sm:p-5">
                {sharedAccountQuery.isError ? (
                  <div className="rounded-md border border-rose-500/25 bg-rose-500/5 px-4 py-3 text-sm">
                    <p className="font-medium text-rose-700 dark:text-rose-300">
                      暂时无法读取共享钱包与分配数据
                    </p>
                    <p className="mt-1 text-muted-foreground text-xs">
                      请稍后重试；在数据恢复前不会用策略账户数值替代钱包权威总额。
                    </p>
                  </div>
                ) : sharedAccountQuery.isFetching && !sharedAccountSummary ? (
                  <div aria-busy="true" className="flex items-center gap-2 py-8 text-muted-foreground text-sm">
                    <RefreshCw className="size-4 animate-spin" /> 正在同步共享钱包与策略分配…
                  </div>
                ) : !sharedAccountSummary ? (
                  <p className="py-8 text-center text-muted-foreground text-sm">
                    共享账户暂无可用数据。
                  </p>
                ) : (
                  <>
                    <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                      {[
                        {
                          label: "钱包总权益 · 权威",
                          value: formatQuote(sharedAccountSummary.wallet.total_equity_quote),
                          detail: "OKX 钱包同步值，不代表当前策略余额",
                          tone: "text-sky-600 dark:text-sky-300",
                        },
                        {
                          label: "钱包可用余额 · 权威",
                          value: formatQuote(sharedAccountSummary.wallet.available_quote),
                          detail: "可用资金，以钱包为准",
                          tone: "text-sky-600 dark:text-sky-300",
                        },
                        {
                          label: "策略可分配余额 · allocator",
                          value: formatQuote(sharedAccountSummary.allocator.available_for_strategies_quote),
                          detail: "扣除当前未结算预留后的可开仓资金",
                          tone: "text-cyan-600 dark:text-cyan-300",
                        },
                        {
                          label: "策略归属 PnL · 归因",
                          value: formatQuote(sharedAccountSummary.strategy_pnl_total_quote),
                          detail: "所有策略归属盈亏合计；不含纸面账本",
                          tone: sharedAccountSummary.strategy_pnl_total_quote == null
                            ? "text-muted-foreground"
                            : sharedAccountSummary.strategy_pnl_total_quote >= 0
                              ? "text-emerald-600 dark:text-emerald-300"
                              : "text-rose-600 dark:text-rose-300",
                        },
                        {
                          label: "未归因权益 · 钱包",
                          value: formatQuote(sharedAccountSummary.wallet.unassigned_equity_quote),
                          detail: "钱包中尚不能归属到策略的部分，不计入策略 PnL",
                          tone: "text-amber-600 dark:text-amber-300",
                        },
                        {
                          label: "钱包 − 策略差额",
                          value: formatQuote(sharedAccountSummary.wallet_strategy_reconciliation_delta_quote),
                          detail: "用于核对，非当前策略账户权益",
                          tone: "text-amber-600 dark:text-amber-300",
                        },
                      ].map((metric) => (
                        <div className="rounded-md border border-border/70 bg-muted/20 px-3 py-3" key={metric.label}>
                          <p className="font-medium text-[10px] text-muted-foreground uppercase tracking-[0.08em]">
                            {metric.label}
                          </p>
                          <p className={cn("mt-2 font-semibold text-lg tabular-nums", metric.tone)}>
                            {metric.value} <span className="font-normal text-xs">USDT</span>
                          </p>
                          <p className="mt-1 text-[11px] text-muted-foreground">{metric.detail}</p>
                        </div>
                      ))}
                    </div>
                    {sharedAccountSummary.data_complete === false ||
                    sharedAccountSummary.wallet.sync_status !== "healthy" ||
                    sharedAccountSummary.wallet.attribution_status !== "complete" ? (
                      <div className="mt-3 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-amber-800 text-xs dark:text-amber-200">
                        <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                        <span>
                          {sharedAccountSummary.incomplete_reason ??
                            (sharedAccountSummary.wallet.sync_status !== "healthy"
                              ? "钱包同步状态异常，权威余额可能暂时不可用。"
                              : "部分策略归因尚未完成，归属 PnL 仅供参考。")}
                        </span>
                      </div>
                    ) : null}
                    {sharedAccountSummary.execution_gate.reasons.length > 0 ? (
                      <div className="mt-3 rounded-md border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-rose-700 text-xs dark:text-rose-300">
                        <span className="font-medium">新开仓门禁：</span>
                        {sharedAccountSummary.execution_gate.reasons.join("；")}
                      </div>
                    ) : null}
                    <div className="mt-5 border-border/70 border-t pt-4">
                      <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
                        <div>
                          <h3 className="font-medium text-sm">OKX 共享钱包权益曲线</h3>
                          <p className="mt-0.5 text-muted-foreground text-xs">
                            仅来自后台持久化的钱包快照，用于核对四策略共同作用后的账户总金额变化
                          </p>
                        </div>
                        <span className="text-muted-foreground text-xs">
                          {sharedAccountSummary.wallet_equity_curve.points.length} 个快照
                        </span>
                      </div>
                      {sharedAccountSummary.wallet_equity_curve.status === "available" && sharedAccountSummary.wallet_equity_curve.points.length > 0 ? (
                        <PnlLineChart
                          data={sharedAccountSummary.wallet_equity_curve.points}
                          height={220}
                          mode="equity"
                          range={equityRange}
                          theme={isDark ? "dark" : "light"}
                        />
                      ) : (
                        <div className="grid h-40 place-items-center text-center text-muted-foreground text-sm">
                          尚无可用的钱包权益快照，后台同步成功后自动显示。
                        </div>
                      )}
                      {sharedAccountSummary.wallet_equity_curve.points.length === 1 ? (
                        <p className="mt-2 text-center text-muted-foreground text-xs">
                          当前只有一个账户快照，下一次同步后将形成变化曲线。
                        </p>
                      ) : null}
                    </div>
                    <div className="mt-5 border-border/70 border-t pt-4">
                      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                        <div>
                          <h3 className="font-medium text-sm">四策略并发运行矩阵</h3>
                          <p className="mt-0.5 text-muted-foreground text-xs">
                            每一行是一套独立策略；规则、批次和成交归属隔离，资金通过共享账户 allocator 竞争与释放
                          </p>
                        </div>
                        <div className="text-right text-xs">
                          <p className="font-medium text-foreground">
                            {(strategiesQuery.data ?? []).filter((strategy) => strategy.status === "running").length} / {(strategiesQuery.data ?? []).length} 运行中
                          </p>
                          <p className="text-muted-foreground">
                            账户利用率 {(sharedAccountSummary.allocator.account_utilization_ratio * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <div className="overflow-x-auto rounded-md border border-border/70">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>策略</TableHead>
                              <TableHead>状态</TableHead>
                              <TableHead className="text-right">预留</TableHead>
                              <TableHead className="text-right">占用</TableHead>
                              <TableHead className="text-right">利用率</TableHead>
                              <TableHead className="text-right">已释放</TableHead>
                              <TableHead className="text-right">已实现</TableHead>
                              <TableHead className="text-right">未实现</TableHead>
                              <TableHead className="text-right">净 PnL / 收益率</TableHead>
                              <TableHead>交易统计</TableHead>
                              <TableHead>资金上限</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {sharedAccountSummary.allocator.allocations.length === 0 ? (
                              <TableRow>
                                <TableCell className="py-7 text-center text-muted-foreground" colSpan={11}>
                                  暂无策略分配记录。
                                </TableCell>
                              </TableRow>
                            ) : (
                              sharedAccountSummary.allocator.allocations.map((allocation) => (
                                <TableRow key={allocation.strategy_id}>
                                <TableCell>
                                    <div className="flex min-w-40 flex-col gap-1">
                                      <span className="font-medium">{strategiesQuery.data?.find((item) => item.strategy_id === allocation.strategy_id)?.name ?? allocation.kind}</span>
                                      <span className="text-[10px] text-muted-foreground">{allocation.kind}</span>
                                      <span className={cn("text-[10px]", allocation.status === "running" ? "text-emerald-500" : "text-muted-foreground")}>
                                        {allocation.status === "running" ? "运行中" : allocation.status === "paused" ? "已暂停" : allocation.status === "archived" ? "已归档" : "已停止"}
                                        {allocation.current_batch_id ? ` · 批次 ${allocation.current_batch_id}` : " · 尚无当前批次"}
                                      </span>
                                      <span className="font-mono text-[10px] text-muted-foreground" title={allocation.strategy_id}>
                                        {allocation.strategy_id}
                                      </span>
                                      <Link
                                        className="w-fit text-[10px] text-sky-600 hover:underline dark:text-sky-300"
                                        onClick={() => setActiveStrategyId(allocation.strategy_id)}
                                        to={`/trades?strategy=${encodeURIComponent(allocation.strategy_id)}`}
                                      >
                                        查看交易明细与条件原因
                                      </Link>
                                      {allocation.strategy_id === strategyId ? (
                                        <Badge className="w-fit border-sky-500/30 bg-sky-500/10 text-[10px] text-sky-600 dark:text-sky-300" variant="outline">
                                          当前选择
                                        </Badge>
                                      ) : null}
                                    </div>
                                  </TableCell>
                                  <TableCell>
                                    <Badge
                                      className={cn(
                                        "text-[10px]",
                                        (allocation.allocation_state === "blocked" || allocation.allocation_state === "submission_unknown" || allocation.allocation_state === "recovery_required") &&
                                          "border-rose-500/30 bg-rose-500/10 text-rose-600 dark:text-rose-300",
                                        allocation.allocation_state !== "blocked" && allocation.allocation_state !== "submission_unknown" && allocation.allocation_state !== "recovery_required" &&
                                          "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300",
                                      )}
                                      variant="outline"
                                    >
                                      {ALLOCATION_STATE_LABELS[allocation.allocation_state] ?? allocation.allocation_state}{allocation.lifecycle_reason ? ` · ${allocation.lifecycle_reason}` : ""}
                                    </Badge>
                                  </TableCell>
                                  <TableCell className="text-right tabular-nums">{formatQuote(allocation.reserved_quote)}</TableCell>
                                  <TableCell className="text-right tabular-nums">{formatQuote(allocation.occupied_quote)}</TableCell>
                                  <TableCell className="text-right tabular-nums">{(allocation.utilization_ratio * 100).toFixed(1)}%</TableCell>
                                  <TableCell className="text-right tabular-nums">{formatQuote(allocation.released_quote)}</TableCell>
                                  <TableCell className="text-right tabular-nums">{formatQuote(allocation.realized_pnl_quote)}</TableCell>
                                  <TableCell className="text-right tabular-nums">{formatQuote(allocation.unrealized_pnl_quote)}</TableCell>
                                  <TableCell className={cn("text-right tabular-nums", allocation.net_pnl_quote == null ? "text-muted-foreground" : allocation.net_pnl_quote >= 0 ? "text-emerald-600 dark:text-emerald-300" : "text-rose-600 dark:text-rose-300")}>
                                    {(() => { const pnl = allocationPnlPresentation(allocation.net_pnl_quote, allocation.return_rate_pct); return <><div>{pnl.value} USDT</div><div className="text-xs">收益率 {pnl.returnRate}</div></>; })()}
                                  </TableCell>
                                  <TableCell className="min-w-56 text-xs tabular-nums">
                                    <div>成交 {allocation.fill_count} 笔 · 完整交易 {allocation.completed_trade_count} 次</div>
                                    <div className="text-muted-foreground">
                                      胜率 {allocation.win_rate == null ? "—" : `${(allocation.win_rate * 100).toFixed(1)}%`} · 周转率 {allocation.turnover_ratio == null ? "—" : `${(allocation.turnover_ratio * 100).toFixed(1)}%`}
                                    </div>
                                    <div className="text-muted-foreground">成交额 {formatQuote(allocation.turnover_quote)} · 手续费 {formatQuote(allocation.fee_quote)} USDT</div>
                                  </TableCell>
                                  <TableCell>
                                    <StrategyAllocationCapEditor
                                      allocation={allocation}
                                      credentialId={sharedCredentialId}
                                    />
                                  </TableCell>
                                </TableRow>
                              ))
                            )}
                          </TableBody>
                        </Table>
                      </div>
                      {sharedAccountSummary.allocator.unallocated_strategies.length > 0 ? (
                        <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
                          <p className="font-medium text-xs">未纳入该共享账户资金池的策略（只读）</p>
                          <p className="mt-1 text-[10px] text-muted-foreground">
                            这些策略存在，但不是该钱包的资金分配对象，因此不计入上方的预留、占用与利用率。这不代表策略未在运行。
                          </p>
                          <ul className="mt-2 grid gap-1.5">
                            {sharedAccountSummary.allocator.unallocated_strategies.map((item) => (
                              <li className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[11px]" key={item.strategy_id}>
                                <span className="font-medium">{item.name}</span>
                                <span className="text-muted-foreground">{item.kind}</span>
                                <Badge className="text-[10px]" variant="outline">
                                  {item.status === "running" ? "运行中" : item.status === "paused" ? "已暂停" : "已停止"}
                                </Badge>
                                <span className="text-muted-foreground">
                                  {item.environment === "paper" ? "Paper 独立账本" : item.environment === "okx_demo" ? "OKX Demo" : "未绑定环境"}
                                </span>
                                <span className="text-amber-600 dark:text-amber-300">{item.reason}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </section>
        ) : null}
        <section
          className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]"
          aria-label="策略监控与风险"
        >
          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <CardHeader className="gap-1 border-border/70 border-b px-5 py-4">
              <CardTitle className="text-base">监控池</CardTitle>
              <CardDescription>只扫描已准入或持仓保留的币种</CardDescription>
            </CardHeader>
            <CardContent className="p-5">
              <div className="flex flex-wrap gap-2 text-xs">
                {(["candidate", "admitted", "held", "removed"] as const).map(
                  (state) => (
                    <Badge
                      key={state}
                      variant={
                        state === "admitted" || state === "held"
                          ? "default"
                          : "outline"
                      }
                    >
                      {MONITOR_STATE_LABELS[state]} {monitorCounts[state] ?? 0}
                    </Badge>
                  ),
                )}
              </div>
              <div className="mt-3 max-h-28 space-y-2 overflow-auto text-xs">
                {monitorRows.map((row) => (
                  <div
                    className="flex items-center justify-between gap-3"
                    key={row.symbol}
                  >
                    <span className="font-mono">{row.symbol}</span>
                    <span className="text-muted-foreground">
                      {displayReason(row.reason_code, row.reason_detail)}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <CardHeader className="gap-1 border-border/70 border-b px-5 py-4">
              <CardTitle className="text-base">账户风险</CardTitle>
              <CardDescription>下单前读取并持续刷新的风险状态</CardDescription>
            </CardHeader>
            <CardContent className="p-5 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">状态</span>
                <Badge
                  variant={
                    riskStateQuery.data?.state === "normal"
                      ? "default"
                      : "destructive"
                  }
                >
                  {riskStateQuery.data?.state
                    ? (RISK_STATE_LABELS[riskStateQuery.data.state] ?? "未知状态")
                    : "读取中"}
                </Badge>
              </div>
              <p className="mt-3 text-muted-foreground text-xs">
                回撤 {formatOptionalPercent(riskStateQuery.data?.current_drawdown_pct)}
              </p>
              <p className="mt-1 text-muted-foreground text-xs">
                {displayReason(
                  riskStateQuery.data?.reason_code,
                  riskStateQuery.data?.reason_detail,
                )}
              </p>
            </CardContent>
          </Card>
        </section>

      </div>
    </div>
  );
}
