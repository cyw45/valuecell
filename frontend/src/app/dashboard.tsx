import {
  ArrowUpRight,
  BarChart3,
  CandlestickChart,
  CircleDollarSign,
  Clock3,
  Cpu,
  Crosshair,
  Layers3,
  Moon,
  RadioTower,
  RefreshCw,
  Settings2,
  Sun,
  TrendingDown,
  TrendingUp,
  WalletCards,
} from "lucide-react";
import { useTheme } from "next-themes";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useGetCryptoMarketIndicators } from "@/api/crypto-market";
import {
  useRuleStrategy,
  useRuleStrategyEvaluations,
  useRuleStrategyPnlCurve,
  useRuleStrategyTrades,
} from "@/api/rule-strategy";
import {
  useSandboxBalance,
  useSandboxConnections,
  useSandboxPositions,
} from "@/api/sandbox-exchange";
import { RuleStrategyConfiguration } from "@/app/strategies/strategies";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import CandlestickChartComponent, {
  type CandlestickData,
  type CandlestickMovingAverage,
} from "@/components/valuecell/charts/candlestick-chart";
import {
  MarketIndicatorPanelChart,
  type RsiBollingerMode,
} from "@/components/valuecell/charts/market-indicator-panel";
import { PnlLineChart } from "@/components/valuecell/charts/pnl-line-chart";
import { ThresholdGauge } from "@/components/valuecell/charts/threshold-gauge";
import { useActiveRuleStrategyId } from "@/hooks/use-active-rule-strategy";
import { cn } from "@/lib/utils";

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
] as const;
const MARKET_HISTORY_RANGES = [
  { value: "10d", label: "10D", days: 10 },
  { value: "30d", label: "30D", days: 30 },
  { value: "90d", label: "90D", days: 90 },
  { value: "1y", label: "1Y", days: 365 },
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
};

function toDashboardSymbol(symbol: string) {
  return symbol.replace("-", "/");
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
          className="mt-3 truncate text-muted-foreground text-xs"
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
  const [strategyId] = useActiveRuleStrategyId();
  const { data: ruleStrategy } = useRuleStrategy(strategyId);
  const execution = ruleStrategy?.config.execution;
  const isOkxDemo = execution?.environment === "okx_demo";
  const connectionId = isOkxDemo ? execution?.sandbox_connection_id : undefined;
  const { data: sandboxConnections = [] } = useSandboxConnections();
  const demoConnection = sandboxConnections.find(
    (connection) => connection.id === connectionId,
  );
  const {
    data: demoBalance,
    isError: demoBalanceError,
    isFetching: demoBalanceLoading,
  } = useSandboxBalance(connectionId, isOkxDemo);
  const { data: demoPositionsData, isError: demoPositionsError } =
    useSandboxPositions(connectionId, isOkxDemo);
  const demoPositions = demoPositionsData?.positions ?? [];
  const { data: pnlCurve } = useRuleStrategyPnlCurve(
    isOkxDemo ? undefined : strategyId || undefined,
  );
  const { data: trades = [] } = useRuleStrategyTrades(
    isOkxDemo ? undefined : strategyId || undefined,
  );
  const { data: evaluations = [] } = useRuleStrategyEvaluations(
    strategyId || undefined,
  );
  const trackedSymbols = ruleStrategy?.config.symbols ?? [];
  const activeSymbols = Object.keys(ruleStrategy?.account.positions ?? {});
  const marketSymbols =
    activeSymbols.length > 0 ? activeSymbols : trackedSymbols;
  const [selectedSymbol, setSelectedSymbol] = useState("BTC-USDT");
  const [marketInterval, setMarketInterval] =
    useState<(typeof MARKET_INTERVALS)[number]>("1h");
  const [historyRange, setHistoryRange] =
    useState<(typeof MARKET_HISTORY_RANGES)[number]["value"]>("10d");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [requestNowMs, setRequestNowMs] = useState(() => Date.now());
  const [rsiMode, setRsiMode] = useState<RsiBollingerMode>("both");
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
  const useSharedSnapshot =
    marketInterval === "1h" && historyRange === "10d" && !fromDate && !toDate;
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
    isError: marketError,
  } = useGetCryptoMarketIndicators({
    symbols: [effectiveSymbol],
    interval: marketInterval,
    lookback: useSharedSnapshot ? 240 : lookback,
    fromTsMs: useSharedSnapshot ? undefined : fromTsMs,
    toTsMs: useSharedSnapshot ? undefined : toTsMs,
    enabled: !invalidDateRange,
  });
  const market = marketData?.symbols[0];
  const marketFailure = marketData?.failed_symbols[effectiveSymbol];

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
  const account = ruleStrategy?.account;
  const demoUsdt = Number(
    demoBalance?.balances.find((balance) => balance.currency === "USDT")
      ?.free ?? 0,
  );
  const displayEquity = isOkxDemo
    ? (demoBalance?.total_usdt_value ?? 0)
    : (account?.equity_quote ?? 0);
  const displayCash = isOkxDemo ? demoUsdt : (account?.quote_balance ?? 0);
  const holdingRows = useMemo(
    () =>
      isOkxDemo
        ? demoPositions.map((position) => ({
            symbol: position.symbol.replace("/", "-"),
            position: {
              quantity: position.quantity,
              entry_price: position.mark_price ?? 0,
              mark_price: position.mark_price ?? 0,
            },
            value: position.notional_usdt ?? 0,
            profit: 0,
          }))
        : Object.entries(account?.positions ?? {}).map(([symbol, position]) => {
            const value = position.quantity * position.mark_price;
            const profit =
              position.quantity * (position.mark_price - position.entry_price);
            return { symbol, position, value, profit };
          }),
    [account?.positions, demoPositions, isOkxDemo],
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
  const requestedCapital = latestEvaluation?.sizing.requested_quote ?? 0;

  const scrollToStrategyConfiguration = useCallback(() => {
    document.getElementById("strategy-configuration")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, []);

  useEffect(() => {
    if (window.location.hash !== "#strategy-configuration") return;
    const frameId = window.requestAnimationFrame(scrollToStrategyConfiguration);
    return () => window.cancelAnimationFrame(frameId);
  }, [scrollToStrategyConfiguration]);

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
                  ? demoBalanceError || demoPositionsError
                    ? "OKX Demo 数据读取失败"
                    : demoBalanceLoading
                      ? "同步 OKX Demo 账户中"
                      : `OKX Demo · ${demoConnection?.label ?? "已绑定连接"}`
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
            <Button
              className="bg-sky-600 text-white hover:bg-sky-500"
              onClick={scrollToStrategyConfiguration}
              type="button"
            >
              配置策略 <ArrowUpRight />
            </Button>
          </div>
        </header>

        <section
          className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4"
          aria-label="投资组合概览"
        >
          <KpiCard
            icon={WalletCards}
            label="组合权益"
            value={
              <>
                <TerminalValue value={displayEquity} /> USDT
              </>
            }
            detail={
              isOkxDemo
                ? `OKX Demo 实时估值 · ${demoBalance?.checked_at ? new Date(demoBalance.checked_at).toLocaleTimeString() : "等待账户同步"}`
                : `初始资金 ${currency.format(account?.initial_capital_quote ?? 10_000)} USDT`
            }
          />
          <KpiCard
            icon={pnl >= 0 ? TrendingUp : TrendingDown}
            label={isOkxDemo ? "账户盈亏" : "总收益与亏损"}
            value={
              <>
                <TerminalValue signed value={isOkxDemo ? 0 : pnl} /> USDT
              </>
            }
            detail={
              isOkxDemo
                ? "OKX Demo 未提供可靠成本基准；不展示估算盈亏"
                : `策略启动以来 ${pnlPercent >= 0 ? "+" : ""}${pnlPercent.toFixed(2)}%`
            }
            trend={isOkxDemo ? "neutral" : pnl >= 0 ? "positive" : "negative"}
          />
          <KpiCard
            icon={CircleDollarSign}
            label="可用资金"
            value={
              <>
                <TerminalValue value={displayCash} /> USDT
              </>
            }
            detail={`已投入 ${currency.format(Math.max(invested, 0))} USDT`}
          />
          <KpiCard
            icon={Layers3}
            label="策略执行情况"
            value={`${holdingRows.length} / ${ruleStrategy?.config.risk.max_positions ?? 0}`}
            detail={`已成交 ${trades.length} 笔模拟交易，扫描 ${trackedSymbols.length} 个币种`}
            trend="neutral"
          />
        </section>

        <section
          aria-label="关键阈值仪表"
          className="grid gap-3 lg:grid-cols-2"
        >
          <ThresholdGauge
            description={rsiDescription}
            displayValue={latestRsi === null ? "—" : latestRsi.toFixed(1)}
            label="RSI（14）"
            thresholds={["30 超卖", "50 中性", "70 超买"]}
            value={latestRsi}
          />
          <ThresholdGauge
            description={utilizationDescription}
            displayValue={
              capitalUtilization === null
                ? "—"
                : `${capitalUtilization.toFixed(1)}%`
            }
            label="资金使用率"
            thresholds={["0%", "50%", "100% 上限"]}
            value={capitalUtilization}
          />
        </section>

        <section
          className="terminal-strip grid gap-px overflow-hidden rounded-lg border border-sky-500/20 bg-border/50 lg:grid-cols-[1.2fr_1fr_1fr_1.1fr]"
          aria-label="策略终端状态"
        >
          <div className="bg-card/95 px-3 py-3">
            <div className="flex items-center gap-2 text-sky-500">
              <Cpu className="size-3.5" />
              <span className="terminal-label">当前策略</span>
            </div>
            <p className="mt-2 truncate font-medium text-sm">
              {ruleStrategy?.name ?? "尚未配置策略"}
            </p>
            <p className="mt-1 text-muted-foreground text-xs">
              {ruleStrategy
                ? `${trackedSymbols.length} 个币种 · ${ruleStrategy.config.interval} 周期`
                : "前往策略配置页面启用扫描"}
            </p>
          </div>
          <div className="bg-card/95 px-3 py-3">
            <div className="flex items-center gap-2 text-emerald-500">
              <Crosshair className="size-3.5" />
              <span className="terminal-label">本轮筛选</span>
            </div>
            <p className="terminal-number mt-2 font-semibold text-lg">
              {holdingRows.length}{" "}
              <span className="text-muted-foreground text-xs">个当前持仓</span>
            </p>
            <p className="mt-1 text-muted-foreground text-xs">
              合格币种按本轮数量均分资金
            </p>
          </div>
          <div className="bg-card/95 px-3 py-3">
            <div className="flex items-center gap-2 text-amber-500">
              <CircleDollarSign className="size-3.5" />
              <span className="terminal-label">最新单币额度</span>
            </div>
            <p className="terminal-number mt-2 font-semibold text-lg">
              <TerminalValue value={requestedCapital} />{" "}
              <span className="text-muted-foreground text-xs">USDT</span>
            </p>
            <p className="mt-1 text-muted-foreground text-xs">
              总资金 ÷ 本轮合格币种数，向下取整
            </p>
          </div>
          <div className="bg-card/95 px-3 py-3">
            <div className="flex items-center gap-2 text-violet-500">
              <Clock3 className="size-3.5" />
              <span className="terminal-label">最近评估</span>
            </div>
            <p className="mt-2 truncate font-medium text-sm">
              {latestEvaluation?.symbol
                ? toDashboardSymbol(latestEvaluation.symbol)
                : "等待首轮扫描"}
            </p>
            <p className="mt-1 text-muted-foreground text-xs">
              {latestEvaluation?.evaluated_at
                ? new Intl.DateTimeFormat("zh-CN", {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  }).format(new Date(latestEvaluation.evaluated_at))
                : "策略启动后自动更新"}
            </p>
          </div>
        </section>

        <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex items-center justify-between border-border/70 border-b px-4 py-3">
              <div>
                <h2 className="font-semibold">策略参数总览</h2>
                <p className="mt-0.5 text-muted-foreground text-xs">
                  首页同步展示正在使用的筛选与风控配置
                </p>
              </div>
              <Button
                onClick={scrollToStrategyConfiguration}
                size="sm"
                type="button"
                variant="ghost"
              >
                <Settings2 /> 编辑
              </Button>
            </div>
            <CardContent className="grid gap-px bg-border/60 p-px sm:grid-cols-2 lg:grid-cols-4">
              <div className="bg-card px-3 py-3">
                <p className="terminal-label">均线趋势</p>
                <p className="mt-1 font-medium text-sm">
                  {ruleStrategy?.config.moving_average.enabled
                    ? `MA ${ruleStrategy.config.moving_average.short_window} / ${ruleStrategy.config.moving_average.long_window}`
                    : "未启用"}
                </p>
              </div>
              <div className="bg-card px-3 py-3">
                <p className="terminal-label">RSI 阈值</p>
                <p className="mt-1 font-medium text-sm">
                  {ruleStrategy?.config.rsi.enabled
                    ? `${ruleStrategy.config.rsi.oversold} / ${ruleStrategy.config.rsi.overbought}`
                    : "未启用"}
                </p>
              </div>
              <div className="bg-card px-3 py-3">
                <p className="terminal-label">止盈 / 止损</p>
                <p className="mt-1 font-medium text-sm">
                  {ruleStrategy?.config.risk.take_profit_pct
                    ? `${(ruleStrategy.config.risk.take_profit_pct * 100).toFixed(1)}%`
                    : "--"}{" "}
                  /{" "}
                  {ruleStrategy?.config.risk.stop_loss_pct
                    ? `${(ruleStrategy.config.risk.stop_loss_pct * 100).toFixed(1)}%`
                    : "--"}
                </p>
              </div>
              <div className="bg-card px-3 py-3">
                <p className="terminal-label">启用指标</p>
                <p
                  className="mt-1 truncate font-medium text-sm"
                  title={enabledIndicators.join("，")}
                >
                  {enabledIndicators.join(" · ") || "未启用"}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex items-center justify-between border-border/70 border-b px-4 py-3">
              <div>
                <h2 className="font-semibold">扫描回放</h2>
                <p className="mt-0.5 text-muted-foreground text-xs">
                  最近评估的策略动作
                </p>
              </div>
              <RefreshCw className="size-4 text-sky-500" />
            </div>
            <CardContent className="grid grid-cols-2 gap-px bg-border/60 p-px sm:grid-cols-4">
              {recentlyScanned.length === 0 ? (
                <p className="col-span-full bg-card px-4 py-7 text-center text-muted-foreground text-sm">
                  策略启动后，这里会显示本轮扫描的币种与动作。
                </p>
              ) : (
                recentlyScanned.map((item) => (
                  <button
                    className="bg-card px-3 py-2 text-left transition-colors hover:bg-sky-500/5"
                    key={item.evaluation_id}
                    onClick={() =>
                      item.symbol && setSelectedSymbol(item.symbol)
                    }
                    type="button"
                  >
                    <span className="block truncate font-medium text-xs">
                      {item.symbol
                        ? toDashboardSymbol(item.symbol)
                        : "市场评估"}
                    </span>
                    <span
                      className={cn(
                        "mt-1 block text-xs",
                        item.action === "buy"
                          ? "text-emerald-500"
                          : item.action === "sell"
                            ? "text-rose-500"
                            : "text-muted-foreground",
                      )}
                    >
                      {item.action === "buy"
                        ? "符合入场"
                        : item.action === "sell"
                          ? "符合离场"
                          : "继续观察"}
                    </span>
                  </button>
                ))
              )}
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-4 xl:grid-cols-[minmax(0,1.72fr)_minmax(300px,0.78fr)]">
          <Card className="dashboard-panel overflow-hidden rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex flex-col gap-3 border-border/70 border-b px-4 py-3">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="grid size-9 shrink-0 place-items-center rounded-md bg-sky-500/10 text-sky-500">
                    <CandlestickChart className="size-4" />
                  </span>
                  <div className="min-w-0">
                    <h2 className="font-semibold">
                      {toDashboardSymbol(effectiveSymbol)} 市场走势
                    </h2>
                    <p className="text-muted-foreground text-xs">
                      {market?.latest_price
                        ? `${market.latest_price.toLocaleString()} USDT`
                        : "等待市场价格"}
                      ，{market?.interval ?? marketInterval} K 线
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-3 md:justify-end">
                  <div className="min-w-[132px] text-right">
                    <p className="terminal-label">当前价格</p>
                    {market?.latest_price != null ? (
                      <p className="mt-1 whitespace-nowrap text-amber-500 text-lg dark:text-amber-300">
                        <TerminalValue value={market.latest_price} />{" "}
                        <span className="text-xs">USDT</span>
                      </p>
                    ) : (
                      <p className="mt-1 text-muted-foreground text-sm">
                        等待行情
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={
                        marketFailure || marketError ? "destructive" : "outline"
                      }
                      className="gap-1 text-[11px]"
                    >
                      <RadioTower className="size-3" />{" "}
                      {market?.provider ?? "行情数据源"}
                    </Badge>
                    {market?.freshness_status === "stale" ? (
                      <Badge variant="outline">数据延迟</Badge>
                    ) : null}
                  </div>
                </div>
              </div>
              <div className="flex flex-col gap-2 border-border/60 border-t pt-3 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex flex-wrap items-center gap-2">
                  <Select
                    value={effectiveSymbol}
                    onValueChange={setSelectedSymbol}
                  >
                    <SelectTrigger
                      aria-label="图表交易对"
                      className="h-8 w-32 text-xs"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Array.from(new Set(["BTC-USDT", ...marketSymbols])).map(
                        (symbol) => (
                          <SelectItem key={symbol} value={symbol}>
                            {toDashboardSymbol(symbol)}
                          </SelectItem>
                        ),
                      )}
                    </SelectContent>
                  </Select>
                  <Select
                    value={marketInterval}
                    onValueChange={(value) =>
                      setMarketInterval(
                        value as (typeof MARKET_INTERVALS)[number],
                      )
                    }
                  >
                    <SelectTrigger
                      aria-label="K 线周期"
                      className="h-8 w-20 text-xs"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {MARKET_INTERVALS.map((interval) => (
                        <SelectItem key={interval} value={interval}>
                          {interval}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <div
                    className="flex items-center gap-1"
                    aria-label="历史范围"
                  >
                    {MARKET_HISTORY_RANGES.map((range) => (
                      <Button
                        key={range.value}
                        onClick={() => {
                          setHistoryRange(range.value);
                          setFromDate("");
                          setToDate("");
                          setRequestNowMs(Date.now());
                        }}
                        size="sm"
                        type="button"
                        variant={
                          historyRange === range.value && !fromDate && !toDate
                            ? "secondary"
                            : "ghost"
                        }
                      >
                        {range.label}
                      </Button>
                    ))}
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <Input
                    aria-label="图表开始日期"
                    className="h-8 w-36"
                    onChange={(event) => {
                      setFromDate(event.target.value);
                      setRequestNowMs(Date.now());
                    }}
                    type="date"
                    value={fromDate}
                  />
                  <span className="text-muted-foreground">至</span>
                  <Input
                    aria-label="图表结束日期"
                    className="h-8 w-36"
                    onChange={(event) => {
                      setToDate(event.target.value);
                      setRequestNowMs(Date.now());
                    }}
                    type="date"
                    value={toDate}
                  />
                </div>
              </div>
            </div>
            <CardContent className="p-0">
              {invalidDateRange ? (
                <p className="py-28 text-center text-destructive text-sm">
                  开始日期不能晚于结束日期。
                </p>
              ) : marketError || marketFailure ? (
                <p className="py-28 text-center text-destructive text-sm">
                  市场数据暂时不可用。
                </p>
              ) : (
                <CandlestickChartComponent
                  currentPrice={market?.latest_price}
                  data={candles}
                  movingAverages={movingAverages}
                  loading={marketLoading}
                  height={410}
                  theme={isDark ? "dark" : "light"}
                />
              )}
            </CardContent>
          </Card>

          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex items-center justify-between border-border/70 border-b px-4 py-3">
              <div>
                <h2 className="font-semibold">符合策略的持仓</h2>
                <p className="mt-0.5 text-muted-foreground text-xs">
                  已根据信号建立的模拟持仓
                </p>
              </div>
              <Badge variant="outline">{holdingRows.length} 个持仓</Badge>
            </div>
            <CardContent className="p-0">
              {holdingRows.length === 0 ? (
                <div className="px-4 py-12 text-center">
                  <BarChart3 className="mx-auto size-6 text-muted-foreground/60" />
                  <p className="mt-3 text-muted-foreground text-sm">
                    暂时没有符合条件的持仓
                  </p>
                  <p className="mt-1 text-muted-foreground text-xs">
                    策略在入场信号成交后会将币种展示在这里。
                  </p>
                </div>
              ) : (
                <div className="max-h-96 divide-y divide-border/70 overflow-y-auto overscroll-contain">
                  {holdingRows.map(({ symbol, position, value, profit }) => (
                    <button
                      className="group flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition-colors hover:bg-sky-500/5"
                      key={symbol}
                      onClick={() => setSelectedSymbol(symbol)}
                      type="button"
                    >
                      <span className="min-w-0">
                        <span className="block font-medium group-hover:text-sky-500">
                          {toDashboardSymbol(symbol)}
                        </span>
                        <span className="block text-muted-foreground text-xs">
                          {position.quantity.toFixed(6)} 个，入场价{" "}
                          {compactCurrency.format(position.entry_price)}
                        </span>
                      </span>
                      <span className="text-right tabular-nums">
                        <span className="block font-medium text-sm">
                          {currency.format(value)}
                        </span>
                        <span
                          className={cn(
                            "block text-xs",
                            profit >= 0 ? "text-emerald-500" : "text-rose-500",
                          )}
                        >
                          {profit >= 0 ? "+" : ""}
                          {currency.format(profit)}
                        </span>
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-4 xl:grid-cols-[minmax(0,1.72fr)_minmax(300px,0.78fr)]">
          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex flex-col gap-3 border-border/70 border-b px-4 py-3">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="font-semibold">所选币种技术指标</h2>
                  <p className="mt-0.5 text-muted-foreground text-xs">
                    {toDashboardSymbol(effectiveSymbol)} 的 RSI、布林带与 MACD
                  </p>
                </div>
                <div className="flex max-w-full gap-1 overflow-x-auto pb-1 sm:pb-0">
                  {[
                    "BTC-USDT",
                    ...marketSymbols.filter((symbol) => symbol !== "BTC-USDT"),
                  ]
                    .slice(0, 12)
                    .map((symbol) => (
                      <button
                        className={cn(
                          "shrink-0 rounded-md border px-2.5 py-1 font-medium text-xs transition-colors",
                          effectiveSymbol === symbol
                            ? "border-sky-500/50 bg-sky-500/10 text-sky-600 dark:text-sky-300"
                            : "border-border text-muted-foreground hover:bg-muted",
                        )}
                        key={symbol}
                        onClick={() => setSelectedSymbol(symbol)}
                        type="button"
                      >
                        {toDashboardSymbol(symbol)}
                      </button>
                    ))}
                </div>
              </div>
              <fieldset
                aria-label="RSI 与布林带显示方式"
                className="flex w-fit rounded-md border border-cyan-500/25 bg-cyan-500/5 p-0.5"
              >
                <button
                  aria-pressed={rsiMode === "rsi"}
                  className={cn(
                    "rounded px-2.5 py-1 font-medium text-xs transition-colors",
                    rsiMode === "rsi"
                      ? "bg-violet-500/20 text-violet-700 dark:text-violet-200"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                  onClick={() => setRsiMode("rsi")}
                  type="button"
                >
                  RSI
                </button>
                <button
                  aria-pressed={rsiMode === "bollinger"}
                  className={cn(
                    "rounded px-2.5 py-1 font-medium text-xs transition-colors",
                    rsiMode === "bollinger"
                      ? "bg-cyan-500/20 text-cyan-700 dark:text-cyan-200"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                  onClick={() => setRsiMode("bollinger")}
                  type="button"
                >
                  布林带
                </button>
                <button
                  aria-pressed={rsiMode === "both"}
                  className={cn(
                    "rounded px-2.5 py-1 font-medium text-xs transition-colors",
                    rsiMode === "both"
                      ? "bg-sky-500/20 text-sky-700 dark:text-sky-200"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                  onClick={() => setRsiMode("both")}
                  type="button"
                >
                  同时显示
                </button>
              </fieldset>
            </div>
            <CardContent className="grid gap-px overflow-hidden bg-border/60 p-px md:grid-cols-2">
              <div className="flex flex-col gap-2 bg-card p-2">
                {rsiMode !== "bollinger" ? (
                  <MarketIndicatorPanelChart
                    data={market?.indicators ?? []}
                    panel="rsi"
                    height={rsiMode === "both" ? 160 : 190}
                    theme={isDark ? "dark" : "light"}
                  />
                ) : null}
                {rsiMode !== "rsi" ? (
                  <MarketIndicatorPanelChart
                    candles={market?.candles ?? []}
                    data={market?.indicators ?? []}
                    panel="bollinger"
                    height={rsiMode === "both" ? 180 : 190}
                    theme={isDark ? "dark" : "light"}
                  />
                ) : null}
              </div>
              <div className="bg-card p-2">
                <MarketIndicatorPanelChart
                  data={market?.indicators ?? []}
                  panel="macd"
                  height={rsiMode === "both" ? 350 : 190}
                  theme={isDark ? "dark" : "light"}
                />
              </div>
            </CardContent>
          </Card>

          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="border-border/70 border-b px-4 py-3">
              <h2 className="font-semibold">最近执行记录</h2>
              <p className="mt-0.5 text-muted-foreground text-xs">
                最近确认的买入与卖出决策
              </p>
            </div>
            <CardContent className="p-0">
              {recentSignals.length === 0 ? (
                <p className="px-4 py-12 text-center text-muted-foreground text-sm">
                  暂时没有完成的执行周期。
                </p>
              ) : (
                <div className="divide-y divide-border/70">
                  {recentSignals.map((signal) => (
                    <div
                      className="flex items-center justify-between gap-3 px-4 py-3"
                      key={signal.evaluation_id}
                    >
                      <span>
                        <span className="block font-medium text-sm">
                          {signal.action === "buy" ? "买入信号" : "卖出信号"}
                        </span>
                        <span className="block max-w-48 truncate text-muted-foreground text-xs">
                          {signal.reason}
                        </span>
                      </span>
                      <Badge
                        className={
                          signal.action === "buy"
                            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300"
                            : "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300"
                        }
                        variant="outline"
                      >
                        {signal.action === "buy" ? "买入" : "卖出"}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
          <div className="flex flex-col gap-2 border-border/70 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="font-semibold">组合收益与亏损</h2>
              <p className="mt-0.5 text-muted-foreground text-xs">
                模拟执行产生的已实现与未实现收益汇总
              </p>
            </div>
            <span
              className={cn(
                "font-semibold text-sm tabular-nums",
                pnl >= 0 ? "text-emerald-500" : "text-rose-500",
              )}
            >
              {pnl >= 0 ? "+" : ""}
              {currency.format(pnl)} USDT
            </span>
          </div>
          <CardContent className="p-2 sm:p-4">
            {pnlCurve?.length ? (
              <PnlLineChart
                data={pnlCurve}
                height={240}
                theme={isDark ? "dark" : "light"}
              />
            ) : (
              <div className="grid h-60 place-items-center text-center">
                <p className="text-muted-foreground text-sm">
                  首个评估周期完成后将显示收益曲线。
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        <section
          aria-label="策略配置"
          className="scroll-mt-4"
          id="strategy-configuration"
        >
          <RuleStrategyConfiguration embedded />
        </section>
      </div>
    </div>
  );
}
