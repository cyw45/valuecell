import { useTheme } from "next-themes";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
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
  RefreshCw,
  RadioTower,
  Settings2,
  Sun,
  TrendingDown,
  TrendingUp,
  WalletCards,
} from "lucide-react";
import {
  useRuleStrategy,
  useRuleStrategyEvaluations,
  useRuleStrategyPnlCurve,
  useRuleStrategyTrades,
} from "@/api/rule-strategy";
import { useGetCryptoMarketIndicators } from "@/api/crypto-market";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import CandlestickChartComponent, {
  type CandlestickData,
  type CandlestickMovingAverage,
} from "@/components/valuecell/charts/candlestick-chart";
import { MarketIndicatorPanelChart } from "@/components/valuecell/charts/market-indicator-panel";
import { PnlLineChart } from "@/components/valuecell/charts/pnl-line-chart";
import { cn } from "@/lib/utils";

const currency = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const compactCurrency = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 2,
});

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
    <span className={cn("terminal-number relative inline-block tabular-nums", className)}>
      <span>{visibleValue}</span>
      {showFlash ? <span aria-hidden className="terminal-value-flash">{visibleValue}</span> : null}
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
          <span className={cn(
            "grid size-9 shrink-0 place-items-center rounded-md border",
            isPositive ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-500" :
              isNegative ? "border-rose-500/30 bg-rose-500/10 text-rose-500" :
                "border-sky-500/30 bg-sky-500/10 text-sky-500",
          )}>
            <Icon className="size-4" />
          </span>
        </div>
        <p className="mt-3 truncate text-muted-foreground text-xs" title={detail}>{detail}</p>
      </CardContent>
    </Card>
  );
}

export default function DashboardPage() {
  const { resolvedTheme, setTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const strategyId = localStorage.getItem("valuecell.rule-strategy-id") ?? "";
  const { data: ruleStrategy } = useRuleStrategy(strategyId);
  const { data: pnlCurve } = useRuleStrategyPnlCurve(strategyId || undefined);
  const { data: trades = [] } = useRuleStrategyTrades(strategyId || undefined);
  const { data: evaluations = [] } = useRuleStrategyEvaluations(strategyId || undefined);
  const trackedSymbols = ruleStrategy?.config.symbols ?? [];
  const activeSymbols = Object.keys(ruleStrategy?.account.positions ?? {});
  const marketSymbols = activeSymbols.length > 0 ? activeSymbols : trackedSymbols;
  const [selectedSymbol, setSelectedSymbol] = useState("BTC-USDT");
  const selectedIsAvailable = marketSymbols.includes(selectedSymbol);
  const effectiveSymbol = selectedIsAvailable || marketSymbols.length === 0
    ? selectedSymbol
    : marketSymbols[0];
  const { data: marketData, isFetching: marketLoading, isError: marketError } = useGetCryptoMarketIndicators({
    symbols: [effectiveSymbol],
    interval: ruleStrategy?.config.interval ?? "1h",
    lookback: 240,
  });
  const market = marketData?.symbols[0];
  const marketFailure = marketData?.failed_symbols[effectiveSymbol];

  const candles: CandlestickData[] = market?.candles.map((candle) => ({
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
  const pnl = (account?.realized_pnl_quote ?? 0) + (account?.unrealized_pnl_quote ?? 0);
  const pnlPercent = account && account.initial_capital_quote > 0
    ? (pnl / account.initial_capital_quote) * 100
    : 0;
  const invested = account ? account.equity_quote - account.quote_balance : 0;
  const holdingRows = useMemo(() => Object.entries(account?.positions ?? {}).map(([symbol, position]) => {
    const value = position.quantity * position.mark_price;
    const profit = position.quantity * (position.mark_price - position.entry_price);
    return { symbol, position, value, profit };
  }), [account?.positions]);
  const recentSignals = evaluations
    .filter((item) => item.action !== "no_op")
    .slice(0, 5);
  const latestEvaluation = evaluations[0];
  const enabledIndicators = [
    ruleStrategy?.config.moving_average.enabled ? `均线 ${ruleStrategy.config.moving_average.short_window}/${ruleStrategy.config.moving_average.long_window}` : null,
    ruleStrategy?.config.rsi.enabled ? `RSI ${ruleStrategy.config.rsi.period}` : null,
    ruleStrategy?.config.bollinger.enabled ? `布林带 ${ruleStrategy.config.bollinger.period}` : null,
    ruleStrategy?.config.momentum_macd.enabled ? `MACD ${ruleStrategy.config.momentum_macd.macd_fast_window}/${ruleStrategy.config.momentum_macd.macd_slow_window}/${ruleStrategy.config.momentum_macd.macd_signal_window}` : null,
  ].filter((item): item is string => item !== null);
  const recentlyScanned = evaluations.slice(0, 8);
  const requestedCapital = latestEvaluation?.sizing.requested_quote ?? 0;

  return (
    <div className="scroll-container dashboard-shell flex size-full flex-col">
      <div className="mx-auto flex w-full max-w-[1800px] flex-col gap-4 p-4 lg:p-5">
        <header className="dashboard-header flex flex-col gap-4 rounded-lg border border-sky-400/15 px-4 py-4 md:flex-row md:items-center md:justify-between md:px-5">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="live-pulse size-2 rounded-full bg-emerald-400" />
              <span className="font-semibold text-[11px] text-sky-500 tracking-[0.14em] dark:text-sky-300">模拟交易终端</span>
              <Badge variant="outline" className="border-sky-500/30 bg-sky-500/10 text-sky-600 dark:text-sky-300">
                {ruleStrategy?.status === "running" ? "策略扫描中" : "策略待命"}
              </Badge>
            </div>
            <h1 className="dashboard-title mt-2 font-semibold text-2xl tracking-[0.02em]">市场指挥中心</h1>
            <p className="mt-1 text-muted-foreground text-sm">
              {ruleStrategy ? `${ruleStrategy.name}，正在监测 ${trackedSymbols.length} 个市场` : "BTC 市场数据已就绪。配置策略后即可开启模拟执行。"}
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
              <TooltipContent>{isDark ? "切换为浅色主题" : "切换为深色主题"}</TooltipContent>
            </Tooltip>
            <Button asChild className="bg-sky-600 text-white hover:bg-sky-500">
              <Link to="/strategies">配置策略 <ArrowUpRight /></Link>
            </Button>
          </div>
        </header>

        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="投资组合概览">
          <KpiCard icon={WalletCards} label="组合权益" value={<><TerminalValue value={account?.equity_quote ?? 0} /> USDT</>} detail={`初始资金 ${currency.format(account?.initial_capital_quote ?? 10_000)} USDT`} />
          <KpiCard icon={pnl >= 0 ? TrendingUp : TrendingDown} label="总收益与亏损" value={<><TerminalValue signed value={pnl} /> USDT</>} detail={`策略启动以来 ${pnlPercent >= 0 ? "+" : ""}${pnlPercent.toFixed(2)}%`} trend={pnl >= 0 ? "positive" : "negative"} />
          <KpiCard icon={CircleDollarSign} label="可用资金" value={<><TerminalValue value={account?.quote_balance ?? 0} /> USDT</>} detail={`已投入 ${currency.format(Math.max(invested, 0))} USDT`} />
          <KpiCard icon={Layers3} label="策略执行情况" value={`${holdingRows.length} / ${ruleStrategy?.config.risk.max_positions ?? 0}`} detail={`已成交 ${trades.length} 笔模拟交易，扫描 ${trackedSymbols.length} 个币种`} trend="neutral" />
        </section>

        <section className="terminal-strip grid gap-px overflow-hidden rounded-lg border border-sky-500/20 bg-border/50 lg:grid-cols-[1.2fr_1fr_1fr_1.1fr]" aria-label="策略终端状态">
          <div className="bg-card/95 px-3 py-3">
            <div className="flex items-center gap-2 text-sky-500"><Cpu className="size-3.5" /><span className="terminal-label">当前策略</span></div>
            <p className="mt-2 truncate font-medium text-sm">{ruleStrategy?.name ?? "尚未配置策略"}</p>
            <p className="mt-1 text-muted-foreground text-xs">{ruleStrategy ? `${trackedSymbols.length} 个币种 · ${ruleStrategy.config.interval} 周期` : "前往策略配置页面启用扫描"}</p>
          </div>
          <div className="bg-card/95 px-3 py-3">
            <div className="flex items-center gap-2 text-emerald-500"><Crosshair className="size-3.5" /><span className="terminal-label">本轮筛选</span></div>
            <p className="terminal-number mt-2 font-semibold text-lg">{holdingRows.length} <span className="text-muted-foreground text-xs">个当前持仓</span></p>
            <p className="mt-1 text-muted-foreground text-xs">合格币种按本轮数量均分资金</p>
          </div>
          <div className="bg-card/95 px-3 py-3">
            <div className="flex items-center gap-2 text-amber-500"><CircleDollarSign className="size-3.5" /><span className="terminal-label">最新单币额度</span></div>
            <p className="terminal-number mt-2 font-semibold text-lg"><TerminalValue value={requestedCapital} /> <span className="text-muted-foreground text-xs">USDT</span></p>
            <p className="mt-1 text-muted-foreground text-xs">总资金 ÷ 本轮合格币种数，向下取整</p>
          </div>
          <div className="bg-card/95 px-3 py-3">
            <div className="flex items-center gap-2 text-violet-500"><Clock3 className="size-3.5" /><span className="terminal-label">最近评估</span></div>
            <p className="mt-2 truncate font-medium text-sm">{latestEvaluation?.symbol ? toDashboardSymbol(latestEvaluation.symbol) : "等待首轮扫描"}</p>
            <p className="mt-1 text-muted-foreground text-xs">{latestEvaluation?.evaluated_at ? new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(latestEvaluation.evaluated_at)) : "策略启动后自动更新"}</p>
          </div>
        </section>

        <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex items-center justify-between border-border/70 border-b px-4 py-3">
              <div><h2 className="font-semibold">策略参数总览</h2><p className="mt-0.5 text-muted-foreground text-xs">首页同步展示正在使用的筛选与风控配置</p></div>
              <Button asChild size="sm" variant="ghost"><Link to="/strategies"><Settings2 /> 编辑</Link></Button>
            </div>
            <CardContent className="grid gap-px bg-border/60 p-px sm:grid-cols-2 lg:grid-cols-4">
              <div className="bg-card px-3 py-3"><p className="terminal-label">均线趋势</p><p className="mt-1 font-medium text-sm">{ruleStrategy?.config.moving_average.enabled ? `MA ${ruleStrategy.config.moving_average.short_window} / ${ruleStrategy.config.moving_average.long_window}` : "未启用"}</p></div>
              <div className="bg-card px-3 py-3"><p className="terminal-label">RSI 阈值</p><p className="mt-1 font-medium text-sm">{ruleStrategy?.config.rsi.enabled ? `${ruleStrategy.config.rsi.oversold} / ${ruleStrategy.config.rsi.overbought}` : "未启用"}</p></div>
              <div className="bg-card px-3 py-3"><p className="terminal-label">止盈 / 止损</p><p className="mt-1 font-medium text-sm">{ruleStrategy?.config.risk.take_profit_pct ? `${(ruleStrategy.config.risk.take_profit_pct * 100).toFixed(1)}%` : "--"} / {ruleStrategy?.config.risk.stop_loss_pct ? `${(ruleStrategy.config.risk.stop_loss_pct * 100).toFixed(1)}%` : "--"}</p></div>
              <div className="bg-card px-3 py-3"><p className="terminal-label">启用指标</p><p className="mt-1 truncate font-medium text-sm" title={enabledIndicators.join("，")}>{enabledIndicators.join(" · ") || "未启用"}</p></div>
            </CardContent>
          </Card>

          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex items-center justify-between border-border/70 border-b px-4 py-3"><div><h2 className="font-semibold">扫描回放</h2><p className="mt-0.5 text-muted-foreground text-xs">最近评估的策略动作</p></div><RefreshCw className="size-4 text-sky-500" /></div>
            <CardContent className="grid grid-cols-2 gap-px bg-border/60 p-px sm:grid-cols-4">
              {recentlyScanned.length === 0 ? <p className="col-span-full bg-card px-4 py-7 text-center text-muted-foreground text-sm">策略启动后，这里会显示本轮扫描的币种与动作。</p> : recentlyScanned.map((item) => <button className="bg-card px-3 py-2 text-left transition-colors hover:bg-sky-500/5" key={item.evaluation_id} onClick={() => item.symbol && setSelectedSymbol(item.symbol)} type="button"><span className="block truncate font-medium text-xs">{item.symbol ? toDashboardSymbol(item.symbol) : "市场评估"}</span><span className={cn("mt-1 block text-xs", item.action === "buy" ? "text-emerald-500" : item.action === "sell" ? "text-rose-500" : "text-muted-foreground")}>{item.action === "buy" ? "符合入场" : item.action === "sell" ? "符合离场" : "继续观察"}</span></button>)}
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-4 xl:grid-cols-[minmax(0,1.72fr)_minmax(300px,0.78fr)]">
          <Card className="dashboard-panel overflow-hidden rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex flex-col gap-3 border-border/70 border-b px-4 py-3 md:flex-row md:items-center md:justify-between">
              <div className="flex min-w-0 items-center gap-3">
                <span className="grid size-9 shrink-0 place-items-center rounded-md bg-sky-500/10 text-sky-500"><CandlestickChart className="size-4" /></span>
                <div className="min-w-0">
                  <h2 className="font-semibold">{toDashboardSymbol(effectiveSymbol)} 市场走势</h2>
                  <p className="text-muted-foreground text-xs">{market?.latest_price ? `${market.latest_price.toLocaleString()} USDT` : "等待市场价格"}，{market?.interval ?? ruleStrategy?.config.interval ?? "1h"} K 线</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={marketFailure || marketError ? "destructive" : "outline"} className="gap-1 text-[11px]">
                  <RadioTower className="size-3" /> {market?.provider ?? "行情数据源"}
                </Badge>
                {market?.freshness_status === "stale" ? <Badge variant="outline">数据延迟</Badge> : null}
              </div>
            </div>
            <CardContent className="p-0">
              {marketError || marketFailure ? <p className="py-28 text-center text-destructive text-sm">市场数据暂时不可用。</p> : (
                <CandlestickChartComponent data={candles} movingAverages={movingAverages} loading={marketLoading} height={410} theme={isDark ? "dark" : "light"} />
              )}
            </CardContent>
          </Card>

          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex items-center justify-between border-border/70 border-b px-4 py-3">
              <div><h2 className="font-semibold">符合策略的持仓</h2><p className="mt-0.5 text-muted-foreground text-xs">已根据信号建立的模拟持仓</p></div>
              <Badge variant="outline">{holdingRows.length} 个持仓</Badge>
            </div>
            <CardContent className="p-0">
              {holdingRows.length === 0 ? <div className="px-4 py-12 text-center"><BarChart3 className="mx-auto size-6 text-muted-foreground/60" /><p className="mt-3 text-muted-foreground text-sm">暂时没有符合条件的持仓</p><p className="mt-1 text-muted-foreground text-xs">策略在入场信号成交后会将币种展示在这里。</p></div> : (
                <div className="divide-y divide-border/70">
                  {holdingRows.map(({ symbol, position, value, profit }) => <button className="group flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition-colors hover:bg-sky-500/5" key={symbol} onClick={() => setSelectedSymbol(symbol)} type="button">
                    <span className="min-w-0"><span className="block font-medium group-hover:text-sky-500">{toDashboardSymbol(symbol)}</span><span className="block text-muted-foreground text-xs">{position.quantity.toFixed(6)} 个，入场价 {compactCurrency.format(position.entry_price)}</span></span>
                    <span className="text-right tabular-nums"><span className="block font-medium text-sm">{currency.format(value)}</span><span className={cn("block text-xs", profit >= 0 ? "text-emerald-500" : "text-rose-500")}>{profit >= 0 ? "+" : ""}{currency.format(profit)}</span></span>
                  </button>)}
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-4 xl:grid-cols-[minmax(0,1.72fr)_minmax(300px,0.78fr)]">
          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="flex flex-col gap-3 border-border/70 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
              <div><h2 className="font-semibold">所选币种技术指标</h2><p className="mt-0.5 text-muted-foreground text-xs">{toDashboardSymbol(effectiveSymbol)} 的 RSI、MACD 与成交量动量</p></div>
              <div className="flex max-w-full gap-1 overflow-x-auto pb-1 sm:pb-0">
                {["BTC-USDT", ...marketSymbols.filter((symbol) => symbol !== "BTC-USDT")].slice(0, 12).map((symbol) => <button className={cn("shrink-0 rounded-md border px-2.5 py-1 font-medium text-xs transition-colors", effectiveSymbol === symbol ? "border-sky-500/50 bg-sky-500/10 text-sky-600 dark:text-sky-300" : "border-border text-muted-foreground hover:bg-muted")} key={symbol} onClick={() => setSelectedSymbol(symbol)} type="button">{toDashboardSymbol(symbol)}</button>)}
              </div>
            </div>
            <CardContent className="grid gap-px overflow-hidden bg-border/60 p-px md:grid-cols-2">
              <div className="bg-card p-2"><MarketIndicatorPanelChart data={market?.indicators ?? []} panel="rsi" height={190} theme={isDark ? "dark" : "light"} /></div>
              <div className="bg-card p-2"><MarketIndicatorPanelChart data={market?.indicators ?? []} panel="macd" height={190} theme={isDark ? "dark" : "light"} /></div>
            </CardContent>
          </Card>

          <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
            <div className="border-border/70 border-b px-4 py-3"><h2 className="font-semibold">最近执行记录</h2><p className="mt-0.5 text-muted-foreground text-xs">最近确认的买入与卖出决策</p></div>
            <CardContent className="p-0">
              {recentSignals.length === 0 ? <p className="px-4 py-12 text-center text-muted-foreground text-sm">暂时没有完成的执行周期。</p> : <div className="divide-y divide-border/70">{recentSignals.map((signal) => <div className="flex items-center justify-between gap-3 px-4 py-3" key={signal.evaluation_id}><span><span className="block font-medium text-sm">{signal.action === "buy" ? "买入信号" : "卖出信号"}</span><span className="block max-w-48 truncate text-muted-foreground text-xs">{signal.reason}</span></span><Badge className={signal.action === "buy" ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300" : "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300"} variant="outline">{signal.action === "buy" ? "买入" : "卖出"}</Badge></div>)}</div>}
            </CardContent>
          </Card>
        </section>

        <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
          <div className="flex flex-col gap-2 border-border/70 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between"><div><h2 className="font-semibold">组合收益与亏损</h2><p className="mt-0.5 text-muted-foreground text-xs">模拟执行产生的已实现与未实现收益汇总</p></div><span className={cn("font-semibold text-sm tabular-nums", pnl >= 0 ? "text-emerald-500" : "text-rose-500")}>{pnl >= 0 ? "+" : ""}{currency.format(pnl)} USDT</span></div>
          <CardContent className="p-2 sm:p-4">{pnlCurve?.length ? <PnlLineChart data={pnlCurve} height={240} theme={isDark ? "dark" : "light"} /> : <div className="grid h-60 place-items-center text-center"><p className="text-muted-foreground text-sm">首个评估周期完成后将显示收益曲线。</p></div>}</CardContent>
        </Card>
      </div>
    </div>
  );
}
