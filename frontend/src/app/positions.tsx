import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { AlertTriangle, CandlestickChart, CircleDollarSign } from "lucide-react";
import { useGetCryptoMarketIndicators } from "@/api/crypto-market";
import {
  useRuleStrategy,
  useRuleStrategyAccount,
  useRuleStrategyDemoExecution,
  useRuleStrategyEvaluations,
  useRuleStrategyTrades,
} from "@/api/rule-strategy";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import CandlestickChartComponent, {
  type CandlestickData,
  type CandlestickMovingAverage,
  type CandlestickTradeMarker,
} from "@/components/valuecell/charts/candlestick-chart";
import { RuleStrategyEvaluationPath } from "@/components/valuecell/rule-strategy-evaluation-path";
import { useActiveRuleStrategyId } from "@/hooks/use-active-rule-strategy";
import type { SandboxOrder } from "@/types/sandbox-exchange";

const intervals = ["1h", "4h"] as const;
const ranges = {
  "1d": { interval: "1h", lookback: 25 },
  "5d": { interval: "1h", lookback: 121 },
  "1w": { interval: "1h", lookback: 169 },
  "1m": { interval: "4h", lookback: 187 },
} as const;
type Range = keyof typeof ranges;

type PositionDetail = {
  symbol: string;
  quantity: number;
  entryPrice: number | null;
  currentPrice: number | null;
  value: number | null;
  pnl: number | null;
};

const amount = (value: number | string | null | undefined) => {
  const parsed = typeof value === "string" ? Number(value) : value;
  return typeof parsed === "number" && Number.isFinite(parsed) ? parsed : null;
};

const canonical = (symbol: string) => symbol.toUpperCase().replace("/", "-");
const money = (value: number | null | undefined) =>
  value == null ? "—" : `${value.toLocaleString("en-US", { maximumFractionDigits: 4 })} USDT`;

function entryPrice(orders: SandboxOrder[], symbol: string) {
  let quantity = 0;
  let cost = 0;
  for (const order of orders) {
    if (canonical(order.symbol) !== canonical(symbol) || order.status !== "filled") continue;
    const filled = amount(order.filled_quantity);
    const quote = amount(order.filled_quote) ?? ((filled ?? 0) * (amount(order.average_fill_price) ?? 0));
    if (!filled || filled <= 0 || quote == null) continue;
    if (order.side === "buy") {
      quantity += filled;
      cost += quote;
    } else if (quantity > 0) {
      const sold = Math.min(quantity, filled);
      cost -= (cost / quantity) * sold;
      quantity -= sold;
    }
  }
  return quantity > 0 ? cost / quantity : null;
}

export default function PositionsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeStrategyId] = useActiveRuleStrategyId();
  const strategyId = searchParams.get("strategyId") ?? activeStrategyId;
  const routeSymbol = searchParams.get("symbol") ?? "";
  const orderId = searchParams.get("orderId");
  const evaluationId = searchParams.get("evaluationId");
  const [selectedSymbol, setSelectedSymbol] = useState(routeSymbol);
  const [range, setRange] = useState<Range>("1m");
  const strategy = useRuleStrategy(strategyId || undefined);
  const isDemo = strategy.data?.config.execution.environment === "okx_demo";
  const account = useRuleStrategyAccount(strategy.data && !isDemo ? strategyId || undefined : undefined);
  const demo = useRuleStrategyDemoExecution(strategyId || undefined, isDemo, 1, 100);
  const trades = useRuleStrategyTrades(strategyId || undefined, !isDemo);
  const evaluations = useRuleStrategyEvaluations(strategyId || undefined);
  const positions = useMemo<PositionDetail[]>(() => {
    if (isDemo) {
      return (demo.data?.positions.data.positions ?? []).map((position) => {
        const entry = entryPrice(demo.data?.orders ?? [], position.symbol);
        const mark = amount(position.mark_price);
        const value = amount(position.notional_usdt);
        return {
          symbol: canonical(position.symbol),
          quantity: position.quantity,
          entryPrice: entry,
          currentPrice: mark,
          value,
          pnl: entry != null && mark != null ? position.quantity * (mark - entry) : null,
        };
      });
    }
    return Object.entries(account.data?.positions ?? {}).map(([symbol, position]) => ({
      symbol: canonical(symbol),
      quantity: position.quantity,
      entryPrice: position.entry_price,
      currentPrice: position.mark_price,
      value: position.quantity * position.mark_price,
      pnl: position.quantity * (position.mark_price - position.entry_price),
    }));
  }, [account.data?.positions, demo.data, isDemo]);
  const selectedOrder = demo.data?.orders.find((order) => order.id === orderId);
  const selectedEvaluation = evaluations.data?.find((evaluation) =>
    evaluation.evaluation_id === evaluationId || evaluation.evaluation_id === selectedOrder?.evaluation_id,
  );
  const chartSymbol = selectedSymbol || routeSymbol || positions[0]?.symbol || "";
  useEffect(() => {
    if (!selectedSymbol && chartSymbol) setSelectedSymbol(chartSymbol);
  }, [chartSymbol, selectedSymbol]);
  const marketRequest = ranges[range];
  const market = useGetCryptoMarketIndicators({
    symbols: chartSymbol ? [chartSymbol] : [],
    interval: marketRequest.interval,
    lookback: marketRequest.lookback,
    enabled: Boolean(chartSymbol),
  });
  const marketSymbol = market.data?.symbols.find((item) => canonical(item.symbol) === canonical(chartSymbol));
  const candles = useMemo<CandlestickData[]>(() => (marketSymbol?.candles ?? []).map((candle) => ({
    time: new Date(candle.ts).toISOString(), open: candle.open, high: candle.high, low: candle.low, close: candle.close, volume: candle.volume,
  })), [marketSymbol?.candles]);
  const movingAverages = useMemo<CandlestickMovingAverage[]>(() => marketSymbol?.indicators.length ? ["ma5", "ma20", "ma60"].map((name, index) => ({
    name: name.toUpperCase(), values: marketSymbol.indicators.map((item) => item.ma[name] ?? null), color: ["#fbbf24", "#38bdf8", "#c084fc"][index],
  })) : [], [marketSymbol?.indicators]);
  const markers = useMemo<CandlestickTradeMarker[]>(() => {
    if (isDemo) {
      const orders = selectedOrder ? [selectedOrder] : demo.data?.orders ?? [];
      return orders.flatMap((order) => {
        const price = amount(order.average_fill_price);
        const time = order.filled_at ?? order.updated_at ?? order.created_at;
        return canonical(order.symbol) === canonical(chartSymbol) && price != null
          ? [{ time, price, side: order.side, label: order.side === "buy" ? "买入" : "卖出" }]
          : [];
      });
    }
    return (trades.data ?? []).flatMap((trade) => {
      if (canonical(trade.symbol) !== canonical(chartSymbol)) return [];
      if (evaluationId && trade.evaluation_id !== evaluationId) return [];
      const side = ["sell", "close", "reduce"].includes(trade.action) ? "sell" : "buy";
      return [{ time: trade.evaluated_at, price: trade.price, side, label: side === "buy" ? "买入" : "卖出" }];
    });
  }, [chartSymbol, demo.data?.orders, evaluationId, isDemo, selectedOrder, trades.data]);
  const selectedPosition = positions.find((position) => canonical(position.symbol) === canonical(chartSymbol));
  const changeSymbol = (symbol: string) => {
    setSelectedSymbol(symbol);
    setSearchParams((current) => {
      current.set("symbol", symbol);
      current.delete("orderId");
      current.delete("evaluationId");
      return current;
    });
  };

  if (!strategyId) return <main className="flex flex-1 items-center justify-center p-6 text-muted-foreground">请先选择策略。</main>;
  if (strategy.isLoading || account.isLoading || demo.isLoading) return <main className="flex flex-1 items-center justify-center p-6 text-muted-foreground">正在读取持仓事实…</main>;
  if (strategy.isError || account.isError || demo.isError) return <main className="flex flex-1 items-center justify-center p-6 text-destructive">持仓数据暂不可用，请稍后重试。</main>;
  return <main className="scroll-container flex flex-1 flex-col gap-5 p-5 lg:p-8">
    <header className="flex flex-wrap items-start justify-between gap-3"><div><p className="font-medium text-sky-600 text-sm dark:text-sky-300">{isDemo ? "OKX Demo 后台快照" : "纸面策略账户"}</p><h1 className="mt-1 font-semibold text-2xl">我的持仓</h1><p className="mt-2 text-muted-foreground text-sm">价格、盈亏、成交买入点和服务器策略条件在同一条链路中核对。</p></div><Button asChild variant="outline"><Link to="/trades">交易明细</Link></Button></header>
    {selectedOrder || selectedEvaluation ? <Card><CardHeader><CardTitle>这笔交易详情</CardTitle><CardDescription>订单执行事实与服务端策略条件对应展示；没有评估记录时不会补造原因。</CardDescription></CardHeader><CardContent className="space-y-4">{selectedOrder ? <div className="grid gap-2 rounded-md border bg-muted/30 p-3 text-sm sm:grid-cols-2"><p>订单：{selectedOrder.symbol} · {selectedOrder.side === "buy" ? "买入" : "卖出"}</p><p>状态：{selectedOrder.status}</p><p>请求金额：{money(amount(selectedOrder.requested_quote))}</p><p>成交均价：{money(amount(selectedOrder.average_fill_price))}</p>{selectedOrder.error_message || selectedOrder.error_code ? <p className="text-destructive sm:col-span-2">失败原因：{selectedOrder.error_message ?? selectedOrder.error_code}</p> : null}</div> : null}<RuleStrategyEvaluationPath evaluation={selectedEvaluation} /></CardContent></Card> : null}
    <section className="grid gap-5 xl:grid-cols-[minmax(0,1.65fr)_minmax(320px,0.85fr)]"><Card><CardHeader className="flex flex-row items-start justify-between gap-3"><div><CardTitle className="flex items-center gap-2"><CandlestickChart className="size-5 text-sky-500" />{chartSymbol.replace("-", "/")} K 线</CardTitle><CardDescription>绿色标记为买入，红色标记为卖出；黄线为当前市价。</CardDescription></div><Select onValueChange={(value) => setRange(value as Range)} value={range}><SelectTrigger className="w-20"><SelectValue /></SelectTrigger><SelectContent>{Object.keys(ranges).map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}</SelectContent></Select></CardHeader><CardContent>{market.isError ? <p className="py-20 text-center text-destructive">K 线数据暂不可用。</p> : <CandlestickChartComponent currentPrice={marketSymbol?.latest_price} data={candles} height={430} loading={market.isLoading} movingAverages={movingAverages} theme="dark" tradeMarkers={markers} />}</CardContent></Card>
    <Card><CardHeader><CardTitle>币种持仓明细</CardTitle><CardDescription>{isDemo ? "Demo 账户仓位；买入成本仅从当前策略已归属成交重建。" : "纸面账户的策略专属持仓。"}</CardDescription></CardHeader><CardContent className="space-y-2">{positions.length ? positions.map((position) => <button className="w-full rounded-md border p-3 text-left hover:border-sky-500/50 hover:bg-sky-500/5" key={position.symbol} onClick={() => changeSymbol(position.symbol)} type="button"><div className="flex justify-between gap-3"><strong>{position.symbol.replace("-", "/")}</strong><span className={position.pnl == null ? "text-muted-foreground" : position.pnl >= 0 ? "text-emerald-500" : "text-rose-500"}>{position.pnl == null ? "盈亏不可用" : money(position.pnl)}</span></div><p className="mt-1 text-muted-foreground text-xs">数量 {position.quantity} · 买入价 {money(position.entryPrice)} · 市价 {money(position.currentPrice)} · 名义价值 {money(position.value)}</p></button>) : <p className="py-8 text-center text-muted-foreground text-sm">当前没有可展示的持仓。</p>}</CardContent></Card></section>
    {selectedPosition ? <Card><CardHeader><CardTitle>当前选择：{selectedPosition.symbol.replace("-", "/")}</CardTitle><CardDescription>买入价 {money(selectedPosition.entryPrice)} · 当前市价 {money(selectedPosition.currentPrice)} · 浮动盈亏 {money(selectedPosition.pnl)}</CardDescription></CardHeader></Card> : null}
    <Card><CardHeader><CardTitle className="flex items-center gap-2"><CircleDollarSign className="size-5 text-sky-500" />策略条件与执行结果</CardTitle><CardDescription>选择交易明细后，此处显示对应 evaluation_id 的买卖原因、执行路径、条件状态及失败事实。</CardDescription></CardHeader><CardContent><RuleStrategyEvaluationPath evaluation={selectedEvaluation} /></CardContent></Card>
  </main>;
}
