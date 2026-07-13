import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { BarChart3, ExternalLink, RadioTower } from "lucide-react";
import { useGetCryptoMarketIndicators } from "@/api/crypto-market";
import { MarketIndicatorPanelChart, type MarketIndicatorPanel } from "@/components/valuecell/charts/market-indicator-panel";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
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
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import CandlestickChart, { type CandlestickData, type CandlestickMovingAverage } from "@/components/valuecell/charts/candlestick-chart";

const SYMBOLS = [
  { value: "BTC-USDT", labelKey: "saas.charts.symbols.bitcoin" },
  { value: "ETH-USDT", labelKey: "saas.charts.symbols.ether" },
  { value: "SOL-USDT", labelKey: "saas.charts.symbols.solana" },
] as const;

const INTERVALS = [
  { value: "15m", label: "15m" }, { value: "1h", label: "1h" },
  { value: "4h", label: "4h" }, { value: "1d", label: "1D" },
  { value: "1w", label: "1W" }, { value: "1M", label: "1M" },
  { value: "3M", label: "3M" }, { value: "1Y", label: "1Y" },
] as const;
const HISTORY_RANGES = [
  { value: "10d", label: "10D", days: 10 }, { value: "30d", label: "30D", days: 30 },
  { value: "90d", label: "90D", days: 90 }, { value: "1y", label: "1Y", days: 365 },
] as const;
const INTERVAL_SECONDS: Record<string, number> = {
  "15m": 900, "1h": 3_600, "4h": 14_400, "1d": 86_400,
  "1w": 604_800, "1M": 2_592_000, "3M": 7_776_000, "1Y": 31_536_000,
};
type HistoryRange = typeof HISTORY_RANGES[number]["value"];

export default function ChartsPage() {
  const { t } = useTranslation();
  const [symbol, setSymbol] = useState<string>(SYMBOLS[0].value);
  const [interval, setInterval] = useState("1h");
  const [indicatorPanel, setIndicatorPanel] = useState<MarketIndicatorPanel>("rsi");
  const [historyRange, setHistoryRange] = useState<HistoryRange>("10d");
  const [fromDate, setFromDate] = useState("");
  const [requestNowMs, setRequestNowMs] = useState(() => Date.now());
  const [toDate, setToDate] = useState("");
  const fromTsMs = useMemo(() => {
    if (fromDate) return new Date(`${fromDate}T00:00:00Z`).getTime();
    const days = HISTORY_RANGES.find((range) => range.value === historyRange)?.days ?? 10;
    return requestNowMs - days * 24 * 60 * 60 * 1000;
  }, [fromDate, historyRange, requestNowMs]);
  const toTsMs = useMemo(() => (
    toDate ? new Date(`${toDate}T23:59:59.999Z`).getTime() : requestNowMs
  ), [requestNowMs, toDate]);
  const lookback = useMemo(() => Math.min(
    5_000,
    Math.ceil((toTsMs - fromTsMs) / (INTERVAL_SECONDS[interval] * 1000)) + 2,
  ), [fromTsMs, interval, toTsMs]);
  const { data, isFetching, isError } = useGetCryptoMarketIndicators({
    symbols: [symbol],
    interval,
    lookback,
    fromTsMs,
    toTsMs,
  });
  const market = data?.symbols.find((item) => item.symbol === symbol);
  const failedReason = data?.failed_symbols?.[symbol];
  const candles = useMemo<CandlestickData[]>(() => market?.candles.map((candle) => ({
    time: new Date(candle.ts).toISOString(),
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume,
  })) ?? [], [market]);
  const movingAverages = useMemo<CandlestickMovingAverage[]>(() => {
    if (!market?.indicators.length) return [];
    const overlays = [
      { key: "ma5", name: "MA5", color: "#f59e0b" },
      { key: "ma20", name: "MA20", color: "#3b82f6" },
      { key: "ma60", name: "MA60", color: "#a855f7" },
      { key: "upper", name: "BB Upper", color: "#94a3b8" },
      { key: "middle", name: "BB Middle", color: "#64748b" },
      { key: "lower", name: "BB Lower", color: "#94a3b8" },
    ] as const;
    return overlays.map((overlay) => ({
      name: overlay.name,
      color: overlay.color,
      values: market.indicators.map((indicator) => {
        if (overlay.key === "ma5" || overlay.key === "ma20" || overlay.key === "ma60") {
          return indicator.ma[overlay.key] ?? null;
        }
        return indicator.bollinger[overlay.key] ?? null;
      }),
    }));
  }, [market]);

  return (
    <div className="scroll-container size-full bg-muted/40">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-4 md:p-6 lg:p-8">
        <header className="flex flex-col gap-2">
          <div className="flex items-center gap-2"><Badge variant="secondary">{t("saas.charts.researchChart")}</Badge><Badge variant="outline">{t("saas.charts.paperOnly")}</Badge></div>
          <h1 className="text-2xl font-semibold tracking-tight">{t("saas.charts.title")}</h1>
          <p className="text-sm text-muted-foreground">{t("saas.charts.subtitle")}</p>
        </header>

        <Card>
          <CardHeader className="gap-4 sm:flex sm:flex-row sm:items-end sm:justify-between">
            <div>
              <CardTitle className="flex items-center gap-2"><BarChart3 className="size-5" /> {t("saas.charts.marketChart")}</CardTitle>
              <CardDescription className="mt-1">{market?.latest_price != null ? t("saas.charts.latestPrice", { price: market.latest_price.toLocaleString() }) : t("saas.charts.marketSource")}</CardDescription>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row">
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger className="w-full sm:w-52"><SelectValue /></SelectTrigger>
                <SelectContent>{SYMBOLS.map((option) => <SelectItem key={option.value} value={option.value}>{t(option.labelKey)}</SelectItem>)}</SelectContent>
              </Select>
              <Select value={interval} onValueChange={setInterval}>
                <SelectTrigger className="w-full sm:w-24"><SelectValue /></SelectTrigger>
                <SelectContent>{INTERVALS.map((option) => <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <div className="flex flex-wrap gap-1" aria-label="Historical range">
              {HISTORY_RANGES.map((range) => <Button key={range.value} onClick={() => { setHistoryRange(range.value); setFromDate(""); setToDate(""); setRequestNowMs(Date.now()); }} size="sm" type="button" variant={historyRange === range.value ? "secondary" : "ghost"}>{range.label}</Button>)}
            </div>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Input aria-label="Start date" className="h-8 w-36" onChange={(event) => { setFromDate(event.target.value); setRequestNowMs(Date.now()); }} type="date" value={fromDate} />
              <span className="text-muted-foreground">to</span>
              <Input aria-label="End date" className="h-8 w-36" onChange={(event) => { setToDate(event.target.value); setRequestNowMs(Date.now()); }} type="date" value={toDate} />
            </div>
          </CardHeader>
          <CardContent className="px-0 pb-2 sm:px-2">
            <div className="mb-3 flex justify-end gap-2 px-4 sm:px-2">
              <Badge variant={isError || failedReason ? "destructive" : isFetching ? "outline" : "secondary"} className="gap-1"><RadioTower className="size-3" />{market?.provider ?? "market"}</Badge>
              {market?.freshness_status === "stale" ? <Badge variant="outline">Data delayed</Badge> : null}
            </div>
            {isError || failedReason ? (
              <p className="px-6 py-28 text-center text-sm text-destructive">{t("saas.charts.marketUnavailable")}{failedReason ? `：${failedReason}` : ""}</p>
            ) : !market && !isFetching ? (
              <p className="py-28 text-center text-sm text-muted-foreground">{t("saas.charts.marketUnavailable")}</p>
            ) : (
              <CandlestickChart data={candles} movingAverages={movingAverages} loading={isFetching} height={430} />
            )}
            {market?.indicators.length ? (
              <div className="border-t px-4 py-4 sm:px-5">
                <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="font-medium text-sm">Technical indicator</p>
                    <p className="text-muted-foreground text-xs">Derived from the same exchange OHLCV snapshot as the candlestick chart.</p>
                  </div>
                  <Tabs onValueChange={(value) => setIndicatorPanel(value as MarketIndicatorPanel)} value={indicatorPanel}>
                    <TabsList aria-label="Technical indicator panel">
                      <TabsTrigger value="rsi">RSI</TabsTrigger>
                      <TabsTrigger value="momentum">Momentum</TabsTrigger>
                      <TabsTrigger value="macd">MACD</TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>
                <MarketIndicatorPanelChart data={market.indicators} panel={indicatorPanel} />
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card className="gap-3 py-5">
          <CardHeader className="px-5"><CardTitle className="text-base">{t("saas.charts.researchContext")}</CardTitle></CardHeader>
          <CardContent className="flex flex-col gap-3 px-5 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
            <p>{t("saas.charts.researchContextDescription")}</p>
            <Button asChild variant="outline" size="sm" className="shrink-0"><a href="https://www.tradingview.com/" rel="noreferrer" target="_blank">{t("saas.charts.openTradingView")} <ExternalLink /></a></Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
