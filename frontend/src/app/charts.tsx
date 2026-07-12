import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { BarChart3, ExternalLink, RadioTower } from "lucide-react";
import { useGetCryptoMarketIndicators } from "@/api/crypto-market";
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
import CandlestickChart, { type CandlestickData, type CandlestickMovingAverage } from "@/components/valuecell/charts/candlestick-chart";

const SYMBOLS = [
  { value: "BTC-USDT", labelKey: "saas.charts.symbols.bitcoin" },
  { value: "ETH-USDT", labelKey: "saas.charts.symbols.ether" },
  { value: "SOL-USDT", labelKey: "saas.charts.symbols.solana" },
] as const;

const INTERVALS = [
  { value: "15m", labelKey: "saas.charts.intervals.fifteenMinutes" },
  { value: "1h", labelKey: "saas.charts.intervals.oneHour" },
  { value: "4h", labelKey: "saas.charts.intervals.fourHours" },
  { value: "1d", labelKey: "saas.charts.intervals.oneDay" },
] as const;

export default function ChartsPage() {
  const { t } = useTranslation();
  const [symbol, setSymbol] = useState<string>(SYMBOLS[0].value);
  const [interval, setInterval] = useState("1h");
  const { data, isFetching, isError } = useGetCryptoMarketIndicators({
    symbols: [symbol],
    interval,
    lookback: 240,
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
  const movingAverages = useMemo<CandlestickMovingAverage[]>(() => market?.indicators.length
    ? ["ma5", "ma20", "ma60"].map((key, index) => ({
      name: key.toUpperCase(),
      values: market.indicators.map((indicator) => indicator.ma[key] ?? null),
      color: ["#f59e0b", "#3b82f6", "#a855f7"][index],
    }))
    : [], [market]);

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
                <SelectTrigger className="w-full sm:w-32"><SelectValue /></SelectTrigger>
                <SelectContent>{INTERVALS.map((option) => <SelectItem key={option.value} value={option.value}>{t(option.labelKey)}</SelectItem>)}</SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent className="px-0 pb-2 sm:px-2">
            <div className="mb-3 flex justify-end gap-2 px-4 sm:px-2">
              <Badge variant={isError || failedReason ? "destructive" : isFetching ? "outline" : "secondary"} className="gap-1"><RadioTower className="size-3" />{market?.provider ?? "market"}</Badge>
              {market?.freshness_status === "stale" ? <Badge variant="outline">Data delayed</Badge> : null}
            </div>
            {isError || failedReason ? (
              <p className="py-28 text-center text-sm text-destructive">{t("saas.charts.marketUnavailable")}</p>
            ) : !market && !isFetching ? (
              <p className="py-28 text-center text-sm text-muted-foreground">{t("saas.charts.marketUnavailable")}</p>
            ) : (
              <CandlestickChart data={candles} movingAverages={movingAverages} loading={isFetching} height={520} />
            )}
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
