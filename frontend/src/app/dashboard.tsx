import { useTheme } from "next-themes";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { ArrowUpRight, CircleAlert, RadioTower } from "lucide-react";
import { useRuleStrategy, useRuleStrategyPnlCurve, useRuleStrategySignals } from "@/api/rule-strategy";
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
import CandlestickChart, { type CandlestickData, type CandlestickMovingAverage } from "@/components/valuecell/charts/candlestick-chart";
import { PnlLineChart } from "@/components/valuecell/charts/pnl-line-chart";
export default function DashboardPage() {
  const { t } = useTranslation();
  const { resolvedTheme } = useTheme();
  const strategyId = localStorage.getItem("valuecell.rule-strategy-id") ?? "";
  const { data: ruleStrategy } = useRuleStrategy(strategyId);
  const { data: signals, isLoading: signalsLoading, isError: signalsError } = useRuleStrategySignals(strategyId);
  const { data: pnlCurve } = useRuleStrategyPnlCurve(strategyId || undefined);
  const { data: marketData, isFetching: marketLoading, isError: marketError } = useGetCryptoMarketIndicators({
    symbols: ["BTC-USDT"],
    interval: "1h",
    lookback: 240,
  });
  const market = marketData?.symbols[0];
  const marketFailure = marketData?.failed_symbols["BTC-USDT"];
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
      color: ["#f59e0b", "#3b82f6", "#a855f7"][index],
    }))
    : [];
  const pnlPoints = pnlCurve ?? [];

  return (
    <div className="scroll-container flex size-full flex-col bg-muted/40">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-4 md:p-6 lg:p-8">
        <header className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
          <div>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <Badge variant="secondary">{t("saas.dashboard.paperWorkspace")}</Badge>
              <Badge variant="outline">{t("saas.dashboard.researchOnly")}</Badge>
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">{t("saas.dashboard.title")}</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              {t("saas.dashboard.subtitle")}
            </p>
          </div>
          <Button asChild className="sm:mt-1">
            <Link to="/strategies">{t("saas.dashboard.reviewStrategies")} <ArrowUpRight /></Link>
          </Button>
        </header>


        <section className="grid gap-4 md:grid-cols-3" aria-label={t("saas.dashboard.workspaceOverview")}>
          <Card className="gap-3 py-5">
            <CardHeader className="px-5">
              <CardDescription>Paper account equity</CardDescription>
              <CardTitle className="text-xl">
                {ruleStrategy ? `${ruleStrategy.account.equity_quote.toFixed(2)} USDT` : "No paper account"}
              </CardTitle>
            </CardHeader>
            <CardContent className="px-5 text-sm text-muted-foreground">
              {ruleStrategy
                ? `Initial ${ruleStrategy.account.initial_capital_quote.toFixed(2)} | Cash ${ruleStrategy.account.quote_balance.toFixed(2)}`
                : t("saas.dashboard.saveStrategyPrompt")}
            </CardContent>
          </Card>
          <Card className="gap-3 py-5">
            <CardHeader className="px-5">
              <CardDescription>Paper profit and loss</CardDescription>
              <CardTitle className="text-xl">
                {ruleStrategy
                  ? `${(ruleStrategy.account.realized_pnl_quote + ruleStrategy.account.unrealized_pnl_quote).toFixed(2)} USDT`
                  : "--"}
              </CardTitle>
            </CardHeader>
            <CardContent className="px-5 text-sm text-muted-foreground">
              {ruleStrategy
                ? `Realized ${ruleStrategy.account.realized_pnl_quote.toFixed(2)} | Unrealized ${ruleStrategy.account.unrealized_pnl_quote.toFixed(2)}`
                : t("saas.dashboard.noStrategySelected")}
            </CardContent>
          </Card>
          <Card className="gap-3 py-5">
            <CardHeader className="px-5">
              <CardDescription>{t("saas.dashboard.paperStrategyActivity")}</CardDescription>
              <CardTitle className="text-xl">{ruleStrategy ? ruleStrategy.status === "running" ? t("saas.dashboard.evaluationActive") : t("saas.dashboard.evaluationStopped") : t("saas.dashboard.noStrategySaved")}</CardTitle>
            </CardHeader>
            <CardContent className="px-5 text-sm text-muted-foreground">
              {ruleStrategy ? t("saas.dashboard.showingRecords", { name: ruleStrategy.name }) : t("saas.dashboard.saveStrategyPrompt")}
            </CardContent>
          </Card>
        </section>

        <Card>
          <CardHeader className="gap-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle>{t("saas.dashboard.marketChart")}</CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant={marketFailure || marketError ? "destructive" : marketLoading ? "outline" : "secondary"} className="gap-1">
                  <RadioTower className="size-3" />
                  {market?.provider ?? "market"}
                </Badge>
                {market?.freshness_status === "stale" ? <Badge variant="outline">Data delayed</Badge> : null}
              </div>
            </div>
            <CardDescription>
              {market?.latest_price != null
                ? t("saas.dashboard.latestPrice", { price: market.latest_price.toLocaleString() })
                : t("saas.dashboard.marketSource")}
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0 overflow-hidden">
            {marketError || marketFailure ? (
              <p className="py-20 text-center text-sm text-destructive">{t("saas.dashboard.marketUnavailable")}</p>
            ) : !market && !marketLoading ? (
              <p className="py-20 text-center text-sm text-muted-foreground">{t("saas.dashboard.marketUnavailable")}</p>
            ) : (
              <CandlestickChart data={candles} movingAverages={movingAverages} loading={marketLoading} height={420} />
            )}
          </CardContent>
        </Card>

        <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <Card>
            <CardHeader>
              <CardTitle>{t("saas.dashboard.cumulativePnl")}</CardTitle>
              <CardDescription>{t("saas.dashboard.cumulativePnlDescription")}</CardDescription>
            </CardHeader>
            <CardContent>
              {pnlPoints.length === 0 ? (
                <p className="py-8 text-center text-sm text-muted-foreground">{t("saas.dashboard.noEvaluationData")}</p>
              ) : (
                <PnlLineChart data={pnlPoints} height={200} theme={resolvedTheme === "dark" ? "dark" : "light"} />
              )}
            </CardContent>
          </Card>

          <Card className="border-amber-500/30 bg-amber-500/5">
            <CardHeader>
              <div className="flex items-center gap-2">
                <CircleAlert className="size-4 text-amber-700 dark:text-amber-400" />
                <CardTitle>{t("saas.dashboard.latestRuleConditions")}</CardTitle>
              </div>
              <CardDescription>{t("saas.dashboard.latestRuleConditionsDescription")}</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-2 text-sm text-muted-foreground">
              {!strategyId ? <p>{t("saas.dashboard.noStrategySelected")}</p> : signalsLoading ? <p>{t("saas.dashboard.loadingSignals")}</p> : signalsError ? <p>{t("saas.dashboard.signalsUnavailable")}</p> : signals?.length ? signals.slice(0, 4).map((signal) => <div className="flex items-start justify-between gap-3" key={`${signal.evaluation_id}-${signal.code}`}><span className="min-w-0">{signal.detail}</span><Badge className="shrink-0 capitalize" variant="outline">{t(`saas.dashboard.signalStates.${signal.state}`, { defaultValue: signal.state.replace("_", " ") })}</Badge></div>) : <p>{t("saas.dashboard.noConditions")}</p>}
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}
