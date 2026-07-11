import { useTheme } from "next-themes";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { ArrowUpRight, CircleAlert } from "lucide-react";
import { useRuleStrategy, useRuleStrategyPnlCurve, useRuleStrategySignals } from "@/api/rule-strategy";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import TradingViewTickerTape from "@/components/tradingview/tradingview-ticker-tape";
import TradingViewAdvancedChart from "@/components/tradingview/tradingview-advanced-chart";
import { PnlLineChart } from "@/components/valuecell/charts/pnl-line-chart";

const MARKET_SYMBOLS = ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:SOLUSDT"];

export default function DashboardPage() {
  const { t } = useTranslation();
  const { resolvedTheme } = useTheme();
  const strategyId = localStorage.getItem("valuecell.rule-strategy-id") ?? "";
  const { data: ruleStrategy } = useRuleStrategy(strategyId);
  const { data: signals, isLoading: signalsLoading, isError: signalsError } = useRuleStrategySignals(strategyId);
  const { data: pnlCurve } = useRuleStrategyPnlCurve(strategyId || undefined);
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

        <section aria-label={t("saas.dashboard.marketTicker")} className="overflow-hidden rounded-lg border bg-card">
          <TradingViewTickerTape symbols={MARKET_SYMBOLS} theme={resolvedTheme === "dark" ? "dark" : "light"} />
        </section>

        <section className="grid gap-4 md:grid-cols-3" aria-label={t("saas.dashboard.workspaceOverview")}>
          <Card className="gap-3 py-5">
            <CardHeader className="px-5">
              <CardDescription>{t("saas.dashboard.workspaceStatus")}</CardDescription>
              <CardTitle className="text-xl">{t("saas.dashboard.paperOnly")}</CardTitle>
            </CardHeader>
            <CardContent className="px-5 text-sm text-muted-foreground">
              {t("saas.dashboard.workspaceStatusDescription")}
            </CardContent>
          </Card>
          <Card className="gap-3 py-5">
            <CardHeader className="px-5">
              <CardDescription>{t("saas.dashboard.strategyWorkflow")}</CardDescription>
              <CardTitle className="text-xl">{t("saas.dashboard.configureRules")}</CardTitle>
            </CardHeader>
            <CardContent className="px-5 text-sm text-muted-foreground">
              {t("saas.dashboard.strategyWorkflowDescription")}
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
          <CardContent className="p-0 overflow-hidden">
            <TradingViewAdvancedChart
              ticker="BTCUSDT"
              theme={resolvedTheme === "dark" ? "dark" : "light"}
              minHeight={420}
            />
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
