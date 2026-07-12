import {
  BarChart3,
  CircleDollarSign,
  ClipboardList,
  Clock3,
  RadioTower,
  Settings2,
  ShieldAlert,
  Waves,
} from "lucide-react";
import { type FC, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { usePreviewStrategyExperiment } from "@/api/strategy";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { TIME_FORMATS, TimeUtils } from "@/lib/time";
import type {
  PortfolioSummary,
  Position,
  Strategy,
  StrategyCompose,
  StrategyDiagnostics,
} from "@/types/strategy";
import CryptoMarketIndicatorPanel from "./crypto-market-indicator-panel";
import PortfolioPositionsGroup from "./portfolio-positions-group";
import StrategyComposeList from "./strategy-compose-list";
import StrategyDiagnosticsPanel from "./strategy-diagnostics-panel";

interface StrategyWorkspaceProps {
  composes: StrategyCompose[];
  diagnostics?: StrategyDiagnostics;
  positions: Position[];
  priceCurve: Array<Array<number | string>>;
  strategy: Strategy;
  strategySymbols: string[];
  summary?: PortfolioSummary;
}

const StrategyWorkspace: FC<StrategyWorkspaceProps> = ({
  composes,
  diagnostics,
  positions,
  priceCurve,
  strategy,
  strategySymbols,
  summary,
}) => {
  const { t } = useTranslation();
  const health = diagnostics?.latest_cycle?.market_data_health;
  const latestCycle = diagnostics?.latest_cycle;
  const expectedSymbolCount = diagnostics?.expected_symbol_count ?? 0;
  const observedSymbolCount = diagnostics?.observed_symbol_count ?? 0;
  const isMarketHealthy = health?.status
    ? health.status === "healthy"
    : health?.ok ??
      (expectedSymbolCount > 0 && observedSymbolCount >= expectedSymbolCount);
  const hasHealthData = Boolean(health) || expectedSymbolCount > 0;
  const exposureIncreaseAllowed = health?.exposure_increase_allowed ?? isMarketHealthy;
  const isPaper = strategy.trading_mode === "virtual";
  const marketHealthLabel = !hasHealthData
    ? t("strategy.workspace.healthAwaiting", { defaultValue: "Data health awaiting first scan" })
    : !exposureIncreaseAllowed
      ? t("strategy.workspace.healthExposureBlocked", { defaultValue: "Exposure increase blocked" })
      : isMarketHealthy
        ? t("strategy.workspace.healthHealthy", { defaultValue: "Market data healthy" })
        : t("strategy.workspace.healthDegraded", { defaultValue: "Market data degraded" });
  const [experimentInput, setExperimentInput] = useState("");
  const [experimentInputError, setExperimentInputError] = useState<string | null>(null);
  const { data: experimentPreview, error: experimentError, isPending: isPreviewPending, mutate: previewExperiment } = usePreviewStrategyExperiment();
  const canPreviewExperiment = strategy.strategy_type === "LongTermSpotRsiStrategy" || strategy.strategy_type === "ShortTermSpotRsiStrategy";
  const currentParameters = useMemo(() => {
    const params = diagnostics?.config?.strategy_params;
    return params && typeof params === "object" && !Array.isArray(params)
      ? params as Record<string, unknown>
      : {};
  }, [diagnostics?.config?.strategy_params]);
  const submitExperimentPreview = () => {
    if (!canPreviewExperiment) return;
    let parameters = currentParameters;
    setExperimentInputError(null);
    if (experimentInput.trim()) {
      try {
        const parsed = JSON.parse(experimentInput);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("Parameters must be a JSON object.");
        }
        parameters = parsed as Record<string, unknown>;
      } catch (error) {
        setExperimentInputError(
          error instanceof Error
            ? error.message
            : "Parameters must be a JSON object.",
        );
        return;
      }
    }
    previewExperiment({
      strategy_type: strategy.strategy_type as "LongTermSpotRsiStrategy" | "ShortTermSpotRsiStrategy",
      parameters,
    });
  };

  return (
    <div className="flex min-w-0 flex-1 flex-col overflow-hidden bg-background">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b bg-card px-5 py-4 lg:px-6">
        <div className="min-w-0">
          <p className="text-muted-foreground text-xs">{strategy.strategy_type}</p>
          <h2 className="truncate font-semibold text-lg">{strategy.strategy_name}</h2>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge variant={isPaper ? "secondary" : "outline"} className="gap-1.5">
            {isPaper ? <ShieldAlert className="size-3.5" /> : null}
            {isPaper
              ? t("strategy.workspace.paperMode", { defaultValue: "Paper mode" })
              : t("strategy.workspace.legacyLiveMode", { defaultValue: "Legacy live mode" })}
          </Badge>
          <Badge
            variant={hasHealthData && !exposureIncreaseAllowed ? "destructive" : "outline"}
            className="gap-1.5"
          >
            <RadioTower className="size-3.5" />
            {marketHealthLabel}
          </Badge>
          <Badge variant={strategy.status === "running" ? "default" : "outline"} className="gap-1.5">
            <span className={`size-1.5 rounded-full ${strategy.status === "running" ? "bg-emerald-300" : "bg-muted-foreground"}`} />
            {strategy.status === "running" ? t("strategy.status.running") : t("strategy.status.stopped")}
          </Badge>
        </div>
        <p className="flex w-full items-center gap-1.5 text-muted-foreground text-xs">
          <Clock3 className="size-3.5" />
          {latestCycle?.created_at
            ? t("strategy.workspace.lastMarketCheck", {
                defaultValue: "Last successful data-health check: {{time}}",
                time: TimeUtils.formatUTC(latestCycle.created_at, TIME_FORMATS.DATETIME),
              })
            : t("strategy.workspace.lastMarketCheckPending", { defaultValue: "Data freshness unavailable until the first successful market-data check" })}
        </p>
      </header>

      <Tabs defaultValue="overview" className="min-h-0 flex-1 gap-0">
        <div className="overflow-x-auto border-b bg-card px-4 lg:px-6">
          <TabsList className="h-11 rounded-none bg-transparent p-0">
            <TabsTrigger value="overview" className="rounded-none px-4 data-[state=active]:border-primary data-[state=active]:border-b-2 data-[state=active]:shadow-none">
              <BarChart3 />{t("strategy.workspace.overview", { defaultValue: "Overview" })}
            </TabsTrigger>
            <TabsTrigger value="execution" className="rounded-none px-4 data-[state=active]:border-primary data-[state=active]:border-b-2 data-[state=active]:shadow-none">
              <ClipboardList />{t("strategy.workspace.execution", { defaultValue: "Execution" })}
            </TabsTrigger>
            <TabsTrigger value="market" className="rounded-none px-4 data-[state=active]:border-primary data-[state=active]:border-b-2 data-[state=active]:shadow-none">
              <Waves />{t("strategy.workspace.market", { defaultValue: "Market" })}
            </TabsTrigger>
            <TabsTrigger value="portfolio" className="rounded-none px-4 data-[state=active]:border-primary data-[state=active]:border-b-2 data-[state=active]:shadow-none">
              <CircleDollarSign />{t("strategy.workspace.portfolio", { defaultValue: "Portfolio" })}
            </TabsTrigger>
            <TabsTrigger value="configuration" className="rounded-none px-4 data-[state=active]:border-primary data-[state=active]:border-b-2 data-[state=active]:shadow-none">
              <Settings2 />{t("strategy.workspace.configuration", { defaultValue: "Configuration" })}
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="overview" className="scroll-container min-h-0 overflow-y-auto p-4 lg:p-6">
          {hasHealthData && !exposureIncreaseAllowed ? (
            <Alert variant="destructive" className="mb-4">
              <ShieldAlert />
              <AlertTitle>{t("strategy.workspace.exposureBlocked", { defaultValue: "Exposure increases blocked" })}</AlertTitle>
              <AlertDescription>
                {health?.stale_symbols?.length
                  ? t("strategy.workspace.staleMarketData", {
                      defaultValue: "Market data is stale for: {{symbols}}. Paper strategy may reduce or close exposure, but cannot increase it until freshness recovers.",
                      symbols: health.stale_symbols.join(", "),
                    })
                  : health?.missing_symbols.length
                    ? t("strategy.workspace.missingMarketData", {
                        defaultValue: "Market data is unavailable for: {{symbols}}. Paper strategy may reduce or close exposure, but cannot increase it until coverage recovers.",
                        symbols: health.missing_symbols.join(", "),
                      })
                    : t("strategy.workspace.incompleteMarketData", {
                        defaultValue: "Market data is incomplete or degraded. Paper strategy may reduce or close exposure, but cannot increase it until coverage and freshness recover.",
                      })}
              </AlertDescription>
            </Alert>
          ) : null}
          <StrategyDiagnosticsPanel
            diagnostics={diagnostics}
            priceCurve={priceCurve}
            positions={positions}
            strategy={strategy}
            summary={summary}
            view="overview"
          />
        </TabsContent>
        <TabsContent value="execution" className="grid min-h-0 grid-cols-1 overflow-hidden xl:grid-cols-[minmax(340px,0.8fr)_minmax(0,1.2fr)]">
          <StrategyComposeList composes={composes} tradingMode={strategy.trading_mode} />
          <div className="scroll-container min-w-0 overflow-y-auto p-4 lg:p-6">
            <StrategyDiagnosticsPanel diagnostics={diagnostics} positions={positions} view="execution" />
          </div>
        </TabsContent>
        <TabsContent value="market" className="scroll-container min-h-0 overflow-y-auto p-4 lg:p-6">
          <CryptoMarketIndicatorPanel
            strategyRefreshIntervalSeconds={undefined}
            strategySymbols={strategySymbols}
          />
        </TabsContent>
        <TabsContent value="portfolio" className="scroll-container min-h-0 overflow-y-auto">
          <PortfolioPositionsGroup summary={summary} priceCurve={priceCurve} positions={positions} strategy={strategy} />
        </TabsContent>
        <TabsContent value="configuration" className="scroll-container min-h-0 overflow-y-auto p-4 lg:p-6">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="font-semibold text-base">{t("strategy.workspace.runtimeConfiguration", { defaultValue: "Runtime configuration" })}</h3>
              <p className="mt-1 text-muted-foreground text-sm">
                {t("strategy.workspace.runtimeConfigurationDesc", { defaultValue: "Read-only parameters currently used by this strategy. Create a paper experiment to test a new parameter set." })}
              </p>
            </div>
            <Badge variant="outline">{t("strategy.workspace.readOnly", { defaultValue: "Read-only" })}</Badge>
          </div>
          <div className="mb-4 rounded-lg border bg-muted/30 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h4 className="font-medium text-sm">{t("strategy.workspace.paperExperiment", { defaultValue: "Paper parameter preview" })}</h4>
                <p className="mt-1 text-muted-foreground text-xs">
                  {t("strategy.workspace.paperExperimentDesc", { defaultValue: "Validate a candidate RSI parameter set and receive a reproducible fingerprint. This is not a backtest and returns no performance projection." })}
                </p>
              </div>
              <Badge variant="secondary">paper only</Badge>
            </div>
            {canPreviewExperiment ? (
              <>
                <Input
                  className="mt-3 font-mono text-xs"
                  value={experimentInput}
                  onChange={(event) => setExperimentInput(event.target.value)}
                  placeholder='Optional JSON overrides, e.g. {"entry_rsi_thresholds":[30,25,20]}'
                />
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <Button type="button" size="sm" onClick={submitExperimentPreview} disabled={isPreviewPending}>
                    {isPreviewPending ? t("strategy.workspace.validating", { defaultValue: "Validating…" }) : t("strategy.workspace.validateCandidate", { defaultValue: "Validate candidate" })}
                  </Button>
                  <span className="text-muted-foreground text-xs">{Object.keys(currentParameters).length > 0 ? t("strategy.workspace.usingCurrentOverrides", { defaultValue: "Uses current overrides when JSON is empty." }) : t("strategy.workspace.usingProfileDefaults", { defaultValue: "Uses the strategy profile defaults when JSON is empty." })}</span>
                </div>
              </>
            ) : (
              <p className="mt-3 text-muted-foreground text-sm">{t("strategy.workspace.previewSupportedRsiOnly", { defaultValue: "Parameter preview is currently available for the paper RSI strategies only." })}</p>
            )}
            {experimentInputError ? <p className="mt-3 text-destructive text-sm">{experimentInputError}</p> : null}
            {experimentError ? <p className="mt-3 text-destructive text-sm">{experimentError instanceof Error ? experimentError.message : t("strategy.workspace.previewFailed", { defaultValue: "Paper parameter preview could not be validated." })}</p> : null}
            {experimentPreview?.data ? (
              <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(16rem,0.7fr)]">
                <div className="rounded-md border bg-background p-3">
                  <p className="text-muted-foreground text-xs">{t("strategy.workspace.canonicalParameters", { defaultValue: "Canonical paper parameters" })}</p>
                  <pre className="mt-2 overflow-x-auto text-xs">{JSON.stringify(experimentPreview.data.parameters, null, 2)}</pre>
                </div>
                <div className="space-y-2 rounded-md border bg-background p-3 text-sm">
                  <p className="text-muted-foreground text-xs">fingerprint</p>
                  <p className="break-all font-mono text-xs">{experimentPreview.data.fingerprint}</p>
                  <p>{t("strategy.workspace.riskLevel", { defaultValue: "Risk level" })}: {experimentPreview.data.candidate_summary.risk_level}</p>
                  <p>{t("strategy.workspace.entrySteps", { defaultValue: "Entry steps" })}: {experimentPreview.data.candidate_summary.entry_steps}</p>
                  <p>{t("strategy.workspace.exitSteps", { defaultValue: "Exit steps" })}: {experimentPreview.data.candidate_summary.exit_steps}</p>
                  <p>{t("strategy.workspace.maxExposure", { defaultValue: "Maximum exposure" })}: {experimentPreview.data.candidate_summary.max_exposure_ratio}</p>
                  {experimentPreview.data.warnings.map((warning) => <p key={warning} className="text-amber-700 dark:text-amber-300">{warning}</p>)}
                  {experimentPreview.data.diagnostics.map((diagnostic) => <p key={`${diagnostic.code}-${diagnostic.field ?? ""}`} className={diagnostic.severity === "error" ? "text-destructive" : diagnostic.severity === "warning" ? "text-amber-700 dark:text-amber-300" : "text-muted-foreground"}>{diagnostic.message}</p>)}
                </div>
              </div>
            ) : null}
          </div>
          <StrategyDiagnosticsPanel diagnostics={diagnostics} strategy={strategy} summary={summary} view="configuration" />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default StrategyWorkspace;
