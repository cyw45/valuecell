import {
  Activity,
  AlertTriangle,
  BarChart3,
  CandlestickChart,
  CircleDollarSign,
  Gauge,
  Layers3,
  LockKeyhole,
  Percent,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  TrendingUp,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { useGetCryptoSymbols } from "@/api/crypto-market";
import {
  useCreateRuleStrategy,
  useEvaluateRuleStrategy,
  useRuleStrategy,
  useStartRuleStrategy,
  useStopRuleStrategy,
  useUpdateRuleStrategy,
} from "@/api/rule-strategy";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import type {
  EvaluateRuleStrategyRequest,
  RuleConditionState,
  RuleStrategyConfig,
  RuleStrategyEvaluation,
  RuleStrategyInterval,
} from "@/types/rule-strategy";

const TIMEFRAME_OPTIONS: RuleStrategyInterval[] = ["5m", "15m", "1h", "4h", "1d"];


type StrategyFormValues = {
  symbols: string[];
  timeframe: RuleStrategyInterval;
  fastMa: number;
  slowMa: number;
  rsiEnabled: boolean;
  rsiPeriod: number;
  rsiOversold: number;
  rsiOverbought: number;
  bollingerEnabled: boolean;
  bollingerPeriod: number;
  bollingerDeviation: number;
  momentumEnabled: boolean;
  macdFast: number;
  macdSlow: number;
  macdSignal: number;
  initialCapital: number;
  positionSize: number;
  takeProfit: number;
  stopLoss: number;
  maximumPositions: number;
  leverage: number;
};

const initialValues: StrategyFormValues = {
  symbols: ["BTC/USDT", "ETH/USDT"],
  timeframe: "1h",
  fastMa: 20,
  slowMa: 50,
  rsiEnabled: true,
  rsiPeriod: 14,
  rsiOversold: 30,
  rsiOverbought: 70,
  bollingerEnabled: true,
  bollingerPeriod: 20,
  bollingerDeviation: 2,
  momentumEnabled: true,
  macdFast: 12,
  macdSlow: 26,
  macdSignal: 9,
  initialCapital: 10_000,
  positionSize: 5,
  takeProfit: 4,
  stopLoss: 2,
  maximumPositions: 2,
  leverage: 1,
};

function NumericField({
  id,
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  hint,
  error,
  disabled = false,
  unit,
}: {
  id: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  hint?: string;
  error?: string;
  disabled?: boolean;
  unit?: string;
}) {
  return (
    <div className="grid gap-1.5">
      <Label htmlFor={id}>{label}</Label>
      <div className="relative">
        <Input
          aria-describedby={error ? `${id}-error` : hint ? `${id}-hint` : undefined}
          aria-invalid={Boolean(error)}
          disabled={disabled}
          id={id}
          inputMode="decimal"
          max={max}
          min={min}
          onChange={(event) => onChange(Number(event.target.value))}
          step={step}
          type="number"
          value={Number.isFinite(value) ? value : ""}
        />
        {unit ? (
          <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-muted-foreground text-xs">
            {unit}
          </span>
        ) : null}
      </div>
      {error ? (
        <p className="text-destructive text-xs" id={`${id}-error`} role="alert">
          {error}
        </p>
      ) : hint ? (
        <p className="text-muted-foreground text-xs" id={`${id}-hint`}>
          {hint}
        </p>
      ) : null}
    </div>
  );
}

function RuleHeading({
  icon: Icon,
  title,
  description,
  enabled,
  onEnabledChange,
}: {
  icon: typeof Activity;
  title: string;
  description: string;
  enabled?: boolean;
  onEnabledChange?: (enabled: boolean) => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex items-start justify-between gap-4 border-b px-4 py-4 sm:px-5">
      <div className="flex min-w-0 gap-3">
        <div className="mt-0.5 rounded-md bg-muted p-2 text-muted-foreground">
          <Icon aria-hidden="true" className="size-4" />
        </div>
        <div>
          <h2 className="font-semibold text-base">{title}</h2>
          <p className="mt-1 text-muted-foreground text-sm">{description}</p>
        </div>
      </div>
      {onEnabledChange ? (
        <div className="flex shrink-0 items-center gap-2">
          <Label className="text-muted-foreground text-xs" htmlFor={`${title}-enabled`}>
            {enabled ? t("saas.operations.strategy.included") : t("saas.operations.strategy.off")}
          </Label>
          <Switch
            aria-label={t("saas.operations.strategy.ruleToggle", { title, status: enabled ? t("saas.operations.strategy.included") : t("saas.operations.strategy.off") })}
            id={`${title}-enabled`}
            onCheckedChange={onEnabledChange}
          />
        </div>
      ) : null}
    </div>
  );
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 py-2 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium tabular-nums">{value}</span>
    </div>
  );
}
function toRuleStrategyConfig(values: StrategyFormValues): RuleStrategyConfig {
  return {
    mode: "paper",
    initial_capital_quote: values.initialCapital,
    confirmation_mode: "all",
    symbols: values.symbols.map((symbol) => symbol.replace("/", "-")),
    interval: values.timeframe,
    decide_interval_s: null,
    moving_average: { enabled: true, short_window: values.fastMa, long_window: values.slowMa },
    rsi: { enabled: values.rsiEnabled, period: values.rsiPeriod, oversold: values.rsiOversold, overbought: values.rsiOverbought },
    bollinger: { enabled: values.bollingerEnabled, period: values.bollingerPeriod, standard_deviations: values.bollingerDeviation },
    momentum_macd: { enabled: values.momentumEnabled, momentum_period: values.macdSignal, macd_fast_window: values.macdFast, macd_slow_window: values.macdSlow, macd_signal_window: values.macdSignal },
    risk: { size_mode: "equity_fraction", size_value: values.positionSize / 100, take_profit_pct: values.takeProfit / 100, stop_loss_pct: values.stopLoss / 100, max_positions: values.maximumPositions, leverage: values.leverage },
  };
}

function conditionBadgeClass(state: RuleConditionState) {
  if (state === "triggered") return "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
  if (state === "blocked") return "border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-200";
  if (state === "unavailable") return "border-muted-foreground/30 bg-muted text-muted-foreground";
  return "border-slate-500/30 bg-slate-500/10 text-slate-700 dark:text-slate-300";
}

function parseEvaluationRequest(input: string): EvaluateRuleStrategyRequest | null {
  if (!input.trim()) return null;
  try {
    const parsed: unknown = JSON.parse(input);
    if (typeof parsed !== "object" || parsed === null || !("candles" in parsed) || !("market" in parsed) || !Array.isArray(parsed.candles) || parsed.candles.length === 0 || typeof parsed.market !== "object" || parsed.market === null) return null;
    return parsed as EvaluateRuleStrategyRequest;
  } catch {
    return null;
  }
}


export default function Strategies() {
  const { t } = useTranslation();
  const [values, setValues] = useState<StrategyFormValues>(initialValues);
  const [strategyId, setStrategyId] = useState(() => localStorage.getItem("valuecell.rule-strategy-id") ?? "");
  const [name, setName] = useState(() => t("saas.operations.strategy.defaultName"));
  const [description, setDescription] = useState("");
  const [evaluationInput, setEvaluationInput] = useState("");
  const [evaluation, setEvaluation] = useState<RuleStrategyEvaluation | null>(null);
  const symbolsQuery = useGetCryptoSymbols();
  const strategyQuery = useRuleStrategy(strategyId);
  const createStrategy = useCreateRuleStrategy();
  const updateStrategy = useUpdateRuleStrategy(strategyId);
  const startStrategy = useStartRuleStrategy(strategyId);
  const stopStrategy = useStopRuleStrategy(strategyId);
  const evaluateStrategy = useEvaluateRuleStrategy(strategyId);

  useEffect(() => {
    if (!strategyQuery.data) return;
    setValues((current) => ({
      ...current,
      initialCapital: strategyQuery.data.config.initial_capital_quote,
      symbols: strategyQuery.data.config.symbols?.map((symbol) => symbol.replace("-", "/")) ?? initialValues.symbols,
      timeframe: strategyQuery.data.config.interval ?? initialValues.timeframe,
    }));
  }, [strategyQuery.data]);

  const errors = useMemo(() => {
    const next: Partial<Record<keyof StrategyFormValues, string>> = {};
    if (values.symbols.length === 0) next.symbols = t("saas.operations.strategy.errors.selectMarket");
    if (!Number.isFinite(values.fastMa) || values.fastMa < 2 || values.fastMa > 200) next.fastMa = t("saas.operations.strategy.errors.wholeNumber", { min: 2, max: 200 });
    if (!Number.isFinite(values.slowMa) || values.slowMa < 3 || values.slowMa > 400) next.slowMa = t("saas.operations.strategy.errors.wholeNumber", { min: 3, max: 400 });
    if (values.fastMa >= values.slowMa) next.slowMa = t("saas.operations.strategy.errors.slowAverage");
    if (values.rsiEnabled) {
      if (values.rsiPeriod < 2 || values.rsiPeriod > 100) next.rsiPeriod = t("saas.operations.strategy.errors.wholeNumber", { min: 2, max: 100 });
      if (values.rsiOversold < 1 || values.rsiOversold >= values.rsiOverbought) next.rsiOversold = t("saas.operations.strategy.errors.belowUpperThreshold");
      if (values.rsiOverbought > 99 || values.rsiOverbought <= values.rsiOversold) next.rsiOverbought = t("saas.operations.strategy.errors.aboveLowerThreshold");
    }
    if (values.bollingerEnabled) {
      if (values.bollingerPeriod < 2 || values.bollingerPeriod > 100) next.bollingerPeriod = t("saas.operations.strategy.errors.wholeNumber", { min: 2, max: 100 });
      if (values.bollingerDeviation < 0.5 || values.bollingerDeviation > 5) next.bollingerDeviation = t("saas.operations.strategy.errors.valueRange", { min: 0.5, max: 5 });
    }
    if (values.momentumEnabled) {
      if (values.macdFast < 2 || values.macdFast > 100) next.macdFast = t("saas.operations.strategy.errors.wholeNumber", { min: 2, max: 100 });
      if (values.macdSlow < 3 || values.macdSlow > 200 || values.macdSlow <= values.macdFast) next.macdSlow = t("saas.operations.strategy.errors.largerThanFastPeriod");
      if (values.macdSignal < 2 || values.macdSignal > 100) next.macdSignal = t("saas.operations.strategy.errors.wholeNumber", { min: 2, max: 100 });
    }
    if (values.positionSize <= 0 || values.positionSize > 25) next.positionSize = t("saas.operations.strategy.errors.percentRange", { min: 0.1, max: 25 });
    if (values.takeProfit <= 0 || values.takeProfit > 100) next.takeProfit = t("saas.operations.strategy.errors.percentRange", { min: 0.1, max: 100 });
    if (values.stopLoss <= 0 || values.stopLoss >= values.takeProfit) next.stopLoss = t("saas.operations.strategy.errors.positiveBelowTakeProfit");
    if (!Number.isInteger(values.maximumPositions) || values.maximumPositions < 1 || values.maximumPositions > 10) next.maximumPositions = t("saas.operations.strategy.errors.wholeNumber", { min: 1, max: 10 });
    if (!Number.isFinite(values.initialCapital) || values.initialCapital <= 0 || values.initialCapital > 100_000_000) next.initialCapital = "Enter a paper capital amount between 1 and 100,000,000 USDT.";
    if (values.leverage < 1 || values.leverage > 5) next.leverage = t("saas.operations.strategy.errors.leverageRange");
    return next;
  }, [t, values]);

  const update = <Key extends keyof StrategyFormValues>(key: Key, value: StrategyFormValues[Key]) => {
    setValues((current) => ({ ...current, [key]: value }));
  };

  const isValid = Object.keys(errors).length === 0;
  const activeFilters = [
    values.rsiEnabled ? `RSI ${values.rsiPeriod}` : null,
    values.bollingerEnabled ? `Bollinger ${values.bollingerPeriod}` : null,
    values.momentumEnabled ? `MACD ${values.macdFast}/${values.macdSlow}/${values.macdSignal}` : null,
  ].filter(Boolean);
  const isPending = createStrategy.isPending || updateStrategy.isPending || startStrategy.isPending || stopStrategy.isPending || evaluateStrategy.isPending;
  const storedStrategy = strategyQuery.data;
  const symbolOptions = symbolsQuery.data?.symbols.map((symbol) => symbol.replace("-", "/")) ?? [];
  const savePending = createStrategy.isPending || updateStrategy.isPending;
  const selectionLimitReached = values.symbols.length >= 10;

  const saveStrategy = async () => {
    if (!isValid) return;
    const request = {
      name: name.trim() || t("saas.operations.strategy.defaultName"),
      description: description.trim() || undefined,
      config: toRuleStrategyConfig(values),
    };
    try {
      const response = strategyId
        ? await updateStrategy.mutateAsync(request)
        : await createStrategy.mutateAsync({
            ...request,
            initial_capital_quote: values.initialCapital,
          });
      const saved = response.data;
      setStrategyId(saved.strategy_id);
      localStorage.setItem("valuecell.rule-strategy-id", saved.strategy_id);
      if (saved.strategy_id === strategyId) {
        await strategyQuery.refetch();
      }
      toast.success(t("saas.operations.strategy.toasts.saved"));
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : t("saas.operations.strategy.toasts.operationFailed"),
      );
    }
  };

  const runEvaluation = async () => {
    const request = parseEvaluationRequest(evaluationInput);
    if (!request) {
      toast.error(t("saas.operations.strategy.toasts.invalidEvaluation"));
      return;
    }
    try {
      const response = await evaluateStrategy.mutateAsync(request);
      setEvaluation(response.data);
      toast.success(t("saas.operations.strategy.toasts.evaluationComplete"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("saas.operations.strategy.toasts.operationFailed"));
    }
  };

  return (
    <div className="scroll-container flex size-full flex-col bg-background">
      <header className="border-b px-4 py-4 sm:px-6">
        <div className="mx-auto flex max-w-[1600px] flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-xl font-semibold">{t("saas.operations.strategy.title")}</h1>
              <Badge className="border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-300" variant="outline">
                <ShieldCheck /> {t("saas.operations.strategy.paperOnly")}
              </Badge>
            </div>
            <p className="mt-1 text-muted-foreground text-sm">{t("saas.operations.strategy.subtitle")}</p>
          </div>
          <Badge className="w-fit" variant="outline">
            <LockKeyhole /> {storedStrategy?.status === "running" ? t("saas.operations.strategy.status.active") : t("saas.operations.strategy.status.ready")}
          </Badge>
        </div>
      </header>

      <div className="mx-auto grid w-full max-w-[1600px] gap-4 p-4 sm:p-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <form className="grid min-w-0 gap-4" noValidate>
          <Alert className="border-sky-500/30 bg-sky-500/5">
            <AlertTriangle />
            <AlertTitle>{t("saas.operations.strategy.explicitInputs.title")}</AlertTitle>
            <AlertDescription>
              {t("saas.operations.strategy.explicitInputs.description")}
            </AlertDescription>
          </Alert>
          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <CardContent className="grid gap-4 px-4 py-4 sm:grid-cols-2 sm:px-5">
              <div className="grid gap-1.5"><Label htmlFor="strategy-name">{t("saas.operations.strategy.fields.name")}</Label><Input id="strategy-name" maxLength={200} onChange={(event) => setName(event.target.value)} value={name} /></div>
              <div className="grid gap-1.5"><Label htmlFor="strategy-description">{t("saas.operations.strategy.fields.description")} <span className="text-muted-foreground">{t("saas.operations.strategy.fields.optional")}</span></Label><Input id="strategy-description" maxLength={1000} onChange={(event) => setDescription(event.target.value)} value={description} /></div>
            </CardContent>
          </Card>

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <RuleHeading icon={CandlestickChart} title={t("saas.operations.strategy.sections.marketScope.title")} description={t("saas.operations.strategy.sections.marketScope.description")} />
            <CardContent className="grid gap-5 px-4 py-4 sm:px-5">
              <fieldset aria-describedby={errors.symbols ? "symbols-error" : "symbols-hint"}>
                <legend className="font-medium text-sm">{t("saas.operations.strategy.fields.markets")}</legend>
                <p className="mt-1 text-muted-foreground text-xs" id="symbols-hint">{t("saas.operations.strategy.fields.marketsHint", { max: 10 })}</p>
                {symbolsQuery.isLoading ? <p className="mt-2 text-muted-foreground text-xs" role="status">{t("saas.operations.strategy.fields.marketsLoading")}</p> : null}
                {symbolsQuery.isError ? <p className="mt-2 text-destructive text-xs" role="alert">{t("saas.operations.strategy.fields.marketsError")}</p> : null}
                {!symbolsQuery.isLoading && !symbolsQuery.isError ? (
                  <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label={t("saas.operations.strategy.fields.paperMarkets")}>
                    {symbolOptions.map((symbol) => {
                      const selected = values.symbols.includes(symbol);
                      return (
                        <Button
                          aria-pressed={selected}
                          disabled={!selected && selectionLimitReached}
                          key={symbol}
                          onClick={() => update("symbols", selected ? values.symbols.filter((item) => item !== symbol) : [...values.symbols, symbol])}
                          type="button"
                          variant={selected ? "default" : "outline"}
                        >
                          {symbol}
                        </Button>
                      );
                    })}
                  </div>
                ) : null}
                {errors.symbols ? <p className="mt-2 text-destructive text-xs" id="symbols-error" role="alert">{errors.symbols}</p> : null}
              </fieldset>
              <div className="grid gap-1.5 sm:max-w-56">
                <Label htmlFor="timeframe">{t("saas.operations.strategy.fields.candleTimeframe")}</Label>
                <Select value={values.timeframe} onValueChange={(value) => update("timeframe", value as RuleStrategyInterval)}>
                  <SelectTrigger id="timeframe"><SelectValue /></SelectTrigger>
                  <SelectContent>{TIMEFRAME_OPTIONS.map((timeframe) => <SelectItem key={timeframe} value={timeframe}>{timeframe}</SelectItem>)}</SelectContent>
                </Select>
                <p className="text-muted-foreground text-xs">{t("saas.operations.strategy.fields.candleTimeframeHint")}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <RuleHeading icon={TrendingUp} title={t("saas.operations.strategy.sections.trendEntry.title")} description={t("saas.operations.strategy.sections.trendEntry.description")} />
            <CardContent className="grid gap-4 px-4 py-4 sm:grid-cols-2 sm:px-5">
              <NumericField error={errors.fastMa} id="fast-ma" label={t("saas.operations.strategy.fields.fastMovingAverage")} max={200} min={2} onChange={(value) => update("fastMa", value)} value={values.fastMa} />
              <NumericField error={errors.slowMa} id="slow-ma" label={t("saas.operations.strategy.fields.slowMovingAverage")} max={400} min={3} onChange={(value) => update("slowMa", value)} value={values.slowMa} />
            </CardContent>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="gap-0 rounded-lg py-0 shadow-none">
              <RuleHeading enabled={values.rsiEnabled} icon={Gauge} onEnabledChange={(enabled) => update("rsiEnabled", enabled)} title={t("saas.operations.strategy.sections.rsi.title")} description={t("saas.operations.strategy.sections.rsi.description")} />
              <CardContent className="grid gap-4 px-4 py-4 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3">
                <NumericField disabled={!values.rsiEnabled} error={errors.rsiPeriod} id="rsi-period" label={t("saas.operations.strategy.fields.period")} max={100} min={2} onChange={(value) => update("rsiPeriod", value)} value={values.rsiPeriod} />
                <NumericField disabled={!values.rsiEnabled} error={errors.rsiOversold} id="rsi-oversold" label={t("saas.operations.strategy.fields.lowerThreshold")} max={98} min={1} onChange={(value) => update("rsiOversold", value)} value={values.rsiOversold} />
                <NumericField disabled={!values.rsiEnabled} error={errors.rsiOverbought} id="rsi-overbought" label={t("saas.operations.strategy.fields.upperThreshold")} max={99} min={2} onChange={(value) => update("rsiOverbought", value)} value={values.rsiOverbought} />
              </CardContent>
            </Card>

            <Card className="gap-0 rounded-lg py-0 shadow-none">
              <RuleHeading enabled={values.bollingerEnabled} icon={BarChart3} onEnabledChange={(enabled) => update("bollingerEnabled", enabled)} title={t("saas.operations.strategy.sections.bollinger.title")} description={t("saas.operations.strategy.sections.bollinger.description")} />
              <CardContent className="grid gap-4 px-4 py-4 sm:grid-cols-2">
                <NumericField disabled={!values.bollingerEnabled} error={errors.bollingerPeriod} id="bollinger-period" label={t("saas.operations.strategy.fields.period")} max={100} min={2} onChange={(value) => update("bollingerPeriod", value)} value={values.bollingerPeriod} />
                <NumericField disabled={!values.bollingerEnabled} error={errors.bollingerDeviation} id="bollinger-deviation" label={t("saas.operations.strategy.fields.standardDeviations")} max={5} min={0.5} onChange={(value) => update("bollingerDeviation", value)} step={0.1} value={values.bollingerDeviation} />
              </CardContent>
            </Card>
          </div>

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <RuleHeading enabled={values.momentumEnabled} icon={Activity} onEnabledChange={(enabled) => update("momentumEnabled", enabled)} title={t("saas.operations.strategy.sections.momentum.title")} description={t("saas.operations.strategy.sections.momentum.description")} />
            <CardContent className="grid gap-4 px-4 py-4 sm:grid-cols-3 sm:px-5">
              <NumericField disabled={!values.momentumEnabled} error={errors.macdFast} id="macd-fast" label={t("saas.operations.strategy.fields.macdFastPeriod")} max={100} min={2} onChange={(value) => update("macdFast", value)} value={values.macdFast} />
              <NumericField disabled={!values.momentumEnabled} error={errors.macdSlow} id="macd-slow" label={t("saas.operations.strategy.fields.macdSlowPeriod")} max={200} min={3} onChange={(value) => update("macdSlow", value)} value={values.macdSlow} />
              <NumericField disabled={!values.momentumEnabled} error={errors.macdSignal} id="macd-signal" label={t("saas.operations.strategy.fields.macdSignalPeriod")} max={100} min={2} onChange={(value) => update("macdSignal", value)} value={values.macdSignal} />
            </CardContent>
          </Card>

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <RuleHeading icon={SlidersHorizontal} title={t("saas.operations.strategy.sections.riskLimits.title")} description={t("saas.operations.strategy.sections.riskLimits.description")} />
            <CardContent className="grid gap-4 px-4 py-4 sm:grid-cols-2 lg:grid-cols-3 sm:px-5">
              <NumericField error={errors.initialCapital} id="initial-capital" label="Initial paper capital" min={1} max={100_000_000} onChange={(value) => update("initialCapital", value)} step={100} unit="USDT" value={values.initialCapital} />
              <NumericField error={errors.positionSize} id="position-size" label={t("saas.operations.strategy.fields.positionSize")} max={25} min={0.1} onChange={(value) => update("positionSize", value)} step={0.1} unit="%" value={values.positionSize} />
              <NumericField error={errors.takeProfit} id="take-profit" label={t("saas.operations.strategy.fields.takeProfit")} max={100} min={0.1} onChange={(value) => update("takeProfit", value)} step={0.1} unit="%" value={values.takeProfit} />
              <NumericField error={errors.stopLoss} id="stop-loss" label={t("saas.operations.strategy.fields.stopLoss")} max={99.9} min={0.1} onChange={(value) => update("stopLoss", value)} step={0.1} unit="%" value={values.stopLoss} />
              <NumericField error={errors.maximumPositions} id="maximum-positions" label={t("saas.operations.strategy.fields.maximumOpenPositions")} max={10} min={1} onChange={(value) => update("maximumPositions", value)} value={values.maximumPositions} />
              <NumericField error={errors.leverage} hint={t("saas.operations.strategy.fields.paperEvaluationOnly")} id="leverage" label={t("saas.operations.strategy.fields.maximumLeverage")} max={5} min={1} onChange={(value) => update("leverage", value)} step={0.5} unit="x" value={values.leverage} />
            </CardContent>
          </Card>
        </form>

        <aside className="flex min-w-0 flex-col gap-4 xl:sticky xl:top-0 xl:self-start">
          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <CardHeader className="border-b px-4 py-4">
              <CardTitle className="flex items-center gap-2 text-base"><Layers3 /> {t("saas.operations.strategy.summary.title")}</CardTitle>
              <CardDescription>{t("saas.operations.strategy.summary.description")}</CardDescription>
            </CardHeader>
            <CardContent className="divide-y px-4 py-2" aria-live="polite">
              <SummaryRow label={t("saas.operations.strategy.summary.paperMarkets")} value={values.symbols.join(", ") || t("saas.operations.strategy.summary.noneSelected")} />
              <SummaryRow label={t("saas.operations.strategy.summary.timeframe")} value={values.timeframe} />
              <SummaryRow label={t("saas.operations.strategy.summary.entry")} value={t("saas.operations.strategy.summary.entryValue", { fast: values.fastMa, slow: values.slowMa })} />
              <SummaryRow label={t("saas.operations.strategy.summary.confirmations")} value={activeFilters.join(" | ") || t("saas.operations.strategy.summary.none")} />
              <SummaryRow label="Initial paper capital" value={`${values.initialCapital.toLocaleString()} USDT`} />
              <SummaryRow label={t("saas.operations.strategy.summary.positionSize")} value={t("saas.operations.strategy.summary.positionSizeValue", { value: values.positionSize })} />
              <SummaryRow label={t("saas.operations.strategy.summary.exitLimits")} value={t("saas.operations.strategy.summary.exitLimitsValue", { takeProfit: values.takeProfit, stopLoss: values.stopLoss })} />
              <SummaryRow label={t("saas.operations.strategy.summary.exposureCap")} value={t("saas.operations.strategy.summary.exposureCapValue", { positions: values.maximumPositions, leverage: values.leverage })} />
            </CardContent>
          </Card>

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <CardHeader className="border-b px-4 py-4">
              <CardTitle className="flex items-center gap-2 text-base"><CircleDollarSign /> {t("saas.operations.strategy.paperStrategy.title")}</CardTitle>
              <CardDescription>{storedStrategy ? t("saas.operations.strategy.paperStrategy.status", { name: storedStrategy.name, status: storedStrategy.status }) : isValid ? t("saas.operations.strategy.paperStrategy.valid") : t("saas.operations.strategy.paperStrategy.resolve")}</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 px-4 py-4">
              <Button disabled={!isValid || isPending} onClick={saveStrategy} type="button"><Target /> <span aria-live="polite">{savePending ? t("saas.operations.strategy.actions.saving") : t("saas.operations.strategy.actions.save")}</span></Button>
              {storedStrategy?.status === "running" ? <Button disabled={isPending} onClick={async () => { try { await stopStrategy.mutateAsync(); toast.success(t("saas.operations.strategy.toasts.stopped")); } catch (err) { toast.error(err instanceof Error ? err.message : t("saas.operations.strategy.toasts.operationFailed")); } }} type="button" variant="outline">{t("saas.operations.strategy.actions.stop")}</Button> : <Button disabled={!strategyId || isPending} onClick={async () => { try { await startStrategy.mutateAsync(); toast.success(t("saas.operations.strategy.toasts.started")); } catch (err) { toast.error(err instanceof Error ? err.message : t("saas.operations.strategy.toasts.operationFailed")); } }} type="button" variant="outline">{t("saas.operations.strategy.actions.start")}</Button>}
              <p className="text-muted-foreground text-xs leading-relaxed">{t("saas.operations.strategy.paperStrategy.help")}</p>
              <Alert className="border-amber-500/30 bg-amber-500/5">
                <AlertTriangle />
                <AlertTitle>{t("saas.operations.strategy.executionMode.title")}</AlertTitle>
                <AlertDescription>
                  {t("saas.operations.strategy.executionMode.description")}
                  <Link
                    className="ml-1 underline underline-offset-2"
                    to="/settings/sandbox-exchanges"
                  >
                    {t("saas.operations.strategy.executionMode.link")}
                  </Link>
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
          {storedStrategy ? (
            <Card className="gap-0 rounded-lg py-0 shadow-none">
              <CardHeader className="border-b px-4 py-4">
                <CardTitle className="text-base">Paper account</CardTitle>
                <CardDescription>Server-recorded paper balance and marked equity.</CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-x-4 gap-y-3 px-4 py-4 text-sm">
                <SummaryRow label="Cash" value={`${storedStrategy.account.quote_balance.toFixed(2)} USDT`} />
                <SummaryRow label="Equity" value={`${storedStrategy.account.equity_quote.toFixed(2)} USDT`} />
                <SummaryRow label="Realized PnL" value={`${storedStrategy.account.realized_pnl_quote.toFixed(2)} USDT`} />
                <SummaryRow label="Unrealized PnL" value={`${storedStrategy.account.unrealized_pnl_quote.toFixed(2)} USDT`} />
              </CardContent>
            </Card>
          ) : null}

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <CardHeader className="border-b px-4 py-4"><CardTitle className="text-base">{t("saas.operations.strategy.evaluate.title")}</CardTitle><CardDescription>{t("saas.operations.strategy.evaluate.description")}</CardDescription></CardHeader>
            <CardContent className="grid gap-3 px-4 py-4"><Textarea className="min-h-40 font-mono text-xs" onChange={(event) => setEvaluationInput(event.target.value)} placeholder={'{"candles":[{"timestamp_ms":...,"open":...,"high":...,"low":...,"close":...,"volume":...}],"market":{"symbol":"BTC-USDT","price":...}}'} value={evaluationInput} /><Button disabled={!strategyId || storedStrategy?.status !== "running" || isPending} onClick={runEvaluation} type="button" variant="outline"><Percent /> {t("saas.operations.strategy.actions.runEvaluation")}</Button></CardContent>
          </Card>

          {evaluation ? <Card className="gap-0 rounded-lg py-0 shadow-none"><CardHeader className="border-b px-4 py-4"><CardTitle className="flex items-center gap-2 text-base">{t("saas.operations.strategy.evaluation.title")} <Badge className="capitalize" variant="outline">{evaluation.action.replace("_", " ")}</Badge></CardTitle><CardDescription>{evaluation.reason}</CardDescription></CardHeader><CardContent className="grid gap-4 px-4 py-4"><div className="grid gap-2">{evaluation.conditions.map((condition) => <div className="rounded-md border p-3" key={condition.code}><div className="flex flex-wrap items-center justify-between gap-2"><span className="font-medium text-sm">{condition.code}</span><Badge className={conditionBadgeClass(condition.state)} variant="outline">{condition.state.replace("_", " ")}</Badge></div><p className="mt-1 text-muted-foreground text-xs">{condition.detail}</p></div>)}</div><div className="rounded-md bg-muted/50 p-3 text-sm"><p className="font-medium">{t("saas.operations.strategy.evaluation.fundingImpact")}</p><p className="mt-1 text-muted-foreground">{t("saas.operations.strategy.evaluation.fundingSummary", { direction: evaluation.funding.direction, payment: evaluation.funding.estimated_payment_quote.toFixed(4), rate: (evaluation.funding.funding_rate * 100).toFixed(4) })}</p></div></CardContent></Card> : null}
        </aside>
      </div>
    </div>
  );
}
