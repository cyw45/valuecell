import {
  Activity,
  AlertTriangle,
  BarChart3,
  CandlestickChart,
  BrainCircuit,
  CircleDollarSign,
  FileText,
  Gauge,
  Layers3,
  LockKeyhole,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  TrendingUp,
  WandSparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { useGetCryptoSymbols } from "@/api/crypto-market";
import {
  useCreateRuleStrategy,
  useParseRuleStrategyText,
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
  AdvancedRuleSetConfig,
  RuleStrategyConfig,
  RuleStrategyInterval,
} from "@/types/rule-strategy";
import { cn } from "@/lib/utils";

const TIMEFRAME_OPTIONS: RuleStrategyInterval[] = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"];
const ADVANCED_TIMEFRAME_OPTIONS = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"] as const;
const DEFAULT_STRATEGY_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"];

const PAPER_DEMO_ADVANCED_RULES: AdvancedRuleSetConfig = {
  enabled: true,
  entry_confirmation_mode: "any",
  exit_confirmation_mode: "any",
  moving_average: { enabled: false, interval: "1d", period: 20, entry_comparator: "above" },
  macd: { enabled: false, interval: "5m", fast_window: 12, slow_window: 26, signal_window: 9, entry_cross: "golden" },
  bollinger: { enabled: false, interval: "15m", period: 20, standard_deviations: 2, entry_reference: "middle", entry_comparator: "above" },
  rsi: { enabled: true, interval: "1m", period: 1, entry_comparator: "above", entry_threshold: -1, exit_enabled: true, exit_comparator: "above", exit_threshold: -1 },
  momentum: { enabled: false, interval: "1m", period: 14, entry_comparator: "below", entry_threshold: 20, exit_enabled: false, exit_comparator: "above", exit_threshold: 85 },
  brar: { enabled: false, interval: "1m", period: 26, component: "br", entry_comparator: "below", entry_threshold: 30, exit_enabled: false, exit_comparator: "above", exit_threshold: 85 },
};

function createPaperDemoAdvancedRules(): AdvancedRuleSetConfig {
  return {
    ...PAPER_DEMO_ADVANCED_RULES,
    moving_average: { ...PAPER_DEMO_ADVANCED_RULES.moving_average },
    macd: { ...PAPER_DEMO_ADVANCED_RULES.macd },
    bollinger: { ...PAPER_DEMO_ADVANCED_RULES.bollinger },
    rsi: { ...PAPER_DEMO_ADVANCED_RULES.rsi },
    momentum: { ...PAPER_DEMO_ADVANCED_RULES.momentum },
    brar: { ...PAPER_DEMO_ADVANCED_RULES.brar },
  };
}


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
  advancedRules: AdvancedRuleSetConfig;
  takeProfitEnabled: boolean;
  takeProfit: number;
  stopLossEnabled: boolean;
  stopLoss: number;
  maximumPositions: number;
  leverage: number;
};

type AdvancedIndicatorKey = Exclude<
  keyof AdvancedRuleSetConfig,
  "enabled" | "entry_confirmation_mode" | "exit_confirmation_mode"
>;

const initialValues: StrategyFormValues = {
  symbols: [],
  timeframe: "15m",
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
  advancedRules: {
    enabled: true,
    entry_confirmation_mode: "all",
    exit_confirmation_mode: "any",
    moving_average: { enabled: true, interval: "1d", period: 20, entry_comparator: "above" },
    macd: { enabled: true, interval: "5m", fast_window: 12, slow_window: 26, signal_window: 9, entry_cross: "golden" },
    bollinger: { enabled: true, interval: "15m", period: 20, standard_deviations: 2, entry_reference: "middle", entry_comparator: "above" },
    rsi: { enabled: true, interval: "15m", period: 14, entry_comparator: "below", entry_threshold: 20, exit_enabled: true, exit_comparator: "above", exit_threshold: 85 },
    momentum: { enabled: true, interval: "15m", period: 14, entry_comparator: "below", entry_threshold: 20, exit_enabled: true, exit_comparator: "above", exit_threshold: 85 },
    brar: { enabled: true, interval: "15m", period: 26, component: "br", entry_comparator: "below", entry_threshold: 30, exit_enabled: false, exit_comparator: "above", exit_threshold: 85 },
  },
  takeProfitEnabled: false,
  takeProfit: 4,
  stopLossEnabled: false,
  stopLoss: 2,
  maximumPositions: 100,
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

function IntervalSelect({
  id,
  value,
  onChange,
  disabled = false,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <Select disabled={disabled} value={value} onValueChange={onChange}>
      <SelectTrigger id={id}><SelectValue /></SelectTrigger>
      <SelectContent>
        {ADVANCED_TIMEFRAME_OPTIONS.map((interval) => <SelectItem key={interval} value={interval}>{interval}</SelectItem>)}
      </SelectContent>
    </Select>
  );
}

function AdvancedThresholdEditor({
  label,
  description,
  rule,
  onChange,
  brar = false,
}: {
  label: string;
  description: string;
  rule: AdvancedRuleSetConfig["rsi"] & { component?: "ar" | "br" };
  onChange: (changes: Partial<AdvancedRuleSetConfig["rsi"]> & { component?: "ar" | "br" }) => void;
  brar?: boolean;
}) {
  return (
    <div className="grid gap-3 rounded-md border border-border/70 p-3">
      <div className="flex items-center justify-between gap-3"><div><p className="font-medium text-sm">{label}</p><p className="text-muted-foreground text-xs">{description}</p></div><Switch checked={rule.enabled} onCheckedChange={(enabled) => onChange({ enabled })} /></div>
      <div className={cn("grid gap-3 sm:grid-cols-2", brar ? "lg:grid-cols-5" : "lg:grid-cols-4")}>
        <div className="grid gap-1.5"><Label>周期</Label><IntervalSelect disabled={!rule.enabled} id={`${label}-interval`} onChange={(interval) => onChange({ interval: interval as "15m" })} value={rule.interval} /></div>
        <NumericField disabled={!rule.enabled} id={`${label}-period`} label="计算周期" max={500} min={brar ? 2 : 1} onChange={(period) => onChange({ period })} value={rule.period} />
        {brar ? <div className="grid gap-1.5"><Label>BRAR 分量</Label><Select disabled={!rule.enabled} value={rule.component ?? "br"} onValueChange={(component) => onChange({ component: component as "ar" | "br" })}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="ar">AR</SelectItem><SelectItem value="br">BR</SelectItem></SelectContent></Select></div> : null}
        <div className="grid gap-1.5"><Label>买入比较</Label><Select disabled={!rule.enabled} value={rule.entry_comparator} onValueChange={(entry_comparator) => onChange({ entry_comparator: entry_comparator as "above" | "below" })}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="below">低于阈值</SelectItem><SelectItem value="above">高于阈值</SelectItem></SelectContent></Select></div>
        <NumericField disabled={!rule.enabled} id={`${label}-entry`} label="买入阈值" max={100_000_000} min={-100_000_000} onChange={(entry_threshold) => onChange({ entry_threshold })} step={0.1} value={rule.entry_threshold} />
      </div>
      <div className="grid gap-3 border-border/60 border-t pt-3 sm:grid-cols-4">
        <div className="flex items-center justify-between gap-3"><Label htmlFor={`${label}-exit`}>启用卖出阈值</Label><Switch checked={rule.exit_enabled} disabled={!rule.enabled} id={`${label}-exit`} onCheckedChange={(exit_enabled) => onChange({ exit_enabled })} /></div>
        <div className="grid gap-1.5"><Label>卖出比较</Label><Select disabled={!rule.enabled || !rule.exit_enabled} value={rule.exit_comparator} onValueChange={(exit_comparator) => onChange({ exit_comparator: exit_comparator as "above" | "below" })}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="above">高于阈值</SelectItem><SelectItem value="below">低于阈值</SelectItem></SelectContent></Select></div>
        <NumericField disabled={!rule.enabled || !rule.exit_enabled} id={`${label}-exit-value`} label="卖出阈值" max={100_000_000} min={-100_000_000} onChange={(exit_threshold) => onChange({ exit_threshold })} step={0.1} value={rule.exit_threshold} />
      </div>
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
    advanced_rules: values.advancedRules,
    risk: { size_mode: "equal_split", size_value: 1, take_profit_pct: values.takeProfitEnabled ? values.takeProfit / 100 : undefined, stop_loss_pct: values.stopLossEnabled ? values.stopLoss / 100 : undefined, max_positions: values.maximumPositions, leverage: values.leverage },
  };
}

export function RuleStrategyConfiguration({ embedded = false }: { embedded?: boolean }) {
  const { t } = useTranslation();
  const [values, setValues] = useState<StrategyFormValues>(initialValues);
  const [strategyId, setStrategyId] = useState(() => localStorage.getItem("valuecell.rule-strategy-id") ?? "");
  const [name, setName] = useState(() => t("saas.operations.strategy.defaultName"));
  const [description, setDescription] = useState("");
  const [strategyText, setStrategyText] = useState("");
  const [textImportSummary, setTextImportSummary] = useState("");
  const [unresolvedItems, setUnresolvedItems] = useState<string[]>([]);
  const symbolsQuery = useGetCryptoSymbols();
  const strategyQuery = useRuleStrategy(strategyId);
  const createStrategy = useCreateRuleStrategy();
  const updateStrategy = useUpdateRuleStrategy(strategyId);
  const startStrategy = useStartRuleStrategy(strategyId);
  const stopStrategy = useStopRuleStrategy(strategyId);
  const parseStrategyText = useParseRuleStrategyText();
  const symbolOptions = symbolsQuery.data?.symbols.map((symbol) => symbol.replace("-", "/")) ?? [];

  useEffect(() => {
    if (!strategyQuery.data) return;
    setValues((current) => ({
      ...current,
      initialCapital: strategyQuery.data.config.initial_capital_quote,
      symbols: strategyQuery.data.config.symbols?.map((symbol) => symbol.replace("-", "/")) ?? initialValues.symbols,
      timeframe: strategyQuery.data.config.interval ?? initialValues.timeframe,
      advancedRules: strategyQuery.data.config.advanced_rules ?? initialValues.advancedRules,
      takeProfitEnabled: strategyQuery.data.config.risk.take_profit_pct !== undefined,
      takeProfit: (strategyQuery.data.config.risk.take_profit_pct ?? current.takeProfit / 100) * 100,
      stopLossEnabled: strategyQuery.data.config.risk.stop_loss_pct !== undefined,
      stopLoss: (strategyQuery.data.config.risk.stop_loss_pct ?? current.stopLoss / 100) * 100,
    }));
  }, [strategyQuery.data]);

  useEffect(() => {
    if (strategyId || symbolOptions.length === 0) return;
    setValues((current) => {
      if (current.symbols.length > 0) return current;
      const symbols = DEFAULT_STRATEGY_SYMBOLS.filter((symbol) =>
        symbolOptions.includes(symbol),
      );
      return {
        ...current,
        symbols: symbols.length > 0 ? symbols : [symbolOptions[0]],
        maximumPositions: 1,
      };
    });
  }, [strategyId, symbolOptions]);

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
    if (values.takeProfitEnabled && (values.takeProfit <= 0 || values.takeProfit > 100)) next.takeProfit = t("saas.operations.strategy.errors.percentRange", { min: 0.1, max: 100 });
    if (values.stopLossEnabled && (values.stopLoss <= 0 || values.stopLoss > 100)) next.stopLoss = t("saas.operations.strategy.errors.percentRange", { min: 0.1, max: 100 });
    if (!Number.isInteger(values.maximumPositions) || values.maximumPositions < 1 || values.maximumPositions > 100) next.maximumPositions = t("saas.operations.strategy.errors.wholeNumber", { min: 1, max: 100 });
    if (!Number.isFinite(values.initialCapital) || values.initialCapital <= 0 || values.initialCapital > 100_000_000) next.initialCapital = "请输入 1 至 100,000,000 USDT 之间的初始模拟资金。";
    if (values.leverage < 1 || values.leverage > 5) next.leverage = t("saas.operations.strategy.errors.leverageRange");
    return next;
  }, [t, values]);

  const update = <Key extends keyof StrategyFormValues>(key: Key, value: StrategyFormValues[Key]) => {
    setValues((current) => ({ ...current, [key]: value }));
  };
  const updateAdvancedRule = <
    Section extends AdvancedIndicatorKey,
    Key extends keyof AdvancedRuleSetConfig[Section],
  >(section: Section, key: Key, value: AdvancedRuleSetConfig[Section][Key]) => {
    setValues((current) => ({
      ...current,
      advancedRules: {
        ...current.advancedRules,
        [section]: {
          ...current.advancedRules[section],
          [key]: value,
        },
      },
    }));
  };

  const applyPaperDemoPreset = () => {
    setValues((current) => ({
      ...current,
      timeframe: "1m",
      rsiEnabled: false,
      bollingerEnabled: false,
      momentumEnabled: false,
      advancedRules: createPaperDemoAdvancedRules(),
      takeProfitEnabled: false,
      stopLossEnabled: false,
      maximumPositions: 100,
      leverage: 1,
    }));
    setName("纸面交易演示策略");
    toast.success("极限演示预设已回填。请保存策略后重新启动；行情数据就绪后会买入，下一轮扫描会模拟卖出。");
  };

  const isValid = Object.keys(errors).length === 0;
  const activeFilters = [
    values.rsiEnabled ? `RSI ${values.rsiPeriod}` : null,
    values.bollingerEnabled ? `Bollinger ${values.bollingerPeriod}` : null,
    values.momentumEnabled ? `MACD ${values.macdFast}/${values.macdSlow}/${values.macdSignal}` : null,
  ].filter(Boolean);
  const isPending = createStrategy.isPending || updateStrategy.isPending || startStrategy.isPending || stopStrategy.isPending || parseStrategyText.isPending;
  const storedStrategy = strategyQuery.data;
  const savePending = createStrategy.isPending || updateStrategy.isPending;
  const selectionLimitReached = values.symbols.length >= 100;
  const ConfigurationHeading = embedded ? "h2" : "h1";

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

  const importStrategyText = async () => {
    if (strategyText.trim().length < 10) {
      toast.error("请至少输入一条完整的策略规则。");
      return;
    }
    try {
      const response = await parseStrategyText.mutateAsync(strategyText);
      const proposal = response.data;
      setValues((current) => ({
        ...current,
        timeframe: proposal.config.interval,
        advancedRules: proposal.config.advanced_rules,
        takeProfitEnabled: proposal.config.risk.take_profit_pct !== null && proposal.config.risk.take_profit_pct !== undefined,
        takeProfit: (proposal.config.risk.take_profit_pct ?? current.takeProfit / 100) * 100,
        stopLossEnabled: proposal.config.risk.stop_loss_pct !== null && proposal.config.risk.stop_loss_pct !== undefined,
        stopLoss: (proposal.config.risk.stop_loss_pct ?? current.stopLoss / 100) * 100,
        maximumPositions: proposal.config.risk.max_positions,
        leverage: proposal.config.risk.leverage,
      }));
      if (proposal.strategy_name) setName(proposal.strategy_name);
      setTextImportSummary(proposal.summary);
      setUnresolvedItems(proposal.unresolved_items);
      toast.success("策略参数已回填，请审核后保存。");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "策略文本解析失败。");
    }
  };

  return (
    <div className={cn("flex flex-col", embedded ? "gap-4" : "scroll-container size-full bg-background")}>
      <header className={cn("border-b px-4 py-4 sm:px-6", embedded && "rounded-lg border border-sky-400/15 bg-card/90")}>
        <div className={cn("mx-auto flex max-w-[1600px] flex-col gap-4 lg:flex-row lg:items-center lg:justify-between", embedded && "max-w-none")}>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <ConfigurationHeading className="font-semibold text-xl">{t("saas.operations.strategy.title")}</ConfigurationHeading>
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

      <div className={cn("mx-auto grid w-full max-w-[1600px] gap-4 p-4 sm:p-6 xl:grid-cols-[minmax(0,1fr)_360px]", embedded && "max-w-none px-0 pb-0 pt-4 sm:px-0 sm:pb-0 sm:pt-4")}>
        <form className="grid min-w-0 gap-4" noValidate>
          <Alert className="border-sky-500/30 bg-sky-500/5">
            <AlertTriangle />
            <AlertTitle>{t("saas.operations.strategy.explicitInputs.title")}</AlertTitle>
            <AlertDescription>
              {t("saas.operations.strategy.explicitInputs.description")}
            </AlertDescription>
          </Alert>
          <Alert className="border-amber-500/35 bg-amber-500/5">
            <Activity className="text-amber-500" />
            <AlertTitle>纸面交易演示预设</AlertTitle>
            <AlertDescription className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <span>使用 1 分钟 RSI 高于 -1 的必过规则：行情数据就绪即模拟买入，下一轮扫描模拟卖出。仅用于验证买卖、持仓和盈亏链路，不适用于真实策略。</span>
              <Button className="w-fit shrink-0" onClick={applyPaperDemoPreset} type="button" variant="secondary">
                应用演示预设
              </Button>
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
                <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                  <span className="text-muted-foreground text-xs">已选择 {values.symbols.length} / {symbolOptions.length}</span>
                  <div className="flex items-center gap-2">
                    <Button
                      disabled={symbolOptions.length === 0 || values.symbols.length === symbolOptions.length}
                      onClick={() => update("symbols", symbolOptions)}
                      size="sm"
                      type="button"
                      variant="secondary"
                    >
                      全选
                    </Button>
                    <Button
                      disabled={values.symbols.length === 0}
                      onClick={() => update("symbols", [])}
                      size="sm"
                      type="button"
                      variant="ghost"
                    >
                      清空
                    </Button>
                  </div>
                </div>
                <p className="mt-1 text-muted-foreground text-xs" id="symbols-hint">默认选择所有支持的 USDT 币种。扫描器会逐一计算技术指标，并将可用资金按符合条件的币种数量均分。</p>
                {symbolsQuery.isLoading ? <output className="mt-2 block text-muted-foreground text-xs">{t("saas.operations.strategy.fields.marketsLoading")}</output> : null}
                {symbolsQuery.isError ? <p className="mt-2 text-destructive text-xs" role="alert">{t("saas.operations.strategy.fields.marketsError")}</p> : null}
                {!symbolsQuery.isLoading && !symbolsQuery.isError ? (
                  <div className="mt-3 flex flex-wrap gap-2">
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

          <Card className="gap-0 rounded-lg border-sky-500/30 py-0 shadow-none">
            <RuleHeading
              enabled={values.advancedRules.enabled}
              icon={Layers3}
              onEnabledChange={(enabled) => update("advancedRules", { ...values.advancedRules, enabled })}
              title="多周期高级规则"
              description="每项指标独立设置周期、参数、比较方向和进出场阈值。高级模式开启后，以下规则替代基础指标区。"
            />
            <CardContent className="grid gap-5 px-4 py-4 sm:px-5">
              <div className="grid gap-4 rounded-md border border-border/70 p-3 sm:grid-cols-2">
                <div className="grid gap-1.5"><Label htmlFor="advanced-entry-mode">买入确认方式</Label><Select value={values.advancedRules.entry_confirmation_mode} onValueChange={(value) => update("advancedRules", { ...values.advancedRules, entry_confirmation_mode: value as "all" | "any" })}><SelectTrigger id="advanced-entry-mode"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">全部条件满足才买入</SelectItem><SelectItem value="any">任一条件满足即可买入</SelectItem></SelectContent></Select></div>
                <div className="grid gap-1.5"><Label htmlFor="advanced-exit-mode">卖出确认方式</Label><Select value={values.advancedRules.exit_confirmation_mode} onValueChange={(value) => update("advancedRules", { ...values.advancedRules, exit_confirmation_mode: value as "all" | "any" })}><SelectTrigger id="advanced-exit-mode"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="any">任一卖出条件满足即卖出</SelectItem><SelectItem value="all">全部卖出条件满足才卖出</SelectItem></SelectContent></Select></div>
              </div>

              <div className="grid gap-3 rounded-md border border-border/70 p-3">
                <div className="flex items-center justify-between gap-3"><div><p className="font-medium text-sm">日线价格与均线</p><p className="text-muted-foreground text-xs">可判断价格高于或低于任意周期均线。</p></div><Switch checked={values.advancedRules.moving_average.enabled} onCheckedChange={(enabled) => updateAdvancedRule("moving_average", "enabled", enabled)} /></div>
                <div className="grid gap-3 sm:grid-cols-3"><div className="grid gap-1.5"><Label>周期</Label><IntervalSelect disabled={!values.advancedRules.moving_average.enabled} id="advanced-ma-interval" onChange={(interval) => updateAdvancedRule("moving_average", "interval", interval as "1d")} value={values.advancedRules.moving_average.interval} /></div><NumericField disabled={!values.advancedRules.moving_average.enabled} id="advanced-ma-period" label="均线周期" max={500} min={2} onChange={(period) => updateAdvancedRule("moving_average", "period", period)} value={values.advancedRules.moving_average.period} /><div className="grid gap-1.5"><Label>价格关系</Label><Select disabled={!values.advancedRules.moving_average.enabled} value={values.advancedRules.moving_average.entry_comparator} onValueChange={(value) => updateAdvancedRule("moving_average", "entry_comparator", value as "above" | "below")}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="above">价格高于均线</SelectItem><SelectItem value="below">价格低于均线</SelectItem></SelectContent></Select></div></div>
              </div>

              <div className="grid gap-3 rounded-md border border-border/70 p-3">
                <div className="flex items-center justify-between gap-3"><div><p className="font-medium text-sm">MACD 金叉 / 死叉</p><p className="text-muted-foreground text-xs">可单独使用 5 分钟等周期，不受主图周期限制。</p></div><Switch checked={values.advancedRules.macd.enabled} onCheckedChange={(enabled) => updateAdvancedRule("macd", "enabled", enabled)} /></div>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5"><div className="grid gap-1.5"><Label>周期</Label><IntervalSelect disabled={!values.advancedRules.macd.enabled} id="advanced-macd-interval" onChange={(interval) => updateAdvancedRule("macd", "interval", interval as "5m")} value={values.advancedRules.macd.interval} /></div><NumericField disabled={!values.advancedRules.macd.enabled} id="advanced-macd-fast" label="快线" max={500} min={1} onChange={(value) => updateAdvancedRule("macd", "fast_window", value)} value={values.advancedRules.macd.fast_window} /><NumericField disabled={!values.advancedRules.macd.enabled} id="advanced-macd-slow" label="慢线" max={500} min={2} onChange={(value) => updateAdvancedRule("macd", "slow_window", value)} value={values.advancedRules.macd.slow_window} /><NumericField disabled={!values.advancedRules.macd.enabled} id="advanced-macd-signal" label="信号线" max={500} min={1} onChange={(value) => updateAdvancedRule("macd", "signal_window", value)} value={values.advancedRules.macd.signal_window} /><div className="grid gap-1.5"><Label>买入交叉</Label><Select disabled={!values.advancedRules.macd.enabled} value={values.advancedRules.macd.entry_cross} onValueChange={(value) => updateAdvancedRule("macd", "entry_cross", value as "golden" | "death")}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="golden">金叉</SelectItem><SelectItem value="death">死叉</SelectItem></SelectContent></Select></div></div>
              </div>

              <div className="grid gap-3 rounded-md border border-border/70 p-3">
                <div className="flex items-center justify-between gap-3"><div><p className="font-medium text-sm">布林带价格关系</p><p className="text-muted-foreground text-xs">可比较价格与上轨、中线或下轨的高低。</p></div><Switch checked={values.advancedRules.bollinger.enabled} onCheckedChange={(enabled) => updateAdvancedRule("bollinger", "enabled", enabled)} /></div>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5"><div className="grid gap-1.5"><Label>周期</Label><IntervalSelect disabled={!values.advancedRules.bollinger.enabled} id="advanced-bollinger-interval" onChange={(interval) => updateAdvancedRule("bollinger", "interval", interval as "15m")} value={values.advancedRules.bollinger.interval} /></div><NumericField disabled={!values.advancedRules.bollinger.enabled} id="advanced-bollinger-period" label="布林周期" max={500} min={2} onChange={(value) => updateAdvancedRule("bollinger", "period", value)} value={values.advancedRules.bollinger.period} /><NumericField disabled={!values.advancedRules.bollinger.enabled} id="advanced-bollinger-deviation" label="标准差" max={10} min={0.1} onChange={(value) => updateAdvancedRule("bollinger", "standard_deviations", value)} step={0.1} value={values.advancedRules.bollinger.standard_deviations} /><div className="grid gap-1.5"><Label>比较线</Label><Select disabled={!values.advancedRules.bollinger.enabled} value={values.advancedRules.bollinger.entry_reference} onValueChange={(value) => updateAdvancedRule("bollinger", "entry_reference", value as "upper" | "middle" | "lower")}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="upper">上轨</SelectItem><SelectItem value="middle">中线</SelectItem><SelectItem value="lower">下轨</SelectItem></SelectContent></Select></div><div className="grid gap-1.5"><Label>价格关系</Label><Select disabled={!values.advancedRules.bollinger.enabled} value={values.advancedRules.bollinger.entry_comparator} onValueChange={(value) => updateAdvancedRule("bollinger", "entry_comparator", value as "above" | "below")}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="above">价格高于</SelectItem><SelectItem value="below">价格低于</SelectItem></SelectContent></Select></div></div>
              </div>

              <AdvancedThresholdEditor label="RSI 阈值规则" description="可分别配置超卖买入和超买卖出。" rule={values.advancedRules.rsi} onChange={(changes) => update("advancedRules", { ...values.advancedRules, rsi: { ...values.advancedRules.rsi, ...changes } })} />
              <AdvancedThresholdEditor label="动能阈值规则" description="动能为当前收盘价与 N 根 K 线前收盘价之差，可设置 0 轴附近的自定义阈值。" rule={values.advancedRules.momentum} onChange={(changes) => update("advancedRules", { ...values.advancedRules, momentum: { ...values.advancedRules.momentum, ...changes } })} />
              <AdvancedThresholdEditor brar label="BRAR 阈值规则" description="支持选择 AR 或 BR 分量，并分别设置买入与卖出阈值。" rule={values.advancedRules.brar} onChange={(changes) => update("advancedRules", { ...values.advancedRules, brar: { ...values.advancedRules.brar, ...changes } })} />
            </CardContent>
          </Card>

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <RuleHeading icon={SlidersHorizontal} title={t("saas.operations.strategy.sections.riskLimits.title")} description={t("saas.operations.strategy.sections.riskLimits.description")} />
            <CardContent className="grid gap-4 px-4 py-4 sm:grid-cols-2 sm:px-5 lg:grid-cols-3">
              <NumericField error={errors.initialCapital} id="initial-capital" label="初始模拟资金" min={1} max={100_000_000} onChange={(value) => update("initialCapital", value)} step={100} unit="USDT" value={values.initialCapital} />
              <div className="grid gap-2"><div className="flex items-center justify-between gap-3"><Label htmlFor="take-profit-enabled">启用止盈</Label><Switch checked={values.takeProfitEnabled} id="take-profit-enabled" onCheckedChange={(enabled) => update("takeProfitEnabled", enabled)} /></div><NumericField disabled={!values.takeProfitEnabled} error={errors.takeProfit} id="take-profit" label={t("saas.operations.strategy.fields.takeProfit")} max={100} min={0.1} onChange={(value) => update("takeProfit", value)} step={0.1} unit="%" value={values.takeProfit} /></div>
              <div className="grid gap-2"><div className="flex items-center justify-between gap-3"><Label htmlFor="stop-loss-enabled">启用止损</Label><Switch checked={values.stopLossEnabled} id="stop-loss-enabled" onCheckedChange={(enabled) => update("stopLossEnabled", enabled)} /></div><NumericField disabled={!values.stopLossEnabled} error={errors.stopLoss} id="stop-loss" label={t("saas.operations.strategy.fields.stopLoss")} max={100} min={0.1} onChange={(value) => update("stopLoss", value)} step={0.1} unit="%" value={values.stopLoss} /></div>
              <NumericField error={errors.maximumPositions} id="maximum-positions" label={t("saas.operations.strategy.fields.maximumOpenPositions")} max={100} min={1} onChange={(value) => update("maximumPositions", value)} value={values.maximumPositions} />
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
              <SummaryRow label="初始模拟资金" value={`${values.initialCapital.toLocaleString()} USDT`} />
              <SummaryRow label="资金分配" value="可用资金 / 符合条件币种数量（向下取整）" />
              <SummaryRow label={t("saas.operations.strategy.summary.exitLimits")} value={t("saas.operations.strategy.summary.exitLimitsValue", { takeProfit: values.takeProfit, stopLoss: values.stopLoss })} />
              <SummaryRow label={t("saas.operations.strategy.summary.exposureCap")} value={t("saas.operations.strategy.summary.exposureCapValue", { positions: values.maximumPositions, leverage: values.leverage })} />
            </CardContent>
          </Card>

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <CardHeader className="border-b px-4 py-4">
              <CardTitle className="flex items-center gap-2 text-base"><BrainCircuit /> AI 策略审阅</CardTitle>
              <CardDescription>在独立页面阅读完整中文分析与确定性评估证据。</CardDescription>
            </CardHeader>
            <CardContent className="px-4 py-4">
              {strategyId ? <Button asChild className="w-full" type="button" variant="outline"><Link to="/strategies/advisory"><BrainCircuit /> 打开 AI 策略审阅</Link></Button> : <Button className="w-full" disabled type="button" variant="outline"><BrainCircuit /> 请先保存策略</Button>}
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

          <Card className="gap-0 rounded-lg border-sky-500/30 py-0 shadow-none">
            <CardHeader className="border-b px-4 py-4"><CardTitle className="flex items-center gap-2 text-base"><WandSparkles className="text-sky-500" /> AI 策略文本导入</CardTitle><CardDescription>粘贴自然语言策略。AI 只会生成待审核的参数草稿，不会保存、启动或下单。</CardDescription></CardHeader>
            <CardContent className="grid gap-3 px-4 py-4"><Textarea className="min-h-44 text-sm" onChange={(event) => setStrategyText(event.target.value)} placeholder="例如：买入以 15 分钟为主，价格高于日线 20 日均线；5 分钟 MACD 金叉；15 分钟价格高于布林中线；RSI 低于 20；动能低于 20；BR 低于 30。卖出：RSI 或动能高于 85 时全部卖出。" value={strategyText} /><Button disabled={parseStrategyText.isPending || strategyText.trim().length < 10} onClick={importStrategyText} type="button" variant="outline"><FileText /> {parseStrategyText.isPending ? "AI 正在拆解策略" : "解析并回填参数"}</Button>{textImportSummary ? <Alert className="border-sky-500/30 bg-sky-500/5"><WandSparkles /><AlertTitle>AI 结构化结果</AlertTitle><AlertDescription>{textImportSummary}{unresolvedItems.length ? <span className="mt-2 block text-amber-700 dark:text-amber-300">待人工确认：{unresolvedItems.join("；")}</span> : <span className="mt-2 block text-emerald-700 dark:text-emerald-300">已回填到多周期高级规则，请检查后保存。</span>}</AlertDescription></Alert> : null}</CardContent>
          </Card>
        </aside>
      </div>
    </div>
  );
}


export default function Strategies() {
  return <RuleStrategyConfiguration />;
}
