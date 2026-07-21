import type {
  AdvancedRuleSetConfig,
  RuleStrategyConfig,
  RuleStrategyInterval,
  RuleStrategyStatus,
} from "../../types/rule-strategy";

export type StrategyFormValues = {
  symbols: string[];
  timeframe: RuleStrategyInterval;
  confirmationMode: "all" | "any";
  decideIntervalSeconds: number | null;
  movingAverageEnabled: boolean;
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
  momentumPeriod: number;
  macdFast: number;
  macdSlow: number;
  macdSignal: number;
  initialCapital: number;
  orderQuoteAmount: number;
  advancedRules: AdvancedRuleSetConfig;
  takeProfitEnabled: boolean;
  takeProfit: number;
  stopLossEnabled: boolean;
  stopLoss: number;
  maximumPositions: number;
  leverage: number;
  executionEnvironment: "paper" | "okx_demo";
  sandboxConnectionId: string;
  maxDemoOrderQuoteAmount: number;
  maxDemoDailyQuoteAmount: number;
  maxDemoTotalQuoteAmount: number;
};

export const defaultStrategyFormValues: StrategyFormValues = {
  symbols: [],
  timeframe: "15m",
  confirmationMode: "all",
  decideIntervalSeconds: null,
  movingAverageEnabled: true,
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
  momentumPeriod: 14,
  macdFast: 12,
  macdSlow: 26,
  macdSignal: 9,
  initialCapital: 10_000,
  orderQuoteAmount: 100,
  advancedRules: {
    enabled: true,
    entry_confirmation_mode: "all",
    entry_confirmation_count: 1,
    entry_confirmation_ratio: 1,
    exit_confirmation_mode: "any",
    moving_average: {
      enabled: true,
      interval: "1d",
      period: 20,
      entry_comparator: "above",
    },
    macd: {
      enabled: true,
      interval: "5m",
      fast_window: 12,
      slow_window: 26,
      signal_window: 9,
      entry_cross: "golden",
    },
    bollinger: {
      enabled: true,
      interval: "15m",
      period: 20,
      standard_deviations: 2,
      entry_reference: "middle",
      entry_comparator: "above",
    },
    rsi: {
      enabled: true,
      interval: "15m",
      period: 14,
      entry_comparator: "below",
      entry_threshold: 20,
      exit_enabled: true,
      exit_comparator: "above",
      exit_threshold: 85,
    },
    momentum: {
      enabled: true,
      interval: "15m",
      period: 14,
      entry_comparator: "below",
      entry_threshold: 20,
      exit_enabled: true,
      exit_comparator: "above",
      exit_threshold: 85,
    },
    brar: {
      enabled: true,
      interval: "15m",
      period: 26,
      component: "br",
      entry_comparator: "below",
      entry_threshold: 30,
      exit_enabled: false,
      exit_comparator: "above",
      exit_threshold: 85,
    },
  },
  takeProfitEnabled: false,
  takeProfit: 4,
  stopLossEnabled: false,
  stopLoss: 2,
  maximumPositions: 100,
  leverage: 1,
  executionEnvironment: "paper",
  sandboxConnectionId: "",
  maxDemoOrderQuoteAmount: 100,
  maxDemoDailyQuoteAmount: 500,
  maxDemoTotalQuoteAmount: 1_000,
};

export type PersistedRuleStrategyConfig = Omit<
  RuleStrategyConfig,
  "advanced_rules" | "execution" | "initial_capital_quote"
> &
  Partial<
    Pick<
      RuleStrategyConfig,
      "advanced_rules" | "execution" | "initial_capital_quote"
    >
  >;

export function ruleStrategyConfigToFormValues(
  config: PersistedRuleStrategyConfig,
): StrategyFormValues {
  const execution = config.execution;
  return {
    symbols: config.symbols.map((symbol) => symbol.replace("-", "/")),
    timeframe: config.interval,
    confirmationMode: config.confirmation_mode,
    decideIntervalSeconds: config.decide_interval_s ?? null,
    movingAverageEnabled: config.moving_average.enabled,
    fastMa: config.moving_average.short_window,
    slowMa: config.moving_average.long_window,
    rsiEnabled: config.rsi.enabled,
    rsiPeriod: config.rsi.period,
    rsiOversold: config.rsi.oversold,
    rsiOverbought: config.rsi.overbought,
    bollingerEnabled: config.bollinger.enabled,
    bollingerPeriod: config.bollinger.period,
    bollingerDeviation: config.bollinger.standard_deviations,
    momentumEnabled: config.momentum_macd.enabled,
    momentumPeriod: config.momentum_macd.momentum_period,
    macdFast: config.momentum_macd.macd_fast_window,
    macdSlow: config.momentum_macd.macd_slow_window,
    macdSignal: config.momentum_macd.macd_signal_window,
    initialCapital:
      config.initial_capital_quote ?? defaultStrategyFormValues.initialCapital,
    orderQuoteAmount: config.risk.order_quote_amount,
    advancedRules: structuredClone(
      config.advanced_rules ?? defaultStrategyFormValues.advancedRules,
    ),
    takeProfitEnabled: config.risk.take_profit_pct !== undefined,
    takeProfit:
      config.risk.take_profit_pct !== undefined
        ? config.risk.take_profit_pct * 100
        : defaultStrategyFormValues.takeProfit,
    stopLossEnabled: config.risk.stop_loss_pct !== undefined,
    stopLoss:
      config.risk.stop_loss_pct !== undefined
        ? config.risk.stop_loss_pct * 100
        : defaultStrategyFormValues.stopLoss,
    maximumPositions: config.risk.max_positions,
    leverage: config.risk.leverage,
    executionEnvironment: execution?.environment ?? "paper",
    sandboxConnectionId: execution?.sandbox_connection_id ?? "",
    maxDemoOrderQuoteAmount:
      execution?.max_order_quote_amount ??
      defaultStrategyFormValues.maxDemoOrderQuoteAmount,
    maxDemoDailyQuoteAmount:
      execution?.max_daily_quote_amount ??
      defaultStrategyFormValues.maxDemoDailyQuoteAmount,
    maxDemoTotalQuoteAmount:
      execution?.max_total_quote_amount ??
      defaultStrategyFormValues.maxDemoTotalQuoteAmount,
  };
}

export function strategyFormValuesToConfig(
  values: StrategyFormValues,
): RuleStrategyConfig {
  return {
    mode: "paper",
    initial_capital_quote: values.initialCapital,
    confirmation_mode: values.confirmationMode,
    symbols: values.symbols.map((symbol) => symbol.replace("/", "-")),
    interval: values.timeframe,
    decide_interval_s: values.decideIntervalSeconds,
    moving_average: {
      enabled: values.movingAverageEnabled,
      short_window: values.fastMa,
      long_window: values.slowMa,
    },
    rsi: {
      enabled: values.rsiEnabled,
      period: values.rsiPeriod,
      oversold: values.rsiOversold,
      overbought: values.rsiOverbought,
    },
    bollinger: {
      enabled: values.bollingerEnabled,
      period: values.bollingerPeriod,
      standard_deviations: values.bollingerDeviation,
    },
    momentum_macd: {
      enabled: values.momentumEnabled,
      momentum_period: values.momentumPeriod,
      macd_fast_window: values.macdFast,
      macd_slow_window: values.macdSlow,
      macd_signal_window: values.macdSignal,
    },
    advanced_rules: structuredClone(values.advancedRules),
    execution: {
      environment: values.executionEnvironment,
      ...(values.executionEnvironment === "okx_demo"
        ? { sandbox_connection_id: values.sandboxConnectionId }
        : {}),
      max_order_quote_amount: values.maxDemoOrderQuoteAmount,
      max_daily_quote_amount: values.maxDemoDailyQuoteAmount,
      max_total_quote_amount: values.maxDemoTotalQuoteAmount,
    },
    risk: {
      order_quote_amount: values.orderQuoteAmount,
      take_profit_pct: values.takeProfitEnabled
        ? values.takeProfit / 100
        : undefined,
      stop_loss_pct: values.stopLossEnabled ? values.stopLoss / 100 : undefined,
      max_positions: values.maximumPositions,
      leverage: values.leverage,
    },
  };
}

type ConfigurationLifecycleInput = {
  strategiesPending: boolean;
  strategyCount: number;
  activeStrategyId: string;
  detailPending: boolean;
  hasDetail: boolean;
  hasError?: boolean;
  status?: RuleStrategyStatus;
  dirty?: boolean;
};

type ConfigurationAction = "save" | "start" | "stop";

export function configurationLifecycle({
  strategiesPending,
  strategyCount,
  activeStrategyId,
  detailPending,
  hasDetail,
  hasError = false,
  status,
  dirty = false,
}: ConfigurationLifecycleInput): {
  loading: boolean;
  readOnly: boolean;
  actions: ConfigurationAction[];
} {
  const selectionPending = !hasError && strategyCount > 0 && !activeStrategyId;
  const selectedDetailPending =
    !hasError && Boolean(activeStrategyId) && (detailPending || !hasDetail);
  const loading =
    !hasError && (strategiesPending || selectionPending || selectedDetailPending);

  if (loading) return { loading: true, readOnly: true, actions: [] };
  if (status === "running") {
    return { loading: false, readOnly: true, actions: ["stop"] };
  }

  return {
    loading: false,
    readOnly: false,
    actions:
      activeStrategyId && !dirty ? ["save", "start"] : ["save"],
  };
}
