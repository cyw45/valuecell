import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { RuleStrategyConfig } from "../../types/rule-strategy.ts";
import {
  configurationLifecycle,
  defaultStrategyFormValues,
  ruleStrategyConfigToFormValues,
  strategyFormValuesToConfig,
} from "./strategy-configuration-lifecycle.ts";

const persistedConfig: RuleStrategyConfig = {
  mode: "paper",
  initial_capital_quote: 43_210,
  confirmation_mode: "any",
  symbols: ["BTC-USDT", "SOL-USDT"],
  interval: "4h",
  decide_interval_s: 45,
  moving_average: { enabled: false, short_window: 7, long_window: 91 },
  rsi: { enabled: false, period: 9, oversold: 17, overbought: 83 },
  bollinger: { enabled: false, period: 31, standard_deviations: 3.25 },
  momentum_macd: {
    enabled: false,
    momentum_period: 19,
    macd_fast_window: 8,
    macd_slow_window: 34,
    macd_signal_window: 13,
  },
  advanced_rules: {
    enabled: false,
    entry_confirmation_mode: "ratio",
    entry_confirmation_count: 3,
    entry_confirmation_ratio: 0.75,
    exit_confirmation_mode: "all",
    moving_average: {
      enabled: false,
      interval: "1h",
      period: 37,
      entry_comparator: "below",
    },
    macd: {
      enabled: false,
      interval: "30m",
      fast_window: 6,
      slow_window: 41,
      signal_window: 11,
      entry_cross: "death",
    },
    bollinger: {
      enabled: false,
      interval: "3m",
      period: 29,
      standard_deviations: 1.75,
      entry_reference: "lower",
      entry_comparator: "below",
    },
    rsi: {
      enabled: false,
      interval: "5m",
      period: 18,
      entry_comparator: "above",
      entry_threshold: 63,
      exit_enabled: false,
      exit_comparator: "below",
      exit_threshold: 22,
    },
    momentum: {
      enabled: false,
      interval: "1d",
      period: 23,
      entry_comparator: "above",
      entry_threshold: 12,
      exit_enabled: false,
      exit_comparator: "below",
      exit_threshold: -7,
    },
    brar: {
      enabled: false,
      interval: "4h",
      period: 44,
      component: "ar",
      entry_comparator: "above",
      entry_threshold: 140,
      exit_enabled: false,
      exit_comparator: "below",
      exit_threshold: 80,
    },
  },
  program: null,
  execution: {
    environment: "okx_demo",
    sandbox_connection_id: "connection-new",
    max_order_quote_amount: 321,
    max_daily_quote_amount: 4_321,
    max_total_quote_amount: 12_345,
  },
  risk: {
    order_quote_amount: 275,
    take_profit_pct: undefined,
    stop_loss_pct: undefined,
    trailing_take_profit_pct: undefined,
    max_total_position_pct: 0.6,
    max_symbol_position_pct: 0.1,
    add_to_winners: false,
    max_additions: 0,
    max_positions: 7,
    leverage: 1,
  },
};

test("complete config mapping replaces every form value without retaining an old strategy", () => {
  const staleValues = {
    ...defaultStrategyFormValues,
    takeProfitEnabled: true,
    takeProfit: 99,
    stopLossEnabled: true,
    stopLoss: 88,
    maximumPositions: 99,
    leverage: 5,
    sandboxConnectionId: "connection-old",
  };

  const mapped = ruleStrategyConfigToFormValues(persistedConfig);

  assert.notDeepEqual(mapped, staleValues);
  assert.deepEqual(mapped, {
    symbols: ["BTC/USDT", "SOL/USDT"],
    timeframe: "4h",
    confirmationMode: "any",
    decideIntervalSeconds: 45,
    movingAverageEnabled: false,
    fastMa: 7,
    slowMa: 91,
    rsiEnabled: false,
    rsiPeriod: 9,
    rsiOversold: 17,
    rsiOverbought: 83,
    bollingerEnabled: false,
    bollingerPeriod: 31,
    bollingerDeviation: 3.25,
    momentumEnabled: false,
    momentumPeriod: 19,
    macdFast: 8,
    macdSlow: 34,
    macdSignal: 13,
    initialCapital: 43_210,
    orderQuoteAmount: 275,
    advancedRules: persistedConfig.advanced_rules,
    program: null,
    takeProfitEnabled: false,
    takeProfit: defaultStrategyFormValues.takeProfit,
    stopLossEnabled: false,
    stopLoss: defaultStrategyFormValues.stopLoss,
    trailingTakeProfitEnabled: false,
    trailingTakeProfit: defaultStrategyFormValues.trailingTakeProfit,
    maxTotalPosition: 60,
    maxSymbolPosition: 10,
    addToWinners: false,
    maxAdditions: 0,
    maximumPositions: 7,
    leverage: 1,
    executionEnvironment: "okx_demo",
    sandboxConnectionId: "connection-new",
    maxDemoOrderQuoteAmount: 321,
    maxDemoDailyQuoteAmount: 4_321,
    maxDemoTotalQuoteAmount: 12_345,
  });
});

test("legacy persisted configuration without later nested fields uses safe defaults", () => {
  const {
    advanced_rules: _advancedRules,
    execution: _execution,
    initial_capital_quote: _initialCapital,
    ...legacyConfig
  } = structuredClone(persistedConfig);

  const values = ruleStrategyConfigToFormValues(legacyConfig);

  assert.equal(values.executionEnvironment, "paper");
  assert.equal(values.sandboxConnectionId, "");
  assert.equal(values.initialCapital, defaultStrategyFormValues.initialCapital);
  assert.deepEqual(values.advancedRules, defaultStrategyFormValues.advancedRules);
  assert.equal(
    values.maxDemoOrderQuoteAmount,
    defaultStrategyFormValues.maxDemoOrderQuoteAmount,
  );
  assert.equal(
    values.maxDemoDailyQuoteAmount,
    defaultStrategyFormValues.maxDemoDailyQuoteAmount,
  );
  assert.equal(
    values.maxDemoTotalQuoteAmount,
    defaultStrategyFormValues.maxDemoTotalQuoteAmount,
  );
});

test("persisted execution parameters survive a form round trip", () => {
  assert.deepEqual(
    strategyFormValuesToConfig(ruleStrategyConfigToFormValues(persistedConfig)),
    persistedConfig,
  );
});

test("every persisted base execution parameter has a visible form control", () => {
  const source = readFileSync(
    new URL("./strategies.tsx", import.meta.url),
    "utf8",
  );

  for (const controlId of [
    "base-confirmation-mode",
    "decide-interval-seconds",
    "moving-average-enabled",
    "momentum-period",
    "trailing-take-profit-enabled",
    "max-total-position",
    "max-symbol-position",
    "add-to-winners",
    "max-additions",
    "compiled-program",
  ]) {
    assert.match(source, new RegExp(`id=["']${controlId}["']`));
  }
});

test("running configuration is read-only, cannot save, and only exposes stop", () => {
  assert.deepEqual(
    configurationLifecycle({
      strategiesPending: false,
      strategyCount: 1,
      activeStrategyId: "active",
      detailPending: false,
      hasDetail: true,
      status: "running",
    }),
    { loading: false, readOnly: true, actions: ["stop"] },
  );
});

test("initial strategy selection and selected detail pending keep the whole page loading", () => {
  const selecting = configurationLifecycle({
    strategiesPending: false,
    strategyCount: 1,
    activeStrategyId: "",
    detailPending: false,
    hasDetail: false,
  });
  const loadingDetail = configurationLifecycle({
    strategiesPending: false,
    strategyCount: 1,
    activeStrategyId: "active",
    detailPending: true,
    hasDetail: false,
  });

  assert.equal(selecting.loading, true);
  assert.equal(loadingDetail.loading, true);
});

test("a failed strategy request exits loading so its error can render", () => {
  assert.equal(
    configurationLifecycle({
      strategiesPending: false,
      strategyCount: 1,
      activeStrategyId: "active",
      detailPending: false,
      hasDetail: false,
      hasError: true,
    }).loading,
    false,
  );
});

test("stopping unlocks editing, then saving permits a restart", () => {
  const stopped = configurationLifecycle({
    strategiesPending: false,
    strategyCount: 1,
    activeStrategyId: "active",
    detailPending: false,
    hasDetail: true,
    status: "stopped",
    dirty: false,
  });
  const edited = configurationLifecycle({
    strategiesPending: false,
    strategyCount: 1,
    activeStrategyId: "active",
    detailPending: false,
    hasDetail: true,
    status: "stopped",
    dirty: true,
  });
  const saved = configurationLifecycle({
    strategiesPending: false,
    strategyCount: 1,
    activeStrategyId: "active",
    detailPending: false,
    hasDetail: true,
    status: "stopped",
    dirty: false,
  });

  assert.equal(stopped.readOnly, false);
  assert.deepEqual(edited.actions, ["save"]);
  assert.deepEqual(saved.actions, ["save", "start"]);
});
