import type { DemoConnection, StrategyConfig } from "../types";

export const STRATEGY_INTERVALS = [
  "1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d",
] as const;

export function createStrategyConfig(): StrategyConfig {
  return {
    mode: "paper",
    initial_capital_quote: 10_000,
    confirmation_mode: "all",
    symbols: ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
    interval: "15m",
    decide_interval_s: null,
    moving_average: { enabled: true, short_window: 20, long_window: 50 },
    rsi: { enabled: true, period: 14, oversold: 30, overbought: 70 },
    bollinger: { enabled: true, period: 20, standard_deviations: 2 },
    momentum_macd: {
      enabled: true,
      momentum_period: 9,
      macd_fast_window: 12,
      macd_slow_window: 26,
      macd_signal_window: 9,
    },
    advanced_rules: {
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
    execution: { environment: "paper", max_order_quote_amount: 100, max_daily_quote_amount: 500, max_total_quote_amount: 1_000 },
    risk: { order_quote_amount: 100, max_positions: 100, leverage: 1 },
  };
}

export function normalizeSymbols(rawSymbols: string[]): string[] {
  return [...new Set(rawSymbols.map((raw) => raw.trim().toUpperCase().replace("/", "-")).filter(Boolean).map((symbol) => symbol.endsWith("-USDT") ? symbol : `${symbol.replace(/-USDT$/, "")}-USDT`))];
}

export function validOkxDemoConnections(connections: DemoConnection[]): DemoConnection[] {
  return connections.filter((connection) => connection.provider === "okx" && connection.metadata.sandbox && connection.metadata.market_type === "spot" && !connection.revoked);
}

export function validateStrategyConfig(
  config: StrategyConfig,
  connections: DemoConnection[],
): string | null {
  const symbols = normalizeSymbols(config.symbols);
  if (symbols.length === 0 || symbols.some((symbol) => !symbol.endsWith("-USDT"))) return "至少需要一个唯一的 USDT 交易对。";
  if (config.initial_capital_quote <= 0 || config.risk.order_quote_amount <= 0 || config.risk.max_positions <= 0 || config.risk.leverage <= 0) return "初始资金、单笔金额、最大持仓和杠杆必须大于 0。";
  const execution = config.execution;
  if (execution.max_order_quote_amount <= 0 || execution.max_daily_quote_amount <= 0 || execution.max_total_quote_amount <= 0) return "执行额度必须大于 0。";
  if (execution.environment === "okx_demo") {
    if (config.risk.leverage !== 1) return "OKX Demo 策略的杠杆必须为 1。";
    if (!execution.sandbox_connection_id || !validOkxDemoConnections(connections).some((item) => item.id === execution.sandbox_connection_id)) return "请选择当前工作区已验证、未撤销的 OKX Demo 现货连接。";
    if (execution.max_daily_quote_amount < execution.max_order_quote_amount || execution.max_total_quote_amount < execution.max_order_quote_amount) return "Demo 的每日及总额度不得低于单笔额度。";
  }
  return null;
}
