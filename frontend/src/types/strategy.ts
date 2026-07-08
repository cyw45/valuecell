// Strategy types

export interface Strategy {
  strategy_id: number;
  strategy_name: string;
  strategy_type: "PromptBasedStrategy" | "GridStrategy";
  status: "running" | "stopped";
  stop_reason?: string;
  trading_mode: "live" | "virtual";
  total_pnl: number;
  total_pnl_pct: number;
  created_at: string;
  exchange_id: string;
  model_id: string;
}

// Strategy Performance types
export type StrategyPerformance = {
  strategy_id: string;
  initial_capital: number;
  return_rate_pct: number;
  llm_provider: string;
  llm_model_id: string;
  exchange_id: string;
  strategy_type: Strategy["strategy_type"];
  max_leverage: number;
  symbols: string[];
  prompt: string;
  prompt_name: string;
  trading_mode: Strategy["trading_mode"];
  decide_interval: number;
};

export interface StrategyMarketDataHealth {
  ok: boolean;
  provider?: string;
  fetched_count: number;
  missing_count: number;
  missing_symbols: string[];
}

export interface StrategyDiagnosticsCycle {
  compose_id: string;
  cycle_index?: number;
  created_at?: string;
  rationale?: string;
  instruction_count: number;
  order_count: number;
  no_order_count: number;
  market_data_health: StrategyMarketDataHealth;
}

export interface StrategySymbolDecision {
  symbol: string;
  intervals_seen: string[];
  has_market_snapshot: boolean;
  latest_price?: number;
  action?: string;
  quantity?: number;
  reason?: string;
}

export interface StrategyDiagnostics {
  strategy_id: string;
  strategy_name?: string;
  status?: string;
  trading_mode?: string;
  exchange_id?: string;
  strategy_type?: Strategy["strategy_type"];
  runtime_health: Record<string, unknown>;
  config: Record<string, unknown>;
  observed_symbol_count: number;
  expected_symbol_count: number;
  latest_cycle?: StrategyDiagnosticsCycle;
  symbol_decisions: StrategySymbolDecision[];
  recent_cycles: StrategyDiagnosticsCycle[];
}

// Position types
export interface Position {
  symbol: string;
  type: "LONG" | "SHORT";
  leverage: number;
  entry_price: number;
  quantity: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

// Strategy Action types
export interface StrategyAction {
  instruction_id: string;
  symbol: string;
  action: "open_long" | "open_short" | "close_long" | "close_short";
  action_display: string;
  side: "BUY" | "SELL";
  quantity: number;
  leverage: number;
  entry_price: number;
  exit_price?: number;
  entry_at: string;
  exit_at?: string;
  fee_cost: number;
  realized_pnl: number;
  realized_pnl_pct: number;
  rationale: string;
  holding_time_ms: number;
}

// Strategy Compose types
export interface StrategyCompose {
  compose_id: string;
  created_at: string;
  rationale: string;
  cycle_index: number;
  actions: StrategyAction[];
}

// Strategy Prompt types
export interface StrategyPrompt {
  id: string;
  name: string;
  content: string;
}

// Create Strategy types
export interface CreateStrategy {
  // LLM Model Configuration
  llm_model_config: {
    provider: string; // e.g. 'openrouter'
    model_id: string; // e.g. 'deepseek-ai/deepseek-v3.1'
    api_key: string;
  };

  // Exchange Configuration
  exchange_config: {
    exchange_id: string; // e.g. 'okx'
    trading_mode: "live" | "virtual";
    api_key: string;
    secret_key: string;
    passphrase: string; // Required for some exchanges like OKX
    wallet_address: string;
    private_key: string;
  };

  // Trading Strategy Configuration
  trading_config: {
    strategy_name: string;
    initial_capital: number;
    max_leverage: number;
    symbols: string[]; // e.g. ['BTC', 'ETH', ...]
    template_id: string;
    decide_interval: number;
    strategy_type: Strategy["strategy_type"];
  };
}

// Copy Strategy types
export interface CopyStrategy {
  // LLM Model Configuration
  llm_model_config: {
    provider: string; // e.g. 'openrouter'
    model_id: string; // e.g. 'deepseek-ai/deepseek-v3.1'
    api_key: string;
  };

  // Exchange Configuration
  exchange_config: {
    exchange_id: string; // e.g. 'okx'
    trading_mode: "live" | "virtual";
    api_key: string;
    secret_key: string;
    passphrase: string; // Required for some exchanges like OKX
    wallet_address: string;
    private_key: string;
  };

  // Trading Strategy Configuration
  trading_config: {
    strategy_name: string;
    initial_capital: number;
    max_leverage: number;
    symbols: string[]; // e.g. ['BTC', 'ETH', ...]
    decide_interval: number;
    strategy_type: Strategy["strategy_type"];
    prompt_name: string;
    prompt: string;
  };
}

// Portfolio Summary types
export interface PortfolioSummary {
  cash: number;
  total_value: number;
  total_pnl: number;
  total_pnl_pct?: number;
  gross_exposure?: number;
  net_exposure?: number;
}
