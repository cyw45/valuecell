import type {
  AccountStrategyOverview,
  AllocationState,
  ExecutionGate,
  StrategyAllocation,
  StrategyStatus,
} from "./multi-strategy";

/**
 * Mobile twin of the Web dashboard's concurrency view models. Both clients read
 * the same persisted allocator facts, so the arithmetic here deliberately mirrors
 * `frontend/src/app/dashboard.tsx` rather than inventing a second opinion: the
 * same denominators, the same zero-baseline scaling, and the same rule that a
 * missing fact renders as unavailable instead of zero.
 */

export type ConcurrencyTone = "default" | "positive" | "negative" | "warning";

/** Minimal shape needed to name a strategy; keeps this module free of client types. */
export type NamedStrategy = {
  strategy_id: string;
  name: string;
};

const SYNC_STATUS_LABELS: Record<string, string> = {
  healthy: "钱包已同步",
  stale: "钱包数据延迟",
  unavailable: "钱包不可用",
};

const ATTRIBUTION_STATUS_LABELS: Record<string, string> = {
  complete: "归因完整",
  partial: "归因不完整",
  unavailable: "归因不可用",
};

const ALLOCATION_STATE_LABELS: Record<AllocationState, string> = {
  available: "可分配",
  reserved: "已预留",
  occupied: "已占用",
  partially_released: "部分释放",
  released: "已释放",
  submission_unknown: "提交结果待对账",
  recovery_required: "等待对账恢复",
  blocked: "已阻断",
};

const STRATEGY_STATUS_LABELS: Record<StrategyStatus, string> = {
  running: "运行中",
  paused: "已暂停",
  stopped: "已停止",
  archived: "已归档",
};

const EXECUTION_GATE_LABELS: Record<ExecutionGate["status"], string> = {
  ready: "可开仓",
  protected: "只读保护",
  blocked: "开仓已阻断",
};

const STRATEGY_KIND_LABELS: Record<string, string> = {
  configurable_rule: "多币种趋势共振",
  dual_ma_trend: "双均线趋势",
  pair_rotation: "配对套利",
  leader_breakout: "现货龙头",
};

export function strategyKindLabel(kind: string): string {
  return STRATEGY_KIND_LABELS[kind] ?? kind;
}

export function syncStatusLabel(status: string): string {
  return SYNC_STATUS_LABELS[status] ?? "钱包状态未知";
}

export function attributionStatusLabel(status: string): string {
  return ATTRIBUTION_STATUS_LABELS[status] ?? "归因状态未知";
}

export function allocationStateLabel(state: AllocationState): string {
  return ALLOCATION_STATE_LABELS[state] ?? "状态未知";
}

export function strategyStatusText(status: StrategyStatus): string {
  return STRATEGY_STATUS_LABELS[status] ?? "状态未知";
}

export function executionGateLabel(status: ExecutionGate["status"]): string {
  return EXECUTION_GATE_LABELS[status] ?? "开仓状态未知";
}

export function executionGateTone(status: ExecutionGate["status"]): ConcurrencyTone {
  if (status === "ready") return "positive";
  if (status === "protected") return "warning";
  return "negative";
}

export function syncStatusTone(status: string): ConcurrencyTone {
  if (status === "healthy") return "positive";
  if (status === "stale") return "warning";
  return "negative";
}

export function attributionStatusTone(status: string): ConcurrencyTone {
  if (status === "complete") return "positive";
  if (status === "partial") return "warning";
  return "negative";
}

/** Unsettled allocation states are the ones an operator must act on. */
export function allocationStateTone(state: AllocationState): ConcurrencyTone {
  if (state === "blocked" || state === "submission_unknown" || state === "recovery_required") {
    return "negative";
  }
  if (state === "reserved" || state === "occupied" || state === "partially_released") {
    return "warning";
  }
  return "default";
}

export function strategyStatusTone(status: StrategyStatus): ConcurrencyTone {
  if (status === "running") return "positive";
  if (status === "paused") return "warning";
  return "default";
}

export function winRateTone(percent: number): ConcurrencyTone {
  if (percent >= 60) return "positive";
  if (percent >= 40) return "warning";
  return "negative";
}

export function utilizationTone(percent: number): ConcurrencyTone {
  if (percent >= 85) return "negative";
  if (percent >= 55) return "warning";
  return "positive";
}

export function formatUsdt(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} USDT`;
}

/** Signed money for diverging readouts, matching the Web `+`/`−` convention. */
export function formatSignedUsdt(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : "−"}${formatUsdt(Math.abs(value))}`;
}

export function formatPercent(value: number | null | undefined, fractionDigits = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(fractionDigits)}%`;
}

/** Ratios arrive as 0-1 from the allocator; the UI always shows them as percent. */
export function formatRatioPercent(
  ratio: number | null | undefined,
  fractionDigits = 1,
): string {
  if (ratio == null || !Number.isFinite(ratio)) return "—";
  return `${(ratio * 100).toFixed(fractionDigits)}%`;
}

export function resolveAllocationName(
  allocation: StrategyAllocation,
  strategies: readonly NamedStrategy[] | undefined,
): string {
  return (
    strategies?.find((item) => item.strategy_id === allocation.strategy_id)?.name ??
    allocation.kind
  );
}

export type CapitalMeterRow = {
  strategyId: string;
  label: string;
  cap: number | null;
  reserved: number;
  occupied: number;
  reservedPercent: number;
  occupiedPercent: number;
};

/**
 * 策略资金水位: each strategy is measured against its own persisted cap so the
 * picture cannot drift from the limit the allocator actually enforces.
 */
export function capitalMeterRows(
  allocations: readonly StrategyAllocation[],
  resolveName: (allocation: StrategyAllocation) => string,
): CapitalMeterRow[] {
  return allocations.map((allocation) => {
    const cap = allocation.max_reserved_quote;
    const reserved = Math.max(allocation.reserved_quote, 0);
    const occupied = Math.max(allocation.occupied_quote, 0);
    const denominator = cap != null && cap > 0 ? cap : Math.max(reserved + occupied, 1);
    return {
      strategyId: allocation.strategy_id,
      label: resolveName(allocation),
      cap,
      reserved,
      occupied,
      reservedPercent: Math.min((reserved / denominator) * 100, 100),
      occupiedPercent: Math.min((occupied / denominator) * 100, 100),
    };
  });
}

export type PnlComparisonRow = {
  strategyId: string;
  label: string;
  value: number;
  returnRatePercent: number | null;
  completedTrades: number;
  positive: boolean;
  widthPercent: number;
};

/** 各策略净收益对比: diverging bars around a shared zero baseline. */
export function pnlComparisonRows(
  allocations: readonly StrategyAllocation[],
  resolveName: (allocation: StrategyAllocation) => string,
): PnlComparisonRow[] {
  const settled = allocations.filter((allocation) => allocation.net_pnl_quote != null);
  const maxAbs = Math.max(
    ...settled.map((allocation) => Math.abs(allocation.net_pnl_quote ?? 0)),
    1,
  );
  return settled.map((allocation) => {
    const value = allocation.net_pnl_quote ?? 0;
    return {
      strategyId: allocation.strategy_id,
      label: resolveName(allocation),
      value,
      returnRatePercent: allocation.return_rate_pct,
      completedTrades: allocation.completed_trade_count,
      positive: value >= 0,
      widthPercent: (Math.abs(value) / maxAbs) * 50,
    };
  });
}

export type WalletBudgetSegment = {
  strategyId: string;
  label: string;
  cap: number;
  /** Share of the wallet's usable balance this cap promises, as a 0-100 percent. */
  sharePercent: number | null;
  widthPercent: number;
  colorIndex: number;
};

export type WalletBudgetChart = {
  segments: WalletBudgetSegment[];
  allocated: number;
  unallocated: number;
  unallocatedPercent: number | null;
  unallocatedWidthPercent: number;
  /** True when the caps collectively promise more than the wallet holds. */
  overCommit: boolean;
  base: number | null;
};

/**
 * 共享钱包资金预算: one stacked bar of every persisted cap plus the unallocated
 * buffer, scaled so over-commitment stays visible instead of being clipped.
 */
export function walletBudgetChart(
  allocations: readonly StrategyAllocation[],
  resolveName: (allocation: StrategyAllocation) => string,
  walletAvailableQuote: number | null,
  walletEquityQuote: number | null,
): WalletBudgetChart {
  const base = walletAvailableQuote ?? walletEquityQuote;
  const rows = allocations
    .map((allocation, index) => ({
      strategyId: allocation.strategy_id,
      label: resolveName(allocation),
      cap: Math.max(allocation.max_reserved_quote ?? 0, 0),
      colorIndex: index,
    }))
    .filter((row) => row.cap > 0)
    .sort((left, right) => right.cap - left.cap);
  const allocated = rows.reduce((total, row) => total + row.cap, 0);
  const scale = Math.max(allocated, base ?? 0, 1);
  const unallocated = Math.max((base ?? allocated) - allocated, 0);
  return {
    segments: rows.map((row) => ({
      ...row,
      sharePercent: base != null && base > 0 ? (row.cap / base) * 100 : null,
      widthPercent: (row.cap / scale) * 100,
    })),
    allocated,
    unallocated,
    unallocatedPercent: base != null && base > 0 ? (unallocated / base) * 100 : null,
    unallocatedWidthPercent: (unallocated / scale) * 100,
    overCommit: base != null && allocated > base,
    base,
  };
}

export type TradeQualityRow = {
  strategyId: string;
  label: string;
  winRate: number | null;
  winPercent: number | null;
  turnoverRatio: number | null;
  fills: number;
  completed: number;
  fee: number;
};

export type TradeQualityChart = {
  rows: TradeQualityRow[];
  /** Fills that have not yet closed a round trip, so win rate stays blank. */
  pendingFillCount: number;
};

export function tradeQualityChart(
  allocations: readonly StrategyAllocation[],
  resolveName: (allocation: StrategyAllocation) => string,
): TradeQualityChart {
  const rows = allocations
    .filter((allocation) => allocation.completed_trade_count > 0)
    .map((allocation) => ({
      strategyId: allocation.strategy_id,
      label: resolveName(allocation),
      winRate: allocation.win_rate,
      winPercent: allocation.win_rate == null ? null : allocation.win_rate * 100,
      turnoverRatio: allocation.turnover_ratio,
      fills: allocation.fill_count,
      completed: allocation.completed_trade_count,
      fee: allocation.fee_quote,
    }));
  const pendingFillCount = allocations.reduce(
    (total, allocation) => total + Math.max(allocation.fill_count, 0),
    0,
  );
  return { rows, pendingFillCount };
}

export type ConcurrencyMatrixRow = {
  strategyId: string;
  name: string;
  kind: string;
  status: StrategyStatus;
  statusLabel: string;
  batchId: string | null;
  allocationState: AllocationState;
  reserved: number;
  occupied: number;
  released: number;
  utilizationPercent: number;
  realizedPnl: number | null;
  unrealizedPnl: number | null;
  netPnl: number | null;
  returnRatePercent: number | null;
  fillCount: number;
  completedTradeCount: number;
  winPercent: number | null;
  maxReserved: number | null;
  maxOccupied: number | null;
  lifecycleReason: string | null;
};

/**
 * 四策略并发运行矩阵 as data. Every strategy keeps its own row so the operator can
 * compare them side by side without treating the shared wallet as one strategy.
 */
export function concurrencyMatrixRows(
  allocations: readonly StrategyAllocation[],
  strategies: readonly NamedStrategy[] | undefined,
): ConcurrencyMatrixRow[] {
  return allocations.map((allocation) => ({
    strategyId: allocation.strategy_id,
    name: resolveAllocationName(allocation, strategies),
    kind: allocation.kind,
    status: allocation.status,
    statusLabel: strategyStatusText(allocation.status),
    batchId: allocation.current_batch_id,
    allocationState: allocation.allocation_state,
    reserved: allocation.reserved_quote,
    occupied: allocation.occupied_quote,
    released: allocation.released_quote,
    utilizationPercent: allocation.utilization_ratio * 100,
    realizedPnl: allocation.realized_pnl_quote,
    unrealizedPnl: allocation.unrealized_pnl_quote,
    netPnl: allocation.net_pnl_quote,
    returnRatePercent: allocation.return_rate_pct,
    fillCount: allocation.fill_count,
    completedTradeCount: allocation.completed_trade_count,
    winPercent: allocation.win_rate == null ? null : allocation.win_rate * 100,
    maxReserved: allocation.max_reserved_quote,
    maxOccupied: allocation.max_occupied_quote,
    lifecycleReason: allocation.lifecycle_reason ?? null,
  }));
}

/**
 * The wallet curve is already a persisted fact, so it is passed through as chart
 * points rather than re-derived. An unavailable curve stays empty on purpose.
 */
export function walletEquityCurvePoints(
  summary: AccountStrategyOverview | undefined,
): Array<{
  ts: string;
  cumulative_pnl: number;
  daily_pnl_quote: number;
  equity_quote: number;
  action: string;
}> {
  const curve = summary?.wallet_equity_curve;
  if (!curve || curve.status !== "available") return [];
  return curve.points.map((point) => ({
    ts: point.ts,
    cumulative_pnl: point.cumulative_pnl,
    daily_pnl_quote: point.daily_pnl_quote,
    equity_quote: point.equity_quote,
    action: point.action,
  }));
}

/** Account utilization denominator description, matching the Web wording. */
export function accountUtilizationDescription(
  summary: AccountStrategyOverview | undefined,
): string {
  if (!summary) return "等待共享账户快照";
  const denominator = summary.allocator.utilization_denominator_quote;
  return `已占用名义 ÷ 可用于策略的权益（${formatUsdt(denominator)}）`;
}
