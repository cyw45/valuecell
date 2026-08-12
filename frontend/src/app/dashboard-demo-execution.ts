import type {
  DemoEquityCurve,
  DemoPurchaseState,
  RuleStrategyDemoExecutionPnl,
} from "@/types/rule-strategy-demo-execution";

const numberFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function formatOptionalAmount(value?: string | number | null): string {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numberFormatter.format(numeric) : "—";
}

export function demoPurchaseStatePresentation(state?: DemoPurchaseState | null) {
  const presentations = {
    bought: { label: "已买入", tone: "positive" },
    not_bought: { label: "尚未买入", tone: "neutral" },
    pending: { label: "待确认", tone: "warning" },
    partially_filled: { label: "待确认（部分成交）", tone: "warning" },
    failed: { label: "失败", tone: "negative" },
    unknown: { label: "待确认", tone: "warning" },
  } as const;
  return presentations[state ?? "unknown"] ?? presentations.unknown;
}

const ORDER_STATUS_LABELS: Record<string, string> = {
  pending: "待提交",
  submitted: "已提交",
  open: "挂单中",
  live: "挂单中",
  partially_filled: "部分成交",
  partial: "部分成交",
  filled: "已成交",
  submission_unknown: "待远端对账",
  canceled: "已取消",
  cancelled: "已取消",
  failed: "失败",
  rejected: "已拒绝",
};

export function demoOrderStatusLabel(status?: string | null): string {
  if (!status) return "未知状态";
  return ORDER_STATUS_LABELS[status.toLowerCase()] ?? `未知状态（${status}）`;
}

const PNL_REASONS: Record<string, { reason: string; recovery: string }> = {
  missing_fill_history: {
    reason: "缺少成交历史，无法可靠计算盈亏。",
    recovery: "同步完整订单成交明细后自动恢复。",
  },
  shared_account_without_strategy_cost_basis: {
    reason: "共享账户缺少可归属的成本基础。",
    recovery: "订单成交可完整归属并建立成本基础后自动恢复。",
  },
  no_filled_orders: {
    reason: "尚无已成交订单，暂无可计算的交易盈亏。",
    recovery: "首笔订单成交并完成估值后自动恢复。",
  },
  incomplete_valuation: {
    reason: "部分资产尚未完成估值。",
    recovery: "全部相关资产取得有效行情后自动恢复。",
  },
  legacy_fill_metadata_unavailable: {
    reason: "存在历史成交记录，但缺少成交数量、均价或成本明细。",
    recovery: "完成历史订单成交明细回填后自动恢复。",
  },
  insufficient_strategy_fill_history: {
    reason: "策略成交历史不完整，无法可靠还原成本基础。",
    recovery: "同步完整策略成交历史后自动恢复。",
  },
  incomplete_fill_metadata: {
    reason: "部分成交记录缺少数量、均价或成本明细。",
    recovery: "成交明细同步完整后自动恢复。",
  },
  mark_price_unavailable: {
    reason: "当前持仓缺少有效行情，无法计算未实现盈亏。",
    recovery: "相关币种取得有效行情后自动恢复。",
  },
  strategy_equity_history_unavailable: {
    reason: "尚未保存可审计的策略历史估值快照，无法绘制收益曲线。",
    recovery: "后端开始持久化策略估值快照后自动显示。",
  },
};

export function demoPnlPresentation(pnl?: RuleStrategyDemoExecutionPnl | null) {
  const status = pnl?.status ?? "unavailable";
  if (status === "available") {
    const total = pnl?.total_pnl !== undefined
      ? pnl.total_pnl
      : pnl?.total !== undefined
        ? pnl.total
        : pnl?.value ?? null;
    const realized = formatOptionalAmount(pnl?.realized_pnl ?? pnl?.realized);
    const unrealized = formatOptionalAmount(pnl?.unrealized_pnl ?? pnl?.unrealized);
    const returnLabel =
      pnl?.return_pct == null
        ? "—"
        : `${(Number(pnl.return_pct) * 100).toFixed(2)}%`;
    return {
      available: true,
      partial: false,
      totalPnl: total == null ? null : Number(total),
      detail: `已实现 ${realized} · 未实现 ${unrealized} · 收益率 ${returnLabel} · 未含手续费`,
    };
  }
  const localized = pnl?.reason_code ? PNL_REASONS[pnl.reason_code] : undefined;
  const chineseReason =
    pnl?.reason && /[\u4e00-\u9fff]/.test(pnl.reason) ? pnl.reason : undefined;
  return {
    available: false,
    partial: false,
    totalPnl: null,
    detail: `${localized?.reason ?? chineseReason ?? "当前数据不足，无法可靠计算盈亏。"} 恢复条件：${localized?.recovery ?? "订单成交、成本基础与资产估值数据完整后自动恢复。"}`,
  };
}

export function buildDemoEquityCurve(curve?: DemoEquityCurve | null) {
  if (!Array.isArray(curve?.points)) return [];
  return curve.points.flatMap((point) => {
    const ts = point.ts ?? point.timestamp;
    const pnl = point.cumulative_pnl ?? point.total_pnl ?? point.pnl ?? point.value;
    if (!ts || !Number.isFinite(Date.parse(ts)) || pnl == null || !Number.isFinite(Number(pnl))) return [];
    return [{
      ts,
      cumulative_pnl: Number(pnl),
      action: "mark_to_market" as const,
    }];
  });
}

export function formatDemoTime(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("zh-CN");
}
