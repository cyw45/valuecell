import type {
  RuleStrategyAction,
  RuleStrategyCondition,
  RuleStrategyConditionState,
  RuleStrategyEvaluation,
  RuleStrategyPaperPosition,
  RuleStrategyStatus,
  SandboxOrder,
} from "../types";
import { formatQuote } from "./workbench";

export type StrategyPresentationTone = "default" | "positive" | "negative" | "warning";

const HAN_CHARACTER = /[\u3400-\u9fff]/;

const CONDITION_LABELS: Readonly<Record<string, string>> = {
  available_collateral: "可用保证金",
  bollinger: "布林带",
  bollinger_price: "布林带价格",
  brar_entry: "BRAR 入场",
  brar_exit: "BRAR 退出",
  leverage_limit: "杠杆额度",
  ma_crossover: "MA 交叉",
  macd_cross: "MACD 交叉",
  max_positions: "最大持仓数",
  momentum_entry: "动量入场",
  momentum_exit: "动量退出",
  momentum_macd: "动量 / MACD",
  price_ma: "价格与 MA",
  rsi: "RSI",
  rsi_entry: "RSI 入场",
  rsi_exit: "RSI 退出",
  stop_loss: "止损",
  take_profit: "止盈",
};

const REASON_LABELS: Readonly<Record<string, string>> = {
  advanced_entry_confirmed: "已确认多周期入场规则，生成买入建议。",
  advanced_entry_not_confirmed: "多周期入场规则尚未确认，本次暂不操作。",
  advanced_exit_confirmed: "已确认多周期退出规则，生成卖出建议。",
  available_collateral: "可用 USDT 不足，已阻止新增仓位。",
  indicator_buy_confirmed: "配置的指标已确认买入信号。",
  indicator_sell_confirmed: "配置的指标已确认卖出信号。",
  indicators_not_confirmed: "配置的指标尚未确认入场信号，本次暂不操作。",
  insufficient_candle_history: "提供的 K 线历史不足，无法完成所需指标计算。",
  leverage_limit: "计划金额超过基于权益计算的杠杆上限，已阻止新增仓位。",
  max_positions: "已达到最大开放持仓数，已阻止新增仓位。",
  no_enabled_indicators: "没有启用指标规则，本次暂不操作。",
  no_exit_signal: "没有已确认的退出信号，本次暂不操作。",
  sell_signal_without_position: "当前没有持仓，卖出信号不能开立纸面多头仓位。",
  stop_loss_triggered: "已触发止损阈值，生成卖出建议。",
  take_profit_triggered: "已触发止盈阈值，生成卖出建议。",
};

const CONDITION_VALUE_LABELS: Readonly<Record<string, string>> = {
  affordable_quote: "可负担额度",
  bollinger_lower: "布林下轨",
  bollinger_middle: "布林中轨",
  bollinger_upper: "布林上轨",
  brar_ar: "BRAR AR",
  brar_br: "BRAR BR",
  interval: "周期",
  macd: "MACD",
  macd_signal: "MACD 信号线",
  max_allowed_quote: "最大允许额度",
  max_positions: "最大持仓数",
  moving_average_long: "长期 MA",
  moving_average_short: "短期 MA",
  momentum: "动量",
  open_position_count: "当前持仓数",
  previous_macd: "前一 MACD",
  previous_macd_signal: "前一 MACD 信号线",
  previous_moving_average_long: "前一长期 MA",
  previous_moving_average_short: "前一短期 MA",
  price: "价格",
  requested_quote: "计划额度",
  required_candles: "需要 K 线数",
  return_pct: "当前收益率",
  rsi: "RSI",
  supplied_candles: "已提供 K 线数",
  threshold_pct: "阈值",
};

function hasChineseText(value: string | null | undefined): value is string {
  return typeof value === "string" && HAN_CHARACTER.test(value);
}

function formatNumber(value: number): string {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: Number.isInteger(value) ? 0 : 6,
  });
}

function formatConditionValue(key: string, value: string | number | boolean | null): string {
  if (value == null) return "—";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "string") return key === "interval" ? `${value} 周期` : value;
  if (key.endsWith("_quote")) return formatQuote(value);
  if (key === "return_pct" || key === "threshold_pct") {
    return `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 4 })}%`;
  }
  return formatNumber(value);
}

function fallbackConditionDetail(condition: RuleStrategyCondition): string {
  const values = condition.values;
  if (condition.state === "unavailable") {
    const required = values.required_candles;
    const supplied = values.supplied_candles;
    if (typeof required === "number" && typeof supplied === "number") {
      return `K 线历史不足：需要 ${formatNumber(required)} 根，已提供 ${formatNumber(supplied)} 根。`;
    }
    return "所需市场数据暂不可用，无法完成该条件计算。";
  }

  switch (condition.code) {
    case "max_positions":
      return condition.state === "blocked"
        ? "当前开放持仓数已达到配置上限，禁止新增仓位。"
        : "当前开放持仓数仍在配置上限内。";
    case "available_collateral":
      return condition.state === "blocked"
        ? "可用 USDT 无法覆盖配置的计划额度，禁止新增仓位。"
        : "可用 USDT 可以覆盖配置的计划额度。";
    case "leverage_limit":
      return condition.state === "blocked"
        ? "配置的计划额度超过权益对应的杠杆上限，禁止新增仓位。"
        : "配置的计划额度处于权益对应的杠杆上限内。";
    case "take_profit":
      return condition.state === "triggered" ? "已达到止盈阈值。" : "尚未达到止盈阈值。";
    case "stop_loss":
      return condition.state === "triggered" ? "已达到止损阈值。" : "尚未达到止损阈值。";
    case "ma_crossover":
      return condition.state === "triggered" ? "MA 交叉条件已触发。" : "MA 交叉条件尚未触发。";
    case "rsi":
    case "rsi_entry":
    case "rsi_exit":
      return condition.state === "triggered" ? "RSI 条件已触发。" : "RSI 条件尚未触发。";
    case "macd_cross":
      return condition.state === "triggered" ? "MACD 交叉条件已触发。" : "MACD 交叉条件尚未触发。";
    case "bollinger":
    case "bollinger_price":
      return condition.state === "triggered" ? "布林带条件已触发。" : "布林带条件尚未触发。";
    case "momentum_macd":
      return condition.state === "triggered" ? "动量与 MACD 条件已触发。" : "动量与 MACD 条件尚未触发。";
    default:
      return `服务端条件“${condition.code}”当前为${conditionStateLabel(condition.state)}。`;
  }
}

export function strategyStatusLabel(status: RuleStrategyStatus, archivedAt?: string | null): string {
  if (archivedAt) return "已归档";
  return status === "running" ? "运行中" : "已停止";
}

export function strategyStatusTone(status: RuleStrategyStatus, archivedAt?: string | null): StrategyPresentationTone {
  if (archivedAt) return "default";
  return status === "running" ? "positive" : "warning";
}

export function executionEnvironmentLabel(environment: "paper" | "okx_demo"): string {
  return environment === "okx_demo" ? "OKX Demo" : "纸面交易";
}

export function strategyActionLabel(action: RuleStrategyAction): string {
  switch (action) {
    case "buy":
      return "买入建议";
    case "sell":
      return "卖出建议";
    default:
      return "暂不操作";
  }
}

export function strategyActionTone(action: RuleStrategyAction): StrategyPresentationTone {
  if (action === "buy") return "positive";
  if (action === "sell") return "warning";
  return "default";
}

export function conditionStateLabel(state: RuleStrategyConditionState): string {
  switch (state) {
    case "triggered":
      return "已触发";
    case "not_triggered":
      return "未触发";
    case "blocked":
      return "已阻止";
    case "unavailable":
      return "不可用";
  }
}

export function conditionStateTone(state: RuleStrategyConditionState): StrategyPresentationTone {
  switch (state) {
    case "triggered":
      return "positive";
    case "blocked":
      return "warning";
    case "unavailable":
      return "negative";
    default:
      return "default";
  }
}

export function conditionLabel(code: string): string {
  return CONDITION_LABELS[code] ?? `策略条件 · ${code}`;
}

export function conditionCategoryLabel(category: RuleStrategyCondition["category"]): string {
  return category === "indicator" ? "指标" : category === "exit" ? "退出" : "风险";
}

export function conditionDetail(condition: RuleStrategyCondition): string {
  return hasChineseText(condition.detail) ? condition.detail : fallbackConditionDetail(condition);
}

export function conditionFacts(condition: RuleStrategyCondition): Array<{ label: string; value: string }> {
  return Object.entries(condition.values).map(([key, value]) => ({
    label: CONDITION_VALUE_LABELS[key] ?? `参数 ${key}`,
    value: formatConditionValue(key, value),
  }));
}

export function evaluationReason(reasonCode: string, reason?: string | null): string {
  if (hasChineseText(reason)) return reason;
  return REASON_LABELS[reasonCode] ?? `服务端给出的决策原因代码为“${reasonCode}”。`;
}

export function conditionStateCounts(conditions: readonly RuleStrategyCondition[]): Record<RuleStrategyConditionState, number> {
  return conditions.reduce<Record<RuleStrategyConditionState, number>>(
    (counts, condition) => ({ ...counts, [condition.state]: counts[condition.state] + 1 }),
    { triggered: 0, not_triggered: 0, blocked: 0, unavailable: 0 },
  );
}

export function conditionStateSummary(conditions: readonly RuleStrategyCondition[]): string {
  const counts = conditionStateCounts(conditions);
  return [
    `已触发 ${counts.triggered}`,
    `未触发 ${counts.not_triggered}`,
    `已阻止 ${counts.blocked}`,
    `不可用 ${counts.unavailable}`,
  ].join(" · ");
}

export function primaryConditionState(conditions: readonly RuleStrategyCondition[]): RuleStrategyConditionState | null {
  if (conditions.length === 0) return null;
  return ["blocked", "unavailable", "triggered", "not_triggered"].find((state) =>
    conditions.some((condition) => condition.state === state),
  ) as RuleStrategyConditionState;
}

export function fundingDirectionLabel(direction: RuleStrategyEvaluation["funding"]["direction"]): string {
  return direction === "credit" ? "预计收取" : direction === "debit" ? "预计支付" : "无资金费影响";
}

export type ExecutionFunnelFact = {
  label: string;
  value: string;
  caption: string;
  tone: StrategyPresentationTone;
};

export function executionFunnelFacts(evaluation: RuleStrategyEvaluation): ExecutionFunnelFact[] {
  if (!evaluation.funnel?.length) return evaluationExecutionFunnel(evaluation);
  return evaluation.funnel.map((stage) => ({
    label: stage.label,
    value: stage.status === "passed" ? "通过" : stage.status === "filled" ? "已成交" : stage.status === "partial" ? "部分成交" : stage.status === "blocked" ? "阻塞" : stage.status === "rejected" ? "已拒绝" : "等待",
    caption: stage.detail,
    tone: stage.status === "passed" || stage.status === "filled" ? "positive" : stage.status === "blocked" || stage.status === "rejected" ? "negative" : stage.status === "partial" ? "warning" : "default",
  }));
}

export function evaluationExecutionFunnel(evaluation: RuleStrategyEvaluation): ExecutionFunnelFact[] {
  const primaryState = primaryConditionState(evaluation.conditions);
  const conditions = conditionStateSummary(evaluation.conditions);
  const execution = evaluation.paper_fill === true
    ? "纸面账本已记录模拟成交"
    : evaluation.execution_ledger === "external"
      ? "外部执行账本"
      : evaluation.action === "no_op"
        ? "本次评估为暂不操作"
        : "服务端本次未提供成交账本标识";
  const executionCaption = evaluation.paper_fill === true
    ? "paper_fill: true"
    : evaluation.execution_ledger === "external"
      ? "execution_ledger: external"
      : `action: ${evaluation.action}`;

  return [
    {
      label: "策略决策",
      value: strategyActionLabel(evaluation.action),
      caption: `action: ${evaluation.action}`,
      tone: strategyActionTone(evaluation.action),
    },
    {
      label: "条件门槛",
      value: conditions,
      caption: `${evaluation.conditions.length} 项服务端条件`,
      tone: primaryState === "blocked"
        ? "warning"
        : primaryState === "unavailable"
          ? "negative"
          : "default",
    },
    {
      label: "计划额度",
      value: formatQuote(evaluation.sizing.requested_quote),
      caption: `最大允许 ${formatQuote(evaluation.sizing.max_allowed_quote)} · 可负担 ${formatQuote(evaluation.sizing.affordable_quote)}`,
      tone: "default",
    },
    {
      label: "资金费影响",
      value: formatQuote(evaluation.funding.estimated_payment_quote),
      caption: `${fundingDirectionLabel(evaluation.funding.direction)} · 预计名义金额 ${formatQuote(evaluation.funding.projected_notional_quote)}`,
      tone: evaluation.funding.direction === "credit" ? "positive" : evaluation.funding.direction === "debit" ? "warning" : "default",
    },
    {
      label: "执行账本",
      value: execution,
      caption: executionCaption,
      tone: evaluation.paper_fill === true ? "positive" : "default",
    },
  ];
}

export function paperPositionValue(positions: Record<string, RuleStrategyPaperPosition>): number {
  return Object.values(positions).reduce(
    (total, position) => total + position.quantity * position.mark_price,
    0,
  );
}

export function orderSideLabel(side: SandboxOrder["side"]): string {
  return side === "buy" ? "买入" : "卖出";
}

export function orderTypeLabel(type: SandboxOrder["type"]): string {
  return type === "market" ? "市价" : "限价";
}

export function orderStatusLabel(status: string): string {
  const labels: Readonly<Record<string, string>> = {
    canceled: "已撤销",
    cancelled: "已撤销",
    failed: "已失败",
    filled: "已成交",
    open: "已挂单",
    pending: "处理中",
    rejected: "已拒绝",
    stale: "状态过期",
    submitted: "已提交",
    submission_unknown: "提交状态待确认",
  };
  return labels[status] ?? `订单状态：${status}`;
}

export function demoSourceLabel(source: string): string {
  return source === "okx_demo_spot" ? "OKX Demo 现货" : `交易所执行来源：${source}`;
}

export function demoPnlReason(reason: string | null | undefined): string {
  if (hasChineseText(reason)) return reason;
  return "OKX Demo 交易所未提供可直接展示的已实现或未实现 PnL；移动端不会用纸面账本替代。";
}
