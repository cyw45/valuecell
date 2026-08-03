import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  CircleSlash,
  ShieldCheck,
  Target,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type {
  RuleConditionState,
  RuleStrategyAction,
  RuleStrategyCondition,
  RuleStrategyEvaluationHistoryEntry,
} from "@/types/rule-strategy";

type ExecutionStageState =
  | "passed"
  | "triggered"
  | "not_triggered"
  | "blocked"
  | "unavailable"
  | "not_recorded";

type ExecutionStage = {
  name: string;
  state: ExecutionStageState;
  summary: string;
};

const numberFormatter = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 4,
});

const CONDITION_CODE_LABELS: Record<string, string> = {
  price_ma: "价格与 MA",
  macd_cross: "MACD 交叉",
  bollinger_price: "价格与布林带",
  rsi_entry: "RSI 入场",
  rsi_exit: "RSI 出场",
  momentum_entry: "动量入场",
  momentum_exit: "动量出场",
  brar_entry: "BRAR 入场",
  brar_exit: "BRAR 出场",
  ma_crossover: "MA 交叉",
  rsi: "RSI",
  bollinger: "布林带",
  momentum_macd: "动量与 MACD",
  take_profit: "止盈",
  stop_loss: "止损",
  max_positions: "持仓数量上限",
  available_collateral: "可用计价余额",
  leverage_limit: "杠杆额度上限",
};

const CONDITION_VALUE_LABELS: Record<string, string> = {
  required_candles: "所需 K 线",
  supplied_candles: "已提供 K 线",
  price: "价格",
  interval: "周期",
  moving_average_short: "短期 MA",
  moving_average_long: "长期 MA",
  previous_moving_average_short: "前序短期 MA",
  previous_moving_average_long: "前序长期 MA",
  rsi: "RSI",
  bollinger_upper: "布林带上轨",
  bollinger_middle: "布林带中轨",
  bollinger_lower: "布林带下轨",
  momentum: "动量",
  macd: "MACD",
  macd_signal: "MACD 信号线",
  previous_macd: "前序 MACD",
  previous_macd_signal: "前序 MACD 信号线",
  return_pct: "当前收益率",
  threshold_pct: "阈值",
  open_position_count: "当前持仓数",
  max_positions: "持仓上限",
  requested_quote: "请求额度",
  affordable_quote: "可用额度",
  max_allowed_quote: "杠杆额度上限",
};

const REASON_LABELS: Record<string, string> = {
  advanced_exit_confirmed: "建议卖出：已确认配置的多周期出场规则。",
  advanced_entry_confirmed: "建议买入：已确认配置的多周期入场规则。",
  advanced_entry_not_confirmed: "不执行操作：未确认配置的多周期入场规则。",
  take_profit_triggered: "建议卖出：已触及止盈阈值。",
  stop_loss_triggered: "建议卖出：已触及止损阈值。",
  indicator_sell_confirmed: "建议卖出：配置指标已确认卖出信号。",
  indicator_buy_confirmed: "建议买入：配置指标已确认买入信号。",
  indicators_not_confirmed: "不执行操作：未确认配置的入场信号。",
  no_exit_signal: "不执行操作：未确认配置的出场信号。",
  no_enabled_indicators: "不执行操作：未启用任何指标规则。",
  insufficient_candle_history: "不执行操作：提供的 K 线历史不足，无法评估配置指标。",
  sell_signal_without_position: "不执行操作：卖出信号不能开启模拟多头仓位。",
  max_positions: "不执行操作：已达到最大持仓数量。",
  available_collateral: "不执行操作：计价币余额不足以覆盖配置仓位。",
  leverage_limit: "不执行操作：配置仓位超过基于权益的杠杆上限。",
};

const statePresentation: Record<
  ExecutionStageState,
  { label: string; className: string; icon: typeof CheckCircle2 }
> = {
  passed: {
    label: "已通过",
    className:
      "border-emerald-500/25 bg-emerald-500/8 text-emerald-700 dark:text-emerald-300",
    icon: CheckCircle2,
  },
  triggered: {
    label: "已触发",
    className:
      "border-emerald-500/25 bg-emerald-500/8 text-emerald-700 dark:text-emerald-300",
    icon: CheckCircle2,
  },
  not_triggered: {
    label: "未触发",
    className:
      "border-slate-400/25 bg-slate-400/8 text-slate-700 dark:text-slate-300",
    icon: CircleSlash,
  },
  blocked: {
    label: "已阻止",
    className:
      "border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-200",
    icon: ShieldCheck,
  },
  unavailable: {
    label: "不可用",
    className:
      "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300",
    icon: AlertTriangle,
  },
  not_recorded: {
    label: "未记录",
    className:
      "border-border bg-muted/40 text-muted-foreground",
    icon: CircleSlash,
  },
};

function containsChinese(value: string) {
  return /[\u3400-\u9fff]/.test(value);
}


function stageFromConditions(
  conditions: RuleStrategyCondition[],
  recordedSummary: string,
  noRecordSummary: string,
): Pick<ExecutionStage, "state" | "summary"> {
  if (conditions.length === 0) {
    return { state: "not_recorded", summary: noRecordSummary };
  }
  if (conditions.some((condition) => condition.state === "blocked")) {
    return { state: "blocked", summary: "本轮存在已阻止的条件。" };
  }
  if (conditions.some((condition) => condition.state === "unavailable")) {
    return { state: "unavailable", summary: "本轮存在不可用的条件。" };
  }
  if (conditions.every((condition) => condition.state === "triggered")) {
    return { state: "triggered", summary: recordedSummary };
  }
  if (conditions.some((condition) => condition.state === "not_triggered")) {
    return { state: "not_triggered", summary: "本轮存在未触发的条件。" };
  }
  return { state: "passed", summary: recordedSummary };
}

function confirmationStage(
  evaluation: RuleStrategyEvaluationHistoryEntry,
  indicatorConditions: RuleStrategyCondition[],
): Pick<ExecutionStage, "state" | "summary"> {
  const reasonCode = evaluation.reason_code;
  if (reasonCode === "no_enabled_indicators") {
    return { state: "not_recorded", summary: "未启用指标，服务器未产生确认结果。" };
  }
  if (
    reasonCode === "insufficient_candle_history" ||
    indicatorConditions.some((condition) => condition.state === "unavailable")
  ) {
    return { state: "unavailable", summary: "指标数据不足，无法完成确认。" };
  }
  if (
    reasonCode === "take_profit_triggered" ||
    reasonCode === "stop_loss_triggered"
  ) {
    return { state: "passed", summary: "退出规则直接形成决策，无需指标确认。" };
  }
  if (reasonCode.endsWith("_confirmed")) {
    return { state: "triggered", summary: "服务器已确认本轮策略信号。" };
  }
  if (
    [
      "advanced_entry_not_confirmed",
      "indicators_not_confirmed",
      "no_exit_signal",
      "sell_signal_without_position",
    ].includes(reasonCode)
  ) {
    return { state: "not_triggered", summary: "配置的确认条件未满足。" };
  }
  if (
    ["max_positions", "available_collateral", "leverage_limit"].includes(
      reasonCode,
    )
  ) {
    return { state: "not_recorded", summary: "服务器未单列确认结果，已进入风控门禁。" };
  }
  return stageFromConditions(
    indicatorConditions,
    "服务器已记录指标确认结果。",
    "服务器未记录确认条件。",
  );
}

function deriveExecutionStages(
  evaluation: RuleStrategyEvaluationHistoryEntry,
): ExecutionStage[] {
  const conditions = evaluation.conditions ?? [];
  const indicatorConditions = conditions.filter(
    (condition) => condition.category === "indicator",
  );
  const riskConditions = conditions.filter(
    (condition) => condition.category === "risk",
  );
  const hasUnavailable = conditions.some(
    (condition) => condition.state === "unavailable",
  );
  const hasRiskBlock = riskConditions.some(
    (condition) => condition.state === "blocked",
  );
  const dataStage: Pick<ExecutionStage, "state" | "summary"> =
    conditions.length === 0
      ? { state: "not_recorded", summary: "本轮评估未记录条件数据。" }
      : hasUnavailable
        ? { state: "unavailable", summary: "本轮评估存在不可用的数据条件。" }
        : {
            state: "passed",
            summary: `服务器已记录 ${conditions.length} 项条件。`,
          };
  const indicatorStage = stageFromConditions(
    indicatorConditions,
    `已完成 ${indicatorConditions.length} 项指标检查。`,
    "服务器未记录指标检查。",
  );
  const confirmation = confirmationStage(evaluation, indicatorConditions);
  const riskStage: Pick<ExecutionStage, "state" | "summary"> =
    riskConditions.length === 0
      ? { state: "not_recorded", summary: "服务器未记录风控门禁。" }
      : hasRiskBlock
        ? { state: "blocked", summary: "风控门禁阻止了本轮动作。" }
        : riskConditions.some((condition) => condition.state === "unavailable")
          ? { state: "unavailable", summary: "风控门禁存在不可用条件。" }
          : {
              state: "passed",
              summary: `已通过 ${riskConditions.length} 项风控检查。`,
            };
  const decision: Pick<ExecutionStage, "state" | "summary"> =
    evaluation.action === "buy" || evaluation.action === "sell"
      ? {
          state: "triggered",
          summary: `服务器建议${ruleStrategyActionLabel(evaluation.action)}。`,
        }
      : hasRiskBlock
        ? { state: "blocked", summary: "风控已阻止执行，服务器不执行。" }
        : hasUnavailable
          ? { state: "unavailable", summary: "存在不可用条件，服务器不执行。" }
          : confirmation.state === "not_triggered"
            ? { state: "not_triggered", summary: "确认条件未触发，服务器不执行。" }
            : {
                state: "not_recorded",
                summary: "本轮未形成执行动作。",
              };

  return [
    { name: "数据", ...dataStage },
    { name: "指标检查", ...indicatorStage },
    { name: "确认", ...confirmation },
    { name: "风控门禁", ...riskStage },
    { name: "决策", ...decision },
  ];
}

function formatConditionValue(key: string, value: string | number | boolean) {
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "string") {
    if (value === "above") return "高于或等于";
    if (value === "below") return "低于或等于";
    return key === "interval" ? value : "已记录";
  }
  if (key.endsWith("_pct")) {
    return `${numberFormatter.format(value * 100)}%`;
  }
  if (key.endsWith("_quote")) return `${numberFormatter.format(value)} USDT`;
  return numberFormatter.format(value);
}

function fallbackConditionDetail(condition: RuleStrategyCondition) {
  if (condition.state === "unavailable") {
    const required = condition.values.required_candles;
    const supplied = condition.values.supplied_candles;
    if (typeof required === "number" && typeof supplied === "number") {
      return `K 线历史不足：需要 ${required} 根，已提供 ${supplied} 根。`;
    }
    return "K 线历史不足，该条件暂不可用。";
  }
  if (condition.code === "take_profit") {
    return condition.state === "triggered" ? "已触及止盈阈值。" : "未触及止盈阈值。";
  }
  if (condition.code === "stop_loss") {
    return condition.state === "triggered" ? "已触及止损阈值。" : "未触及止损阈值。";
  }
  if (condition.code === "max_positions") {
    return condition.state === "blocked"
      ? "已达到最大持仓数量。"
      : "当前持仓数量未达到上限，允许入场。";
  }
  if (condition.code === "available_collateral") {
    return condition.state === "blocked"
      ? "计价币余额不足以覆盖配置仓位。"
      : "计价币余额足以覆盖配置仓位。";
  }
  if (condition.code === "leverage_limit") {
    return condition.state === "blocked"
      ? "配置仓位超过基于权益的杠杆上限。"
      : "配置仓位在基于权益的杠杆上限内。";
  }
  return `${ruleStrategyConditionLabel(condition.code)}已由服务器判定为${ruleStrategyConditionStateLabel(condition.state)}。`;
}

function conditionDetail(condition: RuleStrategyCondition) {
  return containsChinese(condition.detail)
    ? condition.detail
    : fallbackConditionDetail(condition);
}

export function ruleStrategyEvaluationReason(
  evaluation: RuleStrategyEvaluationHistoryEntry,
) {
  if (containsChinese(evaluation.reason)) return evaluation.reason;
  return (
    REASON_LABELS[evaluation.reason_code] ??
    (evaluation.action === "buy"
      ? "服务器已记录买入建议。"
      : evaluation.action === "sell"
        ? "服务器已记录卖出建议。"
        : "服务器已记录本轮不执行决策。")
  );
}

function evaluationTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "服务器已记录";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

export function ruleStrategyActionLabel(action: RuleStrategyAction) {
  if (action === "buy") return "买入";
  if (action === "sell") return "卖出";
  return "不执行";
}

export function ruleStrategyActionTone(action: RuleStrategyAction) {
  if (action === "buy") return "text-emerald-500";
  if (action === "sell") return "text-rose-500";
  return "text-muted-foreground";
}

export function ruleStrategyConditionStateLabel(state: RuleConditionState) {
  if (state === "triggered") return "已触发";
  if (state === "not_triggered") return "未触发";
  if (state === "blocked") return "已阻止";
  return "不可用";
}

export function ruleStrategyConditionCategoryLabel(
  category: RuleStrategyCondition["category"],
) {
  if (category === "indicator") return "指标条件";
  if (category === "exit") return "退出条件";
  return "风控条件";
}

export function ruleStrategyConditionLabel(code: string) {
  return CONDITION_CODE_LABELS[code] ?? "策略条件";
}

export function RuleStrategyEvaluationPath({
  evaluation,
}: {
  evaluation?: RuleStrategyEvaluationHistoryEntry;
}) {
  if (!evaluation) {
    return (
      <Card className="dashboard-panel rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
        <div className="flex items-center gap-3 border-border/70 border-b px-4 py-3">
          <span className="grid size-9 place-items-center rounded-md bg-sky-500/10 text-sky-500">
            <Target className="size-4" />
          </span>
          <div>
            <h2 className="font-semibold">当前评估条件</h2>
            <p className="mt-0.5 text-muted-foreground text-xs">
              仅展示服务器记录的最近一次评估，不推演或补造执行状态。
            </p>
          </div>
        </div>
        <CardContent className="grid min-h-40 place-items-center px-4 py-8 text-center">
          <div>
            <BarChart3 className="mx-auto size-6 text-muted-foreground/60" />
            <p className="mt-3 text-muted-foreground text-sm">
              等待最近一次实际评估。
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const conditions = evaluation.conditions ?? [];
  const stages = deriveExecutionStages(evaluation);

  return (
    <Card
      className="dashboard-panel overflow-hidden rounded-lg border-white/10 bg-card/90 py-0 shadow-none"
      data-evaluation-id={evaluation.evaluation_id}
      data-reason-code={evaluation.reason_code}
    >
      <div className="flex flex-col gap-3 border-border/70 border-b px-4 py-3 md:flex-row md:items-start md:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <span className="grid size-9 shrink-0 place-items-center rounded-md bg-sky-500/10 text-sky-500">
            <Target className="size-4" />
          </span>
          <div className="min-w-0">
            <h2 className="font-semibold">当前评估条件</h2>
            <p className="mt-0.5 text-muted-foreground text-xs">
              仅展示服务器记录的最近一次评估，不推演或补造执行状态。
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 md:justify-end">
          {evaluation.symbol ? (
            <Badge variant="outline">{evaluation.symbol.replace("-", "/")}</Badge>
          ) : null}
          <Badge
            className={cn(
              "border",
              evaluation.action === "buy"
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                : evaluation.action === "sell"
                  ? "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300"
                  : "border-border bg-muted/50 text-muted-foreground",
            )}
            data-action={evaluation.action}
            variant="outline"
          >
            {ruleStrategyActionLabel(evaluation.action)}
          </Badge>
          <span className="text-muted-foreground text-xs">
            {evaluationTime(evaluation.evaluated_at)}
          </span>
        </div>
      </div>
      <CardContent className="space-y-4 p-4">
        <section aria-label="本轮执行漏斗">
          <div className="mb-2 flex items-center gap-2">
            <h3 className="font-medium text-sm">本轮执行路径</h3>
            <span className="text-muted-foreground text-xs">
              数据 → 指标检查 → 确认 → 风控门禁 → 决策
            </span>
          </div>
          <ol className="grid gap-2 md:grid-cols-[repeat(9,minmax(0,1fr))]">
            {stages.map((stage, index) => {
              const presentation = statePresentation[stage.state];
              const StageIcon = presentation.icon;
              return (
                <li
                  className={cn(
                    "contents md:col-span-1",
                    index === stages.length - 1 && "md:col-span-1",
                  )}
                  key={stage.name}
                >
                  <div
                    className={cn(
                      "min-w-0 rounded-md border px-3 py-2.5",
                      presentation.className,
                    )}
                    data-stage-state={stage.state}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-xs">{stage.name}</span>
                      <StageIcon className="size-3.5 shrink-0" />
                    </div>
                    <p className="mt-1 font-semibold text-xs">
                      {presentation.label}
                    </p>
                    <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed opacity-80">
                      {stage.summary}
                    </p>
                  </div>
                  {index < stages.length - 1 ? (
                    <div
                      aria-hidden
                      className="hidden items-center justify-center text-muted-foreground md:col-span-1 md:flex"
                    >
                      <ArrowRight className="size-4" />
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ol>
        </section>

        <section
          className="rounded-md border border-sky-500/20 bg-sky-500/5 px-3 py-3"
          aria-label="本轮决策原因"
        >
          <p className="font-medium text-sky-800 text-xs dark:text-sky-200">
            本轮结论
          </p>
          <p className="mt-1 text-sm leading-relaxed">
            {ruleStrategyEvaluationReason(evaluation)}
          </p>
        </section>

        <section aria-label="本轮条件明细">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-medium text-sm">条件明细</h3>
            <span className="text-muted-foreground text-xs">
              {conditions.length} 项服务器判定
            </span>
          </div>
          {conditions.length === 0 ? (
            <p className="rounded-md border border-dashed px-3 py-4 text-center text-muted-foreground text-sm">
              服务器未记录本轮条件明细。
            </p>
          ) : (
            <div className="grid gap-2 lg:grid-cols-2">
              {conditions.map((condition, index) => {
                const presentation = statePresentation[condition.state];
                const ConditionIcon = presentation.icon;
                const values = Object.entries(condition.values).filter(
                  (
                    entry,
                  ): entry is [string, string | number | boolean] =>
                    entry[1] !== null && entry[1] !== undefined,
                );
                return (
                  <article
                    aria-label={`${ruleStrategyConditionCategoryLabel(condition.category)}：${ruleStrategyConditionLabel(condition.code)}，${ruleStrategyConditionStateLabel(condition.state)}`}
                    className={cn(
                      "rounded-md border p-3",
                      presentation.className,
                    )}
                    data-condition-category={condition.category}
                    data-condition-code={condition.code}
                    data-condition-state={condition.state}
                    key={`${condition.code}-${index}`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <ConditionIcon className="size-4 shrink-0" />
                        <span className="truncate font-medium text-sm">
                          {ruleStrategyConditionLabel(condition.code)}
                        </span>
                      </div>
                      <span className="flex shrink-0 items-center gap-1 text-xs">
                        <span>{ruleStrategyConditionCategoryLabel(condition.category)}</span>
                        <span aria-hidden>·</span>
                        <span className="font-medium">
                          {ruleStrategyConditionStateLabel(condition.state)}
                        </span>
                      </span>
                    </div>
                    <p className="mt-2 text-xs leading-relaxed opacity-90">
                      {conditionDetail(condition)}
                    </p>
                    {values.length > 0 ? (
                      <dl className="mt-2 flex flex-wrap gap-x-3 gap-y-1 border-white/20 border-t pt-2 text-[11px]">
                        {values.map(([key, value]) => (
                          <div className="flex gap-1" key={key}>
                            <dt className="opacity-70">
                              {CONDITION_VALUE_LABELS[key] ?? "指标值"}
                            </dt>
                            <dd className="font-medium">
                              {formatConditionValue(key, value)}
                            </dd>
                          </div>
                        ))}
                      </dl>
                    ) : null}
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </CardContent>
    </Card>
  );
}
