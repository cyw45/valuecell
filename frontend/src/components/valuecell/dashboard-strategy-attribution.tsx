import { Link } from "react-router";
import { Activity, ArrowUpRight, Clock, Target } from "lucide-react";
import {
  buildDashboardFunnel,
  conditionSatisfactionPercent,
  dashboardConditionSummary,
} from "@/app/dashboard-funnel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ThresholdGauge } from "@/components/valuecell/charts/threshold-gauge";
import { DashboardEvaluationRhythm } from "@/components/valuecell/dashboard-evaluation-rhythm";
import {
  RuleStrategyEvaluationPath,
  ruleStrategyActionLabel,
  ruleStrategyActionTone,
  ruleStrategyEvaluationReason,
} from "@/components/valuecell/rule-strategy-evaluation-path";
import { cn } from "@/lib/utils";
import type {
  RuleStrategyEvaluationHistoryEntry,
  RuleStrategyFunnelStatus,
} from "@/types/rule-strategy";

const FUNNEL_PRESENTATION: Record<
  RuleStrategyFunnelStatus,
  { label: string; chip: string; dot: string }
> = {
  passed: {
    label: "通过",
    chip: "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
    dot: "bg-emerald-500",
  },
  filled: {
    label: "已成交",
    chip: "border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-300",
    dot: "bg-sky-400",
  },
  partial: {
    label: "部分成交",
    chip: "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    dot: "bg-amber-500",
  },
  pending: {
    label: "等待",
    chip: "border-border bg-muted/40 text-muted-foreground",
    dot: "bg-muted-foreground/40",
  },
  blocked: {
    label: "受阻",
    chip: "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300",
    dot: "bg-rose-500",
  },
  rejected: {
    label: "被拒",
    chip: "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300",
    dot: "bg-rose-500",
  },
};

const scanTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

function scanTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "服务器已记录";
  return scanTimeFormatter.format(date);
}

/**
 * Answers the operator's two questions on the dashboard itself: which rule
 * stopped this strategy, and which numbers satisfied or failed each condition.
 * Everything shown here is the server's own journal for the selected strategy.
 */
export function DashboardStrategyAttribution({
  strategyId,
  strategyName,
  strategyRunning,
  evaluations,
  capitalUtilization,
  capitalUtilizationDescription,
  className,
}: {
  strategyId: string;
  strategyName: string;
  strategyRunning: boolean;
  evaluations: RuleStrategyEvaluationHistoryEntry[];
  capitalUtilization: number | null;
  capitalUtilizationDescription: string;
  className?: string;
}) {
  const latestEvaluation = evaluations[0];
  const { steps, firstBlocker } = buildDashboardFunnel({
    strategyRunning,
    evaluation: latestEvaluation,
  });
  const summary = dashboardConditionSummary(latestEvaluation);
  const satisfaction = conditionSatisfactionPercent(summary);
  const recentScans = evaluations.slice(0, 6);

  return (
    <>
      <section aria-label="策略执行归因" className={cn("dashboard-rise", className)}>
        <Card className="dashboard-panel overflow-hidden rounded-lg border-sky-500/20 bg-card/90 py-0 shadow-none">
          <CardHeader className="gap-1 border-border/70 border-b px-5 py-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Target className="size-4 text-sky-500" />
                  策略执行归因
                </CardTitle>
                <CardDescription>
                  {strategyName} · 本轮为什么成交或没有成交，逐项列出服务器判定的条件与实际数值
                </CardDescription>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  className={cn(
                    "shrink-0",
                    strategyRunning
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300"
                      : "border-border bg-muted/50 text-muted-foreground",
                  )}
                  variant="outline"
                >
                  {strategyRunning ? "调度器扫描中" : "策略未运行"}
                </Badge>
                <Button asChild size="sm" type="button" variant="outline">
                  <Link to={`/trades?strategy=${encodeURIComponent(strategyId)}`}>
                    查看完整交易明细
                    <ArrowUpRight />
                  </Link>
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 p-4">
            <section aria-label="本轮执行链路">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <h3 className="flex items-center gap-2 font-medium text-sm">
                  <Activity className="size-4 text-sky-500" />
                  本轮执行链路
                </h3>
                <span
                  className={cn(
                    "text-xs",
                    firstBlocker
                      ? "font-medium text-rose-600 dark:text-rose-300"
                      : "text-muted-foreground",
                  )}
                >
                  {firstBlocker ? `当前卡点：${firstBlocker}` : "六个环节全部通过"}
                </span>
              </div>
              <ol className="grid gap-2 sm:grid-cols-2 xl:grid-cols-6">
                {steps.map((step) => {
                  const presentation = FUNNEL_PRESENTATION[step.status];
                  return (
                    <li className="min-w-0" key={step.code}>
                      <div
                        className={cn(
                          "h-full rounded-md border px-3 py-2.5",
                          presentation.chip,
                        )}
                        data-funnel-code={step.code}
                        data-funnel-status={step.status}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate font-medium text-xs">
                            {step.label}
                          </span>
                          <span
                            aria-hidden
                            className={cn(
                              "size-2 shrink-0 rounded-full",
                              presentation.dot,
                            )}
                          />
                        </div>
                        <p className="mt-1 font-semibold text-xs">
                          {presentation.label}
                        </p>
                        <p className="mt-1 line-clamp-3 text-[11px] leading-relaxed opacity-80">
                          {step.detail}
                        </p>
                      </div>
                    </li>
                  );
                })}
              </ol>
            </section>

            <section
              aria-label="关键比例"
              className="grid gap-3 lg:grid-cols-2"
            >
              <ThresholdGauge
                description={
                  summary
                    ? `最近一次评估通过 ${summary.matched}/${summary.total} 项，要求 ${summary.required} 项（${summary.available} 项数据可用）`
                    : "服务器尚未记录本轮条件判定"
                }
                displayValue={
                  satisfaction === null
                    ? "—"
                    : `${Math.round(satisfaction)}%`
                }
                label="条件满足度"
                thresholds={[
                  "0 项",
                  summary ? `要求 ${summary.required} 项` : "无判定",
                  summary ? `共 ${summary.total} 项` : "无判定",
                ]}
                value={satisfaction}
              />
              <ThresholdGauge
                description={capitalUtilizationDescription}
                displayValue={
                  capitalUtilization === null
                    ? "—"
                    : `${capitalUtilization.toFixed(1)}%`
                }
                label="资金利用率"
                thresholds={["0%", "50%", "100%"]}
                value={capitalUtilization}
              />
            </section>

            {recentScans.length > 0 ? (
              <section aria-label="最近扫描时间线">
                <h3 className="mb-2 flex items-center gap-2 font-medium text-sm">
                  <Clock className="size-4 text-sky-500" />
                  最近扫描时间线
                  <span className="font-normal text-muted-foreground text-xs">
                    最新 {recentScans.length} 次评估
                  </span>
                </h3>
                <ol className="space-y-1.5">
                  {recentScans.map((item) => (
                    <li
                      className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-border/60 bg-muted/10 px-3 py-2 text-xs"
                      key={item.evaluation_id}
                    >
                      <span className="font-mono text-muted-foreground tabular-nums">
                        {scanTime(item.evaluated_at)}
                      </span>
                      {item.symbol ? (
                        <Badge className="text-[10px]" variant="outline">
                          {item.symbol.replace("-", "/")}
                        </Badge>
                      ) : null}
                      <span
                        className={cn(
                          "font-medium",
                          ruleStrategyActionTone(item.action),
                        )}
                      >
                        {ruleStrategyActionLabel(item.action)}
                      </span>
                      <span className="min-w-40 flex-1 truncate text-muted-foreground">
                        {ruleStrategyEvaluationReason(item)}
                      </span>
                    </li>
                  ))}
                </ol>
              </section>
            ) : null}

            <DashboardEvaluationRhythm
              evaluations={evaluations}
              strategyName={strategyName}
            />
          </CardContent>
        </Card>
      </section>
      <RuleStrategyEvaluationPath evaluation={latestEvaluation} />
    </>
  );
}
