import { useId } from "react";
import { Activity } from "lucide-react";
import {
  conditionSatisfactionPercent,
  conditionStateSatisfactionPercent,
  dashboardConditionSummary,
} from "@/app/dashboard-funnel";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type {
  RuleStrategyAction,
  RuleStrategyEvaluationHistoryEntry,
} from "@/types/rule-strategy";

const TREND_POINT_LIMIT = 24;

export interface EvaluationTrendPoint {
  evaluatedAt: string;
  label: string;
  percent: number;
}

export interface EvaluationActionBucket {
  code: string;
  label: string;
  count: number;
  share: number;
  bar: string;
  dot: string;
  actions: RuleStrategyAction[];
}

const ACTION_BUCKETS: ReadonlyArray<{
  code: string;
  label: string;
  bar: string;
  dot: string;
  actions: RuleStrategyAction[];
}> = [
  {
    code: "open",
    label: "开仓/加仓",
    bar: "bg-emerald-500/80",
    dot: "bg-emerald-500",
    actions: ["entry", "add", "buy", "long_entry", "short_entry"],
  },
  {
    code: "close",
    label: "减仓/平仓",
    bar: "bg-rose-500/80",
    dot: "bg-rose-500",
    actions: ["reduce", "close", "sell", "exit"],
  },
  {
    code: "hold",
    label: "观望",
    bar: "bg-slate-400/70",
    dot: "bg-slate-400",
    actions: ["no_op", "hold", "no_signal"],
  },
  {
    code: "blocked",
    label: "受阻",
    bar: "bg-amber-500/80",
    dot: "bg-amber-500",
    actions: ["blocked"],
  },
];

const ACTION_BUCKET_BY_ACTION = new Map<RuleStrategyAction, string>(
  ACTION_BUCKETS.flatMap((bucket) =>
    bucket.actions.map((action) => [action, bucket.code] as const),
  ),
);

/**
 * Oldest-to-newest strip of the recorded decisions. Every evaluation has an
 * action, so this strip is populated even when the journal carries no usable
 * condition counters.
 */
export function evaluationTimelineSegments(
  evaluations: RuleStrategyEvaluationHistoryEntry[],
  limit = TREND_POINT_LIMIT,
): { key: string; code: string; action: RuleStrategyAction }[] {
  return evaluations
    .slice(0, limit)
    .map((evaluation) => ({
      key: evaluation.evaluation_id,
      code: ACTION_BUCKET_BY_ACTION.get(evaluation.action) ?? "hold",
      action: evaluation.action,
    }))
    .reverse();
}

const BUCKET_BAR: Record<string, string> = Object.fromEntries(
  ACTION_BUCKETS.map((bucket) => [bucket.code, bucket.bar]),
);

const trendTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});

/**
 * Oldest-to-newest condition-satisfaction history. Each point comes from what
 * the journal recorded for that evaluation: the summary counters when they
 * carry a denominator, otherwise the persisted condition states. Evaluations
 * with nothing recorded are skipped rather than plotted as zero.
 */
export function evaluationSatisfactionTrend(
  evaluations: RuleStrategyEvaluationHistoryEntry[],
  limit = TREND_POINT_LIMIT,
): EvaluationTrendPoint[] {
  const points: EvaluationTrendPoint[] = [];
  for (const evaluation of evaluations.slice(0, limit)) {
    const summaryPercent = conditionSatisfactionPercent(
      dashboardConditionSummary(evaluation),
    );
    const percent =
      summaryPercent ??
      conditionStateSatisfactionPercent(evaluation.conditions ?? []);
    if (percent === null) continue;
    const date = new Date(evaluation.evaluated_at);
    points.push({
      evaluatedAt: evaluation.evaluated_at,
      label: Number.isNaN(date.getTime())
        ? "服务器已记录"
        : trendTimeFormatter.format(date),
      percent,
    });
  }
  return points.reverse();
}

/**
 * Counts how the recorded evaluations were decided. Shares are computed over
 * the evaluations actually read, so the bars always add up to what is shown.
 */
export function evaluationActionDistribution(
  evaluations: RuleStrategyEvaluationHistoryEntry[],
  limit = TREND_POINT_LIMIT,
): EvaluationActionBucket[] {
  const window = evaluations.slice(0, limit);
  const total = window.length;
  return ACTION_BUCKETS.map((bucket) => {
    const count = window.filter((evaluation) =>
      bucket.actions.includes(evaluation.action),
    ).length;
    return {
      code: bucket.code,
      label: bucket.label,
      count,
      share: total === 0 ? 0 : (count / total) * 100,
      bar: bucket.bar,
      dot: bucket.dot,
      actions: bucket.actions,
    };
  });
}

const VIEW_WIDTH = 640;
const VIEW_HEIGHT = 132;
const PAD_X = 10;
const PAD_TOP = 10;
const PAD_BOTTOM = 20;

function trendPoints(trend: EvaluationTrendPoint[]) {
  const usableHeight = VIEW_HEIGHT - PAD_TOP - PAD_BOTTOM;
  const usableWidth = VIEW_WIDTH - PAD_X * 2;
  return trend.map((point, index) => {
    const ratio = trend.length > 1 ? index / (trend.length - 1) : 0.5;
    return {
      ...point,
      x: PAD_X + usableWidth * ratio,
      y: PAD_TOP + usableHeight * (1 - point.percent / 100),
    };
  });
}

function SatisfactionTrendChart({ trend }: { trend: EvaluationTrendPoint[] }) {
  const gradientId = useId().replace(/:/g, "");
  const points = trendPoints(trend);
  const line = points.map((point) => `${point.x},${point.y}`).join(" ");
  const area = `${PAD_X},${VIEW_HEIGHT - PAD_BOTTOM} ${line} ${VIEW_WIDTH - PAD_X},${
    VIEW_HEIGHT - PAD_BOTTOM
  }`;
  const last = points.at(-1);

  return (
    <svg
      aria-hidden="true"
      className="h-[132px] w-full text-sky-500"
      preserveAspectRatio="none"
      viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.4" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 50, 100].map((value) => {
        const y =
          PAD_TOP +
          (VIEW_HEIGHT - PAD_TOP - PAD_BOTTOM) * (1 - value / 100);
        return (
          <line
            className="text-border"
            key={value}
            stroke="currentColor"
            strokeDasharray="4 6"
            strokeWidth="1"
            x1={PAD_X}
            x2={VIEW_WIDTH - PAD_X}
            y1={y}
            y2={y}
          />
        );
      })}
      <polygon fill={`url(#${gradientId})`} points={area} />
      <polyline
        fill="none"
        points={line}
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2.5"
        vectorEffect="non-scaling-stroke"
      />
      {last ? (
        <>
          <circle
            className="trend-point-pulse"
            cx={last.x}
            cy={last.y}
            fill="currentColor"
            r="4.5"
          />
          <circle
            className="text-card"
            cx={last.x}
            cy={last.y}
            fill="currentColor"
            r="7.5"
          />
          <circle cx={last.x} cy={last.y} fill="currentColor" r="4" />
        </>
      ) : null}
    </svg>
  );
}

function ActionDistributionBars({
  buckets,
  total,
}: {
  buckets: EvaluationActionBucket[];
  total: number;
}) {
  return (
    <div className="grid gap-2">
      <span aria-hidden className="flex h-2.5 overflow-hidden rounded-full bg-muted/60">
        {buckets.map((bucket) => (
          <span
            className={cn("bar-grow h-full", bucket.bar)}
            key={bucket.code}
            style={{ width: `${bucket.share}%` }}
          />
        ))}
      </span>
      <ul className="grid gap-1.5">
        {buckets.map((bucket) => (
          <li
            className="flex items-center justify-between gap-3 text-xs"
            key={bucket.code}
          >
            <span className="flex items-center gap-2">
              <span
                aria-hidden
                className={cn("size-2 rounded-full", bucket.dot)}
              />
              {bucket.label}
            </span>
            <span className="text-muted-foreground tabular-nums">
              {bucket.count} 次 ·{" "}
              {total === 0 ? "0.0" : bucket.share.toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DecisionTimelineStrip({
  segments,
}: {
  segments: { key: string; code: string; action: RuleStrategyAction }[];
}) {
  return (
    <div className="grid gap-1.5">
      <span className="terminal-label">决策时间带（左旧右新）</span>
      <span aria-hidden className="flex h-1.5 gap-0.5">
        {segments.map((segment) => (
          <span
            className={cn(
              "h-full flex-1 rounded-full transition-colors",
              BUCKET_BAR[segment.code] ?? "bg-slate-400/70",
            )}
            data-action={segment.action}
            key={segment.key}
          />
        ))}
      </span>
    </div>
  );
}

/**
 * Renders the shape of a strategy's own journal over time: how close the
 * conditions came to firing, and how the recorded decisions were split. All
 * input is the server's evaluation history for the selected strategy, so
 * switching strategies on the dashboard switches this panel too.
 */
export function DashboardEvaluationRhythm({
  strategyName,
  evaluations,
}: {
  strategyName: string;
  evaluations: RuleStrategyEvaluationHistoryEntry[];
}) {
  const window = evaluations.slice(0, TREND_POINT_LIMIT);
  const trend = evaluationSatisfactionTrend(evaluations);
  const buckets = evaluationActionDistribution(evaluations);
  const segments = evaluationTimelineSegments(evaluations);
  const latest = trend.at(-1);
  const average =
    trend.length === 0
      ? null
      : trend.reduce((sum, point) => sum + point.percent, 0) / trend.length;

  return (
    <Card className="dashboard-panel overflow-hidden rounded-lg border-sky-500/20 bg-card/90 py-0 shadow-none">
      <CardHeader className="gap-1 border-border/70 border-b px-5 py-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Activity className="size-4 text-sky-500" />
          策略评估节奏
        </CardTitle>
        <CardDescription className="text-xs">
          {strategyName} · 最近 {window.length} 次服务器评估的条件通过比例与决策分布
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 px-5 py-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        {trend.length > 1 ? (
          <div className="grid gap-2">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="terminal-label">条件通过比例趋势</span>
              <span className="flex flex-wrap items-center gap-x-3 text-[11px] text-muted-foreground tabular-nums">
                <span>最新 {latest?.percent.toFixed(0)}%</span>
                <span>
                  平均 {average === null ? "—" : average.toFixed(0)}%
                </span>
                <span>{trend.length} 个记录点</span>
              </span>
            </div>
            <SatisfactionTrendChart trend={trend} />
            <div className="flex flex-wrap justify-between gap-2 text-[10px] text-muted-foreground tabular-nums">
              <span>{trend[0]?.label}</span>
              <span>100%</span>
              <span>{trend.at(-1)?.label}</span>
            </div>
          </div>
        ) : (
          <div className="grid place-items-center rounded-md border border-border/70 border-dashed px-4 py-10 text-center">
            <p className="text-muted-foreground text-xs">
              服务器记录的带条件判定的评估还不足两次，暂时无法绘制趋势。
            </p>
          </div>
        )}
        <div className="grid content-start gap-3">
          <DecisionTimelineStrip segments={segments} />
          <span className="terminal-label">决策分布</span>
          <ActionDistributionBars buckets={buckets} total={window.length} />
          <p className="text-[10px] text-muted-foreground leading-relaxed">
            分布只统计服务器已经写入评估日志的 {window.length} 次决策，不推算未记录的轮次。
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
