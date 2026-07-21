import {
  BrainCircuit,
  ChevronLeft,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";
import { toast } from "sonner";
import {
  useRuleStrategy,
  useRuleStrategyAdvisory,
  useRuleStrategyEvaluations,
} from "@/api/rule-strategy";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useActiveRuleStrategyId } from "@/hooks/use-active-rule-strategy";
import type { RuleStrategyAdvisory } from "@/types/rule-strategy";
import { initialCapitalLabel } from "./advisory-values";

const formatTimestamp = (value?: string) =>
  value
    ? new Intl.DateTimeFormat("zh-CN", {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(new Date(value))
    : "尚无评估记录";

export default function StrategyAdvisoryPage() {
  const [strategyId] = useActiveRuleStrategyId();
  const strategyQuery = useRuleStrategy(strategyId);
  const evaluationsQuery = useRuleStrategyEvaluations(strategyId);
  const advisoryMutation = useRuleStrategyAdvisory(strategyId);
  const [advisory, setAdvisory] = useState<RuleStrategyAdvisory | null>(null);
  const latestEvaluation = evaluationsQuery.data?.[0];

  const generateReview = async () => {
    try {
      const response = await advisoryMutation.mutateAsync();
      setAdvisory(response.data);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "AI 审阅暂不可用。");
    }
  };

  if (!strategyId) {
    return <EmptyState />;
  }

  return (
    <div className="scroll-container size-full bg-muted/40">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-4 md:p-6 lg:p-8">
        <header className="flex flex-col gap-4 border-b pb-5 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 gap-3">
            <Button
              asChild
              aria-label="返回策略配置"
              size="icon"
              variant="outline"
            >
              <Link to="/strategies">
                <ChevronLeft />
              </Link>
            </Button>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary">AI 策略审阅</Badge>
                <Badge variant="outline">仅建议</Badge>
              </div>
              <h1 className="mt-2 text-2xl font-semibold tracking-tight">
                {strategyQuery.data?.name ?? "策略配置审阅"}
              </h1>
              <p className="mt-1 text-sm text-muted-foreground">
                使用 HiCode GPT-5.5
                对已保存配置和确定性评估记录进行中文只读分析。
              </p>
            </div>
          </div>
          <Button
            disabled={advisoryMutation.isPending || strategyQuery.isLoading}
            onClick={generateReview}
          >
            <RefreshCw
              className={advisoryMutation.isPending ? "animate-spin" : ""}
            />
            {advisoryMutation.isPending
              ? "正在生成…"
              : advisory
                ? "重新生成审阅"
                : "生成 AI 审阅"}
          </Button>
        </header>

        <Alert className="border-sky-500/30 bg-sky-500/5">
          <ShieldCheck />
          <AlertTitle>AI 不参与交易决策</AlertTitle>
          <AlertDescription>
            审阅只读取已保存的规则与评估事实，不能修改策略，不能改变买入、卖出或不交易结论，也不能创建纸面或真实订单。
          </AlertDescription>
        </Alert>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_280px]">
          <Card className="gap-0 py-0 shadow-none">
            <CardHeader className="border-b px-5 py-5">
              <CardTitle className="flex items-center gap-2 text-base">
                <BrainCircuit /> AI 审阅内容
              </CardTitle>
              <CardDescription>
                {advisory
                  ? `模型：${advisory.model_id} · 只读建议`
                  : "点击“生成 AI 审阅”获取中文配置建议。"}
              </CardDescription>
            </CardHeader>
            <CardContent className="min-h-80 px-5 py-5">
              {advisory ? (
                <article className="whitespace-pre-wrap text-sm leading-7">
                  {advisory.content}
                </article>
              ) : (
                <div className="flex min-h-64 flex-col items-center justify-center text-center text-muted-foreground">
                  <BrainCircuit className="mb-3 size-8" />
                  <p>尚未生成审阅。</p>
                  <p className="mt-1 text-xs">
                    生成后会在这里完整显示中文分析，不会挤压策略配置页面。
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <aside className="flex flex-col gap-4">
            <Card className="gap-0 py-0 shadow-none">
              <CardHeader className="border-b px-4 py-4">
                <CardTitle className="text-base">已审阅的配置</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-3 px-4 py-4 text-sm">
                <Fact
                  label="市场"
                  value={
                    strategyQuery.data?.config.symbols.join(", ") ?? "加载中"
                  }
                />
                <Fact
                  label="K 线周期"
                  value={strategyQuery.data?.config.interval ?? "—"}
                />
                <Fact
                  label="确认方式"
                  value={
                    strategyQuery.data?.config.confirmation_mode === "all"
                      ? "全部条件满足"
                      : "任一条件满足"
                  }
                />
                <Fact
                  label="初始纸面资金"
                  value={initialCapitalLabel(
                    strategyQuery.data?.config.initial_capital_quote,
                  )}
                />
              </CardContent>
            </Card>

            <Card className="gap-0 py-0 shadow-none">
              <CardHeader className="border-b px-4 py-4">
                <CardTitle className="text-base">最近确定性评估</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-3 px-4 py-4 text-sm">
                {latestEvaluation ? (
                  <>
                    <Badge className="w-fit capitalize" variant="outline">
                      {latestEvaluation.action.replace("_", " ")}
                    </Badge>
                    <p>{latestEvaluation.reason}</p>
                    <Separator />
                    <Fact
                      label="评估时间"
                      value={formatTimestamp(latestEvaluation.evaluated_at)}
                    />
                    <Fact
                      label="未触发或受阻条件"
                      value={String(
                        latestEvaluation.conditions.filter(
                          (condition) => condition.state !== "triggered",
                        ).length,
                      )}
                    />
                  </>
                ) : (
                  <p className="text-muted-foreground">
                    策略尚无评估记录。启动策略后，调度器会按其 K
                    线周期记录确定性评估。
                  </p>
                )}
              </CardContent>
            </Card>
          </aside>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex size-full items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>尚未选择策略</CardTitle>
          <CardDescription>
            请先保存或选择一个纸面策略，再生成 AI 配置审阅。
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button asChild>
            <Link to="/strategies">前往策略配置</Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-muted-foreground text-xs">{label}</p>
      <p className="mt-1 break-words font-medium">{value}</p>
    </div>
  );
}
