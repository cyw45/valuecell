import { FileDown, Pencil, Play, Plus, Square, Trash2 } from "lucide-react";
import { type FormEvent, useRef, useState } from "react";
import { Link } from "react-router";
import { toast } from "sonner";
import {
  useCreateFixedRuleStrategy,
  useExportRuleStrategy,
  useRuleStrategies,
  useRuleStrategyLifecycleAction,
} from "@/api/rule-strategy";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useActiveRuleStrategyId } from "@/hooks/use-active-rule-strategy";
import { cn } from "@/lib/utils";
import { useSaaSSession } from "@/store/system-store";
import type { RuleStrategy } from "@/types/rule-strategy";
import { strategyManagementActions } from "@/app/strategies/strategy-management";

const STATUS_LABELS: Record<RuleStrategy["status"], string> = {
  running: "运行中",
  stopped: "已停止",
  archived: "已归档",
};

function StrategyExportPopover({ strategy }: { strategy: RuleStrategy }) {
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const exportStrategy = useExportRuleStrategy();
  const downloadInFlight = useRef(false);
  const invalidDateRange = Boolean(fromDate && toDate && fromDate > toDate);

  const download = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (invalidDateRange || exportStrategy.isPending || downloadInFlight.current) {
      return;
    }
    downloadInFlight.current = true;
    setErrorMessage("");
    const selectedDate = fromDate || toDate;
    try {
      const workbook = await exportStrategy.mutateAsync({
        strategyId: strategy.strategy_id,
        fromDate: fromDate || selectedDate || undefined,
        toDate: toDate || selectedDate || undefined,
      });
      const objectUrl = URL.createObjectURL(workbook.blob);
      const serverFilename = workbook.filename?.trim();
      const filename = serverFilename?.toLowerCase().endsWith(".xlsx")
        ? serverFilename
        : `${serverFilename || "策略导出"}.xlsx`;
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filename;
      anchor.style.display = "none";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
      toast.success("策略历史已开始下载。");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "导出失败，请稍后重试。");
    } finally {
      downloadInFlight.current = false;
    }
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button size="sm" type="button" variant="outline">
          <FileDown /> 导出
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80">
        <form className="grid gap-3" noValidate onSubmit={download}>
          <div>
            <p className="font-medium text-sm">导出策略历史</p>
            <p className="mt-1 text-muted-foreground text-xs">
              可选择单日或日期范围；全部留空则导出全部历史。
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="grid gap-1.5">
              <Label htmlFor={`${strategy.strategy_id}-export-from`}>开始日期</Label>
              <Input
                id={`${strategy.strategy_id}-export-from`}
                onChange={(event) => {
                  setFromDate(event.target.value);
                  setErrorMessage("");
                }}
                type="date"
                value={fromDate}
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`${strategy.strategy_id}-export-to`}>结束日期</Label>
              <Input
                id={`${strategy.strategy_id}-export-to`}
                onChange={(event) => {
                  setToDate(event.target.value);
                  setErrorMessage("");
                }}
                type="date"
                value={toDate}
              />
            </div>
          </div>
          {invalidDateRange ? (
            <p className="text-destructive text-xs" role="alert">
              开始日期不得晚于结束日期。
            </p>
          ) : null}
          {errorMessage ? (
            <p className="text-destructive text-xs" role="alert">
              {errorMessage}
            </p>
          ) : null}
          <Button disabled={invalidDateRange || exportStrategy.isPending} type="submit">
            <FileDown />
            {exportStrategy.isPending ? "正在生成文件…" : "下载 Excel"}
          </Button>
        </form>
      </PopoverContent>
    </Popover>
  );
}

export function DashboardStrategyManagement() {
  const { tenantId } = useSaaSSession();
  const [activeStrategyId, setActiveStrategyId] = useActiveRuleStrategyId();
  const strategiesQuery = useRuleStrategies(tenantId);
  const createFixed = useCreateFixedRuleStrategy();
  const lifecycleAction = useRuleStrategyLifecycleAction();
  const strategies = strategiesQuery.data ?? [];
  const fixedDefinitions = [
    { kind: "dual_ma_trend" as const, name: "双均线趋势策略" },
    { kind: "pair_rotation" as const, name: "配对套利策略" },
    { kind: "leader_breakout" as const, name: "现货龙头策略" },
  ];

  const runLifecycleAction = async (
    strategy: RuleStrategy,
    action: "start" | "stop" | "delete",
  ) => {
    setActiveStrategyId(strategy.strategy_id);
    try {
      const response = await lifecycleAction.mutateAsync({
        strategyId: strategy.strategy_id,
        action,
      });
      if (action === "delete") {
        if (activeStrategyId === strategy.strategy_id) setActiveStrategyId("");
        const archived = "archived" in response.data && response.data.archived;
        toast.success(archived ? "策略已安全归档。" : "策略已删除。");
        return;
      }
      toast.success(action === "start" ? "策略已启动。" : "策略已停止。");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "策略操作失败，请稍后重试。");
    }
  };
  const createFixedStrategy = async (kind: "dual_ma_trend" | "pair_rotation" | "leader_breakout", name: string) => {
    try {
      await createFixed.mutateAsync({
        kind,
        name,
        initial_capital_quote: 10_000,
        environment: "paper",
      });
      toast.success(`${name}已创建，规则固定且当前处于停止状态。`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "固定策略创建失败。");
    }
  };

  return (
    <Card className="dashboard-panel order-8 rounded-lg border-white/10 bg-card/90 py-0 shadow-none">
      <CardHeader className="flex flex-col gap-3 border-border/70 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle className="text-base">策略管理</CardTitle>
          <CardDescription>
            选择任一策略后，资金、执行、交易、分析、权益与监控信息同步切换。
          </CardDescription>
        </div>
        <Button asChild size="sm" type="button">
          <Link to="/strategies/new">
            <Plus /> 新增策略
          </Link>
        </Button>
      </CardHeader>
      <CardContent className="p-4">
        <section className="mb-4 rounded-lg border border-sky-500/20 bg-sky-500/5 p-3">
          <p className="font-medium text-sm">代码固定策略</p>
          <p className="mt-1 text-muted-foreground text-xs">规则由代码版本管理，创建后可统一启停，但不能在前端编辑参数。</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {fixedDefinitions.map((definition) => <Button disabled={createFixed.isPending} key={definition.kind} onClick={() => void createFixedStrategy(definition.kind, definition.name)} size="sm" type="button" variant="outline">{definition.name}</Button>)}
          </div>
        </section>
        {strategiesQuery.isPending ? (
          <p className="text-muted-foreground text-sm">正在加载策略…</p>
        ) : strategies.length === 0 ? (
          <div className="rounded-md border border-dashed px-4 py-6 text-center">
            <p className="font-medium text-sm">尚未创建策略</p>
            <p className="mt-1 text-muted-foreground text-xs">
              新增策略后，即可在此选择、启动、编辑、删除和导出历史。
            </p>
          </div>
        ) : (
          <div className="grid gap-3 xl:grid-cols-2">
            {strategies.map((strategy) => {
              const selected = strategy.strategy_id === activeStrategyId;
              const actions = strategyManagementActions({
                selectedStatus: strategy.status,
              });
              return (
                <article
                  className={cn(
                    "rounded-lg border bg-background/50 p-4 transition-colors",
                    selected
                      ? "border-sky-500/60 bg-sky-500/5 shadow-sm"
                      : "border-border hover:border-sky-500/35",
                  )}
                  key={strategy.strategy_id}
                >
                  <button
                    aria-pressed={selected}
                    className="flex w-full items-start justify-between gap-3 text-left"
                    onClick={() => setActiveStrategyId(strategy.strategy_id)}
                    type="button"
                  >
                    <span className="min-w-0">
                      <span className="block truncate font-medium text-sm">
                        {strategy.name}
                      </span>
                      <span className="mt-1 block text-muted-foreground text-xs">
                        {strategy.config.symbols.join("、")} · {strategy.config.interval} 周期
                      </span>
                    </span>
                    <Badge variant={strategy.status === "running" ? "default" : "outline"}>
                      {STATUS_LABELS[strategy.status]}
                    </Badge>
                  </button>
                  <div className="mt-4 flex flex-wrap gap-2 border-border/70 border-t pt-3">
                    {actions.canStart ? (
                      <Button
                        disabled={lifecycleAction.isPending}
                        onClick={() => void runLifecycleAction(strategy, "start")}
                        size="sm"
                        type="button"
                      >
                        <Play /> 启动
                      </Button>
                    ) : null}
                    {actions.canStop ? (
                      <Button
                        disabled={lifecycleAction.isPending}
                        onClick={() => void runLifecycleAction(strategy, "stop")}
                        size="sm"
                        type="button"
                        variant="outline"
                      >
                        <Square /> 停止
                      </Button>
                    ) : null}
                    {strategy.strategy_kind === "configurable_rule" ? <Button asChild size="sm" type="button" variant="outline">
                      <Link to={`/strategies/${strategy.strategy_id}/edit`}>
                        <Pencil /> 编辑
                      </Link>
                    </Button> : <span className="rounded-md border border-border px-3 py-2 text-muted-foreground text-xs">规则固定，只读</span>}
                    <StrategyExportPopover strategy={strategy} />
                    {actions.canDelete ? (
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button size="sm" type="button" variant="ghost">
                            <Trash2 /> 删除
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>删除“{strategy.name}”？</AlertDialogTitle>
                            <AlertDialogDescription>
                              仅已停止策略可删除；已有交易审计记录的策略会安全归档。
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>取消</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => void runLifecycleAction(strategy, "delete")}
                            >
                              确认删除
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    ) : null}
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
