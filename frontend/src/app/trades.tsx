import {
  ChevronLeft,
  ChevronRight,
  ClipboardList,
  FileClock,
  FileDown,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useSearchParams } from "react-router";
import { toast } from "sonner";
import {
  useAllTradeFacts,
  useExportRuleStrategy,
  useRuleStrategies,
  useRuleStrategy,
  useRuleStrategyBatches,
  useRuleStrategyDemoExecution,
  useRuleStrategyTrades,
} from "@/api/rule-strategy";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useActiveRuleStrategyId } from "@/hooks/use-active-rule-strategy";
import { useSaaSSession } from "@/store/system-store";
import type { UnifiedTradeFact } from "@/types/multi-strategy";
import {
  decisionConditions,
  decisionLabel,
  formatConditionValues,
} from "./trades-strategy-conditions";
import {
  demoOrderAveragePriceLabel,
  demoOrderFilledQuantityLabel,
  selectTradesSource,
} from "./trades-source";
function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export default function TradesPage() {
  const { t } = useTranslation();
  const { tenantId } = useSaaSSession();
  const [strategyId] = useActiveRuleStrategyId();
  const [searchParams] = useSearchParams();
  const [demoOrdersPage, setDemoOrdersPage] = useState(1);
  const [selectedBatchId, setSelectedBatchId] = useState<string | null>(null);
  const [factStrategyFilter, setFactStrategyFilter] = useState("all");
  const [factStatusFilter, setFactStatusFilter] = useState("all");
  const exportStrategy = useExportRuleStrategy();
  const strategyQuery = useRuleStrategy(strategyId);
  const strategiesQuery = useRuleStrategies(tenantId, true);
  const batchesQuery = useRuleStrategyBatches(strategyId);
  const allHistory = selectedBatchId === "__all__";
  const source = selectTradesSource(
    strategyQuery.data !== undefined,
    strategyQuery.data?.config.execution?.environment,
  );
  const factsQuery = useAllTradeFacts(
    factStrategyFilter === "all" ? null : factStrategyFilter,
    allHistory ? null : selectedBatchId,
    Boolean(tenantId),
  );
  const paperTradesQuery = useRuleStrategyTrades(strategyId, source === "paper", selectedBatchId);
  const demoExecutionQuery = useRuleStrategyDemoExecution(
    strategyId, source === "okx_demo", demoOrdersPage, 10,
    allHistory ? null : selectedBatchId, allHistory,
  );
  useEffect(() => {
    setDemoOrdersPage(1);
    setSelectedBatchId(null);
    setFactStrategyFilter("all");
  }, [strategyId]);
  useEffect(() => {
    if (!selectedBatchId && batchesQuery.data?.current_batch_id) {
      setSelectedBatchId(batchesQuery.data.current_batch_id);
    }
  }, [batchesQuery.data?.current_batch_id, selectedBatchId]);
  const selectedOrderId = searchParams.get("orderId");
  const paperTrades = paperTradesQuery.data ?? [];
  const demoOrders = demoExecutionQuery.data?.orders ?? [];
  const demoOrdersPagination = demoExecutionQuery.data?.pagination;
  const facts = factsQuery.data ?? [];
  const filteredFacts = useMemo(
    () => facts.filter((fact) => factStrategyFilter === "all" || fact.identity.strategy_id === factStrategyFilter)
      .filter((fact) => factStatusFilter === "all" || fact.status === factStatusFilter),
    [factStatusFilter, factStrategyFilter, facts],
  );
  useEffect(() => {
    if (demoOrdersPagination && demoOrdersPage > demoOrdersPagination.total_pages) {
      setDemoOrdersPage(demoOrdersPagination.total_pages);
    }
  }, [demoOrdersPage, demoOrdersPagination]);
  const downloadAllDemoOrders = async () => {
    if (!strategyId || exportStrategy.isPending) return;
    try {
      const workbook = await exportStrategy.mutateAsync({ strategyId, batchId: allHistory ? undefined : selectedBatchId ?? undefined });
      const objectUrl = URL.createObjectURL(workbook.blob);
      const filename = workbook.filename?.toLowerCase().endsWith(".xlsx") ? workbook.filename : `${workbook.filename || "策略订单导出"}.xlsx`;
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
      toast.success("全部订单记录已开始下载。");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "订单导出失败，请稍后重试。");
    }
  };
  const isPaper = source === "paper";
  const isDemo = source === "okx_demo";
  const recordsLoading = isPaper ? paperTradesQuery.isLoading : demoExecutionQuery.isLoading;
  const recordsError = isPaper ? paperTradesQuery.isError : demoExecutionQuery.isError;
  const hasRecords = isPaper ? paperTrades.length > 0 : demoOrders.length > 0;


  return (
    <div className="scroll-container size-full bg-muted/40">
      <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col gap-6 p-4 md:p-6 lg:p-8">
        <header>
          <div className="mb-2 flex gap-2">
            <Badge variant="secondary">
              {isDemo ? "OKX Demo 交易所订单" : "纸面交易账本"}
            </Badge>
            <Badge variant="outline">
              {isDemo ? "共享 Demo 连接账户" : "无交易所执行账户"}
            </Badge>
          </div>
          <h1 className="font-semibold text-2xl tracking-tight">
            {t("saas.operations.trades.title")}
          </h1>
          <p className="mt-1 text-muted-foreground text-sm">
            {isDemo
              ? "仅显示此策略在 OKX Demo 的交易所订单；不混入纸面成交、纸面账本或纸面盈亏。"
              : t("saas.operations.trades.subtitle")}
          </p>
        </header>

        {strategyId ? (
          <label className="flex flex-wrap items-center gap-3 rounded-lg border bg-background p-3 font-medium text-sm">
            执行批次
            <select
              className="min-w-72 rounded-md border bg-background px-3 py-2 font-normal"
              onChange={(event) => {
                setSelectedBatchId(event.target.value || null);
                setDemoOrdersPage(1);
              }}
              value={selectedBatchId ?? ""}
            >
              <option value="__all__">全部历史订单（跨执行批次）</option>
              <option value="">当前执行批次</option>
              {(batchesQuery.data?.items ?? []).map((batch) => (
                <option key={batch.batch_id} value={batch.batch_id}>
                  {batch.status === "running" ? "运行中" : "已归档，未删除"} · {formatDate(batch.started_at)}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        {tenantId ? (
          <UnifiedTradeFactsCard
            facts={filteredFacts}
            factStrategyFilter={factStrategyFilter}
            factStatusFilter={factStatusFilter}
            factsQuery={factsQuery}
            onStrategyChange={setFactStrategyFilter}
            onStatusChange={setFactStatusFilter}
            strategies={strategiesQuery.data ?? []}
          />
        ) : null}

        {!strategyId ? (
          <EmptyState
            action={t("saas.operations.trades.actions.configure")}
            description={t("saas.operations.trades.noStrategy.description")}
            title={t("saas.operations.trades.noStrategy.title")}
          />
        ) : strategyQuery.isLoading || source === "pending" ? (
          <MessageCard>正在读取策略执行环境…</MessageCard>
        ) : strategyQuery.isError ? (
          <MessageCard error>无法加载策略，不能确定交易记录来源。</MessageCard>
        ) : recordsLoading ? (
          <MessageCard>
            {isDemo
              ? "正在加载 OKX Demo 订单…"
              : t("saas.operations.trades.loading")}
          </MessageCard>
        ) : recordsError ? (
          <MessageCard error>
            {isDemo
              ? "OKX Demo 订单加载失败；不会回退展示纸面交易记录。"
              : t("saas.operations.trades.unavailable")}
          </MessageCard>
        ) : hasRecords ? (
          isDemo ? (
            <Card>
              <CardHeader className="flex flex-row items-start justify-between gap-4">
                <div>
                  <CardTitle>OKX Demo 订单</CardTitle>
                  <CardDescription>
                    订单状态来自 Demo
                    执行端点。当前接口未提供已成交量和成交均价时会明确显示“不可用”。
                  </CardDescription>
                </div>
                <Button
                  disabled={!strategyId || exportStrategy.isPending}
                  onClick={() => void downloadAllDemoOrders()}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <FileDown className="size-4" />
                  {exportStrategy.isPending ? "正在导出" : "导出全部"}
                </Button>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>创建时间</TableHead>
                      <TableHead>市场</TableHead>
                      <TableHead>方向</TableHead>
                      <TableHead>类型</TableHead>
                      <TableHead>订单状态</TableHead>
                      <TableHead className="text-right">委托金额</TableHead>
                      <TableHead className="text-right">已成交量</TableHead>
                      <TableHead className="text-right">成交均价</TableHead>
                      <TableHead>策略依据</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {demoOrders.map((order) => (
                      <TableRow className={selectedOrderId === order.id ? "bg-sky-500/10" : ""} key={order.id}>
                        <TableCell className="whitespace-nowrap">{formatDate(order.created_at)}</TableCell>
                        <TableCell><Link className="font-medium text-sky-600 hover:underline dark:text-sky-300" to={`/positions?strategyId=${encodeURIComponent(strategyId ?? "")}&symbol=${encodeURIComponent(order.symbol.replace("/", "-"))}&orderId=${encodeURIComponent(order.id)}${selectedBatchId ? `&batch_id=${encodeURIComponent(selectedBatchId)}` : ""}${order.evaluation_id ? `&evaluationId=${encodeURIComponent(order.evaluation_id)}` : ""}`}>{order.symbol}</Link></TableCell>
                        <TableCell className="uppercase">{order.side}</TableCell>
                        <TableCell>{order.type}</TableCell>
                        <TableCell><Badge variant="outline">{order.status}</Badge></TableCell>
                        <TableCell className="text-right tabular-nums">{order.requested_quote} USDT</TableCell>
                        <TableCell className="text-right tabular-nums">{demoOrderFilledQuantityLabel(order)}</TableCell>
                        <TableCell className="text-right tabular-nums">{demoOrderAveragePriceLabel(order)}</TableCell>
                        <TableCell className="max-w-96 whitespace-normal text-xs">
                          <div className="font-medium">{decisionLabel(order)}</div>
                          {decisionConditions(order).length === 0 ? (
                            <div className="text-muted-foreground">未找到该订单对应的持久化条件记录，未使用当前策略反推。</div>
                          ) : (
                            decisionConditions(order).map((condition) => (
                              <div className="text-muted-foreground" key={`${order.id}-${condition.code}`}>
                                {condition.label || condition.code}：{condition.state === "triggered" ? "满足" : condition.state === "not_triggered" ? "不满足" : "不可用"}{formatConditionValues(condition.values)}
                              </div>
                            ))
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-border border-t pt-4 text-xs">
                  <span className="text-muted-foreground">
                    共 {demoOrdersPagination?.total_items ?? demoOrders.length} 条 · 第 {demoOrdersPagination?.page ?? demoOrdersPage} / {demoOrdersPagination?.total_pages ?? 1} 页
                  </span>
                  <div className="flex items-center gap-2">
                    <Button
                      aria-label="上一页订单"
                      disabled={demoOrdersPage <= 1 || demoExecutionQuery.isFetching}
                      onClick={() => setDemoOrdersPage((page) => Math.max(1, page - 1))}
                      size="icon"
                      type="button"
                      variant="outline"
                    >
                      <ChevronLeft />
                    </Button>
                    <Button
                      aria-label="下一页订单"
                      disabled={demoOrdersPage >= (demoOrdersPagination?.total_pages ?? 1) || demoExecutionQuery.isFetching}
                      onClick={() => setDemoOrdersPage((page) => page + 1)}
                      size="icon"
                      type="button"
                      variant="outline"
                    >
                      <ChevronRight />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>
                  {t("saas.operations.trades.recommendations.title")}
                </CardTitle>
                <CardDescription>
                  {t("saas.operations.trades.recommendations.description")}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>
                        {t("saas.operations.trades.table.evaluated")}
                      </TableHead>
                      <TableHead>
                        {t("saas.operations.trades.table.action")}
                      </TableHead>
                      <TableHead>Market</TableHead>
                      <TableHead className="text-right">Fill price</TableHead>
                      <TableHead className="text-right">Notional</TableHead>
                      <TableHead className="text-right">Realized PnL</TableHead>
                      <TableHead>
                        {t("saas.operations.trades.table.reason")}
                      </TableHead>
                      <TableHead>
                        {t("saas.operations.trades.table.execution")}
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {paperTrades.map((trade) => (
                      <TableRow key={`${trade.evaluation_id}-${trade.action}`}>
                        <TableCell className="whitespace-nowrap">{formatDate(trade.evaluated_at)}</TableCell>
                        <TableCell className="capitalize">{trade.action}</TableCell>
                        <TableCell><Link className="font-medium text-sky-600 hover:underline dark:text-sky-300" to={`/positions?strategyId=${encodeURIComponent(strategyId ?? "")}&symbol=${encodeURIComponent(trade.symbol)}&evaluationId=${encodeURIComponent(trade.evaluation_id)}${selectedBatchId && selectedBatchId !== "__all__" ? `&batch_id=${encodeURIComponent(selectedBatchId)}` : ""}`}>{trade.symbol}</Link></TableCell>
                        <TableCell className="text-right tabular-nums">{trade.price.toFixed(4)}</TableCell>
                        <TableCell className="text-right tabular-nums">{trade.quote_amount.toFixed(2)} USDT</TableCell>
                        <TableCell className="text-right tabular-nums">{trade.realized_pnl_quote.toFixed(2)} USDT</TableCell>
                        <TableCell className="whitespace-normal break-words">{trade.reason}</TableCell>
                        <TableCell><Badge variant="outline">{trade.execution.replace("_", " ")}</Badge></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )
        ) : (
          <EmptyState
            action={isDemo ? "配置策略" : t("saas.operations.trades.actions.evaluate")}
            description={isDemo ? "该策略尚无 OKX Demo 订单。纸面交易记录不会显示在此视图中。" : t("saas.operations.trades.empty.description")}
            title={isDemo ? "暂无 OKX Demo 订单" : t("saas.operations.trades.empty.title")}
          />
        )}

        {isPaper ? (
          <section className="grid gap-4 md:grid-cols-2">
            <InfoCard
              icon={FileClock}
              title={t("saas.operations.trades.ledger.title")}
            >
              {t("saas.operations.trades.ledger.description")}
            </InfoCard>

            <InfoCard
              icon={ShieldCheck}
              title={t("saas.operations.trades.separation.title")}
            >
              {t("saas.operations.trades.separation.description")}
            </InfoCard>
          </section>
        ) : null}
      </div>
    </div>
  );
}

function UnifiedTradeFactsCard({
  facts,
  factStrategyFilter,
  factStatusFilter,
  factsQuery,
  onStrategyChange,
  onStatusChange,
  strategies,
}: {
  facts: UnifiedTradeFact[];
  factStrategyFilter: string;
  factStatusFilter: string;
  factsQuery: { isLoading: boolean; isError: boolean };
  onStrategyChange: (value: string) => void;
  onStatusChange: (value: string) => void;
  strategies: Array<{ strategy_id: string; name: string; strategy_kind: string }>;
}) {
  const statusLabel: Record<string, string> = {
    signal: "信号",
    blocked: "已拦截",
    pending: "待处理",
    submitted: "已提交",
    partially_filled: "部分成交",
    filled: "已成交",
    cancelled: "已取消",
    failed: "失败",
  };
  return (
    <Card>
      <CardHeader className="gap-4">
        <div>
          <CardTitle>跨策略归因交易事实</CardTitle>
          <CardDescription>每条记录均来自持久化执行事实；不会以当前策略参数反推历史原因。</CardDescription>
        </div>
        <div className="flex flex-wrap gap-2">
          <label className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">策略</span>
            <select className="h-9 rounded-md border bg-background px-3" value={factStrategyFilter} onChange={(event) => onStrategyChange(event.target.value)}>
              <option value="all">全部策略</option>
              {strategies.map((strategy) => <option key={strategy.strategy_id} value={strategy.strategy_id}>{strategy.name} · {strategy.strategy_kind}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">状态</span>
            <select className="h-9 rounded-md border bg-background px-3" value={factStatusFilter} onChange={(event) => onStatusChange(event.target.value)}>
              <option value="all">全部状态</option>
              {Object.entries(statusLabel).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
        </div>
      </CardHeader>
      <CardContent>
        {factsQuery.isLoading ? <div className="py-8 text-center text-muted-foreground text-sm">正在加载归因事实…</div> : factsQuery.isError ? <div className="py-8 text-center text-destructive text-sm" role="alert">归因事实暂时不可用；原有来源专属记录仍保持独立。</div> : facts.length === 0 ? <div className="py-8 text-center text-muted-foreground text-sm">暂无符合筛选条件的归因交易事实。</div> : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader><TableRow><TableHead>时间 / 策略</TableHead><TableHead>市场</TableHead><TableHead>方向</TableHead><TableHead>状态</TableHead><TableHead className="text-right">金额 / 数量</TableHead><TableHead className="text-right">均价 / 费用</TableHead><TableHead>失败</TableHead><TableHead>解释</TableHead></TableRow></TableHeader>
              <TableBody>
                {facts.map((fact) => <UnifiedTradeFactRow fact={fact} key={`${fact.identity.strategy_id}-${fact.order_id ?? fact.evaluation_id ?? fact.created_at}`} statusLabel={statusLabel} />)}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function UnifiedTradeFactRow({ fact, statusLabel }: { fact: UnifiedTradeFact; statusLabel: Record<string, string> }) {
  const explanation = fact.explanation;
  return (
    <TableRow>
      <TableCell className="whitespace-nowrap"><div>{formatDate(fact.created_at)}</div><div className="text-muted-foreground text-xs">{fact.identity.kind} · {fact.identity.strategy_id.slice(0, 8)}</div></TableCell>
      <TableCell className="font-medium">{fact.symbol}</TableCell>
      <TableCell className="uppercase">{fact.side}</TableCell>
      <TableCell><Badge variant={fact.status === "failed" || fact.status === "blocked" ? "destructive" : "outline"}>{statusLabel[fact.status] ?? fact.status}</Badge></TableCell>
      <TableCell className="text-right tabular-nums"><div>{fact.filled_quote ?? fact.requested_quote ?? "不可用"} USDT</div><div className="text-muted-foreground text-xs">数量 {fact.filled_quantity ?? fact.requested_quantity ?? "不可用"}</div></TableCell>
      <TableCell className="text-right tabular-nums"><div>{fact.average_fill_price ?? "不可用"}</div><div className="text-muted-foreground text-xs">费用 {fact.fee_quote == null ? "不可用" : `${fact.fee_quote} USDT`}</div></TableCell>
      <TableCell className="max-w-48 whitespace-normal text-xs">{fact.failure_reason ?? fact.failure_code ?? "—"}</TableCell>
      <TableCell className="min-w-64">
        <Collapsible>
          <CollapsibleTrigger className="rounded-md px-2 py-1 text-left text-primary text-sm hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">查看：{explanation.decision || "未提供"}</CollapsibleTrigger>
          <CollapsibleContent className="space-y-2 pt-2 text-xs">
            <div className="text-muted-foreground">{explanation.decision_reason || "暂无持久化解释。"}</div>
            {explanation.conditions.length === 0 ? <div className="text-muted-foreground">条件明细不可用</div> : explanation.conditions.map((condition) => <div className="rounded border bg-muted/30 p-2" key={`${condition.code}-${condition.data_at}`}><div className="font-medium">{condition.label || condition.code} · {condition.state}</div><div>实际值：{condition.actual ?? "不可用"}　{condition.operator ?? "对比"}　阈值：{condition.threshold ?? "不可用"}</div><div className="text-muted-foreground">{condition.detail}{condition.data_at ? ` · 数据时间 ${formatDate(condition.data_at)}` : ""}</div></div>)}
          </CollapsibleContent>
        </Collapsible>
      </TableCell>
    </TableRow>
  );
}
function MessageCard({
  children,
  error = false,
}: {
  children: React.ReactNode;
  error?: boolean;
}) {
  return (
    <Card>
      <CardContent
        className={
          error
            ? "p-6 text-destructive text-sm"
            : "p-6 text-muted-foreground text-sm"
        }
        role={error ? "alert" : undefined}
      >
        {children}
      </CardContent>
    </Card>
  );
}

function EmptyState({
  action,
  description,
  title,
}: {
  action: string;
  description: string;
  title: string;
}) {
  return (
    <Card className="flex flex-1 items-center justify-center border-dashed">
      <CardHeader className="max-w-md items-center text-center">
        <div className="mb-2 flex size-12 items-center justify-center rounded-full bg-secondary">
          <ClipboardList className="size-6 text-muted-foreground" />
        </div>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="flex justify-center pb-8">
        <Button asChild variant="outline">
          <Link to="/strategies">{action}</Link>
        </Button>
      </CardContent>
    </Card>
  );
}

function InfoCard({
  children,
  icon: Icon,
  title,
}: {
  children: React.ReactNode;
  icon: typeof FileClock;
  title: string;
}) {
  return (
    <Card className="gap-3 py-5">
      <CardHeader className="px-5">
        <div className="flex items-center gap-2">
          <Icon className="size-4 text-muted-foreground" />
          <CardTitle className="text-base">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="px-5 text-muted-foreground text-sm">
        {children}
      </CardContent>
    </Card>
  );
}
