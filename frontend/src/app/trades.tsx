import { ClipboardList, FileClock, ShieldCheck } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import {
  useRuleStrategy,
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useActiveRuleStrategyId } from "@/hooks/use-active-rule-strategy";
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
  const [strategyId] = useActiveRuleStrategyId();
  const strategyQuery = useRuleStrategy(strategyId);
  const source = selectTradesSource(
    strategyQuery.data?.config.execution.environment,
  );
  const paperTradesQuery = useRuleStrategyTrades(
    strategyId,
    source === "paper",
  );
  const demoExecutionQuery = useRuleStrategyDemoExecution(
    strategyId,
    source === "okx_demo",
  );
  const paperTrades = paperTradesQuery.data ?? [];
  const demoOrders = demoExecutionQuery.data?.orders ?? [];
  const isPaper = source === "paper";
  const isDemo = source === "okx_demo";
  const recordsLoading = isPaper
    ? paperTradesQuery.isLoading
    : demoExecutionQuery.isLoading;
  const recordsError = isPaper
    ? paperTradesQuery.isError
    : demoExecutionQuery.isError;
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
              <CardHeader>
                <CardTitle>OKX Demo 订单</CardTitle>
                <CardDescription>
                  订单状态来自 Demo
                  执行端点。当前接口未提供已成交量和成交均价时会明确显示“不可用”。
                </CardDescription>
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
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {demoOrders.map((order) => (
                      <TableRow key={order.id}>
                        <TableCell className="whitespace-nowrap">
                          {formatDate(order.created_at)}
                        </TableCell>
                        <TableCell>{order.symbol}</TableCell>
                        <TableCell className="uppercase">
                          {order.side}
                        </TableCell>
                        <TableCell>{order.type}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{order.status}</Badge>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {order.requested_quote} USDT
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {demoOrderFilledQuantityLabel(order)}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {demoOrderAveragePriceLabel(order)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
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
                        <TableCell className="whitespace-nowrap">
                          {formatDate(trade.evaluated_at)}
                        </TableCell>
                        <TableCell className="capitalize">
                          {trade.action}
                        </TableCell>
                        <TableCell>{trade.symbol}</TableCell>
                        <TableCell className="text-right tabular-nums">
                          {trade.price.toFixed(4)}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {trade.quote_amount.toFixed(2)} USDT
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {trade.realized_pnl_quote.toFixed(2)} USDT
                        </TableCell>
                        <TableCell className="whitespace-normal break-words">
                          {trade.reason}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">
                            {trade.execution.replace("_", " ")}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )
        ) : (
          <EmptyState
            action={
              isDemo ? "配置策略" : t("saas.operations.trades.actions.evaluate")
            }
            description={
              isDemo
                ? "该策略尚无 OKX Demo 订单。纸面交易记录不会显示在此视图中。"
                : t("saas.operations.trades.empty.description")
            }
            title={
              isDemo
                ? "暂无 OKX Demo 订单"
                : t("saas.operations.trades.empty.title")
            }
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
