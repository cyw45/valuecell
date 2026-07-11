import { ClipboardList, FileClock, ShieldCheck } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import { useRuleStrategyTrades } from "@/api/rule-strategy";
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

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

export default function TradesPage() {
  const { t } = useTranslation();
  const strategyId = localStorage.getItem("valuecell.rule-strategy-id") ?? "";
  const { data: trades, isLoading, isError } = useRuleStrategyTrades(strategyId);
  const tradeEntries = trades ?? [];
  const hasTrades = tradeEntries.length > 0;

  return (
    <div className="scroll-container size-full bg-muted/40">
      <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col gap-6 p-4 md:p-6 lg:p-8">
        <header><div className="mb-2 flex gap-2"><Badge variant="secondary">{t("saas.operations.trades.paperOnly")}</Badge><Badge variant="outline">{t("saas.operations.trades.noExecutionAccount")}</Badge></div><h1 className="text-2xl font-semibold tracking-tight">{t("saas.operations.trades.title")}</h1><p className="mt-1 text-sm text-muted-foreground">{t("saas.operations.trades.subtitle")}</p></header>

        {!strategyId ? (
          <Card className="flex flex-1 items-center justify-center border-dashed">
            <CardHeader className="max-w-md items-center text-center">
              <div className="mb-2 flex size-12 items-center justify-center rounded-full bg-secondary"><ClipboardList className="size-6 text-muted-foreground" /></div>
              <CardTitle>{t("saas.operations.trades.noStrategy.title")}</CardTitle>
              <CardDescription>{t("saas.operations.trades.noStrategy.description")}</CardDescription>
            </CardHeader>
            <CardContent className="flex justify-center pb-8"><Button asChild><Link to="/strategies">{t("saas.operations.trades.actions.configure")}</Link></Button></CardContent>
          </Card>
        ) : isLoading ? (
          <Card><CardContent className="p-6 text-sm text-muted-foreground">{t("saas.operations.trades.loading")}</CardContent></Card>
        ) : isError ? (
          <Card><CardContent className="p-6 text-sm text-muted-foreground">{t("saas.operations.trades.unavailable")}</CardContent></Card>
        ) : hasTrades ? (
          <Card>
            <CardHeader>
              <CardTitle>{t("saas.operations.trades.recommendations.title")}</CardTitle>
              <CardDescription>{t("saas.operations.trades.recommendations.description")}</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("saas.operations.trades.table.evaluated")}</TableHead>
                    <TableHead>{t("saas.operations.trades.table.action")}</TableHead>
                    <TableHead>{t("saas.operations.trades.table.reason")}</TableHead>
                    <TableHead className="text-right">{t("saas.operations.trades.table.quantity")}</TableHead>
                    <TableHead>{t("saas.operations.trades.table.execution")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tradeEntries.map((trade) => (
                    <TableRow key={`${trade.evaluation_id}-${trade.action}`}>
                      <TableCell className="whitespace-nowrap">{formatDate(trade.evaluated_at)}</TableCell>
                      <TableCell className="capitalize">{trade.action}</TableCell>
                      <TableCell className="whitespace-normal break-words">{trade.reason}</TableCell>
                      <TableCell className="text-right tabular-nums">{trade.sizing.quantity}</TableCell>
                      <TableCell><Badge variant="outline">{trade.execution.replace("_", " ")}</Badge></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        ) : (
          <Card className="flex flex-1 items-center justify-center border-dashed">
            <CardHeader className="max-w-md items-center text-center">
              <div className="mb-2 flex size-12 items-center justify-center rounded-full bg-secondary"><ClipboardList className="size-6 text-muted-foreground" /></div>
              <CardTitle>{t("saas.operations.trades.empty.title")}</CardTitle>
              <CardDescription>{t("saas.operations.trades.empty.description")}</CardDescription>
            </CardHeader>
            <CardContent className="flex justify-center pb-8"><Button asChild variant="outline"><Link to="/strategies">{t("saas.operations.trades.actions.evaluate")}</Link></Button></CardContent>
          </Card>
        )}

        <section className="grid gap-4 md:grid-cols-2">
          <Card className="gap-3 py-5">
            <CardHeader className="px-5"><div className="flex items-center gap-2"><FileClock className="size-4 text-muted-foreground" /><CardTitle className="text-base">{t("saas.operations.trades.ledger.title")}</CardTitle></div></CardHeader>
            <CardContent className="px-5 text-sm text-muted-foreground">{t("saas.operations.trades.ledger.description")}</CardContent>
          </Card>
          <Card className="gap-3 py-5">
            <CardHeader className="px-5"><div className="flex items-center gap-2"><ShieldCheck className="size-4 text-muted-foreground" /><CardTitle className="text-base">{t("saas.operations.trades.separation.title")}</CardTitle></div></CardHeader>
            <CardContent className="px-5 text-sm text-muted-foreground">{t("saas.operations.trades.separation.description")}</CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}
