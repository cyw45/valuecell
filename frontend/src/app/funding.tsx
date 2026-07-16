import { Landmark, LockKeyhole } from "lucide-react";
import { useTheme } from "next-themes";
import { useTranslation } from "react-i18next";
import {
  useRuleStrategyFunding,
  useRuleStrategyPnlCurve,
} from "@/api/rule-strategy";
import { Badge } from "@/components/ui/badge";
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
import { PnlLineChart } from "@/components/valuecell/charts/pnl-line-chart";

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export default function FundingPage() {
  const { t } = useTranslation();
  const { resolvedTheme } = useTheme();
  const strategyId = localStorage.getItem("valuecell.rule-strategy-id") ?? "";
  const {
    data: funding,
    isLoading,
    isError,
  } = useRuleStrategyFunding(strategyId);
  const { data: pnlCurve } = useRuleStrategyPnlCurve(strategyId || undefined);
  const fundingEntries = funding ?? [];
  const pnlPoints = pnlCurve ?? [];
  const hasFunding = fundingEntries.length > 0;

  return (
    <div className="scroll-container size-full bg-muted/40">
      <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col gap-6 p-4 md:p-6 lg:p-8">
        <Card>
          <CardHeader>
            <CardTitle>{t("saas.operations.funding.pnl.title")}</CardTitle>
            <CardDescription>
              {t("saas.operations.funding.pnl.description")}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {pnlPoints.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                {t("saas.operations.funding.pnl.empty")}
              </p>
            ) : (
              <PnlLineChart
                data={pnlPoints}
                height={200}
                theme={resolvedTheme === "dark" ? "dark" : "light"}
              />
            )}
          </CardContent>
        </Card>

        {!strategyId ? (
          <Card className="border-dashed">
            <CardContent className="p-6 text-sm text-muted-foreground">
              {t("saas.operations.funding.noStrategy")}
            </CardContent>
          </Card>
        ) : isLoading ? (
          <Card>
            <CardContent className="p-6 text-sm text-muted-foreground">
              {t("saas.operations.funding.loading")}
            </CardContent>
          </Card>
        ) : isError ? (
          <Card>
            <CardContent className="p-6 text-sm text-muted-foreground">
              {t("saas.operations.funding.unavailable")}
            </CardContent>
          </Card>
        ) : hasFunding ? (
          <Card>
            <CardHeader>
              <CardTitle>{t("saas.operations.funding.impact.title")}</CardTitle>
              <CardDescription>
                {t("saas.operations.funding.impact.description")}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>
                      {t("saas.operations.funding.table.evaluated")}
                    </TableHead>
                    <TableHead>
                      {t("saas.operations.funding.table.direction")}
                    </TableHead>
                    <TableHead className="text-right">
                      {t("saas.operations.funding.table.rate")}
                    </TableHead>
                    <TableHead className="text-right">
                      {t("saas.operations.funding.table.currentNotional")}
                    </TableHead>
                    <TableHead className="text-right">
                      {t("saas.operations.funding.table.estimatedPayment")}
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {fundingEntries.map((entry) => (
                    <TableRow key={entry.evaluation_id}>
                      <TableCell>{formatDate(entry.evaluated_at)}</TableCell>
                      <TableCell className="capitalize">
                        <Badge variant="outline">{entry.direction}</Badge>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {(entry.funding_rate * 100).toFixed(4)}%
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {entry.current_notional_quote.toFixed(4)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {entry.estimated_payment_quote.toFixed(4)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        ) : (
          <Card className="border-dashed">
            <CardContent className="p-6 text-sm text-muted-foreground">
              {t("saas.operations.funding.empty")}
            </CardContent>
          </Card>
        )}

        <section className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <div className="flex size-11 items-center justify-center rounded-lg bg-secondary">
                <Landmark className="size-5" />
              </div>
              <CardTitle className="mt-2">
                {t("saas.operations.funding.connections.title")}
              </CardTitle>
              <CardDescription>
                {t("saas.operations.funding.connections.description")}
              </CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {t("saas.operations.funding.connections.empty")}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <div className="flex size-11 items-center justify-center rounded-lg bg-secondary">
                <LockKeyhole className="size-5" />
              </div>
              <CardTitle className="mt-2">
                {t("saas.operations.funding.safety.title")}
              </CardTitle>
              <CardDescription>
                {t("saas.operations.funding.safety.description")}
              </CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {t("saas.operations.funding.safety.body")}
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}
