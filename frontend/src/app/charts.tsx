import { useState } from "react";
import { useTheme } from "next-themes";
import { useTranslation } from "react-i18next";
import { BarChart3, ExternalLink } from "lucide-react";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import TradingViewAdvancedChart from "@/components/tradingview/tradingview-advanced-chart";

const SYMBOLS = [
  { value: "BINANCE:BTCUSDT", labelKey: "saas.charts.symbols.bitcoin" },
  { value: "BINANCE:ETHUSDT", labelKey: "saas.charts.symbols.ether" },
  { value: "BINANCE:SOLUSDT", labelKey: "saas.charts.symbols.solana" },
] as const;

export default function ChartsPage() {
  const { t } = useTranslation();
  const { resolvedTheme } = useTheme();
  const [symbol, setSymbol] = useState<string>(SYMBOLS[0].value);
  const [interval, setInterval] = useState("60");

  return (
    <div className="scroll-container size-full bg-muted/40">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-4 md:p-6 lg:p-8">
        <header className="flex flex-col gap-2">
          <div className="flex items-center gap-2"><Badge variant="secondary">{t("saas.charts.researchChart")}</Badge><Badge variant="outline">{t("saas.charts.paperOnly")}</Badge></div>
          <h1 className="text-2xl font-semibold tracking-tight">{t("saas.charts.title")}</h1>
          <p className="text-sm text-muted-foreground">{t("saas.charts.subtitle")}</p>
        </header>

        <Card>
          <CardHeader className="gap-4 sm:flex sm:flex-row sm:items-end sm:justify-between">
            <div>
              <CardTitle className="flex items-center gap-2"><BarChart3 className="size-5" /> {t("saas.charts.marketChart")}</CardTitle>
              <CardDescription className="mt-1">{t("saas.charts.poweredByTradingView")}</CardDescription>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row">
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger className="w-full sm:w-52"><SelectValue /></SelectTrigger>
                <SelectContent>{SYMBOLS.map((option) => <SelectItem key={option.value} value={option.value}>{t(option.labelKey)}</SelectItem>)}</SelectContent>
              </Select>
              <Select value={interval} onValueChange={setInterval}>
                <SelectTrigger className="w-full sm:w-32"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="15">{t("saas.charts.intervals.fifteenMinutes")}</SelectItem>
                  <SelectItem value="60">{t("saas.charts.intervals.oneHour")}</SelectItem>
                  <SelectItem value="240">{t("saas.charts.intervals.fourHours")}</SelectItem>
                  <SelectItem value="D">{t("saas.charts.intervals.oneDay")}</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent className="px-0 pb-2 sm:px-2">
            <TradingViewAdvancedChart ticker={symbol} interval={interval} minHeight={520} theme={resolvedTheme === "dark" ? "dark" : "light"} />
          </CardContent>
        </Card>

        <Card className="gap-3 py-5">
          <CardHeader className="px-5"><CardTitle className="text-base">{t("saas.charts.researchContext")}</CardTitle></CardHeader>
          <CardContent className="flex flex-col gap-3 px-5 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
            <p>{t("saas.charts.researchContextDescription")}</p>
            <Button asChild variant="outline" size="sm" className="shrink-0"><a href="https://www.tradingview.com/" rel="noreferrer" target="_blank">{t("saas.charts.openTradingView")} <ExternalLink /></a></Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
