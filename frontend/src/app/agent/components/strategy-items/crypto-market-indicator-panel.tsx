import { CandlestickChart, LineChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from "echarts/components";
import type { ECharts } from "echarts/core";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts/types/dist/shared";
import { Activity, AlertTriangle, RadioTower } from "lucide-react";
import { type FC, useEffect, useMemo, useRef, useState } from "react";
import {
  useGetCryptoMarketIndicators,
  useGetCryptoSymbols,
} from "@/api/crypto-market";
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
import { useChartResize } from "@/hooks/use-chart-resize";
import { TIME_FORMATS, TimeUtils } from "@/lib/time";
import { cn, numberFixed } from "@/lib/utils";
import {
  getMarketDataRefreshIntervalMs,
  type MarketDataRefreshMode,
  useMarketDataRefreshMode,
  useSettingsActions,
} from "@/store/settings-store";
import type { CryptoSymbolIndicators } from "@/types/crypto-market";

echarts.use([
  CandlestickChart,
  LineChart,
  DataZoomComponent,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer,
]);

const INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"];
const REFRESH_OPTIONS: Array<{ value: MarketDataRefreshMode; label: string }> =
  [
    { value: "manual", label: "手动" },
    { value: "5s", label: "5秒" },
    { value: "15s", label: "15秒" },
    { value: "30s", label: "30秒" },
    { value: "1m", label: "1分钟" },
    { value: "5m", label: "5分钟" },
  ];
const DEFAULT_SYMBOL = "BTC-USDT";

interface CryptoMarketIndicatorPanelProps {
  strategySymbols?: string[];
  strategyRefreshIntervalSeconds?: number;
}

const buildChartOption = (data?: CryptoSymbolIndicators): EChartsOption => {
  const candles = data?.candles ?? [];
  const indicators = data?.indicators ?? [];
  const categories = candles.map((item) =>
    TimeUtils.formatUTC(item.ts, TIME_FORMATS.MARKET),
  );
  const candleValues = candles.map((item) => [
    item.open,
    item.close,
    item.low,
    item.high,
  ]);
  const closeValues = candles.map((item) => item.close);
  const volumes = candles.map((item) => item.volume);
  const indicatorByTs = new Map(indicators.map((item) => [item.ts, item]));
  const seriesFor = (
    selector: (
      point: NonNullable<CryptoSymbolIndicators["indicators"]>[number],
    ) => number | null | undefined,
  ) =>
    candles.map((item) => {
      const point = indicatorByTs.get(item.ts);
      const value = point ? selector(point) : null;
      return value ?? null;
    });

  return {
    animation: false,
    legend: {
      top: 0,
      type: "scroll",
      textStyle: { color: "#64748b" },
    },
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    grid: [
      { left: 52, right: 24, top: 34, height: "43%" },
      { left: 52, right: 24, top: "55%", height: "15%" },
      { left: 52, right: 24, top: "74%", height: "14%" },
      { left: 52, right: 24, top: "91%", height: "7%" },
    ],
    xAxis: [
      {
        type: "category",
        data: categories,
        boundaryGap: true,
        axisLabel: { show: false },
      },
      {
        type: "category",
        data: categories,
        gridIndex: 1,
        axisLabel: { show: false },
      },
      {
        type: "category",
        data: categories,
        gridIndex: 2,
        axisLabel: { show: false },
      },
      { type: "category", data: categories, gridIndex: 3 },
    ],
    yAxis: [
      { scale: true },
      { scale: true, gridIndex: 1, min: 0, max: 100 },
      { scale: true, gridIndex: 2 },
      { scale: true, gridIndex: 3 },
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1, 2, 3], start: 55, end: 100 },
      {
        type: "slider",
        xAxisIndex: [0, 1, 2, 3],
        bottom: 0,
        height: 18,
        start: 55,
        end: 100,
      },
    ],
    series: [
      {
        name: "K线",
        type: "candlestick",
        data: candleValues,
        itemStyle: {
          color: "#26A69A",
          color0: "#EF5350",
          borderColor: "#26A69A",
          borderColor0: "#EF5350",
        },
      },
      {
        name: "Close",
        type: "line",
        data: closeValues,
        showSymbol: false,
        smooth: true,
      },
      {
        name: "MA5",
        type: "line",
        data: seriesFor((item) => item.ma.ma5),
        showSymbol: false,
      },
      {
        name: "MA20",
        type: "line",
        data: seriesFor((item) => item.ma.ma20),
        showSymbol: false,
      },
      {
        name: "MA60",
        type: "line",
        data: seriesFor((item) => item.ma.ma60),
        showSymbol: false,
      },
      {
        name: "BB Upper",
        type: "line",
        data: seriesFor((item) => item.bollinger.upper),
        showSymbol: false,
        lineStyle: { type: "dashed" },
      },
      {
        name: "BB Mid",
        type: "line",
        data: seriesFor((item) => item.bollinger.middle),
        showSymbol: false,
        lineStyle: { type: "dashed" },
      },
      {
        name: "BB Lower",
        type: "line",
        data: seriesFor((item) => item.bollinger.lower),
        showSymbol: false,
        lineStyle: { type: "dashed" },
      },
      {
        name: "RSI",
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: seriesFor((item) => item.rsi),
        showSymbol: false,
        smooth: true,
      },
      {
        name: "Momentum",
        type: "line",
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: seriesFor((item) => item.momentum),
        showSymbol: false,
        smooth: true,
      },
      {
        name: "MACD",
        type: "line",
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: seriesFor((item) => item.macd),
        showSymbol: false,
      },
      {
        name: "MACD Signal",
        type: "line",
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: seriesFor((item) => item.macd_signal),
        showSymbol: false,
      },
      {
        name: "Volume",
        type: "line",
        xAxisIndex: 3,
        yAxisIndex: 3,
        data: volumes,
        showSymbol: false,
        lineStyle: { opacity: 0.45 },
      },
    ],
  };
};

const IndicatorChart: FC<{ data?: CryptoSymbolIndicators }> = ({ data }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<ECharts | null>(null);
  const option = useMemo(() => buildChartOption(data), [data]);
  useChartResize(chartInstance, chartRef, [option]);

  useEffect(() => {
    if (!chartRef.current) return;
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }
    chartInstance.current.setOption(option, true);
    return () => {
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, [option]);

  return <div ref={chartRef} className="h-[520px] min-h-[520px] w-full" />;
};

const normalizeSymbols = (symbols?: string[]) =>
  (symbols ?? [])
    .map((item) => item.trim().toUpperCase().replace("/", "-"))
    .filter((item) => item.endsWith("-USDT"));

const CryptoMarketIndicatorPanel: FC<CryptoMarketIndicatorPanelProps> = ({
  strategySymbols,
  strategyRefreshIntervalSeconds,
}) => {
  const marketDataRefreshMode = useMarketDataRefreshMode();
  const { setMarketDataRefreshMode } = useSettingsActions();
  const normalizedStrategySymbols = useMemo(
    () => normalizeSymbols(strategySymbols),
    [strategySymbols],
  );
  const { data: catalog } = useGetCryptoSymbols();
  const allSymbols = catalog?.symbols ?? [];
  const preferredSymbols =
    normalizedStrategySymbols.length > 0
      ? normalizedStrategySymbols
      : [DEFAULT_SYMBOL];
  const [symbol, setSymbol] = useState(preferredSymbols[0] ?? DEFAULT_SYMBOL);
  const [interval, setInterval] = useState("15m");

  useEffect(() => {
    if (!preferredSymbols.includes(symbol)) {
      setSymbol(preferredSymbols[0] ?? DEFAULT_SYMBOL);
    }
  }, [preferredSymbols, symbol]);

  const { data, isFetching, error } = useGetCryptoMarketIndicators({
    symbols: [symbol],
    interval,
    lookback: 240,
    enabled: Boolean(symbol),
    refreshIntervalSeconds: strategyRefreshIntervalSeconds,
  });
  const current = data?.symbols[0];
  const latestIndicator = current?.indicators[current.indicators.length - 1];
  const failedReason = data?.failed_symbols?.[symbol];
  const effectiveRefreshInterval = getMarketDataRefreshIntervalMs(
    marketDataRefreshMode,
    strategyRefreshIntervalSeconds,
  );

  return (
    <div className="scroll-container flex min-w-0 flex-col gap-4 overflow-y-auto p-4">
      <Card className="overflow-hidden">
        <CardHeader className="gap-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="size-4" />
                行情指标工作台
              </CardTitle>
              <CardDescription>
                加密货币 USDT 交易对，实时拉取 K线并计算
                MA、RSI、布林带、动能/MACD。
              </CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant={isFetching ? "outline" : "secondary"}
                className="gap-1"
              >
                <RadioTower className="size-3" />
                {current?.provider ?? "provider pool"}
              </Badge>
              {failedReason && (
                <Badge variant="destructive" className="gap-1">
                  <AlertTriangle className="size-3" />
                  fallback failed
                </Badge>
              )}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {(preferredSymbols.length > 0 ? preferredSymbols : allSymbols)
              .slice(0, 18)
              .map((item) => (
                <Button
                  key={item}
                  type="button"
                  variant={symbol === item ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSymbol(item)}
                >
                  {item.replace("-", "/")}
                </Button>
              ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {INTERVALS.map((item) => (
              <Button
                key={item}
                type="button"
                variant={interval === item ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setInterval(item)}
              >
                {item}
              </Button>
            ))}
            <div className="ml-auto flex items-center gap-2">
              {strategyRefreshIntervalSeconds ? (
                <Badge variant="outline">
                  按策略 {Math.max(strategyRefreshIntervalSeconds, 5)}秒刷新
                </Badge>
              ) : (
                <>
                  <span className="text-muted-foreground text-xs">刷新</span>
                  <Select
                    value={marketDataRefreshMode}
                    onValueChange={(value) =>
                      setMarketDataRefreshMode(value as MarketDataRefreshMode)
                    }
                  >
                    <SelectTrigger className="h-8 w-24 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {REFRESH_OPTIONS.map((item) => (
                        <SelectItem key={item.value} value={item.value}>
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </>
              )}
              <Badge variant="secondary">
                {effectiveRefreshInterval === false
                  ? "手动更新"
                  : `${Math.round(effectiveRefreshInterval / 1000)}秒`}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="text-muted-foreground text-xs">最新价</p>
              <p className="mt-1 font-semibold text-lg">
                {numberFixed(current?.latest_price ?? undefined, 4)}
              </p>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="text-muted-foreground text-xs">RSI</p>
              <p className="mt-1 font-semibold text-lg">
                {numberFixed(latestIndicator?.rsi ?? undefined, 2)}
              </p>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="text-muted-foreground text-xs">Momentum</p>
              <p className="mt-1 font-semibold text-lg">
                {numberFixed(latestIndicator?.momentum ?? undefined, 4)}
              </p>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="text-muted-foreground text-xs">MACD Hist</p>
              <p className="mt-1 font-semibold text-lg">
                {numberFixed(latestIndicator?.macd_histogram ?? undefined, 4)}
              </p>
            </div>
          </div>
          <div
            className={cn(
              "rounded-lg border bg-background",
              !current &&
                "flex h-[520px] items-center justify-center text-muted-foreground",
            )}
          >
            {current ? (
              <IndicatorChart data={current} />
            ) : error ? (
              "行情暂不可用，请稍后重试。"
            ) : (
              "等待行情数据..."
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CryptoMarketIndicatorPanel;
