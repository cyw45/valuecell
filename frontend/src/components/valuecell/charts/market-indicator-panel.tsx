import { BarChart, LineChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import type { ECharts } from "echarts/core";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts/types/dist/shared";
import { useEffect, useMemo, useRef } from "react";
import { useChartResize } from "@/hooks/use-chart-resize";
import { TIME_FORMATS, TimeUtils } from "@/lib/time";
import { cn } from "@/lib/utils";
import type { CryptoCandle, CryptoIndicatorPoint } from "@/types/crypto-market";

echarts.use([
  BarChart,
  LineChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  CanvasRenderer,
]);

export type MarketIndicatorPanel = "rsi" | "bollinger" | "momentum" | "macd";
export type RsiBollingerMode = "rsi" | "bollinger" | "both";

interface MarketIndicatorPanelProps {
  data: CryptoIndicatorPoint[];
  candles?: CryptoCandle[];
  panel: MarketIndicatorPanel;
  height?: number;
  className?: string;
  theme?: "light" | "dark";
}

const PANEL_LABELS: Record<MarketIndicatorPanel, string> = {
  rsi: "RSI (14)",
  bollinger: "布林带 (20, 2)",
  momentum: "Momentum (14)",
  macd: "MACD (12, 26, 9)",
};

const finiteValue = (value: number | null | undefined): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

export function MarketIndicatorPanelChart({
  data,
  candles = [],
  panel,
  height = 220,
  className,
  theme = "light",
}: MarketIndicatorPanelProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<ECharts | null>(null);

  const option: EChartsOption = useMemo(() => {
    const dates = data.map((point) =>
      TimeUtils.formatUTC(new Date(point.ts).toISOString(), TIME_FORMATS.DATETIME_SHORT),
    );
    const textColor = theme === "dark" ? "#a8b3cf" : "#64748b";
    const gridColor = theme === "dark" ? "rgba(137, 160, 205, 0.16)" : "#e2e8f0";
    const closeByTimestamp = new Map(candles.map((candle) => [candle.ts, candle.close]));
    const bollingerValues = panel === "bollinger"
      ? data.flatMap((point) => [
          finiteValue(point.bollinger.upper),
          finiteValue(point.bollinger.middle),
          finiteValue(point.bollinger.lower),
          finiteValue(closeByTimestamp.get(point.ts)),
        ]).filter((value): value is number => value !== null)
      : [];
    const bollingerMin = bollingerValues.length > 0 ? Math.min(...bollingerValues) : undefined;
    const bollingerMax = bollingerValues.length > 0 ? Math.max(...bollingerValues) : undefined;
    const bollingerPadding = bollingerMin === undefined || bollingerMax === undefined
      ? undefined
      : Math.max((bollingerMax - bollingerMin) * 0.12, bollingerMax * 0.002);
    const series: EChartsOption["series"] = [];

    if (panel === "rsi") {
      series.push({
        name: PANEL_LABELS.rsi,
        type: "line",
        data: data.map((point) => finiteValue(point.rsi)),
        showSymbol: false,
        smooth: true,
        lineStyle: { color: "#8b5cf6", width: 2 },
        markLine: {
          symbol: "none",
          label: { color: textColor, formatter: "RSI {b}" },
          lineStyle: { color: "#a1a1aa", type: "dashed" },
          data: [
            { yAxis: 70, name: "70" },
            { yAxis: 30, name: "30" },
          ],
        },
      });
    } else if (panel === "bollinger") {
      series.push(
        {
          name: "布林区间基线",
          type: "line",
          data: data.map((point) => finiteValue(point.bollinger.lower)),
          stack: "bollinger-band",
          showSymbol: false,
          silent: true,
          tooltip: { show: false },
          lineStyle: { opacity: 0 },
          areaStyle: { opacity: 0 },
        },
        {
          name: "布林区间",
          type: "line",
          data: data.map((point) => {
            const upper = finiteValue(point.bollinger.upper);
            const lower = finiteValue(point.bollinger.lower);
            return upper === null || lower === null ? null : upper - lower;
          }),
          stack: "bollinger-band",
          showSymbol: false,
          silent: true,
          tooltip: { show: false },
          lineStyle: { opacity: 0 },
          areaStyle: { color: "rgba(6, 182, 212, 0.18)" },
        },
        {
          name: "布林上轨",
          type: "line",
          data: data.map((point) => finiteValue(point.bollinger.upper)),
          showSymbol: false,
          smooth: true,
          lineStyle: { color: "#06b6d4", type: "dashed", width: 2 },
        },
        {
          name: "布林中线",
          type: "line",
          data: data.map((point) => finiteValue(point.bollinger.middle)),
          showSymbol: false,
          smooth: true,
          lineStyle: { color: "#f59e0b", width: 1.8 },
        },
        {
          name: "布林下轨",
          type: "line",
          data: data.map((point) => finiteValue(point.bollinger.lower)),
          showSymbol: false,
          smooth: true,
          lineStyle: { color: "#06b6d4", type: "dashed", width: 2 },
        },
        {
          name: "收盘价",
          type: "line",
          data: data.map((point) => closeByTimestamp.get(point.ts) ?? null),
          showSymbol: false,
          smooth: true,
          lineStyle: { color: theme === "dark" ? "#e2e8f0" : "#334155", width: 1.2 },
        },
      );
    } else {
      const values = data.map((point) => {
        if (panel === "momentum") return finiteValue(point.momentum);
        return finiteValue(point.macd);
      });
      series.push({
        name: PANEL_LABELS[panel],
        type: "line",
        data: values,
        showSymbol: false,
        smooth: true,
        lineStyle: { color: "#0ea5e9", width: 1.8 },
        markLine: panel === "momentum"
          ? {
              symbol: "none",
              label: { show: false },
              lineStyle: { color: "#a1a1aa", type: "dashed" },
              data: [{ yAxis: 0 }],
            }
          : undefined,
      });
    }

    if (panel === "macd") {
      series.push(
        {
          name: "Signal",
          type: "line",
          data: data.map((point) => finiteValue(point.macd_signal)),
          showSymbol: false,
          smooth: true,
          lineStyle: { color: "#f59e0b", width: 1.5 },
        },
        {
          name: "Histogram",
          type: "bar",
          data: data.map((point) => finiteValue(point.macd_histogram)),
          itemStyle: {
            color: (params) => Number(params.value) >= 0 ? "#26a69a" : "#ef5350",
          },
        },
      );
    }

    return {
      animation: false,
      grid: { top: 28, right: 22, bottom: 42, left: 56 },
      legend: panel === "rsi" || panel === "bollinger" || panel === "macd"
        ? {
            data: panel === "bollinger"
              ? ["布林上轨", "布林中线", "布林下轨", "收盘价"]
              : undefined,
            top: 2,
            textStyle: { color: textColor, fontSize: 11 },
          }
        : undefined,
      tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
      xAxis: {
        type: "category",
        data: dates,
        boundaryGap: false,
        axisLine: { lineStyle: { color: gridColor } },
        axisTick: { show: false },
        axisLabel: { color: textColor, fontSize: 11 },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        min: panel === "rsi"
          ? -20
          : bollingerMin === undefined || bollingerPadding === undefined
            ? undefined
            : bollingerMin - bollingerPadding,
        max: panel === "rsi"
          ? 120
          : bollingerMax === undefined || bollingerPadding === undefined
            ? undefined
            : bollingerMax + bollingerPadding,
        axisLabel: { color: textColor, fontSize: 11 },
        splitLine: { lineStyle: { color: gridColor } },
      },
      dataZoom: [
        { type: "inside", xAxisIndex: 0 },
        { type: "slider", xAxisIndex: 0, bottom: 4, height: 18 },
      ],
      series,
    };
  }, [candles, data, panel, theme]);

  useChartResize(chartInstance);

  useEffect(() => {
    if (!chartRef.current) return;
    chartInstance.current = echarts.init(chartRef.current);
    return () => {
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, []);

  useEffect(() => {
    chartInstance.current?.setOption(option, { notMerge: true });
  }, [option]);

  return <div className={cn("w-full", className)} ref={chartRef} style={{ height }} />;
}
