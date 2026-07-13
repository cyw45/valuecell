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
import type { CryptoIndicatorPoint } from "@/types/crypto-market";

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

export type MarketIndicatorPanel = "rsi" | "momentum" | "macd";

interface MarketIndicatorPanelProps {
  data: CryptoIndicatorPoint[];
  panel: MarketIndicatorPanel;
  height?: number;
  className?: string;
  theme?: "light" | "dark";
}

const PANEL_LABELS: Record<MarketIndicatorPanel, string> = {
  rsi: "RSI (14)",
  momentum: "Momentum (14)",
  macd: "MACD (12, 26, 9)",
};

const finiteValue = (value: number | null | undefined): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

export function MarketIndicatorPanelChart({
  data,
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
    const values = data.map((point) => {
      if (panel === "rsi") return finiteValue(point.rsi);
      if (panel === "momentum") return finiteValue(point.momentum);
      return finiteValue(point.macd);
    });

    const series: EChartsOption["series"] = [
      {
        name: PANEL_LABELS[panel],
        type: "line",
        data: values,
        showSymbol: false,
        smooth: true,
        lineStyle: { color: panel === "rsi" ? "#8b5cf6" : "#0ea5e9", width: 1.8 },
        markLine: panel === "rsi"
          ? {
              symbol: "none",
              label: { color: textColor, formatter: "{b}" },
              lineStyle: { color: "#a1a1aa", type: "dashed" },
              data: [
                { yAxis: 70, name: "Overbought" },
                { yAxis: 30, name: "Oversold" },
              ],
            }
          : panel === "momentum"
            ? {
                symbol: "none",
                label: { show: false },
                lineStyle: { color: "#a1a1aa", type: "dashed" },
                data: [{ yAxis: 0 }],
              }
            : undefined,
      },
    ];

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
      legend: panel === "macd"
        ? { top: 2, textStyle: { color: textColor, fontSize: 11 } }
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
        min: panel === "rsi" ? 0 : undefined,
        max: panel === "rsi" ? 100 : undefined,
        axisLabel: { color: textColor, fontSize: 11 },
        splitLine: { lineStyle: { color: gridColor } },
      },
      dataZoom: [
        { type: "inside", xAxisIndex: 0 },
        { type: "slider", xAxisIndex: 0, bottom: 4, height: 18 },
      ],
      series,
    };
  }, [data, panel, theme]);

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
