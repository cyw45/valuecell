import { LineChart } from "echarts/charts";
import {
  AxisPointerComponent,
  DataZoomComponent,
  GridComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import type { ECharts } from "echarts/core";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts/types/dist/shared";
import { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { useChartResize } from "@/hooks/use-chart-resize";
import type { RuleStrategyPnlPoint } from "@/types/rule-strategy";

echarts.use([
  LineChart,
  AxisPointerComponent,
  DataZoomComponent,
  GridComponent,
  MarkLineComponent,
  TooltipComponent,
  CanvasRenderer,
]);

type EquityCurvePoint = RuleStrategyPnlPoint & { equity_quote?: number };

interface PnlLineChartProps {
  data: EquityCurvePoint[];
  height?: number;
  mode?: "pnl" | "equity";
  theme?: "light" | "dark";
}

export function PnlLineChart({
  data,
  height = 200,
  mode = "pnl",
  theme = "light",
}: PnlLineChartProps) {
  const { t, i18n } = useTranslation();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<ECharts | null>(null);
  const locale = i18n.language.replace("_", "-");

  const option: EChartsOption = useMemo(() => {
    const isEquity = mode === "equity";
    const textColor = theme === "dark" ? "#a1a1aa" : "#71717a";
    const lineColor = isEquity
      ? theme === "dark"
        ? "#2dd4bf"
        : "#0f766e"
      : theme === "dark"
        ? "#818cf8"
        : "#4f46e5";
    const axisLineColor = theme === "dark" ? "#3f3f46" : "#e4e4e7";
    const areaColor = isEquity
      ? theme === "dark"
        ? "rgba(45,212,191,0.28)"
        : "rgba(13,148,136,0.2)"
      : theme === "dark"
        ? "rgba(129,140,248,0.3)"
        : "rgba(79,70,229,0.2)";
    const initialEquity = data.find(
      (point) => point.action === "initial",
    )?.equity_quote;
    const seriesName = isEquity ? "组合权益" : t("saas.chart.pnl");
    const currencyFormatter = new Intl.NumberFormat(locale, {
      maximumFractionDigits: 2,
    });

    return {
      animation: false,
      backgroundColor: "transparent",
      grid: { top: 16, right: 20, bottom: 58, left: 64 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        borderColor: axisLineColor,
        backgroundColor: theme === "dark" ? "#18181b" : "#ffffff",
        textStyle: { color: textColor },
        formatter: (params: unknown) => {
          const point = (
            params as Array<{ dataIndex: number; name: string; value: number }>
          )[0];
          const value = Number(point?.value);
          if (!Number.isFinite(value))
            return `${point?.name ?? ""}<br/>${seriesName}: —`;
          if (!isEquity)
            return `${point.name}<br/>${seriesName}: ${value >= 0 ? "+" : ""}${value.toFixed(4)}`;

          const cumulativePnl = data[point.dataIndex]?.cumulative_pnl;
          const cumulativePnlLabel =
            cumulativePnl === undefined
              ? "—"
              : `${cumulativePnl >= 0 ? "+" : ""}${currencyFormatter.format(cumulativePnl)} USDT`;
          return `${point.name}<br/>组合权益: ${currencyFormatter.format(value)} USDT<br/>累计盈亏: ${cumulativePnlLabel}`;
        },
      },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: data.map((point) =>
          new Intl.DateTimeFormat(locale, {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          }).format(new Date(point.ts)),
        ),
        axisLine: { lineStyle: { color: axisLineColor } },
        axisTick: { show: false },
        axisLabel: { color: textColor, fontSize: 11, interval: "auto" },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: textColor,
          fontSize: 11,
          formatter: (value: number) => currencyFormatter.format(value),
        },
        splitLine: { lineStyle: { color: axisLineColor, type: "dashed" } },
      },
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: 0,
          filterMode: "none",
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
          moveOnMouseWheel: true,
        },
        {
          type: "slider",
          xAxisIndex: 0,
          bottom: 8,
          height: 28,
          borderColor: axisLineColor,
          backgroundColor: theme === "dark" ? "#27272a" : "#f4f4f5",
          fillerColor: isEquity
            ? theme === "dark"
              ? "rgba(45,212,191,0.3)"
              : "rgba(13,148,136,0.22)"
            : theme === "dark"
              ? "rgba(129,140,248,0.3)"
              : "rgba(79,70,229,0.22)",
          handleSize: 28,
          handleStyle: { color: lineColor, borderColor: lineColor },
          moveHandleSize: 28,
          showDetail: false,
        },
      ],
      series: [
        {
          name: seriesName,
          type: "line",
          data: data.map((point) =>
            isEquity ? (point.equity_quote ?? null) : point.cumulative_pnl,
          ),
          smooth: true,
          connectNulls: false,
          symbol: "none",
          lineStyle: { color: lineColor, width: 2 },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: areaColor },
              { offset: 1, color: "rgba(0,0,0,0)" },
            ]),
          },
          markLine:
            isEquity && initialEquity !== undefined
              ? {
                  symbol: "none",
                  lineStyle: { color: "#f59e0b", type: "dashed", width: 1.5 },
                  label: {
                    formatter: "初始资金",
                    color: theme === "dark" ? "#fcd34d" : "#92400e",
                    position: "insideEndTop",
                  },
                  data: [{ yAxis: initialEquity }],
                }
              : undefined,
        },
      ],
    };
  }, [data, locale, mode, theme, t]);

  useChartResize(chartInstance);

  useEffect(() => {
    if (!chartRef.current) return;
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }
    chartInstance.current.setOption(option, { notMerge: true });
  }, [option]);

  useEffect(() => {
    return () => {
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, []);

  return (
    <div
      aria-label={
        mode === "equity"
          ? "策略权益曲线。可使用滚轮、拖动或下方滑块缩放和浏览时间范围。"
          : t("saas.chart.pnl")
      }
      ref={chartRef}
      role="img"
      style={{ width: "100%", height }}
    />
  );
}
