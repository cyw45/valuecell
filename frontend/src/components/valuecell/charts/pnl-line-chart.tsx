import { LineChart } from "echarts/charts";
import { GridComponent, TooltipComponent } from "echarts/components";
import type { ECharts } from "echarts/core";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts/types/dist/shared";
import { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { useChartResize } from "@/hooks/use-chart-resize";
import type { RuleStrategyPnlPoint } from "@/types/rule-strategy";

echarts.use([LineChart, GridComponent, TooltipComponent, CanvasRenderer]);

interface PnlLineChartProps {
  data: RuleStrategyPnlPoint[];
  height?: number;
  theme?: "light" | "dark";
}


export function PnlLineChart({ data, height = 200, theme = "light" }: PnlLineChartProps) {
  const { t, i18n } = useTranslation();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<ECharts | null>(null);
  const locale = i18n.language.replace("_", "-");

  const option: EChartsOption = useMemo(() => {
    const textColor = theme === "dark" ? "#a1a1aa" : "#71717a";
    const lineColor = theme === "dark" ? "#6366f1" : "#4f46e5";
    const axisLineColor = theme === "dark" ? "#3f3f46" : "#e4e4e7";

    return {
      backgroundColor: "transparent",
      grid: { top: 12, right: 16, bottom: 32, left: 60, containLabel: false },
      tooltip: {
        trigger: "axis",
        formatter: (params: unknown) => {
          const p = (params as Array<{ name: string; value: number }>)[0];
          return `${p.name}<br/>${t("saas.chart.pnl")}: ${p.value >= 0 ? "+" : ""}${p.value.toFixed(4)}`;
        },
      },
      xAxis: {
        type: "category",
        data: data.map((d) => new Intl.DateTimeFormat(locale, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(d.ts))),
        axisLine: { lineStyle: { color: axisLineColor } },
        axisTick: { show: false },
        axisLabel: { color: textColor, fontSize: 11, interval: "auto" },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: textColor, fontSize: 11 },
        splitLine: { lineStyle: { color: axisLineColor } },
      },
      series: [
        {
          type: "line",
          data: data.map((d) => d.cumulative_pnl),
          smooth: true,
          symbol: "none",
          lineStyle: { color: lineColor, width: 2 },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: theme === "dark" ? "rgba(99,102,241,0.3)" : "rgba(79,70,229,0.2)" },
              { offset: 1, color: "rgba(0,0,0,0)" },
            ]),
          },
        },
      ],
    };
  }, [data, locale, theme, t]);

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

  return <div ref={chartRef} style={{ width: "100%", height }} />;
}
