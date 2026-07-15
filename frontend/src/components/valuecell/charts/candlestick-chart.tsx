import {
  BarChart,
  CandlestickChart as ECandlestickChart,
  LineChart,
} from "echarts/charts";
import {
  AxisPointerComponent,
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
import { useStockColors } from "@/store/settings-store";

// Register ECharts components
echarts.use([
  BarChart,
  ECandlestickChart,
  LineChart,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  AxisPointerComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

export interface CandlestickData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
export interface CandlestickMovingAverage {
  name: string;
  values: Array<number | null>;
  color?: string;
}



interface CandlestickChartProps {
  data: CandlestickData[];
  movingAverages?: CandlestickMovingAverage[];
  currentPrice?: number | null;
  width?: number | string;
  height?: number | string;
  className?: string;
  loading?: boolean;
  showVolume?: boolean;
  dateFormat?: string;
  theme?: "light" | "dark";
}

function CandlestickChart({
  data,
  movingAverages = [],
  currentPrice,
  width = "100%",
  height = 500,
  className,
  loading,
  showVolume = true,
  dateFormat = TIME_FORMATS.DATETIME_SHORT,
  theme = "light",
}: CandlestickChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<ECharts | null>(null);
  const stockColors = useStockColors();

  const option: EChartsOption = useMemo(() => {
    if (!data || data.length === 0) return {};

    const dates = data.map((item) =>
      TimeUtils.formatUTC(item.time, dateFormat),
    );
    const ohlcData = data.map((item) => [
      item.open,
      item.close,
      item.low,
      item.high,
    ]);
    const volumes = data.map((item, index) => [
      index,
      item.volume,
      item.close >= item.open ? 1 : -1, // 1 = up (positive), -1 = down (negative)
    ]);

    const mainGridHeight = showVolume ? "50%" : "75%";
    const volumeGridTop = showVolume ? "63%" : undefined;

    const series: EChartsOption["series"] = [
      {
        name: "K线",
        type: "candlestick",
        data: ohlcData,
        clip: true,
        itemStyle: {
          color: stockColors.positive, // up candle fill
          color0: stockColors.negative, // down candle fill
          borderColor: stockColors.positive, // up candle border
          borderColor0: stockColors.negative, // down candle border
        },
        markLine: currentPrice != null
          ? {
              symbol: "none",
              lineStyle: { color: "#fbbf24", type: "dashed", width: 1 },
              label: {
                color: theme === "dark" ? "#fef3c7" : "#78350f",
                formatter: `现价 ${currentPrice.toLocaleString()}`,
                position: "end",
              },
              data: [{ yAxis: currentPrice }],
            }
          : undefined,
      },
    ];
    for (const movingAverage of movingAverages) {
      series.push({
        name: movingAverage.name,
        type: "line",
        data: movingAverage.values,
        showSymbol: false,
        smooth: true,
        lineStyle: { width: 1.5, color: movingAverage.color },
      });
    }

    if (showVolume) {
      series.push({
        name: "成交量",
        type: "bar",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: volumes,
        itemStyle: {
          color: (params) => {
            const value = params.value as [number, number, number];
            // value[2]: 1 = up (positive color), -1 = down (negative color)
            return value[2] === 1 ? stockColors.positive : stockColors.negative;
          },
        },
      });
    }

    const grids: EChartsOption["grid"] = [
      {
        left: "4%",
        right: "4%",
        top: "3%",
        height: mainGridHeight,
        containLabel: true,
      },
    ];

    const xAxes: EChartsOption["xAxis"] = [
      {
        type: "category" as const,
        data: dates,
        boundaryGap: true,
        axisLine: { onZero: false },
        splitLine: { show: false },
        min: "dataMin",
        max: "dataMax",
        axisPointer: { z: 100 },
      },
    ];

    const yAxes: EChartsOption["yAxis"] = [
      {
        scale: true,
        splitArea: { show: false },
        splitLine: { show: false },
      },
    ];

    if (showVolume) {
      grids.push({
        left: "8%",
        right: "6%",
        top: volumeGridTop,
        height: "16%",
        containLabel: true,
      });

      xAxes.push({
        type: "category" as const,
        gridIndex: 1,
        data: dates,
        boundaryGap: true,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: "dataMin",
        max: "dataMax",
      });

      yAxes.push({
        scale: true,
        gridIndex: 1,
        splitNumber: 2,
        axisLabel: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
      });
    }

    const textColor = theme === "dark" ? "#a8b3cf" : "#566075";
    const gridColor = theme === "dark" ? "rgba(137, 160, 205, 0.12)" : "rgba(71, 85, 105, 0.12)";
    return {
      animation: false,
      legend: {
        top: 2,
        left: "center",
        textStyle: { color: textColor, fontSize: 10 },
        itemWidth: 14,
        itemHeight: 2,
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        borderWidth: 1,
        borderColor: gridColor,
        padding: 10,
        backgroundColor: theme === "dark" ? "#101828" : "#ffffff",
        textStyle: { color: textColor },
      },
      axisPointer: {
        link: [{ xAxisIndex: "all" }],
        label: { backgroundColor: theme === "dark" ? "#34405c" : "#64748b" },
      },
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: showVolume ? [0, 1] : [0],
        },
        {
          show: true,
          xAxisIndex: showVolume ? [0, 1] : [0],
          type: "slider",
          top: "85%",
        },
      ],
      series,
    };
  }, [data, movingAverages, currentPrice, stockColors, showVolume, dateFormat, theme]);

  useChartResize(chartInstance);

  useEffect(() => {
    if (!chartRef.current) return;

    chartInstance.current = echarts.init(chartRef.current);
    chartInstance.current.setOption(option);

    return () => {
      chartInstance.current?.dispose();
    };
  }, [option]);

  useEffect(() => {
    if (chartInstance.current) {
      chartInstance.current.setOption(option);
    }
  }, [option]);

  useEffect(() => {
    if (chartInstance.current) {
      if (loading) {
        chartInstance.current.showLoading();
      } else {
        chartInstance.current.hideLoading();
      }
    }
  }, [loading]);

  return (
    <div
      ref={chartRef}
      className={cn("w-fit", className)}
      style={{ width, height }}
    />
  );
}

export default CandlestickChart;
