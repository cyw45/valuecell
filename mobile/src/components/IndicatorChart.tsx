import { useMemo, useState } from "react";
import { LayoutChangeEvent, StyleSheet, Text, View } from "react-native";
import Svg, { Line, Path, Rect } from "react-native-svg";
import type { CryptoCandle, CryptoIndicatorPoint } from "../types";
import { CHART_HORIZONTAL_INSETS, type ChartWindow } from "./CandlestickChart";
import { palette, radius, spacing } from "../theme";

export type IndicatorPanel = "rsi" | "bollinger" | "momentum" | "macd";

type Props = {
  panel: IndicatorPanel;
  candles: CryptoCandle[];
  indicators: CryptoIndicatorPoint[];
  window?: ChartWindow;
  selectedTimestamp?: CryptoCandle["ts"] | null;
  height?: number;
};

const PADDING = { top: 28, ...CHART_HORIZONTAL_INSETS, bottom: 24 };

function finite(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeWindow(window: ChartWindow | undefined, candleCount: number): ChartWindow {
  if (!window) return { start: 0, end: candleCount };
  const start = Math.max(0, Math.min(window.start, candleCount));
  const end = Math.max(start, Math.min(window.end, candleCount));
  return { start, end };
}

function linePath(
  values: Array<number | null>,
  xAt: (index: number) => number,
  yAt: (value: number) => number,
) {
  let path = "";
  let inSegment = false;

  values.forEach((value, index) => {
    if (value == null) {
      inSegment = false;
      return;
    }
    path += `${inSegment ? "L" : "M"}${xAt(index).toFixed(2)},${yAt(value).toFixed(2)} `;
    inSegment = true;
  });

  return path;
}

function bounds(values: Array<number | null>, includeZero = false) {
  const finiteValues = values.filter((value): value is number => value != null);
  if (includeZero) finiteValues.push(0);
  if (!finiteValues.length) return null;

  const low = Math.min(...finiteValues);
  const high = Math.max(...finiteValues);
  const padding = Math.max((high - low) * 0.12, Math.max(Math.abs(high), Math.abs(low)) * 0.002, Number.EPSILON);
  return { minimum: low - padding, maximum: high + padding };
}

function formatAxisValue(value: number) {
  const absolute = Math.abs(value);
  const maximumFractionDigits =
    absolute >= 1_000 ? 2 : absolute >= 1 ? 4 : absolute >= 0.01 ? 6 : 8;
  return value.toLocaleString("zh-CN", { maximumFractionDigits });
}

function panelLabel(panel: IndicatorPanel) {
  switch (panel) {
    case "rsi":
      return "RSI (14)";
    case "bollinger":
      return "布林带 (20, 2)";
    case "momentum":
      return "动量 (14)";
    case "macd":
      return "MACD (12, 26, 9)";
  }
}

/**
 * A passive companion panel to CandlestickChart. The parent supplies the same
 * visible candle window so every value remains aligned to the selected OHLCV
 * snapshot rather than being recalculated on the device.
 */
export default function IndicatorChart({
  panel,
  candles,
  indicators,
  window,
  selectedTimestamp,
  height = 184,
}: Props) {
  const [width, setWidth] = useState(360);
  const selectedWindow = useMemo(
    () => normalizeWindow(window, candles.length),
    [candles.length, window],
  );
  const visibleCandles = useMemo(
    () => candles.slice(selectedWindow.start, selectedWindow.end),
    [candles, selectedWindow.end, selectedWindow.start],
  );
  const indicatorByTimestamp = useMemo(
    () => new Map(indicators.map((point) => [point.ts, point])),
    [indicators],
  );
  const visiblePoints = useMemo(
    () => visibleCandles.map((candle) => indicatorByTimestamp.get(candle.ts)),
    [indicatorByTimestamp, visibleCandles],
  );
  const selectedLocalIndex = useMemo(
    () =>
      selectedTimestamp == null
        ? -1
        : visibleCandles.findIndex((candle) => candle.ts === selectedTimestamp),
    [selectedTimestamp, visibleCandles],
  );

  const setChartWidth = (event: LayoutChangeEvent) => {
    setWidth(Math.max(1, event.nativeEvent.layout.width));
  };

  if (!visibleCandles.length) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyText}>当前 K 线范围没有可显示的技术指标</Text>
      </View>
    );
  }

  const rsi = visiblePoints.map((point) => finite(point?.rsi));
  const upper = visiblePoints.map((point) => finite(point?.bollinger?.upper));
  const middle = visiblePoints.map((point) => finite(point?.bollinger?.middle));
  const lower = visiblePoints.map((point) => finite(point?.bollinger?.lower));
  const close = visibleCandles.map((candle) => finite(candle.close));
  const momentum = visiblePoints.map((point) => finite(point?.momentum));
  const macd = visiblePoints.map((point) => finite(point?.macd));
  const macdSignal = visiblePoints.map((point) => finite(point?.macd_signal));
  const macdHistogram = visiblePoints.map((point) => finite(point?.macd_histogram));

  const indicatorBounds = (() => {
    if (panel === "rsi") return { minimum: 0, maximum: 100 };
    if (panel === "bollinger") return bounds([...upper, ...middle, ...lower, ...close]);
    if (panel === "momentum") return bounds(momentum, true);
    return bounds([...macd, ...macdSignal, ...macdHistogram], true);
  })();

  if (!indicatorBounds) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyText}>{panelLabel(panel)} 暂无服务端指标数据</Text>
      </View>
    );
  }

  const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
  const plotHeight = Math.max(1, height - PADDING.top - PADDING.bottom);
  const range = Math.max(
    indicatorBounds.maximum - indicatorBounds.minimum,
    Number.EPSILON,
  );
  const xAt = (index: number) =>
    PADDING.left + ((index + 0.5) / visibleCandles.length) * plotWidth;
  const yAt = (value: number) =>
    PADDING.top + (1 - (value - indicatorBounds.minimum) / range) * plotHeight;
  const histogramWidth = Math.max(
    1,
    Math.min(12, (plotWidth / visibleCandles.length) * 0.72),
  );
  const zeroY = yAt(0);

  return (
    <View accessibilityLabel={`${panelLabel(panel)} 技术指标图`} onLayout={setChartWidth} style={styles.container}>
      <Svg height={height} viewBox={`0 0 ${width} ${height}`} width={width}>
        {[0, 0.5, 1].map((fraction) => {
          const lineY = PADDING.top + plotHeight * fraction;
          return (
            <Line
              key={fraction}
              stroke={palette.border}
              strokeDasharray="3 4"
              strokeWidth={1}
              x1={PADDING.left}
              x2={width - PADDING.right}
              y1={lineY}
              y2={lineY}
            />
          );
        })}
        {panel === "rsi" ? (
          <>
            {[30, 70].map((guide) => (
              <Line
                key={guide}
                stroke={palette.textMuted}
                strokeDasharray="4 4"
                strokeWidth={1}
                x1={PADDING.left}
                x2={width - PADDING.right}
                y1={yAt(guide)}
                y2={yAt(guide)}
              />
            ))}
            <Path d={linePath(rsi, xAt, yAt)} fill="none" stroke="#B58CFF" strokeWidth={1.8} />
          </>
        ) : null}
        {panel === "bollinger" ? (
          <>
            <Path d={linePath(upper, xAt, yAt)} fill="none" stroke="#2AB5F6" strokeDasharray="4 3" strokeWidth={1.5} />
            <Path d={linePath(middle, xAt, yAt)} fill="none" stroke="#F5B544" strokeWidth={1.5} />
            <Path d={linePath(lower, xAt, yAt)} fill="none" stroke="#2AB5F6" strokeDasharray="4 3" strokeWidth={1.5} />
            <Path d={linePath(close, xAt, yAt)} fill="none" stroke={palette.text} strokeWidth={1.2} />
          </>
        ) : null}
        {panel === "momentum" ? (
          <>
            <Line
              stroke={palette.textMuted}
              strokeDasharray="4 4"
              strokeWidth={1}
              x1={PADDING.left}
              x2={width - PADDING.right}
              y1={zeroY}
              y2={zeroY}
            />
            <Path d={linePath(momentum, xAt, yAt)} fill="none" stroke={palette.primary} strokeWidth={1.8} />
          </>
        ) : null}
        {panel === "macd" ? (
          <>
            <Line
              stroke={palette.textMuted}
              strokeDasharray="4 4"
              strokeWidth={1}
              x1={PADDING.left}
              x2={width - PADDING.right}
              y1={zeroY}
              y2={zeroY}
            />
            {macdHistogram.map((value, index) => {
              if (value == null) return null;
              const y = yAt(value);
              return (
                <Rect
                  fill={value >= 0 ? palette.positive : palette.negative}
                  fillOpacity={0.62}
                  height={Math.max(1, Math.abs(zeroY - y))}
                  key={`${visibleCandles[index].ts}-histogram`}
                  width={histogramWidth}
                  x={xAt(index) - histogramWidth / 2}
                  y={Math.min(zeroY, y)}
                />
              );
            })}
            <Path d={linePath(macd, xAt, yAt)} fill="none" stroke={palette.primary} strokeWidth={1.8} />
            <Path d={linePath(macdSignal, xAt, yAt)} fill="none" stroke="#F5B544" strokeWidth={1.5} />
          </>
        ) : null}
        {selectedLocalIndex >= 0 ? (
          <Line
            stroke={palette.textMuted}
            strokeDasharray="4 4"
            strokeWidth={1}
            x1={xAt(selectedLocalIndex)}
            x2={xAt(selectedLocalIndex)}
            y1={PADDING.top}
            y2={PADDING.top + plotHeight}
          />
        ) : null}
      </Svg>
      <View pointerEvents="none" style={styles.title}>
        <Text style={styles.titleText}>{panelLabel(panel)}</Text>
      </View>
      <View pointerEvents="none" style={styles.maxLabel}>
        <Text style={styles.axisText}>{formatAxisValue(indicatorBounds.maximum)}</Text>
      </View>
      <View pointerEvents="none" style={styles.minLabel}>
        <Text style={styles.axisText}>{formatAxisValue(indicatorBounds.minimum)}</Text>
      </View>
      <View pointerEvents="none" style={styles.legend}>
        {panel === "rsi" ? <Text style={[styles.legendText, { color: "#B58CFF" }]}>RSI · 30 / 70</Text> : null}
        {panel === "bollinger" ? <Text style={styles.legendText}>上轨 / 中线 / 下轨 / 收盘</Text> : null}
        {panel === "momentum" ? <Text style={styles.legendText}>零轴</Text> : null}
        {panel === "macd" ? <Text style={styles.legendText}>MACD / 信号 / 柱状</Text> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: palette.surface,
    borderColor: palette.border,
    borderRadius: radius.md,
    borderWidth: 1,
    overflow: "hidden",
  },
  empty: {
    alignItems: "center",
    backgroundColor: palette.surface,
    borderColor: palette.border,
    borderRadius: radius.md,
    borderWidth: 1,
    justifyContent: "center",
    padding: spacing.md,
  },
  emptyText: { color: palette.textMuted, fontSize: 13, textAlign: "center" },
  title: { left: PADDING.left, position: "absolute", top: spacing.xs },
  titleText: { color: palette.text, fontSize: 11, fontWeight: "800" },
  maxLabel: { position: "absolute", right: spacing.xs, top: spacing.xs },
  minLabel: { bottom: spacing.xs, position: "absolute", right: spacing.xs },
  axisText: { color: palette.textMuted, fontSize: 10 },
  legend: { bottom: spacing.xs, left: PADDING.left, position: "absolute" },
  legendText: { color: palette.textMuted, fontSize: 10 },
});
