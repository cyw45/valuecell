import { StyleSheet, Text, View } from "react-native";
import Svg, { G, Line, Path, Rect } from "react-native-svg";
import type { Candle, IndicatorPoint } from "../types";
import { palette, radius, spacing } from "../theme";

type Props = {
  candles: Candle[];
  indicators: IndicatorPoint[];
  height?: number;
};

const CHART_WIDTH = 360;
const PADDING = { top: 20, right: 8, bottom: 24, left: 52 };

function toPath(values: Array<number | null>, minimum: number, maximum: number, height: number) {
  const availableWidth = CHART_WIDTH - PADDING.left - PADDING.right;
  const availableHeight = height - PADDING.top - PADDING.bottom;
  const denominator = Math.max(maximum - minimum, Number.EPSILON);
  let path = "";
  values.forEach((value, index) => {
    if (value == null) return;
    const x = PADDING.left + (index / Math.max(values.length - 1, 1)) * availableWidth;
    const y = PADDING.top + (1 - (value - minimum) / denominator) * availableHeight;
    path += `${path ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)} `;
  });
  return path;
}

export default function CandlestickChart({ candles, indicators, height = 272 }: Props) {
  if (!candles.length) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyText}>当前范围没有可显示的 K 线</Text>
      </View>
    );
  }

  const low = Math.min(...candles.map((candle) => candle.low));
  const high = Math.max(...candles.map((candle) => candle.high));
  const pricePadding = Math.max((high - low) * 0.08, high * 0.001);
  const minimum = low - pricePadding;
  const maximum = high + pricePadding;
  const plotWidth = CHART_WIDTH - PADDING.left - PADDING.right;
  const plotHeight = height - PADDING.top - PADDING.bottom;
  const denominator = Math.max(maximum - minimum, Number.EPSILON);
  const candleWidth = Math.max(2, Math.min(10, (plotWidth / candles.length) * 0.62));
  const ma5 = indicators.map((indicator) => indicator.ma.ma5 ?? null);
  const ma20 = indicators.map((indicator) => indicator.ma.ma20 ?? null);

  function y(value: number) {
    return PADDING.top + (1 - (value - minimum) / denominator) * plotHeight;
  }

  return (
    <View accessibilityLabel="K 线图" style={styles.container}>
      <Svg height={height} preserveAspectRatio="none" viewBox={`0 0 ${CHART_WIDTH} ${height}`} width="100%">
        {[0, 0.5, 1].map((fraction) => {
          const lineY = PADDING.top + plotHeight * fraction;
          return (
            <Line
              key={fraction}
              stroke={palette.border}
              strokeDasharray="3 4"
              strokeWidth={1}
              x1={PADDING.left}
              x2={CHART_WIDTH - PADDING.right}
              y1={lineY}
              y2={lineY}
            />
          );
        })}
        {candles.map((candle, index) => {
          const x = PADDING.left + ((index + 0.5) / candles.length) * plotWidth;
          const color = candle.close >= candle.open ? palette.positive : palette.negative;
          const bodyTop = y(Math.max(candle.open, candle.close));
          const bodyBottom = y(Math.min(candle.open, candle.close));
          return (
            <G key={candle.ts}>
              <Line stroke={color} strokeWidth={1} x1={x} x2={x} y1={y(candle.high)} y2={y(candle.low)} />
              <Rect
                fill={color}
                height={Math.max(1.5, bodyBottom - bodyTop)}
                rx={1}
                width={candleWidth}
                x={x - candleWidth / 2}
                y={bodyTop}
              />
            </G>
          );
        })}
        <Path d={toPath(ma5, minimum, maximum, height)} fill="none" stroke="#F5B544" strokeWidth={1.4} />
        <Path d={toPath(ma20, minimum, maximum, height)} fill="none" stroke={palette.primary} strokeWidth={1.4} />
      </Svg>
      <View pointerEvents="none" style={styles.axisTop}>
        <Text style={styles.axisText}>{high.toLocaleString(undefined, { maximumFractionDigits: 4 })}</Text>
      </View>
      <View pointerEvents="none" style={styles.axisBottom}>
        <Text style={styles.axisText}>{low.toLocaleString(undefined, { maximumFractionDigits: 4 })}</Text>
      </View>
      <View style={styles.legend}>
        <Text style={[styles.legendText, { color: "#F5B544" }]}>MA5</Text>
        <Text style={[styles.legendText, { color: palette.primary }]}>MA20</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, overflow: "hidden" },
  empty: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, justifyContent: "center" },
  emptyText: { color: palette.textMuted, fontSize: 13 },
  axisTop: { position: "absolute", right: spacing.xs, top: spacing.xs },
  axisBottom: { bottom: spacing.xs, position: "absolute", right: spacing.xs },
  axisText: { color: palette.textMuted, fontSize: 10 },
  legend: { flexDirection: "row", gap: spacing.sm, left: PADDING.left, position: "absolute", top: spacing.xs },
  legendText: { fontSize: 10, fontWeight: "700" },
});
