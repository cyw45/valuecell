import { useEffect, useMemo, useRef, useState } from "react";
import {
  LayoutChangeEvent,
  PanResponder,
  StyleSheet,
  Text,
  View,
} from "react-native";
import Svg, { G, Line, Path, Rect } from "react-native-svg";
import type { Candle, IndicatorPoint } from "../types";
import { palette, radius, spacing } from "../theme";

type Props = {
  candles: Candle[];
  indicators: IndicatorPoint[];
  height?: number;
  onSelectCandle?: (candle: Candle | null) => void;
};

const PADDING = { top: 22, right: 10, bottom: 28, left: 54 };
const MIN_VISIBLE_CANDLES = 12;

function formatTimestamp(timestamp: number) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function linePath(
  values: Array<number | null>,
  minimum: number,
  maximum: number,
  width: number,
  height: number,
) {
  const availableWidth = width - PADDING.left - PADDING.right;
  const availableHeight = height - PADDING.top - PADDING.bottom;
  const range = Math.max(maximum - minimum, Number.EPSILON);
  let path = "";
  values.forEach((value, index) => {
    if (value == null) return;
    const x = PADDING.left + (index / Math.max(values.length - 1, 1)) * availableWidth;
    const y = PADDING.top + (1 - (value - minimum) / range) * availableHeight;
    path += `${path ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)} `;
  });
  return path;
}

export default function CandlestickChart({
  candles,
  indicators,
  height = 292,
  onSelectCandle,
}: Props) {
  const [width, setWidth] = useState(360);
  const [visibleCount, setVisibleCount] = useState(80);
  const [endIndex, setEndIndex] = useState(candles.length);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const visibleCountRef = useRef(visibleCount);
  const endIndexRef = useRef(endIndex);
  const pinchStartDistance = useRef<number | null>(null);
  const pinchStartCount = useRef(visibleCount);
  const panStartX = useRef<number | null>(null);
  const panStartEnd = useRef(endIndex);

  useEffect(() => {
    const nextVisibleCount = Math.min(Math.max(MIN_VISIBLE_CANDLES, visibleCountRef.current), candles.length || 1);
    setVisibleCount(nextVisibleCount);
    visibleCountRef.current = nextVisibleCount;
    setEndIndex(candles.length);
    endIndexRef.current = candles.length;
    setSelectedIndex(null);
    onSelectCandle?.(null);
  }, [candles.length, onSelectCandle]);

  const window = useMemo(() => {
    const count = Math.min(Math.max(MIN_VISIBLE_CANDLES, visibleCount), candles.length || 1);
    const end = Math.min(Math.max(count, endIndex), candles.length);
    return { start: Math.max(0, end - count), end, candles: candles.slice(Math.max(0, end - count), end) };
  }, [candles, endIndex, visibleCount]);

  const indicatorByTimestamp = useMemo(
    () => new Map(indicators.map((indicator) => [indicator.ts, indicator])),
    [indicators],
  );
  const visibleIndicators = window.candles.map(
    (candle) => indicatorByTimestamp.get(candle.ts) ?? { ts: candle.ts, ma: {} },
  );

  const setChartWidth = (event: LayoutChangeEvent) => {
    setWidth(Math.max(1, event.nativeEvent.layout.width));
  };

  const chooseCandle = (locationX: number) => {
    if (!window.candles.length) return;
    const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
    const ratio = Math.min(1, Math.max(0, (locationX - PADDING.left) / plotWidth));
    const index = Math.min(window.candles.length - 1, Math.max(0, Math.round(ratio * (window.candles.length - 1))));
    setSelectedIndex(index);
    onSelectCandle?.(window.candles[index]);
  };

  const responder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,
        onPanResponderGrant: (event) => {
          const touches = event.nativeEvent.touches;
          panStartX.current = touches[0]?.locationX ?? null;
          panStartEnd.current = endIndexRef.current;
          if (touches.length >= 2) {
            pinchStartDistance.current = Math.abs(touches[0].locationX - touches[1].locationX);
            pinchStartCount.current = visibleCountRef.current;
          }
        },
        onPanResponderMove: (event) => {
          const touches = event.nativeEvent.touches;
          if (touches.length >= 2 && pinchStartDistance.current) {
            const distance = Math.abs(touches[0].locationX - touches[1].locationX);
            const scale = Math.max(distance / Math.max(pinchStartDistance.current, 1), 0.1);
            const next = Math.round(pinchStartCount.current / scale);
            const clamped = Math.min(candles.length, Math.max(MIN_VISIBLE_CANDLES, next));
            setVisibleCount(clamped);
            visibleCountRef.current = clamped;
            setEndIndex((current) => Math.min(candles.length, Math.max(clamped, current)));
            return;
          }
          const startX = panStartX.current;
          if (startX == null || !window.candles.length) return;
          const moved = event.nativeEvent.touches[0]?.locationX - startX;
          const shift = Math.round((-moved / Math.max(width, 1)) * visibleCountRef.current);
          const next = Math.min(candles.length, Math.max(visibleCountRef.current, panStartEnd.current + shift));
          setEndIndex(next);
          endIndexRef.current = next;
        },
        onPanResponderRelease: (event) => {
          if (pinchStartDistance.current == null) chooseCandle(event.nativeEvent.locationX);
          pinchStartDistance.current = null;
          panStartX.current = null;
        },
      }),
    [candles.length, endIndexRef, onSelectCandle, visibleCountRef, width, window.candles.length],
  );

  if (!window.candles.length) {
    return <View style={[styles.empty, { height }]}><Text style={styles.emptyText}>当前范围没有可显示的 K 线</Text></View>;
  }

  const low = Math.min(...window.candles.map((candle) => candle.low));
  const high = Math.max(...window.candles.map((candle) => candle.high));
  const pricePadding = Math.max((high - low) * 0.08, high * 0.001);
  const minimum = low - pricePadding;
  const maximum = high + pricePadding;
  const plotWidth = width - PADDING.left - PADDING.right;
  const plotHeight = height - PADDING.top - PADDING.bottom;
  const range = Math.max(maximum - minimum, Number.EPSILON);
  const candleWidth = Math.max(2, Math.min(10, (plotWidth / window.candles.length) * 0.62));
  const ma5 = visibleIndicators.map((indicator) => indicator.ma.ma5 ?? null);
  const ma20 = visibleIndicators.map((indicator) => indicator.ma.ma20 ?? null);
  const selectedCandle = selectedIndex == null ? null : window.candles[selectedIndex];

  function y(value: number) {
    return PADDING.top + (1 - (value - minimum) / range) * plotHeight;
  }

  return (
    <View accessibilityLabel="可缩放的 K 线图" onLayout={setChartWidth} style={styles.container} {...responder.panHandlers}>
      <Svg height={height} viewBox={`0 0 ${width} ${height}`} width={width}>
        {[0, 0.5, 1].map((fraction) => {
          const lineY = PADDING.top + plotHeight * fraction;
          return <Line key={fraction} stroke={palette.border} strokeDasharray="3 4" strokeWidth={1} x1={PADDING.left} x2={width - PADDING.right} y1={lineY} y2={lineY} />;
        })}
        {window.candles.map((candle, index) => {
          const x = PADDING.left + ((index + 0.5) / window.candles.length) * plotWidth;
          const color = candle.close >= candle.open ? palette.positive : palette.negative;
          const bodyTop = y(Math.max(candle.open, candle.close));
          const bodyBottom = y(Math.min(candle.open, candle.close));
          return <G key={candle.ts}><Line stroke={color} strokeWidth={1} x1={x} x2={x} y1={y(candle.high)} y2={y(candle.low)} /><Rect fill={color} height={Math.max(1.5, bodyBottom - bodyTop)} rx={1} width={candleWidth} x={x - candleWidth / 2} y={bodyTop} /></G>;
        })}
        <Path d={linePath(ma5, minimum, maximum, width, height)} fill="none" stroke="#F5B544" strokeWidth={1.4} />
        <Path d={linePath(ma20, minimum, maximum, width, height)} fill="none" stroke={palette.primary} strokeWidth={1.4} />
        {selectedIndex != null ? <Line stroke={palette.textMuted} strokeDasharray="4 4" strokeWidth={1} x1={PADDING.left + ((selectedIndex + 0.5) / window.candles.length) * plotWidth} x2={PADDING.left + ((selectedIndex + 0.5) / window.candles.length) * plotWidth} y1={PADDING.top} y2={height - PADDING.bottom} /> : null}
      </Svg>
      <View pointerEvents="none" style={styles.axisTop}><Text style={styles.axisText}>{high.toLocaleString(undefined, { maximumFractionDigits: 4 })}</Text></View>
      <View pointerEvents="none" style={styles.axisBottom}><Text style={styles.axisText}>{low.toLocaleString(undefined, { maximumFractionDigits: 4 })}</Text></View>
      <View pointerEvents="none" style={styles.legend}><Text style={[styles.legendText, { color: "#F5B544" }]}>MA5</Text><Text style={[styles.legendText, { color: palette.primary }]}>MA20</Text><Text style={styles.gestureText}>双指缩放 · 拖动回看 · 点按明细</Text></View>
      {selectedCandle ? <View pointerEvents="none" style={styles.tooltip}><Text style={styles.tooltipTime}>{formatTimestamp(selectedCandle.ts)}</Text><Text style={styles.tooltipText}>开 {selectedCandle.open.toLocaleString()} · 高 {selectedCandle.high.toLocaleString()}</Text><Text style={styles.tooltipText}>低 {selectedCandle.low.toLocaleString()} · 收 {selectedCandle.close.toLocaleString()}</Text></View> : null}
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
  legend: { alignItems: "center", flexDirection: "row", gap: spacing.sm, left: PADDING.left, position: "absolute", top: spacing.xs },
  legendText: { fontSize: 10, fontWeight: "700" },
  gestureText: { color: palette.textMuted, fontSize: 9, marginLeft: spacing.xs },
  tooltip: { backgroundColor: "rgba(7,17,31,0.94)", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, bottom: spacing.sm, left: spacing.sm, padding: spacing.xs, position: "absolute" },
  tooltipTime: { color: palette.primary, fontSize: 10, fontWeight: "800", marginBottom: 2 },
  tooltipText: { color: palette.text, fontSize: 10, lineHeight: 15 },
});
