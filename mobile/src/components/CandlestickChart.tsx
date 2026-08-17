import { useEffect, useMemo, useRef, useState } from "react";
import {
  LayoutChangeEvent,
  PanResponder,
  StyleSheet,
  Text,
  View,
} from "react-native";
import Svg, { Circle, G, Line, Path, Rect, Text as SvgText } from "react-native-svg";
import type { CryptoCandle, CryptoIndicatorPoint } from "../types";
import { palette, radius, spacing } from "../theme";

export type ChartWindow = Readonly<{
  start: number;
  end: number;
}>;


type Props = {
  candles: CryptoCandle[];
  indicators: CryptoIndicatorPoint[];
  height?: number;
  onSelectCandle?: (candle: CryptoCandle | null) => void;
  onWindowChange?: (window: ChartWindow) => void;
};

export const CHART_HORIZONTAL_INSETS = { left: 8, right: 54 } as const;
const PADDING = { top: 28, ...CHART_HORIZONTAL_INSETS, bottom: 22 };
const MIN_VISIBLE_CANDLES = 12;
const INITIAL_VISIBLE_CANDLES = 64;
const MAX_VISIBLE_CANDLES = 500;
const VOLUME_HEIGHT = 54;
const PANEL_GAP = 8;

function finite(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

type TouchPoint = {
  locationX?: number;
  pageX?: number;
};

function coordinate(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function touchLocationX(touch: TouchPoint | undefined) {
  return coordinate(touch?.locationX);
}

function horizontalPinch(touches: readonly TouchPoint[]) {
  const first = touches[0];
  const second = touches[1];
  const firstLocationX = touchLocationX(first);
  const secondLocationX = touchLocationX(second);
  if (firstLocationX == null || secondLocationX == null) return null;

  const firstPageX = coordinate(first?.pageX) ?? firstLocationX;
  const secondPageX = coordinate(second?.pageX) ?? secondLocationX;
  const distance = Math.abs(firstPageX - secondPageX);
  if (distance < 12) return null;

  return {
    distance,
    midpoint: (firstLocationX + secondLocationX) / 2,
  };
}

function formatAxisValue(value: number) {
  const absolute = Math.abs(value);
  const maximumFractionDigits =
    absolute >= 1_000 ? 2 : absolute >= 1 ? 4 : absolute >= 0.01 ? 6 : 8;
  return value.toLocaleString("zh-CN", { maximumFractionDigits });
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

/**
 * Native OHLCV renderer. It deliberately only visualizes indicators returned
 * with the market snapshot; it does not calculate a technical series locally.
 */
export default function CandlestickChart({
  candles,
  indicators,
  height = 420,
  onSelectCandle,
  onWindowChange,
}: Props) {
  const [width, setWidth] = useState(360);
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_CANDLES);
  const [endIndex, setEndIndex] = useState(candles.length);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const visibleCountRef = useRef(visibleCount);
  const endIndexRef = useRef(endIndex);
  const pinchStartDistance = useRef<number | null>(null);
  const pinchStartCount = useRef(visibleCount);
  const pinchStartEnd = useRef(endIndex);
  const pinchStartMidpoint = useRef<number | null>(null);
  const panStartX = useRef<number | null>(null);
  const panStartEnd = useRef(endIndex);
  const movedDuringGesture = useRef(false);
  const pinchedDuringGesture = useRef(false);
  const onSelectCandleRef = useRef(onSelectCandle);
  const onWindowChangeRef = useRef(onWindowChange);

  onSelectCandleRef.current = onSelectCandle;
  onWindowChangeRef.current = onWindowChange;

  const candleCount = candles.length;
  const lastTimestamp = candles[candleCount - 1]?.ts;

  useEffect(() => {
    if (candleCount === 0) {
      setEndIndex(0);
      endIndexRef.current = 0;
      setSelectedIndex(null);
      return;
    }

    const maximumVisibleCount = Math.min(MAX_VISIBLE_CANDLES, candleCount);
    const nextVisibleCount = clamp(
      visibleCountRef.current,
      Math.min(MIN_VISIBLE_CANDLES, maximumVisibleCount),
      maximumVisibleCount,
    );
    setVisibleCount(nextVisibleCount);
    visibleCountRef.current = nextVisibleCount;
    setEndIndex(candleCount);
    endIndexRef.current = candleCount;
    setSelectedIndex(candleCount - 1);
  }, [candleCount, lastTimestamp]);

  const window = useMemo<ChartWindow>(() => {
    if (candleCount === 0) return { start: 0, end: 0 };
    const maximumVisibleCount = Math.min(MAX_VISIBLE_CANDLES, candleCount);
    const count = clamp(
      visibleCount,
      Math.min(MIN_VISIBLE_CANDLES, maximumVisibleCount),
      maximumVisibleCount,
    );
    const end = clamp(endIndex, count, candleCount);
    return { start: Math.max(0, end - count), end };
  }, [candleCount, endIndex, visibleCount]);

  const visibleCandles = useMemo(
    () => candles.slice(window.start, window.end),
    [candles, window.end, window.start],
  );

  useEffect(() => {
    onWindowChangeRef.current?.(window);
  }, [window]);

  useEffect(() => {
    if (selectedIndex == null || selectedIndex < window.start || selectedIndex >= window.end) {
      setSelectedIndex(visibleCandles.length ? window.end - 1 : null);
    }
  }, [selectedIndex, visibleCandles.length, window.end, window.start]);

  useEffect(() => {
    const selectedCandle =
      selectedIndex != null && selectedIndex >= window.start && selectedIndex < window.end
        ? candles[selectedIndex] ?? null
        : null;
    onSelectCandleRef.current?.(selectedCandle);
  }, [candles, selectedIndex, window.end, window.start]);

  const indicatorByTimestamp = useMemo(
    () => new Map(indicators.map((indicator) => [indicator.ts, indicator])),
    [indicators],
  );
  const visibleIndicators = useMemo(
    () => visibleCandles.map((candle) => indicatorByTimestamp.get(candle.ts)),
    [indicatorByTimestamp, visibleCandles],
  );

  const setChartWidth = (event: LayoutChangeEvent) => {
    setWidth(Math.max(1, event.nativeEvent.layout.width));
  };

  const chooseCandle = (locationX: number) => {
    if (!visibleCandles.length) return;
    const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
    const ratio = clamp((locationX - PADDING.left) / plotWidth, 0, 1);
    const localIndex = clamp(
      Math.round(ratio * (visibleCandles.length - 1)),
      0,
      visibleCandles.length - 1,
    );
    setSelectedIndex(window.start + localIndex);
  };

  const responder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,
        onPanResponderGrant: (event) => {
          const touches = event.nativeEvent.touches;
          const pinch = horizontalPinch(touches);
          panStartX.current = touchLocationX(touches[0]);
          panStartEnd.current = endIndexRef.current;
          movedDuringGesture.current = false;
          pinchedDuringGesture.current = pinch != null;
          if (pinch) {
            pinchStartDistance.current = pinch.distance;
            pinchStartCount.current = visibleCountRef.current;
            pinchStartEnd.current = endIndexRef.current;
            pinchStartMidpoint.current = pinch.midpoint;
          }
        },
        onPanResponderMove: (event) => {
          const touches = event.nativeEvent.touches;
          const pinch = horizontalPinch(touches);
          if (pinch) {
            const maximumVisibleCount = Math.min(MAX_VISIBLE_CANDLES, candles.length);
            if (maximumVisibleCount === 0) return;

            if (pinchStartDistance.current == null) {
              pinchStartDistance.current = pinch.distance;
              pinchStartCount.current = visibleCountRef.current;
              pinchStartEnd.current = endIndexRef.current;
              pinchStartMidpoint.current = pinch.midpoint;
            }

            pinchedDuringGesture.current = true;
            const minimumVisibleCount = Math.min(MIN_VISIBLE_CANDLES, maximumVisibleCount);
            const scale = clamp(
              pinch.distance / Math.max(pinchStartDistance.current, 1),
              0.25,
              4,
            );
            const initialVisibleCount = clamp(
              pinchStartCount.current,
              minimumVisibleCount,
              maximumVisibleCount,
            );
            const nextVisibleCount = clamp(
              Math.round(initialVisibleCount / scale),
              minimumVisibleCount,
              maximumVisibleCount,
            );
            const initialEnd = clamp(
              pinchStartEnd.current,
              initialVisibleCount,
              candles.length,
            );
            const initialStart = Math.max(0, initialEnd - initialVisibleCount);
            const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
            const pinchMidpoint = pinchStartMidpoint.current ?? pinch.midpoint;
            const focusRatio = clamp(
              (pinchMidpoint - PADDING.left) / plotWidth,
              0,
              1,
            );
            const focalCandle = initialStart + initialVisibleCount * focusRatio;
            const targetStart = Math.round(
              focalCandle - nextVisibleCount * focusRatio,
            );
            const nextEnd = clamp(
              targetStart + nextVisibleCount,
              nextVisibleCount,
              candles.length,
            );

            setVisibleCount(nextVisibleCount);
            visibleCountRef.current = nextVisibleCount;
            setEndIndex(nextEnd);
            endIndexRef.current = nextEnd;
            return;
          }

          if (pinchStartDistance.current != null) {
            pinchStartDistance.current = null;
            pinchStartMidpoint.current = null;
            panStartX.current = touchLocationX(touches[0]);
            panStartEnd.current = endIndexRef.current;
            return;
          }

          const startX = panStartX.current;
          if (startX == null || !visibleCandles.length) return;
          const currentX = touchLocationX(touches[0]) ?? startX;
          const moved = currentX - startX;
          if (Math.abs(moved) >= 4) movedDuringGesture.current = true;
          const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
          const shift = Math.round(
            (-moved / plotWidth) * visibleCountRef.current,
          );
          const nextEnd = clamp(
            panStartEnd.current + shift,
            Math.min(visibleCountRef.current, candles.length),
            candles.length,
          );
          setEndIndex(nextEnd);
          endIndexRef.current = nextEnd;
        },
        onPanResponderRelease: (event) => {
          if (!pinchedDuringGesture.current && !movedDuringGesture.current) {
            chooseCandle(event.nativeEvent.locationX);
          }
          pinchStartDistance.current = null;
          pinchStartMidpoint.current = null;
          panStartX.current = null;
          movedDuringGesture.current = false;
          pinchedDuringGesture.current = false;
        },
        onPanResponderTerminate: () => {
          pinchStartDistance.current = null;
          pinchStartMidpoint.current = null;
          panStartX.current = null;
          movedDuringGesture.current = false;
          pinchedDuringGesture.current = false;
        },
      }),
    [candles.length, visibleCandles.length, width, window.start],
  );

  if (!visibleCandles.length) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyText}>当前范围没有可显示的 K 线</Text>
      </View>
    );
  }

  const priceTop = PADDING.top;
  const volumeBottom = height - PADDING.bottom;
  const volumeTop = volumeBottom - VOLUME_HEIGHT;
  const priceBottom = volumeTop - PANEL_GAP;
  const priceHeight = Math.max(1, priceBottom - priceTop);
  const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
  const priceValues = visibleCandles.flatMap((candle, index) => {
    const indicator = visibleIndicators[index];
    return [
      candle.low,
      candle.high,
      finite(indicator?.ma.ma5),
      finite(indicator?.ma.ma20),
      finite(indicator?.ma.ma60),
    ].filter((value): value is number => value != null);
  });
  const low = Math.min(...priceValues);
  const high = Math.max(...priceValues);
  const pricePadding = Math.max((high - low) * 0.08, Math.abs(high) * 0.001, Number.EPSILON);
  const minimum = low - pricePadding;
  const maximum = high + pricePadding;
  const priceRange = Math.max(maximum - minimum, Number.EPSILON);
  const maxVolume = Math.max(
    ...visibleCandles.map((candle) => Math.max(0, candle.volume)),
    Number.EPSILON,
  );
  const candleWidth = Math.max(
    2.5,
    Math.min(14, (plotWidth / visibleCandles.length) * 0.76),
  );
  const volumeWidth = Math.max(
    1,
    Math.min(12, (plotWidth / visibleCandles.length) * 0.72),
  );
  const ma5 = visibleIndicators.map((indicator) => finite(indicator?.ma.ma5));
  const ma20 = visibleIndicators.map((indicator) => finite(indicator?.ma.ma20));
  const ma60 = visibleIndicators.map((indicator) => finite(indicator?.ma.ma60));
  const selectedLocalIndex =
    selectedIndex == null ? null : selectedIndex - window.start;
  const selectedCandle =
    selectedLocalIndex == null ||
    selectedLocalIndex < 0 ||
    selectedLocalIndex >= visibleCandles.length
      ? null
      : visibleCandles[selectedLocalIndex];

  const xAt = (index: number) =>
    PADDING.left + ((index + 0.5) / visibleCandles.length) * plotWidth;
  const priceY = (value: number) =>
    priceTop + (1 - (value - minimum) / priceRange) * priceHeight;
  const volumeY = (value: number) =>
    volumeBottom - (Math.max(0, value) / maxVolume) * VOLUME_HEIGHT;

  return (
    <View
      accessibilityLabel="可缩放、可拖动、可点按的 K 线图"
      onLayout={setChartWidth}
      style={styles.container}
      {...responder.panHandlers}
    >
      <Svg height={height} viewBox={`0 0 ${width} ${height}`} width={width}>
        {[0, 0.5, 1].map((fraction) => {
          const lineY = priceTop + priceHeight * fraction;
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
        <Line
          stroke={palette.border}
          strokeWidth={1}
          x1={PADDING.left}
          x2={width - PADDING.right}
          y1={volumeTop}
          y2={volumeTop}
        />
        {[maximum, (maximum + minimum) / 2, minimum].map((value, index) => (
          <SvgText
            fill={palette.textMuted}
            fontSize={10}
            key={`price-scale-${index}`}
            x={width - PADDING.right + 5}
            y={
              index === 0
                ? priceTop + 9
                : index === 2
                  ? priceBottom - 2
                  : priceTop + priceHeight / 2 + 3
            }
          >
            {formatAxisValue(value)}
          </SvgText>
        ))}
        <SvgText
          fill={palette.textMuted}
          fontSize={10}
          x={PADDING.left}
          y={volumeTop + 14}
        >
          {`成交量 ${formatAxisValue(maxVolume)}`}
        </SvgText>
        {visibleCandles.map((candle, index) => {
          const x = xAt(index);
          const color = candle.close >= candle.open ? palette.positive : palette.negative;
          const bodyTop = priceY(Math.max(candle.open, candle.close));
          const bodyBottom = priceY(Math.min(candle.open, candle.close));
          return (
            <G key={candle.ts}>
              <Rect
                fill={color}
                fillOpacity={0.48}
                height={Math.max(1, volumeBottom - volumeY(candle.volume))}
                width={volumeWidth}
                x={x - volumeWidth / 2}
                y={volumeY(candle.volume)}
              />
              <Line
                stroke={color}
                strokeWidth={1}
                x1={x}
                x2={x}
                y1={priceY(candle.high)}
                y2={priceY(candle.low)}
              />
              <Rect
                fill={candle.close >= candle.open ? color : palette.surface}
                stroke={color}
                strokeWidth={1}
                height={Math.max(1.5, bodyBottom - bodyTop)}
                rx={1}
                width={candleWidth}
                x={x - candleWidth / 2}
                y={bodyTop}
              />
            </G>
          );
        })}
        <Path d={linePath(ma5, xAt, priceY)} fill="none" stroke="#F5B544" strokeWidth={1.35} />
        <Path d={linePath(ma20, xAt, priceY)} fill="none" stroke={palette.primary} strokeWidth={1.35} />
        <Path d={linePath(ma60, xAt, priceY)} fill="none" stroke="#B58CFF" strokeWidth={1.35} />
        {selectedCandle && selectedLocalIndex != null ? (
          <>
            <Line
              stroke={palette.textMuted}
              strokeDasharray="4 4"
              strokeWidth={1}
              x1={xAt(selectedLocalIndex)}
              x2={xAt(selectedLocalIndex)}
              y1={priceTop}
              y2={volumeBottom}
            />
            <Line
              stroke={palette.textMuted}
              strokeDasharray="4 4"
              strokeWidth={1}
              x1={PADDING.left}
              x2={width - PADDING.right}
              y1={priceY(selectedCandle.close)}
              y2={priceY(selectedCandle.close)}
            />
            <Circle
              cx={xAt(selectedLocalIndex)}
              cy={priceY(selectedCandle.close)}
              fill={palette.surface}
              r={3.4}
              stroke={selectedCandle.close >= selectedCandle.open ? palette.positive : palette.negative}
              strokeWidth={1.5}
            />
          </>
        ) : null}
      </Svg>
      <View pointerEvents="none" style={styles.legend}>
        <Text style={[styles.legendText, { color: "#F5B544" }]}>MA5</Text>
        <Text style={[styles.legendText, { color: palette.primary }]}>MA20</Text>
        <Text style={[styles.legendText, { color: "#B58CFF" }]}>MA60</Text>
        <Text style={styles.windowText}>可见 {visibleCandles.length} 根</Text>
      </View>
      <View pointerEvents="none" style={styles.gestureCopy}>
        <Text style={styles.gestureText}>双指横向开合缩放 · 单指横向拖动回看 · 点按查看</Text>
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
  },
  emptyText: { color: palette.textMuted, fontSize: 13 },
  legend: {
    alignItems: "center",
    flexDirection: "row",
    gap: spacing.sm,
    left: PADDING.left,
    position: "absolute",
    top: spacing.xs,
  },
  legendText: { fontSize: 10, fontWeight: "700" },
  windowText: { color: palette.textMuted, fontSize: 10, fontWeight: "700" },
  gestureCopy: {
    bottom: spacing.xs,
    left: PADDING.left,
    position: "absolute",
  },
  gestureText: { color: palette.textMuted, fontSize: 9 }
});
