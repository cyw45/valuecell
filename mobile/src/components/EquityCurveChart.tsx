import { useEffect, useMemo, useRef, useState } from "react";
import {
  LayoutChangeEvent,
  PanResponder,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import Svg, { Circle, Line, Path, Text as SvgText } from "react-native-svg";
import type { RuleStrategyPnlPoint } from "../types";
import { palette, radius, spacing } from "../theme";

type EquityPoint = RuleStrategyPnlPoint & { equity_quote: number };

type Props = {
  points: readonly RuleStrategyPnlPoint[];
  formatQuote: (value: number | null | undefined) => string;
  formatTimestamp: (value: string) => string;
  height?: number;
};

const PADDING = { top: 30, right: 64, bottom: 26, left: 10 };
const MIN_VISIBLE_POINTS = 12;
const INITIAL_VISIBLE_POINTS = 64;
const MAX_VISIBLE_POINTS = 500;

function isFiniteNumber(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isEquityPoint(point: RuleStrategyPnlPoint): point is EquityPoint {
  return isFiniteNumber(point.equity_quote);
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

function latestCumulativePnl(points: readonly RuleStrategyPnlPoint[]) {
  for (let index = points.length - 1; index >= 0; index -= 1) {
    const value = points[index]?.cumulative_pnl;
    if (isFiniteNumber(value)) return value;
  }
  return undefined;
}

function formatAxisValue(value: number) {
  const absolute = Math.abs(value);
  const maximumFractionDigits =
    absolute >= 1_000 ? 2 : absolute >= 1 ? 4 : absolute >= 0.01 ? 6 : 8;
  return value.toLocaleString("zh-CN", { maximumFractionDigits });
}

function formatShortTimestamp(value: string) {
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) return value;
  return timestamp.toLocaleString("zh-CN", {
    day: "2-digit",
    hour: "2-digit",
    hour12: false,
    minute: "2-digit",
    month: "2-digit",
  });
}

function formatSignedQuote(
  value: number | undefined,
  formatQuote: Props["formatQuote"],
) {
  if (!isFiniteNumber(value)) return "—";
  return `${value > 0 ? "+" : ""}${formatQuote(value)}`;
}

function actionLabel(action: string) {
  switch (action) {
    case "initial":
      return "初始本金";
    case "buy":
      return "买入";
    case "sell":
      return "卖出";
    case "no_op":
      return "无操作";
    default:
      return action || "未标注";
  }
}

/**
 * Native equity renderer for the tenant-scoped strategy PnL curve. It only
 * draws server-returned `equity_quote` values and never infers equity locally.
 */
export function EquityCurveChart({
  points,
  formatQuote,
  formatTimestamp,
  height = 220,
}: Props) {
  const [width, setWidth] = useState(360);
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_POINTS);
  const [endIndex, setEndIndex] = useState(0);
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

  const curvePoints = useMemo(() => points.filter(isEquityPoint), [points]);
  const initialPoint = useMemo(
    () => points.find((point): point is EquityPoint => point.action === "initial" && isEquityPoint(point)),
    [points],
  );
  const pointCount = curvePoints.length;
  const lastTimestamp = curvePoints[pointCount - 1]?.ts;
  const latestPoint = curvePoints[pointCount - 1];
  const latestPnl = latestCumulativePnl(points);
  const initialCapital = initialPoint?.equity_quote;

  useEffect(() => {
    if (pointCount === 0) {
      setEndIndex(0);
      endIndexRef.current = 0;
      setSelectedIndex(null);
      return;
    }

    const maximumVisibleCount = Math.min(MAX_VISIBLE_POINTS, pointCount);
    const nextVisibleCount = clamp(
      visibleCountRef.current,
      Math.min(MIN_VISIBLE_POINTS, maximumVisibleCount),
      maximumVisibleCount,
    );
    setVisibleCount(nextVisibleCount);
    visibleCountRef.current = nextVisibleCount;
    setEndIndex(pointCount);
    endIndexRef.current = pointCount;
    setSelectedIndex(pointCount - 1);
  }, [lastTimestamp, pointCount]);

  const window = useMemo(() => {
    if (pointCount === 0) return { start: 0, end: 0 };
    const maximumVisibleCount = Math.min(MAX_VISIBLE_POINTS, pointCount);
    const count = clamp(
      visibleCount,
      Math.min(MIN_VISIBLE_POINTS, maximumVisibleCount),
      maximumVisibleCount,
    );
    const end = clamp(endIndex, count, pointCount);
    return { start: Math.max(0, end - count), end };
  }, [endIndex, pointCount, visibleCount]);

  const visiblePoints = useMemo(
    () => curvePoints.slice(window.start, window.end),
    [curvePoints, window.end, window.start],
  );

  useEffect(() => {
    if (selectedIndex == null || selectedIndex < window.start || selectedIndex >= window.end) {
      setSelectedIndex(visiblePoints.length ? window.end - 1 : null);
    }
  }, [selectedIndex, visiblePoints.length, window.end, window.start]);

  const setChartWidth = (event: LayoutChangeEvent) => {
    setWidth(Math.max(1, event.nativeEvent.layout.width));
  };

  const choosePoint = (locationX: number) => {
    if (!visiblePoints.length) return;
    const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
    const ratio = clamp((locationX - PADDING.left) / plotWidth, 0, 1);
    const localIndex = clamp(
      Math.round(ratio * (visiblePoints.length - 1)),
      0,
      visiblePoints.length - 1,
    );
    setSelectedIndex(window.start + localIndex);
  };

  const showLatest = () => {
    if (pointCount === 0) return;
    const maximumVisibleCount = Math.min(MAX_VISIBLE_POINTS, pointCount);
    const count = clamp(
      visibleCountRef.current,
      Math.min(MIN_VISIBLE_POINTS, maximumVisibleCount),
      maximumVisibleCount,
    );
    setVisibleCount(count);
    visibleCountRef.current = count;
    setEndIndex(pointCount);
    endIndexRef.current = pointCount;
    setSelectedIndex(pointCount - 1);
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
            const maximumVisibleCount = Math.min(MAX_VISIBLE_POINTS, pointCount);
            if (maximumVisibleCount === 0) return;

            if (pinchStartDistance.current == null) {
              pinchStartDistance.current = pinch.distance;
              pinchStartCount.current = visibleCountRef.current;
              pinchStartEnd.current = endIndexRef.current;
              pinchStartMidpoint.current = pinch.midpoint;
            }

            pinchedDuringGesture.current = true;
            const minimumVisibleCount = Math.min(MIN_VISIBLE_POINTS, maximumVisibleCount);
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
              pointCount,
            );
            const initialStart = Math.max(0, initialEnd - initialVisibleCount);
            const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
            const pinchMidpoint = pinchStartMidpoint.current ?? pinch.midpoint;
            const focusRatio = clamp(
              (pinchMidpoint - PADDING.left) / plotWidth,
              0,
              1,
            );
            const focalPoint = initialStart + initialVisibleCount * focusRatio;
            const targetStart = Math.round(
              focalPoint - nextVisibleCount * focusRatio,
            );
            const nextEnd = clamp(
              targetStart + nextVisibleCount,
              nextVisibleCount,
              pointCount,
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
          if (startX == null || !visiblePoints.length) return;
          const currentX = touchLocationX(touches[0]) ?? startX;
          const moved = currentX - startX;
          if (Math.abs(moved) >= 4) movedDuringGesture.current = true;
          const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
          const shift = Math.round(
            (-moved / plotWidth) * visibleCountRef.current,
          );
          const nextEnd = clamp(
            panStartEnd.current + shift,
            Math.min(visibleCountRef.current, pointCount),
            pointCount,
          );
          setEndIndex(nextEnd);
          endIndexRef.current = nextEnd;
        },
        onPanResponderRelease: (event) => {
          if (!pinchedDuringGesture.current && !movedDuringGesture.current) {
            choosePoint(event.nativeEvent.locationX);
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
    [pointCount, visiblePoints.length, width, window.start],
  );

  if (!points.length) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyTitle}>尚无权益时点</Text>
        <Text style={styles.emptyCopy}>服务端尚未返回策略权益曲线点。</Text>
      </View>
    );
  }

  if (!visiblePoints.length) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyTitle}>权益数据暂不可绘制</Text>
        <Text style={styles.emptyCopy}>服务端曲线未返回权益字段，移动端不会推算权益。</Text>
      </View>
    );
  }

  const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
  const plotHeight = Math.max(1, height - PADDING.top - PADDING.bottom);
  const equityValues = visiblePoints.map((point) => point.equity_quote);
  if (initialCapital != null) equityValues.push(initialCapital);
  const low = Math.min(...equityValues);
  const high = Math.max(...equityValues);
  const axisPadding = Math.max(
    (high - low) * 0.08,
    Math.abs(high) * 0.002,
    Math.abs(low) * 0.002,
    0.01,
  );
  const minimum = low - axisPadding;
  const maximum = high + axisPadding;
  const range = Math.max(maximum - minimum, Number.EPSILON);
  const xAt = (index: number) =>
    PADDING.left +
    (visiblePoints.length === 1 ? 0.5 : index / (visiblePoints.length - 1)) * plotWidth;
  const yAt = (value: number) =>
    PADDING.top + (1 - (value - minimum) / range) * plotHeight;
  const linePath = visiblePoints
    .map((point, index) => `${index === 0 ? "M" : "L"}${xAt(index).toFixed(2)},${yAt(point.equity_quote).toFixed(2)}`)
    .join(" ");
  const areaPath = visiblePoints.length > 1
    ? `${linePath} L${xAt(visiblePoints.length - 1).toFixed(2)},${(PADDING.top + plotHeight).toFixed(2)} L${xAt(0).toFixed(2)},${(PADDING.top + plotHeight).toFixed(2)} Z`
    : "";
  const selectedLocalIndex = selectedIndex == null ? null : selectedIndex - window.start;
  const selectedPoint =
    selectedLocalIndex == null ||
    selectedLocalIndex < 0 ||
    selectedLocalIndex >= visiblePoints.length
      ? null
      : visiblePoints[selectedLocalIndex] ?? null;
  const selectedPnl = selectedPoint?.cumulative_pnl;
  const curveColor = isFiniteNumber(latestPnl) && latestPnl < 0
    ? palette.negative
    : isFiniteNumber(latestPnl) && latestPnl > 0
      ? palette.positive
      : palette.primary;
  const selectedPnlColor = isFiniteNumber(selectedPnl) && selectedPnl < 0
    ? palette.negative
    : isFiniteNumber(selectedPnl) && selectedPnl > 0
      ? palette.positive
      : palette.text;
  const baselineY = initialCapital == null ? null : yAt(initialCapital);
  const firstVisibleTime = visiblePoints[0]?.ts;
  const lastVisibleTime = visiblePoints[visiblePoints.length - 1]?.ts;

  return (
    <View style={styles.root}>
      <View style={styles.summaryGrid}>
        <View style={styles.summaryMetric}>
          <Text style={styles.summaryLabel}>初始本金</Text>
          <Text numberOfLines={1} style={styles.summaryValue}>{formatQuote(initialCapital)}</Text>
        </View>
        <View style={styles.summaryMetric}>
          <Text style={styles.summaryLabel}>最新权益</Text>
          <Text numberOfLines={1} style={styles.summaryValue}>{formatQuote(latestPoint?.equity_quote)}</Text>
        </View>
        <View style={styles.summaryMetric}>
          <Text style={styles.summaryLabel}>累计 PnL</Text>
          <Text numberOfLines={1} style={[styles.summaryValue, { color: curveColor }]}>{formatSignedQuote(latestPnl, formatQuote)}</Text>
        </View>
      </View>

      <View
        accessible
        accessibilityHint="单指左右拖动回看，双指横向开合缩放，点按查看对应时间点。"
        accessibilityLabel={`可交互策略权益曲线，当前可见 ${visiblePoints.length} 个服务端权益时点。`}
        onLayout={setChartWidth}
        style={styles.chartSurface}
        {...responder.panHandlers}
      >
        <View pointerEvents="none" style={styles.chartLegend}>
          <Text style={[styles.legendText, { color: curveColor }]}>权益</Text>
          <Text style={styles.legendText}>初始本金虚线</Text>
          <Text style={styles.legendText}>可见 {visiblePoints.length} / {pointCount}</Text>
        </View>
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
          {baselineY != null ? (
            <>
              <Line
                stroke={palette.textMuted}
                strokeDasharray="6 4"
                strokeWidth={1}
                x1={PADDING.left}
                x2={width - PADDING.right}
                y1={baselineY}
                y2={baselineY}
              />
              <SvgText
                fill={palette.textMuted}
                fontSize={9}
                x={PADDING.left + 4}
                y={Math.max(PADDING.top + 10, baselineY - 4)}
              >
                初始本金
              </SvgText>
            </>
          ) : null}
          {areaPath ? <Path d={areaPath} fill={curveColor} fillOpacity={0.12} /> : null}
          <Path d={linePath} fill="none" stroke={curveColor} strokeWidth={2.4} />
          {[maximum, (maximum + minimum) / 2, minimum].map((value, index) => (
            <SvgText
              fill={palette.textMuted}
              fontSize={9}
              key={`equity-scale-${index}`}
              x={width - PADDING.right + 5}
              y={
                index === 0
                  ? PADDING.top + 8
                  : index === 2
                    ? PADDING.top + plotHeight - 2
                    : PADDING.top + plotHeight / 2 + 3
              }
            >
              {formatAxisValue(value)}
            </SvgText>
          ))}
          <SvgText fill={palette.textMuted} fontSize={9} x={PADDING.left} y={height - 7}>
            {firstVisibleTime ? formatShortTimestamp(firstVisibleTime) : "—"}
          </SvgText>
          <SvgText
            fill={palette.textMuted}
            fontSize={9}
            textAnchor="end"
            x={width - PADDING.right}
            y={height - 7}
          >
            {lastVisibleTime ? formatShortTimestamp(lastVisibleTime) : "—"}
          </SvgText>
          {selectedPoint && selectedLocalIndex != null ? (
            <>
              <Line
                stroke={palette.textMuted}
                strokeDasharray="4 4"
                strokeWidth={1}
                x1={xAt(selectedLocalIndex)}
                x2={xAt(selectedLocalIndex)}
                y1={PADDING.top}
                y2={PADDING.top + plotHeight}
              />
              <Circle
                cx={xAt(selectedLocalIndex)}
                cy={yAt(selectedPoint.equity_quote)}
                fill={palette.surfaceRaised}
                r={4}
                stroke={curveColor}
                strokeWidth={2}
              />
            </>
          ) : null}
        </Svg>
        <View pointerEvents="none" style={styles.gestureCopy}>
          <Text style={styles.gestureText}>单指左右拖动回看 · 双指横向开合缩放 · 点按查看</Text>
        </View>
      </View>

      {selectedPoint ? (
        <View style={styles.selectionCard}>
          <View style={styles.selectionHeader}>
            <View style={styles.selectionCopy}>
              <Text style={styles.selectionTitle}>时间轴详情</Text>
              <Text numberOfLines={1} style={styles.selectionTime}>{formatTimestamp(selectedPoint.ts)}</Text>
            </View>
            <Pressable
              accessibilityLabel="回到最新权益时点"
              accessibilityRole="button"
              onPress={showLatest}
              style={({ pressed }) => [styles.latestButton, pressed && styles.pressed]}
            >
              <Text style={styles.latestButtonText}>回到最新</Text>
            </Pressable>
          </View>
          <View style={styles.detailGrid}>
            <View style={styles.detailMetric}>
              <Text style={styles.detailLabel}>权益</Text>
              <Text numberOfLines={1} style={styles.detailValue}>{formatQuote(selectedPoint.equity_quote)}</Text>
            </View>
            <View style={styles.detailMetric}>
              <Text style={styles.detailLabel}>累计 PnL</Text>
              <Text numberOfLines={1} style={[styles.detailValue, { color: selectedPnlColor }]}>{formatSignedQuote(selectedPnl, formatQuote)}</Text>
            </View>
            <View style={styles.detailMetric}>
              <Text style={styles.detailLabel}>当日盈亏</Text>
              <Text numberOfLines={1} style={[styles.detailValue, { color: selectedPnlColor }]}>{formatSignedQuote(selectedPoint.daily_pnl_quote, formatQuote)}</Text>
            </View>
            <View style={styles.detailMetric}>
              <Text style={styles.detailLabel}>服务端事件</Text>
              <Text numberOfLines={1} style={styles.detailValue}>{actionLabel(selectedPoint.action)}</Text>
            </View>
          </View>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { gap: spacing.sm },
  summaryGrid: { flexDirection: "row", gap: spacing.xs },
  summaryMetric: {
    backgroundColor: palette.surfaceRaised,
    borderRadius: radius.sm,
    flex: 1,
    gap: spacing.xxs,
    minHeight: 62,
    padding: spacing.xs,
  },
  summaryLabel: { color: palette.textMuted, fontSize: 10, fontWeight: "800" },
  summaryValue: { color: palette.text, fontSize: 12, fontWeight: "900" },
  chartSurface: {
    backgroundColor: palette.surfaceRaised,
    borderColor: palette.border,
    borderRadius: radius.md,
    borderWidth: 1,
    overflow: "hidden",
  },
  chartLegend: {
    alignItems: "center",
    flexDirection: "row",
    gap: spacing.sm,
    left: PADDING.left,
    position: "absolute",
    top: spacing.xs,
  },
  legendText: { color: palette.textMuted, fontSize: 10, fontWeight: "800" },
  gestureCopy: { alignItems: "center", minHeight: 28, paddingHorizontal: spacing.sm },
  gestureText: { color: palette.textMuted, fontSize: 10, lineHeight: 16, textAlign: "center" },
  selectionCard: {
    backgroundColor: palette.surfaceRaised,
    borderColor: palette.border,
    borderRadius: radius.md,
    borderWidth: 1,
    gap: spacing.sm,
    padding: spacing.sm,
  },
  selectionHeader: { alignItems: "center", flexDirection: "row", gap: spacing.sm },
  selectionCopy: { flex: 1, gap: 2 },
  selectionTitle: { color: palette.text, fontSize: 13, fontWeight: "900" },
  selectionTime: { color: palette.textMuted, fontSize: 11, lineHeight: 16 },
  latestButton: {
    alignItems: "center",
    borderColor: palette.primary,
    borderRadius: radius.sm,
    borderWidth: 1,
    justifyContent: "center",
    minHeight: 44,
    paddingHorizontal: spacing.sm,
  },
  latestButtonText: { color: palette.primary, fontSize: 12, fontWeight: "900" },
  detailGrid: { flexDirection: "row", gap: spacing.xs },
  detailMetric: { flex: 1, gap: spacing.xxs, minWidth: 0 },
  detailLabel: { color: palette.textMuted, fontSize: 10, fontWeight: "800" },
  detailValue: { color: palette.text, fontSize: 12, fontWeight: "900" },
  empty: {
    alignItems: "center",
    backgroundColor: palette.surfaceRaised,
    borderColor: palette.border,
    borderRadius: radius.md,
    borderWidth: 1,
    gap: spacing.xxs,
    justifyContent: "center",
    padding: spacing.md,
  },
  emptyTitle: { color: palette.text, fontSize: 14, fontWeight: "900" },
  emptyCopy: { color: palette.textMuted, fontSize: 12, lineHeight: 18, textAlign: "center" },
  pressed: { opacity: 0.76 },
});
