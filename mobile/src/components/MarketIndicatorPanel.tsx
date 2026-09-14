import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { ChevronRight, RefreshCw, SlidersHorizontal } from "lucide-react-native";
import { api } from "../api";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import type { CryptoCandle } from "../types";
import { AnimatedQuote } from "./AnimatedQuote";
import CandlestickChart, { type ChartWindow, type PriceOverlay } from "./CandlestickChart";
import IndicatorChart, { type IndicatorPanel } from "./IndicatorChart";

/**
 * Inline market and indicator block for the concurrency console.
 *
 * It renders the same server market snapshot the Web dashboard renders (candles,
 * volume, price overlays and a lower indicator panel) and follows the selected
 * strategy's symbol set. Indicators are never computed locally: a missing fact
 * renders as unavailable instead of as a derived number.
 */
export type MarketIndicatorPanelProps = {
  strategyName: string;
  symbols: readonly string[];
  focusSymbol?: string | null;
  onOpenFull?: () => void;
};

type MarketInterval = "5m" | "15m" | "30m" | "1h" | "4h" | "1d";
type HistoryRange = "1D" | "5D" | "10D" | "30D" | "90D";

const INTERVAL_OPTIONS: ReadonlyArray<{ value: MarketInterval; label: string }> = [
  { value: "5m", label: "5分" },
  { value: "15m", label: "15分" },
  { value: "30m", label: "30分" },
  { value: "1h", label: "1小时" },
  { value: "4h", label: "4小时" },
  { value: "1d", label: "1日" },
];

const RANGE_OPTIONS: ReadonlyArray<{ value: HistoryRange; label: string }> = [
  { value: "1D", label: "1日" },
  { value: "5D", label: "5日" },
  { value: "10D", label: "10日" },
  { value: "30D", label: "30日" },
  { value: "90D", label: "90日" },
];

const RANGE_DAYS: Record<HistoryRange, number> = {
  "1D": 1,
  "5D": 5,
  "10D": 10,
  "30D": 30,
  "90D": 90,
};

const INTERVAL_MS: Record<MarketInterval, number> = {
  "5m": 300_000,
  "15m": 900_000,
  "30m": 1_800_000,
  "1h": 3_600_000,
  "4h": 14_400_000,
  "1d": 86_400_000,
};

const PRICE_OVERLAYS: readonly PriceOverlay[] = ["ma5", "ma20", "bollinger"];
const OVERLAY_SUMMARY = "MA5 · MA20 · 布林带";
const LOWER_PANELS: ReadonlyArray<{ value: IndicatorPanel; label: string }> = [
  { value: "rsi", label: "RSI" },
  { value: "macd", label: "MACD" },
  { value: "momentum", label: "动量" },
  { value: "bollinger", label: "布林带" },
];

function dashboardSymbol(symbol: string) {
  return symbol.replace("-", "/");
}

function formatPrice(value: number) {
  const absolute = Math.abs(value);
  const maximumFractionDigits = absolute >= 1_000 ? 2 : absolute >= 1 ? 2 : absolute >= 0.01 ? 4 : 8;
  return value.toLocaleString("zh-CN", { maximumFractionDigits });
}

function formatTimestamp(value: number) {
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function priceFractionDigits(value: number) {
  if (value >= 1_000) return 2;
  if (value >= 1) return 4;
  return 6;
}
export default function MarketIndicatorPanel({
  strategyName,
  symbols,
  focusSymbol,
  onOpenFull,
}: MarketIndicatorPanelProps) {
  const { session } = useSession();
  const tenantId = session?.tenantId ?? "public";
  const [symbol, setSymbol] = useState("");
  const [interval, setInterval] = useState<MarketInterval>("1h");
  const [range, setRange] = useState<HistoryRange>("10D");
  const [rangeAnchor, setRangeAnchor] = useState(() => Date.now());
  const [lowerPanel, setLowerPanel] = useState<IndicatorPanel>("rsi");
  const [visibleWindow, setVisibleWindow] = useState<ChartWindow>();
  const [selectedCandle, setSelectedCandle] = useState<CryptoCandle | null>(null);

  // Keep the chart on a symbol that still belongs to the selected strategy so a
  // card switch never leaves a stale pair behind; an explicit focus request from
  // the concurrency matrix wins over the previous selection.
  useEffect(() => {
    if (symbols.length === 0) {
      if (symbol !== "") setSymbol("");
      return;
    }
    if (focusSymbol && symbols.includes(focusSymbol)) {
      if (focusSymbol !== symbol) setSymbol(focusSymbol);
      return;
    }
    if (!symbols.includes(symbol)) setSymbol(symbols[0]);
  }, [focusSymbol, symbol, symbols]);

  const dateRange = useMemo(() => {
    const toTsMs = rangeAnchor;
    return {
      fromTsMs: toTsMs - RANGE_DAYS[range] * 24 * 60 * 60 * 1_000,
      toTsMs,
    };
  }, [range, rangeAnchor]);

  const lookback = useMemo(
    () =>
      Math.min(
        5_000,
        Math.max(
          1,
          Math.ceil((dateRange.toTsMs - dateRange.fromTsMs) / INTERVAL_MS[interval]) + 2,
        ),
      ),
    [dateRange.fromTsMs, dateRange.toTsMs, interval],
  );

  const market = useQuery({
    queryKey: [
      "mobile",
      tenantId,
      "crypto-market",
      symbol,
      interval,
      lookback,
      dateRange.fromTsMs,
      dateRange.toTsMs,
    ],
    queryFn: () =>
      api.market(symbol, interval, lookback, {
        from_ts_ms: dateRange.fromTsMs,
        to_ts_ms: dateRange.toTsMs,
      }),
    enabled: Boolean(symbol),
  });

  const marketSymbol = market.data?.symbols.find((item) => item.symbol === symbol);
  const failureReason = symbol ? market.data?.failed_symbols[symbol] : undefined;
  const latestPrice = marketSymbol?.latest_price ?? null;
  const candleChange = useMemo(() => {
    if (!selectedCandle || !selectedCandle.open) return null;
    return ((selectedCandle.close - selectedCandle.open) / selectedCandle.open) * 100;
  }, [selectedCandle]);

  useEffect(() => {
    setVisibleWindow(undefined);
    setSelectedCandle(null);
  }, [interval, range, symbol]);
  const reload = () => {
    setRangeAnchor(Date.now());
    void market.refetch();
  };

  return (
    <View style={styles.block}>
      <View style={styles.header}>
        <View style={styles.headerCopy}>
          <Text style={styles.title}>
            {symbol ? `${dashboardSymbol(symbol)} 市场走势` : "市场走势"}
          </Text>
          <Text style={styles.meta}>
            {strategyName} · 监测 {symbols.length} 个币种 · {interval} K 线
            {marketSymbol?.freshness_status === "stale" ? " · 数据延迟" : ""}
          </Text>
        </View>
        <View style={styles.priceBlock}>
          <Text style={styles.priceLabel}>当前价格</Text>
          {latestPrice != null && Number.isFinite(latestPrice) ? (
            <AnimatedQuote
              fractionDigits={priceFractionDigits(latestPrice)}
              style={styles.priceValue}
              suffix=" USDT"
              value={latestPrice}
            />
          ) : (
            <Text style={styles.priceUnavailable}>等待行情</Text>
          )}
        </View>
      </View>

      {symbols.length > 0 ? (
        <ScrollView
          contentContainerStyle={styles.chipRow}
          horizontal
          showsHorizontalScrollIndicator={false}
        >
          {symbols.map((item) => {
            const selected = item === symbol;
            return (
              <Pressable
                accessibilityRole="button"
                key={item}
                onPress={() => setSymbol(item)}
                style={[styles.chip, selected && styles.chipSelected]}
              >
                <Text style={[styles.chipText, selected && styles.chipTextSelected]}>
                  {dashboardSymbol(item)}
                </Text>
              </Pressable>
            );
          })}
        </ScrollView>
      ) : (
        <Text style={styles.muted}>该策略尚未配置币种，无法展示走势。</Text>
      )}

      <View style={styles.controlGroup}>
        <Text style={styles.controlLabel}>K 线周期</Text>
        <ScrollView
          contentContainerStyle={styles.chipRow}
          horizontal
          showsHorizontalScrollIndicator={false}
        >
          {INTERVAL_OPTIONS.map((option) => (
            <Pressable
              accessibilityRole="button"
              key={option.value}
              onPress={() => setInterval(option.value)}
              style={[styles.smallChip, option.value === interval && styles.chipSelected]}
            >
              <Text style={[styles.smallChipText, option.value === interval && styles.chipTextSelected]}>
                {option.label}
              </Text>
            </Pressable>
          ))}
        </ScrollView>
        <Text style={styles.controlLabel}>历史范围</Text>
        <ScrollView
          contentContainerStyle={styles.chipRow}
          horizontal
          showsHorizontalScrollIndicator={false}
        >
          {RANGE_OPTIONS.map((option) => (
            <Pressable
              accessibilityRole="button"
              key={option.value}
              onPress={() => setRange(option.value)}
              style={[styles.smallChip, option.value === range && styles.chipSelected]}
            >
              <Text style={[styles.smallChipText, option.value === range && styles.chipTextSelected]}>
                {option.label}
              </Text>
            </Pressable>
          ))}
        </ScrollView>
      </View>

      {market.isLoading && !marketSymbol ? (
        <View style={styles.loading}>
          <ActivityIndicator color={palette.primary} />
          <Text style={styles.muted}>正在读取服务端行情快照。</Text>
        </View>
      ) : null}

      {!market.isLoading && !marketSymbol ? (
        <View style={styles.errorPanel}>
          <Text style={styles.errorTitle}>当前行情暂不可用</Text>
          <Text style={styles.muted}>
            {failureReason ??
              (market.error as Error | null)?.message ??
              "服务端尚未返回该币种的行情快照。"}
          </Text>
          <Pressable accessibilityRole="button" onPress={reload} style={styles.retryButton}>
            <RefreshCw color={palette.canvas} size={16} />
            <Text style={styles.retryText}>重新请求</Text>
          </Pressable>
        </View>
      ) : null}
      {marketSymbol ? (
        <>
          <CandlestickChart
            candles={marketSymbol.candles}
            height={360}
            indicators={marketSymbol.indicators}
            onSelectCandle={setSelectedCandle}
            onWindowChange={setVisibleWindow}
            priceOverlays={PRICE_OVERLAYS}
          />
          {selectedCandle ? (
            <View style={styles.ohlcStrip}>
              <View style={styles.ohlcHeader}>
                <Text style={styles.ohlcTitle}>已选 K 线</Text>
                <Text style={styles.meta}>{formatTimestamp(selectedCandle.ts)}</Text>
              </View>
              <View style={styles.ohlcGrid}>
                {(
                  [
                    ["开", formatPrice(selectedCandle.open)],
                    ["高", formatPrice(selectedCandle.high)],
                    ["低", formatPrice(selectedCandle.low)],
                    ["收", formatPrice(selectedCandle.close)],
                    ["量", formatPrice(selectedCandle.volume)],
                    [
                      "涨跌",
                      candleChange == null
                        ? "—"
                        : `${candleChange >= 0 ? "+" : ""}${candleChange.toFixed(2)}%`,
                    ],
                  ] as ReadonlyArray<readonly [string, string]>
                ).map(([label, value]) => (
                  <View key={label} style={styles.ohlcItem}>
                    <Text style={styles.ohlcLabel}>{label}</Text>
                    <Text
                      style={[
                        styles.ohlcValue,
                        label === "涨跌" && candleChange != null
                          ? { color: candleChange >= 0 ? palette.positive : palette.negative }
                          : null,
                      ]}
                    >
                      {value}
                    </Text>
                  </View>
                ))}
              </View>
            </View>
          ) : null}

          <View style={styles.indicatorHeader}>
            <View style={styles.headerCopy}>
              <View style={styles.indicatorTitleRow}>
                <SlidersHorizontal color={palette.primary} size={16} />
                <Text style={styles.indicatorTitle}>技术指标</Text>
              </View>
              <Text style={styles.meta}>价格图层：{OVERLAY_SUMMARY}</Text>
            </View>
            <Pressable accessibilityRole="button" onPress={reload} style={styles.reloadButton}>
              <RefreshCw color={palette.primary} size={15} />
              <Text style={styles.reloadText}>刷新</Text>
            </Pressable>
          </View>
          <ScrollView
            contentContainerStyle={styles.chipRow}
            horizontal
            showsHorizontalScrollIndicator={false}
          >
            {LOWER_PANELS.map((option) => (
              <Pressable
                accessibilityRole="button"
                key={option.value}
                onPress={() => setLowerPanel(option.value)}
                style={[styles.smallChip, option.value === lowerPanel && styles.chipSelected]}
              >
                <Text style={[styles.smallChipText, option.value === lowerPanel && styles.chipTextSelected]}>
                  {option.label}
                </Text>
              </Pressable>
            ))}
          </ScrollView>
          <IndicatorChart
            candles={marketSymbol.candles}
            height={168}
            indicators={marketSymbol.indicators}
            panel={lowerPanel}
            selectedTimestamp={selectedCandle?.ts ?? null}
            window={visibleWindow}
          />
        </>
      ) : null}

      {onOpenFull ? (
        <Pressable
          accessibilityRole="button"
          onPress={onOpenFull}
          style={({ pressed }) => [styles.linkRow, pressed && styles.pressed]}
        >
          <View style={styles.rowCopy}>
            <Text style={styles.linkValue}>打开完整行情页</Text>
            <Text style={styles.muted}>在行情页可切换目录标的并自定义日期区间</Text>
          </View>
          <ChevronRight color={palette.textMuted} size={18} />
        </Pressable>
      ) : null}
    </View>
  );
}
const styles = StyleSheet.create({
  block: {
    gap: spacing.sm,
  },
  header: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: spacing.sm,
    justifyContent: "space-between",
  },
  headerCopy: {
    flex: 1,
    gap: 2,
  },
  title: {
    color: palette.text,
    fontSize: 16,
    fontWeight: "700",
  },
  meta: {
    color: palette.textMuted,
    fontSize: 12,
  },
  priceBlock: {
    alignItems: "flex-end",
  },
  priceLabel: {
    color: palette.textMuted,
    fontSize: 11,
    letterSpacing: 0.4,
  },
  priceValue: {
    color: palette.warning,
    fontSize: 17,
    fontWeight: "700",
  },
  priceUnavailable: {
    color: palette.textMuted,
    fontSize: 13,
  },
  chipRow: {
    gap: spacing.xs,
    paddingRight: spacing.sm,
  },
  chip: {
    backgroundColor: palette.surfaceMuted,
    borderColor: palette.border,
    borderRadius: radius.pill,
    borderWidth: 1,
    paddingHorizontal: spacing.sm,
    paddingVertical: 6,
  },
  chipSelected: {
    backgroundColor: palette.primarySoft,
    borderColor: palette.primary,
  },
  chipText: {
    color: palette.textMuted,
    fontSize: 12,
    fontWeight: "600",
  },
  chipTextSelected: {
    color: palette.primary,
  },
  smallChip: {
    backgroundColor: palette.surface,
    borderColor: palette.border,
    borderRadius: radius.pill,
    borderWidth: 1,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
  },
  smallChipText: {
    color: palette.textMuted,
    fontSize: 11,
    fontWeight: "600",
  },
  controlGroup: {
    gap: spacing.xs,
  },
  controlLabel: {
    color: palette.textMuted,
    fontSize: 11,
    letterSpacing: 0.4,
  },
  loading: {
    alignItems: "center",
    flexDirection: "row",
    gap: spacing.xs,
    paddingVertical: spacing.sm,
  },
  muted: {
    color: palette.textMuted,
    fontSize: 12,
  },
  errorPanel: {
    backgroundColor: palette.surfaceMuted,
    borderColor: palette.warning,
    borderRadius: radius.md,
    borderWidth: 1,
    gap: spacing.xs,
    padding: spacing.sm,
  },
  errorTitle: {
    color: palette.text,
    fontSize: 13,
    fontWeight: "700",
  },
  retryButton: {
    alignItems: "center",
    alignSelf: "flex-start",
    backgroundColor: palette.primary,
    borderRadius: radius.pill,
    flexDirection: "row",
    gap: 6,
    paddingHorizontal: spacing.sm,
    paddingVertical: 6,
  },
  retryText: {
    color: palette.canvas,
    fontSize: 12,
    fontWeight: "700",
  },
  ohlcStrip: {
    backgroundColor: palette.surface,
    borderColor: palette.border,
    borderRadius: radius.md,
    borderWidth: 1,
    gap: spacing.xs,
    padding: spacing.sm,
  },
  ohlcHeader: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
  },
  ohlcTitle: {
    color: palette.text,
    fontSize: 12,
    fontWeight: "700",
  },
  ohlcGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.sm,
  },
  ohlcItem: {
    gap: 2,
    minWidth: 56,
  },
  ohlcLabel: {
    color: palette.textMuted,
    fontSize: 10,
  },
  ohlcValue: {
    color: palette.text,
    fontSize: 12,
    fontWeight: "600",
  },
  indicatorHeader: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
  },
  indicatorTitleRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: 6,
  },
  indicatorTitle: {
    color: palette.text,
    fontSize: 14,
    fontWeight: "700",
  },
  reloadButton: {
    alignItems: "center",
    borderColor: palette.border,
    borderRadius: radius.pill,
    borderWidth: 1,
    flexDirection: "row",
    gap: 5,
    paddingHorizontal: spacing.sm,
    paddingVertical: 5,
  },
  reloadText: {
    color: palette.primary,
    fontSize: 11,
    fontWeight: "700",
  },
  linkRow: {
    alignItems: "center",
    backgroundColor: palette.surface,
    borderColor: palette.border,
    borderRadius: radius.md,
    borderWidth: 1,
    flexDirection: "row",
    gap: spacing.sm,
    padding: spacing.sm,
  },
  rowCopy: {
    flex: 1,
    gap: 2,
  },
  linkValue: {
    color: palette.text,
    fontSize: 13,
    fontWeight: "700",
  },
  pressed: {
    opacity: 0.75,
  },
});