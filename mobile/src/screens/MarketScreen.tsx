import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ActivityIndicator,
  Linking,
  Modal,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useNavigation, type NavigationProp, type ParamListBase } from "@react-navigation/native";
import {
  CandlestickChart as CandlestickIcon,
  ChevronDown,
  ExternalLink,
  RefreshCw,
  Search,
  SlidersHorizontal,
} from "lucide-react-native";
import { api, MobileApiError } from "../api";
import CandlestickChart, { type ChartWindow, type PriceOverlay } from "../components/CandlestickChart";
import IndicatorChart, { type IndicatorPanel } from "../components/IndicatorChart";
import { useSession } from "../session";
import { marketDataRefreshInterval, usePreferences } from "../preferences";
import { palette, radius, spacing } from "../theme";
import type { CryptoCandle, Strategy } from "../types";

type MarketScreenProps = {
  route?: {
    params?: {
      strategyId?: string;
      symbol?: string;
    };
  };
};

type MarketInterval = "1m" | "3m" | "5m" | "15m" | "30m" | "1h" | "4h" | "1d";
type HistoryRange = "1D" | "5D" | "1W" | "1M" | "10D" | "30D" | "90D" | "1Y" | "custom";
type SymbolScope = "strategy" | "catalog";

const INTERVAL_OPTIONS: ReadonlyArray<{ value: MarketInterval; label: string }> = [
  { value: "1m", label: "1分" },
  { value: "3m", label: "3分" },
  { value: "5m", label: "5分" },
  { value: "15m", label: "15分" },
  { value: "30m", label: "30分" },
  { value: "1h", label: "1小时" },
  { value: "4h", label: "4小时" },
  { value: "1d", label: "1日" },
];
const TIME_RANGE_OPTIONS: ReadonlyArray<{
  value: Exclude<HistoryRange, "custom">;
  label: string;
}> = [
  { value: "1D", label: "1日" },
  { value: "5D", label: "5日" },
  { value: "1W", label: "1周" },
  { value: "1M", label: "1月" },
  { value: "10D", label: "10日" },
  { value: "30D", label: "30日" },
  { value: "90D", label: "90日" },
  { value: "1Y", label: "1年" },
];
const RANGE_DAYS: Record<Exclude<HistoryRange, "custom">, number> = {
  "1D": 1,
  "5D": 5,
  "1W": 7,
  "1M": 30,
  "10D": 10,
  "30D": 30,
  "90D": 90,
  "1Y": 365,
};
const INTERVAL_MS: Record<MarketInterval, number> = {
  "1m": 60_000,
  "3m": 180_000,
  "5m": 300_000,
  "15m": 900_000,
  "30m": 1_800_000,
  "1h": 3_600_000,
  "4h": 14_400_000,
  "1d": 86_400_000,
};
type LowerIndicatorPanel = Exclude<IndicatorPanel, "bollinger">;
const PRICE_OVERLAY_OPTIONS: ReadonlyArray<{ value: PriceOverlay; label: string }> = [
  { value: "ma5", label: "MA5" },
  { value: "ma10", label: "MA10" },
  { value: "ma20", label: "MA20" },
  { value: "ma60", label: "MA60" },
  { value: "bollinger", label: "布林带" },
];
const LOWER_INDICATOR_OPTIONS: ReadonlyArray<{ value: LowerIndicatorPanel; label: string }> = [
  { value: "rsi", label: "RSI" },
  { value: "momentum", label: "动量" },
  { value: "macd", label: "MACD" },
];

function toggleSelection<T extends string>(values: readonly T[], value: T): T[] {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function parseUtcDay(value: string, endOfDay: boolean) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const timestamp = Date.UTC(year, month - 1, day, endOfDay ? 23 : 0, endOfDay ? 59 : 0, endOfDay ? 59 : 0, endOfDay ? 999 : 0);
  const date = new Date(timestamp);
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) {
    return null;
  }
  return timestamp;
}

function formatTimestamp(timestamp: number | string | null | undefined) {
  if (timestamp == null) return "未提供";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "未提供";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatAge(ageMs: number | null | undefined) {
  if (ageMs == null || !Number.isFinite(ageMs)) return "年龄未提供";
  if (ageMs < 60_000) return `${Math.max(0, Math.round(ageMs / 1_000))} 秒前`;
  if (ageMs < 3_600_000) return `${Math.round(ageMs / 60_000)} 分钟前`;
  return `${Math.round(ageMs / 3_600_000)} 小时前`;
}

function formatPrice(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 8 });
}

function selectedStrategy(
  strategies: Strategy[] | undefined,
  requestedStrategyId: string | undefined,
) {
  if (!strategies?.length) return undefined;
  return (
    strategies.find((strategy) => strategy.strategy_id === requestedStrategyId) ??
    strategies.find((strategy) => strategy.status === "running") ??
    strategies[0]
  );
}

export default function MarketScreen({ route }: MarketScreenProps) {
  const navigation = useNavigation<NavigationProp<ParamListBase>>();
  const { session } = useSession();
  const { preferences, ready: preferencesReady } = usePreferences();
  const marketRefreshInterval = preferencesReady
    ? marketDataRefreshInterval(preferences.marketDataRefreshMode)
    : false;
  const tenantId = session?.tenantId ?? "public";
  const requestedStrategyId = route?.params?.strategyId;
  const requestedSymbol = route?.params?.symbol?.trim().toUpperCase().replace("/", "-");
  const [scope, setScope] = useState<SymbolScope>("strategy");
  const [symbol, setSymbol] = useState("");
  const [symbolFilter, setSymbolFilter] = useState("");
  const [interval, setInterval] = useState<MarketInterval>("1h");
  const [range, setRange] = useState<HistoryRange>("10D");
  const [rangeAnchor, setRangeAnchor] = useState(() => Date.now());
  const [dateSheetVisible, setDateSheetVisible] = useState(false);
  const [draftFromDate, setDraftFromDate] = useState("");
  const [draftToDate, setDraftToDate] = useState("");
  const [customFromDate, setCustomFromDate] = useState("");
  const [customToDate, setCustomToDate] = useState("");
  const [priceOverlays, setPriceOverlays] = useState<PriceOverlay[]>(["ma5", "ma20", "bollinger"]);
  const [lowerPanel, setLowerPanel] = useState<LowerIndicatorPanel | null>("rsi");
  const [indicatorSheetVisible, setIndicatorSheetVisible] = useState(false);
  const [visibleWindow, setVisibleWindow] = useState<ChartWindow>();
  const [selectedCandle, setSelectedCandle] = useState<CryptoCandle | null>(null);
  const [externalReferenceError, setExternalReferenceError] = useState<string | null>(null);

  const strategies = useQuery({
    queryKey: ["mobile", tenantId, "strategies", "market"],
    queryFn: () => api.strategies(false),
    enabled: Boolean(session?.tenantId),
  });
  const catalog = useQuery({
    queryKey: ["mobile", "crypto-market", "symbols"],
    queryFn: () => api.cryptoSymbols(),
  });
  const activeStrategy = useMemo(
    () => selectedStrategy(strategies.data, requestedStrategyId),
    [requestedStrategyId, strategies.data],
  );
  const strategySymbols = useMemo(
    () => activeStrategy?.config.symbols.filter(Boolean) ?? [],
    [activeStrategy?.config.symbols],
  );
  const catalogSymbols = catalog.data?.symbols ?? [];
  const allSymbols = useMemo(() => {
    const strategySymbolSet = new Set(strategySymbols);
    return [...strategySymbols, ...catalogSymbols.filter((item) => !strategySymbolSet.has(item))];
  }, [catalogSymbols, strategySymbols]);
  const scopedSymbols = scope === "strategy" ? strategySymbols : allSymbols;
  const filteredSymbols = useMemo(() => {
    const normalizedFilter = symbolFilter.trim().toLocaleLowerCase();
    if (!normalizedFilter) return scopedSymbols;
    return scopedSymbols.filter((item) => item.toLocaleLowerCase().includes(normalizedFilter));
  }, [scopedSymbols, symbolFilter]);

  useEffect(() => {
    if (scope === "strategy" && strategySymbols.length === 0 && allSymbols.length > 0) {
      setScope("catalog");
    }
  }, [allSymbols.length, scope, strategySymbols.length]);
  useEffect(() => {
    if (!requestedSymbol || !allSymbols.includes(requestedSymbol)) return;
    setScope(strategySymbols.includes(requestedSymbol) ? "strategy" : "catalog");
    setSymbol(requestedSymbol);
  }, [allSymbols, requestedSymbol, strategySymbols]);

  useEffect(() => {
    if (!scopedSymbols.includes(symbol)) setSymbol(scopedSymbols[0] ?? "");
  }, [scopedSymbols, symbol]);

  const dateRange = useMemo(() => {
    if (range !== "custom") {
      const toTsMs = rangeAnchor;
      return {
        fromTsMs: toTsMs - RANGE_DAYS[range] * 24 * 60 * 60 * 1_000,
        toTsMs,
        valid: true,
        label:
          TIME_RANGE_OPTIONS.find((option) => option.value === range)?.label ??
          "时间范围",
      };
    }
    const fromTsMs = parseUtcDay(customFromDate, false);
    const toTsMs = parseUtcDay(customToDate, true);
    return {
      fromTsMs,
      toTsMs,
      valid: fromTsMs != null && toTsMs != null && fromTsMs <= toTsMs,
      label: customFromDate && customToDate ? `${customFromDate} 至 ${customToDate}` : "自定义日期",
    };
  }, [customFromDate, customToDate, range, rangeAnchor]);

  useEffect(() => {
    setVisibleWindow(undefined);
    setSelectedCandle(null);
  }, [dateRange.fromTsMs, dateRange.toTsMs, interval, symbol]);

  const chartSnapshotKey = `${symbol}:${interval}:${dateRange.fromTsMs ?? "invalid"}:${dateRange.toTsMs ?? "invalid"}`;
  const lookback = useMemo(() => {
    if (dateRange.fromTsMs == null || dateRange.toTsMs == null) return 1;
    return Math.min(
      5_000,
      Math.max(
        1,
        Math.ceil((dateRange.toTsMs - dateRange.fromTsMs) / INTERVAL_MS[interval]) + 2,
      ),
    );
  }, [dateRange.fromTsMs, dateRange.toTsMs, interval]);
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
        from_ts_ms: dateRange.fromTsMs ?? undefined,
        to_ts_ms: dateRange.toTsMs ?? undefined,
      }),
    enabled: Boolean(symbol) && dateRange.valid,
    refetchInterval: marketRefreshInterval,
  });
  const marketSymbol = market.data?.symbols.find((item) => item.symbol === symbol);
  const failedSymbolReason = symbol ? market.data?.failed_symbols[symbol] : undefined;
  const marketError = market.error instanceof Error ? market.error.message : "行情数据加载失败。";
  const isWarming = market.error instanceof MobileApiError && market.error.status === 503;
  const marketUnavailable =
    !dateRange.valid ||
    market.isError ||
    Boolean(failedSymbolReason) ||
    (Boolean(symbol) && !market.isLoading && !marketSymbol);
  const overlaySummary = priceOverlays.length ? priceOverlays.map((overlay) => PRICE_OVERLAY_OPTIONS.find((option) => option.value === overlay)?.label).filter(Boolean).join(" · ") : "无价格叠加";
  const lowerPanelSummary = lowerPanel ? LOWER_INDICATOR_OPTIONS.find((option) => option.value === lowerPanel)?.label ?? lowerPanel : "不显示";
  const draftFromTs = parseUtcDay(draftFromDate, false);
  const draftToTs = parseUtcDay(draftToDate, true);
  const draftRangeValid =
    draftFromTs != null && draftToTs != null && draftFromTs <= draftToTs;
  const tradingViewUrl = useMemo(() => {
    const match = /^([A-Z0-9]+)-([A-Z0-9]+)$/.exec(symbol.trim().toUpperCase());
    if (!match) return null;
    return `https://www.tradingview.com/symbols/${encodeURIComponent(`${match[1]}${match[2]}`)}/`;
  }, [symbol]);
  const candleChange =
    selectedCandle && selectedCandle.open !== 0
      ? ((selectedCandle.close - selectedCandle.open) / selectedCandle.open) * 100
      : null;

  const refresh = () => {
    if (range !== "custom") setRangeAnchor(Date.now());
    void Promise.all([strategies.refetch(), catalog.refetch(), market.refetch()]);
  };

  const openCustomRange = () => {
    setDraftFromDate(customFromDate);
    setDraftToDate(customToDate);
    setDateSheetVisible(true);
  };

  const applyCustomRange = () => {
    if (!draftRangeValid) return;
    setCustomFromDate(draftFromDate);
    setCustomToDate(draftToDate);
    setRange("custom");
    setRangeAnchor(Date.now());
    setDateSheetVisible(false);
  };

  const openTradingView = async () => {
    if (!tradingViewUrl) return;
    setExternalReferenceError(null);
    try {
      await Linking.openURL(tradingViewUrl);
    } catch {
      setExternalReferenceError("无法打开 TradingView 外部参考。请稍后重试。");
    }
  };

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          onRefresh={refresh}
          refreshing={strategies.isRefetching || catalog.isRefetching || market.isRefetching}
          tintColor={palette.primary}
        />
      }
      style={styles.page}
    >
      <View style={styles.header}>
        <View style={styles.headerCopy}>
          <Text style={styles.eyebrow}>服务端行情 · 研究</Text>
          <Text style={styles.title}>行情研究</Text>
          <Text style={styles.subtitle}>
            {activeStrategy
              ? `${activeStrategy.name} · 优先展示策略观察币种`
              : "从服务端符号目录选择行情，不写入策略配置"}
          </Text>
        </View>
        <View style={styles.headerActions}>
          <Pressable
            accessibilityLabel="打开全球情报研究"
            accessibilityRole="button"
            onPress={() => navigation.navigate("WorldMonitor")}
            style={styles.iconAction}
          >
            <Text style={styles.iconActionText}>全球情报</Text>
          </Pressable>
          <Pressable
            accessibilityLabel="打开预测市场研究"
            accessibilityRole="button"
            onPress={() => navigation.navigate("Polymarket")}
            style={styles.iconAction}
          >
            <Text style={styles.iconActionText}>预测研究</Text>
          </Pressable>
        </View>
      </View>

      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <View style={styles.cardTitleRow}>
            <CandlestickIcon color={palette.primary} size={19} />
            <Text style={styles.cardTitle}>选择观察标的</Text>
          </View>
          <Text style={styles.cardMeta}>
            {scope === "strategy" ? "策略币种优先" : "服务端目录研究"}
          </Text>
        </View>
        <View style={styles.scopeRow}>
          <Pressable
            accessibilityRole="button"
            accessibilityState={{ selected: scope === "strategy" }}
            disabled={strategySymbols.length === 0}
            onPress={() => setScope("strategy")}
            style={[
              styles.scopeButton,
              scope === "strategy" && styles.scopeButtonActive,
              strategySymbols.length === 0 && styles.controlDisabled,
            ]}
          >
            <Text style={[styles.scopeButtonText, scope === "strategy" && styles.scopeButtonTextActive]}>
              策略币种 {strategySymbols.length ? `(${strategySymbols.length})` : ""}
            </Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityState={{ selected: scope === "catalog" }}
            onPress={() => setScope("catalog")}
            style={[styles.scopeButton, scope === "catalog" && styles.scopeButtonActive]}
          >
            <Text style={[styles.scopeButtonText, scope === "catalog" && styles.scopeButtonTextActive]}>
              全部目录 {catalogSymbols.length ? `(${catalogSymbols.length})` : ""}
            </Text>
          </Pressable>
        </View>
        <View style={styles.searchBox}>
          <Search color={palette.textMuted} size={18} />
          <TextInput
            accessibilityLabel="搜索行情符号"
            autoCapitalize="characters"
            onChangeText={setSymbolFilter}
            placeholder="搜索，例如 BTC 或 BTC-USDT"
            placeholderTextColor={palette.textMuted}
            style={styles.searchInput}
            value={symbolFilter}
          />
        </View>
        {catalog.isError ? (
          <View style={styles.inlineError}>
            <Text style={styles.inlineErrorText}>{catalog.error instanceof Error ? catalog.error.message : "符号目录加载失败。"}</Text>
            <Pressable accessibilityRole="button" onPress={() => void catalog.refetch()} style={styles.smallAction}>
              <RefreshCw color={palette.primary} size={16} />
              <Text style={styles.smallActionText}>重试目录</Text>
            </Pressable>
          </View>
        ) : null}
        {catalog.isLoading && !catalog.data ? (
          <View style={styles.catalogLoading}>
            <ActivityIndicator color={palette.primary} />
            <Text style={styles.mutedText}>正在读取可用符号目录…</Text>
          </View>
        ) : null}
        <ScrollView contentContainerStyle={styles.symbolRow} horizontal showsHorizontalScrollIndicator={false}>
          {filteredSymbols.map((item) => (
            <Pressable
              accessibilityRole="button"
              accessibilityState={{ selected: item === symbol }}
              key={item}
              onPress={() => setSymbol(item)}
              style={[styles.symbolChip, item === symbol && styles.symbolChipActive]}
            >
              <Text style={[styles.symbolChipText, item === symbol && styles.symbolChipTextActive]}>
                {item.replace("-", "/")}
              </Text>
            </Pressable>
          ))}
        </ScrollView>
        {!catalog.isLoading && filteredSymbols.length === 0 ? (
          <Text style={styles.emptyText}>
            {scope === "strategy"
              ? "当前策略没有可用观察币种。切换到全部目录可以继续研究。"
              : "没有匹配的服务端目录符号。"}
          </Text>
        ) : null}
      </View>

      <View style={styles.card}>
        <View style={styles.chartTitleRow}>
          <View>
            <Text style={styles.chartTitle}>{symbol ? `${symbol.replace("-", "/")} K 线` : "选择一个标的"}</Text>
            <Text style={styles.cardMeta}>成交量、均线与技术指标均来自同一服务端快照</Text>
          </View>
          <Pressable
            accessibilityLabel="打开外部行情图"
            accessibilityRole="button"
            disabled={!tradingViewUrl}
            onPress={() => void openTradingView()}
            style={[styles.tradingViewButton, !tradingViewUrl && styles.controlDisabled]}
          >
            <ExternalLink color={tradingViewUrl ? palette.primary : palette.textMuted} size={17} />
            <Text style={[styles.tradingViewText, !tradingViewUrl && styles.tradingViewTextDisabled]}>外部图表</Text>
          </Pressable>
        </View>
        {externalReferenceError ? <Text style={styles.errorText}>{externalReferenceError}</Text> : null}

        <Text style={styles.controlLabel}>周期</Text>
        <ScrollView contentContainerStyle={styles.controlRow} horizontal showsHorizontalScrollIndicator={false}>
          {INTERVAL_OPTIONS.map((item) => (
            <Pressable
              accessibilityRole="button"
              accessibilityState={{ selected: interval === item.value }}
              key={item.value}
              onPress={() => setInterval(item.value)}
              style={[styles.controlChip, interval === item.value && styles.controlChipActive]}
            >
              <Text style={[styles.controlChipText, interval === item.value && styles.controlChipTextActive]}>{item.label}</Text>
            </Pressable>
          ))}
        </ScrollView>

        <View style={styles.rangeHeader}>
          <Text style={styles.controlLabel}>时间范围</Text>
          <Text style={styles.rangeSummary}>{dateRange.label}</Text>
        </View>
        <ScrollView contentContainerStyle={styles.controlRow} horizontal showsHorizontalScrollIndicator={false}>
          {TIME_RANGE_OPTIONS.map((option) => (
            <Pressable
              accessibilityRole="button"
              accessibilityState={{ selected: range === option.value }}
              key={option.value}
              onPress={() => {
                setRange(option.value);
                setRangeAnchor(Date.now());
              }}
              style={[styles.controlChip, range === option.value && styles.controlChipActive]}
            >
              <Text style={[styles.controlChipText, range === option.value && styles.controlChipTextActive]}>{option.label}</Text>
            </Pressable>
          ))}
          <Pressable
            accessibilityRole="button"
            accessibilityState={{ selected: range === "custom" }}
            onPress={openCustomRange}
            style={[styles.controlChip, range === "custom" && styles.controlChipActive]}
          >
            <Text style={[styles.controlChipText, range === "custom" && styles.controlChipTextActive]}>日期范围</Text>
          </Pressable>
        </ScrollView>

        {marketSymbol ? (
          <View style={styles.sourceStrip}>
            <View style={styles.sourcePill}>
              <Text style={styles.sourcePillText}>{marketSymbol.provider}</Text>
            </View>
            <Text style={styles.sourceText}>新鲜度：{marketSymbol.freshness_status}</Text>
            <Text style={styles.sourceText}>{formatAge(marketSymbol.freshness_age_ms)}</Text>
            <Text style={styles.sourceText}>覆盖：{marketSymbol.coverage_status}</Text>
            {marketSymbol.snapshot_ts_ms != null ? (
              <Text style={styles.sourceText}>快照 {formatTimestamp(marketSymbol.snapshot_ts_ms)}</Text>
            ) : null}
            {marketSymbol.warning ? <Text style={styles.warningText}>{marketSymbol.warning}</Text> : null}
          </View>
        ) : null}

        {market.isLoading && !market.data ? (
          <View style={styles.statePanel}>
            <ActivityIndicator color={palette.primary} />
            <Text style={styles.mutedText}>正在请求服务端行情快照…</Text>
          </View>
        ) : null}
        {marketUnavailable && !market.isLoading ? (
          <View style={styles.errorPanel}>
            <Text style={styles.errorTitle}>{isWarming ? "行情正在预热" : "当前行情暂不可用"}</Text>
            <Text style={styles.errorText}>
              {!dateRange.valid
                ? "开始日期不能晚于结束日期，并且两者都必须是有效的 YYYY-MM-DD 日期。"
                : failedSymbolReason ?? marketError}
            </Text>
            <Pressable accessibilityRole="button" onPress={() => void market.refetch()} style={styles.retryButton}>
              <RefreshCw color={palette.canvas} size={17} />
              <Text style={styles.retryText}>重新请求</Text>
            </Pressable>
          </View>
        ) : null}
        {marketSymbol && !marketUnavailable ? (
          <>
            <CandlestickChart
              key={chartSnapshotKey}
              candles={marketSymbol.candles}
              height={420}
              indicators={marketSymbol.indicators}
              onSelectCandle={setSelectedCandle}
              onWindowChange={setVisibleWindow}
              priceOverlays={priceOverlays}
            />
            {selectedCandle ? (
              <View style={styles.inspectionStrip}>
                <View style={styles.inspectionHeader}>
                  <Text style={styles.inspectionTitle}>已选 K 线</Text>
                  <Text style={styles.inspectionTimestamp}>{formatTimestamp(selectedCandle.ts)}</Text>
                </View>
                <View style={styles.ohlcGrid}>
                  {[["开", formatPrice(selectedCandle.open)], ["高", formatPrice(selectedCandle.high)], ["低", formatPrice(selectedCandle.low)], ["收", formatPrice(selectedCandle.close)], ["量", formatPrice(selectedCandle.volume)], ["涨跌", candleChange == null ? "—" : `${candleChange >= 0 ? "+" : ""}${candleChange.toFixed(2)}%`]].map(([label, value]) => <View key={label} style={styles.ohlcItem}><Text style={styles.ohlcLabel}>{label}</Text><Text style={[styles.ohlcValue, label === "涨跌" && candleChange != null ? { color: candleChange >= 0 ? palette.positive : palette.negative } : null]}>{value}</Text></View>)}
                </View>
              </View>
            ) : null}
            <View style={styles.indicatorHeader}>
              <View style={styles.indicatorCopy}>
                <Text style={styles.indicatorTitle}>技术指标</Text>
                <Text style={styles.cardMeta}>价格图层：{overlaySummary} · 副图：{lowerPanelSummary}</Text>
              </View>
              <Pressable accessibilityLabel="配置技术指标" accessibilityRole="button" onPress={() => setIndicatorSheetVisible(true)} style={styles.selectorButton}>
                <SlidersHorizontal color={palette.primary} size={17} />
                <Text style={styles.selectorButtonText}>配置</Text>
                <ChevronDown color={palette.textMuted} size={17} />
              </Pressable>
            </View>
            {lowerPanel ? <IndicatorChart candles={marketSymbol.candles} height={184} indicators={marketSymbol.indicators} panel={lowerPanel} selectedTimestamp={selectedCandle?.ts ?? null} window={visibleWindow} /> : <View style={styles.lowerPanelHint}><Text style={styles.mutedText}>副图已关闭。价格指标已直接叠加在同一张 K 线图中。</Text></View>}
          </>
        ) : null}
      </View>

      <View style={styles.researchNote}>
        <Text style={styles.researchNoteTitle}>研究边界</Text>
        <Text style={styles.researchNoteText}>
          图表数据用于研究与策略观察；切换目录标的不会更改策略配置。外部行情图仅作为参考。
        </Text>
      </View>

      <Modal
        animationType="slide"
        onRequestClose={() => setDateSheetVisible(false)}
        transparent
        visible={dateSheetVisible}
      >
        <View style={styles.sheetBackdrop}>
          <Pressable
            accessibilityLabel="关闭日期范围选择"
            accessibilityRole="button"
            onPress={() => setDateSheetVisible(false)}
            style={StyleSheet.absoluteFill}
          />
          <View style={styles.sheet}>
            <Text style={styles.sheetTitle}>自定义日期范围</Text>
            <Text style={styles.sheetCopy}>按协调世界时日期提交；应用后会以服务端时间戳范围请求行情。</Text>
            <Text style={styles.inputLabel}>开始日期</Text>
            <TextInput
              accessibilityLabel="开始日期 YYYY-MM-DD"
              autoCapitalize="none"
              keyboardType="numbers-and-punctuation"
              maxLength={10}
              onChangeText={setDraftFromDate}
              placeholder="YYYY-MM-DD"
              placeholderTextColor={palette.textMuted}
              style={styles.dateInput}
              value={draftFromDate}
            />
            <Text style={styles.inputLabel}>结束日期</Text>
            <TextInput
              accessibilityLabel="结束日期 YYYY-MM-DD"
              autoCapitalize="none"
              keyboardType="numbers-and-punctuation"
              maxLength={10}
              onChangeText={setDraftToDate}
              placeholder="YYYY-MM-DD"
              placeholderTextColor={palette.textMuted}
              style={styles.dateInput}
              value={draftToDate}
            />
            {(draftFromDate || draftToDate) && !draftRangeValid ? (
              <Text style={styles.errorText}>请输入有效日期，且开始日期不能晚于结束日期。</Text>
            ) : null}
            <View style={styles.sheetActions}>
              <Pressable accessibilityRole="button" onPress={() => setDateSheetVisible(false)} style={styles.secondarySheetAction}>
                <Text style={styles.secondarySheetActionText}>取消</Text>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                disabled={!draftRangeValid}
                onPress={applyCustomRange}
                style={[styles.primarySheetAction, !draftRangeValid && styles.controlDisabled]}
              >
                <Text style={styles.primarySheetActionText}>应用范围</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>

      <Modal
        animationType="slide"
        onRequestClose={() => setIndicatorSheetVisible(false)}
        transparent
        visible={indicatorSheetVisible}
      >
        <View style={styles.sheetBackdrop}>
          <Pressable
            accessibilityLabel="关闭技术指标选择"
            accessibilityRole="button"
            onPress={() => setIndicatorSheetVisible(false)}
            style={StyleSheet.absoluteFill}
          />
          <View style={styles.sheet}>
            <Text style={styles.sheetTitle}>图层和副图</Text>
            <Text style={styles.sheetCopy}>均线与布林带共用价格坐标，叠加在同一张 K 线图；RSI、动量、MACD 只保留一个下方副图，避免拆成多张图。</Text>
            <Text style={styles.modalSectionTitle}>价格图层</Text>
            <View style={styles.indicatorOptions}>
              {PRICE_OVERLAY_OPTIONS.map((option) => {
                const selected = priceOverlays.includes(option.value);
                return <Pressable accessibilityRole="checkbox" accessibilityState={{ checked: selected }} key={option.value} onPress={() => setPriceOverlays((current) => toggleSelection(current, option.value))} style={[styles.indicatorOption, selected && styles.indicatorOptionActive]}><Text style={[styles.indicatorOptionText, selected && styles.indicatorOptionTextActive]}>{selected ? "✓ " : ""}{option.label}</Text></Pressable>;
              })}
            </View>
            <Text style={styles.modalSectionTitle}>下方副图</Text>
            <View style={styles.indicatorOptions}>
              <Pressable accessibilityRole="radio" accessibilityState={{ selected: lowerPanel === null }} onPress={() => setLowerPanel(null)} style={[styles.indicatorOption, lowerPanel === null && styles.indicatorOptionActive]}><Text style={[styles.indicatorOptionText, lowerPanel === null && styles.indicatorOptionTextActive]}>不显示副图</Text></Pressable>
              {LOWER_INDICATOR_OPTIONS.map((option) => <Pressable accessibilityRole="radio" accessibilityState={{ selected: lowerPanel === option.value }} key={option.value} onPress={() => setLowerPanel(option.value)} style={[styles.indicatorOption, lowerPanel === option.value && styles.indicatorOptionActive]}><Text style={[styles.indicatorOptionText, lowerPanel === option.value && styles.indicatorOptionTextActive]}>{option.label}</Text></Pressable>)}
            </View>
            <Pressable accessibilityRole="button" onPress={() => setIndicatorSheetVisible(false)} style={styles.primarySheetAction}><Text style={styles.primarySheetActionText}>完成</Text></Pressable>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  header: { gap: spacing.sm, paddingTop: spacing.sm },
  headerCopy: { gap: 3 },
  eyebrow: { color: palette.primary, fontSize: 10, fontWeight: "800", letterSpacing: 1.2 },
  title: { color: palette.text, fontSize: 28, fontWeight: "800", letterSpacing: -0.8 },
  subtitle: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  headerActions: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  iconAction: {
    alignItems: "center",
    backgroundColor: palette.primarySoft,
    borderColor: palette.border,
    borderRadius: radius.sm,
    borderWidth: 1,
    justifyContent: "center",
    minHeight: 44,
    minWidth: 96,
    paddingHorizontal: spacing.sm,
  },
  iconActionText: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  card: {
    backgroundColor: palette.surface,
    borderColor: palette.border,
    borderRadius: radius.md,
    borderWidth: 1,
    gap: spacing.sm,
    padding: spacing.md,
  },
  cardHeader: { alignItems: "center", flexDirection: "row", justifyContent: "space-between" },
  cardTitleRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
  cardTitle: { color: palette.text, fontSize: 16, fontWeight: "800" },
  cardMeta: { color: palette.textMuted, fontSize: 11, lineHeight: 16 },
  scopeRow: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  scopeButton: {
    alignItems: "center",
    backgroundColor: palette.surfaceMuted,
    borderColor: palette.border,
    borderRadius: radius.sm,
    borderWidth: 1,
    justifyContent: "center",
    minHeight: 44,
    paddingHorizontal: spacing.sm,
  },
  scopeButtonActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  scopeButtonText: { color: palette.textMuted, fontSize: 12, fontWeight: "800" },
  scopeButtonTextActive: { color: palette.primary },
  controlDisabled: { opacity: 0.45 },
  searchBox: {
    alignItems: "center",
    backgroundColor: palette.canvas,
    borderColor: palette.border,
    borderRadius: radius.sm,
    borderWidth: 1,
    flexDirection: "row",
    gap: spacing.xs,
    paddingHorizontal: spacing.sm,
  },
  searchInput: { color: palette.text, flex: 1, fontSize: 14, height: 46 },
  inlineError: {
    alignItems: "center",
    backgroundColor: palette.negativeSoft,
    borderColor: palette.negative,
    borderRadius: radius.sm,
    borderWidth: 1,
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.xs,
    justifyContent: "space-between",
    padding: spacing.sm,
  },
  inlineErrorText: { color: palette.text, flex: 1, fontSize: 12, lineHeight: 18, minWidth: 180 },
  smallAction: { alignItems: "center", flexDirection: "row", gap: 6, minHeight: 44, paddingHorizontal: spacing.xs },
  smallActionText: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  catalogLoading: { alignItems: "center", flexDirection: "row", gap: spacing.xs, minHeight: 44 },
  mutedText: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  symbolRow: { gap: spacing.xs },
  symbolChip: {
    alignItems: "center",
    backgroundColor: palette.surfaceMuted,
    borderColor: palette.border,
    borderRadius: radius.pill,
    borderWidth: 1,
    justifyContent: "center",
    minHeight: 44,
    paddingHorizontal: spacing.sm,
  },
  symbolChipActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  symbolChipText: { color: palette.textMuted, fontSize: 12, fontWeight: "800" },
  symbolChipTextActive: { color: palette.primary },
  emptyText: { color: palette.textMuted, fontSize: 13, lineHeight: 20 },
  chartTitleRow: { alignItems: "flex-start", flexDirection: "row", gap: spacing.sm, justifyContent: "space-between" },
  chartTitle: { color: palette.text, fontSize: 18, fontWeight: "800", marginBottom: 3 },
  tradingViewButton: {
    alignItems: "center",
    borderColor: palette.border,
    borderRadius: radius.sm,
    borderWidth: 1,
    flexDirection: "row",
    gap: 5,
    justifyContent: "center",
    minHeight: 44,
    paddingHorizontal: spacing.sm,
  },
  tradingViewText: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  tradingViewTextDisabled: { color: palette.textMuted },
  controlLabel: { color: palette.text, fontSize: 12, fontWeight: "800", marginTop: spacing.xs },
  controlRow: { gap: spacing.xs },
  controlChip: {
    alignItems: "center",
    backgroundColor: palette.surfaceMuted,
    borderColor: palette.border,
    borderRadius: radius.sm,
    borderWidth: 1,
    justifyContent: "center",
    minHeight: 44,
    minWidth: 52,
    paddingHorizontal: spacing.sm,
  },
  controlChipActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  controlChipText: { color: palette.textMuted, fontSize: 12, fontWeight: "800" },
  controlChipTextActive: { color: palette.primary },
  rangeHeader: { alignItems: "center", flexDirection: "row", justifyContent: "space-between" },
  rangeSummary: { color: palette.textMuted, fontSize: 11, maxWidth: "52%", textAlign: "right" },
  sourceStrip: {
    alignItems: "center",
    backgroundColor: palette.surfaceMuted,
    borderColor: palette.border,
    borderRadius: radius.sm,
    borderWidth: 1,
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.xs,
    padding: spacing.sm,
  },
  sourcePill: { alignSelf: "flex-start", backgroundColor: palette.primarySoft, borderRadius: radius.pill, paddingHorizontal: 8, paddingVertical: 4 },
  sourcePillText: { color: palette.primary, fontSize: 11, fontWeight: "800" },
  sourceText: { color: palette.textMuted, fontSize: 11, lineHeight: 16 },
  warningText: { color: palette.warning, flexBasis: "100%", fontSize: 12, lineHeight: 18 },
  statePanel: { alignItems: "center", gap: spacing.sm, justifyContent: "center", minHeight: 220, padding: spacing.md },
  errorPanel: {
    alignItems: "flex-start",
    backgroundColor: palette.negativeSoft,
    borderColor: palette.negative,
    borderRadius: radius.md,
    borderWidth: 1,
    gap: spacing.sm,
    minHeight: 200,
    justifyContent: "center",
    padding: spacing.md,
  },
  errorTitle: { color: palette.negative, fontSize: 16, fontWeight: "800" },
  errorText: { color: palette.text, fontSize: 13, lineHeight: 20 },
  retryButton: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.md },
  retryText: { color: palette.canvas, fontSize: 13, fontWeight: "800" },
  inspectionStrip: { backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, gap: spacing.sm, padding: spacing.sm },
  inspectionHeader: { alignItems: "center", flexDirection: "row", justifyContent: "space-between" },
  inspectionTitle: { color: palette.text, fontSize: 12, fontWeight: "800" },
  inspectionTimestamp: { color: palette.textMuted, fontSize: 11 },
  ohlcGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  ohlcItem: { backgroundColor: palette.canvas, borderRadius: radius.sm, flexGrow: 1, gap: 3, minWidth: "28%", padding: spacing.xs },
  ohlcLabel: { color: palette.textMuted, fontSize: 10 },
  ohlcValue: { color: palette.text, fontSize: 12, fontWeight: "800" },
  indicatorHeader: { alignItems: "center", flexDirection: "row", gap: spacing.sm, justifyContent: "space-between", marginTop: spacing.xs },
  indicatorCopy: { flex: 1 },
  lowerPanelHint: { alignItems: "center", backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, minHeight: 72, justifyContent: "center", padding: spacing.sm },
  indicatorTitle: { color: palette.text, fontSize: 15, fontWeight: "800", marginBottom: 2 },
  selectorButton: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: 5, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.sm },
  selectorButtonText: { color: palette.text, fontSize: 12, fontWeight: "800" },
  researchNote: { backgroundColor: palette.primarySoft, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: 4, padding: spacing.md },
  researchNoteTitle: { color: palette.primary, fontSize: 13, fontWeight: "800" },
  researchNoteText: { color: palette.textMuted, fontSize: 12, lineHeight: 19 },
  sheetBackdrop: { backgroundColor: "rgba(0,0,0,0.62)", flex: 1, justifyContent: "flex-end" },
  sheet: { backgroundColor: palette.surface, borderColor: palette.border, borderTopLeftRadius: radius.lg, borderTopRightRadius: radius.lg, borderWidth: 1, gap: spacing.sm, padding: spacing.md, paddingBottom: spacing.lg },
  sheetTitle: { color: palette.text, fontSize: 18, fontWeight: "800" },
  sheetCopy: { color: palette.textMuted, fontSize: 12, lineHeight: 19 },
  modalSectionTitle: { color: palette.text, fontSize: 13, fontWeight: "900", marginTop: spacing.xs },
  inputLabel: { color: palette.text, fontSize: 12, fontWeight: "800", marginTop: spacing.xs },
  dateInput: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, fontSize: 15, height: 48, paddingHorizontal: spacing.sm },
  sheetActions: { flexDirection: "row", gap: spacing.xs, justifyContent: "flex-end", marginTop: spacing.sm },
  secondarySheetAction: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, justifyContent: "center", minHeight: 44, minWidth: 92, paddingHorizontal: spacing.sm },
  secondarySheetActionText: { color: palette.text, fontSize: 13, fontWeight: "800" },
  primarySheetAction: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, justifyContent: "center", minHeight: 44, minWidth: 104, paddingHorizontal: spacing.sm },
  primarySheetActionText: { color: palette.canvas, fontSize: 13, fontWeight: "800" },
  indicatorOptions: { gap: spacing.xs, marginTop: spacing.xs },
  indicatorOption: { alignItems: "center", backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.sm },
  indicatorOptionActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  indicatorOptionText: { color: palette.text, fontSize: 14, fontWeight: "800" },
  indicatorOptionTextActive: { color: palette.primary },
});
