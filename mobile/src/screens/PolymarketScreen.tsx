import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ActivityIndicator,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import {
  Activity,
  Database,
  FlaskConical,
  Gauge,
  Layers3,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
} from "lucide-react-native";
import { api } from "../api";
import { palette, radius, spacing } from "../theme";
import type {
  PredictionMarketBookLevel,
  PredictionMarketCatalog,
  PredictionMarketFreshnessStatus,
  PredictionMarketSnapshot,
} from "../types";

type PolymarketScreenProps = {
  route?: {
    params?: {
      marketId?: string;
      outcome?: string;
    };
  };
};

function formatProbability(value: string | number | null | undefined) {
  const numeric = value == null ? Number.NaN : Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(2)}%` : "—";
}

function formatNumber(value: string | number | null | undefined, decimals = 4) {
  const numeric = value == null ? Number.NaN : Number(value);
  return Number.isFinite(numeric) ? numeric.toLocaleString(undefined, { maximumFractionDigits: decimals }) : "—";
}

function formatTimestamp(timestamp: number | null | undefined) {
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

function freshnessColors(status: PredictionMarketFreshnessStatus | undefined) {
  if (status === "fresh") return { backgroundColor: palette.positiveSoft, color: palette.positive };
  if (status === "delayed") return { backgroundColor: palette.warningSoft, color: palette.warning };
  if (status === "stale") return { backgroundColor: palette.warningSoft, color: palette.warning };
  return { backgroundColor: palette.negativeSoft, color: palette.negative };
}

function BookSide({ levels, side }: { levels: PredictionMarketBookLevel[]; side: "bid" | "ask" }) {
  const tone = side === "bid" ? palette.positive : palette.negative;
  const sideLabel = side === "bid" ? "买方深度" : "卖方深度";

  return (
    <View style={styles.depthSide}>
      <Text style={[styles.depthTitle, { color: tone }]}>{sideLabel}</Text>
      <View style={styles.depthHeader}>
        <Text style={styles.depthLabel}>概率</Text>
        <Text style={styles.depthLabel}>合约</Text>
      </View>
      {levels.slice(0, 8).map((level, index) => (
        <View key={`${side}-${level.price}-${index}`} style={styles.depthRow}>
          <Text style={[styles.depthValue, { color: tone }]}>{formatProbability(level.price)}</Text>
          <Text style={styles.depthValue}>{formatNumber(level.size, 2)}</Text>
        </View>
      ))}
      {levels.length === 0 ? <Text style={styles.noDepth}>没有公开 {side === "bid" ? "买" : "卖"} 方深度</Text> : null}
    </View>
  );
}

export default function PolymarketScreen({ route }: PolymarketScreenProps) {
  const routeMarketId = route?.params?.marketId;
  const routeOutcome = route?.params?.outcome;
  const [marketId, setMarketId] = useState(routeMarketId ?? "");
  const [outcome, setOutcome] = useState(routeOutcome ?? "");
  const [query, setQuery] = useState("");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [size, setSize] = useState("100");
  const [latencyMs, setLatencyMs] = useState("250");
  const [history, setHistory] = useState<string[]>([]);
  const observedReferenceKey = useRef<string | null>(null);
  const appliedRouteMarket = useRef<string | null>(null);
  const appliedRouteOutcome = useRef<string | null>(null);

  const catalog = useQuery({
    queryKey: ["mobile", "prediction-markets", "catalog", 50],
    queryFn: () => api.predictionMarketCatalog(50),
  });
  const activeMarkets = useMemo(
    () => catalog.data?.markets.filter((market) => market.active && !market.closed) ?? [],
    [catalog.data?.markets],
  );
  const selectedMarket = useMemo(
    () => catalog.data?.markets.find((market) => market.market_id === marketId),
    [catalog.data?.markets, marketId],
  );
  const filteredMarkets = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    if (!normalizedQuery) return activeMarkets;
    return activeMarkets.filter(
      (market) =>
        market.question.toLocaleLowerCase().includes(normalizedQuery) ||
        market.market_id.toLocaleLowerCase().includes(normalizedQuery) ||
        market.outcomes.some((item) => item.outcome.toLocaleLowerCase().includes(normalizedQuery)),
    );
  }, [activeMarkets, query]);

  useEffect(() => {
    if (!activeMarkets.length) return;
    const shouldApplyRouteMarket =
      routeMarketId != null && routeMarketId !== appliedRouteMarket.current;
    const routeMarket = shouldApplyRouteMarket
      ? activeMarkets.find((market) => market.market_id === routeMarketId)
      : undefined;
    if (shouldApplyRouteMarket) appliedRouteMarket.current = routeMarketId ?? null;
    const currentMarket = activeMarkets.find((market) => market.market_id === marketId);
    const nextMarket = routeMarket ?? currentMarket ?? activeMarkets[0];
    if (nextMarket.market_id !== marketId) {
      setMarketId(nextMarket.market_id);
      setOutcome(nextMarket.outcomes[0]?.outcome ?? "");
      setHistory([]);
      observedReferenceKey.current = null;
    }
  }, [activeMarkets, marketId, routeMarketId]);

  useEffect(() => {
    if (!selectedMarket) return;
    const routeOutcomeKey = routeOutcome
      ? `${routeMarketId ?? selectedMarket.market_id}:${routeOutcome}`
      : null;
    const shouldApplyRouteOutcome =
      routeOutcomeKey != null && routeOutcomeKey !== appliedRouteOutcome.current;
    const routeSelection = shouldApplyRouteOutcome
      ? selectedMarket.outcomes.find((item) => item.outcome === routeOutcome)?.outcome
      : undefined;
    if (shouldApplyRouteOutcome) appliedRouteOutcome.current = routeOutcomeKey;
    const currentSelection = selectedMarket.outcomes.find((item) => item.outcome === outcome)?.outcome;
    const nextOutcome = routeSelection ?? currentSelection ?? selectedMarket.outcomes[0]?.outcome ?? "";
    if (nextOutcome !== outcome) {
      setOutcome(nextOutcome);
      setHistory([]);
      observedReferenceKey.current = null;
    }
  }, [outcome, routeMarketId, routeOutcome, selectedMarket]);

  const snapshot = useQuery({
    queryKey: ["mobile", "prediction-markets", "snapshot", marketId, outcome],
    queryFn: () => api.predictionMarketSnapshot(marketId, outcome),
    enabled: Boolean(marketId && outcome),
  });

  useEffect(() => {
    const reference = snapshot.data?.book.microprice ?? snapshot.data?.book.midpoint;
    const observedAt = snapshot.data?.observed_at_ms;
    if (!reference || observedAt == null) return;
    const key = `${observedAt}:${reference}`;
    if (observedReferenceKey.current === key) return;
    observedReferenceKey.current = key;
    setHistory((current) => [...current, reference].slice(-32));
  }, [snapshot.data?.book.microprice, snapshot.data?.book.midpoint, snapshot.data?.observed_at_ms]);

  const signal = useQuery({
    queryKey: ["mobile", "prediction-markets", "signal", marketId, outcome, history],
    queryFn: () => api.predictionMarketSignal(marketId, outcome, history),
    enabled: Boolean(marketId && outcome && snapshot.data),
  });
  const replay = useMutation({ mutationFn: api.predictionReplayPreview });
  const detail = signal.data ?? snapshot.data;
  const book = detail?.book;
  const numericSize = Number(size);
  const numericLatency = Number(latencyMs);
  const replayEnabled =
    Boolean(detail?.source_timestamp_ms && detail.observed_at_ms && book) &&
    Number.isFinite(numericSize) &&
    numericSize > 0 &&
    Number.isFinite(numericLatency) &&
    numericLatency >= 0 &&
    !replay.isPending;
  const sourceMetadata: PredictionMarketCatalog | PredictionMarketSnapshot | undefined =
    detail ?? catalog.data;
  const sourceFreshness = freshnessColors(sourceMetadata?.freshness_status);
  const dataError = catalog.error ?? snapshot.error ?? signal.error;
  const unavailable =
    Boolean(dataError) ||
    (!catalog.isLoading && catalog.data != null && activeMarkets.length === 0);

  const selectMarket = (nextMarketId: string) => {
    const nextMarket = activeMarkets.find((market) => market.market_id === nextMarketId);
    if (!nextMarket) return;
    setMarketId(nextMarket.market_id);
    setOutcome(nextMarket.outcomes[0]?.outcome ?? "");
    setHistory([]);
    observedReferenceKey.current = null;
  };

  const selectOutcome = (nextOutcome: string) => {
    if (nextOutcome === outcome) return;
    setOutcome(nextOutcome);
    setHistory([]);
    observedReferenceKey.current = null;
  };

  const refresh = () => {
    void Promise.all([catalog.refetch(), snapshot.refetch(), signal.refetch()]);
  };

  const runReplay = () => {
    if (!detail || !book || !replayEnabled) return;
    replay.mutate({
      decision_time_ms: Date.now(),
      latency_ms: numericLatency,
      order: {
        side,
        size: numericSize,
        max_levels: 8,
        extra_slippage_bps: 0,
      },
      snapshots: [
        {
          source_timestamp_ms: detail.source_timestamp_ms,
          observed_at_ms: detail.observed_at_ms,
          bids: book.bids,
          asks: book.asks,
        },
      ],
    });
  };

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl onRefresh={refresh} refreshing={catalog.isRefetching || snapshot.isRefetching || signal.isRefetching} tintColor={palette.primary} />}
      style={styles.page}
    >
      <View style={styles.header}>
        <View style={styles.headerTitleRow}>
          <Text style={styles.title}>预测市场研究</Text>
          <View style={styles.researchBadge}>
            <FlaskConical color={palette.primary} size={14} />
            <Text style={styles.researchBadgeText}>研究与模拟</Text>
          </View>
        </View>
        <Text style={styles.subtitle}>公开 Gamma 目录与 CLOB 订单簿观测。无钱包、签名、账户数据或预测市场实盘下单。</Text>
      </View>

      <View style={styles.catalogCard}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>公开市场目录</Text>
          <Pressable accessibilityLabel="刷新公开预测市场数据" accessibilityRole="button" onPress={refresh} style={styles.refreshButton}>
            <RefreshCw color={palette.primary} size={17} />
            <Text style={styles.refreshText}>刷新</Text>
          </Pressable>
        </View>
        <View style={styles.searchBox}>
          <Search color={palette.textMuted} size={18} />
          <TextInput
            accessibilityLabel="搜索预测市场目录"
            onChangeText={setQuery}
            placeholder="搜索问题、结果或市场 ID"
            placeholderTextColor={palette.textMuted}
            style={styles.searchInput}
            value={query}
          />
        </View>
        {catalog.isLoading && !catalog.data ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={palette.primary} />
            <Text style={styles.mutedText}>正在加载公开目录…</Text>
          </View>
        ) : null}
        {!catalog.isLoading && !catalog.error && filteredMarkets.length === 0 ? (
          <Text style={styles.emptyText}>{query ? "没有匹配的活跃公开市场。" : "当前没有可用的活跃公开市场。"}</Text>
        ) : null}
        <View style={styles.marketList}>
          {filteredMarkets.map((market) => (
            <Pressable
              accessibilityRole="button"
              accessibilityState={{ selected: market.market_id === marketId }}
              key={market.market_id}
              onPress={() => selectMarket(market.market_id)}
              style={[styles.marketOption, market.market_id === marketId && styles.marketOptionActive]}
            >
              <Text numberOfLines={2} style={[styles.marketQuestion, market.market_id === marketId && styles.marketQuestionActive]}>
                {market.question}
              </Text>
              <Text style={styles.marketOptionMeta}>{market.outcomes.map((item) => item.outcome).join(" · ")}</Text>
            </Pressable>
          ))}
        </View>
      </View>

      {selectedMarket ? (
        <View style={styles.outcomeCard}>
          <Text style={styles.outcomeLabel}>结果</Text>
          <Text style={styles.selectedQuestion}>{selectedMarket.question}</Text>
          <View style={styles.outcomeRow}>
            {selectedMarket.outcomes.map((item) => (
              <Pressable
                accessibilityRole="button"
                accessibilityState={{ selected: item.outcome === outcome }}
                key={item.token_id}
                onPress={() => selectOutcome(item.outcome)}
                style={[styles.outcomeChip, item.outcome === outcome && styles.outcomeChipActive]}
              >
                <Text style={[styles.outcomeChipText, item.outcome === outcome && styles.outcomeChipTextActive]}>
                  {item.outcome} {item.price != null ? formatProbability(item.price) : ""}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>
      ) : null}

      <View style={styles.sourceStrip}>
        <View style={styles.sourcePill}>
          <Database color={palette.primary} size={14} />
          <Text style={styles.sourcePillText}>{sourceMetadata?.source ?? "Public source"}</Text>
        </View>
        <Text style={styles.sourceText}>源时间 {formatTimestamp(sourceMetadata?.source_timestamp_ms)}</Text>
        <Text style={styles.sourceText}>观测 {formatTimestamp(sourceMetadata?.observed_at_ms)}</Text>
        <Text style={styles.sourceText}>{formatAge(sourceMetadata?.freshness_age_ms)}</Text>
        <View style={[styles.freshnessBadge, { backgroundColor: sourceFreshness.backgroundColor }]}>
          <Text style={[styles.freshnessText, { color: sourceFreshness.color }]}>{sourceMetadata?.freshness_status ?? "unavailable"}</Text>
        </View>
        {sourceMetadata?.warnings?.map((warning) => <Text key={warning} style={styles.warningText}>{warning}</Text>)}
      </View>

      {unavailable ? (
        <View style={styles.errorPanel}>
          <Text style={styles.errorTitle}>{activeMarkets.length === 0 ? "没有可用公开市场" : "公开市场数据暂不可用"}</Text>
          <Text style={styles.errorText}>{dataError instanceof Error ? dataError.message : "服务端未返回可用于研究的公开快照。"}</Text>
          <Pressable accessibilityRole="button" onPress={refresh} style={styles.retryButton}>
            <RefreshCw color={palette.canvas} size={17} />
            <Text style={styles.retryText}>重新请求</Text>
          </Pressable>
        </View>
      ) : null}

      {!unavailable && !detail && (snapshot.isLoading || signal.isLoading) ? (
        <View style={styles.snapshotLoading}>
          <ActivityIndicator color={palette.primary} />
          <Text style={styles.mutedText}>正在读取所选公开订单簿与研究信号…</Text>
        </View>
      ) : null}

      {detail && !unavailable ? (
        <>
          <View style={styles.metricGrid}>
            {[
              ["参考概率", formatProbability(book?.microprice ?? book?.midpoint), "优先使用微价格"],
              ["最佳买价", formatProbability(book?.best_bid), "公开 CLOB 买价"],
              ["最佳卖价", formatProbability(book?.best_ask), "公开 CLOB 卖价"],
              ["盘口状态", book?.health?.status ?? "unknown", book?.health?.reason ?? "请在研究中验证"],
            ].map(([label, value, hint]) => (
              <View key={label} style={styles.metricCard}>
                <Text style={styles.metricLabel}>{label}</Text>
                <Text style={styles.metricValue}>{value}</Text>
                <Text style={styles.metricHint}>{hint}</Text>
              </View>
            ))}
          </View>

          <View style={styles.depthCard}>
            <View style={styles.depthCardHeader}>
              <Layers3 color={palette.primary} size={18} />
              <View style={styles.depthHeaderCopy}>
                <Text style={styles.sectionTitle}>公开订单簿深度</Text>
                <Text style={styles.mutedText}>显示的层级是观测值，并非可执行报价或流动性保证。</Text>
              </View>
            </View>
            <View style={styles.depthGrid}>
              <BookSide levels={book?.bids ?? []} side="bid" />
              <BookSide levels={book?.asks ?? []} side="ask" />
            </View>
          </View>

          <View style={styles.signalCard}>
            <View style={styles.signalHeader}>
              <Activity color={palette.primary} size={18} />
              <Text style={styles.sectionTitle}>研究信号</Text>
            </View>
            <Text style={styles.signalCopy}>信号只描述本地观测的公开概率变化，不估计事件结果或回报。</Text>
            <View style={styles.signalGrid}>
              <View style={styles.signalItem}>
                <Text style={styles.signalLabel}>参考</Text>
                <Text style={styles.signalValue}>{formatProbability(detail.signal?.reference_price)}</Text>
              </View>
              <View style={styles.signalItem}>
                <Text style={styles.signalLabel}>波动</Text>
                <Text style={styles.signalValue}>{formatNumber(detail.signal?.volatility, 6)}</Text>
              </View>
              <View style={styles.signalItem}>
                <Text style={styles.signalLabel}>方法</Text>
                <Text style={styles.signalValue}>{detail.signal?.reference_method ?? "等待观测"}</Text>
              </View>
              <View style={styles.signalItem}>
                <Text style={styles.signalLabel}>观测数</Text>
                <Text style={styles.signalValue}>{detail.signal?.observation_count ?? history.length}</Text>
              </View>
            </View>
            <View style={styles.caveat}>
              <Gauge color={palette.warning} size={18} />
              <View style={styles.caveatCopy}>
                <Text style={styles.caveatTitle}>{detail.signal?.volatility_status ?? "研究提示"}</Text>
                <Text style={styles.caveatText}>公开数据可能延迟、不完整、过时或不可用；请以其新鲜度与盘口状态为准。</Text>
              </View>
            </View>
          </View>

          <View style={styles.replayCard}>
            <View style={styles.replayHeader}>
              <Play color={palette.primary} size={18} />
              <View style={styles.depthHeaderCopy}>
                <Text style={styles.sectionTitle}>纸面回放</Text>
                <Text style={styles.mutedText}>针对当前接收的冻结公开盘口进行确定性模拟，不会提交实时订单。</Text>
              </View>
            </View>
            <Text style={styles.fieldLabel}>模拟方向</Text>
            <View style={styles.sideRow}>
              {(["buy", "sell"] as const).map((item) => (
                <Pressable
                  accessibilityRole="button"
                  accessibilityState={{ selected: side === item }}
                  key={item}
                  onPress={() => setSide(item)}
                  style={[styles.sideButton, side === item && styles.sideButtonActive]}
                >
                  <Text style={[styles.sideButtonText, side === item && styles.sideButtonTextActive]}>{item === "buy" ? "模拟买入" : "模拟卖出"}</Text>
                </Pressable>
              ))}
            </View>
            <View style={styles.inputGrid}>
              <View style={styles.inputField}>
                <Text style={styles.fieldLabel}>模拟数量</Text>
                <TextInput
                  accessibilityLabel="纸面回放模拟数量"
                  keyboardType="decimal-pad"
                  onChangeText={setSize}
                  placeholder="数量"
                  placeholderTextColor={palette.textMuted}
                  style={styles.numberInput}
                  value={size}
                />
              </View>
              <View style={styles.inputField}>
                <Text style={styles.fieldLabel}>假设延迟（毫秒）</Text>
                <TextInput
                  accessibilityLabel="纸面回放假设延迟毫秒"
                  keyboardType="number-pad"
                  onChangeText={setLatencyMs}
                  placeholder="毫秒"
                  placeholderTextColor={palette.textMuted}
                  style={styles.numberInput}
                  value={latencyMs}
                />
              </View>
            </View>
            <Text style={styles.assumptionText}>假设：{latencyMs || "0"} ms 延迟、可见冻结公开盘口、零额外滑点，未成交余量取消。结果是模拟结果，并非表现保证。</Text>
            <Pressable
              accessibilityRole="button"
              disabled={!replayEnabled}
              onPress={runReplay}
              style={[styles.replayButton, !replayEnabled && styles.controlDisabled]}
            >
              <FlaskConical color={palette.canvas} size={17} />
              <Text style={styles.replayButtonText}>{replay.isPending ? "模拟中" : "运行纸面回放"}</Text>
            </Pressable>
            {replay.error ? <Text style={styles.errorText}>{replay.error instanceof Error ? replay.error.message : "模拟不可用。"}</Text> : null}
            {replay.data ? (
              <View style={styles.resultCard}>
                <Text style={styles.resultTitle}>模拟结果 · {replay.data.simulation_mode}</Text>
                {[
                  ["模拟 VWAP", replay.data.fill.vwap == null ? "—" : formatProbability(replay.data.fill.vwap)],
                  ["成交 / 请求", `${formatNumber(replay.data.fill.filled_size, 2)} / ${formatNumber(replay.data.fill.requested_size, 2)}`],
                  ["未成交余量", formatNumber(replay.data.fill.unfilled_size, 2)],
                  ["按盘口标记 P&L", `${formatNumber(replay.data.mark_to_book.pnl, 4)} ${replay.data.mark_to_book.currency}`],
                ].map(([label, value]) => (
                  <View key={label} style={styles.resultRow}>
                    <Text style={styles.resultLabel}>{label}</Text>
                    <Text style={styles.resultValue}>{value}</Text>
                  </View>
                ))}
              </View>
            ) : null}
          </View>
        </>
      ) : null}

      <View style={styles.safeguardCard}>
        <ShieldAlert color={palette.warning} size={19} />
        <View style={styles.safeguardCopy}>
          <Text style={styles.safeguardTitle}>仅研究与模拟</Text>
          <Text style={styles.safeguardText}>本屏没有钱包连接、签名、账户访问或实时预测市场订单流。</Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  header: { gap: spacing.xs, paddingTop: spacing.sm },
  headerTitleRow: { alignItems: "center", flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  title: { color: palette.text, fontSize: 27, fontWeight: "800", letterSpacing: -0.8 },
  subtitle: { color: palette.textMuted, fontSize: 13, lineHeight: 20 },
  researchBadge: { alignItems: "center", backgroundColor: palette.primarySoft, borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, flexDirection: "row", gap: 5, paddingHorizontal: 8, paddingVertical: 5 },
  researchBadgeText: { color: palette.primary, fontSize: 10, fontWeight: "800" },
  catalogCard: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.md },
  sectionHeader: { alignItems: "center", flexDirection: "row", justifyContent: "space-between" },
  sectionTitle: { color: palette.text, fontSize: 16, fontWeight: "800" },
  refreshButton: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: 6, justifyContent: "center", minHeight: 44, minWidth: 78, paddingHorizontal: spacing.sm },
  refreshText: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  searchBox: { alignItems: "center", backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, paddingHorizontal: spacing.sm },
  searchInput: { color: palette.text, flex: 1, fontSize: 14, height: 46 },
  loadingRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs, minHeight: 44 },
  snapshotLoading: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, justifyContent: "center", minHeight: 160, padding: spacing.md },
  mutedText: { color: palette.textMuted, fontSize: 12, lineHeight: 18 },
  emptyText: { color: palette.textMuted, fontSize: 13, lineHeight: 20 },
  marketList: { gap: spacing.xs },
  marketOption: { backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, gap: 4, minHeight: 56, padding: spacing.sm },
  marketOptionActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  marketQuestion: { color: palette.text, fontSize: 13, fontWeight: "800", lineHeight: 19 },
  marketQuestionActive: { color: palette.primary },
  marketOptionMeta: { color: palette.textMuted, fontSize: 11, lineHeight: 16 },
  outcomeCard: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.xs, padding: spacing.md },
  outcomeLabel: { color: palette.textMuted, fontSize: 11, fontWeight: "800" },
  selectedQuestion: { color: palette.text, fontSize: 14, fontWeight: "800", lineHeight: 20 },
  outcomeRow: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  outcomeChip: { alignItems: "center", backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.sm },
  outcomeChipActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  outcomeChipText: { color: palette.textMuted, fontSize: 12, fontWeight: "800" },
  outcomeChipTextActive: { color: palette.primary },
  sourceStrip: { backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: 5, padding: spacing.sm },
  sourcePill: { alignItems: "center", alignSelf: "flex-start", backgroundColor: palette.primarySoft, borderRadius: radius.pill, flexDirection: "row", gap: 5, paddingHorizontal: 8, paddingVertical: 4 },
  sourcePillText: { color: palette.primary, fontSize: 11, fontWeight: "800" },
  sourceText: { color: palette.textMuted, fontSize: 11, lineHeight: 17 },
  freshnessBadge: { alignSelf: "flex-start", borderRadius: radius.pill, paddingHorizontal: 8, paddingVertical: 4 },
  freshnessText: { fontSize: 11, fontWeight: "800" },
  warningText: { color: palette.warning, fontSize: 12, lineHeight: 18 },
  errorPanel: { alignItems: "flex-start", backgroundColor: palette.negativeSoft, borderColor: palette.negative, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.md },
  errorTitle: { color: palette.negative, fontSize: 16, fontWeight: "800" },
  errorText: { color: palette.text, fontSize: 13, lineHeight: 20 },
  retryButton: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.md },
  retryText: { color: palette.canvas, fontSize: 13, fontWeight: "800" },
  metricGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  metricCard: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexGrow: 1, gap: 4, minWidth: "44%", padding: spacing.sm },
  metricLabel: { color: palette.textMuted, fontSize: 11 },
  metricValue: { color: palette.text, fontSize: 17, fontWeight: "800" },
  metricHint: { color: palette.textMuted, fontSize: 10, lineHeight: 15 },
  depthCard: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.md },
  depthCardHeader: { alignItems: "flex-start", flexDirection: "row", gap: spacing.xs },
  depthHeaderCopy: { flex: 1, gap: 3 },
  depthGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  depthSide: { backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexGrow: 1, gap: 5, minWidth: "44%", padding: spacing.sm },
  depthTitle: { fontSize: 13, fontWeight: "800" },
  depthHeader: { flexDirection: "row", justifyContent: "space-between" },
  depthLabel: { color: palette.textMuted, fontSize: 10, fontWeight: "800" },
  depthRow: { flexDirection: "row", justifyContent: "space-between" },
  depthValue: { color: palette.text, fontSize: 12, fontVariant: ["tabular-nums"] },
  noDepth: { color: palette.textMuted, fontSize: 11, paddingVertical: spacing.sm, textAlign: "center" },
  signalCard: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.md },
  signalHeader: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
  signalCopy: { color: palette.textMuted, fontSize: 12, lineHeight: 19 },
  signalGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  signalItem: { backgroundColor: palette.surfaceMuted, borderRadius: radius.sm, flexGrow: 1, gap: 4, minWidth: "44%", padding: spacing.sm },
  signalLabel: { color: palette.textMuted, fontSize: 10 },
  signalValue: { color: palette.text, fontSize: 13, fontWeight: "800" },
  caveat: { alignItems: "flex-start", backgroundColor: palette.warningSoft, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xs, padding: spacing.sm },
  caveatCopy: { flex: 1, gap: 3 },
  caveatTitle: { color: palette.warning, fontSize: 12, fontWeight: "800" },
  caveatText: { color: palette.text, fontSize: 11, lineHeight: 17 },
  replayCard: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.md },
  replayHeader: { alignItems: "flex-start", flexDirection: "row", gap: spacing.xs },
  fieldLabel: { color: palette.text, fontSize: 12, fontWeight: "800" },
  sideRow: { flexDirection: "row", gap: spacing.xs },
  sideButton: { alignItems: "center", backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flex: 1, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.sm },
  sideButtonActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  sideButtonText: { color: palette.textMuted, fontSize: 13, fontWeight: "800" },
  sideButtonTextActive: { color: palette.primary },
  inputGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  inputField: { flexGrow: 1, gap: 6, minWidth: "44%" },
  numberInput: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, fontSize: 15, height: 48, paddingHorizontal: spacing.sm },
  assumptionText: { color: palette.textMuted, fontSize: 11, lineHeight: 18 },
  replayButton: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.md },
  replayButtonText: { color: palette.canvas, fontSize: 13, fontWeight: "800" },
  controlDisabled: { opacity: 0.45 },
  resultCard: { borderColor: palette.border, borderTopWidth: 1, gap: spacing.xs, marginTop: spacing.xs, paddingTop: spacing.sm },
  resultTitle: { color: palette.primary, fontSize: 13, fontWeight: "800" },
  resultRow: { alignItems: "center", flexDirection: "row", justifyContent: "space-between" },
  resultLabel: { color: palette.textMuted, fontSize: 12 },
  resultValue: { color: palette.text, fontSize: 12, fontWeight: "800", textAlign: "right" },
  safeguardCard: { alignItems: "flex-start", backgroundColor: palette.warningSoft, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, padding: spacing.md },
  safeguardCopy: { flex: 1, gap: 3 },
  safeguardTitle: { color: palette.warning, fontSize: 13, fontWeight: "800" },
  safeguardText: { color: palette.text, fontSize: 12, lineHeight: 19 },
});
