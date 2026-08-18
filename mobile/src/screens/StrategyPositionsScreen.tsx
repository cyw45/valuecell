import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { useNavigation, useRoute, type RouteProp } from "@react-navigation/native";
import { ChartCandlestick, ShieldAlert } from "lucide-react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { ConfirmSheet, DangerButton, EquityCurveChart, SectionCard, StatePanel } from "../components";
import CandlestickChart, { type TradeMarker } from "../components/CandlestickChart";
import type { WorkbenchStackParamList } from "../navigation/types";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import type { RuleStrategyPaperPosition, RuleStrategyTradeLogEntry, SandboxOrder } from "../types";
import { formatQuote, formatTimestamp } from "./workbench";

type Route = RouteProp<WorkbenchStackParamList, "StrategyPositions">;
type Range = "1d" | "5d" | "1w" | "1m";
type PositionView = { symbol: string; quantity: number; entryPrice: number | null; currentPrice: number | null; value: number | null; pnl: number | null };
type PendingClose = { scope: "symbol" | "all"; symbol?: string } | null;

const RANGE_DAYS: Record<Range, number> = { "1d": 1, "5d": 5, "1w": 7, "1m": 31 };

function numberValue(value: number | string | null | undefined): number | null {
  const number = typeof value === "string" ? Number(value) : value;
  return typeof number === "number" && Number.isFinite(number) ? number : null;
}

function canonical(symbol: string): string { return symbol.toUpperCase().replace("/", "-"); }

function entryFromOrders(orders: SandboxOrder[], symbol: string): number | null {
  const target = canonical(symbol);
  let quantity = 0;
  let cost = 0;
  for (const order of orders.filter((item) => canonical(item.symbol) === target && item.status === "filled")) {
    const filledQuantity = numberValue(order.filled_quantity);
    const filledQuote = numberValue(order.filled_quote);
    const averagePrice = numberValue(order.average_fill_price);
    if (!filledQuantity || filledQuantity <= 0) continue;
    const quote = filledQuote ?? (averagePrice == null ? null : filledQuantity * averagePrice);
    if (quote == null) continue;
    if (order.side === "buy") {
      quantity += filledQuantity;
      cost += quote;
    } else if (order.side === "sell" && quantity > 0) {
      const sold = Math.min(quantity, filledQuantity);
      cost -= (cost / quantity) * sold;
      quantity -= sold;
    }
  }
  return quantity > 0 ? cost / quantity : null;
}

function paperPositions(account?: { positions: Record<string, RuleStrategyPaperPosition> }): PositionView[] {
  return Object.entries(account?.positions ?? {}).map(([symbol, position]) => {
    const value = position.quantity * position.mark_price;
    return { symbol, quantity: position.quantity, entryPrice: position.entry_price, currentPrice: position.mark_price, value, pnl: value - position.quantity * position.entry_price };
  });
}

function demoPositions(positions: Array<{ symbol: string; quantity: number; mark_price: number | null; notional_usdt: number | null }>, orders: SandboxOrder[]): PositionView[] {
  return positions.map((position) => {
    const entryPrice = entryFromOrders(orders, position.symbol);
    const currentPrice = numberValue(position.mark_price);
    const value = numberValue(position.notional_usdt);
    return { symbol: canonical(position.symbol), quantity: position.quantity, entryPrice, currentPrice, value, pnl: entryPrice != null && currentPrice != null ? position.quantity * (currentPrice - entryPrice) : null };
  });
}

function paperMarkers(entries: RuleStrategyTradeLogEntry[], symbol: string): TradeMarker[] {
  return entries.filter((entry) => canonical(entry.symbol) === canonical(symbol) && ["entry", "add", "buy"].includes(entry.action)).flatMap((entry) => {
    const price = numberValue(entry.price);
    const ts = Date.parse(entry.evaluated_at);
    return price != null && Number.isFinite(ts) ? [{ ts, price, side: "buy" as const }] : [];
  });
}

function demoMarkers(orders: SandboxOrder[], symbol: string): TradeMarker[] {
  return orders.filter((order) => canonical(order.symbol) === canonical(symbol) && order.side === "buy" && order.status === "filled").flatMap((order) => {
    const price = numberValue(order.average_fill_price);
    const ts = Date.parse(order.filled_at ?? order.updated_at ?? order.created_at);
    return price != null && Number.isFinite(ts) ? [{ ts, price, side: "buy" as const }] : [];
  });
}

function alignMarkers(markers: TradeMarker[], candles: Array<{ ts: number }>): TradeMarker[] {
  return markers.flatMap((marker) => {
    if (!candles.length) return [];
    const candle = candles.reduce((closest, current) => Math.abs(current.ts - marker.ts) < Math.abs(closest.ts - marker.ts) ? current : closest);
    return [{ ...marker, ts: candle.ts }];
  });
}

export default function StrategyPositionsScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<Route>();
  const { session } = useSession();
  const queryClient = useQueryClient();
  const strategyId = route.params.strategyId;
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [range, setRange] = useState<Range>("1m");
  const [pendingClose, setPendingClose] = useState<PendingClose>(null);
  const [closeError, setCloseError] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState("");
  const strategy = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId], queryFn: () => api.strategy(strategyId), enabled: Boolean(session && strategyId) });
  const isDemo = strategy.data?.config.execution.environment === "okx_demo";
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const account = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "account"], queryFn: () => api.strategyAccount(strategyId), enabled: Boolean(strategyId && strategy.data && !isDemo) });
  const trades = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "trades", 500], queryFn: () => api.strategyLog(strategyId, "trades", 500), enabled: Boolean(strategyId && !isDemo) });
  const demo = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "demo-execution", "all"], queryFn: () => api.strategyDemoExecutionAll(strategyId), enabled: Boolean(strategyId && isDemo), retry: false });
  const positions = useMemo(() => isDemo ? demoPositions(demo.data?.positions.data.positions ?? [], demo.data?.orders ?? []) : paperPositions(account.data), [account.data, demo.data, isDemo]);
  useEffect(() => { if (!positions.some((item) => item.symbol === selectedSymbol)) setSelectedSymbol(positions[0]?.symbol ?? ""); }, [positions, selectedSymbol]);
  const selected = positions.find((item) => item.symbol === selectedSymbol) ?? positions[0];
  const toTs = Date.now();
  const fromTs = toTs - RANGE_DAYS[range] * 86_400_000;
  const market = useQuery({ queryKey: ["mobile", "positions", strategyId, selected?.symbol, range], queryFn: () => api.market(selected?.symbol ?? "", "1h", Math.ceil((toTs - fromTs) / 3_600_000) + 2, { from_ts_ms: fromTs, to_ts_ms: toTs }), enabled: Boolean(selected?.symbol) });
  const marketSymbol = market.data?.symbols.find((item) => item.symbol === selected.symbol || canonical(item.symbol) === selected.symbol);
  const markers = alignMarkers(
    isDemo
      ? demoMarkers(demo.data?.orders ?? [], selected?.symbol ?? "")
      : paperMarkers(trades.data?.entries ?? [], selected?.symbol ?? ""),
    marketSymbol?.candles ?? [],
  );
  const closeMutation = useMutation({ mutationFn: (request: { scope: "symbol" | "all"; symbol?: string }) => api.manualCloseStrategy(strategyId, { ...request, confirmation, idempotency_key: globalThis.crypto?.randomUUID?.() ?? `close-${Date.now()}-${Math.random().toString(16).slice(2)}` }), onSuccess: async () => { await Promise.all([queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId] }), queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "demo-execution"] })]); setPendingClose(null); setConfirmation(""); setCloseError(null); }, onError: (error) => setCloseError(error instanceof Error ? error.message : "手动平仓被服务器拒绝。") });
  const totalPnl = isDemo ? numberValue(demo.data?.pnl.total_pnl ?? demo.data?.pnl.total) : (numberValue(account.data?.realized_pnl_quote) ?? 0) + (numberValue(account.data?.unrealized_pnl_quote) ?? 0);
  const expectedConfirmation = pendingClose?.scope === "all" ? "CLOSE ALL POSITIONS" : `CLOSE ${(pendingClose?.symbol ?? "").replace("-", "/")}`;
  const refresh = () => { void Promise.all([strategy.refetch(), account.refetch(), trades.refetch(), demo.refetch(), market.refetch()]); };
  const canManualClose = isDemo && canMutate(access.data, "trade.execute");

  if (strategy.isLoading) return <StatePanel description="正在读取策略持仓。" title="我的持仓" />;
  if (strategy.isError || !strategy.data) return <StatePanel actionLabel="重试" description={(strategy.error as Error)?.message ?? "策略持仓暂不可用。"} onAction={refresh} title="我的持仓暂不可用" tone="error" />;
  return <ScrollView contentContainerStyle={styles.content} refreshControl={undefined} style={styles.page}>
    <SectionCard description={`${isDemo ? "OKX Demo 真实账户" : "纸面账户"} · ${positions.length} 个持仓`} title="持仓总览">
      <View style={styles.summaryGrid}><View style={styles.summaryMetric}><Text style={styles.label}>总收益</Text><Text style={[styles.summaryValue, { color: (totalPnl ?? 0) >= 0 ? palette.positive : palette.negative }]}>{formatQuote(totalPnl)}</Text></View><View style={styles.summaryMetric}><Text style={styles.label}>持仓数量</Text><Text style={styles.summaryValue}>{positions.length}</Text></View></View>
      {canManualClose ? <DangerButton label="一键强平全部策略持仓" leading={<ShieldAlert color={palette.negative} size={18} />} onPress={() => { setConfirmation(""); setPendingClose({ scope: "all" }); }} /> : isDemo ? <Text style={styles.muted}>当前角色没有交易执行权限，无法提交手动平仓。</Text> : <Text style={styles.muted}>纸面持仓可通过策略评估归因；OKX Demo 手动平仓需要真实账户快照和策略成交归属均可验证。</Text>}
    </SectionCard>
    {closeError ? <Text style={styles.error}>{closeError}</Text> : null}
    <SectionCard title="持仓列表">
      {positions.length ? positions.map((position) => <View key={position.symbol} style={styles.positionCard}><Pressable accessibilityRole="button" onPress={() => setSelectedSymbol(position.symbol)} style={[styles.positionSelect, selected?.symbol === position.symbol && styles.selected]}><View style={styles.copy}><Text style={styles.symbol}>{position.symbol.replace("-", "/")}</Text><Text style={styles.muted}>数量 {position.quantity} · 买入价 {formatQuote(position.entryPrice)} · 当前价 {formatQuote(position.currentPrice)}</Text></View><Text style={[styles.pnl, { color: position.pnl == null || position.pnl >= 0 ? palette.positive : palette.negative }]}>{formatQuote(position.pnl)}</Text></Pressable>{canManualClose ? <DangerButton fullWidth={false} label="平仓" onPress={() => { setConfirmation(""); setPendingClose({ scope: "symbol", symbol: position.symbol }); }} /> : null}</View>) : <Text style={styles.muted}>服务端当前没有持仓事实。</Text>}
    </SectionCard>
    {selected ? <SectionCard description="买入点由服务端成交记录与行情时间戳对齐，绿色点为策略买入。" title={`${selected.symbol.replace("-", "/")} K 线`}>
      <View style={styles.rangeRow}>{(Object.keys(RANGE_DAYS) as Range[]).map((item) => <Pressable accessibilityRole="button" key={item} onPress={() => setRange(item)} style={[styles.range, range === item && styles.rangeActive]}><Text style={[styles.rangeText, range === item && styles.rangeTextActive]}>{item}</Text></Pressable>)}</View>
      {market.isError ? <Text style={styles.muted}>{(market.error as Error).message}</Text> : marketSymbol ? <CandlestickChart candles={marketSymbol.candles} height={360} indicators={marketSymbol.indicators} onWindowChange={() => undefined} tradeMarkers={markers} /> : <Text style={styles.muted}>正在读取当前币种行情。</Text>}
    </SectionCard> : null}
    <ConfirmSheet confirmDisabled={confirmation.trim().toUpperCase() !== expectedConfirmation} confirming={closeMutation.isPending} destructive message={`该操作会向 OKX Demo 提交卖出订单。确认文本：${expectedConfirmation}。只会尝试平掉已验证的策略归属持仓，不会清空共享账户其他策略资产。`} onCancel={() => !closeMutation.isPending && setPendingClose(null)} onConfirm={() => pendingClose && closeMutation.mutate(pendingClose)} title="危险操作：手动平仓" visible={Boolean(pendingClose)}><TextInput autoCapitalize="characters" onChangeText={setConfirmation} placeholder={expectedConfirmation} placeholderTextColor={palette.textMuted} style={styles.confirmInput} value={confirmation} /></ConfirmSheet>
  </ScrollView>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  summaryGrid: { flexDirection: "row", gap: spacing.xs },
  summaryMetric: { backgroundColor: palette.surfaceRaised, borderRadius: radius.sm, flex: 1, gap: spacing.xxs, padding: spacing.sm },
  label: { color: palette.textMuted, fontSize: 11, fontWeight: "800" },
  summaryValue: { color: palette.text, fontSize: 18, fontWeight: "900" },
  error: { backgroundColor: palette.negativeSoft, borderColor: palette.negative, borderRadius: radius.sm, borderWidth: 1, color: palette.negative, fontSize: 13, lineHeight: 19, padding: spacing.sm },
  positionCard: { borderTopColor: palette.border, borderTopWidth: 1, gap: spacing.xs, paddingVertical: spacing.sm },
  positionSelect: { alignItems: "center", flexDirection: "row", gap: spacing.sm, minHeight: 60 },
  selected: { backgroundColor: palette.primarySoft, borderRadius: radius.sm, paddingHorizontal: spacing.xs },
  copy: { flex: 1, gap: 3 },
  symbol: { color: palette.text, fontSize: 16, fontWeight: "900" },
  muted: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  pnl: { fontSize: 13, fontWeight: "900" },
  rangeRow: { flexDirection: "row", gap: spacing.xs },
  range: { borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, minHeight: 36, paddingHorizontal: spacing.sm, justifyContent: "center" },
  rangeActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  rangeText: { color: palette.textMuted, fontSize: 12, fontWeight: "800" },
  rangeTextActive: { color: palette.primary },
  confirmInput: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, fontSize: 14, height: 48, paddingHorizontal: spacing.sm },
});
