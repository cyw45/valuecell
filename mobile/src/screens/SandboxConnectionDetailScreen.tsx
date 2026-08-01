import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Pressable, RefreshControl, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { useRoute } from "@react-navigation/native";
import { RefreshCw, Send } from "lucide-react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { ScreenHeader, SectionCard, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import { formatTimestamp } from "./workbench";

type RouteParams = { params: { connectionId: string } };
function idempotencyKey(): string { return globalThis.crypto?.randomUUID?.() ?? `mobile-${Date.now()}-${Math.random().toString(16).slice(2)}`; }

export default function SandboxConnectionDetailScreen() {
  const route = useRoute() as RouteParams;
  const { session } = useSession();
  const queryClient = useQueryClient();
  const connectionId = route.params.connectionId;
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [type, setType] = useState<"market" | "limit">("market");
  const [quoteAmount, setQuoteAmount] = useState("100");
  const [price, setPrice] = useState("");
  const orderSubmission = useRef<{ fingerprint: string; key: string } | null>(null);
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const connections = useQuery({ queryKey: ["mobile", session?.tenantId, "sandbox-connections"], queryFn: api.sandboxConnections, enabled: Boolean(session) });
  const balance = useQuery({ queryKey: ["mobile", session?.tenantId, "sandbox", connectionId, "balance"], queryFn: () => api.sandboxBalance(connectionId), enabled: Boolean(connectionId) });
  const positions = useQuery({ queryKey: ["mobile", session?.tenantId, "sandbox", connectionId, "positions"], queryFn: () => api.sandboxPositions(connectionId), enabled: Boolean(connectionId) });
  const connection = connections.data?.find((item) => item.id === connectionId);
  const supportsSymbolCatalog = connection?.provider === "okx";
  const symbols = useQuery({ queryKey: ["mobile", session?.tenantId, "sandbox", connectionId, "symbols"], queryFn: () => api.sandboxSymbols(connectionId), enabled: Boolean(connectionId && supportsSymbolCatalog) });
  const orders = useQuery({ queryKey: ["mobile", session?.tenantId, "sandbox", connectionId, "orders"], queryFn: () => api.sandboxOrders(connectionId, false), enabled: Boolean(connectionId) });
  const orderKey = (request: { symbol: string; side: "buy" | "sell"; type: "market" | "limit"; quote_amount: number; price?: number }) => {
    const fingerprint = JSON.stringify(request);
    if (orderSubmission.current?.fingerprint !== fingerprint) orderSubmission.current = { fingerprint, key: idempotencyKey() };
    return orderSubmission.current.key;
  };
  const createOrder = useMutation({ mutationFn: (request: { symbol: string; side: "buy" | "sell"; type: "market" | "limit"; quote_amount: number; price?: number }) => api.createSandboxOrder({ credential_id: connectionId, sandbox: true, ...request }, orderKey(request)), onSuccess: () => { orderSubmission.current = null; void queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "sandbox", connectionId, "orders"] }); } });
  const canTrade = canMutate(access.data, "trade.execute");
  const refresh = () => {
    void balance.refetch();
    void positions.refetch();
    void orders.refetch();
    if (supportsSymbolCatalog) void symbols.refetch();
  };
  const symbolLabels = useMemo(() => symbols.data?.map((item) => item.symbol).filter(Boolean) ?? [], [symbols.data]);
  const submit = async () => {
    const amount = Number(quoteAmount);
    const limitPrice = Number(price);
    if (!symbol.trim() || !Number.isFinite(amount) || amount <= 0 || (type === "limit" && (!Number.isFinite(limitPrice) || limitPrice <= 0))) { Alert.alert("订单无效", "请填写交易对、正数的 USDT 金额，以及限价单价格。" ); return; }
    Alert.alert("提交模拟盘订单", `${side === "buy" ? "买入" : "卖出"} ${symbol}，金额 ${amount} USDT。`, [{ text: "取消", style: "cancel" }, { text: "提交", onPress: () => void createOrder.mutateAsync({ symbol: symbol.trim().toUpperCase().replace("-", "/"), side, type, quote_amount: amount, ...(type === "limit" ? { price: limitPrice } : {}) }).catch((error: Error) => Alert.alert("提交失败", error.message)) }]);
  };
  if (balance.isLoading) return <StatePanel description="正在读取服务器代理的模拟盘数据。" title="模拟盘详情" />;
  const ordersData = orders.data ?? [];
  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={refresh} refreshing={balance.isRefetching || positions.isRefetching || orders.isRefetching} tintColor={palette.primary} />} style={styles.page}>
    <ScreenHeader subtitle="余额、仓位和订单来自服务器代理的模拟盘；凭据不会回传。" title="模拟盘详情" />
    {(balance.isError || positions.isError || symbols.isError || orders.isError) ? <StatePanel actionLabel="重试" description={((balance.error ?? positions.error ?? symbols.error ?? orders.error) as Error).message} onAction={refresh} title="部分模拟盘数据暂不可用" tone="error" /> : null}
    <SectionCard title="余额"><Text style={styles.caption}>来源 {balance.data?.source ?? "—"} · 核验 {formatTimestamp(balance.data?.checked_at)}</Text>{balance.data?.balances.length ? balance.data.balances.map((item) => <Text key={item.currency} style={styles.row}>{item.currency} · 可用 {item.free} · 总计 {item.total}</Text>) : <Text style={styles.empty}>没有可显示余额。</Text>}</SectionCard>
    <SectionCard title="仓位"><Text style={styles.caption}>来源 {positions.data?.source ?? "—"} · 核验 {formatTimestamp(positions.data?.checked_at)}</Text>{positions.data?.positions.length ? positions.data.positions.map((item) => <Text key={item.symbol} style={styles.row}>{item.symbol} · {item.quantity} · 名义价值 {item.notional_usdt ?? "—"}</Text>) : <Text style={styles.empty}>当前没有模拟盘仓位。</Text>}</SectionCard>
    <SectionCard title="手动模拟盘订单"><Text style={styles.caption}>每次提交生成独立 UUID，并同时发送请求体和 Idempotency-Key 请求头。</Text><Field label="交易对" onChangeText={setSymbol} value={symbol} />{symbolLabels.length ? <ScrollView contentContainerStyle={styles.chips} horizontal showsHorizontalScrollIndicator={false}>{symbolLabels.slice(0, 20).map((item) => <Pressable accessibilityRole="button" key={item} onPress={() => setSymbol(item)} style={[styles.chip, symbol === item && styles.chipActive]}><Text style={[styles.chipText, symbol === item && styles.chipTextActive]}>{item}</Text></Pressable>)}</ScrollView> : null}<View style={styles.options}>{(["buy", "sell"] as const).map((item) => <Choice key={item} onPress={() => setSide(item)} selected={side === item} text={item === "buy" ? "买入" : "卖出"} />)}{(["market", "limit"] as const).map((item) => <Choice key={item} onPress={() => setType(item)} selected={type === item} text={item === "market" ? "市价" : "限价"} />)}</View><Field label="金额（USDT）" onChangeText={setQuoteAmount} value={quoteAmount} />{type === "limit" ? <Field label="限价" onChangeText={setPrice} value={price} /> : null}<Pressable accessibilityRole="button" disabled={!canTrade || createOrder.isPending} onPress={() => void submit()} style={[styles.submit, (!canTrade || createOrder.isPending) && styles.disabled]}><Send color={palette.canvas} size={18} /><Text style={styles.submitText}>{createOrder.isPending ? "正在提交…" : "提交模拟盘订单"}</Text></Pressable></SectionCard>
    <SectionCard actionLabel="刷新订单" onAction={() => void orders.refetch()} title="订单历史">{ordersData.length ? ordersData.map((order) => <View key={order.id} style={styles.order}><Text style={styles.row}>{order.side.toUpperCase()} · {order.symbol} · {order.status}</Text><Pressable accessibilityRole="button" onPress={() => void api.sandboxOrderStatus(order.id).then(() => orders.refetch())} style={styles.statusRefresh}><RefreshCw color={palette.primary} size={16} /><Text style={styles.statusRefreshText}>刷新状态</Text></Pressable></View>) : <Text style={styles.empty}>尚无模拟盘订单。</Text>}</SectionCard>
  </ScrollView>;
}

function Choice({ onPress, selected, text }: { text: string; selected: boolean; onPress: () => void }) { return <Pressable accessibilityRole="button" onPress={onPress} style={[styles.choice, selected && styles.choiceActive]}><Text style={[styles.choiceText, selected && styles.choiceTextActive]}>{text}</Text></Pressable>; }
function Field({ label, onChangeText, value }: { label: string; value: string; onChangeText: (value: string) => void }) { return <View style={styles.field}><Text style={styles.label}>{label}</Text><TextInput accessibilityLabel={label} autoCapitalize="characters" onChangeText={onChangeText} style={styles.input} value={value} /></View>; }

const styles = StyleSheet.create({ page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, caption: { color: palette.textMuted, fontSize: 12, lineHeight: 19 }, row: { color: palette.text, fontSize: 14, lineHeight: 25 }, empty: { color: palette.textMuted, fontSize: 13 }, field: { gap: spacing.xs }, label: { color: palette.textMuted, fontSize: 12, fontWeight: "700" }, input: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, minHeight: 46, paddingHorizontal: spacing.sm }, chips: { gap: spacing.xs }, chip: { borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, minHeight: 40, justifyContent: "center", paddingHorizontal: spacing.sm }, chipActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, chipText: { color: palette.textMuted, fontSize: 12 }, chipTextActive: { color: palette.primary }, options: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs }, choice: { borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, minHeight: 40, justifyContent: "center", paddingHorizontal: spacing.md }, choiceActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, choiceText: { color: palette.textMuted, fontSize: 12, fontWeight: "800" }, choiceTextActive: { color: palette.primary }, submit: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.sm, justifyContent: "center", minHeight: 50 }, submitText: { color: palette.canvas, fontSize: 15, fontWeight: "800" }, disabled: { opacity: 0.45 }, order: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.sm, justifyContent: "space-between", paddingVertical: spacing.sm }, statusRefresh: { alignItems: "center", flexDirection: "row", gap: 4, minHeight: 44 }, statusRefreshText: { color: palette.primary, fontSize: 12, fontWeight: "800" }, });
