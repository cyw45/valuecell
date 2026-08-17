import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useNavigation, useRoute, type RouteProp } from "@react-navigation/native";
import { api } from "../api";
import { StatePanel } from "../components";
import type { WorkbenchStackParamList } from "../navigation/types";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import type { RuleStrategyDemoExecution } from "../types";
import { orderSideLabel, orderStatusLabel, orderTypeLabel } from "./strategy-presentation";
import { formatQuote, formatTimestamp } from "./workbench";
type Route = RouteProp<WorkbenchStackParamList, "ExecutionFacts">;
type FactKind = "positions" | "balances" | "orders";

const pageCopy: Record<FactKind, { title: string; empty: string }> = {
  positions: { title: "交易所仓位", empty: "交易所当前没有返回持仓。" },
  balances: { title: "交易所余额", empty: "交易所当前没有返回余额。" },
  orders: { title: "策略归属订单", empty: "当前没有归因到该策略的交易所订单。" },
};

function formatNumericQuote(value: number | string | null | undefined): string {
  const number = typeof value === "string" ? Number(value) : value;
  return typeof number === "number" && Number.isFinite(number) ? formatQuote(number) : "—";
}

export default function ExecutionFactsScreen() {
  const route = useRoute<Route>();
  const { session } = useSession();
  const navigation = useNavigation<any>();
  const { strategyId, kind } = route.params;
  const [page, setPage] = useState(1);
  const strategy = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", strategyId],
    queryFn: () => api.strategy(strategyId),
    enabled: Boolean(session && strategyId),
  });
  const isDemo = strategy.data?.config.execution.environment === "okx_demo";
  const execution = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "demo-execution", page],
    queryFn: () => api.strategyDemoExecution(strategyId, page, 20),
    enabled: Boolean(strategyId && isDemo),
    retry: false,
  });
  const paperAccount = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "account"],
    queryFn: () => api.strategyAccount(strategyId),
    enabled: Boolean(strategyId && strategy.data && !isDemo && kind === "positions"),
  });
  const loading = strategy.isLoading || execution.isLoading || paperAccount.isLoading;
  const error = strategy.error ?? execution.error ?? paperAccount.error;
  const detail = pageCopy[kind];
  useEffect(() => {
    navigation.setOptions({ title: detail.title });
  }, [detail.title, navigation]);
  const paperPositions = useMemo(() => Object.entries(paperAccount.data?.positions ?? {}), [paperAccount.data?.positions]);
  const demo = execution.data;

  if (loading) return <StatePanel description="正在读取服务端执行事实。" title={detail.title} />;
  if (error || !strategy.data) return <StatePanel actionLabel="重试" description={(error as Error)?.message ?? "无法读取执行事实。"} onAction={() => { void strategy.refetch(); void execution.refetch(); void paperAccount.refetch(); }} title={`${detail.title}暂不可用`} tone="error" />;
  if (!isDemo && kind !== "positions") return <StatePanel description="纸面策略不使用交易所共享账户；此信息仅对 OKX Demo 策略提供。" title={detail.title} />;
  if (!isDemo && kind === "positions") return <PaperPositions positions={paperPositions} />;
  if (!demo) return <StatePanel description="交易所执行数据尚未返回。" title={detail.title} />;

  if (kind === "positions") return <DemoPositions positions={demo.positions.data.positions} />;
  if (kind === "balances") return <DemoBalances balances={demo.account.data.balances} />;
  return <DemoOrders page={page} setPage={setPage} execution={demo} />;
}

function PaperPositions({ positions }: { positions: Array<[string, { quantity: number; mark_price: number }]> }) {
  return <FactsList empty="纸面账本当前没有持仓。" title="纸面账户仓位">{positions.map(([symbol, position]) => <FactRow detail={`数量 ${position.quantity} · 标记价 ${position.mark_price}`} key={symbol} title={symbol} value={formatQuote(position.quantity * position.mark_price)} />)}</FactsList>;
}

function DemoPositions({ positions }: { positions: Array<{ symbol: string; quantity: number; available_quantity: number; notional_usdt: number | null }> }) {
  return <FactsList empty={pageCopy.positions.empty} title={pageCopy.positions.title}>{positions.map((position) => <FactRow detail={`数量 ${position.quantity} · 可用 ${position.available_quantity}`} key={position.symbol} title={position.symbol} value={formatQuote(position.notional_usdt)} />)}</FactsList>;
}

function DemoBalances({ balances }: { balances: Array<{ currency: string; free: string | number; total: string | number; usdt_value: number | null }> }) {
  return <FactsList empty={pageCopy.balances.empty} title={pageCopy.balances.title}>{balances.map((balance) => <FactRow detail={`可用 ${balance.free} · 总计 ${balance.total}`} key={balance.currency} title={balance.currency} value={formatQuote(balance.usdt_value)} />)}</FactsList>;
}

function DemoOrders({ execution, page, setPage }: { execution: RuleStrategyDemoExecution; page: number; setPage: (page: number) => void }) {
  return <FactsList empty={pageCopy.orders.empty} title={`策略归属订单 · 第 ${execution.pagination.page}/${execution.pagination.total_pages || 1} 页`}>
    {execution.orders.map((order) => <FactRow detail={`${orderTypeLabel(order.type)} · 请求 ${formatNumericQuote(order.requested_quote)} · ${formatTimestamp(order.updated_at)}${order.error_code ? ` · ${order.error_code}` : ""}`} key={order.id} title={`${orderSideLabel(order.side)} · ${order.symbol}`} value={orderStatusLabel(order.status)} />)}
    {execution.pagination.total_pages > 1 ? <View style={styles.pagination}><Pressable accessibilityRole="button" disabled={page <= 1} onPress={() => setPage(Math.max(1, page - 1))} style={[styles.pageButton, page <= 1 && styles.disabled]}><Text style={styles.pageButtonText}>上一页</Text></Pressable><Pressable accessibilityRole="button" disabled={page >= execution.pagination.total_pages} onPress={() => setPage(Math.min(execution.pagination.total_pages, page + 1))} style={[styles.pageButton, page >= execution.pagination.total_pages && styles.disabled]}><Text style={styles.pageButtonText}>下一页</Text></Pressable></View> : null}
  </FactsList>;
}

function FactsList({ children, empty, title }: { children: ReactNode; empty: string; title: string }) {
  const hasItems = Array.isArray(children) ? children.length > 0 : Boolean(children);
  return <ScrollView contentContainerStyle={styles.content} style={styles.page}><View style={styles.card}><Text style={styles.title}>{title}</Text>{hasItems ? <View style={styles.list}>{children}</View> : <Text style={styles.empty}>{empty}</Text>}</View></ScrollView>;
}

function FactRow({ detail, title, value }: { detail: string; title: string; value: string }) {
  return <View style={styles.row}><View style={styles.copy}><Text style={styles.rowTitle}>{title}</Text><Text style={styles.detail}>{detail}</Text></View><Text style={styles.value}>{value}</Text></View>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { padding: spacing.md, paddingBottom: spacing.xl },
  card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.md },
  title: { color: palette.text, fontSize: 16, fontWeight: "900" },
  list: { gap: 0 },
  row: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 62, paddingVertical: spacing.sm },
  copy: { flex: 1, gap: 3 },
  rowTitle: { color: palette.text, fontSize: 14, fontWeight: "800" },
  detail: { color: palette.textMuted, fontSize: 12, lineHeight: 17 },
  value: { color: palette.text, fontSize: 12, fontWeight: "900", textAlign: "right" },
  empty: { color: palette.textMuted, fontSize: 13, paddingVertical: spacing.lg, textAlign: "center" },
  pagination: { flexDirection: "row", gap: spacing.sm, justifyContent: "flex-end", paddingTop: spacing.sm },
  pageButton: { borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, minHeight: 40, paddingHorizontal: spacing.md, justifyContent: "center" },
  pageButtonText: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  disabled: { opacity: 0.45 },
});
