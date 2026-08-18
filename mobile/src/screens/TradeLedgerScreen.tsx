import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { useRoute } from "@react-navigation/native";
import { api } from "../api";
import { BottomSheetSelector, ScreenHeader, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import { formatQuote, formatTimestamp, selectActiveStrategyId } from "./workbench";

type RouteParams = { params?: { strategyId?: string } };
type TradeEntry = {
  evaluation_id?: string;
  evaluated_at?: string;
  action?: string;
  symbol?: string;
  quantity?: number;
  quote_amount?: number;
  realized_pnl_quote?: number;
  reason?: string;
};

export default function TradeLedgerScreen() {
  const route = useRoute() as RouteParams;
  const { session } = useSession();
  const [strategyId, setStrategyId] = useState(route.params?.strategyId ?? "");
  const [pickerVisible, setPickerVisible] = useState(false);
  const [page, setPage] = useState(1);
  const strategies = useQuery({ queryKey: ["mobile", session?.tenantId, "strategies"], queryFn: () => api.strategies(false), enabled: Boolean(session) });
  const selectedId = useMemo(
    () => selectActiveStrategyId(strategies.data ?? [], strategyId),
    [strategies.data, strategyId],
  );
  const selectedStrategy = strategies.data?.find((item) => item.strategy_id === selectedId);
  const isDemo = selectedStrategy?.config.execution.environment === "okx_demo";
  const trades = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", selectedId, "trades", 100], queryFn: () => api.strategyLog(selectedId, "trades", 100), enabled: Boolean(selectedId && !isDemo) });
  useEffect(() => { if (selectedId !== strategyId) setStrategyId(selectedId); }, [selectedId, strategyId]);
  const entries = ((trades.data as { entries?: TradeEntry[] } | undefined)?.entries ?? []);
  const pageSize = 20;
  const totalPages = Math.max(1, Math.ceil(entries.length / pageSize));
  const pageEntries = entries.slice((page - 1) * pageSize, page * pageSize);

  if (strategies.isLoading) return <StatePanel description="正在加载可用策略。" title="成交账本" />;
  if (!selectedId) return <StatePanel description="创建策略并完成评估后，成交会保留在服务端账本中。" title="暂无策略" />
  if (isDemo) return <StatePanel description="OKX Demo 的真实订单、成交状态与错误信息来自交易所执行记录；请在策略工作台或策略详情查看。纸面成交账本不会混入 Demo 订单。" title="OKX Demo 交易记录" />
  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={() => void Promise.all([strategies.refetch(), trades.refetch()])} refreshing={strategies.isRefetching || trades.isRefetching} tintColor={palette.primary} />} style={styles.page}>
    <ScreenHeader actionLabel="切换策略" onAction={() => setPickerVisible(true)} subtitle={`服务端归因成交 · 第 ${page}/${totalPages} 页`} title="成交账本" />
    {trades.isError ? <StatePanel actionLabel="重试" description={(trades.error as Error).message} onAction={() => void trades.refetch()} title="成交记录暂不可用" tone="error" /> : null}
    {!trades.isError && entries.length === 0 ? <StatePanel description="没有已成交的策略订单。等待下一次服务端策略评估。" title="尚无成交" /> : null}
    {pageEntries.map((trade, index) => <View key={`${trade.evaluation_id ?? "trade"}-${index}`} style={styles.card}>
      <View style={styles.row}><Text style={[styles.action, trade.action === "buy" ? styles.buy : styles.sell]}>{trade.action === "buy" ? "买入" : "卖出"}</Text><Text style={styles.symbol}>{trade.symbol ?? "—"}</Text><Text style={styles.time}>{formatTimestamp(trade.evaluated_at)}</Text></View>
      <View style={styles.metrics}><Text style={styles.metric}>数量 {trade.quantity ?? "—"}</Text><Text style={styles.metric}>成交额 {formatQuote(trade.quote_amount)}</Text><Text style={styles.metric}>已实现 {formatQuote(trade.realized_pnl_quote)}</Text></View>
      <Text style={styles.reason}>{trade.reason ?? "服务端未提供成交原因。"}</Text>
    </View>)}
    {entries.length > pageSize ? <View style={styles.pagination}><Pressable accessibilityRole="button" disabled={page <= 1} onPress={() => setPage((current) => Math.max(1, current - 1))} style={[styles.pageButton, page <= 1 && styles.disabled]}><Text style={styles.pageButtonText}>上一页</Text></Pressable><Text style={styles.pageLabel}>{page} / {totalPages}</Text><Pressable accessibilityRole="button" disabled={page >= totalPages} onPress={() => setPage((current) => Math.min(totalPages, current + 1))} style={[styles.pageButton, page >= totalPages && styles.disabled]}><Text style={styles.pageButtonText}>下一页</Text></Pressable></View> : null}
    <BottomSheetSelector onClose={() => setPickerVisible(false)} onSelect={(id) => { setStrategyId(id); setPickerVisible(false); }} options={(strategies.data ?? []).map((item) => ({ label: item.name, value: item.strategy_id }))} selectedValue={selectedId} title="选择策略" visible={pickerVisible} />
  </ScrollView>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.sm, padding: spacing.md, paddingBottom: spacing.xl }, card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.md }, row: { alignItems: "center", flexDirection: "row", gap: spacing.sm }, action: { borderRadius: radius.pill, fontSize: 12, fontWeight: "800", overflow: "hidden", paddingHorizontal: spacing.sm, paddingVertical: spacing.xs }, buy: { backgroundColor: palette.positiveSoft, color: palette.positive }, sell: { backgroundColor: palette.negativeSoft, color: palette.negative }, symbol: { color: palette.text, fontSize: 16, fontWeight: "800" }, time: { color: palette.textMuted, flex: 1, fontSize: 11, textAlign: "right" }, metrics: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm }, metric: { color: palette.textMuted, fontSize: 12 }, reason: { color: palette.text, fontSize: 13, lineHeight: 20 },
  pagination: { alignItems: "center", flexDirection: "row", gap: spacing.sm, justifyContent: "center", paddingVertical: spacing.sm },
  pageButton: { borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, minHeight: 40, paddingHorizontal: spacing.md, justifyContent: "center" },
  pageButtonText: { color: palette.primary, fontSize: 13, fontWeight: "800" },
  pageLabel: { color: palette.textMuted, fontSize: 13, fontWeight: "800" },
  disabled: { opacity: 0.45 },
});
