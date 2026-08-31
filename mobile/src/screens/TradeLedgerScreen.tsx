import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { useNavigation, useRoute, type NavigationProp } from "@react-navigation/native";
import { api } from "../api";
import { BottomSheetSelector, ScreenHeader, StatePanel, TradeDecisionConditions } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import type { WorkbenchStackParamList } from "../navigation/types";
import type { SandboxOrder } from "../types";
import type { ExplanationCondition, UnifiedTradeFact } from "../multi-strategy";
import { formatQuote, formatTimestamp, selectActiveStrategyId } from "./workbench";

const ALL_BATCHES = "__all__";

type RouteParams = { params?: { strategyId?: string; batchId?: string | null } };

type BatchSummary = { batch_id: string; status: string; started_at: string; execution_generation: number };

function numeric(value: number | string | null | undefined): number | null {
  const parsed = typeof value === "string" ? Number(value) : value;
  return typeof parsed === "number" && Number.isFinite(parsed) ? parsed : null;
}

function displayNumber(value: number | string | null | undefined): string {
  const parsed = numeric(value);
  return parsed == null ? "—" : String(parsed);
}

function displayQuote(value: number | string | null | undefined): string {
  return formatQuote(numeric(value));
}

function batchTitle(batch: Pick<BatchSummary, "status" | "started_at">): string {
  return `${batch.status === "running" ? "运行中" : "已归档"} · ${formatTimestamp(batch.started_at)}`;
}

export default function TradeLedgerScreen() {
  const navigation = useNavigation<NavigationProp<WorkbenchStackParamList>>();
  const route = useRoute() as RouteParams;
  const { session } = useSession();
  const [strategyId, setStrategyId] = useState(route.params?.strategyId ?? "");
  const [batchId, setBatchId] = useState<string | null | undefined>(route.params?.batchId);
  const [pickerVisible, setPickerVisible] = useState(false);
  const [batchPickerVisible, setBatchPickerVisible] = useState(false);
  const [page, setPage] = useState(1);
  const strategies = useQuery({ queryKey: ["mobile", session?.tenantId, "strategies"], queryFn: () => api.strategies(false), enabled: Boolean(session) });
  const selectedId = useMemo(() => selectActiveStrategyId(strategies.data ?? [], strategyId), [strategies.data, strategyId]);
  const selectedStrategy = strategies.data?.find((item) => item.strategy_id === selectedId);
  const isDemo = selectedStrategy?.config.execution.environment === "okx_demo";
  const batches = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", selectedId, "batches"], queryFn: () => api.strategyBatches(selectedId), enabled: Boolean(selectedId) });

  useEffect(() => { if (selectedId !== strategyId) setStrategyId(selectedId); }, [selectedId, strategyId]);
  useEffect(() => {
    if (route.params?.batchId !== undefined || batchId !== undefined) return;
    if (batches.data?.current_batch_id) setBatchId(batches.data.current_batch_id);
  }, [batchId, batches.data?.current_batch_id, route.params?.batchId]);
  useEffect(() => setPage(1), [batchId, selectedId]);

  const allHistory = isDemo && batchId === ALL_BATCHES;
  const scopedBatchId = batchId === ALL_BATCHES ? null : batchId;
  const facts = useQuery({
    queryKey: ["mobile", session?.tenantId, "all-trade-facts", selectedId, scopedBatchId ?? "current"],
    queryFn: () => api.allTradeFacts(selectedId, 100, scopedBatchId),
    enabled: Boolean(selectedId && !isDemo),
  });
  const demo = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", selectedId, "demo-execution", scopedBatchId ?? "current", allHistory, page],
    queryFn: () => api.strategyDemoExecution(selectedId, page, 20, scopedBatchId, allHistory),
    enabled: Boolean(selectedId && isDemo), retry: false,
  });
  const pageSize = 20;
  const paperTotalPages = Math.max(1, Math.ceil((facts.data?.length ?? 0) / pageSize));
  const totalPages = isDemo ? Math.max(1, demo.data?.pagination.total_pages ?? 1) : paperTotalPages;
  const pageFacts = (facts.data ?? []).slice((page - 1) * pageSize, page * pageSize);
  const refresh = () => void Promise.all([strategies.refetch(), batches.refetch(), facts.refetch(), demo.refetch()]);
  const selectBatch = (value: string) => { setBatchId(value === "__current__" ? batches.data?.current_batch_id ?? null : value); setBatchPickerVisible(false); };
  const batchOptions = [
    ...(batches.data?.current_batch_id ? [{ label: "当前执行批次", value: "__current__", description: "只读取当前服务端批次" }] : []),
    ...(isDemo ? [{ label: "全部历史订单", value: ALL_BATCHES, description: "跨执行批次查看已归档记录" }] : []),
    ...(batches.data?.items ?? []).map((batch) => ({ label: batchTitle(batch), value: batch.batch_id, description: `代际 ${batch.execution_generation}` })),
  ];
  const currentBatch = (batches.data?.items ?? []).find((item) => item.batch_id === batchId) as BatchSummary | undefined;

  if (strategies.isLoading) return <StatePanel description="正在加载可用策略。" title="交易明细" />;
  if (!selectedId) return <StatePanel description="创建策略并完成评估后，交易会保留在服务端账本中。" title="暂无策略" />;
  const factsError = facts.isError || demo.isError;
  return (
    <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={refresh} refreshing={strategies.isRefetching || batches.isRefetching || facts.isRefetching || demo.isRefetching} tintColor={palette.primary} />} style={styles.page}>
      <ScreenHeader actionLabel="切换策略" onAction={() => setPickerVisible(true)} subtitle={`${isDemo ? "OKX Demo 策略归属订单" : "统一策略归因事实"} · ${allHistory ? "全部历史" : batchId ? "当前批次" : "等待当前批次"}`} title="交易明细" />
      <Pressable accessibilityRole="button" onPress={() => setBatchPickerVisible(true)} style={styles.batchSelector}>
        <View style={styles.batchCopy}><Text style={styles.batchLabel}>执行范围</Text><Text style={styles.batchValue}>{allHistory ? "全部历史订单" : currentBatch ? batchTitle(currentBatch) : batchId ? "当前执行批次" : "当前批次尚未启动"}</Text></View><Text style={styles.batchAction}>切换</Text>
      </Pressable>
      {factsError ? <StatePanel actionLabel="重试" description={((facts.error ?? demo.error) as Error).message} onAction={refresh} title="交易记录暂不可用" tone="error" /> : null}
      {!factsError && (isDemo ? demo.isLoading : facts.isLoading) ? <StatePanel description="正在读取服务端交易与策略依据。" title="正在同步" /> : null}
      {isDemo ? (
        demo.data?.orders.length ? demo.data.orders.map((order) => <DemoOrderCard key={order.id} order={order} batchId={scopedBatchId} navigation={navigation} strategyId={selectedId} strategyKind={selectedStrategy?.strategy_kind} />) : !demo.isLoading && !demo.isError ? <StatePanel description="该策略尚无 OKX Demo 归属订单；不会回退展示纸面成交。" title="暂无 Demo 订单" /> : null
      ) : (
        pageFacts.length ? pageFacts.map((fact) => <UnifiedFactCard key={`${fact.order_id ?? fact.evaluation_id ?? fact.created_at}-${fact.symbol}`} fact={fact} navigation={navigation} />) : !facts.isLoading && !facts.isError ? <StatePanel description="没有已记录的统一策略交易事实。等待下一次服务端策略评估。" title="尚无成交" /> : null
      )}
      {totalPages > 1 ? <View style={styles.pagination}><Pressable accessibilityRole="button" disabled={page <= 1} onPress={() => setPage((current) => Math.max(1, current - 1))} style={[styles.pageButton, page <= 1 && styles.disabled]}><Text style={styles.pageButtonText}>上一页</Text></Pressable><Text style={styles.pageLabel}>{page} / {totalPages}</Text><Pressable accessibilityRole="button" disabled={page >= totalPages} onPress={() => setPage((current) => Math.min(totalPages, current + 1))} style={[styles.pageButton, page >= totalPages && styles.disabled]}><Text style={styles.pageButtonText}>下一页</Text></Pressable></View> : null}
      <BottomSheetSelector onClose={() => setPickerVisible(false)} onSelect={(id) => { setStrategyId(id); setPickerVisible(false); }} options={(strategies.data ?? []).map((item) => ({ label: item.name, value: item.strategy_id }))} selectedValue={selectedId} title="选择策略" visible={pickerVisible} />
      <BottomSheetSelector onClose={() => setBatchPickerVisible(false)} onSelect={selectBatch} options={batchOptions} selectedValue={allHistory ? ALL_BATCHES : batchId ? batchId : "__current__"} title="选择执行范围" visible={batchPickerVisible} />
    </ScrollView>
  );
}

function UnifiedFactCard({ fact, navigation }: { fact: UnifiedTradeFact; navigation: NavigationProp<WorkbenchStackParamList> }) {
  const [expanded, setExpanded] = useState(false);
  const side = fact.side === "buy" || fact.side === "cover" ? "buy" : "sell";
  return <Pressable accessibilityRole="button" onPress={() => fact.symbol && navigation.navigate("StrategyPositions", { strategyId: fact.identity.strategy_id, symbol: fact.symbol, evaluationId: fact.evaluation_id ?? undefined, batchId: fact.batch_id })} style={styles.card}>
    <View style={styles.row}><Text style={[styles.action, side === "buy" ? styles.buy : styles.sell]}>{directionLabel(fact.side)}</Text><Text style={styles.symbol}>{fact.symbol}</Text><Text style={styles.time}>{formatTimestamp(fact.created_at)}</Text></View>
    <View style={styles.identityRow}><Text style={styles.strategyType}>{strategyKindLabel(fact.identity.kind)}</Text><Text style={[styles.status, statusStyle(fact.status)]}>{statusLabel(fact.status)}</Text></View>
    <View style={styles.metrics}><Text style={styles.metric}>请求额 {displayQuote(fact.requested_quote)}</Text><Text style={styles.metric}>成交额 {displayQuote(fact.filled_quote)}</Text><Text style={styles.metric}>数量 {displayNumber(fact.filled_quantity ?? fact.requested_quantity)}</Text><Text style={styles.metric}>均价 {displayQuote(fact.average_fill_price)}</Text><Text style={styles.metric}>手续费 {displayQuote(fact.fee_quote)}</Text></View>
    {fact.failure_reason || fact.failure_code || fact.explanation?.block_reason || fact.status === "submission_unknown" ? <Text style={styles.error}>{fact.failure_reason ?? fact.failure_code ?? fact.explanation?.block_reason ?? "提交结果未确认：正在向原交易所对账，系统不会重新提交订单。"}</Text> : null}
    <Text style={styles.reason}>{fact.explanation?.decision_reason || "服务端未提供策略决策原因。"}{fact.explanation?.block_reason ? `\n启动/执行阻塞：${fact.explanation.block_reason}` : ""}</Text>
    <Pressable accessibilityRole="button" accessibilityState={{ expanded }} onPress={(event) => { event.stopPropagation(); setExpanded((value) => !value); }} style={styles.conditionsToggle}><Text style={styles.conditionsToggleText}>{expanded ? "收起持久化条件" : "查看持久化条件"}</Text><Text style={styles.conditionsCount}>{fact.explanation?.conditions?.length ?? 0} 项</Text></Pressable>
    {expanded ? <PersistedConditions conditions={fact.explanation?.conditions} /> : null}
    <Text style={styles.openHint}>点击查看持仓、盈亏、K 线与完整执行漏斗</Text>
  </Pressable>;
}

function PersistedConditions({ conditions }: { conditions?: readonly ExplanationCondition[] | null }) {
  if (!conditions?.length) return <Text style={styles.missing}>未找到该交易对应的持久化条件记录，未使用当前策略反推。</Text>;
  return <View style={styles.persistedList}>{conditions.map((condition, index) => <View key={`${condition.code}-${index}`} style={styles.persistedCondition}><View style={styles.conditionHeader}><Text style={styles.conditionLabel}>{condition.label || condition.code}</Text><Text style={styles.conditionState}>{conditionStateLabel(condition.state)}</Text></View><Text style={styles.conditionMeta}>{condition.code} · 实际 {displayFactValue(condition.actual)} · 阈值 {displayFactValue(condition.threshold)} · {condition.operator || "无比较符"}</Text><Text style={styles.conditionDetail}>{condition.detail}</Text>{condition.data_at ? <Text style={styles.conditionAt}>数据时间 {formatTimestamp(condition.data_at)}</Text> : null}</View>)}</View>;
}

function strategyKindLabel(kind: UnifiedTradeFact["identity"]["kind"]): string {
  return ({ configurable_rule: "可配置规则", dual_ma_trend: "双均线趋势", pair_rotation: "配对轮动", leader_breakout: "领涨突破" } as Record<string, string>)[kind] ?? kind;
}

function statusLabel(status: UnifiedTradeFact["status"]): string {
  return ({ signal: "信号", blocked: "已拦截", pending: "待处理", submitted: "已提交", submission_unknown: "待远端对账（不可重提）", partially_filled: "部分成交", filled: "已成交", cancelled: "已取消", failed: "失败" } as Record<string, string>)[status] ?? status;
}

function statusStyle(status: UnifiedTradeFact["status"]) {
  if (status === "filled") return styles.statusPositive;
  if (status === "failed" || status === "blocked" || status === "cancelled") return styles.statusNegative;
  if (status === "partially_filled" || status === "pending" || status === "submission_unknown") return styles.statusWarning;
  return styles.statusNeutral;
}

function conditionStateLabel(state: ExplanationCondition["state"]): string {
  return ({ triggered: "已触发", not_triggered: "未触发", blocked: "已拦截", unavailable: "不可用" } as Record<string, string>)[state] ?? state;
}

function displayFactValue(value: ExplanationCondition["actual"]): string {
  if (value === null || value === undefined || value === "") return "—";
  return typeof value === "number" ? displayNumber(value) : String(value);
}

function directionLabel(side: UnifiedTradeFact["side"]): string {
  return side === "buy" ? "买入" : side === "sell" ? "卖出" : side === "short" ? "做空" : "回补";
}

function DemoOrderCard({ order, batchId, navigation, strategyId, strategyKind }: { order: SandboxOrder; batchId: string | null | undefined; navigation: NavigationProp<WorkbenchStackParamList>; strategyId: string; strategyKind?: UnifiedTradeFact["identity"]["kind"] }) {
  return <Pressable accessibilityRole="button" onPress={() => navigation.navigate("StrategyPositions", { strategyId, symbol: order.symbol, orderId: order.id, evaluationId: order.evaluation_id ?? undefined, batchId })} style={styles.card}>
    <View style={styles.row}><Text style={[styles.action, order.side === "buy" ? styles.buy : styles.sell]}>{order.side === "buy" ? "买入" : "卖出"}</Text><Text style={styles.symbol}>{order.symbol}</Text><Text style={styles.time}>{formatTimestamp(order.created_at)}</Text></View>
    <View style={styles.identityRow}><Text style={styles.strategyType}>{strategyKind ? strategyKindLabel(strategyKind) : "策略类型未知"}</Text><Text style={styles.statusNeutral}>{order.status === "submission_unknown" ? "待远端对账（不可重提）" : order.status === "partially_filled" || order.status === "partial" ? "部分成交" : order.status}</Text></View>
    <View style={styles.metrics}><Text style={styles.metric}>状态 {order.status === "submission_unknown" ? "待远端对账" : order.status}</Text><Text style={styles.metric}>委托 {displayQuote(order.requested_quote)}</Text><Text style={styles.metric}>成交量 {displayNumber(order.filled_quantity)}</Text><Text style={styles.metric}>均价 {displayQuote(order.average_fill_price)}</Text><Text style={styles.metric}>手续费 —</Text></View>
    {order.error_message || order.error_code || order.status === "submission_unknown" ? <Text style={styles.error}>{order.error_message ?? order.error_code ?? "提交结果未确认：正在向原交易所对账，系统不会重新提交订单。"}</Text> : null}
    <Text style={styles.reason}>{order.decision_reason ?? order.decision_reason_code ?? "服务端未提供成交原因。"}</Text>
    <TradeDecisionConditions conditions={order.decision_conditions} side={order.side} />
    <Text style={styles.openHint}>点击查看持仓、盈亏、K 线与完整执行漏斗</Text>
  </Pressable>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.sm, padding: spacing.md, paddingBottom: spacing.xl },
  batchSelector: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.primary, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 58, paddingHorizontal: spacing.md },
  batchCopy: { flex: 1, gap: 3 },
  batchLabel: { color: palette.textMuted, fontSize: 11, fontWeight: "800" },
  batchValue: { color: palette.text, fontSize: 14, fontWeight: "900" },
  batchAction: { color: palette.primary, fontSize: 12, fontWeight: "900" },
  card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.md },
  row: { alignItems: "center", flexDirection: "row", gap: spacing.sm },
  action: { borderRadius: radius.pill, fontSize: 12, fontWeight: "800", overflow: "hidden", paddingHorizontal: spacing.sm, paddingVertical: spacing.xs },
  buy: { backgroundColor: palette.positiveSoft, color: palette.positive },
  sell: { backgroundColor: palette.negativeSoft, color: palette.negative },
  symbol: { color: palette.text, flex: 1, fontSize: 16, fontWeight: "800" },
  time: { color: palette.textMuted, fontSize: 11 },
  identityRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
  strategyType: { color: palette.primary, flex: 1, fontSize: 11, fontWeight: "900" },
  status: { borderRadius: radius.pill, fontSize: 10, fontWeight: "900", overflow: "hidden", paddingHorizontal: spacing.xs, paddingVertical: 4 },
  statusPositive: { backgroundColor: palette.positiveSoft, color: palette.positive },
  statusNegative: { backgroundColor: palette.negativeSoft, color: palette.negative },
  statusWarning: { backgroundColor: palette.warningSoft, color: palette.warning },
  statusNeutral: { backgroundColor: palette.surfaceRaised, color: palette.textMuted },
  metrics: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  metric: { backgroundColor: palette.surfaceRaised, borderRadius: radius.sm, color: palette.textMuted, fontSize: 11, paddingHorizontal: spacing.xs, paddingVertical: 5 },
  conditionsToggle: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", justifyContent: "space-between", paddingHorizontal: spacing.sm, paddingVertical: spacing.xs },
  conditionsToggleText: { color: palette.primary, fontSize: 12, fontWeight: "900" },
  conditionsCount: { color: palette.textMuted, fontSize: 11, fontWeight: "800" },
  persistedList: { backgroundColor: palette.surfaceMuted, borderRadius: radius.sm, gap: spacing.xs, padding: spacing.xs },
  persistedCondition: { backgroundColor: palette.surface, borderLeftColor: palette.primary, borderLeftWidth: 3, borderRadius: radius.sm, gap: 3, padding: spacing.xs },
  conditionHeader: { alignItems: "center", flexDirection: "row", justifyContent: "space-between" },
  conditionLabel: { color: palette.text, flex: 1, fontSize: 12, fontWeight: "900" },
  conditionState: { color: palette.primary, fontSize: 10, fontWeight: "900" },
  conditionMeta: { color: palette.textMuted, fontSize: 10, lineHeight: 16 },
  conditionDetail: { color: palette.text, fontSize: 11, lineHeight: 17 },
  conditionAt: { color: palette.textMuted, fontSize: 10 },
  reason: { color: palette.text, fontSize: 13, lineHeight: 20 },
  error: { backgroundColor: palette.negativeSoft, borderRadius: radius.sm, color: palette.negative, fontSize: 12, lineHeight: 18, padding: spacing.xs },
  missing: { color: palette.textMuted, fontSize: 12, lineHeight: 18 },
  openHint: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  pagination: { alignItems: "center", flexDirection: "row", gap: spacing.sm, justifyContent: "center", paddingVertical: spacing.sm },
  pageButton: { borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, justifyContent: "center", minHeight: 40, paddingHorizontal: spacing.md },
  pageButtonText: { color: palette.primary, fontSize: 13, fontWeight: "800" },
  pageLabel: { color: palette.textMuted, fontSize: 13, fontWeight: "800" },
  disabled: { opacity: 0.45 },
});
