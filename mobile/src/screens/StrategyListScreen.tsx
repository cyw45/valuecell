import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Pressable, RefreshControl, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { useNavigation } from "@react-navigation/native";
import { Archive, CirclePlay, Pause, Plus, Search } from "lucide-react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { ScreenHeader, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";

type Filter = "all" | "running" | "stopped" | "archived";

export default function StrategyListScreen() {
  const navigation = useNavigation<any>();
  const { session } = useSession();
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const strategies = useQuery({ queryKey: ["mobile", session?.tenantId, "strategies", filter === "archived"], queryFn: () => api.strategies(filter === "archived"), enabled: Boolean(session) });
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const invalidate = () => void queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategies"] });
  const start = useMutation({ mutationFn: api.startStrategy, onSuccess: invalidate });
  const stop = useMutation({ mutationFn: api.stopStrategy, onSuccess: invalidate });
  const archive = useMutation({ mutationFn: api.archiveStrategy, onSuccess: invalidate });
  const rows = useMemo(() => (strategies.data ?? []).filter((item) => {
    if (filter === "archived" ? !item.archived_at : item.archived_at) return false;
    if (filter === "running" && item.status !== "running") return false;
    if (filter === "stopped" && item.status !== "stopped") return false;
    return item.name.toLowerCase().includes(search.trim().toLowerCase()) || item.config.symbols.some((symbol) => symbol.toLowerCase().includes(search.trim().toLowerCase()));
  }), [filter, search, strategies.data]);
  const runMutation = (strategyId: string, action: "start" | "stop") => {
    if (!canMutate(access.data, "strategy.manage")) return;
    const verb = action === "start" ? "启动" : "停止";
    Alert.alert(`${verb}策略`, `${verb}后将按当前执行环境${action === "start" ? "执行" : "停止提交新订单"}。`, [{ text: "取消", style: "cancel" }, { text: verb, onPress: () => void (action === "start" ? start.mutateAsync(strategyId) : stop.mutateAsync(strategyId)).catch((error: Error) => Alert.alert(`${verb}失败`, error.message)) }]);
  };
  const confirmArchive = (strategyId: string) => {
    if (!canMutate(access.data, "strategy.manage")) return;
    Alert.alert("归档策略", "归档会保留评估、成交、资金费与审计历史。策略必须已停止，且不能恢复。", [{ text: "取消", style: "cancel" }, { text: "归档", style: "destructive", onPress: () => void archive.mutateAsync(strategyId).catch((error: Error) => Alert.alert("归档失败", error.message)) }]);
  };
  const busy = start.isPending || stop.isPending || archive.isPending;
  const canManage = canMutate(access.data, "strategy.manage");

  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={() => void strategies.refetch()} refreshing={strategies.isRefetching} tintColor={palette.primary} />} style={styles.page}>
    <ScreenHeader actionLabel={canManage ? "新建" : undefined} onAction={canManage ? () => navigation.navigate("StrategyEditor") : undefined} subtitle="配置、启停与归档服务端策略" title="策略" />
    {!canManage && access.data ? <StatePanel description={access.data.status === "active" ? "当前角色仅具备策略查看权限。" : "服务未激活，策略写入已禁用。"} title="只读访问" /> : null}
    <View style={styles.search}><Search color={palette.textMuted} size={18} /><TextInput accessibilityLabel="搜索策略" onChangeText={setSearch} placeholder="名称或交易对" placeholderTextColor={palette.textMuted} style={styles.searchInput} value={search} /></View>
    <ScrollView contentContainerStyle={styles.filters} horizontal showsHorizontalScrollIndicator={false}>{(["all", "running", "stopped", "archived"] as const).map((value) => <Pressable accessibilityRole="button" key={value} onPress={() => setFilter(value)} style={[styles.filter, filter === value && styles.filterActive]}><Text style={[styles.filterText, filter === value && styles.filterTextActive]}>{({ all: "全部", running: "运行中", stopped: "已停止", archived: "已归档" })[value]}</Text></Pressable>)}</ScrollView>
    {strategies.isLoading ? <StatePanel description="正在读取当前租户的策略。" title="同步中" /> : null}
    {strategies.isError ? <StatePanel actionLabel="重试" description={(strategies.error as Error).message} onAction={() => void strategies.refetch()} title="策略列表暂不可用" tone="error" /> : null}
    {!strategies.isLoading && !strategies.isError && rows.length === 0 ? <StatePanel actionLabel={canManage && filter !== "archived" ? "创建策略" : undefined} description={filter === "archived" ? "没有归档策略。" : "新策略会使用完整的风险、执行与指标配置。"} onAction={canManage && filter !== "archived" ? () => navigation.navigate("StrategyEditor") : undefined} title="没有匹配的策略" /> : null}
    {rows.map((strategy) => <View key={strategy.strategy_id} style={styles.card}><Pressable accessibilityRole="button" onPress={() => navigation.navigate("StrategyDetail", { strategyId: strategy.strategy_id })} style={styles.cardPress}><View style={styles.cardHead}><View style={{ flex: 1 }}><Text style={styles.name}>{strategy.name}</Text><Text style={styles.meta}>{strategy.config.symbols.join(" · ")} · {strategy.config.execution.environment === "okx_demo" ? "OKX Demo" : "纸面交易"}</Text></View><Text style={[styles.status, strategy.archived_at ? styles.archived : strategy.status === "running" ? styles.running : styles.stopped]}>{strategy.archived_at ? "已归档" : strategy.status === "running" ? "运行中" : "已停止"}</Text></View></Pressable>
      {!strategy.archived_at ? <View style={styles.actions}><Pressable accessibilityRole="button" disabled={busy || !canManage} onPress={() => runMutation(strategy.strategy_id, strategy.status === "running" ? "stop" : "start")} style={[styles.action, !canManage && styles.disabled]}><>{strategy.status === "running" ? <Pause color={palette.warning} size={18} /> : <CirclePlay color={palette.positive} size={18} />}</><Text style={styles.actionText}>{strategy.status === "running" ? "停止" : "启动"}</Text></Pressable><Pressable accessibilityRole="button" disabled={busy || !canManage || strategy.status === "running"} onPress={() => confirmArchive(strategy.strategy_id)} style={[styles.action, (!canManage || strategy.status === "running") && styles.disabled]}><Archive color={palette.negative} size={18} /><Text style={[styles.actionText, { color: palette.negative }]}>归档</Text></Pressable></View> : <Text style={styles.history}>归档策略仅保留为只读历史。</Text>}</View>)}
  </ScrollView>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, search: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 48, paddingHorizontal: spacing.sm }, searchInput: { color: palette.text, flex: 1, fontSize: 15, minHeight: 44 }, filters: { gap: spacing.xs }, filter: { borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, minHeight: 44, justifyContent: "center", paddingHorizontal: spacing.md }, filterActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, filterText: { color: palette.textMuted, fontSize: 13, fontWeight: "700" }, filterTextActive: { color: palette.primary }, card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, overflow: "hidden" }, cardPress: { padding: spacing.md }, cardHead: { alignItems: "flex-start", flexDirection: "row", gap: spacing.sm }, name: { color: palette.text, fontSize: 16, fontWeight: "800" }, meta: { color: palette.textMuted, fontSize: 12, marginTop: spacing.xs }, status: { borderRadius: radius.pill, fontSize: 11, fontWeight: "800", overflow: "hidden", paddingHorizontal: spacing.sm, paddingVertical: spacing.xs }, running: { backgroundColor: palette.positiveSoft, color: palette.positive }, stopped: { backgroundColor: palette.surfaceMuted, color: palette.textMuted }, archived: { backgroundColor: palette.warningSoft, color: palette.warning }, actions: { borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.xs, padding: spacing.xs }, action: { alignItems: "center", flex: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 44 }, actionText: { color: palette.text, fontSize: 13, fontWeight: "800" }, disabled: { opacity: 0.45 }, history: { color: palette.textMuted, fontSize: 12, padding: spacing.md },
});
