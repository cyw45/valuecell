import { useQuery } from "@tanstack/react-query";
import { Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { useNavigation, useRoute } from "@react-navigation/native";
import { Bot, ChartCandlestick, ChevronRight, Wallet } from "lucide-react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { ScreenHeader, SectionCard, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import { formatQuote, formatTimestamp } from "./workbench";

type RouteParams = { params: { strategyId: string } };
type LogPayload = { entries?: Array<Record<string, unknown>> };

function entries(value: unknown): Array<Record<string, unknown>> {
  return ((value as LogPayload | undefined)?.entries ?? []);
}

export default function StrategyDetailScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute() as RouteParams;
  const { session } = useSession();
  const strategyId = route.params.strategyId;
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const strategy = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId], queryFn: () => api.strategy(strategyId), enabled: Boolean(strategyId) });
  const account = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "account"], queryFn: () => api.strategyAccount(strategyId), enabled: Boolean(strategyId && strategy.data?.config.execution.environment !== "okx_demo") });
  const demo = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "demo-execution"], queryFn: () => api.strategyDemoExecution(strategyId), enabled: Boolean(strategyId && strategy.data?.config.execution.environment === "okx_demo"), retry: false });
  const evaluations = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "evaluations", 20], queryFn: () => api.strategyEvaluations(strategyId, 20), enabled: Boolean(strategyId) });
  const signals = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "signals", 20], queryFn: () => api.strategyLog(strategyId, "signals", 20), enabled: Boolean(strategyId) });
  const trades = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "trades", 20], queryFn: () => api.strategyLog(strategyId, "trades", 20), enabled: Boolean(strategyId) });
  const funding = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "funding", 20], queryFn: () => api.strategyLog(strategyId, "funding", 20), enabled: Boolean(strategyId) });
  const refresh = () => void Promise.all([strategy.refetch(), account.refetch(), demo.refetch(), evaluations.refetch(), signals.refetch(), trades.refetch(), funding.refetch()]);
  if (strategy.isLoading) return <StatePanel description="正在读取策略配置和运行诊断。" title="策略详情" />;
  if (strategy.isError || !strategy.data) return <StatePanel error={(strategy.error as Error)?.message ?? "找不到策略。"} onRetry={refresh} state="error" title="策略详情不可用" />;
  const item = strategy.data;
  const config = item.config;
  const latestEvaluation = (evaluations.data ?? [])[0] as unknown as Record<string, unknown> | undefined;
  const mayManage = canMutate(access.data, "strategy.manage");
  const diagnostics = entries(signals.data);

  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={refresh} refreshing={strategy.isRefetching || account.isRefetching || demo.isRefetching} tintColor={palette.primary} />} style={styles.page}>
    <ScreenHeader actionLabel={!item.archived_at && mayManage ? "编辑" : undefined} onAction={!item.archived_at && mayManage ? () => navigation.navigate("StrategyEditor", { strategyId }) : undefined} subtitle={item.archived_at ? "已归档，只读历史" : item.status === "running" ? "策略正在运行" : "策略已停止"} title={item.name} />
    <SectionCard title={config.execution.environment === "okx_demo" ? "Demo 执行来源" : "纸面账户"}>{config.execution.environment === "okx_demo" ? <Text style={styles.row}>{demo.isError ? (demo.error as Error).message : `交易所来源 ${String((demo.data as Record<string, unknown> | undefined)?.source ?? "OKX Demo")} · ${formatTimestamp((demo.data as Record<string, unknown> | undefined)?.checked_at)}`}</Text> : <Text style={styles.row}>权益 {formatQuote((account.data as Record<string, number> | undefined)?.equity_quote)} · 可用 {formatQuote((account.data as Record<string, number> | undefined)?.quote_balance)}</Text>}</SectionCard>
    <SectionCard actionLabel="策略参考" onAction={() => navigation.navigate("StrategyAdvisory", { strategyId })} title="最近评估"><Text style={styles.row}>{String(latestEvaluation?.action ?? "尚无评估")} · {String(latestEvaluation?.reason ?? "服务器将在评估完成后显示原因")}</Text></SectionCard>
    <SectionCard title="信号诊断">{diagnostics.length ? diagnostics.slice(0, 5).map((entry, index) => <View key={index} style={styles.diagnostic}><Text style={styles.diagnosticCode}>{String(entry.code ?? "signal")}</Text><Text style={styles.diagnosticDetail}>{String(entry.detail ?? "—")}</Text></View>) : <Text style={styles.muted}>尚无信号诊断。</Text>}</SectionCard>
    <SectionCard title="最近运行历史"><Text style={styles.row}>成交 {entries(trades.data).length} 条 · 资金费 {entries(funding.data).length} 条</Text><Text style={styles.muted}>列表请求均限制在服务端允许的 500 条以下。</Text></SectionCard>
    <Pressable accessibilityRole="button" onPress={() => navigation.navigate("工作台", { screen: "StrategyOverview", params: { strategyId } })} style={styles.link}><Wallet color={palette.primary} size={19} /><Text style={styles.linkText}>在工作台查看账户</Text><ChevronRight color={palette.textMuted} size={18} /></Pressable>
    <Pressable accessibilityRole="button" onPress={() => navigation.navigate("行情", { screen: "Market", params: { strategyId } })} style={styles.link}><ChartCandlestick color={palette.primary} size={19} /><Text style={styles.linkText}>用该策略查看行情</Text><ChevronRight color={palette.textMuted} size={18} /></Pressable>
    <Pressable accessibilityRole="button" onPress={() => navigation.navigate("StrategyAdvisory", { strategyId })} style={styles.link}><Bot color={palette.primary} size={19} /><Text style={styles.linkText}>打开策略参考</Text><ChevronRight color={palette.textMuted} size={18} /></Pressable>
  </ScrollView>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, row: { color: palette.text, fontSize: 14, lineHeight: 23 }, muted: { color: palette.textMuted, fontSize: 12, lineHeight: 19 }, diagnostic: { borderTopColor: palette.border, borderTopWidth: 1, gap: spacing.xs, paddingVertical: spacing.sm }, diagnosticCode: { color: palette.primary, fontSize: 12, fontWeight: "800" }, diagnosticDetail: { color: palette.text, fontSize: 13, lineHeight: 20 }, link: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 52, paddingHorizontal: spacing.md }, linkText: { color: palette.text, flex: 1, fontSize: 14, fontWeight: "700" },
});
