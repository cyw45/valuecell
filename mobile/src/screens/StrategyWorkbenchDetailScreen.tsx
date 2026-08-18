import { useQuery } from "@tanstack/react-query";
import { ScrollView, StyleSheet, Text, View } from "react-native";
import { useRoute, type RouteProp } from "@react-navigation/native";
import { api } from "../api";
import { SectionCard, StatePanel, StrategyEvaluationPanel } from "../components";
import type { WorkbenchStackParamList } from "../navigation/types";
import { useSession } from "../session";
import { palette, spacing } from "../theme";
import { conditionStateSummary, strategyActionLabel } from "./strategy-presentation";
import { formatTimestamp } from "./workbench";

type Route = RouteProp<WorkbenchStackParamList, "StrategyWorkbenchDetail">;
const RISK_LABELS: Record<string, string> = { normal: "正常", warn: "预警", only_reduce: "仅允许减仓", blocked: "已阻断", halted: "已暂停" };
const MONITOR_LABELS: Record<string, string> = { candidate: "待准入", admitted: "已准入", held: "持仓保留", removed: "已移除" };
const displayRiskState = (state: string) => RISK_LABELS[state] ?? state;
const displayMonitorState = (state: string) => MONITOR_LABELS[state] ?? state;
export default function StrategyWorkbenchDetailScreen() {
  const route = useRoute<Route>();
  const { session } = useSession();
  const { strategyId, section } = route.params;
  const strategy = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId], queryFn: () => api.strategy(strategyId), enabled: Boolean(session && strategyId) });
  const evaluations = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "evaluations", 100], queryFn: () => api.strategyEvaluations(strategyId, 100), enabled: Boolean(strategyId) });
  const monitor = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "monitor-state"], queryFn: () => api.strategyMonitorState(strategyId), enabled: Boolean(strategyId) });
  const risk = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "risk-state"], queryFn: () => api.strategyRiskState(strategyId), enabled: Boolean(strategyId) });
  if (strategy.isLoading) return <StatePanel description="正在读取策略详情。" title="策略详情" />;
  if (strategy.isError || !strategy.data) return <StatePanel actionLabel="重试" description={(strategy.error as Error)?.message ?? "策略详情暂不可用。"} onAction={() => void strategy.refetch()} title="策略详情暂不可用" tone="error" />;
  const latest = evaluations.data?.[0];
  const title = section === "execution" ? "执行概览" : section === "decision" ? "策略决策" : "监控池与风险";
  return <ScrollView contentContainerStyle={styles.content} style={styles.page}>
    <SectionCard description={`${strategy.data.name} · 服务端数据源`} title={title}>
      {section === "execution" ? <View style={styles.rows}><Text style={styles.row}>执行环境：{strategy.data.config.execution.environment}</Text><Text style={styles.row}>策略状态：{strategy.data.status}</Text><Text style={styles.row}>观察币种：{strategy.data.config.symbols.length} 个</Text><Text style={styles.row}>评估周期：{strategy.data.config.interval}</Text><Text style={styles.row}>执行代际：{strategy.data.execution_generation ?? "—"}</Text></View> : null}
      {section === "decision" ? latest ? <><Text style={styles.row}>最近评估：{formatTimestamp(latest.evaluated_at)}</Text><Text style={styles.row}>决策：{strategyActionLabel(latest.action)} · {conditionStateSummary(latest.conditions)}</Text><StrategyEvaluationPanel evaluation={latest} /></> : <Text style={styles.muted}>服务端尚无评估事实。</Text> : null}
      {section === "risk" ? <View style={styles.rows}><Text style={styles.row}>风险状态：{risk.data ? displayRiskState(risk.data.state) : "同步中"}</Text><Text style={styles.muted}>{risk.data?.reason_detail ?? "暂无持久化风险原因"}</Text><Text style={styles.row}>监控池总数：{monitor.data?.length ?? 0}</Text>{monitor.data?.map((item) => <View key={item.symbol} style={styles.monitorRow}><Text style={styles.row}>{item.symbol} · {displayMonitorState(item.state)}</Text><Text style={styles.muted}>{item.reason_detail ?? "暂无原因"}</Text></View>)}</View> : null}
    </SectionCard>
  </ScrollView>;
}

const styles = StyleSheet.create({ page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, rows: { gap: spacing.sm }, row: { color: palette.text, fontSize: 14, lineHeight: 22 }, muted: { color: palette.textMuted, fontSize: 13, lineHeight: 19 }, monitorRow: { borderTopColor: palette.border, borderTopWidth: 1, gap: spacing.xxs, paddingTop: spacing.sm } });
