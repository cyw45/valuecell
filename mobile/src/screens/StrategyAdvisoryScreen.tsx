import { useQuery } from "@tanstack/react-query";
import { RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { useRoute } from "@react-navigation/native";
import { Bot } from "lucide-react-native";
import { api } from "../api";
import { ScreenHeader, SectionCard, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";

type RouteParams = { params: { strategyId: string } };

export default function StrategyAdvisoryScreen() {
  const route = useRoute() as RouteParams;
  const { session } = useSession();
  const strategyId = route.params.strategyId;
  const advisory = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "advisory"], queryFn: () => api.strategyAdvisory(strategyId), enabled: Boolean(strategyId) });
  const evaluations = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "evaluations", 10], queryFn: () => api.strategyEvaluations(strategyId, 10), enabled: Boolean(strategyId) });
  const error = advisory.error ?? evaluations.error;
  if (advisory.isLoading || evaluations.isLoading) return <StatePanel description="正在读取配置参考与最近评估证据。" title="策略参考" />;
  if (error) return <StatePanel actionLabel="重试" description={(error as Error).message} onAction={() => void Promise.all([advisory.refetch(), evaluations.refetch()])} title="策略参考暂不可用" tone="error" />;
  const details = advisory.data as { content?: string; provider?: string; model_id?: string } | undefined;
  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={() => void Promise.all([advisory.refetch(), evaluations.refetch()])} refreshing={advisory.isRefetching || evaluations.isRefetching} tintColor={palette.primary} />} style={styles.page}>
    <ScreenHeader subtitle="仅供策略参考，不会自动执行或修改策略" title="策略参考" />
    <View style={styles.notice}><Bot color={palette.warning} size={20} /><Text style={styles.noticeText}>仅供策略参考，不会自动执行或修改策略</Text></View>
    <SectionCard title="配置建议"><Text style={styles.contentText}>{details?.content ?? "当前没有可用的策略建议。"}</Text><Text style={styles.provider}>{details?.provider ?? "—"} · {details?.model_id ?? "—"}</Text></SectionCard>
    <SectionCard title="近期评估证据">{(evaluations.data ?? []).length ? (evaluations.data ?? []).map((entry, index) => <View key={String((entry as unknown as Record<string, unknown>).evaluation_id ?? index)} style={styles.evidence}><Text style={styles.action}>{String((entry as unknown as Record<string, unknown>).action ?? "—")}</Text><Text style={styles.reason}>{String((entry as unknown as Record<string, unknown>).reason ?? "—")}</Text></View>) : <Text style={styles.contentText}>尚无服务端评估证据。</Text>}</SectionCard>
  </ScrollView>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, notice: { alignItems: "center", backgroundColor: palette.warningSoft, borderRadius: radius.md, flexDirection: "row", gap: spacing.sm, padding: spacing.md }, noticeText: { color: palette.warning, flex: 1, fontSize: 14, fontWeight: "800", lineHeight: 21 }, contentText: { color: palette.text, fontSize: 14, lineHeight: 23 }, provider: { color: palette.textMuted, fontSize: 12, marginTop: spacing.md }, evidence: { borderTopColor: palette.border, borderTopWidth: 1, gap: spacing.xs, paddingVertical: spacing.sm }, action: { color: palette.primary, fontSize: 12, fontWeight: "800" }, reason: { color: palette.text, fontSize: 13, lineHeight: 20 },
});
