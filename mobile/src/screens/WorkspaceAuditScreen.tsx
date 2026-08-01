import { useQuery } from "@tanstack/react-query";
import { RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { api } from "../api";
import { ScreenHeader, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";

export default function WorkspaceAuditScreen() {
  const { session } = useSession();
  const audit = useQuery({ queryKey: ["mobile", session?.tenantId, "tenant-audit", 100], queryFn: () => api.tenantAudit(100), enabled: Boolean(session) });
  if (audit.isLoading) return <StatePanel description="正在读取当前租户最近 100 条审计事件。" title="审计记录" />;
  if (audit.isError) return <StatePanel actionLabel="重试" description={(audit.error as Error).message} onAction={() => void audit.refetch()} title="审计记录暂不可用" tone="error" />;
  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={() => void audit.refetch()} refreshing={audit.isRefetching} tintColor={palette.primary} />} style={styles.page}><ScreenHeader subtitle="当前工作区的安全、成员、策略及配置操作。" title="审计记录" />{audit.data?.length ? audit.data.map((event) => <View key={event.id} style={styles.card}><View style={styles.top}><Text style={styles.action}>{event.action}</Text><Text style={styles.outcome}>{event.outcome}</Text></View><Text style={styles.target}>{event.target_type} · {event.target_id}</Text><Text style={styles.meta}>{event.created_at} · {event.actor_user_id ?? "系统"}</Text></View>) : <StatePanel description="服务端尚未记录当前租户的审计事件。" title="暂无审计记录" />}</ScrollView>;
}
const styles = StyleSheet.create({ page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.sm, padding: spacing.md, paddingBottom: spacing.xl }, card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.xs, padding: spacing.md }, top: { flexDirection: "row", justifyContent: "space-between" }, action: { color: palette.primary, fontSize: 13, fontWeight: "800" }, outcome: { color: palette.textMuted, fontSize: 12 }, target: { color: palette.text, fontSize: 13 }, meta: { color: palette.textMuted, fontSize: 11 }, });
