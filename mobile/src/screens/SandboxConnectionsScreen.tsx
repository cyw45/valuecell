import { useQuery } from "@tanstack/react-query";
import { Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { useNavigation } from "@react-navigation/native";
import { ChevronRight, Plus, ShieldCheck } from "lucide-react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { ScreenHeader, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";

export default function SandboxConnectionsScreen() {
  const navigation = useNavigation<any>();
  const { session } = useSession();
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const connections = useQuery({ queryKey: ["mobile", session?.tenantId, "sandbox-connections"], queryFn: api.demoConnections, enabled: Boolean(session) });
  const canCreate = canMutate(access.data, "connection.manage");
  if (connections.isLoading) return <StatePanel description="正在读取当前租户的服务器保险库连接。" title="OKX Demo 与模拟盘" />;
  if (connections.isError) return <StatePanel actionLabel="重试" description={(connections.error as Error).message} onAction={() => void connections.refetch()} title="连接暂不可用" tone="error" />;
  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={() => void connections.refetch()} refreshing={connections.isRefetching} tintColor={palette.primary} />} style={styles.page}>
    <ScreenHeader actionLabel={canCreate ? "添加连接" : undefined} onAction={canCreate ? () => navigation.navigate("SandboxConnectionEditor") : undefined} subtitle="仅支持 Binance Testnet 和 OKX Demo 现货；凭据不保存在本机。" title="OKX Demo 与模拟盘" />
    <View style={styles.warning}><ShieldCheck color={palette.warning} size={20} /><Text style={styles.warningText}>仅模拟盘：提交订单不会绕过服务器权限、限额或幂等保护。</Text></View>
    {!connections.data?.length ? <StatePanel actionLabel={canCreate ? "添加模拟盘连接" : undefined} description="创建后服务器只返回安全的连接元数据。" onAction={canCreate ? () => navigation.navigate("SandboxConnectionEditor") : undefined} title="尚无模拟盘连接" /> : connections.data.map((connection) => <Pressable accessibilityRole="button" key={connection.id} onPress={() => navigation.navigate("SandboxConnectionDetail", { connectionId: connection.id })} style={styles.card}><View style={styles.icon}><Text style={styles.iconText}>{connection.provider.toUpperCase()}</Text></View><View style={styles.copy}><Text style={styles.label}>{connection.label}</Text><Text style={styles.meta}>{connection.metadata.sandbox ? "模拟盘" : "—"} · {connection.metadata.market_type ?? "spot"} · {connection.revoked ? "已撤销" : "已验证"}</Text></View><ChevronRight color={palette.textMuted} size={20} /></Pressable>)}
  </ScrollView>;
}

const styles = StyleSheet.create({ page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, warning: { alignItems: "center", backgroundColor: palette.warningSoft, borderRadius: radius.md, flexDirection: "row", gap: spacing.sm, padding: spacing.md }, warningText: { color: palette.warning, flex: 1, fontSize: 13, fontWeight: "700", lineHeight: 20 }, card: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 68, padding: spacing.md }, icon: { alignItems: "center", backgroundColor: palette.primarySoft, borderRadius: radius.sm, height: 38, justifyContent: "center", width: 48 }, iconText: { color: palette.primary, fontSize: 10, fontWeight: "800" }, copy: { flex: 1 }, label: { color: palette.text, fontSize: 15, fontWeight: "800" }, meta: { color: palette.textMuted, fontSize: 12, marginTop: 3 }, });
