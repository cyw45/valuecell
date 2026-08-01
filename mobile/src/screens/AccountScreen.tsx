import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { Building2, ChevronRight, LogOut, ShieldCheck, UserRound } from "lucide-react-native";
import { api } from "../api";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";

export default function AccountScreen() {
  const { session, signOut, switchWorkspace } = useSession();
  const queryClient = useQueryClient();
  const access = useQuery({ queryKey: ["mobile", "access"], queryFn: api.access });
  const workspaces = useQuery({ queryKey: ["mobile", "workspaces"], queryFn: api.workspaces });

  async function selectWorkspace(tenantId: string) {
    try {
      await switchWorkspace(tenantId);
      await queryClient.invalidateQueries();
    } catch (reason) {
      Alert.alert("切换失败", reason instanceof Error ? reason.message : "无法切换工作区。");
    }
  }

  function confirmSignOut() {
    Alert.alert("退出登录", "将清除本设备保存的加密会话。", [
      { text: "取消", style: "cancel" },
      { text: "退出", style: "destructive", onPress: () => void signOut() },
    ]);
  }

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl onRefresh={() => void Promise.all([access.refetch(), workspaces.refetch()])} refreshing={access.isRefetching || workspaces.isRefetching} tintColor={palette.primary} />}
      style={styles.page}
    >
      <Text style={styles.eyebrow}>ACCOUNT & WORKSPACE</Text>
      <Text style={styles.title}>账户与工作区</Text>
      <View style={styles.profileCard}><View style={styles.avatar}><UserRound color={palette.canvas} size={24} /></View><View style={styles.profileCopy}><Text numberOfLines={1} style={styles.email}>{session?.email}</Text><Text style={styles.role}>{access.data?.is_platform_admin ? "平台管理员" : `${access.data?.role ?? "成员"} · ${access.data?.tenant_type === "enterprise" ? "企业租户" : "个人账户"}`}</Text></View></View>
      <View style={styles.statusCard}><ShieldCheck color={access.data?.status === "active" ? palette.positive : palette.warning} size={20} /><View style={styles.statusCopy}><Text style={styles.statusTitle}>{access.data?.status === "active" ? "服务已开通" : "服务待开通"}</Text><Text style={styles.statusText}>{access.data?.commercial_model === "revenue_share" ? "企业利润分成合同" : access.data?.commercial_model === "subscription" ? "订阅服务" : "请联系平台管理员开通工作区服务"}</Text></View></View>
      <Text style={styles.sectionTitle}>工作区切换</Text>
      <View style={styles.list}>{workspaces.data?.map((workspace) => <Pressable accessibilityRole="button" key={workspace.tenant_id} onPress={() => void selectWorkspace(workspace.tenant_id)} style={({ pressed }) => [styles.workspace, workspace.selected && styles.workspaceSelected, pressed && styles.pressed]}><View style={styles.workspaceIcon}><Building2 color={workspace.selected ? palette.primary : palette.textMuted} size={18} /></View><View style={styles.workspaceCopy}><Text style={styles.workspaceName}>{workspace.organization_name ?? workspace.name}</Text><Text style={styles.workspaceMeta}>{workspace.tenant_type === "enterprise" ? "企业" : "个人"} · {workspace.role}</Text></View><View style={styles.workspaceEnd}>{workspace.selected ? <Text style={styles.selectedText}>当前</Text> : <ChevronRight color={palette.textMuted} size={18} />}</View></Pressable>) ?? <Text style={styles.loading}>正在读取可切换工作区…</Text>}</View>
      <View style={styles.security}><Text style={styles.securityTitle}>设备安全</Text><Text style={styles.securityText}>访问令牌通过系统安全存储加密保存。不会将交易所 API Secret、Passphrase 或私钥写入应用日志。</Text></View>
      <Pressable accessibilityRole="button" onPress={confirmSignOut} style={({ pressed }) => [styles.signOut, pressed && styles.pressed]}><LogOut color={palette.negative} size={18} /><Text style={styles.signOutText}>安全退出登录</Text></Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, eyebrow: { color: palette.primary, fontSize: 10, fontWeight: "800", letterSpacing: 1.2, marginTop: spacing.sm }, title: { color: palette.text, fontSize: 27, fontWeight: "800", letterSpacing: -0.8 }, profileCard: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.lg, borderWidth: 1, flexDirection: "row", gap: spacing.md, padding: spacing.md }, avatar: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.pill, height: 48, justifyContent: "center", width: 48 }, profileCopy: { flex: 1 }, email: { color: palette.text, fontSize: 16, fontWeight: "800" }, role: { color: palette.textMuted, fontSize: 12, marginTop: 4 }, statusCard: { alignItems: "center", backgroundColor: palette.positiveSoft, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, padding: spacing.md }, statusCopy: { flex: 1 }, statusTitle: { color: palette.text, fontSize: 14, fontWeight: "800" }, statusText: { color: palette.textMuted, fontSize: 12, lineHeight: 18, marginTop: 2 }, sectionTitle: { color: palette.text, fontSize: 14, fontWeight: "800", marginTop: spacing.sm }, list: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, overflow: "hidden" }, workspace: { alignItems: "center", borderBottomColor: palette.border, borderBottomWidth: StyleSheet.hairlineWidth, flexDirection: "row", gap: spacing.sm, padding: spacing.md }, workspaceSelected: { backgroundColor: palette.primarySoft }, workspaceIcon: { alignItems: "center", height: 30, justifyContent: "center", width: 30 }, workspaceCopy: { flex: 1 }, workspaceName: { color: palette.text, fontSize: 14, fontWeight: "700" }, workspaceMeta: { color: palette.textMuted, fontSize: 12, marginTop: 3 }, workspaceEnd: { minWidth: 34 }, selectedText: { color: palette.primary, fontSize: 12, fontWeight: "800" }, security: { backgroundColor: palette.surfaceMuted, borderRadius: radius.md, gap: 4, padding: spacing.md }, securityTitle: { color: palette.text, fontSize: 13, fontWeight: "800" }, securityText: { color: palette.textMuted, fontSize: 12, lineHeight: 18 }, signOut: { alignItems: "center", borderColor: palette.negative, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48 }, signOutText: { color: palette.negative, fontSize: 14, fontWeight: "800" }, loading: { color: palette.textMuted, padding: spacing.md }, pressed: { opacity: 0.7 },
});
