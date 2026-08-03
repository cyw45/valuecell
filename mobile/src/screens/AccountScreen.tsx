import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { useNavigation } from "@react-navigation/native";
import { Building2, ChevronRight, CreditCard, FileClock, Landmark, LogOut, Settings2, ShieldCheck, UsersRound } from "lucide-react-native";
import { api } from "../api";
import { canMutate, isPlatformAdmin } from "../access";
import { ScreenHeader, SectionCard, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";

function AccountLink({ disabled, icon: Icon, label, onPress, subtitle }: { disabled?: boolean; icon: typeof Settings2; label: string; onPress: () => void; subtitle?: string }) {
  return <Pressable accessibilityRole="button" disabled={disabled} onPress={onPress} style={[styles.link, disabled && styles.disabled]}><Icon color={disabled ? palette.textMuted : palette.primary} size={20} /><View style={styles.linkCopy}><Text style={styles.linkLabel}>{label}</Text>{subtitle ? <Text style={styles.linkSubtitle}>{subtitle}</Text> : null}</View><ChevronRight color={palette.textMuted} size={18} /></Pressable>;
}

export default function AccountScreen() {
  const navigation = useNavigation<any>();
  const { session, signOut, switchWorkspace } = useSession();
  const queryClient = useQueryClient();
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const workspaces = useQuery({ queryKey: ["mobile", session?.tenantId, "workspaces"], queryFn: api.workspaces, enabled: Boolean(session) });
  const active = access.data?.status === "active";
  const mayManageMembers = canMutate(access.data, "member.manage");
  const mayManageBilling = canMutate(access.data, "billing.manage");
  const selectWorkspace = async (tenantId: string) => {
    try { await switchWorkspace(tenantId); await queryClient.invalidateQueries(); } catch (error) { Alert.alert("切换失败", error instanceof Error ? error.message : "无法切换工作区。"); }
  };
  const confirmSignOut = () => Alert.alert("安全退出登录", "将清除本设备保存的加密会话。交易所凭据从未保存在设备中。", [{ text: "取消", style: "cancel" }, { text: "退出", style: "destructive", onPress: () => void signOut() }]);
  const refresh = () => void Promise.all([access.refetch(), workspaces.refetch()]);
  if (access.isError) return <StatePanel actionLabel="重试" description={(access.error as Error).message} onAction={refresh} title="账户状态暂不可用" tone="error" />;
  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={refresh} refreshing={access.isRefetching || workspaces.isRefetching} tintColor={palette.primary} />} style={styles.page}>
    <ScreenHeader subtitle={access.data?.organization_name ?? "量化策略工作区"} title="我的" />
    <View style={styles.profile}><View style={styles.avatar}><Text style={styles.avatarText}>{session?.email.slice(0, 1).toUpperCase() ?? "V"}</Text></View><View style={styles.profileCopy}><Text numberOfLines={1} style={styles.email}>{session?.email}</Text><Text style={styles.role}>{access.data?.is_platform_admin ? "平台管理员" : `${access.data?.role ?? "成员"} · ${access.data?.tenant_type === "enterprise" ? "企业工作区" : "个人工作区"}`}</Text></View></View>
    <View style={[styles.status, { backgroundColor: active ? palette.positiveSoft : palette.warningSoft }]}><ShieldCheck color={active ? palette.positive : palette.warning} size={20} /><View><Text style={[styles.statusTitle, { color: active ? palette.positive : palette.warning }]}>{active ? "服务已开通" : "服务待开通"}</Text><Text style={styles.statusCopy}>{access.data?.commercial_model === "revenue_share" ? "企业分成合约" : access.data?.commercial_model === "subscription" ? "订阅服务" : "请联系平台管理员开通服务"}{access.data?.expires_at ? ` · 至 ${new Date(access.data.expires_at).toLocaleDateString()}` : ""}</Text></View></View>
    <SectionCard title="工作区"><View style={styles.workspaceList}>{(workspaces.data ?? []).map((workspace) => <Pressable accessibilityRole="button" key={workspace.tenant_id} onPress={() => void selectWorkspace(workspace.tenant_id)} style={[styles.workspace, workspace.selected && styles.workspaceSelected]}><Building2 color={workspace.selected ? palette.primary : palette.textMuted} size={19} /><View style={styles.linkCopy}><Text style={styles.linkLabel}>{workspace.organization_name ?? workspace.name}</Text><Text style={styles.linkSubtitle}>{workspace.tenant_type === "enterprise" ? "企业" : "个人"} · {workspace.role}</Text></View>{workspace.selected ? <Text style={styles.selected}>当前</Text> : <ChevronRight color={palette.textMuted} size={18} />}</Pressable>)}</View></SectionCard>
    <SectionCard title="偏好与执行"><AccountLink icon={Settings2} label="偏好设置" onPress={() => navigation.navigate("Preferences")} subtitle="语言、主题、涨跌色与行情刷新" /><AccountLink disabled={!active} icon={Landmark} label="OKX Demo 与模拟盘" onPress={() => navigation.navigate("SandboxConnections")} subtitle="凭据仅发送至服务器保险库" /><AccountLink disabled={!active} icon={ShieldCheck} label="实盘执行" onPress={() => navigation.navigate("LiveExecution")} subtitle="需服务端风控与授权" /></SectionCard>
    <SectionCard title="当前租户"><AccountLink disabled={!active} icon={UsersRound} label="成员管理" onPress={() => navigation.navigate("WorkspaceMembers")} subtitle={access.data?.tenant_type === "personal" ? "个人租户不提供成员管理" : mayManageMembers ? "管理成员角色" : "只读成员状态"} /><AccountLink disabled={!mayManageBilling} icon={CreditCard} label="账单与合约" onPress={() => navigation.navigate("WorkspaceBilling")} subtitle={mayManageBilling ? "订阅、合约与结算" : "角色可能限制账单查看"} /><AccountLink disabled={!active} icon={FileClock} label="审计记录" onPress={() => navigation.navigate("WorkspaceAudit")} subtitle="当前工作区安全与配置事件" /></SectionCard>
    {isPlatformAdmin(access.data) ? <SectionCard title="平台管理"><AccountLink icon={Building2} label="平台租户与合约" onPress={() => navigation.navigate("PlatformAdmin")} subtitle="仅平台管理员可见" /></SectionCard> : null}
    <View style={styles.security}><Text style={styles.securityTitle}>设备安全边界</Text><Text style={styles.securityCopy}>访问令牌使用系统安全存储。API Key、Secret、Passphrase、私钥和实时凭据不会写入会话、偏好、查询缓存或日志。</Text></View>
    <Pressable accessibilityRole="button" onPress={confirmSignOut} style={styles.signOut}><LogOut color={palette.negative} size={19} /><Text style={styles.signOutText}>安全退出登录</Text></Pressable>
  </ScrollView>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, profile: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.lg, borderWidth: 1, flexDirection: "row", gap: spacing.md, padding: spacing.md }, avatar: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.pill, height: 48, justifyContent: "center", width: 48 }, avatarText: { color: palette.canvas, fontSize: 20, fontWeight: "800" }, profileCopy: { flex: 1 }, email: { color: palette.text, fontSize: 16, fontWeight: "800" }, role: { color: palette.textMuted, fontSize: 13, marginTop: 3 }, status: { alignItems: "center", borderRadius: radius.md, flexDirection: "row", gap: spacing.sm, padding: spacing.md }, statusTitle: { fontSize: 14, fontWeight: "800" }, statusCopy: { color: palette.textMuted, fontSize: 12, marginTop: 2 }, workspaceList: { gap: spacing.xs }, workspace: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 52, paddingHorizontal: spacing.sm }, workspaceSelected: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, link: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 56, paddingVertical: spacing.xs }, linkCopy: { flex: 1 }, linkLabel: { color: palette.text, fontSize: 14, fontWeight: "800" }, linkSubtitle: { color: palette.textMuted, fontSize: 12, marginTop: 2 }, selected: { color: palette.primary, fontSize: 12, fontWeight: "800" }, disabled: { opacity: 0.45 }, security: { backgroundColor: palette.surfaceMuted, borderRadius: radius.md, gap: spacing.xs, padding: spacing.md }, securityTitle: { color: palette.text, fontSize: 14, fontWeight: "800" }, securityCopy: { color: palette.textMuted, fontSize: 12, lineHeight: 19 }, signOut: { alignItems: "center", borderColor: palette.negative, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.sm, justifyContent: "center", minHeight: 50 }, signOutText: { color: palette.negative, fontSize: 15, fontWeight: "800" },
});
