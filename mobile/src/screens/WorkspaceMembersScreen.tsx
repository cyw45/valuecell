import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { ScreenHeader, SectionCard, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import type { TenantRole } from "../types";

const ROLES: TenantRole[] = ["owner", "admin", "strategist", "trader", "viewer", "billing_manager"];

export default function WorkspaceMembersScreen() {
  const { session } = useSession(); const queryClient = useQueryClient(); const [email, setEmail] = useState(""); const [role, setRole] = useState<TenantRole>("viewer");
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const members = useQuery({ queryKey: ["mobile", session?.tenantId, "workspace-members"], queryFn: api.workspaceMembers, enabled: Boolean(session) });
  const save = useMutation({ mutationFn: () => api.saveWorkspaceMember({ email: email.trim(), role }), onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "workspace-members"] }) });
  const mayManage = canMutate(access.data, "member.manage");
  const submit = async () => { if (!email.trim()) return; try { await save.mutateAsync(); setEmail(""); } catch (error) { Alert.alert("保存成员失败", error instanceof Error ? error.message : "请检查邮箱和成员状态。"); } };
  if (access.data?.tenant_type === "personal") return <StatePanel description="个人租户不支持邀请或编辑成员。升级为企业工作区后可使用成员管理。" title="成员管理不可用" />;
  if (members.isLoading) return <StatePanel description="正在读取企业工作区成员。" title="成员管理" />;
  if (members.isError) return <StatePanel actionLabel="重试" description={(members.error as Error).message} onAction={() => void members.refetch()} title="成员暂不可用" tone="error" />;
  return <ScrollView contentContainerStyle={styles.content} style={styles.page}><ScreenHeader subtitle={mayManage ? "Owner 或 Admin 可保存成员角色。" : "当前角色仅可查看成员列表。"} title="成员管理" />{mayManage ? <SectionCard title="添加或更新成员"><TextInput accessibilityLabel="成员邮箱" autoCapitalize="none" keyboardType="email-address" onChangeText={setEmail} placeholder="member@example.com" placeholderTextColor={palette.textMuted} style={styles.input} value={email} /><View style={styles.roles}>{ROLES.map((item) => <Pressable accessibilityRole="button" key={item} onPress={() => setRole(item)} style={[styles.role, role === item && styles.roleActive]}><Text style={[styles.roleText, role === item && styles.roleTextActive]}>{item}</Text></Pressable>)}</View><Pressable accessibilityRole="button" disabled={save.isPending} onPress={() => void submit()} style={[styles.action, save.isPending && styles.disabled]}><Text style={styles.actionText}>保存成员角色</Text></Pressable></SectionCard> : null}<SectionCard title="当前成员">{(members.data ?? []).map((member) => <View key={member.user_id} style={styles.member}><View><Text style={styles.email}>{member.email}</Text><Text style={styles.meta}>{member.created_at}</Text></View><Text style={styles.memberRole}>{member.role}</Text></View>)}</SectionCard></ScrollView>;
}
const styles = StyleSheet.create({ page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, input: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, minHeight: 46, paddingHorizontal: spacing.sm }, roles: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs, marginTop: spacing.sm }, role: { borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, minHeight: 40, justifyContent: "center", paddingHorizontal: spacing.sm }, roleActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, roleText: { color: palette.textMuted, fontSize: 12 }, roleTextActive: { color: palette.primary }, action: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, justifyContent: "center", marginTop: spacing.md, minHeight: 46 }, actionText: { color: palette.canvas, fontSize: 14, fontWeight: "800" }, member: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", justifyContent: "space-between", paddingVertical: spacing.sm }, email: { color: palette.text, fontSize: 14, fontWeight: "800" }, meta: { color: palette.textMuted, fontSize: 11, marginTop: 3 }, memberRole: { color: palette.primary, fontSize: 12, fontWeight: "800" }, disabled: { opacity: 0.45 }, });
