import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { useNavigation } from "@react-navigation/native";
import { Save } from "lucide-react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { ScreenHeader, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";

export default function SandboxConnectionEditorScreen() {
  const navigation = useNavigation<any>();
  const { session } = useSession();
  const queryClient = useQueryClient();
  const [provider, setProvider] = useState<"binance" | "okx">("okx");
  const [label, setLabel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [passphrase, setPassphrase] = useState("");
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const create = useMutation({ mutationFn: () => api.createSandboxConnection({ provider, label: label.trim(), api_key: apiKey, api_secret: apiSecret, ...(provider === "okx" ? { passphrase } : {}) }), onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "sandbox-connections"] }) });
  const clearCredentials = () => { setApiKey(""); setApiSecret(""); setPassphrase(""); };
  const submit = async () => {
    if (!label.trim() || !apiKey || !apiSecret || (provider === "okx" && !passphrase)) { Alert.alert("缺少凭据", "请填写标签和服务端验证所需的全部凭据。OKX Demo 还需要 Passphrase。"); return; }
    try { await create.mutateAsync(); Alert.alert("连接已保存", "服务器保险库已保存经过验证的模拟盘凭据。"); navigation.goBack(); } catch (error) { Alert.alert("连接失败", error instanceof Error ? error.message : "服务端未接受该凭据。"); } finally { clearCredentials(); }
  };
  const allowed = canMutate(access.data, "connection.manage");
  if (access.isLoading) return <StatePanel description="正在验证工作区访问状态。" title="添加模拟盘连接" />;
  return <ScrollView contentContainerStyle={styles.content} style={styles.page}>
    <ScreenHeader subtitle="凭据仅通过 HTTPS 发送给服务器保险库，提交后立即从输入框清除。" title="添加模拟盘连接" />
    {!allowed ? <StatePanel description={access.data?.status === "active" ? "当前角色不能管理交易所连接。" : "服务未激活，连接写入已禁用。"} title="无法添加连接" /> : null}
    <View style={styles.options}>{(["okx", "binance"] as const).map((item) => <Pressable accessibilityRole="button" key={item} onPress={() => setProvider(item)} style={[styles.option, provider === item && styles.optionActive]}><Text style={[styles.optionText, provider === item && styles.optionTextActive]}>{item === "okx" ? "OKX Demo" : "Binance Testnet"}</Text></Pressable>)}</View>
    <Field label="连接标签" onChangeText={setLabel} value={label} /><Field label="API Key" onChangeText={setApiKey} sensitive value={apiKey} /><Field label="API Secret" onChangeText={setApiSecret} sensitive value={apiSecret} />{provider === "okx" ? <Field label="OKX Passphrase" onChangeText={setPassphrase} sensitive value={passphrase} /> : null}
    <View style={styles.notice}><Text style={styles.noticeTitle}>模拟盘限制</Text><Text style={styles.noticeText}>连接只用于服务器验证、只读余额/订单和明确的模拟盘订单。不会回显 API Secret 或 Passphrase。</Text></View>
    <Pressable accessibilityRole="button" disabled={!allowed || create.isPending} onPress={() => void submit()} style={[styles.save, (!allowed || create.isPending) && styles.disabled]}><Save color={palette.canvas} size={19} /><Text style={styles.saveText}>{create.isPending ? "正在验证…" : "验证并保存连接"}</Text></Pressable>
  </ScrollView>;
}

function Field({ label, onChangeText, sensitive = false, value }: { label: string; value: string; sensitive?: boolean; onChangeText: (value: string) => void }) { return <View style={styles.field}><Text style={styles.label}>{label}</Text><TextInput accessibilityLabel={label} autoCapitalize="none" autoCorrect={false} onChangeText={onChangeText} secureTextEntry={sensitive} style={styles.input} value={value} /></View>; }

const styles = StyleSheet.create({ page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, options: { flexDirection: "row", gap: spacing.sm }, option: { borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flex: 1, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.sm }, optionActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, optionText: { color: palette.textMuted, fontSize: 13, fontWeight: "800", textAlign: "center" }, optionTextActive: { color: palette.primary }, field: { gap: spacing.xs }, label: { color: palette.textMuted, fontSize: 12, fontWeight: "700" }, input: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, minHeight: 48, paddingHorizontal: spacing.sm }, notice: { backgroundColor: palette.warningSoft, borderRadius: radius.md, gap: spacing.xs, padding: spacing.md }, noticeTitle: { color: palette.warning, fontSize: 14, fontWeight: "800" }, noticeText: { color: palette.text, fontSize: 12, lineHeight: 19 }, save: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.sm, justifyContent: "center", minHeight: 52 }, saveText: { color: palette.canvas, fontSize: 16, fontWeight: "800" }, disabled: { opacity: 0.45 }, });
