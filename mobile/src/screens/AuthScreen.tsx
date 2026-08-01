import { useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { BarChart3, LockKeyhole, Mail, ShieldCheck } from "lucide-react-native";
import { api } from "../api";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";

export default function AuthScreen() {
  const { signIn, signOut } = useSession();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    if (!email.trim() || !password) {
      setError("请输入企业邮箱和密码。");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await signIn(email, password);
      await api.access();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "登录失败，请稍后重试。");
      await signOut();
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.select({ ios: "padding", default: undefined })}
      style={styles.page}
    >
      <View style={styles.orbPrimary} />
      <View style={styles.orbSecondary} />
      <View style={styles.content}>
        <View style={styles.brandRow}>
          <View style={styles.brandIcon}><BarChart3 color={palette.canvas} size={21} /></View>
          <Text style={styles.brand}>VALUE CELL</Text>
        </View>
        <View style={styles.hero}>
          <Text style={styles.eyebrow}>SECURE STRATEGY OPERATIONS</Text>
          <Text style={styles.title}>随时掌握策略、资金与市场</Text>
          <Text style={styles.subtitle}>企业级量化工作台。登录后进入已授权的租户工作区。</Text>
        </View>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>登录工作区</Text>
          <Text style={styles.cardCopy}>使用 Value Cell SaaS 账户继续</Text>
          <View style={styles.field}>
            <Mail color={palette.textMuted} size={18} />
            <TextInput
              autoCapitalize="none"
              autoComplete="email"
              keyboardType="email-address"
              onChangeText={setEmail}
              placeholder="企业邮箱"
              placeholderTextColor={palette.textMuted}
              style={styles.input}
              value={email}
            />
          </View>
          <View style={styles.field}>
            <LockKeyhole color={palette.textMuted} size={18} />
            <TextInput
              autoComplete="password"
              onChangeText={setPassword}
              placeholder="密码"
              placeholderTextColor={palette.textMuted}
              secureTextEntry
              style={styles.input}
              value={password}
            />
          </View>
          {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
          <Pressable
            accessibilityRole="button"
            disabled={submitting}
            onPress={() => void submit()}
            style={({ pressed }) => [styles.submit, (pressed || submitting) && styles.submitPressed]}
          >
            {submitting ? <ActivityIndicator color={palette.canvas} /> : <Text style={styles.submitText}>安全登录</Text>}
          </Pressable>
          <View style={styles.securityRow}>
            <ShieldCheck color={palette.positive} size={16} />
            <Text style={styles.securityCopy}>会话令牌仅加密保存在本设备。</Text>
          </View>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: palette.canvas, justifyContent: "center" },
  content: { gap: spacing.xl, paddingHorizontal: spacing.lg, zIndex: 1 },
  orbPrimary: { position: "absolute", top: -140, right: -100, width: 310, height: 310, borderRadius: 155, backgroundColor: "#123F5B" },
  orbSecondary: { position: "absolute", bottom: -160, left: -100, width: 260, height: 260, borderRadius: 130, backgroundColor: "#11352F" },
  brandRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  brandIcon: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, height: 38, justifyContent: "center", width: 38 },
  brand: { color: palette.text, fontSize: 14, fontWeight: "800", letterSpacing: 1.8 },
  hero: { gap: spacing.sm },
  eyebrow: { color: palette.primary, fontSize: 11, fontWeight: "800", letterSpacing: 1.2 },
  title: { color: palette.text, fontSize: 31, fontWeight: "800", letterSpacing: -1.1, lineHeight: 40 },
  subtitle: { color: palette.textMuted, fontSize: 15, lineHeight: 23 },
  card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.lg },
  cardTitle: { color: palette.text, fontSize: 20, fontWeight: "700" },
  cardCopy: { color: palette.textMuted, fontSize: 13, marginTop: -spacing.sm },
  field: { alignItems: "center", backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.sm, height: 54, paddingHorizontal: spacing.sm },
  input: { color: palette.text, flex: 1, fontSize: 16, height: "100%" },
  submit: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, justifyContent: "center", minHeight: 52 },
  submitPressed: { opacity: 0.72 },
  submitText: { color: palette.canvas, fontSize: 16, fontWeight: "800" },
  error: { color: palette.negative, fontSize: 13, lineHeight: 19 },
  securityRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
  securityCopy: { color: palette.textMuted, fontSize: 12 },
});
