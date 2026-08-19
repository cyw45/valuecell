import { useRef, useState } from "react";
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { LockKeyhole } from "lucide-react-native";
import { api, loadRememberedPassword, persistRememberedPassword } from "../api";
import { StatePanel } from "../components";
import { palette, radius, spacing } from "../theme";

export default function ChangePasswordScreen() {
  const scrollRef = useRef<ScrollView>(null);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (newPassword.length < 12) {
      setError("新密码至少需要 12 个字符。");
      return;
    }
    if (newPassword !== confirmation) {
      setError("两次输入的新密码不一致。");
      return;
    }
    setSubmitting(true);
    setError(null);
    setMessage(null);
    try {
      await api.changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      });
      if (await loadRememberedPassword()) await persistRememberedPassword(newPassword);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmation("");
      setMessage("密码已更新。当前设备会话保持有效。其他已登录设备会在其令牌过期后要求重新登录。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "密码修改失败，请稍后重试。");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <KeyboardAvoidingView behavior={Platform.select({ ios: "padding", android: "height" })} style={styles.page}>
      <ScrollView automaticallyAdjustKeyboardInsets contentContainerStyle={styles.content} keyboardDismissMode="interactive" keyboardShouldPersistTaps="handled" ref={scrollRef}>
        <View style={styles.card}>
          <Text style={styles.title}>修改密码</Text>
          <Text style={styles.copy}>请先验证当前密码，再设置至少 12 位的新密码。</Text>
          <PasswordField label="当前密码" onChangeText={setCurrentPassword} onFocus={() => scrollRef.current?.scrollToEnd({ animated: true })} value={currentPassword} />
          <PasswordField label="新密码" onChangeText={setNewPassword} onFocus={() => scrollRef.current?.scrollToEnd({ animated: true })} value={newPassword} />
          <PasswordField label="确认新密码" onChangeText={setConfirmation} onFocus={() => scrollRef.current?.scrollToEnd({ animated: true })} value={confirmation} />
          {error ? <StatePanel error={error} state="error" title="无法修改密码" /> : null}
          {message ? <Text style={styles.success}>{message}</Text> : null}
          <Pressable accessibilityRole="button" disabled={submitting} onPress={() => void submit()} style={({ pressed }) => [styles.submit, (pressed || submitting) && styles.disabled]}><Text style={styles.submitText}>{submitting ? "正在更新…" : "更新密码"}</Text></Pressable>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function PasswordField({ label, onChangeText, onFocus, value }: { label: string; onChangeText: (value: string) => void; onFocus: () => void; value: string }) {
  return <View style={styles.field}><LockKeyhole color={palette.textMuted} size={18} /><TextInput accessibilityLabel={label} autoComplete="new-password" onChangeText={onChangeText} onFocus={onFocus} placeholder={label} placeholderTextColor={palette.textMuted} secureTextEntry style={styles.input} textContentType="newPassword" value={value} /></View>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { flexGrow: 1, justifyContent: "center", padding: spacing.md },
  card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.lg },
  title: { color: palette.text, fontSize: 21, fontWeight: "900" },
  copy: { color: palette.textMuted, fontSize: 13, lineHeight: 20 },
  field: { alignItems: "center", backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 52, paddingHorizontal: spacing.sm },
  input: { color: palette.text, flex: 1, fontSize: 16, minHeight: 50 },
  submit: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md },
  submitText: { color: palette.canvas, fontSize: 16, fontWeight: "800" },
  success: { backgroundColor: palette.positiveSoft, borderRadius: radius.sm, color: palette.positive, fontSize: 13, lineHeight: 19, padding: spacing.sm },
  disabled: { opacity: 0.55 },
});
