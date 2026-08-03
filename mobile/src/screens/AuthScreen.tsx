import { useMemo, useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { BarChart3, Building2, LockKeyhole, Mail, ShieldCheck } from "lucide-react-native";
import { useI18n } from "../i18n";
import { useSession } from "../session";
import { radius, spacing } from "../theme";
import { useTheme } from "../theme-context";

type AuthMode = "login" | "register";
type TenantType = "personal" | "enterprise";

const tenantTranslations = {
  personal: { label: "auth.tenant.personal.label", copy: "auth.tenant.personal.copy" },
  enterprise: { label: "auth.tenant.enterprise.label", copy: "auth.tenant.enterprise.copy" },
} as const;

export default function AuthScreen() {
  const { t } = useI18n();
  const { tokens } = useTheme();
  const { register, signIn } = useSession();
  const [mode, setMode] = useState<AuthMode>("login");
  const [tenantType, setTenantType] = useState<TenantType>("personal");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [workspaceName, setWorkspaceName] = useState("");
  const [organizationName, setOrganizationName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const isRegistration = mode === "register";

  const styles = useMemo(() => StyleSheet.create({
    page: { backgroundColor: tokens.canvas, flex: 1 },
    content: { flexGrow: 1, gap: spacing.xl, justifyContent: "center", paddingHorizontal: spacing.lg, paddingVertical: spacing.xl, zIndex: 1 },
    orbPrimary: { backgroundColor: tokens.primarySoft, borderRadius: 155, height: 310, position: "absolute", right: -100, top: -140, width: 310 },
    orbSecondary: { backgroundColor: tokens.positiveSoft, borderRadius: 130, bottom: -160, height: 260, left: -100, position: "absolute", width: 260 },
    brandRow: { alignItems: "center", flexDirection: "row", gap: spacing.sm },
    brandIcon: { alignItems: "center", backgroundColor: tokens.primary, borderRadius: radius.sm, height: 44, justifyContent: "center", width: 44 },
    brand: { color: tokens.text, fontSize: 14, fontWeight: "800", letterSpacing: 1.8 },
    hero: { gap: spacing.sm },
    eyebrow: { color: tokens.primary, fontSize: 11, fontWeight: "800", letterSpacing: 1.2 },
    title: { color: tokens.text, fontSize: 31, fontWeight: "800", letterSpacing: -1.1, lineHeight: 40 },
    subtitle: { color: tokens.textMuted, fontSize: 15, lineHeight: 23 },
    card: { backgroundColor: tokens.surface, borderColor: tokens.border, borderRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.lg },
    cardTitle: { color: tokens.text, fontSize: 20, fontWeight: "700" },
    cardCopy: { color: tokens.textMuted, fontSize: 13, marginTop: -spacing.sm },
    field: { alignItems: "center", backgroundColor: tokens.canvas, borderColor: tokens.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 52, paddingHorizontal: spacing.sm },
    input: { color: tokens.text, flex: 1, fontSize: 16, minHeight: 50 },
    tenantLabel: { color: tokens.textMuted, fontSize: 13, fontWeight: "700" },
    tenantChoices: { flexDirection: "row", gap: spacing.xs },
    tenantChoice: { borderColor: tokens.border, borderRadius: radius.md, borderWidth: 1, flex: 1, gap: spacing.xxs, minHeight: 76, padding: spacing.sm },
    tenantChoiceActive: { backgroundColor: tokens.primarySoft, borderColor: tokens.primary },
    tenantChoiceTitle: { color: tokens.text, fontSize: 14, fontWeight: "800" },
    tenantChoiceTitleActive: { color: tokens.primary },
    tenantChoiceCopy: { color: tokens.textMuted, fontSize: 11, lineHeight: 16 },
    error: { color: tokens.negative, fontSize: 13, lineHeight: 19 },
    submit: { alignItems: "center", backgroundColor: tokens.primary, borderRadius: radius.sm, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md },
    submitPressed: { opacity: 0.72 },
    submitText: { color: tokens.canvas, fontSize: 16, fontWeight: "800" },
    modeButton: { alignItems: "center", borderColor: tokens.border, borderRadius: radius.sm, borderWidth: 1, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.sm },
    modeButtonText: { color: tokens.primary, fontSize: 14, fontWeight: "800" },
    securityRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
    securityCopy: { color: tokens.textMuted, flex: 1, fontSize: 12, lineHeight: 17 },
  }), [tokens]);

  function switchMode(nextMode: AuthMode) {
    setMode(nextMode);
    setError("");
  }

  async function submit() {
    if (!email.trim() || !password) {
      setError(t("auth.validation.credentials"));
      return;
    }
    if (isRegistration && !workspaceName.trim()) {
      setError(t("auth.validation.registration"));
      return;
    }
    if (isRegistration && tenantType === "enterprise" && !organizationName.trim()) {
      setError(t("auth.validation.organization"));
      return;
    }

    setSubmitting(true);
    setError("");
    try {
      if (isRegistration) {
        await register({
          email: email.trim(),
          password,
          tenant_type: tenantType,
          workspace_name: workspaceName.trim(),
          organization_name: tenantType === "enterprise" ? organizationName.trim() : undefined,
        });
      } else {
        await signIn(email, password);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : t(isRegistration ? "auth.failed.register" : "auth.failed.login"));
    } finally {
      setSubmitting(false);
    }
  }

  const workspacePlaceholder = t(tenantType === "enterprise" ? "auth.workspace.enterprise" : "auth.workspace.personal");

  return (
    <KeyboardAvoidingView behavior={Platform.select({ ios: "padding", default: undefined })} style={styles.page}>
      <View pointerEvents="none" style={styles.orbPrimary} />
      <View pointerEvents="none" style={styles.orbSecondary} />
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <View style={styles.brandRow}>
          <View style={styles.brandIcon}><BarChart3 color={tokens.canvas} size={21} /></View>
          <Text style={styles.brand}>VALUE CELL</Text>
        </View>
        <View style={styles.hero}>
          <Text style={styles.eyebrow}>{t("auth.eyebrow")}</Text>
          <Text style={styles.title}>{t("auth.hero.title")}</Text>
          <Text style={styles.subtitle}>{t("auth.hero.copy")}</Text>
        </View>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>{t(isRegistration ? "auth.register.title" : "auth.login.title")}</Text>
          <Text style={styles.cardCopy}>{t(isRegistration ? "auth.register.copy" : "auth.login.copy")}</Text>
          <View style={styles.field}>
            <Mail color={tokens.textMuted} size={18} />
            <TextInput
              accessibilityLabel={t("auth.email")}
              autoCapitalize="none"
              autoComplete="email"
              keyboardType="email-address"
              onChangeText={setEmail}
              placeholder={t("auth.email")}
              placeholderTextColor={tokens.textMuted}
              style={styles.input}
              textContentType="emailAddress"
              value={email}
            />
          </View>
          <View style={styles.field}>
            <LockKeyhole color={tokens.textMuted} size={18} />
            <TextInput
              accessibilityLabel={t("auth.password")}
              autoComplete="password"
              onChangeText={setPassword}
              placeholder={t("auth.password")}
              placeholderTextColor={tokens.textMuted}
              secureTextEntry
              style={styles.input}
              textContentType="password"
              value={password}
            />
          </View>
          {isRegistration ? (
            <>
              <Text style={styles.tenantLabel}>{t("auth.tenantType")}</Text>
              <View style={styles.tenantChoices}>
                {(["personal", "enterprise"] as const).map((type) => {
                  const selected = tenantType === type;
                  const translation = tenantTranslations[type];
                  return (
                    <Pressable
                      accessibilityRole="button"
                      key={type}
                      onPress={() => setTenantType(type)}
                      style={({ pressed }) => [styles.tenantChoice, selected && styles.tenantChoiceActive, pressed && { opacity: 0.76 }]}
                    >
                      <Text style={[styles.tenantChoiceTitle, selected && styles.tenantChoiceTitleActive]}>{t(translation.label)}</Text>
                      <Text style={styles.tenantChoiceCopy}>{t(translation.copy)}</Text>
                    </Pressable>
                  );
                })}
              </View>
              <View style={styles.field}>
                <Building2 color={tokens.textMuted} size={18} />
                <TextInput
                  accessibilityLabel={workspacePlaceholder}
                  autoCapitalize="words"
                  onChangeText={setWorkspaceName}
                  placeholder={workspacePlaceholder}
                  placeholderTextColor={tokens.textMuted}
                  style={styles.input}
                  value={workspaceName}
                />
              </View>
              {tenantType === "enterprise" ? (
                <View style={styles.field}>
                  <Building2 color={tokens.textMuted} size={18} />
                  <TextInput
                    accessibilityLabel={t("auth.organizationName")}
                    autoCapitalize="words"
                    onChangeText={setOrganizationName}
                    placeholder={t("auth.organizationName")}
                    placeholderTextColor={tokens.textMuted}
                    style={styles.input}
                    value={organizationName}
                  />
                </View>
              ) : null}
            </>
          ) : null}
          {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
          <Pressable
            accessibilityRole="button"
            disabled={submitting}
            onPress={() => void submit()}
            style={({ pressed }) => [styles.submit, (pressed || submitting) && styles.submitPressed]}
          >
            {submitting ? <ActivityIndicator color={tokens.canvas} /> : <Text style={styles.submitText}>{t(isRegistration ? "auth.submit.register" : "auth.submit.login")}</Text>}
          </Pressable>
          <Pressable accessibilityRole="button" disabled={submitting} onPress={() => switchMode(isRegistration ? "login" : "register")} style={({ pressed }) => [styles.modeButton, pressed && !submitting && { backgroundColor: tokens.surfaceRaised }]}>
            <Text style={styles.modeButtonText}>{t(isRegistration ? "auth.switch.login" : "auth.switch.register")}</Text>
          </Pressable>
          <View style={styles.securityRow}>
            <ShieldCheck color={tokens.positive} size={16} />
            <Text style={styles.securityCopy}>{t("auth.security")}</Text>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
