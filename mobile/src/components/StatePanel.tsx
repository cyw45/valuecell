import type { ReactNode } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";
import { useI18n } from "../i18n";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

export type StatePanelProps = {
  state?: "loading" | "empty" | "error";
  title?: string;
  message?: string;
  description?: string;
  error?: Error | string | null;
  onRetry?: () => void;
  onAction?: () => void;
  actionLabel?: string;
  retryLabel?: string;
  tone?: "error" | "warning";
  children?: ReactNode;
};

export function StatePanel({ state = "empty", title, message, description, error, onRetry, onAction, actionLabel, retryLabel, children }: StatePanelProps) {
  const { t } = useI18n();
  const { tokens } = useTheme();
  const resolvedMessage = state === "error" && error
    ? (typeof error === "string" ? error : error.message)
    : message ?? description;
  const titles = {
    loading: t("state.loading.title"),
    empty: t("state.empty.title"),
    error: t("state.error.title"),
  };
  const tone = state === "error" ? tokens.negative : state === "loading" ? tokens.primary : tokens.textMuted;
  const styles = StyleSheet.create({
    root: { alignItems: "center", backgroundColor: tokens.surface, borderColor: tokens.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.lg },
    title: { color: tone, fontSize: 15, fontWeight: "800", textAlign: "center" },
    message: { color: tokens.textMuted, fontSize: 13, lineHeight: 19, textAlign: "center" },
    retry: { alignItems: "center", borderColor: tokens.primary, borderRadius: radius.sm, borderWidth: 1, justifyContent: "center", minHeight: 44, minWidth: 88, paddingHorizontal: spacing.sm },
    retryText: { color: tokens.primary, fontSize: 14, fontWeight: "800" },
  });

  return (
    <View accessibilityLiveRegion="polite" style={styles.root}>
      {state === "loading" ? <ActivityIndicator color={tokens.primary} /> : null}
      <Text style={styles.title}>{title ?? titles[state]}</Text>
      {resolvedMessage ? <Text style={styles.message}>{resolvedMessage}</Text> : null}
      {children}
      {onRetry ?? onAction ? <Pressable accessibilityRole="button" onPress={onRetry ?? onAction} style={({ pressed }) => [styles.retry, pressed && { backgroundColor: tokens.primarySoft }]}><Text style={styles.retryText}>{retryLabel ?? actionLabel ?? t("common.retry")}</Text></Pressable> : null}
    </View>
  );
}
