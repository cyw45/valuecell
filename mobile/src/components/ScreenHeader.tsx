import type { ReactNode } from "react";
import { Pressable, StyleSheet, Text, View, type StyleProp, type ViewStyle } from "react-native";
import { ArrowLeft } from "lucide-react-native";
import { useI18n } from "../i18n";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

export type ScreenHeaderProps = {
  title: string;
  subtitle?: string;
  onBack?: () => void;
  backAccessibilityLabel?: string;
  right?: ReactNode;
  actionLabel?: string;
  onAction?: () => void;
  style?: StyleProp<ViewStyle>;
};

export function ScreenHeader({
  title,
  subtitle,
  onBack,
  backAccessibilityLabel,
  right,
  actionLabel,
  onAction,
  style,
}: ScreenHeaderProps) {
  const { t } = useI18n();
  const { tokens } = useTheme();
  const styles = StyleSheet.create({
    root: { alignItems: "center", flexDirection: "row", gap: spacing.sm, minHeight: 52 },
    back: { alignItems: "center", borderRadius: radius.sm, justifyContent: "center", minHeight: 44, minWidth: 44 },
    content: { flex: 1, gap: spacing.xxs },
    title: { color: tokens.text, fontSize: 20, fontWeight: "800", letterSpacing: -0.3 },
    subtitle: { color: tokens.textMuted, fontSize: 13, lineHeight: 18 },
    right: { alignItems: "center", justifyContent: "center", minHeight: 44 },
  });

  return (
    <View style={[styles.root, style]}>
      {onBack ? (
        <Pressable
          accessibilityLabel={backAccessibilityLabel ?? t("common.back")}
          accessibilityRole="button"
          hitSlop={4}
          onPress={onBack}
          style={({ pressed }) => [styles.back, pressed && { backgroundColor: tokens.surfaceMuted }]}
        >
          <ArrowLeft color={tokens.text} size={20} />
        </Pressable>
      ) : null}
      <View style={styles.content}>
        <Text numberOfLines={1} style={styles.title}>{title}</Text>
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      </View>
      {right ?? (actionLabel && onAction ? <Pressable accessibilityLabel={actionLabel} accessibilityRole="button" onPress={onAction} style={({ pressed }) => [styles.right, pressed && { backgroundColor: tokens.surfaceRaised }]}><Text style={styles.subtitle}>{actionLabel}</Text></Pressable> : null)}
    </View>
  );
}
