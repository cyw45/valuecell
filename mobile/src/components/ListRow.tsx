import type { ReactNode } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { ChevronRight } from "lucide-react-native";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

export type ListRowProps = {
  title: string;
  subtitle?: string;
  leading?: ReactNode;
  trailing?: ReactNode;
  onPress?: () => void;
  disabled?: boolean;
  showChevron?: boolean;
  accessibilityLabel?: string;
};

export function ListRow({
  title,
  subtitle,
  leading,
  trailing,
  onPress,
  disabled = false,
  showChevron = Boolean(onPress),
  accessibilityLabel,
}: ListRowProps) {
  const { tokens } = useTheme();
  const styles = StyleSheet.create({
    root: { alignItems: "center", backgroundColor: tokens.surface, borderColor: tokens.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 56, minWidth: 44, paddingHorizontal: spacing.sm, paddingVertical: spacing.xs },
    leading: { alignItems: "center", justifyContent: "center", minHeight: 44, minWidth: 44 },
    copy: { flex: 1, gap: spacing.xxs },
    title: { color: disabled ? tokens.textMuted : tokens.text, fontSize: 15, fontWeight: "700" },
    subtitle: { color: tokens.textMuted, fontSize: 12, lineHeight: 17 },
    trailing: { alignItems: "center", flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 44 },
  });
  const content = <>{leading ? <View style={styles.leading}>{leading}</View> : null}<View style={styles.copy}><Text numberOfLines={1} style={styles.title}>{title}</Text>{subtitle ? <Text numberOfLines={2} style={styles.subtitle}>{subtitle}</Text> : null}</View><View style={styles.trailing}>{trailing}{showChevron ? <ChevronRight color={tokens.textMuted} size={18} /> : null}</View></>;

  if (!onPress) return <View style={styles.root}>{content}</View>;
  return <Pressable accessibilityLabel={accessibilityLabel ?? title} accessibilityRole="button" disabled={disabled} onPress={onPress} style={({ pressed }) => [styles.root, disabled && { opacity: 0.5 }, pressed && !disabled && { backgroundColor: tokens.surfaceRaised }]}>{content}</Pressable>;
}
