import { Pressable, StyleSheet, Text, View, type StyleProp, type ViewStyle } from "react-native";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

export type MetricTone = "default" | "positive" | "negative" | "warning";

export type MetricCardProps = {
  label: string;
  value: string | number;
  caption?: string;
  tone?: MetricTone;
  onPress?: () => void;
  style?: StyleProp<ViewStyle>;
};

export function MetricCard({ label, value, caption, tone = "default", onPress, style }: MetricCardProps) {
  const { tokens } = useTheme();
  const valueColor = {
    default: tokens.text,
    positive: tokens.positive,
    negative: tokens.negative,
    warning: tokens.warning,
  }[tone];
  const styles = StyleSheet.create({
    root: { backgroundColor: tokens.surface, borderColor: tokens.border, borderRadius: radius.md, borderWidth: 1, flexGrow: 1, gap: spacing.xxs, minHeight: 88, minWidth: 44, padding: spacing.sm },
    label: { color: tokens.textMuted, fontSize: 12, fontWeight: "700" },
    value: { color: valueColor, fontSize: 20, fontWeight: "800", letterSpacing: -0.4 },
    caption: { color: tokens.textMuted, fontSize: 11, lineHeight: 15 },
  });
  const content = <><Text style={styles.label}>{label}</Text><Text numberOfLines={1} style={styles.value}>{value}</Text>{caption ? <Text numberOfLines={2} style={styles.caption}>{caption}</Text> : null}</>;

  if (!onPress) return <View style={[styles.root, style]}>{content}</View>;
  return (
    <Pressable
      accessibilityLabel={label}
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [styles.root, style, pressed && { backgroundColor: tokens.surfaceRaised }]}
    >
      {content}
    </Pressable>
  );
}
