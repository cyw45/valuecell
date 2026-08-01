import type { PropsWithChildren, ReactNode } from "react";
import { Pressable, StyleSheet, Text, View, type StyleProp, type ViewStyle } from "react-native";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

export type SectionCardProps = PropsWithChildren<{
  title?: string;
  description?: string;
  action?: ReactNode;
  actionLabel?: string;
  onAction?: () => void;
  style?: StyleProp<ViewStyle>;
}>;

export function SectionCard({ title, description, action, actionLabel, onAction, children, style }: SectionCardProps) {
  const { tokens } = useTheme();
  const styles = StyleSheet.create({
    root: { backgroundColor: tokens.surface, borderColor: tokens.border, borderRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.md },
    header: { alignItems: "flex-start", flexDirection: "row", gap: spacing.sm },
    heading: { flex: 1, gap: spacing.xxs },
    title: { color: tokens.text, fontSize: 16, fontWeight: "800" },
    description: { color: tokens.textMuted, fontSize: 13, lineHeight: 19 },
    action: { alignItems: "center", justifyContent: "center", minHeight: 44 },
  });
  const resolvedAction = action ?? (actionLabel && onAction ? <Pressable accessibilityLabel={actionLabel} accessibilityRole="button" onPress={onAction} style={({ pressed }) => [styles.action, pressed && { backgroundColor: tokens.surfaceRaised }]}><Text style={styles.description}>{actionLabel}</Text></Pressable> : null);
  const hasHeader = title || description || resolvedAction;

  return (
    <View style={[styles.root, style]}>
      {hasHeader ? (
        <View style={styles.header}>
          <View style={styles.heading}>
            {title ? <Text style={styles.title}>{title}</Text> : null}
            {description ? <Text style={styles.description}>{description}</Text> : null}
          </View>
          {resolvedAction ? <View style={styles.action}>{resolvedAction}</View> : null}
        </View>
      ) : null}
      {children}
    </View>
  );
}
