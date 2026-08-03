import type { ReactNode } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, type PressableProps, type StyleProp, type ViewStyle } from "react-native";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

export type DangerButtonProps = Omit<PressableProps, "children" | "style" | "disabled"> & {
  label: string;
  leading?: ReactNode;
  loading?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
  style?: StyleProp<ViewStyle>;
};

export function DangerButton({
  label,
  leading,
  loading = false,
  disabled = false,
  fullWidth = true,
  style,
  ...pressableProps
}: DangerButtonProps) {
  const { tokens } = useTheme();
  const unavailable = disabled || loading;
  const styles = StyleSheet.create({
    root: { alignItems: "center", backgroundColor: tokens.negativeSoft, borderColor: tokens.negative, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48, minWidth: 44, paddingHorizontal: spacing.md },
    label: { color: tokens.negative, fontSize: 15, fontWeight: "800" },
  });

  return (
    <Pressable
      {...pressableProps}
      accessibilityLabel={pressableProps.accessibilityLabel ?? label}
      accessibilityRole="button"
      disabled={unavailable}
      style={({ pressed }) => [styles.root, fullWidth && { alignSelf: "stretch" }, unavailable && { opacity: 0.48 }, pressed && !unavailable && { opacity: 0.72 }, style]}
    >
      {loading ? <ActivityIndicator color={tokens.negative} /> : leading}
      <Text style={styles.label}>{label}</Text>
    </Pressable>
  );
}
