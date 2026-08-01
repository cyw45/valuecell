import type { ReactNode } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, type PressableProps, type StyleProp, type ViewStyle } from "react-native";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

export type PrimaryButtonProps = Omit<PressableProps, "children" | "style" | "disabled"> & {
  label: string;
  leading?: ReactNode;
  loading?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
  style?: StyleProp<ViewStyle>;
};

export function PrimaryButton({
  label,
  leading,
  loading = false,
  disabled = false,
  fullWidth = true,
  style,
  ...pressableProps
}: PrimaryButtonProps) {
  const { tokens } = useTheme();
  const unavailable = disabled || loading;
  const styles = StyleSheet.create({
    root: { alignItems: "center", backgroundColor: tokens.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48, minWidth: 44, paddingHorizontal: spacing.md },
    label: { color: tokens.canvas, fontSize: 15, fontWeight: "800" },
  });

  return (
    <Pressable
      {...pressableProps}
      accessibilityLabel={pressableProps.accessibilityLabel ?? label}
      accessibilityRole="button"
      disabled={unavailable}
      style={({ pressed }) => [styles.root, fullWidth && { alignSelf: "stretch" }, unavailable && { opacity: 0.48 }, pressed && !unavailable && { opacity: 0.78 }, style]}
    >
      {loading ? <ActivityIndicator color={tokens.canvas} /> : leading}
      <Text style={styles.label}>{label}</Text>
    </Pressable>
  );
}
