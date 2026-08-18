import type { PropsWithChildren } from "react";
import { ActivityIndicator, Modal, Pressable, StyleSheet, Text, View } from "react-native";
import { useI18n } from "../i18n";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

export type ConfirmSheetProps = PropsWithChildren<{
  visible: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  confirming?: boolean;
  confirmDisabled?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}>;

export function ConfirmSheet({
  visible,
  title,
  message,
  confirmLabel,
  cancelLabel,
  destructive = false,
  confirming = false,
  confirmDisabled = false,
  onConfirm,
  onCancel,
  children,
}: ConfirmSheetProps) {
  const { t } = useI18n();
  const { tokens } = useTheme();
  const actionColor = destructive ? tokens.negative : tokens.primary;
  const actionSoft = destructive ? tokens.negativeSoft : tokens.primarySoft;
  const styles = StyleSheet.create({
    overlay: { backgroundColor: "rgba(0, 0, 0, 0.56)", flex: 1, justifyContent: "flex-end" },
    backdrop: { ...StyleSheet.absoluteFill },
    sheet: { backgroundColor: tokens.surface, borderColor: tokens.border, borderTopLeftRadius: radius.lg, borderTopRightRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.lg, paddingBottom: spacing.xl },
    title: { color: tokens.text, fontSize: 19, fontWeight: "800" },
    message: { color: tokens.textMuted, fontSize: 14, lineHeight: 21 },
    actions: { gap: spacing.xs },
    confirm: { alignItems: "center", backgroundColor: actionSoft, borderColor: actionColor, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48, minWidth: 44, paddingHorizontal: spacing.md },
    confirmText: { color: actionColor, fontSize: 15, fontWeight: "800" },
    cancel: { alignItems: "center", borderColor: tokens.border, borderRadius: radius.sm, borderWidth: 1, justifyContent: "center", minHeight: 48, minWidth: 44, paddingHorizontal: spacing.md },
    cancelText: { color: tokens.text, fontSize: 15, fontWeight: "800" },
  });

  return (
    <Modal animationType="slide" onRequestClose={onCancel} transparent visible={visible}>
      <View accessibilityViewIsModal style={styles.overlay}>
        <Pressable accessibilityLabel={t("common.close")} accessibilityRole="button" onPress={onCancel} style={styles.backdrop} />
        <View style={styles.sheet}>
          <Text style={styles.title}>{title}</Text>
          <Text style={styles.message}>{message}</Text>
          {children}
          <View style={styles.actions}>
            <Pressable accessibilityRole="button" disabled={confirming || confirmDisabled} onPress={onConfirm} style={({ pressed }) => [styles.confirm, (confirming || confirmDisabled) && { opacity: 0.55 }, pressed && !confirming && !confirmDisabled && { opacity: 0.78 }]}>
              {confirming ? <ActivityIndicator color={actionColor} /> : null}
              <Text style={styles.confirmText}>{confirmLabel ?? t("common.confirm")}</Text>
            </Pressable>
            <Pressable accessibilityRole="button" disabled={confirming} onPress={onCancel} style={({ pressed }) => [styles.cancel, confirming && { opacity: 0.55 }, pressed && !confirming && { backgroundColor: tokens.surfaceRaised }]}><Text style={styles.cancelText}>{cancelLabel ?? t("common.cancel")}</Text></Pressable>
          </View>
        </View>
      </View>
    </Modal>
  );
}
