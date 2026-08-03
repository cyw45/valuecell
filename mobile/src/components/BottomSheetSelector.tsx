import { Modal, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { Check } from "lucide-react-native";
import { useI18n } from "../i18n";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

export type BottomSheetOption<Value extends string> = {
  value: Value;
  label: string;
  description?: string;
  disabled?: boolean;
};

export type BottomSheetSelectorProps<Value extends string> = {
  visible: boolean;
  title: string;
  options: readonly BottomSheetOption<Value>[];
  value?: Value;
  selectedValue?: Value | null;
  onSelect: (value: Value) => void;
  onClose: () => void;
};

export function BottomSheetSelector<Value extends string>({
  visible,
  title,
  options,
  value,
  selectedValue,
  onSelect,
  onClose,
}: BottomSheetSelectorProps<Value>) {
  const activeValue = value ?? selectedValue;
  const { t } = useI18n();
  const { tokens } = useTheme();
  const styles = StyleSheet.create({
    overlay: { backgroundColor: "rgba(0, 0, 0, 0.56)", flex: 1, justifyContent: "flex-end" },
    backdrop: { ...StyleSheet.absoluteFill },
    sheet: { backgroundColor: tokens.surface, borderColor: tokens.border, borderTopLeftRadius: radius.lg, borderTopRightRadius: radius.lg, borderWidth: 1, gap: spacing.md, maxHeight: "78%", padding: spacing.lg, paddingBottom: spacing.xl },
    title: { color: tokens.text, fontSize: 19, fontWeight: "800" },
    list: { gap: spacing.xs },
    option: { alignItems: "center", borderColor: tokens.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 56, minWidth: 44, paddingHorizontal: spacing.sm, paddingVertical: spacing.xs },
    optionSelected: { backgroundColor: tokens.primarySoft, borderColor: tokens.primary },
    copy: { flex: 1, gap: spacing.xxs },
    label: { color: tokens.text, fontSize: 15, fontWeight: "800" },
    description: { color: tokens.textMuted, fontSize: 12, lineHeight: 17 },
    close: { alignItems: "center", borderColor: tokens.border, borderRadius: radius.sm, borderWidth: 1, justifyContent: "center", minHeight: 44, minWidth: 44 },
    closeText: { color: tokens.text, fontSize: 14, fontWeight: "800" },
  });

  return (
    <Modal animationType="slide" onRequestClose={onClose} transparent visible={visible}>
      <View accessibilityViewIsModal style={styles.overlay}>
        <Pressable accessibilityLabel={t("common.close")} accessibilityRole="button" onPress={onClose} style={styles.backdrop} />
        <View style={styles.sheet}>
          <Text style={styles.title}>{title}</Text>
          <ScrollView contentContainerStyle={styles.list} showsVerticalScrollIndicator={false}>
            {options.map((option) => {
              const selected = option.value === activeValue;
              return (
                <Pressable
                  accessibilityRole="button"
                  disabled={option.disabled}
                  key={option.value}
                  onPress={() => {
                    onSelect(option.value);
                    onClose();
                  }}
                  style={({ pressed }) => [styles.option, selected && styles.optionSelected, option.disabled && { opacity: 0.5 }, pressed && !option.disabled && { backgroundColor: selected ? tokens.primarySoft : tokens.surfaceRaised }]}
                >
                  <View style={styles.copy}>
                    <Text style={styles.label}>{option.label}</Text>
                    {option.description ? <Text style={styles.description}>{option.description}</Text> : null}
                  </View>
                  {selected ? <Check color={tokens.primary} size={20} /> : null}
                </Pressable>
              );
            })}
          </ScrollView>
          <Pressable accessibilityRole="button" onPress={onClose} style={({ pressed }) => [styles.close, pressed && { backgroundColor: tokens.surfaceRaised }]}><Text style={styles.closeText}>{t("common.close")}</Text></Pressable>
        </View>
      </View>
    </Modal>
  );
}
