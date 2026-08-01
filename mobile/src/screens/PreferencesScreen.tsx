import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { usePreferences } from "../preferences";
import { useI18n } from "../i18n";
import { ScreenHeader, StatePanel } from "../components";
import { palette, radius, spacing } from "../theme";

function OptionRow<T extends string>({ label, onSelect, options, value }: { label: string; value: T; options: ReadonlyArray<{ label: string; value: T }>; onSelect: (value: T) => void }) {
  return <View style={styles.group}><Text style={styles.label}>{label}</Text><View style={styles.options}>{options.map((option) => <Pressable accessibilityRole="button" key={option.value} onPress={() => onSelect(option.value)} style={[styles.option, value === option.value && styles.optionActive]}><Text style={[styles.optionText, value === option.value && styles.optionTextActive]}>{option.label}</Text></Pressable>)}</View></View>;
}

export default function PreferencesScreen() {
  const { preferences, ready, setLanguage, setMarketDataRefreshMode, setStockColorMode, setTheme } = usePreferences();
  const { t } = useI18n();
  if (!ready) return <StatePanel description="正在恢复本机偏好。" title="偏好设置" />;
  return <ScrollView contentContainerStyle={styles.content} style={styles.page}>
    <ScreenHeader subtitle="仅存储在本设备；不包含交易所凭据或账户秘密。" title={t("preferences.title") || "偏好设置"} />
    <OptionRow label="语言" onSelect={setLanguage} value={preferences.language} options={[{ value: "zh_CN", label: "简体中文" }, { value: "zh_TW", label: "繁體中文" }, { value: "en", label: "English" }, { value: "ja", label: "日本語" }]} />
    <OptionRow label="主题" onSelect={setTheme} value={preferences.theme} options={[{ value: "dark", label: "深色" }, { value: "light", label: "浅色" }, { value: "system", label: "跟随系统" }]} />
    <OptionRow label="涨跌颜色" onSelect={setStockColorMode} value={preferences.stockColorMode} options={[{ value: "GREEN_UP_RED_DOWN", label: "涨绿跌红" }, { value: "RED_UP_GREEN_DOWN", label: "涨红跌绿" }]} />
    <OptionRow label="行情自动刷新" onSelect={setMarketDataRefreshMode} value={preferences.marketDataRefreshMode} options={[{ value: "manual", label: "手动" }, { value: "5s", label: "5 秒" }, { value: "15s", label: "15 秒" }, { value: "30s", label: "30 秒" }, { value: "1m", label: "1 分钟" }, { value: "5m", label: "5 分钟" }]} />
    <View style={styles.notice}><Text style={styles.noticeTitle}>本机偏好</Text><Text style={styles.noticeCopy}>更改将在应用重启后保留。语言、主题、涨跌色和行情刷新模式不会同步至 SaaS 服务端。</Text></View>
  </ScrollView>;
}

const styles = StyleSheet.create({ page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.lg, padding: spacing.md, paddingBottom: spacing.xl }, group: { gap: spacing.sm }, label: { color: palette.text, fontSize: 16, fontWeight: "800" }, options: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs }, option: { borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, minHeight: 44, justifyContent: "center", paddingHorizontal: spacing.md }, optionActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, optionText: { color: palette.textMuted, fontSize: 13, fontWeight: "700" }, optionTextActive: { color: palette.primary }, notice: { backgroundColor: palette.surfaceMuted, borderRadius: radius.md, gap: spacing.xs, padding: spacing.md }, noticeTitle: { color: palette.text, fontSize: 14, fontWeight: "800" }, noticeCopy: { color: palette.textMuted, fontSize: 13, lineHeight: 20 }, });
