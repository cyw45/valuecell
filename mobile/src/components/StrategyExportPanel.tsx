import { useMemo, useState } from "react";
import { File, Paths } from "expo-file-system";
import * as Sharing from "expo-sharing";
import { Download } from "lucide-react-native";
import { Platform, Pressable, StyleSheet, Text, TextInput, View } from "react-native";
import { api, type StrategyExportFile } from "../api";
import { palette, radius, spacing } from "../theme";
import { PrimaryButton } from "./PrimaryButton";
import { SectionCard } from "./SectionCard";

type StrategyExportPanelProps = {
  strategyId: string;
};

type ResolvedDateRange = {
  fromDate?: string;
  toDate?: string;
  summary: string;
  validationError: string | null;
};

const XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

function isUtcCalendarDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const date = new Date(`${value}T00:00:00.000Z`);
  return !Number.isNaN(date.getTime()) && date.toISOString().slice(0, 10) === value;
}

function resolveDateRange(fromValue: string, toValue: string): ResolvedDateRange {
  const fromDate = fromValue.trim();
  const toDate = toValue.trim();
  if (!fromDate && !toDate) {
    return { summary: "全部历史记录", validationError: null };
  }

  const effectiveFromDate = fromDate || toDate;
  const effectiveToDate = toDate || fromDate;
  if (!isUtcCalendarDate(effectiveFromDate) || !isUtcCalendarDate(effectiveToDate)) {
    return {
      summary: "等待有效日期",
      validationError: "请输入有效的 YYYY-MM-DD 日期。",
    };
  }
  if (effectiveFromDate > effectiveToDate) {
    return {
      summary: "日期范围无效",
      validationError: "开始日期不能晚于结束日期。",
    };
  }
  if (effectiveFromDate === effectiveToDate) {
    return {
      fromDate: effectiveFromDate,
      toDate: effectiveToDate,
      summary: `单日 · ${effectiveFromDate}`,
      validationError: null,
    };
  }
  return {
    fromDate: effectiveFromDate,
    toDate: effectiveToDate,
    summary: `${effectiveFromDate} 至 ${effectiveToDate}`,
    validationError: null,
  };
}

function downloadOnWeb(file: StrategyExportFile): void {
  const documentRef = globalThis.document;
  if (!documentRef || typeof URL.createObjectURL !== "function") {
    throw new Error("当前浏览器不支持 Excel 文件下载。请更换浏览器后重试。");
  }

  const url = URL.createObjectURL(
    new Blob([file.bytes], { type: file.mimeType || XLSX_MIME_TYPE }),
  );
  const anchor = documentRef.createElement("a");
  anchor.download = file.filename;
  anchor.href = url;
  anchor.style.display = "none";
  documentRef.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

async function shareOnDevice(file: StrategyExportFile): Promise<void> {
  const cacheFile = new File(Paths.cache, file.filename);
  cacheFile.write(file.bytes);

  if (!(await Sharing.isAvailableAsync())) {
    throw new Error("当前设备无法打开系统分享面板。请确认已安装支持 Excel 的应用后重试。");
  }
  await Sharing.shareAsync(cacheFile.uri, {
    UTI: "org.openxmlformats.spreadsheetml.sheet",
    dialogTitle: "分享策略历史 Excel",
    mimeType: file.mimeType || XLSX_MIME_TYPE,
  });
}

/** Broker-grade date range controls for a strategy's authenticated XLSX export. */
export function StrategyExportPanel({ strategyId }: StrategyExportPanelProps) {
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [exportError, setExportError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const range = useMemo(() => resolveDateRange(fromDate, toDate), [fromDate, toDate]);
  const isWeb = Platform.OS === "web";

  const updateFromDate = (value: string) => {
    setFromDate(value);
    setExportError(null);
  };
  const updateToDate = (value: string) => {
    setToDate(value);
    setExportError(null);
  };
  const exportHistory = async () => {
    if (isExporting || range.validationError) return;
    setExportError(null);
    setIsExporting(true);
    try {
      const file = await api.strategyExport(strategyId, {
        fromDate: range.fromDate,
        toDate: range.toDate,
      });
      if (isWeb) {
        downloadOnWeb(file);
      } else {
        await shareOnDevice(file);
      }
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "服务未完成本次 Excel 导出。");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <SectionCard
      description="服务端生成策略参数、成交明细、资金变化、执行明细与资金费，不会在设备上拼接或推断数据。"
      title="策略历史导出"
    >
      <View style={styles.rangeStatus}>
        <Text style={styles.rangeStatusLabel}>当前导出范围</Text>
        <Text style={styles.rangeStatusValue}>{range.summary}</Text>
      </View>
      <View style={styles.dateFields}>
        <View style={styles.field}>
          <Text style={styles.label}>开始日期（可选）</Text>
          <TextInput
            accessibilityLabel="导出开始日期 YYYY-MM-DD"
            autoCapitalize="none"
            editable={!isExporting}
            keyboardType="numbers-and-punctuation"
            maxLength={10}
            onChangeText={updateFromDate}
            placeholder="YYYY-MM-DD"
            placeholderTextColor={palette.textMuted}
            style={styles.input}
            value={fromDate}
          />
        </View>
        <View style={styles.field}>
          <Text style={styles.label}>结束日期（可选）</Text>
          <TextInput
            accessibilityLabel="导出结束日期 YYYY-MM-DD"
            autoCapitalize="none"
            editable={!isExporting}
            keyboardType="numbers-and-punctuation"
            maxLength={10}
            onChangeText={updateToDate}
            placeholder="YYYY-MM-DD"
            placeholderTextColor={palette.textMuted}
            style={styles.input}
            value={toDate}
          />
        </View>
      </View>
      <Text style={styles.helper}>
        日期按 UTC 自然日解释并包含开始和结束当天；只填写一个日期会按该单日导出，留空则导出全部历史记录。
      </Text>
      {range.validationError ? <Text style={styles.validationError}>{range.validationError}</Text> : null}
      {exportError ? (
        <View style={styles.serverError}>
          <View style={styles.serverErrorCopy}>
            <Text style={styles.serverErrorTitle}>导出失败</Text>
            <Text style={styles.serverErrorMessage}>{exportError}</Text>
          </View>
          <Pressable
            accessibilityLabel="关闭导出错误"
            accessibilityRole="button"
            onPress={() => setExportError(null)}
            style={({ pressed }) => [styles.dismissButton, pressed && styles.pressed]}
          >
            <Text style={styles.dismissButtonText}>关闭</Text>
          </Pressable>
        </View>
      ) : null}
      <PrimaryButton
        disabled={Boolean(range.validationError)}
        label={isExporting ? "正在生成 Excel…" : isWeb ? "下载 Excel" : "导出并分享 Excel"}
        leading={<Download color={palette.canvas} size={19} />}
        loading={isExporting}
        onPress={() => void exportHistory()}
      />
    </SectionCard>
  );
}

const styles = StyleSheet.create({
  rangeStatus: {
    alignItems: "center",
    backgroundColor: palette.surfaceMuted,
    borderColor: palette.border,
    borderRadius: radius.sm,
    borderWidth: 1,
    flexDirection: "row",
    gap: spacing.sm,
    justifyContent: "space-between",
    minHeight: 44,
    paddingHorizontal: spacing.sm,
  },
  rangeStatusLabel: { color: palette.textMuted, fontSize: 12, fontWeight: "800" },
  rangeStatusValue: { color: palette.text, fontSize: 13, fontWeight: "900", textAlign: "right" },
  dateFields: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  field: { flexGrow: 1, gap: spacing.xs, minWidth: 136 },
  label: { color: palette.textMuted, fontSize: 12, fontWeight: "800" },
  input: {
    backgroundColor: palette.canvas,
    borderColor: palette.border,
    borderRadius: radius.sm,
    borderWidth: 1,
    color: palette.text,
    fontSize: 15,
    height: 48,
    paddingHorizontal: spacing.sm,
  },
  helper: { color: palette.textMuted, fontSize: 12, lineHeight: 19 },
  validationError: { color: palette.negative, fontSize: 12, fontWeight: "800", lineHeight: 19 },
  serverError: {
    alignItems: "flex-start",
    backgroundColor: palette.negativeSoft,
    borderColor: palette.negative,
    borderRadius: radius.sm,
    borderWidth: 1,
    flexDirection: "row",
    gap: spacing.sm,
    padding: spacing.sm,
  },
  serverErrorCopy: { flex: 1, gap: 2 },
  serverErrorTitle: { color: palette.negative, fontSize: 13, fontWeight: "900" },
  serverErrorMessage: { color: palette.text, fontSize: 12, lineHeight: 18 },
  dismissButton: { alignItems: "center", justifyContent: "center", minHeight: 36, paddingHorizontal: spacing.xs },
  dismissButtonText: { color: palette.negative, fontSize: 12, fontWeight: "900" },
  pressed: { opacity: 0.76 },
});
