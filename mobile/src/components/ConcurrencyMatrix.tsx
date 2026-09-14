import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { ChevronRight, CircleDollarSign, LineChart, ReceiptText, Wallet } from "lucide-react-native";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import type { ConcurrencyMatrixRow } from "../concurrency";
import {
  allocationStateLabel,
  allocationStateTone,
  formatRatioPercent,
  formatUsdt,
  strategyKindLabel,
  utilizationTone,
} from "../concurrency";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";
import { AnimatedQuote } from "./AnimatedQuote";

/**
 * 四策略并发运行矩阵 in mobile form: one card per strategy, all reading the
 * same allocator facts as the Web table. A card tap switches the console's
 * active strategy in place instead of reloading the screen.
 */

export type ConcurrencyMatrixCardProps = {
  row: ConcurrencyMatrixRow;
  selected: boolean;
  onSelect: () => void;
  onOpenTrades: () => void;
  onOpenCharts: () => void;
  onEditCap: () => void;
};

function toneColor(
  tone: "default" | "positive" | "negative" | "warning",
  tokens: ReturnType<typeof useTheme>["tokens"],
) {
  if (tone === "positive") return tokens.positive;
  if (tone === "negative") return tokens.negative;
  if (tone === "warning") return tokens.warning;
  return tokens.textMuted;
}

export function ConcurrencyMatrixCard({
  row,
  selected,
  onSelect,
  onOpenTrades,
  onOpenCharts,
  onEditCap,
}: ConcurrencyMatrixCardProps) {
  const { tokens } = useTheme();
  const stateTone = allocationStateTone(row.allocationState);
  const stateColor = toneColor(stateTone, tokens);
  const utilizationPercent = Math.min(Math.max(row.utilizationPercent, 0), 100);
  const utilizationColor = toneColor(utilizationTone(utilizationPercent), tokens);
  const netTone = row.netPnl == null ? "default" : row.netPnl >= 0 ? "positive" : "negative";
  const statusColor = row.status === "running" ? tokens.positive : tokens.textMuted;
  const styles = StyleSheet.create({
    card: {
      backgroundColor: tokens.surface,
      borderColor: selected ? tokens.primary : tokens.border,
      borderRadius: radius.md,
      borderWidth: selected ? 2 : 1,
      gap: spacing.sm,
      padding: spacing.sm,
    },
    header: { gap: spacing.xxs },
    titleRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs, justifyContent: "space-between" },
    name: { color: tokens.text, flexShrink: 1, fontSize: 15, fontWeight: "900" },
    status: { fontSize: 11, fontWeight: "800" },
    meta: { color: tokens.textMuted, fontSize: 10, lineHeight: 15 },
    badgeRow: { alignItems: "center", flexDirection: "row", flexWrap: "wrap", gap: spacing.xxs },
    badge: { borderRadius: radius.sm, borderWidth: 1, paddingHorizontal: spacing.xs, paddingVertical: 3 },
    badgeText: { fontSize: 10, fontWeight: "800" },
    reason: { color: tokens.textMuted, fontSize: 10, lineHeight: 15 },
    utilizationRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
    utilizationLabel: { color: tokens.textMuted, fontSize: 11, fontWeight: "700" },
    utilizationTrack: { backgroundColor: tokens.surfaceMuted, borderRadius: radius.pill, flex: 1, height: 6, overflow: "hidden" },
    utilizationFill: { borderRadius: radius.pill, height: "100%" },
    utilizationValue: { fontSize: 12, fontWeight: "800" },
    metricGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
    metric: { backgroundColor: tokens.surfaceRaised, borderRadius: radius.sm, flexBasis: "30%", flexGrow: 1, gap: 2, minWidth: 92, padding: spacing.xs },
    metricLabel: { color: tokens.textMuted, fontSize: 10, fontWeight: "700" },
    metricValue: { color: tokens.text, fontSize: 13, fontWeight: "800" },
    statsRow: { color: tokens.textMuted, fontSize: 10, lineHeight: 15 },
    actions: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
    action: { alignItems: "center", borderColor: tokens.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xxs, minHeight: 40, paddingHorizontal: spacing.sm },
    actionPrimary: { borderColor: tokens.primary, backgroundColor: tokens.primarySoft },
    actionText: { color: tokens.text, fontSize: 12, fontWeight: "800" },
    actionTextPrimary: { color: tokens.primary },
    selectedHint: { color: tokens.primary, fontSize: 10, fontWeight: "800" },
  });

  const metric = (label: string, node: React.ReactNode) => (
    <View style={styles.metric}>
      <Text style={styles.metricLabel}>{label}</Text>
      {node}
    </View>
  );

  return (
    <Pressable
      accessibilityLabel={`查看 ${row.name} 的并发资金与执行事实`}
      accessibilityRole="button"
      onPress={onSelect}
      style={({ pressed }) => [styles.card, pressed && { opacity: 0.82 }]}
    >
      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Text numberOfLines={1} style={styles.name}>{row.name}</Text>
          <Text style={[styles.status, { color: statusColor }]}>{row.statusLabel}</Text>
        </View>
        <Text style={styles.meta}>
          {strategyKindLabel(row.kind)} · 批次 {row.batchId ?? "尚无当前批次"}
        </Text>
        <View style={styles.badgeRow}>
          <View style={[styles.badge, { backgroundColor: tokens.surfaceRaised, borderColor: stateColor }]}>
            <Text style={[styles.badgeText, { color: stateColor }]}>{allocationStateLabel(row.allocationState)}</Text>
          </View>
          {selected ? <Text style={styles.selectedHint}>当前选择 · 下方图表已切换</Text> : null}
        </View>
        {row.lifecycleReason ? (
          <Text numberOfLines={2} style={styles.reason}>{row.lifecycleReason}</Text>
        ) : null}
      </View>

      <View style={styles.utilizationRow}>
        <Text style={styles.utilizationLabel}>利用率</Text>
        <View style={styles.utilizationTrack}>
          <View style={[styles.utilizationFill, { backgroundColor: utilizationColor, width: `${utilizationPercent}%` }]} />
        </View>
        <Text style={[styles.utilizationValue, { color: utilizationColor }]}>
          {`${utilizationPercent.toFixed(1)}%`}
        </Text>
      </View>

      <View style={styles.metricGrid}>
        {metric("预留", <AnimatedQuote style={styles.metricValue} value={row.reserved} />)}
        {metric("占用", <AnimatedQuote style={styles.metricValue} value={row.occupied} />)}
        {metric("已释放", <AnimatedQuote style={styles.metricValue} value={row.released} />)}
        {metric("已实现 PnL", <AnimatedQuote style={styles.metricValue} value={row.realizedPnl} />)}
        {metric("未实现 PnL", <AnimatedQuote style={styles.metricValue} value={row.unrealizedPnl} />)}
        {metric(
          "净 PnL",
          <AnimatedQuote
            style={[styles.metricValue, { color: toneColor(netTone, tokens) }]}
            tone={netTone}
            value={row.netPnl}
          />,
        )}
      </View>

      <Text style={styles.statsRow}>
        收益率 {row.returnRatePercent == null ? "—" : `${row.returnRatePercent.toFixed(2)}%`} · 成交 {row.fillCount} 笔 ·
        完整交易 {row.completedTradeCount} 次
      </Text>
      <Text style={styles.statsRow}>
        胜率 {row.winPercent == null ? "—" : `${row.winPercent.toFixed(1)}%`} · 资金上限{" "}
        {row.maxReserved == null ? "未设置" : formatUsdt(row.maxReserved)} / 占用上限{" "}
        {row.maxOccupied == null ? "未设置" : formatUsdt(row.maxOccupied)}
      </Text>

      <View style={styles.actions}>
        <Pressable
          accessibilityRole="button"
          onPress={onOpenTrades}
          style={({ pressed }) => [styles.action, styles.actionPrimary, pressed && { opacity: 0.78 }]}
        >
          <ReceiptText color={tokens.primary} size={14} />
          <Text style={[styles.actionText, styles.actionTextPrimary]}>交易明细与原因</Text>
          <ChevronRight color={tokens.primary} size={14} />
        </Pressable>
        <Pressable accessibilityRole="button" onPress={onOpenCharts} style={({ pressed }) => [styles.action, pressed && { opacity: 0.78 }]}>
          <LineChart color={tokens.text} size={14} />
          <Text style={styles.actionText}>行情与指标</Text>
        </Pressable>
        <Pressable accessibilityRole="button" onPress={onEditCap} style={({ pressed }) => [styles.action, pressed && { opacity: 0.78 }]}>
          <CircleDollarSign color={tokens.text} size={14} />
          <Text style={styles.actionText}>资金上限</Text>
        </Pressable>
      </View>
    </Pressable>
  );
}

export type ConcurrencyMatrixProps = {
  rows: readonly ConcurrencyMatrixRow[];
  selectedStrategyId: string | null;
  runningCount: number;
  totalCount: number;
  accountUtilizationRatio: number;
  onSelect: (strategyId: string) => void;
  onOpenTrades: (strategyId: string) => void;
  onOpenCharts: (strategyId: string) => void;
  onEditCap: (row: ConcurrencyMatrixRow) => void;
};

export function ConcurrencyMatrix({
  rows,
  selectedStrategyId,
  runningCount,
  totalCount,
  accountUtilizationRatio,
  onSelect,
  onOpenTrades,
  onOpenCharts,
  onEditCap,
}: ConcurrencyMatrixProps) {
  const { tokens } = useTheme();
  const styles = StyleSheet.create({
    summary: { alignItems: "flex-end", gap: 2 },
    summaryValue: { color: tokens.text, fontSize: 14, fontWeight: "900" },
    summaryMeta: { color: tokens.textMuted, fontSize: 11, fontWeight: "700" },
    list: { gap: spacing.sm },
    empty: { color: tokens.textMuted, fontSize: 12, lineHeight: 18, paddingVertical: spacing.md, textAlign: "center" },
  });

  return (
    <View style={styles.list}>
      <View style={styles.summary}>
        <Text style={styles.summaryValue}>{runningCount} / {totalCount} 运行中</Text>
        <Text style={styles.summaryMeta}>账户利用率 {formatRatioPercent(accountUtilizationRatio)}</Text>
      </View>
      {rows.length === 0 ? (
        <Text style={styles.empty}>暂无策略分配记录。</Text>
      ) : (
        rows.map((row) => (
          <ConcurrencyMatrixCard
            key={row.strategyId}
            onEditCap={() => onEditCap(row)}
            onOpenCharts={() => onOpenCharts(row.strategyId)}
            onOpenTrades={() => onOpenTrades(row.strategyId)}
            onSelect={() => onSelect(row.strategyId)}
            row={row}
            selected={row.strategyId === selectedStrategyId}
          />
        ))
      )}
    </View>
  );
}

export type AllocationCapSheetProps = {
  visible: boolean;
  row: ConcurrencyMatrixRow | null;
  credentialId: string;
  tenantId: string | null | undefined;
  onClose: () => void;
};

/**
 * 策略资金上限编辑。Validation mirrors the Web editor: both caps must be
 * finite and non-negative, and the occupied cap may not exceed the reserved cap.
 */
export function AllocationCapSheet({
  visible,
  row,
  credentialId,
  tenantId,
  onClose,
}: AllocationCapSheetProps) {
  const { tokens } = useTheme();
  const queryClient = useQueryClient();
  const [reserved, setReserved] = useState("");
  const [occupied, setOccupied] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!visible) return;
    setReserved(row?.maxReserved == null ? "" : String(row.maxReserved));
    setOccupied(row?.maxOccupied == null ? "" : String(row.maxOccupied));
    setError(null);
    setPending(false);
  }, [row?.maxOccupied, row?.maxReserved, row?.strategyId, visible]);

  const styles = StyleSheet.create({
    overlay: { backgroundColor: "rgba(0, 0, 0, 0.56)", flex: 1, justifyContent: "flex-end" },
    backdrop: { ...StyleSheet.absoluteFill },
    sheet: { backgroundColor: tokens.surface, borderColor: tokens.border, borderTopLeftRadius: radius.lg, borderTopRightRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.lg, paddingBottom: spacing.xl },
    title: { color: tokens.text, fontSize: 19, fontWeight: "800" },
    description: { color: tokens.textMuted, fontSize: 13, lineHeight: 19 },
    field: { gap: spacing.xxs },
    fieldLabel: { color: tokens.textMuted, fontSize: 12, fontWeight: "800" },
    input: { backgroundColor: tokens.surfaceRaised, borderColor: tokens.border, borderRadius: radius.sm, borderWidth: 1, color: tokens.text, fontSize: 16, fontWeight: "800", minHeight: 48, paddingHorizontal: spacing.sm },
    error: { color: tokens.negative, fontSize: 12, lineHeight: 18 },
    actions: { gap: spacing.xs },
    save: { alignItems: "center", backgroundColor: tokens.primarySoft, borderColor: tokens.primary, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48 },
    saveText: { color: tokens.primary, fontSize: 15, fontWeight: "800" },
    cancel: { alignItems: "center", borderColor: tokens.border, borderRadius: radius.sm, borderWidth: 1, justifyContent: "center", minHeight: 48 },
    cancelText: { color: tokens.text, fontSize: 15, fontWeight: "800" },
  });

  const save = async () => {
    if (!row) return;
    const maxReservedQuote = Number(reserved);
    const maxOccupiedQuote = Number(occupied);
    if (
      !Number.isFinite(maxReservedQuote) ||
      !Number.isFinite(maxOccupiedQuote) ||
      maxReservedQuote < 0 ||
      maxOccupiedQuote < 0 ||
      maxOccupiedQuote > maxReservedQuote
    ) {
      setError("请输入有效上限，已占用上限不能大于预留上限。");
      return;
    }
    setPending(true);
    setError(null);
    try {
      await api.updateStrategyAllocationCap(row.strategyId, credentialId, maxReservedQuote, maxOccupiedQuote);
      await queryClient.invalidateQueries({
        queryKey: ["mobile", tenantId, "shared-account-summary"],
      });
      onClose();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "资金上限更新失败。");
    } finally {
      setPending(false);
    }
  };

  return (
    <Modal animationType="slide" onRequestClose={onClose} transparent visible={visible}>
      <View accessibilityViewIsModal style={styles.overlay}>
        <Pressable accessibilityLabel="关闭资金上限编辑" accessibilityRole="button" onPress={onClose} style={styles.backdrop} />
        <ScrollView contentContainerStyle={styles.sheet} keyboardShouldPersistTaps="handled">
          <Text style={styles.title}>调整资金上限</Text>
          <Text style={styles.description}>
            {row ? `${row.name} · ${strategyKindLabel(row.kind)}` : ""}
            {"\n"}上限由共享账户 allocator 按实时余额强制执行，未结算资金不会被其他策略挪用。
          </Text>
          <View style={styles.field}>
            <Text style={styles.fieldLabel}>预留上限（USDT）</Text>
            <TextInput
              accessibilityLabel="最大预留资金"
              keyboardType="decimal-pad"
              onChangeText={setReserved}
              placeholder="预留上限"
              placeholderTextColor={tokens.textMuted}
              style={styles.input}
              value={reserved}
            />
          </View>
          <View style={styles.field}>
            <Text style={styles.fieldLabel}>占用上限（USDT）</Text>
            <TextInput
              accessibilityLabel="最大占用资金"
              keyboardType="decimal-pad"
              onChangeText={setOccupied}
              placeholder="占用上限"
              placeholderTextColor={tokens.textMuted}
              style={styles.input}
              value={occupied}
            />
          </View>
          {error ? <Text style={styles.error}>{error}</Text> : null}
          <View style={styles.actions}>
            <Pressable
              accessibilityRole="button"
              disabled={pending}
              onPress={() => void save()}
              style={({ pressed }) => [styles.save, pending && { opacity: 0.55 }, pressed && !pending && { opacity: 0.78 }]}
            >
              {pending ? <ActivityIndicator color={tokens.primary} /> : <Wallet color={tokens.primary} size={18} />}
              <Text style={styles.saveText}>保存资金上限</Text>
            </Pressable>
            <Pressable
              accessibilityRole="button"
              disabled={pending}
              onPress={onClose}
              style={({ pressed }) => [styles.cancel, pending && { opacity: 0.55 }, pressed && !pending && { backgroundColor: tokens.surfaceRaised }]}
            >
              <Text style={styles.cancelText}>取消</Text>
            </Pressable>
          </View>
        </ScrollView>
      </View>
    </Modal>
  );
}
