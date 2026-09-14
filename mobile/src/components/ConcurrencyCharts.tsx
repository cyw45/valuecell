import type { ReactNode } from "react";
import { StyleSheet, Text, View } from "react-native";
import type { StrategyAllocation } from "../multi-strategy";
import {
  capitalMeterRows,
  formatRatioPercent,
  formatUsdt,
  pnlComparisonRows,
  tradeQualityChart,
  walletBudgetChart,
  winRateTone,
} from "../concurrency";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";

/**
 * Mobile twins of the Web dashboard's four concurrency charts. Every number is
 * the allocator's own persisted fact, so the visuals can differ in density but
 * never in arithmetic or in what counts as "unavailable".
 */

/** Stable segment palette so a strategy keeps its colour across every chart. */
export const ALLOCATION_SEGMENT_COLORS = [
  "#0EA5E9",
  "#8B5CF6",
  "#10B981",
  "#F59E0B",
  "#F43F5E",
  "#06B6D4",
  "#D946EF",
  "#84CC16",
];

export function allocationSegmentColor(index: number): string {
  return ALLOCATION_SEGMENT_COLORS[index % ALLOCATION_SEGMENT_COLORS.length] as string;
}

function winRateToneColor(percent: number, positive: string, warning: string, negative: string) {
  const tone = winRateTone(percent);
  if (tone === "positive") return positive;
  if (tone === "warning") return warning;
  return negative;
}

type ChartCardProps = {
  title: string;
  meta?: string;
  description: string;
  children: ReactNode;
};

function ChartCard({ title, meta, description, children }: ChartCardProps) {
  const { tokens } = useTheme();
  const styles = StyleSheet.create({
    root: {
      backgroundColor: tokens.surface,
      borderColor: tokens.border,
      borderRadius: radius.md,
      borderWidth: 1,
      gap: spacing.xs,
      padding: spacing.sm,
    },
    header: { alignItems: "baseline", flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
    title: { color: tokens.text, flexShrink: 1, fontSize: 14, fontWeight: "800" },
    meta: { color: tokens.textMuted, fontSize: 11, fontWeight: "700" },
    description: { color: tokens.textMuted, fontSize: 11, lineHeight: 16 },
    body: { gap: spacing.sm, marginTop: spacing.xxs },
    empty: { color: tokens.textMuted, fontSize: 12, lineHeight: 18, paddingVertical: spacing.sm, textAlign: "center" },
  });

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <Text style={styles.title}>{title}</Text>
        {meta ? <Text style={styles.meta}>{meta}</Text> : null}
      </View>
      <Text style={styles.description}>{description}</Text>
      <View style={styles.body}>{children}</View>
    </View>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  const { tokens } = useTheme();
  const styles = StyleSheet.create({
    item: { alignItems: "center", flexDirection: "row", gap: spacing.xxs },
    dot: { borderRadius: radius.pill, height: 8, width: 8 },
    label: { color: tokens.textMuted, fontSize: 10, fontWeight: "700" },
  });
  return (
    <View style={styles.item}>
      <View style={[styles.dot, { backgroundColor: color }]} />
      <Text style={styles.label}>{label}</Text>
    </View>
  );
}

export type StrategyCapitalMeterChartProps = {
  allocations: readonly StrategyAllocation[];
  resolveName: (allocation: StrategyAllocation) => string;
};

/** Stacked reserved/occupied bars measured against each strategy's own cap. */
export function StrategyCapitalMeterChart({ allocations, resolveName }: StrategyCapitalMeterChartProps) {
  const { tokens } = useTheme();
  const rows = capitalMeterRows(allocations, resolveName);
  const styles = StyleSheet.create({
    row: { gap: spacing.xxs },
    rowHeader: { alignItems: "baseline", flexDirection: "row", gap: spacing.xs, justifyContent: "space-between" },
    label: { color: tokens.text, flexShrink: 1, fontSize: 12, fontWeight: "700" },
    cap: { color: tokens.textMuted, fontSize: 11 },
    track: { backgroundColor: tokens.surfaceMuted, borderRadius: radius.pill, height: 12, overflow: "hidden", position: "relative" },
    reservedBar: { backgroundColor: "#0EA5E9", bottom: 0, left: 0, position: "absolute", top: 0 },
    occupiedBar: { backgroundColor: "#F59E0B", bottom: 0, position: "absolute", top: 0 },
    legend: { flexDirection: "row", gap: spacing.sm },
  });

  return (
    <ChartCard
      description="每条策略相对自身资金上限的实时占用，资金在共享钱包内竞争但互不越界"
      meta="上限来自 allocator 持久化事实"
      title="策略资金水位"
    >
      <View style={styles.legend}>
        <LegendDot color="#0EA5E9" label="预留" />
        <LegendDot color="#F59E0B" label="占用" />
      </View>
      {rows.map((row) => (
        <View key={row.strategyId} style={styles.row}>
          <View style={styles.rowHeader}>
            <Text numberOfLines={1} style={styles.label}>{row.label}</Text>
            <Text style={styles.cap}>
              上限 {row.cap == null ? "未设置" : formatUsdt(row.cap)}
            </Text>
          </View>
          <View style={styles.track}>
            <View style={[styles.reservedBar, { width: `${row.reservedPercent}%` }]} />
            <View
              style={[
                styles.occupiedBar,
                {
                  left: `${row.reservedPercent}%`,
                  width: `${Math.min(row.occupiedPercent, 100 - row.reservedPercent)}%`,
                },
              ]}
            />
          </View>
        </View>
      ))}
    </ChartCard>
  );
}

export type StrategyPnlComparisonChartProps = {
  allocations: readonly StrategyAllocation[];
  resolveName: (allocation: StrategyAllocation) => string;
};

/** Diverging bars around a zero baseline so winners and losers read at a glance. */
export function StrategyPnlComparisonChart({ allocations, resolveName }: StrategyPnlComparisonChartProps) {
  const { tokens } = useTheme();
  const rows = pnlComparisonRows(allocations, resolveName);
  const styles = StyleSheet.create({
    row: { gap: spacing.xxs },
    rowHeader: { alignItems: "baseline", flexDirection: "row", gap: spacing.xs, justifyContent: "space-between" },
    label: { color: tokens.text, flexShrink: 1, fontSize: 12, fontWeight: "700" },
    amount: { fontSize: 11, fontWeight: "800" },
    track: { backgroundColor: tokens.surfaceMuted, borderRadius: radius.pill, height: 10, overflow: "hidden", position: "relative" },
    baseline: { backgroundColor: tokens.border, bottom: 0, left: "50%", position: "absolute", top: 0, width: 1 },
    positiveBar: { backgroundColor: tokens.positive, bottom: 0, position: "absolute", top: 0 },
    negativeBar: { backgroundColor: tokens.negative, bottom: 0, position: "absolute", top: 0 },
  });

  return (
    <ChartCard
      description="仅来自各策略自己的成交与成本重放，不摊分共享钱包的未归因金额"
      meta="策略归属净 PnL（USDT）"
      title="各策略净收益对比"
    >
      {rows.length === 0 ? (
        <Text style={{ color: tokens.textMuted, fontSize: 12, lineHeight: 18, paddingVertical: spacing.sm, textAlign: "center" }}>
          暂无可归属的已结算 PnL，策略产生完整交易后自动出现。
        </Text>
      ) : (
        rows.map((row) => (
          <View key={row.strategyId} style={styles.row}>
            <View style={styles.rowHeader}>
              <Text numberOfLines={1} style={styles.label}>{row.label}</Text>
              <Text style={[styles.amount, { color: row.positive ? tokens.positive : tokens.negative }]}>
                {row.positive ? "+" : "−"}
                {formatUsdt(Math.abs(row.value))}
                {row.returnRatePercent == null
                  ? ""
                  : ` · ${row.returnRatePercent >= 0 ? "+" : "−"}${Math.abs(row.returnRatePercent).toFixed(2)}%`}
              </Text>
            </View>
            <View style={styles.track}>
              <View style={styles.baseline} />
              <View
                style={[
                  row.positive ? styles.positiveBar : styles.negativeBar,
                  row.positive
                    ? { left: "50%", width: `${row.widthPercent}%` }
                    : { right: "50%", width: `${row.widthPercent}%` },
                ]}
              />
            </View>
          </View>
        ))
      )}
    </ChartCard>
  );
}

export type SharedWalletBudgetChartProps = {
  allocations: readonly StrategyAllocation[];
  resolveName: (allocation: StrategyAllocation) => string;
  walletAvailableQuote: number | null;
  walletEquityQuote: number | null;
};

/**
 * Single stacked budget bar: how much of the shared wallet is promised to each
 * strategy cap versus left unallocated. Reads the same persisted caps the
 * allocator enforces, so the picture cannot drift from execution limits.
 */
export function SharedWalletBudgetChart({
  allocations,
  resolveName,
  walletAvailableQuote,
  walletEquityQuote,
}: SharedWalletBudgetChartProps) {
  const { tokens } = useTheme();
  const chart = walletBudgetChart(allocations, resolveName, walletAvailableQuote, walletEquityQuote);
  const styles = StyleSheet.create({
    track: { backgroundColor: tokens.surfaceMuted, borderRadius: radius.pill, flexDirection: "row", height: 16, overflow: "hidden" },
    row: { alignItems: "baseline", flexDirection: "row", gap: spacing.xs, justifyContent: "space-between" },
    rowLabel: { alignItems: "center", flexDirection: "row", flexShrink: 1, gap: spacing.xs },
    dot: { borderRadius: radius.pill, height: 8, width: 8 },
    label: { color: tokens.text, flexShrink: 1, fontSize: 12, fontWeight: "700" },
    value: { color: tokens.textMuted, fontSize: 11 },
    bufferRow: { borderTopColor: tokens.border, borderTopWidth: 1, paddingTop: spacing.xs },
    footnote: { color: tokens.textMuted, fontSize: 10, lineHeight: 15 },
  });

  return (
    <ChartCard
      description="每条策略的资金上限都取自共享钱包的同一份余额，剩余部分留作未分配缓冲"
      meta={`上限合计 ${formatUsdt(chart.allocated)}`}
      title="共享钱包资金预算"
    >
      {chart.segments.length === 0 ? (
        <Text style={{ color: tokens.textMuted, fontSize: 12, lineHeight: 18, paddingVertical: spacing.sm, textAlign: "center" }}>
          尚未为任何策略设置资金上限，设置后这里会显示预算拆分。
        </Text>
      ) : (
        <>
          <View style={styles.track}>
            {chart.segments.map((segment) => (
              <View
                key={segment.strategyId}
                style={{ backgroundColor: allocationSegmentColor(segment.colorIndex), height: "100%", width: `${segment.widthPercent}%` }}
              />
            ))}
            {chart.unallocated > 0 ? (
              <View
                style={{ backgroundColor: tokens.surfaceMuted, height: "100%", opacity: 0.9, width: `${chart.unallocatedWidthPercent}%` }}
              />
            ) : null}
          </View>
          {chart.segments.map((segment) => (
            <View key={segment.strategyId} style={styles.row}>
              <View style={styles.rowLabel}>
                <View style={[styles.dot, { backgroundColor: allocationSegmentColor(segment.colorIndex) }]} />
                <Text numberOfLines={1} style={styles.label}>{segment.label}</Text>
              </View>
              <Text style={styles.value}>
                {formatUsdt(segment.cap)} · {segment.sharePercent == null ? "—" : `${segment.sharePercent.toFixed(1)}%`}
              </Text>
            </View>
          ))}
          {chart.base == null ? null : (
            <View style={[styles.row, styles.bufferRow]}>
              <Text style={styles.label}>未分配缓冲</Text>
              <Text style={styles.value}>
                {formatUsdt(chart.unallocated)} ·{" "}
                {chart.base > 0 ? `${chart.unallocatedPercent?.toFixed(1) ?? "—"}%` : "—"}
              </Text>
            </View>
          )}
          <Text style={styles.footnote}>
            钱包可用余额 {walletAvailableQuote == null ? "—" : formatUsdt(walletAvailableQuote)}
            {chart.overCommit
              ? "；当前上限合计已超过可用余额，allocator 会按实时余额拒绝超额开仓。"
              : "。"}
          </Text>
        </>
      )}
    </ChartCard>
  );
}

export type StrategyTradeQualityChartProps = {
  allocations: readonly StrategyAllocation[];
  resolveName: (allocation: StrategyAllocation) => string;
};

/** Win rate and turnover per strategy so trade quality is comparable at a glance. */
export function StrategyTradeQualityChart({ allocations, resolveName }: StrategyTradeQualityChartProps) {
  const { tokens } = useTheme();
  const chart = tradeQualityChart(allocations, resolveName);
  const styles = StyleSheet.create({
    row: { gap: spacing.xxs },
    rowHeader: { alignItems: "baseline", flexDirection: "row", gap: spacing.xs, justifyContent: "space-between" },
    label: { color: tokens.text, flexShrink: 1, fontSize: 12, fontWeight: "700" },
    meta: { color: tokens.textMuted, fontSize: 11 },
    track: { backgroundColor: tokens.surfaceMuted, borderRadius: radius.pill, height: 8, overflow: "hidden" },
    fill: { borderRadius: radius.pill, height: "100%" },
    detail: { color: tokens.textMuted, fontSize: 10 },
    empty: { color: tokens.textMuted, fontSize: 12, lineHeight: 18, paddingVertical: spacing.sm, textAlign: "center" },
  });

  return (
    <ChartCard description="胜率只统计已闭合交易，周转率表示资金被真正动用的比例；开仓未平仓前不计入" meta="胜率 · 资金周转率" title="策略交易质量">
      {chart.rows.length === 0 ? (
        <Text style={styles.empty}>
          {chart.pendingFillCount === 0
            ? "全部策略尚无成交，第一笔完整交易结算后这里会显示胜率与周转率。"
            : `当前 ${chart.pendingFillCount} 笔成交尚未形成闭合交易，平仓结算后这里会显示胜率与周转率。`}
        </Text>
      ) : (
        chart.rows.map((row) => (
          <View key={row.strategyId} style={styles.row}>
            <View style={styles.rowHeader}>
              <Text numberOfLines={1} style={styles.label}>{row.label}</Text>
              <Text style={styles.meta}>
                胜率 {row.winPercent == null ? "—" : `${row.winPercent.toFixed(1)}%`} · 周转率 {formatRatioPercent(row.turnoverRatio)}
              </Text>
            </View>
            <View style={styles.track}>
              {row.winPercent == null ? null : (
                <View
                  style={[
                    styles.fill,
                    {
                      backgroundColor: winRateToneColor(
                        row.winPercent,
                        tokens.positive,
                        tokens.warning,
                        tokens.negative,
                      ),
                      width: `${Math.min(Math.max(row.winPercent, 0), 100)}%`,
                    },
                  ]}
                />
              )}
            </View>
            <Text style={styles.detail}>
              成交 {row.fills} 笔 · 完整交易 {row.completed} 次 · 手续费 {formatUsdt(row.fee)}
            </Text>
          </View>
        ))
      )}
    </ChartCard>
  );
}