import { useQuery } from "@tanstack/react-query";
import { RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { Activity, ChartNoAxesCombined, CircleDollarSign, ShieldCheck } from "lucide-react-native";
import { api } from "../api";
import { palette, radius, spacing } from "../theme";

function formatUsd(value: number) {
  return `${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USDT`;
}

function Metric({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "neutral" | "positive" | "warning" }) {
  const color = tone === "positive" ? palette.positive : tone === "warning" ? palette.warning : palette.text;
  return <View style={styles.metric}><Text style={styles.metricLabel}>{label}</Text><Text style={[styles.metricValue, { color }]}>{value}</Text></View>;
}

export default function HomeScreen() {
  const access = useQuery({ queryKey: ["mobile", "access"], queryFn: api.access });
  const strategies = useQuery({ queryKey: ["mobile", "strategies"], queryFn: api.strategies });
  const active = strategies.data?.find((strategy) => strategy.status === "running") ?? strategies.data?.[0];
  const refresh = () => void Promise.all([access.refetch(), strategies.refetch()]);
  const dataError = access.error ?? strategies.error;

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl onRefresh={refresh} refreshing={access.isRefetching || strategies.isRefetching} tintColor={palette.primary} />}
      style={styles.page}
    >
      <View style={styles.header}>
        <View><Text style={styles.eyebrow}>VALUE CELL · MOBILE WORKSPACE</Text><Text style={styles.title}>市场指挥中心</Text><Text style={styles.subtitle}>{access.data?.organization_name ?? "量化策略工作区"}</Text></View>
        <View style={[styles.status, { backgroundColor: access.data?.status === "active" ? palette.positiveSoft : palette.warningSoft }]}><ShieldCheck color={access.data?.status === "active" ? palette.positive : palette.warning} size={18} /><Text style={[styles.statusText, { color: access.data?.status === "active" ? palette.positive : palette.warning }]}>{access.data?.status === "active" ? "已开通" : "待开通"}</Text></View>
      </View>
      {access.isLoading || strategies.isLoading ? <Text style={styles.loading}>正在同步工作区数据…</Text> : null}
      {dataError ? <Text style={styles.error}>{dataError instanceof Error ? dataError.message : "工作区数据同步失败。下拉刷新后重试。"}</Text> : null}
      <View style={styles.metricGrid}>
        <Metric label="当前权益" value={active ? formatUsd(active.account.equity_quote) : "—"} />
        <Metric label="可用资金" value={active ? formatUsd(active.account.quote_balance) : "—"} />
        <Metric label="运行策略" value={`${strategies.data?.filter((item) => item.status === "running").length ?? 0} 个`} tone="positive" />
        <Metric label="观察币种" value={`${active?.config.symbols.length ?? 0} 个`} tone="warning" />
      </View>
      <View style={styles.card}>
        <View style={styles.cardHeader}><View style={styles.iconBox}><Activity color={palette.primary} size={18} /></View><View><Text style={styles.cardTitle}>当前策略</Text><Text style={styles.cardCopy}>{active?.name ?? "尚未创建策略"}</Text></View></View>
        {active ? <><View style={styles.rule}/><View style={styles.row}><Text style={styles.rowLabel}>执行环境</Text><Text style={styles.rowValue}>{active.config.execution.environment === "okx_demo" ? "OKX Demo" : "纸面交易"}</Text></View><View style={styles.row}><Text style={styles.rowLabel}>单笔开仓</Text><Text style={styles.rowValue}>{formatUsd(active.config.risk.order_quote_amount)}</Text></View><View style={styles.row}><Text style={styles.rowLabel}>状态</Text><Text style={[styles.rowValue, { color: active.status === "running" ? palette.positive : palette.textMuted }]}>{active.status === "running" ? "正在运行" : "已停止"}</Text></View></> : <Text style={styles.emptyCopy}>在“策略”页创建第一条策略后，这里会显示执行状态和资金概览。</Text>}
      </View>
      <View style={styles.notice}><CircleDollarSign color={palette.primary} size={19} /><View style={styles.noticeCopy}><Text style={styles.noticeTitle}>移动端安全边界</Text><Text style={styles.noticeText}>纸面和 OKX Demo 可从策略页安全操作；实盘仍需要桌面端的独立风控、绑定与人工授权。</Text></View></View>
      <View style={styles.card}><View style={styles.cardHeader}><View style={styles.iconBox}><ChartNoAxesCombined color={palette.primary} size={18} /></View><View><Text style={styles.cardTitle}>行情查看</Text><Text style={styles.cardCopy}>行情页支持策略观察币种逐个切换、K 线周期和均线。</Text></View></View></View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, header: { gap: spacing.sm, paddingTop: spacing.sm }, eyebrow: { color: palette.primary, fontSize: 10, fontWeight: "800", letterSpacing: 1 }, title: { color: palette.text, fontSize: 28, fontWeight: "800", letterSpacing: -0.8, marginTop: 2 }, subtitle: { color: palette.textMuted, fontSize: 13, marginTop: 3 }, status: { alignItems: "center", alignSelf: "flex-start", borderRadius: radius.pill, flexDirection: "row", gap: 6, paddingHorizontal: 10, paddingVertical: 6 }, statusText: { fontSize: 12, fontWeight: "800" }, metricGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm }, metric: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexGrow: 1, minWidth: "46%", padding: spacing.sm }, metricLabel: { color: palette.textMuted, fontSize: 11 }, metricValue: { fontSize: 17, fontWeight: "800", marginTop: 7 }, card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, padding: spacing.md }, cardHeader: { alignItems: "center", flexDirection: "row", gap: spacing.sm }, iconBox: { alignItems: "center", backgroundColor: palette.primarySoft, borderRadius: radius.sm, height: 34, justifyContent: "center", width: 34 }, cardTitle: { color: palette.text, fontSize: 15, fontWeight: "800" }, cardCopy: { color: palette.textMuted, fontSize: 12, marginTop: 2 }, rule: { backgroundColor: palette.border, height: 1, marginVertical: spacing.md }, row: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 5 }, rowLabel: { color: palette.textMuted, fontSize: 13 }, rowValue: { color: palette.text, fontSize: 13, fontWeight: "700" }, emptyCopy: { color: palette.textMuted, fontSize: 13, lineHeight: 20, marginTop: spacing.md }, notice: { backgroundColor: palette.primarySoft, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, padding: spacing.md }, noticeCopy: { flex: 1 }, noticeTitle: { color: palette.text, fontSize: 13, fontWeight: "800" }, noticeText: { color: palette.textMuted, fontSize: 12, lineHeight: 18, marginTop: 4 }, loading: { color: palette.textMuted, fontSize: 13 }, error: { color: palette.negative, fontSize: 13 },
});
