import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { useRoute } from "@react-navigation/native";
import Svg, { Line, Path } from "react-native-svg";
import { api } from "../api";
import { BottomSheetSelector, ScreenHeader, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import { formatQuote, formatTimestamp, selectActiveStrategyId } from "./workbench";

type RouteParams = { params?: { strategyId?: string } };
type PnlPoint = { ts: string; cumulative_pnl?: number; equity_quote?: number };
type FundingEntry = { evaluation_id?: string; evaluated_at?: string; funding_rate?: number; direction?: string; current_notional_quote?: number; estimated_payment_quote?: number };

function pathFor(points: PnlPoint[], width: number, height: number): string {
  const values = points.map((point) => point.cumulative_pnl ?? 0);
  const minimum = Math.min(...values, 0);
  const maximum = Math.max(...values, 0);
  const range = Math.max(maximum - minimum, 0.00001);
  return points.map((point, index) => {
    const x = points.length === 1 ? width / 2 : (index / (points.length - 1)) * width;
    const y = height - (((point.cumulative_pnl ?? 0) - minimum) / range) * height;
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
}

export default function FundingPnlScreen() {
  const route = useRoute() as RouteParams;
  const { session } = useSession();
  const [strategyId, setStrategyId] = useState(route.params?.strategyId ?? "");
  const [pickerVisible, setPickerVisible] = useState(false);
  const strategies = useQuery({ queryKey: ["mobile", session?.tenantId, "strategies"], queryFn: () => api.strategies(false), enabled: Boolean(session) });
  const selectedId = useMemo(() => selectActiveStrategyId(strategies.data ?? [], strategyId), [strategies.data, strategyId]);
  const selectedStrategy = strategies.data?.find((item) => item.strategy_id === selectedId);
  const isDemo = selectedStrategy?.config.execution.environment === "okx_demo";
  const pnl = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", selectedId, "pnl"], queryFn: () => api.strategyPnlCurve(selectedId), enabled: Boolean(selectedId && !isDemo) });
  const funding = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", selectedId, "funding", 100], queryFn: () => api.strategyLog(selectedId, "funding", 100), enabled: Boolean(selectedId && !isDemo) });
  const demo = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", selectedId, "demo-execution"], queryFn: () => api.strategyDemoExecution(selectedId), enabled: Boolean(selectedId && isDemo), retry: false });
  useEffect(() => { if (selectedId !== strategyId) setStrategyId(selectedId); }, [selectedId, strategyId]);
  const points = pnl.data as PnlPoint[] | undefined;
  const entries = ((funding.data as { entries?: FundingEntry[] } | undefined)?.entries ?? []);
  const line = points?.length ? pathFor(points, 320, 120) : "";

  if (strategies.isLoading) return <StatePanel description="正在加载可用策略。" title="资金费与 PnL" />;
  if (!selectedId) return <StatePanel description="创建策略后可查看服务端 PnL 与资金费历史。" title="暂无策略" />;
  if (isDemo) return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={() => void demo.refetch()} refreshing={demo.isRefetching} tintColor={palette.primary} />} style={styles.page}><ScreenHeader actionLabel="切换策略" onAction={() => setPickerVisible(true)} subtitle="OKX Demo 使用交易所权威执行数据，不会回退到纸面账本。" title="资金费与 PnL" />{demo.isError ? <StatePanel error={demo.error as Error} onRetry={() => void demo.refetch()} state="error" title="Demo 执行数据暂不可用" /> : <StatePanel message={demo.data?.pnl.reason ?? "交易所未提供 PnL 数据。"} state="empty" title="PnL 暂不可用" />}<View style={styles.chartCard}><Text style={styles.cardTitle}>交易所执行状态</Text><Text style={styles.caption}>来源 {demo.data?.source ?? "OKX Demo"} · 核验 {formatTimestamp(demo.data?.checked_at)} · 关联订单 {demo.data?.orders.length ?? 0}</Text></View><BottomSheetSelector onClose={() => setPickerVisible(false)} onSelect={(id) => { setStrategyId(id); setPickerVisible(false); }} options={(strategies.data ?? []).map((item) => ({ label: item.name, value: item.strategy_id }))} selectedValue={selectedId} title="选择策略" visible={pickerVisible} /></ScrollView>;
  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={() => void Promise.all([pnl.refetch(), funding.refetch()])} refreshing={pnl.isRefetching || funding.isRefetching} tintColor={palette.primary} />} style={styles.page}>
    <ScreenHeader actionLabel="切换策略" onAction={() => setPickerVisible(true)} subtitle="服务端已评估曲线与资金费影响" title="资金费与 PnL" />
    {pnl.isError || funding.isError ? <StatePanel actionLabel="重试" description={((pnl.error ?? funding.error) as Error).message} onAction={() => void Promise.all([pnl.refetch(), funding.refetch()])} title="历史数据暂不可用" tone="error" /> : null}
    <View style={styles.chartCard}><Text style={styles.cardTitle}>累计 PnL</Text>{line ? <Svg height={140} width="100%" viewBox="0 0 320 140"><Line stroke={palette.border} strokeWidth={1} x1={0} x2={320} y1={70} y2={70} /><Path d={line} fill="none" stroke={palette.primary} strokeWidth={3} /></Svg> : <Text style={styles.empty}>尚无服务端 PnL 点。完成评估后将显示真实曲线。</Text>}<Text style={styles.caption}>{points?.length ?? 0} 个曲线点</Text></View>
    <Text style={styles.section}>资金费记录</Text>
    {entries.length === 0 ? <StatePanel description="服务端尚未记录资金费影响。" title="暂无资金费" /> : entries.map((entry, index) => <View key={`${entry.evaluation_id ?? "funding"}-${index}`} style={styles.fundingCard}><View style={styles.fundingTop}><Text style={styles.direction}>{entry.direction ?? "none"}</Text><Text style={styles.time}>{formatTimestamp(entry.evaluated_at)}</Text></View><Text style={styles.amount}>{formatQuote(entry.estimated_payment_quote)}</Text><Text style={styles.caption}>费率 {entry.funding_rate ?? "—"} · 名义金额 {formatQuote(entry.current_notional_quote)}</Text></View>)}
    <BottomSheetSelector onClose={() => setPickerVisible(false)} onSelect={(id) => { setStrategyId(id); setPickerVisible(false); }} options={(strategies.data ?? []).map((item) => ({ label: item.name, value: item.strategy_id }))} selectedValue={selectedId} title="选择策略" visible={pickerVisible} />
  </ScrollView>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, chartCard: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, padding: spacing.md }, cardTitle: { color: palette.text, fontSize: 16, fontWeight: "800" }, section: { color: palette.text, fontSize: 17, fontWeight: "800" }, empty: { color: palette.textMuted, fontSize: 13, paddingVertical: spacing.lg }, caption: { color: palette.textMuted, fontSize: 12, lineHeight: 19 }, fundingCard: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.xs, padding: spacing.md }, fundingTop: { flexDirection: "row", justifyContent: "space-between" }, direction: { color: palette.primary, fontSize: 13, fontWeight: "800" }, time: { color: palette.textMuted, fontSize: 12 }, amount: { color: palette.text, fontSize: 20, fontWeight: "800" },
});
