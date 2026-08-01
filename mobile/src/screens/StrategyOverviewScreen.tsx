import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ActivityIndicator,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import Svg, { Path } from "react-native-svg";
import { useNavigation } from "@react-navigation/native";
import { ChevronRight, LineChart, Plus, RefreshCw, Wallet } from "lucide-react-native";
import { api } from "../api";
import { BottomSheetSelector, MetricCard, SectionCard, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import {
  formatQuote,
  formatTimestamp,
  readActiveStrategyId,
  saveActiveStrategyId,
  selectActiveStrategyId,
} from "./workbench";

const TEN_MINUTES = 10 * 60 * 1000;

type LogPayload = { entries: Array<Record<string, unknown>> };

function listEntries(value: unknown): Array<Record<string, unknown>> {
  if (!value || typeof value !== "object" || !("entries" in value)) return [];
  const entries = (value as LogPayload).entries;
  return Array.isArray(entries) ? entries : [];
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
type PnlPoint = { cumulative_pnl?: number };

function pnlSparkline(points: PnlPoint[]): string {
  if (points.length < 2) return "";
  const values = points.map((point) => point.cumulative_pnl ?? 0);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const range = Math.max(maximum - minimum, Number.EPSILON);
  return values.map((value, index) => `${index === 0 ? "M" : "L"}${(index / (values.length - 1) * 180).toFixed(1)} ${(48 - ((value - minimum) / range) * 44).toFixed(1)}`).join(" ");
}

export default function StrategyOverviewScreen() {
  const navigation = useNavigation<any>();
  const { session } = useSession();
  const [selectedId, setSelectedId] = useState("");
  const [selectorVisible, setSelectorVisible] = useState(false);
  const strategies = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategies"],
    queryFn: () => api.strategies(false),
    enabled: Boolean(session),
  });

  useEffect(() => {
    if (!session) return;
    void readActiveStrategyId(session.userId, session.tenantId).then(setSelectedId);
  }, [session]);

  const activeId = useMemo(
    () => selectActiveStrategyId(strategies.data ?? [], selectedId),
    [selectedId, strategies.data],
  );
  const strategy = strategies.data?.find((item) => item.strategy_id === activeId);
  const isDemo = strategy?.config.execution.environment === "okx_demo";
  const account = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "account"],
    queryFn: () => api.strategyAccount(activeId),
    enabled: Boolean(activeId && !isDemo),
  });
  const pnl = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "pnl"],
    queryFn: () => api.strategyPnlCurve(activeId),
    enabled: Boolean(activeId && !isDemo),
  });
  const evaluations = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "evaluations", 20],
    queryFn: () => api.strategyEvaluations(activeId, 20),
    enabled: Boolean(activeId && !isDemo),
  });
  const trades = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "trades", 10],
    queryFn: () => api.strategyLog(activeId, "trades", 10),
    enabled: Boolean(activeId && !isDemo),
  });
  const funding = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "funding", 10],
    queryFn: () => api.strategyLog(activeId, "funding", 10),
    enabled: Boolean(activeId && !isDemo),
  });
  const demo = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "demo-execution"],
    queryFn: () => api.strategyDemoExecution(activeId),
    enabled: Boolean(activeId && isDemo),
    retry: false,
  });

  const refresh = () => {
    void Promise.all([
      strategies.refetch(),
      account.refetch(),
      pnl.refetch(),
      evaluations.refetch(),
      trades.refetch(),
      funding.refetch(),
      demo.refetch(),
    ]);
  };
  const selectStrategy = (strategyId: string) => {
    setSelectedId(strategyId);
    setSelectorVisible(false);
    if (session) void saveActiveStrategyId(session.userId, session.tenantId, strategyId);
  };
  const accountData = account.data as Record<string, unknown> | undefined;
  const demoData = demo.data;
  const latestEvaluation = (evaluations.data ?? [])[0] as unknown as Record<string, unknown> | undefined;
  const tradeRows = listEntries(trades.data);
  const fundingRows = listEntries(funding.data);

  if (strategies.isLoading) {
    return <StatePanel title="正在同步策略" description="正在读取当前工作区的账户与策略状态。" />;
  }
  if (strategies.isError) {
    return <StatePanel actionLabel="重试" description={(strategies.error as Error).message} onAction={refresh} title="策略数据暂不可用" tone="error" />;
  }
  if (!strategy) {
    return <StatePanel actionLabel="创建策略" description="创建第一条策略后，账户、成交、资金费与 PnL 会出现在这里。" onAction={() => navigation.navigate("策略", { screen: "StrategyEditor" })} title="尚未创建策略" />;
  }

  const equity = numberValue(accountData?.equity_quote);
  const quoteBalance = numberValue(accountData?.quote_balance);
  const positions = accountData?.positions;
  const positionCount = positions && typeof positions === "object" ? Object.keys(positions).length : 0;
  const demoBalances = demoData?.account.data.balances ?? [];
  const demoPositions = demoData?.positions.data.positions ?? [];

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl onRefresh={refresh} refreshing={strategies.isRefetching || account.isRefetching || demo.isRefetching} tintColor={palette.primary} />}
      style={styles.page}
    >
      <View style={styles.heading}>
        <View><Text style={styles.eyebrow}>WORKBENCH</Text><Text style={styles.title}>策略工作台</Text></View>
        <Pressable accessibilityLabel="切换活跃策略" accessibilityRole="button" onPress={() => setSelectorVisible(true)} style={styles.selector}><Text numberOfLines={1} style={styles.selectorText}>{strategy.name}</Text><ChevronRight color={palette.primary} size={18} /></Pressable>
      </View>
      <Text style={styles.environment}>{isDemo ? "OKX Demo · 交易所权威数据" : "纸面交易 · 服务端账本"}</Text>
      {isDemo ? (
        <>
          {demo.isError ? <StatePanel actionLabel="重试" description={(demo.error as Error).message} onAction={() => void demo.refetch()} title="Demo 执行数据暂不可用" tone="error" /> : null}
          <View style={styles.grid}>
            <MetricCard label="资产条目" value={`${demoBalances.length}`} />
            <MetricCard label="持仓" value={`${demoPositions.length}`} />
            <MetricCard label="数据源" value={String(demoData?.source ?? "OKX Demo")} />
            <MetricCard label="核验时间" value={formatTimestamp(demoData?.checked_at)} />
          </View>
          {demoData?.pnl.status === "unavailable" ? <StatePanel description={demoData.pnl.reason} title="PnL 暂不可用" /> : null}
          <SectionCard title="交易所仓位">{demoPositions.length ? demoPositions.slice(0, 5).map((item) => <Text key={item.symbol} style={styles.row}>{item.symbol} · {item.quantity}</Text>) : <Text style={styles.muted}>当前没有交易所持仓。</Text>}</SectionCard>
          <SectionCard title="交易所余额">{demoBalances.length ? demoBalances.slice(0, 4).map((item) => <Text key={item.currency} style={styles.row}>{item.currency} · 可用 {item.free} · 总计 {item.total} · USDT {item.usdt_value ?? "—"}</Text>) : <Text style={styles.muted}>交易所未返回余额。</Text>}</SectionCard>
          <SectionCard title="交易所订单">{demoData?.orders.length ? demoData.orders.slice(0, 5).map((item) => <Text key={item.id} style={styles.row}>{item.side.toUpperCase()} · {item.symbol} · {item.status}</Text>) : <Text style={styles.muted}>当前没有关联订单。</Text>}</SectionCard>
        </>
      ) : (
        <>
          <View style={styles.grid}>
            <MetricCard label="当前权益" value={formatQuote(equity)} />
            <MetricCard label="可用资金" value={formatQuote(quoteBalance)} />
            <MetricCard label="已实现 PnL" value={formatQuote(numberValue(accountData?.realized_pnl_quote))} />
            <MetricCard label="未实现 PnL" value={formatQuote(numberValue(accountData?.unrealized_pnl_quote))} />
          </View>
          <SectionCard title="风险与仓位"><Text style={styles.row}>持仓：{positionCount} / {strategy.config.risk.max_positions}</Text><Text style={styles.row}>单笔上限：{formatQuote(strategy.config.risk.order_quote_amount)}</Text><Text style={styles.row}>杠杆：{strategy.config.risk.leverage}×</Text></SectionCard>
          <SectionCard actionLabel="查看 PnL" onAction={() => navigation.navigate("FundingPnl", { strategyId: activeId })} title="PnL 曲线">{pnlSparkline((pnl.data ?? []) as PnlPoint[]) ? <Svg height={52} viewBox="0 0 180 52" width="100%"><Path d={pnlSparkline((pnl.data ?? []) as PnlPoint[])} fill="none" stroke={palette.primary} strokeWidth={2.5} /></Svg> : <Text style={styles.chartText}>尚无已完成的策略评估。</Text>}<Text style={styles.chartText}>{pnl.data?.length ?? 0} 个服务端曲线点</Text></SectionCard>
          <SectionCard actionLabel="全部成交" onAction={() => navigation.navigate("TradeLedger", { strategyId: activeId })} title="最近成交">{tradeRows.length ? tradeRows.slice(0, 3).map((item, index) => <Text key={index} style={styles.row}>{String(item.action ?? "—").toUpperCase()} · {String(item.symbol ?? "—")} · {formatQuote(numberValue(item.quote_amount))}</Text>) : <Text style={styles.muted}>尚无成交记录。</Text>}</SectionCard>
          <SectionCard actionLabel="资金费与 PnL" onAction={() => navigation.navigate("FundingPnl", { strategyId: activeId })} title="资金费影响">{fundingRows.length ? fundingRows.slice(0, 3).map((item, index) => <Text key={index} style={styles.row}>{String(item.direction ?? "—")} · {formatQuote(numberValue(item.estimated_payment_quote))}</Text>) : <Text style={styles.muted}>尚无资金费记录。</Text>}</SectionCard>
          <SectionCard title="最新评估"><Text style={styles.row}>{String(latestEvaluation?.action ?? "暂无评估")} · {String(latestEvaluation?.reason ?? "服务端将在下一次评估后提供原因")}</Text></SectionCard>
        </>
      )}
      <Pressable accessibilityRole="button" onPress={() => navigation.navigate("策略", { screen: "StrategyDetail", params: { strategyId: activeId } })} style={styles.detailButton}><LineChart color={palette.primary} size={19} /><Text style={styles.detailText}>查看策略诊断与配置</Text></Pressable>
      <Pressable accessibilityRole="button" onPress={refresh} style={styles.refreshButton}><RefreshCw color={palette.textMuted} size={18} /><Text style={styles.refreshText}>刷新数据</Text></Pressable>
      <BottomSheetSelector
        onClose={() => setSelectorVisible(false)}
        onSelect={selectStrategy}
        options={(strategies.data ?? []).map((item) => ({ label: `${item.name} · ${item.status === "running" ? "运行中" : "已停止"}`, value: item.strategy_id }))}
        selectedValue={activeId}
        title="选择活跃策略"
        visible={selectorVisible}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, heading: { alignItems: "flex-start", flexDirection: "row", gap: spacing.sm, justifyContent: "space-between", paddingTop: spacing.xs }, eyebrow: { color: palette.primary, fontSize: 10, fontWeight: "800", letterSpacing: 1.2 }, title: { color: palette.text, fontSize: 28, fontWeight: "800" }, selector: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, minHeight: 44, paddingHorizontal: spacing.sm, maxWidth: 170 }, selectorText: { color: palette.text, fontSize: 13, fontWeight: "700", flexShrink: 1 }, environment: { color: palette.textMuted, fontSize: 13, marginTop: -spacing.sm }, grid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm }, row: { color: palette.text, fontSize: 14, lineHeight: 24 }, muted: { color: palette.textMuted, fontSize: 13, lineHeight: 20 }, chartText: { color: palette.textMuted, fontSize: 13 }, detailButton: { alignItems: "center", borderColor: palette.primary, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.sm, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md }, detailText: { color: palette.primary, fontSize: 15, fontWeight: "800" }, refreshButton: { alignItems: "center", flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 44 }, refreshText: { color: palette.textMuted, fontSize: 14 },
});
