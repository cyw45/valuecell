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
import { useNavigation, useRoute, type RouteProp } from "@react-navigation/native";
import { ChevronRight, LineChart, RefreshCw } from "lucide-react-native";
import { api } from "../api";
import { StrategyExportPanel } from "../components/StrategyExportPanel";
import {
  BottomSheetSelector,
  EquityCurveChart,
  MetricCard,
  PrimaryButton,
  SectionCard,
  StatePanel,
  StrategyEvaluationPanel,
} from "../components";
import type { WorkbenchStackParamList } from "../navigation/types";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import {
  conditionStateLabel,
  conditionStateSummary,
  conditionStateTone,
  demoSourceLabel,
  demoPnlReason,
  evaluationReason,
  executionEnvironmentLabel,
  fundingDirectionLabel,
  orderSideLabel,
  orderStatusLabel,
  orderTypeLabel,
  paperPositionValue,
  primaryConditionState,
  strategyActionLabel,
  strategyStatusLabel,
} from "./strategy-presentation";
import {
  formatQuote,
  formatTimestamp,
  readActiveStrategyId,
  saveActiveStrategyId,
  selectActiveStrategyId,
} from "./workbench";


type StrategyOverviewRoute = RouteProp<WorkbenchStackParamList, "StrategyOverview">;


function demoPositionValue(
  positions: Array<{ notional_usdt: number | null }>,
): number | undefined {
  if (positions.length === 0) return 0;
  const valuedPositions = positions.filter(
    (position): position is { notional_usdt: number } =>
      typeof position.notional_usdt === "number" && Number.isFinite(position.notional_usdt),
  );
  if (valuedPositions.length === 0) return undefined;
  return valuedPositions.reduce((total, position) => total + position.notional_usdt, 0);
}

export default function StrategyOverviewScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<StrategyOverviewRoute>();
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
    let active = true;
    void readActiveStrategyId(session.userId, session.tenantId).then((strategyId) => {
      if (active) setSelectedId(strategyId);
    });
    return () => {
      active = false;
    };
  }, [session?.tenantId, session?.userId]);

  useEffect(() => {
    const requestedStrategyId = route.params?.strategyId;
    if (!requestedStrategyId || requestedStrategyId === selectedId) return;
    setSelectedId(requestedStrategyId);
    if (session) {
      void saveActiveStrategyId(session.userId, session.tenantId, requestedStrategyId);
    }
  }, [route.params?.strategyId, selectedId, session]);

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
    enabled: Boolean(activeId),
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
    if (isDemo) {
      void Promise.all([strategies.refetch(), evaluations.refetch(), demo.refetch()]);
      return;
    }
    void Promise.all([
      strategies.refetch(),
      account.refetch(),
      pnl.refetch(),
      evaluations.refetch(),
      trades.refetch(),
      funding.refetch(),
    ]);
  };
  const selectStrategy = (strategyId: string) => {
    setSelectedId(strategyId);
    setSelectorVisible(false);
    if (session) void saveActiveStrategyId(session.userId, session.tenantId, strategyId);
  };

  if (strategies.isLoading) {
    return <StatePanel description="正在读取当前工作区的策略、账户与执行状态。" title="正在同步策略工作台" />;
  }
  if (strategies.isError) {
    return <StatePanel actionLabel="重试" description={(strategies.error as Error).message} onAction={refresh} title="策略工作台暂不可用" tone="error" />;
  }
  if (!strategy) {
    return <StatePanel actionLabel="创建策略" description="创建第一条策略后，服务端账户、仓位、评估条件与执行记录会在这里汇总。" onAction={() => navigation.navigate("策略", { screen: "StrategyEditor" })} title="尚未创建策略" />;
  }

  const paperAccount = account.data;
  const demoData = demo.data;
  const paperPositions = Object.entries(paperAccount?.positions ?? {});
  const demoPositions = demoData?.positions.data.positions ?? [];
  const demoBalances = demoData?.account.data.balances ?? [];
  const demoOrders = demoData?.orders ?? [];
  const positionValue = isDemo
    ? demoData
      ? demoPositionValue(demoPositions)
      : undefined
    : paperAccount
      ? paperPositionValue(paperAccount.positions)
      : undefined;
  const latestEvaluation = evaluations.data?.[0];
  const activeConditionState = latestEvaluation
    ? primaryConditionState(latestEvaluation.conditions)
    : null;
  const runningCount = (strategies.data ?? []).filter(
    (item) => item.status === "running" && !item.archived_at,
  ).length;

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          onRefresh={refresh}
          refreshing={
            strategies.isRefetching ||
            account.isRefetching ||
            pnl.isRefetching ||
            evaluations.isRefetching ||
            demo.isRefetching
          }
          tintColor={palette.primary}
        />
      }
      style={styles.page}
    >
      <View style={styles.heading}>
        <View style={styles.headingCopy}>
          <Text style={styles.eyebrow}>移动策略经纪工作台</Text>
          <Text style={styles.title}>策略工作台</Text>
          <Text style={styles.subtitle}>账户、条件与执行事实均以服务端数据为准</Text>
        </View>
        <Pressable
          accessibilityLabel="切换活跃策略"
          accessibilityRole="button"
          onPress={() => setSelectorVisible(true)}
          style={({ pressed }) => [styles.selector, pressed && styles.pressed]}
        >
          <View style={styles.selectorCopy}>
            <Text numberOfLines={1} style={styles.selectorText}>{strategy.name}</Text>
            <Text numberOfLines={1} style={styles.selectorMeta}>
              {strategyStatusLabel(strategy.status, strategy.archived_at)} · {executionEnvironmentLabel(strategy.config.execution.environment)}
            </Text>
          </View>
          <ChevronRight color={palette.primary} size={20} />
        </Pressable>
      </View>

      <View style={styles.metricGrid}>
        <MetricCard caption={isDemo ? "OKX Demo 交易所仓位名义价值" : "纸面仓位按服务端标记价格计算"} label="持仓价值" style={styles.metric} tone="default" value={formatQuote(positionValue)} />
        <MetricCard caption="当前工作区未归档的运行策略" label="运行策略" style={styles.metric} tone={runningCount > 0 ? "positive" : "default"} value={`${runningCount} 个`} />
        <MetricCard caption={strategy.config.symbols.join(" · ") || "未配置观察标的"} label="观察币种" style={styles.metric} tone="warning" value={`${strategy.config.symbols.length} 个`} />
        <MetricCard
          caption={latestEvaluation ? latestEvaluation.conditions.length ? conditionStateSummary(latestEvaluation.conditions) : "本次评估未返回条件记录" : evaluations.isLoading ? "正在读取最近评估" : "服务端尚无评估记录"}
          label="活动条件状态"
          style={styles.metric}
          tone={activeConditionState ? conditionStateTone(activeConditionState) : "default"}
          value={activeConditionState ? conditionStateLabel(activeConditionState) : latestEvaluation ? "未返回条件" : evaluations.isLoading ? "同步中" : "尚无评估"}
        />
      </View>

      <SectionCard description="点按交易对可直接打开该策略对应的行情视图。" title="观察标的">
        <View style={styles.symbols}>
          {strategy.config.symbols.map((symbol) => (
            <Pressable
              accessibilityLabel={`查看 ${symbol} 行情`}
              accessibilityRole="button"
              key={symbol}
              onPress={() => navigation.navigate("行情", { screen: "Market", params: { strategyId: activeId, symbol } })}
              style={({ pressed }) => [styles.symbol, pressed && styles.pressed]}
            >
              <Text style={styles.symbolText}>{symbol}</Text>
              <ChevronRight color={palette.primary} size={16} />
            </Pressable>
          ))}
        </View>
      </SectionCard>

      {isDemo ? (
        <>
          {demo.isLoading ? <View style={styles.loading}><ActivityIndicator color={palette.primary} /><Text style={styles.loadingText}>正在读取 OKX Demo 交易所执行数据。</Text></View> : null}
          {demo.isError ? <StatePanel actionLabel="重试" description={(demo.error as Error).message} onAction={() => void demo.refetch()} title="Demo 执行数据暂不可用" tone="error" /> : null}
          {demoData ? (
            <>
              <SectionCard description={`来源：${demoSourceLabel(demoData.source)} · 最近核验 ${formatTimestamp(demoData.checked_at)}`} title="OKX Demo 执行账户">
                <View style={styles.dataRows}>
                  <Text style={styles.row}>交易所余额条目：{demoBalances.length} 项</Text>
                  <Text style={styles.row}>交易所持仓：{demoPositions.length} 项</Text>
                  <Text style={styles.row}>关联订单：{demoOrders.length} 笔</Text>
                </View>
              </SectionCard>
              <SectionCard title="交易所仓位">
                {demoPositions.length ? demoPositions.map((position) => (
                  <View key={position.symbol} style={styles.positionRow}>
                    <View style={styles.rowCopy}><Text style={styles.positionSymbol}>{position.symbol}</Text><Text style={styles.muted}>数量 {position.quantity} · 可用 {position.available_quantity}</Text></View>
                    <Text style={styles.positionValue}>{formatQuote(position.notional_usdt)}</Text>
                  </View>
                )) : <Text style={styles.muted}>交易所当前没有返回持仓。</Text>}
              </SectionCard>
              <SectionCard title="交易所余额">
                {demoBalances.length ? demoBalances.slice(0, 6).map((balance) => (
                  <View key={balance.currency} style={styles.positionRow}>
                    <View style={styles.rowCopy}><Text style={styles.positionSymbol}>{balance.currency}</Text><Text style={styles.muted}>可用 {balance.free} · 总计 {balance.total}</Text></View>
                    <Text style={styles.positionValue}>{formatQuote(balance.usdt_value)}</Text>
                  </View>
                )) : <Text style={styles.muted}>交易所未返回余额条目。</Text>}
              </SectionCard>
              <SectionCard title="关联订单">
                {demoOrders.length ? demoOrders.slice(0, 5).map((order) => (
                  <View key={order.id} style={styles.orderRow}>
                    <View style={styles.rowCopy}><Text style={styles.positionSymbol}>{orderSideLabel(order.side)} · {order.symbol}</Text><Text style={styles.muted}>{orderTypeLabel(order.type)} · {formatTimestamp(order.updated_at)}</Text></View>
                    <Text style={styles.orderStatus}>{orderStatusLabel(order.status)}</Text>
                  </View>
                )) : <Text style={styles.muted}>当前没有归因到该策略的交易所订单。</Text>}
              </SectionCard>
              <StatePanel description={demoPnlReason(demoData.pnl.reason)} title="PnL 暂不可用" />
            </>
          ) : null}
          {evaluations.isError ? <StatePanel actionLabel="重试" description={(evaluations.error as Error).message} onAction={() => void evaluations.refetch()} title="最近评估暂不可用" tone="error" /> : null}
          <SectionCard description={latestEvaluation ? `最近评估：${formatTimestamp(latestEvaluation.evaluated_at)}` : "条件记录来自服务端评估，不会用交易所账户数据推断。"} title="最新策略评估">
            {latestEvaluation ? <StrategyEvaluationPanel evaluation={latestEvaluation} /> : evaluations.isLoading ? <View style={styles.loading}><ActivityIndicator color={palette.primary} /><Text style={styles.loadingText}>正在读取最近评估。</Text></View> : <Text style={styles.muted}>服务端尚无评估记录，无法假定任何条件或执行结果。</Text>}
          </SectionCard>
        </>
      ) : (
        <>
          {account.isError ? <StatePanel actionLabel="重试" description={(account.error as Error).message} onAction={() => void account.refetch()} title="纸面账户暂不可用" tone="error" /> : null}
          <SectionCard description="纸面账户以服务端账本为准，不会混入交易所 Demo 数据。" title="账户与仓位">
            {paperAccount ? (
              <>
                <View style={styles.accountGrid}>
                  <View style={styles.accountMetric}><Text style={styles.accountLabel}>当前权益</Text><Text style={styles.accountValue}>{formatQuote(paperAccount.equity_quote)}</Text></View>
                  <View style={styles.accountMetric}><Text style={styles.accountLabel}>可用资金</Text><Text style={styles.accountValue}>{formatQuote(paperAccount.quote_balance)}</Text></View>
                  <View style={styles.accountMetric}><Text style={styles.accountLabel}>已实现 PnL</Text><Text style={styles.accountValue}>{formatQuote(paperAccount.realized_pnl_quote)}</Text></View>
                  <View style={styles.accountMetric}><Text style={styles.accountLabel}>未实现 PnL</Text><Text style={styles.accountValue}>{formatQuote(paperAccount.unrealized_pnl_quote)}</Text></View>
                </View>
                <View style={styles.rule} />
                <Text style={styles.row}>持仓数 {paperPositions.length} / {strategy.config.risk.max_positions} · 单笔上限 {formatQuote(strategy.config.risk.order_quote_amount)} · 杠杆 {strategy.config.risk.leverage}×</Text>
                {paperPositions.length ? paperPositions.map(([symbol, position]) => (
                  <View key={symbol} style={styles.positionRow}>
                    <View style={styles.rowCopy}><Text style={styles.positionSymbol}>{symbol}</Text><Text style={styles.muted}>数量 {position.quantity} · 标记价 {position.mark_price}</Text></View>
                    <Text style={styles.positionValue}>{formatQuote(position.quantity * position.mark_price)}</Text>
                  </View>
                )) : <Text style={styles.muted}>纸面账本当前没有持仓。</Text>}
              </>
            ) : <Text style={styles.muted}>正在等待服务端纸面账户数据。</Text>}
          </SectionCard>

          {evaluations.isError ? <StatePanel actionLabel="重试" description={(evaluations.error as Error).message} onAction={() => void evaluations.refetch()} title="最近评估暂不可用" tone="error" /> : null}
          <SectionCard description={latestEvaluation ? `最近评估：${formatTimestamp(latestEvaluation.evaluated_at)}` : "评估完成后，这里会按服务端返回顺序展示全部条件状态。"} title="最新策略评估">
            {latestEvaluation ? <StrategyEvaluationPanel evaluation={latestEvaluation} /> : evaluations.isLoading ? <View style={styles.loading}><ActivityIndicator color={palette.primary} /><Text style={styles.loadingText}>正在读取最近评估。</Text></View> : <Text style={styles.muted}>服务端尚无评估记录，无法假定任何条件或执行结果。</Text>}
          </SectionCard>

          <SectionCard
            actionLabel="查看 PnL"
            description="权益、初始本金与累计 PnL 均来自服务端账本；单指横向拖动回看，双指横向开合缩放。"
            onAction={() => navigation.navigate("FundingPnl", { strategyId: activeId })}
            title="策略权益曲线"
          >
            {pnl.isError ? <Text style={styles.muted}>{(pnl.error as Error).message}</Text> : pnl.isLoading ? <View style={styles.loading}><ActivityIndicator color={palette.primary} /><Text style={styles.loadingText}>正在读取服务端权益曲线。</Text></View> : <EquityCurveChart formatQuote={formatQuote} formatTimestamp={formatTimestamp} points={pnl.data ?? []} />}
          </SectionCard>

          <SectionCard actionLabel="全部成交" onAction={() => navigation.navigate("TradeLedger", { strategyId: activeId })} title="最近成交">
            {trades.isError ? <Text style={styles.muted}>{(trades.error as Error).message}</Text> : trades.data?.entries.length ? trades.data.entries.slice(0, 3).map((trade) => <View key={trade.evaluation_id} style={styles.orderRow}><View style={styles.rowCopy}><Text style={styles.positionSymbol}>{strategyActionLabel(trade.action)} · {trade.symbol}</Text><Text style={styles.muted}>{evaluationReason(trade.reason_code, trade.reason)}</Text></View><Text style={styles.positionValue}>{formatQuote(trade.quote_amount)}</Text></View>) : <Text style={styles.muted}>服务端尚无归因成交。</Text>}
          </SectionCard>

          <SectionCard actionLabel="资金费与 PnL" onAction={() => navigation.navigate("FundingPnl", { strategyId: activeId })} title="资金费影响">
            {funding.isError ? <Text style={styles.muted}>{(funding.error as Error).message}</Text> : funding.data?.entries.length ? funding.data.entries.slice(0, 3).map((entry) => <View key={entry.evaluation_id} style={styles.orderRow}><View style={styles.rowCopy}><Text style={styles.positionSymbol}>{fundingDirectionLabel(entry.direction)}</Text><Text style={styles.muted}>名义金额 {formatQuote(entry.current_notional_quote)}</Text></View><Text style={styles.positionValue}>{formatQuote(entry.estimated_payment_quote)}</Text></View>) : <Text style={styles.muted}>服务端尚未记录资金费影响。</Text>}
          </SectionCard>
        </>
      )}

      <StrategyExportPanel strategyId={activeId} />

      <PrimaryButton label="查看策略详情" leading={<LineChart color={palette.canvas} size={19} />} onPress={() => navigation.navigate("策略", { screen: "StrategyDetail", params: { strategyId: activeId } })} />
      <Pressable accessibilityLabel="刷新策略工作台" accessibilityRole="button" onPress={refresh} style={({ pressed }) => [styles.refreshButton, pressed && styles.pressed]}>
        <RefreshCw color={palette.textMuted} size={18} />
        <Text style={styles.refreshText}>刷新服务端数据</Text>
      </Pressable>
      <BottomSheetSelector
        onClose={() => setSelectorVisible(false)}
        onSelect={selectStrategy}
        options={(strategies.data ?? []).map((item) => ({
          description: `${strategyStatusLabel(item.status, item.archived_at)} · ${executionEnvironmentLabel(item.config.execution.environment)} · ${item.config.symbols.join(" · ")}`,
          label: item.name,
          value: item.strategy_id,
        }))}
        selectedValue={activeId}
        title="选择工作台策略"
        visible={selectorVisible}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  heading: { gap: spacing.sm },
  headingCopy: { gap: spacing.xxs },
  eyebrow: { color: palette.primary, fontSize: 11, fontWeight: "900", letterSpacing: 1 },
  title: { color: palette.text, fontSize: 29, fontWeight: "900", letterSpacing: -0.7 },
  subtitle: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  selector: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.primary, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 56, paddingHorizontal: spacing.md },
  selectorCopy: { flex: 1, gap: spacing.xxs },
  selectorText: { color: palette.text, fontSize: 16, fontWeight: "900" },
  selectorMeta: { color: palette.textMuted, fontSize: 12, fontWeight: "700" },
  metricGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  metric: { flexBasis: "47%", minWidth: 148 },
  symbols: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  symbol: { alignItems: "center", backgroundColor: palette.primarySoft, borderColor: palette.primary, borderRadius: radius.pill, borderWidth: 1, flexDirection: "row", gap: spacing.xxs, minHeight: 44, paddingHorizontal: spacing.sm },
  symbolText: { color: palette.primary, fontSize: 13, fontWeight: "900" },
  loading: { alignItems: "center", flexDirection: "row", gap: spacing.sm, minHeight: 48 },
  loadingText: { color: palette.textMuted, flex: 1, fontSize: 14, lineHeight: 20 },
  dataRows: { gap: spacing.xs },
  row: { color: palette.text, fontSize: 14, lineHeight: 22 },
  rowCopy: { flex: 1, gap: 2 },
  muted: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  positionRow: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 52, paddingVertical: spacing.xs },
  positionSymbol: { color: palette.text, fontSize: 14, fontWeight: "800" },
  positionValue: { color: palette.text, fontSize: 13, fontWeight: "900" },
  orderRow: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 54, paddingVertical: spacing.xs },
  orderStatus: { color: palette.warning, fontSize: 12, fontWeight: "900", textAlign: "right" },
  accountGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  accountMetric: { backgroundColor: palette.surfaceRaised, borderRadius: radius.sm, flexBasis: "47%", flexGrow: 1, gap: spacing.xxs, minWidth: 136, padding: spacing.sm },
  accountLabel: { color: palette.textMuted, fontSize: 11, fontWeight: "800" },
  accountValue: { color: palette.text, fontSize: 15, fontWeight: "900" },
  rule: { backgroundColor: palette.border, height: 1 },
  refreshButton: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md },
  refreshText: { color: palette.textMuted, fontSize: 14, fontWeight: "800" },
  pressed: { opacity: 0.76 },
});
