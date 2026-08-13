import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ActivityIndicator,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { ChevronRight, Download, LineChart, Pause, Pencil, Play, RefreshCw, Trash2 } from "lucide-react-native";
import { useNavigation, useRoute, type RouteProp } from "@react-navigation/native";
import { api } from "../api";
import { accessGate, canMutate } from "../access";
import { StrategyExportPanel } from "../components/StrategyExportPanel";
import {
  ConfirmSheet,
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

const MONITOR_LABELS: Record<string, string> = {
  candidate: "待准入",
  admitted: "已准入",
  held: "持仓保留",
  removed: "已移除",
};
const RISK_LABELS: Record<string, string> = {
  normal: "正常",
  warn: "预警",
  only_reduce: "仅允许减仓",
  blocked: "已阻断",
  halted: "已暂停",
};
const displayMonitorState = (state: string) => MONITOR_LABELS[state] ?? "未知状态";
const displayRiskState = (state?: string | null) => (state ? RISK_LABELS[state] ?? "未知状态" : "同步中");

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

function formatNumericQuote(value: number | string | null | undefined): string {
  const number = typeof value === "string" ? Number(value) : value;
  return typeof number === "number" ? formatQuote(number) : "—";
}

export default function StrategyOverviewScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<StrategyOverviewRoute>();
  const { session } = useSession();
  const [selectedId, setSelectedId] = useState("");
  const [selectorVisible, setSelectorVisible] = useState(false);
  const queryClient = useQueryClient();
  const [demoOrdersPage, setDemoOrdersPage] = useState(1);
  const [equityRange, setEquityRange] = useState<"1d" | "5d" | "1w" | "1m" | "1y" | "all">("1m");
  const [exportStrategyId, setExportStrategyId] = useState<string | null>(null);
  const [pendingOperation, setPendingOperation] = useState<{
    action: "start" | "stop" | "archive";
    strategyId: string;
  } | null>(null);
  const [operationError, setOperationError] = useState<string | null>(null);
  const access = useQuery({
    queryKey: ["mobile", session?.tenantId, "access"],
    queryFn: api.access,
    enabled: Boolean(session),
  });
  const lifecycle = useMutation({
    mutationFn: async ({ action, strategyId }: { action: "start" | "stop" | "archive"; strategyId: string }) => {
      if (action === "start") return api.startStrategy(strategyId);
      if (action === "stop") return api.stopStrategy(strategyId);
      return api.archiveStrategy(strategyId);
    },
    onSuccess: async (saved) => {
      await queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategies"] });
      await queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategy", saved.strategy_id] });
    },
  });
  const strategies = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategies"],
    queryFn: () => api.strategies(false),
    enabled: Boolean(session),
    refetchInterval: 15_000,
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
    refetchInterval: 15_000,
  });
  const pnl = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "pnl"],
    queryFn: () => api.strategyPnlCurve(activeId),
    enabled: Boolean(activeId && !isDemo),
    refetchInterval: 15_000,
  });
  const evaluations = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "evaluations", 20],
    queryFn: () => api.strategyEvaluations(activeId, 20),
    enabled: Boolean(activeId),
    refetchInterval: 15_000,
  });
  const trades = useQuery({
    refetchInterval: 15_000,
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "trades", 10],
    queryFn: () => api.strategyLog(activeId, "trades", 10),
    enabled: Boolean(activeId && !isDemo),
  });
  const funding = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "funding", 10],
    refetchInterval: 15_000,
    queryFn: () => api.strategyLog(activeId, "funding", 10),
    enabled: Boolean(activeId && !isDemo),
  });
  const demo = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "demo-execution", demoOrdersPage],
    queryFn: () => api.strategyDemoExecution(activeId, demoOrdersPage, 10),
    enabled: Boolean(activeId && isDemo),
    retry: false,
    refetchInterval: 15_000,
  });
  const monitorState = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "monitor-state"],
    queryFn: () => api.strategyMonitorState(activeId),
    enabled: Boolean(activeId),
    refetchInterval: 15_000,
  });
  const riskState = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "risk-state"],
    refetchInterval: 15_000,
    queryFn: () => api.strategyRiskState(activeId),
    enabled: Boolean(activeId),
  });

  const refresh = () => {
    if (isDemo) {
      void Promise.all([strategies.refetch(), evaluations.refetch(), demo.refetch(), monitorState.refetch(), riskState.refetch()]);
      return;
    }
    void Promise.all([
      strategies.refetch(),
      account.refetch(),
      pnl.refetch(),
      evaluations.refetch(),
      trades.refetch(),
      funding.refetch(),
      monitorState.refetch(),
      riskState.refetch(),
    ]);
  };
  const selectStrategy = (strategyId: string) => {
    setSelectedId(strategyId);
    setSelectorVisible(false);
    if (session) void saveActiveStrategyId(session.userId, session.tenantId, strategyId);
  };
  const managementGate = accessGate(access.data, "strategy.manage");
  const canManage = canMutate(access.data, "strategy.manage");
  const operationStrategy = pendingOperation
    ? (strategies.data ?? []).find(
        (item) => item.strategy_id === pendingOperation.strategyId,
      )
    : undefined;
  const filteredPnlPoints = useMemo(() => {
    const points = pnl.data ?? [];
    if (equityRange === "all" || points.length < 2) return points;
    const durations = {
      "1d": 24 * 60 * 60 * 1_000,
      "5d": 5 * 24 * 60 * 60 * 1_000,
      "1w": 7 * 24 * 60 * 60 * 1_000,
      "1m": 31 * 24 * 60 * 60 * 1_000,
      "1y": 365 * 24 * 60 * 60 * 1_000,
    } as const;
    const latestTs = Date.parse(points[points.length - 1]?.ts ?? "");
    if (!Number.isFinite(latestTs)) return points;
    const initial = points.find((point) => point.action === "initial");
    const ranged = points.filter(
      (point) => Date.parse(point.ts) >= latestTs - durations[equityRange],
    );
    return initial && ranged[0] !== initial ? [initial, ...ranged] : ranged;
  }, [equityRange, pnl.data]);
  const confirmOperation = async () => {
    if (!pendingOperation || !operationStrategy) return;
    try {
      await lifecycle.mutateAsync(pendingOperation);
      if (pendingOperation.action === "archive" && activeId === operationStrategy.strategy_id) {
        selectStrategy("");
      }
      setPendingOperation(null);
      setOperationError(null);
    } catch (error) {
      setOperationError(error instanceof Error ? error.message : "服务未完成本次策略操作。");
    }
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
          {demoData ? (
            <>
              <SectionCard description={`来源：${demoSourceLabel(demoData.source)} · 最近核验 ${formatTimestamp(demoData.checked_at)}`} title="OKX Demo 执行账户">
                <View style={styles.dataRows}>
                  <Text style={styles.row}>交易所余额条目：{demoBalances.length} 项</Text>
                  <Text style={styles.row}>交易所持仓：{demoPositions.length} 项</Text>
                  <Text style={styles.row}>策略归属订单：{demoData.pagination.total_items} 笔</Text>
                  <Text style={styles.row}>已成交 {demoData.trade_summary?.filled_order_count ?? 0} · 部分成交 {demoData.trade_summary?.partially_filled_order_count ?? 0} · 失败 {demoData.trade_summary?.failed_order_count ?? 0}</Text>
                </View>
              </SectionCard>
              <SectionCard title="OKX Demo 已实现交易表现">
                {demoData.pnl.status === "unavailable" ? <Text style={styles.muted}>{demoPnlReason(demoData.pnl.reason)}</Text> : <View style={styles.accountGrid}>
                  <View style={styles.accountMetric}><Text style={styles.accountLabel}>总 PnL</Text><Text style={styles.accountValue}>{formatNumericQuote(demoData.pnl.total_pnl ?? demoData.pnl.total ?? demoData.pnl.value)}</Text></View>
                  <View style={styles.accountMetric}><Text style={styles.accountLabel}>已实现 PnL</Text><Text style={styles.accountValue}>{formatNumericQuote(demoData.pnl.realized_pnl ?? demoData.pnl.realized)}</Text></View>
                  <View style={styles.accountMetric}><Text style={styles.accountLabel}>未实现 PnL</Text><Text style={styles.accountValue}>{formatNumericQuote(demoData.pnl.unrealized_pnl ?? demoData.pnl.unrealized)}</Text></View>
                  <View style={styles.accountMetric}><Text style={styles.accountLabel}>收益率</Text><Text style={styles.accountValue}>{demoData.pnl.return_pct == null ? "—" : `${demoData.pnl.return_pct.toFixed(2)}%`}</Text></View>
                </View>}
                <Text style={styles.muted}>{demoData.pnl.fees_included === true ? "已含交易所费用。" : demoData.pnl.reason ?? "仅展示交易所返回且服务器可核验的 PnL。"}</Text>
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
              <SectionCard title={`策略归属订单 · 第 ${demoData.pagination.page}/${demoData.pagination.total_pages || 1} 页`}>
                {demoOrders.length ? demoOrders.map((order) => (
                  <View key={order.id} style={styles.orderRow}>
                    <View style={styles.rowCopy}><Text style={styles.positionSymbol}>{orderSideLabel(order.side)} · {order.symbol}</Text><Text style={styles.muted}>{orderTypeLabel(order.type)} · 请求 {formatNumericQuote(order.requested_quote)} · {formatTimestamp(order.updated_at)}</Text>{order.error_code ? <Text style={styles.operationError}>{order.error_code}</Text> : null}</View>
                    <Text style={styles.orderStatus}>{orderStatusLabel(order.status)}</Text>
                  </View>
                )) : <Text style={styles.muted}>当前没有归因到该策略的交易所订单。</Text>}
                {demoData.pagination.total_pages > 1 ? <View style={styles.pagination}><Pressable accessibilityRole="button" disabled={demoOrdersPage <= 1} onPress={() => setDemoOrdersPage((page) => Math.max(1, page - 1))} style={[styles.pageButton, demoOrdersPage <= 1 && styles.disabled]}><Text style={styles.pageButtonText}>上一页</Text></Pressable><Pressable accessibilityRole="button" disabled={demoOrdersPage >= demoData.pagination.total_pages} onPress={() => setDemoOrdersPage((page) => Math.min(demoData.pagination.total_pages, page + 1))} style={[styles.pageButton, demoOrdersPage >= demoData.pagination.total_pages && styles.disabled]}><Text style={styles.pageButtonText}>下一页</Text></Pressable></View> : null}
              </SectionCard>
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
            description="权益、初始本金与累计 PnL 均来自服务端账本；可拖动和双指缩放时间轴。"
            onAction={() => navigation.navigate("FundingPnl", { strategyId: activeId })}
            title="策略权益曲线"
          >
            <ScrollView contentContainerStyle={styles.rangeTabs} horizontal showsHorizontalScrollIndicator={false}>
              {(["1d", "5d", "1w", "1m", "1y", "all"] as const).map((value) => (
                <Pressable
                  accessibilityLabel={`查看${({ "1d": "日", "5d": "5日", "1w": "周", "1m": "月", "1y": "年", all: "全部" } as const)[value]}权益曲线`}
                  accessibilityRole="button"
                  key={value}
                  onPress={() => setEquityRange(value)}
                  style={({ pressed }) => [styles.rangeTab, equityRange === value && styles.rangeTabActive, pressed && styles.pressed]}
                >
                  <Text style={[styles.rangeTabText, equityRange === value && styles.rangeTabTextActive]}>
                    {({ "1d": "日", "5d": "5日", "1w": "周", "1m": "月", "1y": "年", all: "全部" } as const)[value]}
                  </Text>
                </Pressable>
              ))}
            </ScrollView>
            {pnl.isError ? <Text style={styles.muted}>{(pnl.error as Error).message}</Text> : pnl.isLoading ? <View style={styles.loading}><ActivityIndicator color={palette.primary} /><Text style={styles.loadingText}>正在读取服务端权益曲线。</Text></View> : <EquityCurveChart formatQuote={formatQuote} formatTimestamp={formatTimestamp} points={filteredPnlPoints} />}
          </SectionCard>

          <SectionCard actionLabel="全部成交" onAction={() => navigation.navigate("TradeLedger", { strategyId: activeId })} title="最近成交">
            {trades.isError ? <Text style={styles.muted}>{(trades.error as Error).message}</Text> : trades.data?.entries.length ? trades.data.entries.slice(0, 3).map((trade) => <View key={trade.evaluation_id} style={styles.orderRow}><View style={styles.rowCopy}><Text style={styles.positionSymbol}>{strategyActionLabel(trade.action)} · {trade.symbol}</Text><Text style={styles.muted}>{evaluationReason(trade.reason_code, trade.reason)}</Text></View><Text style={styles.positionValue}>{formatQuote(trade.quote_amount)}</Text></View>) : <Text style={styles.muted}>服务端尚无归因成交。</Text>}
          </SectionCard>

          <SectionCard actionLabel="资金费与 PnL" onAction={() => navigation.navigate("FundingPnl", { strategyId: activeId })} title="资金费影响">
            {funding.isError ? <Text style={styles.muted}>{(funding.error as Error).message}</Text> : funding.data?.entries.length ? funding.data.entries.slice(0, 3).map((entry) => <View key={entry.evaluation_id} style={styles.orderRow}><View style={styles.rowCopy}><Text style={styles.positionSymbol}>{fundingDirectionLabel(entry.direction)}</Text><Text style={styles.muted}>名义金额 {formatQuote(entry.current_notional_quote)}</Text></View><Text style={styles.positionValue}>{formatQuote(entry.estimated_payment_quote)}</Text></View>) : <Text style={styles.muted}>服务端尚未记录资金费影响。</Text>}
          </SectionCard>
        </>
      )}

      <SectionCard description="服务端持久化的候选池与账户级阻断原因。" title="监控池与风险">
        <View style={styles.stateRows}>
          <Text style={styles.stateText}>待准入 {monitorState.data?.filter((item) => item.state === "candidate").length ?? 0} · 已准入 {monitorState.data?.filter((item) => item.state === "admitted").length ?? 0} · 持仓保留 {monitorState.data?.filter((item) => item.state === "held").length ?? 0} · 已移除 {monitorState.data?.filter((item) => item.state === "removed").length ?? 0}</Text>
          <Text style={styles.stateText}>风险状态：{displayRiskState(riskState.data?.state)}</Text>
          <Text style={styles.muted}>{riskState.data?.reason_detail?.match(/[\u4e00-\u9fff]/) ? riskState.data.reason_detail : "暂无持久化风险原因"}</Text>
          {monitorState.data?.slice(0, 4).map((item) => <Text key={item.symbol} style={styles.muted}>{item.symbol} · {displayMonitorState(item.state)} · {item.reason_detail?.match(/[\u4e00-\u9fff]/) ? item.reason_detail : "暂无中文说明"}</Text>)}
        </View>
      </SectionCard>
      <SectionCard
        actionLabel="新增策略"
        description="选择策略后，资金、执行、交易、分析、权益和监控信息会同步切换。"
        onAction={() => navigation.navigate("策略", { screen: "StrategyEditor" })}
        title="策略管理"
      >
        {!managementGate.mutationAllowed && access.data ? <Text style={styles.muted}>{managementGate.message ?? "当前角色仅具备策略查看权限。"}</Text> : null}
        {operationError ? <Text style={styles.operationError}>{operationError}</Text> : null}
        <View style={styles.managerCards}>
          {(strategies.data ?? []).map((item) => {
            const selected = item.strategy_id === activeId;
            const running = item.status === "running";
            return (
              <View key={item.strategy_id} style={[styles.managerCard, selected && styles.managerCardSelected]}>
                <Pressable
                  accessibilityLabel={`选择策略 ${item.name}`}
                  accessibilityRole="button"
                  onPress={() => selectStrategy(item.strategy_id)}
                  style={({ pressed }) => [styles.managerSelect, pressed && styles.pressed]}
                >
                  <View style={styles.rowCopy}>
                    <Text numberOfLines={1} style={styles.positionSymbol}>{item.name}</Text>
                    <Text numberOfLines={1} style={styles.muted}>{item.config.symbols.join(" · ")} · {item.config.interval} 周期</Text>
                  </View>
                  <Text style={[styles.managerStatus, running ? styles.managerRunning : styles.managerStopped]}>{strategyStatusLabel(item.status, item.archived_at)}</Text>
                </Pressable>
                <View style={styles.managerActions}>
                  <Pressable accessibilityLabel={`编辑策略 ${item.name}`} accessibilityRole="button" onPress={() => navigation.navigate("策略", { screen: "StrategyEditor", params: { strategyId: item.strategy_id } })} style={({ pressed }) => [styles.managerAction, pressed && styles.pressed]}><Pencil color={palette.primary} size={16} /><Text style={styles.managerActionText}>编辑</Text></Pressable>
                  {canManage ? <Pressable accessibilityLabel={`${running ? "停止" : "启动"}策略 ${item.name}`} accessibilityRole="button" disabled={lifecycle.isPending} onPress={() => setPendingOperation({ action: running ? "stop" : "start", strategyId: item.strategy_id })} style={({ pressed }) => [styles.managerAction, running && styles.managerStopAction, lifecycle.isPending && styles.disabled, pressed && !lifecycle.isPending && styles.pressed]}>{running ? <Pause color={palette.warning} size={16} /> : <Play color={palette.primary} size={16} />}<Text style={styles.managerActionText}>{running ? "停止" : "启动"}</Text></Pressable> : null}
                  <Pressable accessibilityLabel={`导出策略 ${item.name} 历史`} accessibilityRole="button" onPress={() => setExportStrategyId(exportStrategyId === item.strategy_id ? null : item.strategy_id)} style={({ pressed }) => [styles.managerAction, pressed && styles.pressed]}><Download color={palette.primary} size={16} /><Text style={styles.managerActionText}>导出</Text></Pressable>
                  {canManage && !running ? <Pressable accessibilityLabel={`删除策略 ${item.name}`} accessibilityRole="button" onPress={() => setPendingOperation({ action: "archive", strategyId: item.strategy_id })} style={({ pressed }) => [styles.managerAction, styles.managerDeleteAction, pressed && styles.pressed]}><Trash2 color={palette.negative} size={16} /><Text style={styles.managerDeleteText}>删除</Text></Pressable> : null}
                </View>
                {exportStrategyId === item.strategy_id ? <StrategyExportPanel strategyId={item.strategy_id} /> : null}
              </View>
            );
          })}
        </View>
      </SectionCard>

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
      <ConfirmSheet
        confirmLabel={pendingOperation?.action === "archive" ? "确认删除" : pendingOperation?.action === "start" ? "确认启动" : "确认停止"}
        confirming={lifecycle.isPending}
        destructive={pendingOperation?.action === "archive"}
        message={pendingOperation?.action === "archive" ? "已停止策略会从工作台列表移除；如有审计记录，服务端会安全归档。" : pendingOperation?.action === "start" ? "策略会按服务端配置恢复扫描。" : "停止后不会再创建新的策略执行。"}
        onCancel={() => !lifecycle.isPending && setPendingOperation(null)}
        onConfirm={() => void confirmOperation()}
        title={pendingOperation ? `${pendingOperation.action === "archive" ? "删除" : pendingOperation.action === "start" ? "启动" : "停止"}“${operationStrategy?.name ?? "策略"}”？` : "确认策略操作"}
        visible={Boolean(pendingOperation)}
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
  stateRows: { gap: spacing.xs },
  stateText: { color: palette.text, fontSize: 13, fontWeight: "800", lineHeight: 20 },
  row: { color: palette.text, fontSize: 14, lineHeight: 22 },
  rowCopy: { flex: 1, gap: 2 },
  muted: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  positionRow: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 52, paddingVertical: spacing.xs },
  rangeTabs: { gap: spacing.xs, paddingBottom: spacing.xs },
  rangeTab: { borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, minHeight: 34, paddingHorizontal: spacing.sm, justifyContent: "center" },
  rangeTabActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  rangeTabText: { color: palette.textMuted, fontSize: 12, fontWeight: "800" },
  rangeTabTextActive: { color: palette.primary },
  managerCards: { gap: spacing.sm },
  managerCard: { backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, gap: spacing.sm, padding: spacing.sm },
  managerCardSelected: { borderColor: palette.primary, backgroundColor: palette.primarySoft },
  managerSelect: { alignItems: "center", flexDirection: "row", gap: spacing.sm },
  managerStatus: { fontSize: 12, fontWeight: "900" },
  managerRunning: { color: palette.positive },
  managerStopped: { color: palette.textMuted },
  managerActions: { borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", flexWrap: "wrap", gap: spacing.xs, paddingTop: spacing.sm },
  managerAction: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: 4, minHeight: 36, paddingHorizontal: spacing.sm },
  managerStopAction: { borderColor: palette.warning },
  managerDeleteAction: { borderColor: palette.negative },
  managerActionText: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  managerDeleteText: { color: palette.negative, fontSize: 12, fontWeight: "800" },
  operationError: { color: palette.negative, fontSize: 12, fontWeight: "800", lineHeight: 19 },
  disabled: { opacity: 0.5 },
  positionSymbol: { color: palette.text, fontSize: 14, fontWeight: "800" },
  positionValue: { color: palette.text, fontSize: 13, fontWeight: "900" },
  orderRow: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 54, paddingVertical: spacing.xs },
  orderStatus: { color: palette.warning, fontSize: 12, fontWeight: "900", textAlign: "right" },
  pagination: { flexDirection: "row", gap: spacing.sm, justifyContent: "flex-end", paddingTop: spacing.sm },
  pageButton: { borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, minHeight: 36, paddingHorizontal: spacing.sm, justifyContent: "center" },
  pageButtonText: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  accountGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  accountMetric: { backgroundColor: palette.surfaceRaised, borderRadius: radius.sm, flexBasis: "47%", flexGrow: 1, gap: spacing.xxs, minWidth: 136, padding: spacing.sm },
  accountLabel: { color: palette.textMuted, fontSize: 11, fontWeight: "800" },
  accountValue: { color: palette.text, fontSize: 15, fontWeight: "900" },
  rule: { backgroundColor: palette.border, height: 1 },
  refreshButton: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md },
  refreshText: { color: palette.textMuted, fontSize: 14, fontWeight: "800" },
  pressed: { opacity: 0.76 },
});
