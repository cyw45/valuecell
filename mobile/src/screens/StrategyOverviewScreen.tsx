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
import { Boxes, ChevronRight, Download, Landmark, LineChart, ListFilter, Pause, Pencil, Play, ReceiptText, RefreshCw, Trash2, Wallet } from "lucide-react-native";
import { useNavigation, useRoute, type RouteProp } from "@react-navigation/native";
import { api } from "../api";
import { accessGate, canMutate } from "../access";
import { StrategyExportPanel } from "../components/StrategyExportPanel";
import {
  BottomSheetSelector,
  ConfirmSheet,
  EquityCurveChart,
  ListRow,
  MetricCard,
  PrimaryButton,
  SectionCard,
  StatePanel,
  StrategyEvaluationPanel,
} from "../components";
import type { WorkbenchStackParamList } from "../navigation/types";
import type { RuleStrategyDemoExecution, RuleStrategyPnlPoint } from "../types";
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

function demoWalletPoints(
  snapshot: RuleStrategyDemoExecution | undefined,
): RuleStrategyPnlPoint[] {
  return (snapshot?.equity_curve?.points ?? []).flatMap((point) => {
    const toNumber = (value: number | string | null | undefined) =>
      typeof value === "string" ? Number(value) : value;
    const equity = toNumber(point.equity_quote ?? point.equity);
    const pnl = toNumber(point.cumulative_pnl ?? point.total_pnl ?? point.pnl);
    const dailyPnl = toNumber(point.daily_pnl_quote);
    const timestamp = point.ts ?? point.timestamp;
    return timestamp && typeof equity === "number" && Number.isFinite(equity) && typeof pnl === "number" && Number.isFinite(pnl)
      ? [{
          ts: timestamp,
          equity_quote: equity,
          cumulative_pnl: pnl,
          daily_pnl_quote: typeof dailyPnl === "number" && Number.isFinite(dailyPnl) ? dailyPnl : undefined,
          action: point.action ?? "wallet_snapshot",
        }]
      : [];
  });
}

export default function StrategyOverviewScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<StrategyOverviewRoute>();
  const { session } = useSession();
  const [selectedId, setSelectedId] = useState("");
  const [selectorVisible, setSelectorVisible] = useState(false);
  const queryClient = useQueryClient();
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
  const demo = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "demo-execution", "summary"],
    queryFn: () => api.strategyDemoExecution(activeId, 1, 1),
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
  const demoCurvePoints = demoWalletPoints(demoData);
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
  const previewSymbols = strategy.config.symbols.slice(0, 3);
  const remainingSymbolCount = Math.max(0, strategy.config.symbols.length - previewSymbols.length);

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          onRefresh={refresh}
          refreshing={
            strategies.isRefetching ||
            account.isRefetching ||
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


      <SectionCard actionLabel={remainingSymbolCount > 0 ? `全部 ${strategy.config.symbols.length} 个` : undefined} description="默认只显示前三个；完整观察池在独立页面中检索和查看。" onAction={remainingSymbolCount > 0 ? () => navigation.navigate("StrategySymbols", { strategyId: activeId }) : undefined} title="观察标的">
        <View style={styles.symbols}>
          {previewSymbols.map((symbol) => (
            <Pressable accessibilityLabel={`查看 ${symbol} 行情`} accessibilityRole="button" key={symbol} onPress={() => navigation.navigate("行情", { screen: "Market", params: { strategyId: activeId, symbol } })} style={({ pressed }) => [styles.symbol, pressed && styles.pressed]}>
              <Text style={styles.symbolText}>{symbol}</Text>
              <ChevronRight color={palette.primary} size={16} />
            </Pressable>
          ))}
        </View>
        {remainingSymbolCount > 0 ? <Pressable accessibilityLabel="查看全部观察标的" accessibilityRole="button" onPress={() => navigation.navigate("StrategySymbols", { strategyId: activeId })} style={({ pressed }) => [styles.moreSymbols, pressed && styles.pressed]}><ListFilter color={palette.primary} size={17} /><Text style={styles.moreSymbolsText}>还有 {remainingSymbolCount} 个标的，查看全部</Text><ChevronRight color={palette.primary} size={17} /></Pressable> : null}
      </SectionCard>

      {isDemo ? (
        <>
          {demo.isLoading ? <View style={styles.loading}><ActivityIndicator color={palette.primary} /><Text style={styles.loadingText}>正在读取 OKX Demo 交易所执行数据。</Text></View> : null}
          {demo.isError ? <StatePanel actionLabel="重试" description={(demo.error as Error).message} onAction={() => void demo.refetch()} title="Demo 执行数据暂不可用" tone="error" /> : null}
          <SectionCard description={demoData ? `来源：${demoSourceLabel(demoData.source)} · 最近核验 ${formatTimestamp(demoData.checked_at)}` : "仅在独立页面展示交易所返回的明细。"} title="OKX Demo 执行概览">
            <View style={styles.accountGrid}>
              <View style={styles.accountMetric}><Text style={styles.accountLabel}>总 PnL</Text><Text style={styles.accountValue}>{formatNumericQuote(demoData?.pnl.total_pnl ?? demoData?.pnl.total ?? demoData?.pnl.value)}</Text></View>
              <View style={styles.accountMetric}><Text style={styles.accountLabel}>已成交订单</Text><Text style={styles.accountValue}>{demoData?.trade_summary?.filled_order_count ?? "—"}</Text></View>
            </View>
            <View style={styles.detailLinks}>
              <ListRow accessibilityLabel="查看交易所仓位" leading={<Boxes color={palette.primary} size={20} />} onPress={() => navigation.navigate("ExecutionFacts", { strategyId: activeId, kind: "positions" })} subtitle={`${demoPositions.length} 项连接级仓位 · 不代表策略分配`} title="交易所仓位" trailing={<Text style={styles.linkValue}>{formatQuote(positionValue)}</Text>} />
              <ListRow accessibilityLabel="查看交易所余额" leading={<Landmark color={palette.primary} size={20} />} onPress={() => navigation.navigate("ExecutionFacts", { strategyId: activeId, kind: "balances" })} subtitle={`${demoBalances.length} 项连接级余额`} title="交易所余额" />
              <ListRow accessibilityLabel="查看策略归属订单" leading={<ReceiptText color={palette.primary} size={20} />} onPress={() => navigation.navigate("ExecutionFacts", { strategyId: activeId, kind: "orders" })} subtitle={`共 ${demoData?.pagination.total_items ?? 0} 笔 · 已成交 ${demoData?.trade_summary?.filled_order_count ?? 0} 笔`} title="策略归属订单" />
              <ListRow accessibilityLabel="查看 Demo 资金费与 PnL" leading={<Wallet color={palette.primary} size={20} />} onPress={() => navigation.navigate("FundingPnl", { strategyId: activeId })} subtitle="查看交易所 PnL 与权益曲线" title="资金费与 PnL" />
            </View>
          </SectionCard>
          <SectionCard actionLabel="查看详情" description="账户钱包每日快照来自 OKX Demo 真实余额与估值；从本次部署开始累计历史。" onAction={() => navigation.navigate("FundingPnl", { strategyId: activeId })} title="OKX Demo 每日收益曲线">
            {demo.isError ? <Text style={styles.muted}>{(demo.error as Error).message}</Text> : demo.isLoading ? <View style={styles.loading}><ActivityIndicator color={palette.primary} /><Text style={styles.loadingText}>正在读取交易所钱包日结事实。</Text></View> : <EquityCurveChart formatQuote={formatQuote} formatTimestamp={formatTimestamp} height={176} points={demoCurvePoints} />}
          </SectionCard>
        </>
      ) : (
        <>
          {account.isError ? <StatePanel actionLabel="重试" description={(account.error as Error).message} onAction={() => void account.refetch()} title="纸面账户暂不可用" tone="error" /> : null}
          <SectionCard description="详细仓位、成交和权益曲线均进入独立页面，工作台只保留当前摘要。" title="纸面执行概览">
            <View style={styles.accountGrid}>
              <View style={styles.accountMetric}><Text style={styles.accountLabel}>当前权益</Text><Text style={styles.accountValue}>{formatQuote(paperAccount?.equity_quote)}</Text></View>
              <View style={styles.accountMetric}><Text style={styles.accountLabel}>可用资金</Text><Text style={styles.accountValue}>{formatQuote(paperAccount?.quote_balance)}</Text></View>
            </View>
            <View style={styles.detailLinks}>
              <ListRow accessibilityLabel="查看纸面仓位" leading={<Boxes color={palette.primary} size={20} />} onPress={() => navigation.navigate("ExecutionFacts", { strategyId: activeId, kind: "positions" })} subtitle={`${paperPositions.length} 项持仓 · 策略上限 ${strategy.config.risk.max_positions}`} title="纸面账户仓位" trailing={<Text style={styles.linkValue}>{formatQuote(positionValue)}</Text>} />
              <ListRow accessibilityLabel="查看全部纸面成交" leading={<ReceiptText color={palette.primary} size={20} />} onPress={() => navigation.navigate("TradeLedger", { strategyId: activeId })} subtitle="查看服务端归因成交记录" title="成交账本" />
              <ListRow accessibilityLabel="查看资金费与权益曲线" leading={<Wallet color={palette.primary} size={20} />} onPress={() => navigation.navigate("FundingPnl", { strategyId: activeId })} subtitle="查看权益曲线与资金费影响" title="资金费与 PnL" />
            </View>
          </SectionCard>
          <SectionCard actionLabel="查看详情" description="按 UTC 日期展示服务端持久化的日结权益与当日盈亏；无账户快照不会补造曲线点。" onAction={() => navigation.navigate("FundingPnl", { strategyId: activeId })} title="每日收益曲线">
            {pnl.isError ? <Text style={styles.muted}>{(pnl.error as Error).message}</Text> : pnl.isLoading ? <View style={styles.loading}><ActivityIndicator color={palette.primary} /><Text style={styles.loadingText}>正在读取每日收益事实。</Text></View> : <EquityCurveChart formatQuote={formatQuote} formatTimestamp={formatTimestamp} height={176} points={pnl.data ?? []} />}
          </SectionCard>
        </>
      )}

      {evaluations.isError ? <StatePanel actionLabel="重试" description={(evaluations.error as Error).message} onAction={() => void evaluations.refetch()} title="最近评估暂不可用" tone="error" /> : null}
      <SectionCard actionLabel="策略详情" description={latestEvaluation ? `最近评估：${formatTimestamp(latestEvaluation.evaluated_at)} · ${evaluationReason(latestEvaluation.reason_code, latestEvaluation.reason)}` : "服务端尚无已保存的策略评估。"} onAction={() => navigation.navigate("策略", { screen: "StrategyDetail", params: { strategyId: activeId } })} title="策略决策">
        <Text style={styles.decisionSummary}>{latestEvaluation ? `${strategyActionLabel(latestEvaluation.action)} · ${conditionStateSummary(latestEvaluation.conditions)}` : "查看策略详情、条件诊断与执行漏斗。"}</Text>
      </SectionCard>

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
  moreSymbols: { alignItems: "center", backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, minHeight: 44, paddingHorizontal: spacing.sm },
  moreSymbolsText: { color: palette.primary, flex: 1, fontSize: 12, fontWeight: "800" },
  detailLinks: { gap: spacing.xs },
  linkValue: { color: palette.text, fontSize: 12, fontWeight: "900" },
  decisionSummary: { color: palette.text, fontSize: 13, lineHeight: 20 },
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
