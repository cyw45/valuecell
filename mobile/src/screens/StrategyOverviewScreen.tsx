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
import { Boxes, ChevronRight, LineChart, ListFilter, ReceiptText, RefreshCw, ShieldAlert, Wallet } from "lucide-react-native";
import { useNavigation, useRoute, type RouteProp } from "@react-navigation/native";
import { api } from "../api";
import {
  BottomSheetSelector,
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
const ALLOCATION_LABELS: Record<string, string> = {
  available: "可分配",
  reserved: "已预留",
  occupied: "已占用",
  partially_released: "部分释放",
  released: "已释放",
  blocked: "已阻断",
};
const SYNC_STATUS_LABELS: Record<string, string> = {
  healthy: "钱包同步正常",
  stale: "钱包数据已过期",
  unavailable: "钱包数据不可用",
};
const ATTRIBUTION_STATUS_LABELS: Record<string, string> = {
  complete: "策略归因完整",
  partial: "策略归因部分完整",
  unavailable: "策略归因不可用",
};
const allocationStateLabel = (state: string) => ALLOCATION_LABELS[state] ?? "状态未知";
const ratioLabel = (value: number | string | null | undefined): string => {
  const number = numberValue(value);
  return typeof number === "number" ? `${(number * 100).toFixed(1)}%` : "—";
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

function numberValue(value: number | string | null | undefined): number | undefined {
  const number = typeof value === "string" ? Number(value) : value;
  return typeof number === "number" && Number.isFinite(number) ? number : undefined;
}

function demoWalletPoints(
  snapshot: RuleStrategyDemoExecution | undefined,
): RuleStrategyPnlPoint[] {
  return (snapshot?.wallet_equity_curve?.points ?? []).flatMap((point) => {
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
  const credentialId = strategy?.config.execution.sandbox_connection_id
    ?? strategies.data?.find(
      (item) => item.config.execution.environment === "okx_demo" && item.config.execution.sandbox_connection_id,
    )?.config.execution.sandbox_connection_id
    ?? null;
  const sharedAccount = useQuery({
    queryKey: ["mobile", session?.tenantId, "shared-account-summary", credentialId ?? ""],
    queryFn: () => api.sharedAccountSummary(credentialId as string),
    enabled: Boolean(credentialId),
    retry: false,
    refetchInterval: 15_000,
  });
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
  const trades = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "trades", 100],
    queryFn: () => api.strategyLog(activeId, "trades", 100),
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
    queryFn: () => api.strategyDemoExecution(activeId, 1, 10),
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
      void Promise.all([strategies.refetch(), sharedAccount.refetch(), evaluations.refetch(), demo.refetch(), monitorState.refetch(), riskState.refetch()]);
      return;
    }
    void Promise.all([
      strategies.refetch(),
      sharedAccount.refetch(),
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
  const sharedData = sharedAccount.data;
  const sharedAllocations = sharedData?.allocator.allocations ?? [];
  const selectedAllocation = sharedAllocations.find((item) => item.strategy_id === activeId);
  const latestEvaluation = evaluations.data?.[0];
  const activeConditionState = latestEvaluation
    ? primaryConditionState(latestEvaluation.conditions)
    : null;
  const runningCount = (strategies.data ?? []).filter(
    (item) => item.status === "running" && !item.archived_at,
  ).length;
  const previewSymbols = strategy.config.symbols.slice(0, 3);
  const remainingSymbolCount = Math.max(0, strategy.config.symbols.length - previewSymbols.length);
  const totalPnl = isDemo
    ? demoData?.pnl.total_pnl ?? demoData?.pnl.total ?? demoData?.pnl.value
    : paperAccount
      ? paperAccount.realized_pnl_quote + paperAccount.unrealized_pnl_quote
      : undefined;

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          onRefresh={refresh}
          refreshing={
            strategies.isRefetching ||
            sharedAccount.isRefetching ||
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
        <MetricCard caption={isDemo ? "OKX Demo 钱包总估值" : `可用资金 ${formatQuote(paperAccount?.quote_balance)}`} label="资金总览" style={styles.metric} tone="default" value={formatQuote(isDemo ? numberValue(demoData?.account.data.total_usdt_value) : paperAccount?.equity_quote)} />
        <MetricCard caption="当前工作区未归档的运行策略" label="运行策略" style={styles.metric} tone={runningCount > 0 ? "positive" : "default"} value={`${runningCount} 个`} />
        <MetricCard caption={strategy.config.symbols.join(" · ") || "未配置观察标的"} label="币种观察" style={styles.metric} tone="warning" value={`${strategy.config.symbols.length} 个`} />
        <MetricCard caption="服务端账户快照累计" label="收益 / 亏损" style={styles.metric} tone={typeof totalPnl === "number" && totalPnl >= 0 ? "positive" : "warning"} value={formatNumericQuote(totalPnl)} />
      </View>


      <SectionCard description={credentialId ? `连接 ${credentialId.slice(0, 8)}… · 钱包与策略归因分开统计` : "当前策略未配置 OKX Sandbox 连接。"} title="共享账户总览">
        {!credentialId ? <StatePanel description="配置 OKX Sandbox 连接后，这里会显示钱包权益、allocator 占用与策略归因。" title="缺少 Sandbox 连接" /> : null}
        {credentialId && sharedAccount.isLoading ? <StatePanel description="正在读取 OKX 钱包与共享 allocator 快照。" state="loading" title="正在同步共享账户" /> : null}
        {credentialId && sharedAccount.isError ? <StatePanel actionLabel="重试" description={(sharedAccount.error as Error).message} onAction={() => void sharedAccount.refetch()} title="共享账户暂不可用" tone="error" /> : null}
        {credentialId && sharedData ? <>
          <View style={styles.stateRows}>
            <Text style={styles.stateText}>{SYNC_STATUS_LABELS[sharedData.wallet.sync_status] ?? "钱包状态未知"} · {ATTRIBUTION_STATUS_LABELS[sharedData.wallet.attribution_status] ?? "归因状态未知"}</Text>
            <Text style={styles.muted}>钱包观测 {formatTimestamp(sharedData.wallet.observed_at)} · allocator 观测 {formatTimestamp(sharedData.allocator.observed_at)}</Text>
            {!sharedData.data_complete ? <Text style={styles.incompleteText}>数据不完整：{sharedData.incomplete_reason ?? "部分共享账户事实尚未就绪"}</Text> : null}
          </View>
          <View style={styles.metricGrid}>
            <MetricCard caption="OKX Sandbox 钱包事实" label="钱包权益" style={styles.metric} value={formatNumericQuote(sharedData.wallet.total_equity_quote)} />
            <MetricCard caption="钱包当前可用余额" label="可用资金" style={styles.metric} value={formatNumericQuote(sharedData.wallet.available_quote)} />
            <MetricCard caption="所有策略归因的累计结果" label="策略归因 PnL" style={styles.metric} tone={typeof sharedData.strategy_pnl_total_quote === "number" && sharedData.strategy_pnl_total_quote >= 0 ? "positive" : "warning"} value={formatNumericQuote(sharedData.strategy_pnl_total_quote)} />
            <MetricCard caption="allocator 已分配待使用" label="已预留" style={styles.metric} value={formatNumericQuote(sharedData.allocator.reserved_quote)} />
            <MetricCard caption="allocator 当前名义占用" label="已占用" style={styles.metric} value={formatNumericQuote(sharedData.allocator.occupied_notional_quote)} />
            <MetricCard caption="可重新分配的 allocator 余额" label="可复用" style={styles.metric} value={formatNumericQuote(sharedData.allocator.reusable_quote)} />
            <MetricCard caption="占用 / allocator 分母" label="账户利用率" style={styles.metric} tone={sharedData.allocator.account_utilization_ratio > 0.9 ? "warning" : "default"} value={ratioLabel(sharedData.allocator.account_utilization_ratio)} />
          </View>
          <View style={styles.allocationRows}>
            <Text style={styles.allocationHeading}>策略分配矩阵</Text>
            {sharedAllocations.length > 0 ? sharedAllocations.map((allocation) => <ListRow key={allocation.strategy_id} subtitle={`${allocationStateLabel(allocation.allocation_state)} · 预留 ${formatNumericQuote(allocation.reserved_quote)} · 占用 ${formatNumericQuote(allocation.occupied_quote)}`} title={allocation.strategy_id === activeId ? `${strategy.name} · 当前选择` : allocation.strategy_id} trailing={<View style={styles.allocationTrailing}><Text style={[styles.linkValue, allocation.net_pnl_quote != null && allocation.net_pnl_quote >= 0 ? styles.positiveText : styles.negativeText]}>{formatNumericQuote(allocation.net_pnl_quote)}</Text><Text style={styles.allocationPnlLabel}>净 PnL</Text></View>} />) : <Text style={styles.muted}>服务端尚未返回策略分配事实。</Text>}
          </View>
          {!selectedAllocation ? <Text style={styles.incompleteText}>当前策略暂无共享分配记录；上方钱包事实不代表当前策略资金。</Text> : null}
        </> : null}
        {credentialId && !sharedAccount.isLoading && !sharedAccount.isError && !sharedData ? <StatePanel description="服务端未返回共享账户快照。" title="共享账户数据不可用" /> : null}
      </SectionCard>

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
              <ListRow accessibilityLabel="查看交易所持仓" leading={<Boxes color={palette.primary} size={20} />} onPress={() => navigation.navigate("StrategyPositions", { strategyId: activeId })} subtitle={`${demoPositions.length} 项连接级仓位 · 不代表策略分配`} title="我的持仓" trailing={<Text style={styles.linkValue}>{formatQuote(positionValue)}</Text>} />
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
              <ListRow accessibilityLabel="查看纸面持仓" leading={<Boxes color={palette.primary} size={20} />} onPress={() => navigation.navigate("StrategyPositions", { strategyId: activeId })} subtitle={`${paperPositions.length} 项持仓 · 策略上限 ${strategy.config.risk.max_positions}`} title="我的持仓" trailing={<Text style={styles.linkValue}>{formatQuote(positionValue)}</Text>} />
              <ListRow accessibilityLabel="查看全部纸面成交" leading={<ReceiptText color={palette.primary} size={20} />} onPress={() => navigation.navigate("TradeLedger", { strategyId: activeId })} subtitle="查看服务端归因成交记录" title="成交账本" />
              <ListRow accessibilityLabel="查看资金费与权益曲线" leading={<Wallet color={palette.primary} size={20} />} onPress={() => navigation.navigate("FundingPnl", { strategyId: activeId })} subtitle="查看权益曲线与资金费影响" title="资金费与 PnL" />
            </View>
          </SectionCard>
          <SectionCard actionLabel="查看详情" description="按 UTC 日期展示服务端持久化的日结权益与当日盈亏；无账户快照不会补造曲线点。" onAction={() => navigation.navigate("FundingPnl", { strategyId: activeId })} title="每日收益曲线">
            {pnl.isError ? <Text style={styles.muted}>{(pnl.error as Error).message}</Text> : pnl.isLoading ? <View style={styles.loading}><ActivityIndicator color={palette.primary} /><Text style={styles.loadingText}>正在读取每日收益事实。</Text></View> : <EquityCurveChart formatQuote={formatQuote} formatTimestamp={formatTimestamp} height={176} points={pnl.data ?? []} />}
          </SectionCard>
        </>
      )}

      <SectionCard actionLabel="全部交易" description="最新交易明细按页查看；Demo 展示策略归属交易所订单。" onAction={() => navigation.navigate(isDemo ? "ExecutionFacts" : "TradeLedger", isDemo ? { strategyId: activeId, kind: "orders" } : { strategyId: activeId })} title="交易明细">
        {isDemo ? demoData?.orders.length ? demoData.orders.slice(0, 3).map((order) => <View key={order.id} style={styles.orderRow}><View style={styles.rowCopy}><Text style={styles.positionSymbol}>{orderSideLabel(order.side)} · {order.symbol}</Text><Text style={styles.muted}>{orderTypeLabel(order.type)} · {formatTimestamp(order.updated_at)}</Text></View><Text style={styles.orderStatus}>{orderStatusLabel(order.status)}</Text></View>) : <Text style={styles.muted}>当前没有策略归属订单。</Text> : trades.data?.entries.length ? trades.data.entries.slice(0, 3).map((trade) => <View key={trade.evaluation_id} style={styles.orderRow}><View style={styles.rowCopy}><Text style={styles.positionSymbol}>{strategyActionLabel(trade.action)} · {trade.symbol}</Text><Text style={styles.muted}>{formatTimestamp(trade.evaluated_at)}</Text></View><Text style={styles.positionValue}>{formatQuote(trade.quote_amount)}</Text></View>) : <Text style={styles.muted}>服务端尚无归因成交。</Text>}
      </SectionCard>

      {evaluations.isError ? <StatePanel actionLabel="重试" description={(evaluations.error as Error).message} onAction={() => void evaluations.refetch()} title="最近评估暂不可用" tone="error" /> : null}
      <SectionCard description="执行、决策和风控细节进入专用页面，首页只保留入口。" title="策略详情入口">
        <View style={styles.detailLinks}>
          <ListRow accessibilityLabel="查看执行概览" leading={<LineChart color={palette.primary} size={20} />} onPress={() => navigation.navigate("StrategyWorkbenchDetail", { strategyId: activeId, section: "execution" })} subtitle="执行环境、策略代际和服务端执行状态" title="执行概览" />
          <ListRow accessibilityLabel="查看策略决策" leading={<ReceiptText color={palette.primary} size={20} />} onPress={() => navigation.navigate("StrategyWorkbenchDetail", { strategyId: activeId, section: "decision" })} subtitle={latestEvaluation ? `最近评估 ${formatTimestamp(latestEvaluation.evaluated_at)}` : "查看条件、执行漏斗和决策原因"} title="策略决策说明" />
          <ListRow accessibilityLabel="查看监控池和风险" leading={<ShieldAlert color={palette.primary} size={20} />} onPress={() => navigation.navigate("StrategyWorkbenchDetail", { strategyId: activeId, section: "risk" })} subtitle="查看监控池状态和账户级风险原因" title="监控池与风险" />
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
  incompleteText: { color: palette.warning, fontSize: 12, fontWeight: "800", lineHeight: 18 },
  allocationRows: { gap: spacing.xs },
  allocationHeading: { color: palette.text, fontSize: 14, fontWeight: "900", paddingTop: spacing.xs },
  allocationTrailing: { alignItems: "flex-end", gap: spacing.xxs },
  allocationPnlLabel: { color: palette.textMuted, fontSize: 10, fontWeight: "700" },
  positiveText: { color: palette.positive },
  negativeText: { color: palette.negative },
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
