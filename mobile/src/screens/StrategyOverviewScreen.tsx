import { useEffect, useMemo, useRef, useState } from "react";
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
import {
  AlertTriangle,
  CandlestickChart as CandlestickIcon,
  ChevronRight,
  LineChart,
  ReceiptText,
  RefreshCw,
  ShieldAlert,
} from "lucide-react-native";
import { useNavigation, useRoute, type RouteProp } from "@react-navigation/native";
import { api } from "../api";
import {
  AllocationCapSheet,
  BottomSheetSelector,
  ConcurrencyMatrix,
  EquityCurveChart,
  MarketIndicatorPanel,
  MetricCard,
  PrimaryButton,
  SectionCard,
  SharedWalletBudgetChart,
  StatePanel,
  StrategyCapitalMeterChart,
  StrategyEvaluationPanel,
  StrategyPnlComparisonChart,
  StrategyTradeQualityChart,
} from "../components";
import {
  accountUtilizationDescription,
  attributionStatusLabel,
  concurrencyMatrixRows,
  executionGateLabel,
  executionGateTone,
  formatRatioPercent,
  formatUsdt,
  resolveAllocationName,
  syncStatusLabel,
  walletEquityCurvePoints,
  type ConcurrencyMatrixRow,
  type ConcurrencyTone,
} from "../concurrency";
import type { WorkbenchStackParamList } from "../navigation/types";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import {
  executionEnvironmentLabel,
  strategyStatusLabel,
} from "./strategy-presentation";
import {
  formatQuote,
  formatTimestamp,
  readActiveStrategyId,
  saveActiveStrategyId,
  selectActiveStrategyId,
} from "./workbench";
import { attributedDecisionReason, executionQueryScope } from "../execution-scope";

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

function toneTextStyle(tone: ConcurrencyTone): { color: string } {
  if (tone === "positive") return { color: palette.positive };
  if (tone === "negative") return { color: palette.negative };
  if (tone === "warning") return { color: palette.warning };
  return { color: palette.textMuted };
}

function formatNumericQuote(value: number | string | null | undefined): string {
  const number = typeof value === "string" ? Number(value) : value;
  return typeof number === "number" && Number.isFinite(number) ? formatQuote(number) : "—";
}

/**
 * 并发策略控制台. The same persisted allocator facts the Web dashboard renders
 * drive this screen; selecting a strategy card swaps the attribution and market
 * context in place, so nothing reloads and no number is re-derived on the client.
 */
export default function StrategyOverviewScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<StrategyOverviewRoute>();
  const { session } = useSession();
  const [selectedId, setSelectedId] = useState("");
  const [selectorVisible, setSelectorVisible] = useState(false);
  const [capRow, setCapRow] = useState<ConcurrencyMatrixRow | null>(null);
  const [marketFocusSymbol, setMarketFocusSymbol] = useState<string | null>(null);
  const scrollRef = useRef<ScrollView>(null);
  const marketBlockY = useRef<number | null>(null);

  // The concurrency console switches context in place: selecting a card or a
  // matrix row re-points the attribution and the market block instead of
  // navigating away and remounting the page.
  const scrollToMarket = (symbol?: string) => {
    if (symbol) setMarketFocusSymbol(symbol);
    const blockY = marketBlockY.current;
    if (blockY == null) return;
    scrollRef.current?.scrollTo({ animated: true, y: Math.max(0, blockY - spacing.md) });
  };

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
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "trades", 20],
    queryFn: () => api.strategyLog(activeId, "trades", 20),
    enabled: Boolean(activeId && !isDemo),
    refetchInterval: 15_000,
  });
  const evaluations = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "evaluations", 20],
    queryFn: () => api.strategyEvaluations(activeId, 20),
    enabled: Boolean(activeId),
    refetchInterval: 15_000,
  });
  const batches = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "batches"],
    queryFn: () => api.strategyBatches(activeId),
    enabled: Boolean(activeId),
  });
  const scope = executionQueryScope({
    environment: strategy?.config.execution.environment,
    status: strategy?.status,
    currentBatchId: batches.isSuccess ? (batches.data.current_batch_id ?? null) : undefined,
  });
  const recordsReady = Boolean(activeId && scope.ready && scope.unavailableReason !== "no_current_batch");
  const tradeFacts = useQuery({
    queryKey: ["mobile", session?.tenantId, "all-trade-facts", activeId, scope.batchId ?? "current"],
    queryFn: () => api.allTradeFacts(activeId, 20, scope.batchId),
    enabled: Boolean(recordsReady && isDemo),
    refetchInterval: 15_000,
  });
  const demo = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", activeId, "demo-execution", scope.batchId ?? "current"],
    queryFn: () => api.strategyDemoExecution(activeId, 1, 10, scope.batchId),
    enabled: Boolean(recordsReady && isDemo),
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
    queryFn: () => api.strategyRiskState(activeId),
    enabled: Boolean(activeId),
    refetchInterval: 15_000,
  });

  const refresh = () => {
    void Promise.all([
      strategies.refetch(),
      sharedAccount.refetch(),
      account.refetch(),
      pnl.refetch(),
      trades.refetch(),
      batches.refetch(),
      tradeFacts.refetch(),
      evaluations.refetch(),
      demo.refetch(),
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
  const sharedData = sharedAccount.data;
  const allocations = sharedData?.allocator.allocations ?? [];
  const matrixRows = concurrencyMatrixRows(allocations, strategies.data);
  const resolveName = (allocation: Parameters<typeof resolveAllocationName>[0]) =>
    resolveAllocationName(allocation, strategies.data);
  const selectedAllocation = allocations.find((item) => item.strategy_id === activeId) ?? null;
  const walletCurve = walletEquityCurvePoints(sharedData);
  const latestEvaluation = evaluations.data?.[0];
  const runningCount = (strategies.data ?? []).filter(
    (item) => item.status === "running" && !item.archived_at,
  ).length;
  const totalCount = (strategies.data ?? []).length;
  const paperAccount = account.data;
  const paperPositions = Object.entries(paperAccount?.positions ?? {});
  const paperPnl = paperAccount
    ? paperAccount.realized_pnl_quote + paperAccount.unrealized_pnl_quote
    : undefined;
  const demoData = demo.data;
  const demoValuedPositions = (demoData?.positions.data.positions ?? []).filter(
    (position): position is typeof position & { notional_usdt: number } =>
      typeof position.notional_usdt === "number" && Number.isFinite(position.notional_usdt),
  );
  const demoPositionNotional = demoValuedPositions.reduce(
    (total, position) => total + position.notional_usdt,
    0,
  );
  const monitoredSymbols = strategy.config.symbols.length;
  const gateTone = sharedData ? executionGateTone(sharedData.execution_gate.status) : "default";
  const incomplete =
    sharedData != null &&
    (!sharedData.data_complete ||
      sharedData.wallet.sync_status !== "healthy" ||
      sharedData.wallet.attribution_status !== "complete");

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      ref={scrollRef}
      refreshControl={
        <RefreshControl
          onRefresh={refresh}
          refreshing={strategies.isRefetching || sharedAccount.isRefetching || evaluations.isRefetching}
          tintColor={palette.primary}
        />
      }
      style={styles.page}
    >
      <View style={styles.heading}>
        <View style={styles.headingCopy}>
          <Text style={styles.eyebrow}>{isDemo ? "OKX DEMO 共享账户并发终端" : "纸面交易终端"}</Text>
          <Text style={styles.title}>并发策略控制台</Text>
          <Text style={styles.subtitle}>
            四策略共用同一 OKX 钱包，各自独立批次与成交归属；点击下方任一策略卡片即可切换图表与归因，无需整页刷新
          </Text>
        </View>
        <Pressable
          accessibilityLabel="切换归因策略"
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
        <MetricCard
          caption={`当前工作区共 ${totalCount} 条策略，未归档且运行中的计入`}
          label="运行策略"
          style={styles.metric}
          tone={runningCount > 0 ? "positive" : "default"}
          value={`${runningCount} / ${totalCount}`}
        />
        <MetricCard
          caption={accountUtilizationDescription(sharedData)}
          label="账户利用率"
          style={styles.metric}
          tone="warning"
          value={sharedData ? formatRatioPercent(sharedData.allocator.account_utilization_ratio) : "—"}
        />
        <MetricCard
          caption="OKX 钱包权威同步值，不归任何单一策略"
          label="钱包总权益"
          style={styles.metric}
          value={formatUsdt(sharedData?.wallet.total_equity_quote)}
        />
        <MetricCard
          caption={`当前策略正在监测 ${monitoredSymbols} 个市场`}
          label="策略归属 PnL"
          style={styles.metric}
          tone={
            sharedData?.strategy_pnl_total_quote == null
              ? "default"
              : sharedData.strategy_pnl_total_quote >= 0
                ? "positive"
                : "negative"
          }
          value={formatUsdt(sharedData?.strategy_pnl_total_quote)}
        />
      </View>
      <SectionCard
        description="钱包权威总额与策略归属分配分开呈现，不将当前策略视为整个账户"
        title="共享账户并发控制台"
      >
        {!credentialId ? (
          <StatePanel
            description="配置 OKX Sandbox 连接后，这里会显示钱包权益、四策略资金分配、并发矩阵与共享钱包曲线。"
            title="缺少 Sandbox 连接"
          />
        ) : null}
        {credentialId && sharedAccount.isLoading && !sharedData ? (
          <StatePanel description="正在读取 OKX 钱包与共享 allocator 快照。" state="loading" title="正在同步共享账户" />
        ) : null}
        {credentialId && sharedAccount.isError ? (
          <StatePanel
            actionLabel="重试"
            description={`${(sharedAccount.error as Error).message}；在数据恢复前不会用策略账户数值替代钱包权威总额。`}
            onAction={() => void sharedAccount.refetch()}
            title="共享账户暂不可用"
            tone="error"
          />
        ) : null}
        {credentialId && sharedData ? (
          <>
            <View style={styles.badgeRow}>
              <View style={[styles.badge, { borderColor: palette.primary }]}>
                <Text style={[styles.badgeText, { color: palette.primary }]}>
                  {syncStatusLabel(sharedData.wallet.sync_status)}
                </Text>
              </View>
              <View style={[styles.badge, { borderColor: palette.border }]}>
                <Text style={styles.badgeText}>{attributionStatusLabel(sharedData.wallet.attribution_status)}</Text>
              </View>
              <View style={[styles.badge, { borderColor: palette.border }]}>
                <Text style={styles.badgeText}>观测 {formatTimestamp(sharedData.wallet.observed_at)}</Text>
              </View>
              <View style={[styles.badge, { borderColor: palette.primary }]}>
                <Text style={[styles.badgeText, toneTextStyle(gateTone)]}>
                  {executionGateLabel(sharedData.execution_gate.status)}
                </Text>
              </View>
            </View>

            {incomplete ? (
              <View style={styles.warningBox}>
                <AlertTriangle color={palette.warning} size={15} />
                <Text style={styles.warningText}>
                  {sharedData.incomplete_reason ??
                    (sharedData.wallet.sync_status !== "healthy"
                      ? "钱包同步状态异常，权威余额可能暂时不可用。"
                      : "部分策略归因尚未完成，归属 PnL 仅供参考。")}
                </Text>
              </View>
            ) : null}

            {sharedData.execution_gate.reasons.length > 0 ? (
              <View style={styles.dangerBox}>
                <Text style={styles.dangerText}>
                  新开仓门禁：{sharedData.execution_gate.reasons.join("；")}
                </Text>
              </View>
            ) : null}

            <View style={styles.accountGrid}>
              <View style={styles.accountMetric}>
                <Text style={styles.accountLabel}>钱包总权益 · 权威</Text>
                <Text style={styles.accountValue}>{formatUsdt(sharedData.wallet.total_equity_quote)}</Text>
              </View>
              <View style={styles.accountMetric}>
                <Text style={styles.accountLabel}>钱包可用余额 · 权威</Text>
                <Text style={styles.accountValue}>{formatUsdt(sharedData.wallet.available_quote)}</Text>
              </View>
              <View style={styles.accountMetric}>
                <Text style={styles.accountLabel}>策略可分配余额 · allocator</Text>
                <Text style={styles.accountValue}>{formatUsdt(sharedData.allocator.available_for_strategies_quote)}</Text>
              </View>
              <View style={styles.accountMetric}>
                <Text style={styles.accountLabel}>已预留 · allocator</Text>
                <Text style={styles.accountValue}>{formatUsdt(sharedData.allocator.reserved_quote)}</Text>
              </View>
              <View style={styles.accountMetric}>
                <Text style={styles.accountLabel}>已占用名义</Text>
                <Text style={styles.accountValue}>{formatUsdt(sharedData.allocator.occupied_notional_quote)}</Text>
              </View>
              <View style={styles.accountMetric}>
                <Text style={styles.accountLabel}>待结算</Text>
                <Text style={styles.accountValue}>{formatUsdt(sharedData.allocator.pending_settlement_quote)}</Text>
              </View>
              <View style={styles.accountMetric}>
                <Text style={styles.accountLabel}>可再投资余额</Text>
                <Text style={styles.accountValue}>{formatUsdt(sharedData.allocator.reusable_quote)}</Text>
              </View>
              <View style={styles.accountMetric}>
                <Text style={styles.accountLabel}>未归因权益 · 钱包</Text>
                <Text style={styles.accountValue}>{formatUsdt(sharedData.wallet.unassigned_equity_quote)}</Text>
              </View>
              <View style={styles.accountMetric}>
                <Text style={styles.accountLabel}>钱包 − 策略差额</Text>
                <Text style={styles.accountValue}>{formatUsdt(sharedData.wallet_strategy_reconciliation_delta_quote)}</Text>
              </View>
            </View>
            <View style={styles.block}>
              <View style={styles.blockHeader}>
                <Text style={styles.blockTitle}>OKX 共享钱包权益曲线</Text>
                <Text style={styles.blockMeta}>{walletCurve.length} 个快照</Text>
              </View>
              <Text style={styles.muted}>
                仅来自后台持久化的钱包快照，用于核对四策略共同作用后的账户总金额变化
              </Text>
              {sharedData.wallet_equity_curve.status === "available" && walletCurve.length > 0 ? (
                <EquityCurveChart
                  formatQuote={formatQuote}
                  formatTimestamp={formatTimestamp}
                  height={200}
                  points={walletCurve}
                />
              ) : (
                <Text style={styles.muted}>尚无可用的钱包权益快照，后台同步成功后自动显示。</Text>
              )}
              {walletCurve.length === 1 ? (
                <Text style={styles.muted}>当前只有一个账户快照，下一次同步后将形成变化曲线。</Text>
              ) : null}
            </View>

            <View style={styles.block}>
              <Text style={styles.blockTitle}>四策略并发运行矩阵</Text>
              <Text style={styles.muted}>
                每一行是一套独立策略；规则、批次和成交归属隔离，资金通过共享账户 allocator 竞争与释放
              </Text>
              <ConcurrencyMatrix
                accountUtilizationRatio={sharedData.allocator.account_utilization_ratio}
                onEditCap={(row) => setCapRow(row)}
                onOpenCharts={(strategyId) => {
                  selectStrategy(strategyId);
                  setMarketFocusSymbol(null);
                  scrollToMarket();
                }}
                onOpenTrades={(strategyId) => {
                  selectStrategy(strategyId);
                  navigation.navigate("TradeLedger", { strategyId });
                }}
                onSelect={selectStrategy}
                rows={matrixRows}
                runningCount={runningCount}
                selectedStrategyId={activeId}
                totalCount={totalCount}
              />
            </View>

            {allocations.length > 0 ? (
              <View style={styles.chartGrid}>
                <StrategyCapitalMeterChart allocations={allocations} resolveName={resolveName} />
                <StrategyPnlComparisonChart allocations={allocations} resolveName={resolveName} />
                <SharedWalletBudgetChart
                  allocations={allocations}
                  resolveName={resolveName}
                  walletAvailableQuote={sharedData.wallet.available_quote}
                  walletEquityQuote={sharedData.wallet.total_equity_quote}
                />
                <StrategyTradeQualityChart allocations={allocations} resolveName={resolveName} />
              </View>
            ) : null}

            {sharedData.allocator.unallocated_strategies.length > 0 ? (
              <View style={styles.warningBox}>
                <View style={styles.unallocatedCopy}>
                  <Text style={styles.warningTitle}>未纳入该共享账户资金池的策略（只读）</Text>
                  <Text style={styles.muted}>
                    这些策略存在，但不是该钱包的资金分配对象，因此不计入上方的预留、占用与利用率。这不代表策略未在运行。
                  </Text>
                  {sharedData.allocator.unallocated_strategies.map((item) => (
                    <Text key={item.strategy_id} style={styles.unallocatedRow}>
                      {item.name} · {item.kind} ·{" "}
                      {item.status === "running" ? "运行中" : item.status === "paused" ? "已暂停" : "已停止"} ·{" "}
                      {item.environment === "paper" ? "Paper 独立账本" : item.environment === "okx_demo" ? "OKX Demo" : "未绑定环境"} ·{" "}
                      {item.reason}
                    </Text>
                  ))}
                </View>
              </View>
            ) : null}
          </>
        ) : null}
      </SectionCard>
      <View
        onLayout={(event) => {
          marketBlockY.current = event.nativeEvent.layout.y;
        }}
      >
        <SectionCard
          description="K 线、成交量与技术指标跟随当前选中策略的币种集合；点击币种即可就地切换走势，不重新加载页面"
          title={`行情走势与技术指标 · ${strategy.name}`}
        >
          <MarketIndicatorPanel
            focusSymbol={marketFocusSymbol}
            onOpenFull={() => navigation.navigate("行情", { screen: "Market", params: { strategyId: activeId } })}
            strategyName={strategy.name}
            symbols={strategy.config.symbols}
          />
        </SectionCard>
      </View>
      <SectionCard
        description="只读取服务端持久化的评估与成交事实；切换上方矩阵卡片即可切换归因对象"
        title={`当前策略归因 · ${strategy.name}`}
      >
        <View style={styles.stateRows}>
          <Text style={styles.stateText}>
            监控池 {displayMonitorState(monitorState.data?.[0]?.state ?? "")} · 风险 {displayRiskState(riskState.data?.state)}
            {selectedAllocation
              ? ` · 该策略利用率 ${formatRatioPercent(selectedAllocation.utilization_ratio)}`
              : " · 该策略未纳入共享资金池"}
          </Text>
          <Text style={styles.muted}>
            最近评估 {latestEvaluation ? formatTimestamp(latestEvaluation.evaluated_at) : "尚无评估记录"} · 预留上限{" "}
            {selectedAllocation?.max_reserved_quote == null ? "未设置" : formatUsdt(selectedAllocation.max_reserved_quote)}
          </Text>
        </View>

        {evaluations.isError ? (
          <StatePanel
            actionLabel="重试"
            description={(evaluations.error as Error).message}
            onAction={() => void evaluations.refetch()}
            title="最近评估暂不可用"
            tone="error"
          />
        ) : null}
        {evaluations.isLoading && !latestEvaluation ? (
          <View style={styles.loading}>
            <ActivityIndicator color={palette.primary} />
            <Text style={styles.loadingText}>正在读取服务端评估与条件事实。</Text>
          </View>
        ) : null}
        {latestEvaluation ? <StrategyEvaluationPanel compact evaluation={latestEvaluation} /> : null}

        <View style={styles.detailLinks}>
          <Pressable
            accessibilityRole="button"
            onPress={() => navigation.navigate("TradeLedger", { strategyId: activeId, batchId: scope.batchId })}
            style={({ pressed }) => [styles.linkRow, pressed && styles.pressed]}
          >
            <ReceiptText color={palette.primary} size={20} />
            <View style={styles.rowCopy}>
              <Text style={styles.linkValue}>交易明细与条件原因</Text>
              <Text style={styles.muted}>按时间节点查看成交数量、价格与触发条件数值</Text>
            </View>
            <ChevronRight color={palette.textMuted} size={18} />
          </Pressable>
          <Pressable
            accessibilityRole="button"
            onPress={() => scrollToMarket()}
            style={({ pressed }) => [styles.linkRow, pressed && styles.pressed]}
          >
            <CandlestickIcon color={palette.primary} size={20} />
            <View style={styles.rowCopy}>
              <Text style={styles.linkValue}>本页行情走势与技术指标</Text>
              <Text style={styles.muted}>滚动到上方区块，点击币种即可就地切换 K 线与指标</Text>
            </View>
            <ChevronRight color={palette.textMuted} size={18} />
          </Pressable>
          <Pressable
            accessibilityRole="button"
            onPress={() => navigation.navigate("行情", { screen: "Market", params: { strategyId: activeId } })}
            style={({ pressed }) => [styles.linkRow, pressed && styles.pressed]}
          >
            <LineChart color={palette.primary} size={20} />
            <View style={styles.rowCopy}>
              <Text style={styles.linkValue}>打开完整行情页</Text>
              <Text style={styles.muted}>支持更长历史范围、自定义日期与目录标的对比</Text>
            </View>
            <ChevronRight color={palette.textMuted} size={18} />
          </Pressable>
          <Pressable
            accessibilityRole="button"
            onPress={() => navigation.navigate("StrategyWorkbenchDetail", { strategyId: activeId, section: "risk" })}
            style={({ pressed }) => [styles.linkRow, pressed && styles.pressed]}
          >
            <ShieldAlert color={palette.primary} size={20} />
            <View style={styles.rowCopy}>
              <Text style={styles.linkValue}>监控池与风险详情</Text>
              <Text style={styles.muted}>查看监控池状态与账户级风险原因</Text>
            </View>
            <ChevronRight color={palette.textMuted} size={18} />
          </Pressable>
        </View>
      </SectionCard>

      {!isDemo ? (
        <SectionCard
          actionLabel="全部交易"
          description="该策略为 Paper 独立账本，不参与共享 OKX 钱包分配"
          onAction={() => navigation.navigate("TradeLedger", { strategyId: activeId, batchId: scope.batchId })}
          title="Paper 独立账本"
        >
          <View style={styles.metricGrid}>
            <MetricCard caption={`可用资金 ${formatQuote(paperAccount?.quote_balance)}`} label="账户权益" style={styles.metric} value={formatNumericQuote(paperAccount?.equity_quote)} />
            <MetricCard caption="服务端账户快照累计" label="收益 / 亏损" style={styles.metric} tone={typeof paperPnl === "number" && paperPnl >= 0 ? "positive" : "warning"} value={formatNumericQuote(paperPnl)} />
            <MetricCard caption="服务端归因持仓" label="持仓数量" style={styles.metric} value={`${paperPositions.length} 个`} />
            <MetricCard caption="当前策略观察标的" label="币种观察" style={styles.metric} tone="warning" value={`${monitoredSymbols} 个`} />
          </View>
          {pnl.isError ? <Text style={styles.muted}>{(pnl.error as Error).message}</Text> : null}
          {pnl.isLoading ? (
            <View style={styles.loading}>
              <ActivityIndicator color={palette.primary} />
              <Text style={styles.loadingText}>正在读取每日收益事实。</Text>
            </View>
          ) : (
            <EquityCurveChart formatQuote={formatQuote} formatTimestamp={formatTimestamp} height={176} points={pnl.data ?? []} />
          )}
          {trades.data?.entries.length ? (
            trades.data.entries.slice(0, 3).map((trade) => (
              <View key={trade.evaluation_id} style={styles.tradeRow}>
                <View style={styles.rowCopy}>
                  <Text style={styles.tradeSymbol}>{trade.symbol}</Text>
                  <Text style={styles.muted}>{formatTimestamp(trade.evaluated_at)}</Text>
                </View>
                <Text style={styles.tradeValue}>{formatQuote(trade.quote_amount)}</Text>
              </View>
            ))
          ) : (
            <Text style={styles.muted}>服务端尚无归因成交。</Text>
          )}
        </SectionCard>
      ) : null}

      {isDemo ? (
        <SectionCard
          actionLabel="交易明细"
          description="订单和成交原因都按当前执行批次读取，不把共享钱包或纸面账本混进来"
          onAction={() => navigation.navigate("TradeLedger", { strategyId: activeId, batchId: scope.batchId })}
          title="OKX Demo 订单与成交原因"
        >
          {scope.unavailableReason === "batch_pending" || (recordsReady && demo.isLoading && !demoData) ? <Text style={styles.muted}>正在按当前执行批次读取订单和成交原因。</Text> : null}
          {scope.unavailableReason === "no_current_batch" ? <Text style={styles.muted}>当前没有执行批次。历史订单请打开交易明细后切换到全部历史。</Text> : null}
          {demo.isError ? <Text style={styles.muted}>{(demo.error as Error).message}</Text> : null}
          {demoData ? (
            <>
              <View style={styles.metricGrid}>
                <MetricCard caption="本策略归属持仓名义" label="持仓名义" style={styles.metric} value={formatNumericQuote(demoValuedPositions.length > 0 ? demoPositionNotional : undefined)} />
                <MetricCard caption="OKX Demo 账户总估值" label="账户估值" style={styles.metric} value={formatNumericQuote(demoData.account.data.total_usdt_value)} />
                <MetricCard caption="当前批次归属订单，不含粉尘空操作" label="订单数量" style={styles.metric} value={`${demoData.pagination.total_items} 笔`} />
              </View>
              {demoData.orders.length ? demoData.orders.slice(0, 4).map((order) => (
                <Pressable accessibilityRole="button" key={order.id} onPress={() => navigation.navigate("TradeLedger", { strategyId: activeId, batchId: scope.batchId })} style={styles.tradeRow}>
                  <View style={styles.rowCopy}>
                    <Text style={styles.tradeSymbol}>{order.side === "buy" ? "买入" : "卖出"} · {order.symbol}</Text>
                    <Text style={styles.muted}>{attributedDecisionReason(order, tradeFacts.data ?? [])}</Text>
                  </View>
                  <Text style={styles.tradeValue}>{order.status}</Text>
                </Pressable>
              )) : <Text style={styles.muted}>当前批次没有归因订单。</Text>}
              {(tradeFacts.data ?? []).slice(0, 4).map((fact) => (
                <View key={`${fact.order_id ?? fact.evaluation_id ?? fact.created_at}-${fact.symbol}`} style={styles.tradeRow}>
                  <View style={styles.rowCopy}>
                    <Text style={styles.tradeSymbol}>{fact.symbol} · {fact.side === "buy" ? "买入" : fact.side === "sell" ? "卖出" : fact.side}</Text>
                    <Text style={styles.muted}>{fact.explanation.decision_reason || "服务端未提供成交原因。"}</Text>
                  </View>
                  <Text style={styles.tradeValue}>{formatTimestamp(fact.created_at)}</Text>
                </View>
              ))}
            </>
          ) : null}
        </SectionCard>
      ) : null}
      <PrimaryButton
        label="查看策略详情"
        leading={<LineChart color={palette.canvas} size={19} />}
        onPress={() => navigation.navigate("策略", { screen: "StrategyDetail", params: { strategyId: activeId } })}
      />
      <Pressable accessibilityLabel="刷新并发控制台" accessibilityRole="button" onPress={refresh} style={({ pressed }) => [styles.refreshButton, pressed && styles.pressed]}>
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
        title="选择归因策略"
        visible={selectorVisible}
      />
      <AllocationCapSheet
        credentialId={credentialId ?? ""}
        onClose={() => setCapRow(null)}
        row={capRow}
        tenantId={session?.tenantId}
        visible={capRow != null && Boolean(credentialId)}
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
  title: { color: palette.text, fontSize: 27, fontWeight: "900", letterSpacing: -0.6 },
  subtitle: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  selector: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.primary, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 56, paddingHorizontal: spacing.md },
  selectorCopy: { flex: 1, gap: spacing.xxs },
  selectorText: { color: palette.text, fontSize: 16, fontWeight: "900" },
  selectorMeta: { color: palette.textMuted, fontSize: 12, fontWeight: "700" },
  metricGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  metric: { flexBasis: "47%", minWidth: 148 },
  badgeRow: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xxs },
  badge: { borderRadius: radius.sm, borderWidth: 1, paddingHorizontal: spacing.xs, paddingVertical: 4 },
  badgeText: { color: palette.textMuted, fontSize: 10, fontWeight: "800" },
  warningBox: { backgroundColor: palette.warningSoft, borderColor: palette.warning, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, padding: spacing.sm },
  warningTitle: { color: palette.warning, fontSize: 12, fontWeight: "900" },
  warningText: { color: palette.warning, flex: 1, fontSize: 12, lineHeight: 18 },
  unallocatedCopy: { flex: 1, gap: spacing.xxs },
  unallocatedRow: { color: palette.text, fontSize: 12, lineHeight: 18 },
  dangerBox: { backgroundColor: palette.negativeSoft, borderColor: palette.negative, borderRadius: radius.sm, borderWidth: 1, padding: spacing.sm },
  dangerText: { color: palette.negative, fontSize: 12, fontWeight: "800", lineHeight: 18 },
  accountGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  accountMetric: { backgroundColor: palette.surfaceRaised, borderRadius: radius.sm, flexBasis: "47%", flexGrow: 1, gap: spacing.xxs, minWidth: 136, padding: spacing.sm },
  accountLabel: { color: palette.textMuted, fontSize: 11, fontWeight: "800" },
  accountValue: { color: palette.text, fontSize: 15, fontWeight: "900" },
  block: { borderTopColor: palette.border, borderTopWidth: 1, gap: spacing.xs, paddingTop: spacing.sm },
  blockHeader: { alignItems: "baseline", flexDirection: "row", gap: spacing.xs, justifyContent: "space-between" },
  blockTitle: { color: palette.text, fontSize: 15, fontWeight: "900" },
  blockMeta: { color: palette.textMuted, fontSize: 11, fontWeight: "700" },
  chartGrid: { gap: spacing.sm },
  stateRows: { gap: spacing.xxs },
  stateText: { color: palette.text, fontSize: 13, fontWeight: "800", lineHeight: 20 },
  muted: { color: palette.textMuted, fontSize: 12, lineHeight: 18 },
  detailLinks: { gap: spacing.xs },
  linkRow: { alignItems: "center", borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 60, paddingHorizontal: spacing.sm, paddingVertical: spacing.xs },
  linkValue: { color: palette.text, fontSize: 14, fontWeight: "900" },
  rowCopy: { flex: 1, gap: 2 },
  tradeRow: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 52, paddingVertical: spacing.xs },
  tradeSymbol: { color: palette.text, fontSize: 14, fontWeight: "800" },
  tradeValue: { color: palette.text, fontSize: 13, fontWeight: "900" },
  loading: { alignItems: "center", flexDirection: "row", gap: spacing.sm, minHeight: 48 },
  loadingText: { color: palette.textMuted, flex: 1, fontSize: 13, lineHeight: 19 },
  refreshButton: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md },
  refreshText: { color: palette.textMuted, fontSize: 14, fontWeight: "800" },
  pressed: { opacity: 0.76 },
});
