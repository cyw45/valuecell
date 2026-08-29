import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { useNavigation, useRoute } from "@react-navigation/native";
import { Archive, Bot, ChartCandlestick, CirclePlay, Pause, Wallet } from "lucide-react-native";
import { api } from "../api";
import { StrategyExportPanel } from "../components/StrategyExportPanel";
import { accessGate, canMutate } from "../access";
import {
  ConfirmSheet,
  DangerButton,
  ListRow,
  PrimaryButton,
  ScreenHeader,
  SectionCard,
  StatePanel,
  StrategyEvaluationPanel,
} from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import type { Strategy } from "../types";
import {
  conditionDetail,
  conditionLabel,
  conditionStateLabel,
  conditionStateTone,
  demoPnlReason,
  demoSourceLabel,
  evaluationReason,
  executionEnvironmentLabel,
  fundingDirectionLabel,
  orderSideLabel,
  orderStatusLabel,
  orderTypeLabel,
  paperPositionValue,
  strategyActionLabel,
  strategyStatusLabel,
} from "./strategy-presentation";
import { formatQuote, formatTimestamp, readActiveStrategyId, saveActiveStrategyId } from "./workbench";

type RouteParams = { params: { strategyId: string } };
type LifecycleAction = "start" | "stop" | "archive";
type Confirmation = { action: LifecycleAction; strategy: Strategy };

function confirmationCopy(confirmation: Confirmation): { destructive: boolean; label: string; message: string; title: string } {
  const environment = executionEnvironmentLabel(confirmation.strategy.config.execution.environment);
  if (confirmation.action === "start") {
    return {
      destructive: false,
      label: "启动策略",
      message: `确认以${environment}启动“${confirmation.strategy.name}”？策略会根据服务器端配置开始后续评估。`,
      title: "启动策略",
    };
  }
  if (confirmation.action === "stop") {
    return {
      destructive: true,
      label: "停止策略",
      message: `确认停止“${confirmation.strategy.name}”？服务端将停止提交新的策略执行。`,
      title: "停止策略",
    };
  }
  return {
    destructive: true,
    label: "归档策略",
    message: `确认归档“${confirmation.strategy.name}”？归档会保留评估、成交、资金费与审计历史。策略必须已停止，且不能恢复。`,
    title: "归档策略",
  };
}

function formatNumericQuote(value: number | string | null | undefined): string {
  const number = typeof value === "string" ? Number(value) : value;
  return typeof number === "number" ? formatQuote(number) : "—";
}

export default function StrategyDetailScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute() as RouteParams;
  const { session } = useSession();
  const queryClient = useQueryClient();
  const strategyId = route.params.strategyId;
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const [demoOrdersPage, setDemoOrdersPage] = useState(1);
  const [operationError, setOperationError] = useState<string | null>(null);
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const strategy = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId], queryFn: () => api.strategy(strategyId), enabled: Boolean(strategyId) });
  const account = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "account"], queryFn: () => api.strategyAccount(strategyId), enabled: Boolean(strategyId && strategy.data?.config.execution.environment !== "okx_demo") });
  const demo = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "demo-execution", demoOrdersPage], queryFn: () => api.strategyDemoExecution(strategyId, demoOrdersPage, 10), enabled: Boolean(strategyId && strategy.data?.config.execution.environment === "okx_demo"), retry: false });
  const evaluations = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "evaluations", 20], queryFn: () => api.strategyEvaluations(strategyId, 20), enabled: Boolean(strategyId) });
  const signals = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "signals", 20], queryFn: () => api.strategyLog(strategyId, "signals", 20), enabled: Boolean(strategyId) });
  const trades = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "trades", 20], queryFn: () => api.strategyLog(strategyId, "trades", 20), enabled: Boolean(strategyId) });
  const funding = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId, "funding", 20], queryFn: () => api.strategyLog(strategyId, "funding", 20), enabled: Boolean(strategyId) });
  const start = useMutation({ mutationFn: api.startStrategy });
  const stop = useMutation({ mutationFn: api.stopStrategy });
  const archive = useMutation({ mutationFn: api.archiveStrategy });
  const refresh = () => void Promise.all([strategy.refetch(), account.refetch(), demo.refetch(), evaluations.refetch(), signals.refetch(), trades.refetch(), funding.refetch(), access.refetch()]);

  if (strategy.isLoading) return <StatePanel description="正在读取策略配置、账户和运行诊断。" title="正在打开策略详情" />;
  if (strategy.isError || !strategy.data) return <StatePanel error={(strategy.error as Error)?.message ?? "找不到策略。"} onRetry={refresh} state="error" title="策略详情不可用" />;

  const item = strategy.data;
  const config = item.config;
  const isDemo = config.execution.environment === "okx_demo";
  const latestEvaluation = evaluations.data?.[0];
  const mayManage = canMutate(access.data, "strategy.manage");
  const gate = accessGate(access.data, "strategy.manage");
  const busy = start.isPending || stop.isPending || archive.isPending;
  const confirmationCopyValue = confirmation ? confirmationCopy(confirmation) : null;
  const paperPositions = Object.entries(account.data?.positions ?? {});
  const demoData = demo.data;
  const demoPositions = demoData?.positions.data.positions ?? [];
  const demoOrders = demoData?.orders ?? [];
  const activeIndicators = [
    config.moving_average.enabled ? "MA" : null,
    config.rsi.enabled ? "RSI" : null,
    config.bollinger.enabled ? "布林带" : null,
    config.momentum_macd.enabled ? "动量 / MACD" : null,
  ].filter(Boolean);

  const invalidateStrategyData = async (targetId: string) => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategies"] }),
      queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategy", targetId] }),
    ]);
  };
  const executeConfirmation = async () => {
    if (!confirmation || !session) return;
    const { action, strategy: target } = confirmation;
    try {
      const saved = action === "start"
        ? await start.mutateAsync(target.strategy_id)
        : action === "stop"
          ? await stop.mutateAsync(target.strategy_id)
          : await archive.mutateAsync(target.strategy_id);
      await invalidateStrategyData(saved.strategy_id);
      if (action === "archive") {
        const selectedId = await readActiveStrategyId(session.userId, session.tenantId);
        if (selectedId === saved.strategy_id) {
          await saveActiveStrategyId(session.userId, session.tenantId, "");
        }
        setConfirmation(null);
        navigation.navigate("StrategyList");
        return;
      }
      setOperationError(null);
      setConfirmation(null);
      void strategy.refetch();
    } catch (error) {
      setOperationError(error instanceof Error ? error.message : "服务未完成本次策略操作。");
    }
  };

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl onRefresh={refresh} refreshing={strategy.isRefetching || account.isRefetching || demo.isRefetching || evaluations.isRefetching} tintColor={palette.primary} />}
      style={styles.page}
    >
      <ScreenHeader actionLabel={!item.archived_at && mayManage && item.strategy_kind === "configurable_rule" ? "编辑" : undefined} onAction={!item.archived_at && mayManage && item.strategy_kind === "configurable_rule" ? () => navigation.navigate("StrategyEditor", { strategyId }) : undefined} subtitle={item.archived_at ? "已归档，只读历史与执行记录" : item.status === "running" ? `运行中 · ${executionEnvironmentLabel(config.execution.environment)}` : `已停止 · ${executionEnvironmentLabel(config.execution.environment)}`} title={item.name} />
      {!gate.mutationAllowed && access.data ? <StatePanel description={gate.message ?? "当前角色仅具备策略查看权限。"} title="策略操作受限" /> : null}
      {operationError ? <StatePanel actionLabel="关闭" description={operationError} onAction={() => setOperationError(null)} title="策略操作未完成" tone="error" /> : null}

      <SectionCard description={item.description ?? "未填写策略说明。"} title="策略配置">
        <View style={styles.configRows}>
          <Text style={styles.row}>观察标的：{config.symbols.join(" · ")}</Text>
          <Text style={styles.row}>评估周期：{config.interval} · 初始资金：{formatQuote(config.initial_capital_quote)}</Text>
          <Text style={styles.row}>执行环境：{executionEnvironmentLabel(config.execution.environment)} · 状态：{strategyStatusLabel(item.status, item.archived_at)}</Text>
          <Text style={styles.row}>风险：单笔 {formatQuote(config.risk.order_quote_amount)} · 最大持仓 {config.risk.max_positions} · 杠杆 {config.risk.leverage}×</Text>
          <Text style={styles.row}>基础指标：{activeIndicators.length ? activeIndicators.join(" · ") : "未启用"}</Text>
          <Text style={styles.row}>高级规则：{config.advanced_rules.enabled ? `已启用（入场${config.advanced_rules.entry_confirmation_mode === "all" ? "全部满足" : "任一满足"} · 退出${config.advanced_rules.exit_confirmation_mode === "all" ? "全部满足" : "任一满足"}）` : "未启用"}</Text>
        </View>
      </SectionCard>

      <StrategyExportPanel strategyId={strategyId} />

      {!item.archived_at ? <SectionCard description={mayManage ? "启停和归档会由服务端再次校验当前租户、状态与执行代际。" : "当前帐号仅可查看策略状态。"} title="策略操作">
        {mayManage ? <>
          <PrimaryButton label={item.status === "running" ? "停止策略" : "启动策略"} leading={item.status === "running" ? <Pause color={palette.canvas} size={19} /> : <CirclePlay color={palette.canvas} size={19} />} loading={busy} onPress={() => { setOperationError(null); setConfirmation({ action: item.status === "running" ? "stop" : "start", strategy: item }); }} />
          {item.status === "stopped" ? <DangerButton disabled={busy} label="归档策略" leading={<Archive color={palette.negative} size={19} />} loading={archive.isPending} onPress={() => { setOperationError(null); setConfirmation({ action: "archive", strategy: item }); }} /> : <Text style={styles.archiveHint}>策略运行中时不能归档。请先停止策略，再归档以保留完整历史。</Text>}
        </> : null}
      </SectionCard> : <SectionCard title="归档状态"><Text style={styles.muted}>归档策略保留配置、评估、成交、资金费和审计历史；移动端不会提供编辑、启动或恢复操作。</Text></SectionCard>}

      {isDemo ? (
        <>
          <SectionCard description={demoData ? `来源：${demoSourceLabel(demoData.source)} · 最近核验 ${formatTimestamp(demoData.checked_at)}` : "仅显示交易所权威数据，绝不回退到纸面账本。"} title="OKX Demo 执行来源">
            {demoData ? <>
              <Text style={styles.row}>交易所总估值：{formatQuote(demoData.account.data.total_usdt_value)}</Text>
              <Text style={styles.row}>持仓：{demoPositions.length} 项 · 策略归属订单：{demoData.pagination.total_items} 笔 · 已成交 {demoData.trade_summary?.filled_order_count ?? 0} 笔</Text>
              {demoData.pnl.status === "unavailable" ? <Text style={styles.muted}>{demoPnlReason(demoData.pnl.reason)}</Text> : <View style={styles.accountGrid}><View style={styles.accountMetric}><Text style={styles.accountLabel}>总 PnL</Text><Text style={styles.accountValue}>{formatNumericQuote(demoData.pnl.total_pnl ?? demoData.pnl.total ?? demoData.pnl.value)}</Text></View><View style={styles.accountMetric}><Text style={styles.accountLabel}>已实现</Text><Text style={styles.accountValue}>{formatNumericQuote(demoData.pnl.realized_pnl ?? demoData.pnl.realized)}</Text></View><View style={styles.accountMetric}><Text style={styles.accountLabel}>未实现</Text><Text style={styles.accountValue}>{formatNumericQuote(demoData.pnl.unrealized_pnl ?? demoData.pnl.unrealized)}</Text></View><View style={styles.accountMetric}><Text style={styles.accountLabel}>收益率</Text><Text style={styles.accountValue}>{demoData.pnl.return_pct == null ? "—" : `${demoData.pnl.return_pct.toFixed(2)}%`}</Text></View></View>}
              {demoPositions.length ? demoPositions.map((position) => <View key={position.symbol} style={styles.executionRow}><View style={styles.executionCopy}><Text style={styles.executionTitle}>{position.symbol}</Text><Text style={styles.muted}>数量 {position.quantity} · 可用 {position.available_quantity}</Text></View><Text style={styles.executionValue}>{formatQuote(position.notional_usdt)}</Text></View>) : <Text style={styles.muted}>交易所当前没有返回持仓。</Text>}
              {demoOrders.length ? <View style={styles.orderList}>{demoOrders.map((order) => <View key={order.id} style={styles.executionRow}><View style={styles.executionCopy}><Text style={styles.executionTitle}>{orderSideLabel(order.side)} · {order.symbol}</Text><Text style={styles.muted}>{orderTypeLabel(order.type)} · 请求 {formatNumericQuote(order.requested_quote)} · {formatTimestamp(order.updated_at)}</Text>{order.error_code ? <Text style={styles.warning}>{order.error_code}</Text> : null}</View><Text style={styles.executionValue}>{orderStatusLabel(order.status)}</Text></View>)}</View> : <Text style={styles.muted}>当前没有归因到该策略的交易所订单。</Text>}
              {demoData.pagination.total_pages > 1 ? <View style={styles.pagination}><Pressable accessibilityRole="button" disabled={demoOrdersPage <= 1} onPress={() => setDemoOrdersPage((page) => Math.max(1, page - 1))} style={[styles.pageButton, demoOrdersPage <= 1 && styles.disabled]}><Text style={styles.pageButtonText}>上一页</Text></Pressable><Pressable accessibilityRole="button" disabled={demoOrdersPage >= demoData.pagination.total_pages} onPress={() => setDemoOrdersPage((page) => Math.min(demoData.pagination.total_pages, page + 1))} style={[styles.pageButton, demoOrdersPage >= demoData.pagination.total_pages && styles.disabled]}><Text style={styles.pageButtonText}>下一页</Text></Pressable></View> : null}
            </> : <Text style={styles.muted}>正在等待交易所执行数据。</Text>}
          </SectionCard>
        </>
      ) : (
        <>
          {account.isError ? <StatePanel actionLabel="重试" description={(account.error as Error).message} onAction={() => void account.refetch()} title="纸面账户暂不可用" tone="error" /> : null}
          <SectionCard description="纸面账户、仓位和 PnL 都来自服务端账本。" title="纸面账户">
            {account.data ? <>
              <View style={styles.accountGrid}><View style={styles.accountMetric}><Text style={styles.accountLabel}>当前权益</Text><Text style={styles.accountValue}>{formatQuote(account.data.equity_quote)}</Text></View><View style={styles.accountMetric}><Text style={styles.accountLabel}>可用资金</Text><Text style={styles.accountValue}>{formatQuote(account.data.quote_balance)}</Text></View><View style={styles.accountMetric}><Text style={styles.accountLabel}>持仓价值</Text><Text style={styles.accountValue}>{formatQuote(paperPositionValue(account.data.positions))}</Text></View><View style={styles.accountMetric}><Text style={styles.accountLabel}>未实现 PnL</Text><Text style={styles.accountValue}>{formatQuote(account.data.unrealized_pnl_quote)}</Text></View></View>
              {paperPositions.length ? paperPositions.map(([symbol, position]) => <View key={symbol} style={styles.executionRow}><View style={styles.executionCopy}><Text style={styles.executionTitle}>{symbol}</Text><Text style={styles.muted}>数量 {position.quantity} · 标记价 {position.mark_price}</Text></View><Text style={styles.executionValue}>{formatQuote(position.quantity * position.mark_price)}</Text></View>) : <Text style={styles.muted}>纸面账本当前没有持仓。</Text>}
            </> : <Text style={styles.muted}>正在等待服务端纸面账户数据。</Text>}
          </SectionCard>
        </>
      )}

      {evaluations.isError ? <StatePanel actionLabel="重试" description={(evaluations.error as Error).message} onAction={() => void evaluations.refetch()} title="最近评估暂不可用" tone="error" /> : null}
      <SectionCard actionLabel="策略参考" description={latestEvaluation ? `最近评估：${formatTimestamp(latestEvaluation.evaluated_at)}` : "服务端尚无已保存的评估记录。"} onAction={() => navigation.navigate("StrategyAdvisory", { strategyId })} title="最新策略评估">
        {latestEvaluation ? <StrategyEvaluationPanel evaluation={latestEvaluation} /> : evaluations.isLoading ? <Text style={styles.muted}>正在读取服务端评估。</Text> : <Text style={styles.muted}>没有评估记录，移动端不会假定条件或执行状态。</Text>}
      </SectionCard>

      <SectionCard description="条件代码保持可核对，说明与状态按中文展示。" title="近期信号诊断">
        {signals.isError ? <Text style={styles.muted}>{(signals.error as Error).message}</Text> : signals.data?.entries.length ? signals.data.entries.slice(0, 10).map((signal) => <View key={`${signal.evaluation_id}-${signal.code}`} style={styles.signalRow}><View style={styles.signalHeader}><Text style={styles.signalTitle}>{conditionLabel(signal.code)}</Text><Text style={[styles.signalState, conditionStateTone(signal.state) === "positive" ? styles.signalPositive : conditionStateTone(signal.state) === "warning" ? styles.signalWarning : conditionStateTone(signal.state) === "negative" ? styles.signalNegative : styles.signalDefault]}>{conditionStateLabel(signal.state)}</Text></View><Text style={styles.signalCode}>code: {signal.code} · {formatTimestamp(signal.evaluated_at)}</Text><Text style={styles.signalDetail}>{conditionDetail(signal)}</Text></View>) : <Text style={styles.muted}>服务端尚无信号诊断记录。</Text>}
      </SectionCard>

      <SectionCard description="仅展示服务端归因的近期记录。" title="近期执行记录">
        {trades.isError || funding.isError ? <Text style={styles.muted}>{((trades.error ?? funding.error) as Error).message}</Text> : <>
          <Text style={styles.executionSectionTitle}>成交</Text>
          {trades.data?.entries.length ? trades.data.entries.slice(0, 5).map((trade) => <View key={trade.evaluation_id} style={styles.executionRow}><View style={styles.executionCopy}><Text style={styles.executionTitle}>{strategyActionLabel(trade.action)} · {trade.symbol}</Text><Text style={styles.muted}>{evaluationReason(trade.reason_code, trade.reason)}</Text></View><Text style={styles.executionValue}>{formatQuote(trade.quote_amount)}</Text></View>) : <Text style={styles.muted}>尚无归因成交。</Text>}
          <Text style={styles.executionSectionTitle}>资金费</Text>
          {funding.data?.entries.length ? funding.data.entries.slice(0, 5).map((entry) => <View key={entry.evaluation_id} style={styles.executionRow}><View style={styles.executionCopy}><Text style={styles.executionTitle}>{fundingDirectionLabel(entry.direction)}</Text><Text style={styles.muted}>当前名义金额 {formatQuote(entry.current_notional_quote)}</Text></View><Text style={styles.executionValue}>{formatQuote(entry.estimated_payment_quote)}</Text></View>) : <Text style={styles.muted}>尚无资金费影响。</Text>}
        </>}
      </SectionCard>

      <ListRow leading={<Wallet color={palette.primary} size={20} />} onPress={() => navigation.navigate("工作台", { screen: "StrategyOverview", params: { strategyId } })} subtitle="带入该策略查看账户、仓位、条件与执行漏斗" title="在工作台查看策略" />
      <ListRow leading={<ChartCandlestick color={palette.primary} size={20} />} onPress={() => navigation.navigate("行情", { screen: "Market", params: { strategyId } })} subtitle="使用该策略的观察交易对查看行情" title="打开策略行情" />
      <ListRow leading={<Bot color={palette.primary} size={20} />} onPress={() => navigation.navigate("StrategyAdvisory", { strategyId })} subtitle="仅供策略参考，不会自动执行或修改策略" title="打开策略参考" />

      <ConfirmSheet confirming={busy} confirmLabel={confirmationCopyValue?.label} destructive={confirmationCopyValue?.destructive} message={confirmationCopyValue?.message ?? ""} onCancel={() => { if (!busy) setConfirmation(null); }} onConfirm={() => void executeConfirmation()} title={confirmationCopyValue?.title ?? "确认策略操作"} visible={Boolean(confirmation)} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  configRows: { gap: spacing.xs },
  row: { color: palette.text, fontSize: 14, lineHeight: 22 },
  muted: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  archiveHint: { color: palette.warning, fontSize: 13, fontWeight: "800", lineHeight: 20 },
  accountGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  accountMetric: { backgroundColor: palette.surfaceRaised, borderRadius: radius.sm, flexBasis: "47%", flexGrow: 1, gap: spacing.xxs, minWidth: 136, padding: spacing.sm },
  accountLabel: { color: palette.textMuted, fontSize: 11, fontWeight: "800" },
  accountValue: { color: palette.text, fontSize: 15, fontWeight: "900" },
  executionRow: { alignItems: "center", borderTopColor: palette.border, borderTopWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 54, paddingVertical: spacing.xs },
  executionCopy: { flex: 1, gap: 2 },
  executionTitle: { color: palette.text, fontSize: 14, fontWeight: "800" },
  executionValue: { color: palette.text, fontSize: 13, fontWeight: "900", textAlign: "right" },
  orderList: { gap: 0 },
  signalRow: { borderTopColor: palette.border, borderTopWidth: 1, gap: spacing.xs, paddingVertical: spacing.sm },
  signalHeader: { alignItems: "center", flexDirection: "row", gap: spacing.sm },
  signalTitle: { color: palette.text, flex: 1, fontSize: 14, fontWeight: "900" },
  signalState: { borderRadius: radius.pill, fontSize: 11, fontWeight: "900", overflow: "hidden", paddingHorizontal: spacing.xs, paddingVertical: spacing.xxs },
  signalPositive: { backgroundColor: palette.positiveSoft, color: palette.positive },
  signalWarning: { backgroundColor: palette.warningSoft, color: palette.warning },
  signalNegative: { backgroundColor: palette.negativeSoft, color: palette.negative },
  signalDefault: { backgroundColor: palette.surfaceRaised, color: palette.textMuted },
  signalCode: { color: palette.textMuted, fontSize: 11, fontWeight: "700" },
  signalDetail: { color: palette.text, fontSize: 13, lineHeight: 20 },
  pagination: { flexDirection: "row", gap: spacing.sm, justifyContent: "flex-end", paddingTop: spacing.sm },
  pageButton: { borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, minHeight: 36, justifyContent: "center", paddingHorizontal: spacing.sm },
  pageButtonText: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  warning: { color: palette.warning, fontSize: 12, fontWeight: "800" },
  disabled: { opacity: 0.5 },
  executionSectionTitle: { color: palette.primary, fontSize: 13, fontWeight: "900", marginTop: spacing.xs },
});
