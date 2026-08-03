import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pressable, RefreshControl, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { useNavigation } from "@react-navigation/native";
import { Archive, Bookmark, CirclePlay, Pause, Plus, Search } from "lucide-react-native";
import { api } from "../api";
import { accessGate, canMutate } from "../access";
import { ConfirmSheet, DangerButton, PrimaryButton, ScreenHeader, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import type { Strategy } from "../types";
import {
  executionEnvironmentLabel,
  strategyStatusLabel,
  strategyStatusTone,
} from "./strategy-presentation";
import {
  readActiveStrategyId,
  saveActiveStrategyId,
  selectActiveStrategyId,
} from "./workbench";

type Filter = "all" | "running" | "stopped" | "archived";
type LifecycleAction = "start" | "stop" | "archive";
type Confirmation = { action: LifecycleAction; strategy: Strategy };

const FILTER_LABELS: Readonly<Record<Filter, string>> = {
  all: "全部",
  archived: "已归档",
  running: "运行中",
  stopped: "已停止",
};

const FILTERS: readonly Filter[] = ["all", "running", "stopped", "archived"];

function confirmationCopy(confirmation: Confirmation): { title: string; message: string; label: string; destructive: boolean } {
  const environment = executionEnvironmentLabel(confirmation.strategy.config.execution.environment);
  if (confirmation.action === "start") {
    return {
      title: "启动策略",
      message: `确认以${environment}启动“${confirmation.strategy.name}”？服务端会按当前配置进行后续评估与执行。`,
      label: "启动策略",
      destructive: false,
    };
  }
  if (confirmation.action === "stop") {
    return {
      title: "停止策略",
      message: `确认停止“${confirmation.strategy.name}”？服务端将停止提交新的策略执行。`,
      label: "停止策略",
      destructive: true,
    };
  }
  return {
    title: "归档策略",
    message: `确认归档“${confirmation.strategy.name}”？归档会保留评估、成交、资金费与审计历史。策略必须已停止，且不能恢复。`,
    label: "归档策略",
    destructive: true,
  };
}

export default function StrategyListScreen() {
  const navigation = useNavigation<any>();
  const { session } = useSession();
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [storedActiveId, setStoredActiveId] = useState("");
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const [operationError, setOperationError] = useState<string | null>(null);
  const [operationNotice, setOperationNotice] = useState<string | null>(null);
  const activeStrategies = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategies", false],
    queryFn: () => api.strategies(false),
    enabled: Boolean(session),
  });
  const archivedStrategies = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategies", true],
    queryFn: () => api.strategies(true),
    enabled: Boolean(session && filter === "archived"),
  });
  const access = useQuery({
    queryKey: ["mobile", session?.tenantId, "access"],
    queryFn: api.access,
    enabled: Boolean(session),
  });
  const start = useMutation({ mutationFn: api.startStrategy });
  const stop = useMutation({ mutationFn: api.stopStrategy });
  const archive = useMutation({ mutationFn: api.archiveStrategy });

  useEffect(() => {
    if (!session) return;
    let active = true;
    void readActiveStrategyId(session.userId, session.tenantId).then((strategyId) => {
      if (active) setStoredActiveId(strategyId);
    });
    return () => {
      active = false;
    };
  }, [session?.tenantId, session?.userId]);

  const visibleQuery = filter === "archived" ? archivedStrategies : activeStrategies;
  const strategies = visibleQuery.data ?? [];
  const activeStrategyId = useMemo(
    () => selectActiveStrategyId(activeStrategies.data ?? [], storedActiveId),
    [activeStrategies.data, storedActiveId],
  );
  const rows = useMemo(() => {
    const normalizedSearch = search.trim().toLocaleLowerCase();
    return strategies.filter((strategy) => {
      if (filter === "archived" ? !strategy.archived_at : strategy.archived_at) return false;
      if (filter === "running" && strategy.status !== "running") return false;
      if (filter === "stopped" && strategy.status !== "stopped") return false;
      if (!normalizedSearch) return true;
      return strategy.name.toLocaleLowerCase().includes(normalizedSearch)
        || strategy.config.symbols.some((symbol) => symbol.toLocaleLowerCase().includes(normalizedSearch));
    });
  }, [filter, search, strategies]);
  const canManage = canMutate(access.data, "strategy.manage");
  const gate = accessGate(access.data, "strategy.manage");
  const busy = start.isPending || stop.isPending || archive.isPending;

  const invalidateStrategyData = async (strategyId: string) => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategies"] }),
      queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId] }),
    ]);
  };
  const selectForWorkbench = (strategyId: string) => {
    if (!session) return;
    setStoredActiveId(strategyId);
    void saveActiveStrategyId(session.userId, session.tenantId, strategyId);
    setOperationNotice("已设为工作台策略。工作台会在下次打开时使用该策略。");
  };
  const executeConfirmation = async () => {
    if (!confirmation || !session) return;
    const { action, strategy } = confirmation;
    try {
      const saved = action === "start"
        ? await start.mutateAsync(strategy.strategy_id)
        : action === "stop"
          ? await stop.mutateAsync(strategy.strategy_id)
          : await archive.mutateAsync(strategy.strategy_id);
      await invalidateStrategyData(saved.strategy_id);
      if (action === "archive" && storedActiveId === saved.strategy_id) {
        setStoredActiveId("");
        await saveActiveStrategyId(session.userId, session.tenantId, "");
      }
      setOperationNotice(
        action === "start"
          ? "策略已请求启动，工作台会显示服务端最新状态。"
          : action === "stop"
            ? "策略已请求停止，工作台会显示服务端最新状态。"
            : "策略已归档，并已从当前工作台选择中移除。",
      );
      setOperationError(null);
      setConfirmation(null);
    } catch (error) {
      setOperationError(error instanceof Error ? error.message : "服务未完成本次策略操作。");
    }
  };

  const confirmationCopyValue = confirmation ? confirmationCopy(confirmation) : null;

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl onRefresh={() => void Promise.all([activeStrategies.refetch(), archivedStrategies.refetch(), access.refetch()])} refreshing={activeStrategies.isRefetching || archivedStrategies.isRefetching || access.isRefetching} tintColor={palette.primary} />}
      style={styles.page}
    >
      <ScreenHeader subtitle="选择工作台策略，并按服务端权限管理启停与归档。" title="策略" />
      {canManage ? <PrimaryButton label="新建策略" leading={<Plus color={palette.canvas} size={19} />} onPress={() => navigation.navigate("StrategyEditor")} /> : null}
      {!gate.mutationAllowed && access.data ? <StatePanel description={gate.message ?? "当前角色仅具备策略查看权限。"} title="策略操作受限" /> : null}
      {operationNotice ? <View style={styles.notice}><Text style={styles.noticeText}>{operationNotice}</Text></View> : null}
      {operationError ? <StatePanel actionLabel="关闭" description={operationError} onAction={() => setOperationError(null)} title="策略操作未完成" tone="error" /> : null}

      <View style={styles.search}>
        <Search color={palette.textMuted} size={18} />
        <TextInput accessibilityLabel="搜索策略" onChangeText={setSearch} placeholder="按名称或交易对搜索" placeholderTextColor={palette.textMuted} style={styles.searchInput} value={search} />
      </View>
      <ScrollView contentContainerStyle={styles.filters} horizontal showsHorizontalScrollIndicator={false}>
        {FILTERS.map((value) => (
          <Pressable
            accessibilityLabel={`筛选${FILTER_LABELS[value]}策略`}
            accessibilityRole="button"
            key={value}
            onPress={() => setFilter(value)}
            style={({ pressed }) => [styles.filter, filter === value && styles.filterActive, pressed && styles.pressed]}
          >
            <Text style={[styles.filterText, filter === value && styles.filterTextActive]}>{FILTER_LABELS[value]}</Text>
          </Pressable>
        ))}
      </ScrollView>

      {visibleQuery.isLoading ? <StatePanel description="正在读取当前租户的策略。" title="正在同步" /> : null}
      {visibleQuery.isError ? <StatePanel actionLabel="重试" description={(visibleQuery.error as Error).message} onAction={() => void visibleQuery.refetch()} title="策略列表暂不可用" tone="error" /> : null}
      {!visibleQuery.isLoading && !visibleQuery.isError && rows.length === 0 ? <StatePanel actionLabel={canManage && filter !== "archived" ? "创建策略" : undefined} description={filter === "archived" ? "没有归档策略。归档历史仍可在此查看。" : "新策略会使用完整的风险、执行与指标配置。"} onAction={canManage && filter !== "archived" ? () => navigation.navigate("StrategyEditor") : undefined} title="没有匹配的策略" /> : null}

      {rows.map((strategy) => {
        const isWorkbenchStrategy = strategy.strategy_id === activeStrategyId;
        const isRunning = strategy.status === "running";
        const isArchived = Boolean(strategy.archived_at);
        return (
          <View key={strategy.strategy_id} style={styles.card}>
            <Pressable
              accessibilityLabel={`查看策略 ${strategy.name}`}
              accessibilityRole="button"
              onPress={() => navigation.navigate("StrategyDetail", { strategyId: strategy.strategy_id })}
              style={({ pressed }) => [styles.cardPress, pressed && styles.pressed]}
            >
              <View style={styles.cardHead}>
                <View style={styles.cardCopy}>
                  <Text numberOfLines={1} style={styles.name}>{strategy.name}</Text>
                  <Text numberOfLines={2} style={styles.meta}>{strategy.config.symbols.join(" · ")}</Text>
                </View>
                <View style={[styles.statusBadge, strategyStatusTone(strategy.status, strategy.archived_at) === "positive" ? styles.statusPositive : strategyStatusTone(strategy.status, strategy.archived_at) === "warning" ? styles.statusWarning : styles.statusDefault]}>
                  <Text style={[styles.statusText, strategyStatusTone(strategy.status, strategy.archived_at) === "positive" ? styles.statusPositiveText : strategyStatusTone(strategy.status, strategy.archived_at) === "warning" ? styles.statusWarningText : styles.statusDefaultText]}>{strategyStatusLabel(strategy.status, strategy.archived_at)}</Text>
                </View>
              </View>
              <Text style={styles.meta}>{executionEnvironmentLabel(strategy.config.execution.environment)} · {strategy.config.interval} · 单笔 {strategy.config.risk.order_quote_amount.toLocaleString()} USDT</Text>
              {isWorkbenchStrategy ? <View style={styles.workbenchMark}><Bookmark color={palette.primary} size={14} /><Text style={styles.workbenchMarkText}>当前工作台策略</Text></View> : null}
            </Pressable>

            {!isArchived ? (
              <View style={styles.actions}>
                {isWorkbenchStrategy ? <View style={styles.currentAction}><Bookmark color={palette.primary} size={18} /><Text style={styles.currentActionText}>当前工作台</Text></View> : <Pressable accessibilityLabel={`设定 ${strategy.name} 为工作台策略`} accessibilityRole="button" disabled={busy} onPress={() => selectForWorkbench(strategy.strategy_id)} style={({ pressed }) => [styles.secondaryAction, busy && styles.disabled, pressed && !busy && styles.pressed]}><Bookmark color={palette.primary} size={18} /><Text style={styles.secondaryActionText}>设为工作台</Text></Pressable>}
                {canManage ? <Pressable accessibilityLabel={`${isRunning ? "停止" : "启动"}策略 ${strategy.name}`} accessibilityRole="button" disabled={busy} onPress={() => { setOperationError(null); setConfirmation({ action: isRunning ? "stop" : "start", strategy }); }} style={({ pressed }) => [styles.lifecycleAction, isRunning && styles.stopAction, busy && styles.disabled, pressed && !busy && styles.pressed]}>{isRunning ? <Pause color={palette.warning} size={18} /> : <CirclePlay color={palette.canvas} size={18} />}<Text style={[styles.lifecycleText, isRunning && styles.stopActionText]}>{isRunning ? "停止策略" : "启动策略"}</Text></Pressable> : null}
                {canManage && !isRunning ? <DangerButton disabled={busy} fullWidth label="归档策略" leading={<Archive color={palette.negative} size={18} />} onPress={() => { setOperationError(null); setConfirmation({ action: "archive", strategy }); }} /> : null}
                {canManage && isRunning ? <Text style={styles.archiveHint}>需先停止策略，才能归档。</Text> : null}
              </View>
            ) : <View style={styles.archivedHint}><Archive color={palette.textMuted} size={17} /><Text style={styles.archivedHintText}>归档策略仅保留查看与历史访问，不可编辑、启动或恢复。</Text></View>}
          </View>
        );
      })}

      <ConfirmSheet
        confirming={busy}
        confirmLabel={confirmationCopyValue?.label}
        destructive={confirmationCopyValue?.destructive}
        message={confirmationCopyValue?.message ?? ""}
        onCancel={() => { if (!busy) setConfirmation(null); }}
        onConfirm={() => void executeConfirmation()}
        title={confirmationCopyValue?.title ?? "确认策略操作"}
        visible={Boolean(confirmation)}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  notice: { backgroundColor: palette.positiveSoft, borderColor: palette.positive, borderRadius: radius.sm, borderWidth: 1, padding: spacing.sm },
  noticeText: { color: palette.positive, fontSize: 13, fontWeight: "800", lineHeight: 19 },
  search: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 48, paddingHorizontal: spacing.sm },
  searchInput: { color: palette.text, flex: 1, fontSize: 16, minHeight: 44 },
  filters: { gap: spacing.xs },
  filter: { borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.md },
  filterActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  filterText: { color: palette.textMuted, fontSize: 13, fontWeight: "800" },
  filterTextActive: { color: palette.primary },
  card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, overflow: "hidden", padding: spacing.sm },
  cardPress: { gap: spacing.xs, minHeight: 82, padding: spacing.xs },
  cardHead: { alignItems: "flex-start", flexDirection: "row", gap: spacing.sm },
  cardCopy: { flex: 1, gap: spacing.xxs },
  name: { color: palette.text, fontSize: 17, fontWeight: "900" },
  meta: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  statusBadge: { borderRadius: radius.pill, paddingHorizontal: spacing.xs, paddingVertical: spacing.xxs },
  statusPositive: { backgroundColor: palette.positiveSoft },
  statusWarning: { backgroundColor: palette.warningSoft },
  statusDefault: { backgroundColor: palette.surfaceRaised },
  statusText: { fontSize: 11, fontWeight: "900" },
  statusPositiveText: { color: palette.positive },
  statusWarningText: { color: palette.warning },
  statusDefaultText: { color: palette.textMuted },
  workbenchMark: { alignItems: "center", flexDirection: "row", gap: spacing.xxs },
  workbenchMarkText: { color: palette.primary, fontSize: 12, fontWeight: "900" },
  actions: { gap: spacing.xs },
  secondaryAction: { alignItems: "center", backgroundColor: palette.primarySoft, borderColor: palette.primary, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md },
  secondaryActionText: { color: palette.primary, fontSize: 15, fontWeight: "900" },
  currentAction: { alignItems: "center", backgroundColor: palette.surfaceRaised, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md },
  currentActionText: { color: palette.primary, fontSize: 15, fontWeight: "900" },
  lifecycleAction: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md },
  lifecycleText: { color: palette.canvas, fontSize: 15, fontWeight: "900" },
  stopAction: { backgroundColor: palette.warningSoft, borderColor: palette.warning, borderWidth: 1 },
  stopActionText: { color: palette.warning },
  archiveHint: { color: palette.textMuted, fontSize: 12, lineHeight: 18, paddingHorizontal: spacing.xs },
  archivedHint: { alignItems: "flex-start", backgroundColor: palette.surfaceRaised, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xs, padding: spacing.sm },
  archivedHintText: { color: palette.textMuted, flex: 1, fontSize: 12, lineHeight: 18 },
  disabled: { opacity: 0.48 },
  pressed: { opacity: 0.76 },
});
