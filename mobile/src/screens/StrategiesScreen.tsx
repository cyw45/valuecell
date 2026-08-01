import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { CirclePlay, Pause, Plus, Save, ShieldAlert, SlidersHorizontal } from "lucide-react-native";
import { api } from "../api";
import type { ExecutionEnvironment, Strategy } from "../types";
import { palette, radius, spacing } from "../theme";

const intervals = ["1m", "5m", "15m", "1h", "4h", "1d"] as const;

function defaultConfig() {
  return {
    mode: "paper",
    initial_capital_quote: 10000,
    confirmation_mode: "all",
    symbols: ["BTC-USDT"],
    interval: "1h",
    execution: { environment: "paper", max_order_quote_amount: 100, max_daily_quote_amount: 500, max_total_quote_amount: 1000 },
    risk: { order_quote_amount: 100, max_positions: 3, leverage: 1 },
  };
}

function NumericInput({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return <View style={styles.field}><Text style={styles.fieldLabel}>{label}</Text><TextInput accessibilityLabel={label} keyboardType="decimal-pad" onChangeText={(text) => onChange(Number(text) || 0)} style={styles.input} value={String(value)} /></View>;
}

export default function StrategiesScreen() {
  const queryClient = useQueryClient();
  const strategies = useQuery({ queryKey: ["mobile", "strategies"], queryFn: api.strategies });
  const demos = useQuery({ queryKey: ["mobile", "demo-connections"], queryFn: api.demoConnections });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = strategies.data?.find((strategy) => strategy.strategy_id === selectedId) ?? strategies.data?.[0];
  const [draft, setDraft] = useState<Strategy | null>(null);
  const [symbolInput, setSymbolInput] = useState("");

  useEffect(() => { if (selected) setDraft(structuredClone(selected)); }, [selected]);
  const invalidate = () => void queryClient.invalidateQueries({ queryKey: ["mobile", "strategies"] });
  const create = useMutation({ mutationFn: () => api.createStrategy({ name: "移动端策略", initial_capital_quote: 10000, config: defaultConfig() }), onSuccess: (strategy) => { setSelectedId(strategy.strategy_id); invalidate(); } });
  const update = useMutation({ mutationFn: (strategy: Strategy) => api.updateStrategy(strategy.strategy_id, { name: strategy.name, description: strategy.description, config: strategy.config }), onSuccess: invalidate });
  const start = useMutation({ mutationFn: (strategyId: string) => api.startStrategy(strategyId), onSuccess: invalidate });
  const stop = useMutation({ mutationFn: (strategyId: string) => api.stopStrategy(strategyId), onSuccess: invalidate });
  const isBusy = create.isPending || update.isPending || start.isPending || stop.isPending;
  const demoConnections = useMemo(() => demos.data?.filter((connection) => connection.provider === "okx" && connection.metadata.sandbox && connection.metadata.market_type === "spot" && !connection.revoked) ?? [], [demos.data]);

  function changeConfig(key: string, value: unknown) {
    setDraft((current) => current ? { ...current, config: { ...current.config, [key]: value } } : current);
  }
  function changeRisk(key: string, value: number) {
    if (!draft) return;
    changeConfig("risk", { ...draft.config.risk, [key]: value });
  }
  function changeExecution(key: string, value: string | number) {
    if (!draft) return;
    changeConfig("execution", { ...draft.config.execution, [key]: value });
  }
  function addSymbol() {
    const symbol = symbolInput.trim().toUpperCase().replace("/", "-");
    if (!draft || !symbol || draft.config.symbols.includes(symbol)) return;
    if (!symbol.endsWith("-USDT")) { Alert.alert("交易对格式", "仅支持 USDT 交易对，例如 BTC-USDT。"); return; }
    changeConfig("symbols", [...draft.config.symbols, symbol]);
    setSymbolInput("");
  }
  function changeEnvironment(environment: ExecutionEnvironment) {
    if (!draft || draft.status === "running") return;
    changeExecution("environment", environment);
    if (environment === "okx_demo") changeRisk("leverage", 1);
  }
  function save() {
    if (!draft) return;
    const execution = draft.config.execution;
    if (execution.environment === "okx_demo" && !execution.sandbox_connection_id) { Alert.alert("缺少 OKX Demo 连接", "请先选择一条已验证的 OKX Demo 现货连接。"); return; }
    if (execution.max_order_quote_amount > execution.max_daily_quote_amount || execution.max_order_quote_amount > execution.max_total_quote_amount) { Alert.alert("额度配置无效", "每日额度和策略总额度都不得低于单笔额度。"); return; }
    void update.mutateAsync(draft).catch((reason) => Alert.alert("保存失败", reason instanceof Error ? reason.message : "请稍后重试。"));
  }
  function toggle() {
    if (!selected) return;
    const mutation = selected.status === "running" ? stop : start;
    const action = selected.status === "running" ? "停止" : "启动";
    Alert.alert(`${action}策略`, `${action}后策略将${action === "启动" ? "按当前执行环境运行" : "停止提交新的策略订单"}。`, [{ text: "取消", style: "cancel" }, { text: action, onPress: () => void mutation.mutateAsync(selected.strategy_id).catch((reason) => Alert.alert(`${action}失败`, reason instanceof Error ? reason.message : "请稍后重试。")) }]);
  }

  return <ScrollView contentContainerStyle={styles.content} style={styles.page}>
    <Text style={styles.eyebrow}>STRATEGY OPERATIONS</Text><Text style={styles.title}>策略配置</Text><Text style={styles.subtitle}>在移动端查看、更新并安全启停现有策略。</Text>
    <ScrollView contentContainerStyle={styles.strategyRow} horizontal showsHorizontalScrollIndicator={false}>{strategies.data?.map((strategy) => <Pressable accessibilityRole="button" key={strategy.strategy_id} onPress={() => setSelectedId(strategy.strategy_id)} style={[styles.strategyChip, selected?.strategy_id === strategy.strategy_id && styles.strategyChipActive]}><Text numberOfLines={1} style={[styles.strategyName, selected?.strategy_id === strategy.strategy_id && styles.strategyNameActive]}>{strategy.name}</Text><Text style={[styles.strategyStatus, { color: strategy.status === "running" ? palette.positive : palette.textMuted }]}>{strategy.status === "running" ? "运行中" : "已停止"}</Text></Pressable>)}</ScrollView>
    {strategies.isLoading ? <Text style={{ color: palette.textMuted, fontSize: 13 }}>正在同步策略列表…</Text> : null}
    {strategies.isError ? <Text style={{ color: palette.negative, fontSize: 13 }}>策略列表加载失败。请检查网络后重试。</Text> : null}
    {!strategies.isLoading && !selected ? <Pressable accessibilityRole="button" onPress={() => void create.mutateAsync().catch((reason) => Alert.alert("创建失败", reason instanceof Error ? reason.message : "请稍后重试。"))} style={styles.create}><Plus color={palette.canvas} size={18} /><Text style={styles.createText}>创建第一条纸面策略</Text></Pressable> : null}
    {draft ? <>
      <View style={styles.card}><View style={styles.cardTitleRow}><SlidersHorizontal color={palette.primary} size={19} /><Text style={styles.cardTitle}>基础配置</Text></View><View style={styles.field}><Text style={styles.fieldLabel}>策略名称</Text><TextInput accessibilityLabel="策略名称" onChangeText={(name) => setDraft({ ...draft, name })} style={styles.input} value={draft.name} /></View><View style={styles.field}><Text style={styles.fieldLabel}>说明</Text><TextInput accessibilityLabel="策略说明" multiline onChangeText={(description) => setDraft({ ...draft, description })} style={[styles.input, styles.multiline]} value={draft.description ?? ""} /></View><NumericInput label="初始纸面资金（USDT）" onChange={(value) => changeConfig("initial_capital_quote", value)} value={draft.config.initial_capital_quote} /><NumericInput label="单笔开仓金额（USDT）" onChange={(value) => changeRisk("order_quote_amount", value)} value={draft.config.risk.order_quote_amount} /><NumericInput label="最大持仓数" onChange={(value) => changeRisk("max_positions", value)} value={draft.config.risk.max_positions} /></View>
      <View style={styles.card}><Text style={styles.cardTitle}>观察币种 · {draft.config.symbols.length}</Text><View style={styles.symbols}>{draft.config.symbols.map((symbol) => <Pressable accessibilityRole="button" key={symbol} onPress={() => changeConfig("symbols", draft.config.symbols.filter((item) => item !== symbol))} style={styles.symbol}><Text style={styles.symbolText}>{symbol.replace("-", "/")} ×</Text></Pressable>)}</View><View style={styles.addSymbol}><TextInput accessibilityLabel="添加观察币种" autoCapitalize="characters" onChangeText={setSymbolInput} placeholder="例如 BTC-USDT" placeholderTextColor={palette.textMuted} style={[styles.input, styles.addInput]} value={symbolInput} /><Pressable accessibilityRole="button" onPress={addSymbol} style={styles.addButton}><Plus color={palette.canvas} size={17} /></Pressable></View><Text style={styles.help}>输入 USDT 交易对；行情页可切换查看每一个观察币种的 K 线。</Text></View>
      <View style={styles.card}><Text style={styles.cardTitle}>执行环境</Text><View style={styles.environmentRow}>{(["paper", "okx_demo"] as const).map((environment) => <Pressable accessibilityRole="button" disabled={draft.status === "running"} key={environment} onPress={() => changeEnvironment(environment)} style={[styles.environment, draft.config.execution.environment === environment && styles.environmentActive, draft.status === "running" && styles.disabled]}><Text style={[styles.environmentTitle, draft.config.execution.environment === environment && styles.environmentTitleActive]}>{environment === "paper" ? "纸面交易" : "OKX Demo"}</Text><Text style={styles.environmentCopy}>{environment === "paper" ? "仅写入模拟账本" : "向模拟盘自动下单"}</Text></Pressable>)}</View>{draft.status === "running" ? <Text style={styles.warning}>策略运行中。请先停止，再切换执行环境或 Demo 连接。</Text> : null}{draft.config.execution.environment === "okx_demo" ? <><Text style={styles.fieldLabel}>已验证 OKX Demo 现货连接</Text><View style={styles.symbols}>{demoConnections.map((connection) => <Pressable accessibilityRole="button" key={connection.id} onPress={() => changeExecution("sandbox_connection_id", connection.id)} style={[styles.symbol, draft.config.execution.sandbox_connection_id === connection.id && styles.symbolSelected]}><Text style={styles.symbolText}>{connection.label}</Text></Pressable>)}</View>{demoConnections.length === 0 ? <Text style={styles.warning}>尚无已验证的 OKX Demo 现货连接。请先在桌面端“模拟交易所”完成连接验证。</Text> : null}<NumericInput label="Demo 单笔额度（USDT）" onChange={(value) => changeExecution("max_order_quote_amount", value)} value={draft.config.execution.max_order_quote_amount} /><NumericInput label="Demo 每日额度（USDT）" onChange={(value) => changeExecution("max_daily_quote_amount", value)} value={draft.config.execution.max_daily_quote_amount} /><NumericInput label="Demo 策略总额度（USDT）" onChange={(value) => changeExecution("max_total_quote_amount", value)} value={draft.config.execution.max_total_quote_amount} /></> : null}<View style={styles.realNotice}><ShieldAlert color={palette.warning} size={18} /><Text style={styles.realNoticeText}>实盘不会在移动端自动开启。仍需在桌面端独立完成实盘连接、风控策略、绑定与人工启动授权。</Text></View></View>
      <View style={styles.actions}><Pressable accessibilityRole="button" disabled={isBusy} onPress={save} style={[styles.save, isBusy && styles.disabled]}><Save color={palette.canvas} size={18} /><Text style={styles.saveText}>保存策略配置</Text></Pressable><Pressable accessibilityRole="button" disabled={isBusy} onPress={toggle} style={[styles.toggle, selected?.status === "running" && styles.stop]}>{selected?.status === "running" ? <Pause color={palette.negative} size={18} /> : <CirclePlay color={palette.positive} size={18} />}<Text style={[styles.toggleText, { color: selected?.status === "running" ? palette.negative : palette.positive }]}>{selected?.status === "running" ? "停止策略" : "启动策略"}</Text></Pressable></View>
    </> : null}
  </ScrollView>;
}

const styles = StyleSheet.create({ page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, eyebrow: { color: palette.primary, fontSize: 10, fontWeight: "800", letterSpacing: 1.2, marginTop: spacing.sm }, title: { color: palette.text, fontSize: 27, fontWeight: "800", letterSpacing: -0.8 }, subtitle: { color: palette.textMuted, fontSize: 13, marginTop: -spacing.sm }, strategyRow: { gap: spacing.xs }, strategyChip: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, maxWidth: 150, padding: spacing.sm }, strategyChipActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, strategyName: { color: palette.text, fontSize: 13, fontWeight: "800" }, strategyNameActive: { color: palette.primary }, strategyStatus: { fontSize: 11, fontWeight: "700", marginTop: 4 }, create: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.md, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 52 }, createText: { color: palette.canvas, fontSize: 15, fontWeight: "800" }, card: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.md, padding: spacing.md }, cardTitleRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs }, cardTitle: { color: palette.text, fontSize: 15, fontWeight: "800" }, field: { gap: 6 }, fieldLabel: { color: palette.textMuted, fontSize: 12, fontWeight: "700" }, input: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, fontSize: 14, minHeight: 44, paddingHorizontal: spacing.sm }, multiline: { minHeight: 70, paddingTop: spacing.sm, textAlignVertical: "top" }, symbols: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs }, symbol: { backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, paddingHorizontal: 10, paddingVertical: 7 }, symbolSelected: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, symbolText: { color: palette.text, fontSize: 12, fontWeight: "700" }, addSymbol: { flexDirection: "row", gap: spacing.xs }, addInput: { flex: 1 }, addButton: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, justifyContent: "center", width: 46 }, help: { color: palette.textMuted, fontSize: 11, lineHeight: 16 }, environmentRow: { flexDirection: "row", gap: spacing.sm }, environment: { borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flex: 1, gap: 4, padding: spacing.sm }, environmentActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, environmentTitle: { color: palette.text, fontSize: 13, fontWeight: "800" }, environmentTitleActive: { color: palette.primary }, environmentCopy: { color: palette.textMuted, fontSize: 11, lineHeight: 15 }, warning: { color: palette.warning, fontSize: 12, lineHeight: 18 }, realNotice: { backgroundColor: palette.warningSoft, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xs, padding: spacing.sm }, realNoticeText: { color: palette.textMuted, flex: 1, fontSize: 12, lineHeight: 18 }, actions: { gap: spacing.sm }, save: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.md, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 50 }, saveText: { color: palette.canvas, fontSize: 14, fontWeight: "800" }, toggle: { alignItems: "center", borderColor: palette.positive, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 50 }, stop: { borderColor: palette.negative }, toggleText: { fontSize: 14, fontWeight: "800" }, disabled: { opacity: 0.5 },
});
