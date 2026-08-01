import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Modal, Pressable, ScrollView, StyleSheet, Switch, Text, TextInput, View } from "react-native";
import { useNavigation, useRoute } from "@react-navigation/native";
import { Bot, Plus, Save, X } from "lucide-react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { BottomSheetSelector, ScreenHeader, SectionCard, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import type { StrategyConfig } from "../types";
import { createStrategyConfig, normalizeSymbols, STRATEGY_INTERVALS, validOkxDemoConnections, validateStrategyConfig } from "./strategy-form";

type RouteParams = { params?: { strategyId?: string } };
type TextProposal = { strategy_name?: string | null; config?: Partial<StrategyConfig>; summary?: string; unresolved_items?: string[] };

function NumberField({ label, onChange, value }: { label: string; value: number | null | undefined; onChange: (value: number) => void }) {
  return <View style={styles.field}><Text style={styles.label}>{label}</Text><TextInput accessibilityLabel={label} keyboardType="decimal-pad" onChangeText={(text) => onChange(Number(text))} style={styles.input} value={String(value ?? "")} /></View>;
}

function Toggle({ label, onChange, value }: { label: string; value: boolean; onChange: (value: boolean) => void }) {
  return <View style={styles.toggle}><Text style={styles.label}>{label}</Text><Switch onValueChange={onChange} thumbColor={value ? palette.primary : palette.textMuted} trackColor={{ false: palette.surfaceMuted, true: palette.primarySoft }} value={value} /></View>;
}

function cloneConfig(config: StrategyConfig): StrategyConfig {
  return structuredClone(config);
}

export default function StrategyEditorScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute() as RouteParams;
  const { session } = useSession();
  const queryClient = useQueryClient();
  const strategyId = route.params?.strategyId;
  const [name, setName] = useState("移动端策略");
  const [description, setDescription] = useState("");
  const [config, setConfig] = useState<StrategyConfig>(createStrategyConfig);
  const [symbolInput, setSymbolInput] = useState("");
  const [connectionPickerVisible, setConnectionPickerVisible] = useState(false);
  const [aiVisible, setAiVisible] = useState(false);
  const [aiText, setAiText] = useState("");
  const [proposal, setProposal] = useState<TextProposal | null>(null);
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const existing = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId], queryFn: () => api.strategy(strategyId ?? ""), enabled: Boolean(strategyId) });
  const connections = useQuery({ queryKey: ["mobile", session?.tenantId, "sandbox-connections"], queryFn: api.demoConnections, enabled: Boolean(session) });
  const mayManage = canMutate(access.data, "strategy.manage");
  const parse = useMutation({ mutationFn: (strategyText: string) => api.parseStrategyText(strategyText) });
  const create = useMutation({ mutationFn: (draft: { name: string; description?: string; config: StrategyConfig }) => api.createStrategy({ ...draft, initial_capital_quote: draft.config.initial_capital_quote }), onSuccess: (saved) => void Promise.all([queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategies"] }), queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategy", saved.strategy_id] })]) });
  const update = useMutation({ mutationFn: (draft: { name: string; description: string | null; config: StrategyConfig }) => api.updateStrategy(strategyId ?? "", draft), onSuccess: (saved) => void Promise.all([queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategies"] }), queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategy", saved.strategy_id] })]) });

  useEffect(() => {
    if (!existing.data) return;
    setName(existing.data.name);
    setDescription(existing.data.description ?? "");
    setConfig(cloneConfig(existing.data.config));
  }, [existing.data]);

  const demoConnections = useMemo(() => validOkxDemoConnections(connections.data ?? []), [connections.data]);
  const lockedTarget = existing.data?.status === "running";
  const patch = <K extends keyof StrategyConfig>(key: K, value: StrategyConfig[K]) => setConfig((current) => ({ ...current, [key]: value }));
  const patchExecution = (key: string, value: string | number | undefined) => setConfig((current) => ({ ...current, execution: { ...current.execution, [key]: value } }));
  const patchRisk = (key: string, value: number) => setConfig((current) => ({ ...current, risk: { ...current.risk, [key]: value } }));
  const patchAdvanced = (key: string, value: unknown) => setConfig((current) => ({ ...current, advanced_rules: { ...current.advanced_rules, [key]: value } }));
  const addSymbol = () => {
    const symbols = normalizeSymbols([...config.symbols, symbolInput]);
    if (symbols.length === config.symbols.length) return;
    patch("symbols", symbols);
    setSymbolInput("");
  };
  const save = async () => {
    if (!name.trim()) { Alert.alert("缺少名称", "策略名称不能为空。"); return; }
    const normalized = { ...config, symbols: normalizeSymbols(config.symbols) };
    const validation = validateStrategyConfig(normalized, connections.data ?? []);
    if (validation) { Alert.alert("配置无效", validation); return; }
    const draft = { name: name.trim(), description: description.trim() || null, config: normalized };
    try {
      const saved = strategyId
        ? await update.mutateAsync(draft)
        : await create.mutateAsync({ ...draft, description: draft.description ?? undefined });
      navigation.replace("StrategyDetail", { strategyId: saved.strategy_id });
    } catch (error) {
      Alert.alert("保存失败", error instanceof Error ? error.message : "服务拒绝了本次保存。");
    }
  };
  const requestAiDraft = async () => {
    const text = aiText.trim();
    if (text.length < 10 || text.length > 8_000) { Alert.alert("草拟长度无效", "策略描述需为 10 至 8,000 个字符。"); return; }
    try { setProposal((await parse.mutateAsync(text)) as TextProposal); } catch (error) { Alert.alert("AI 草拟不可用", error instanceof Error ? error.message : "请稍后重试。"); }
  };
  const applyProposal = () => {
    if (!proposal) return;
    if (proposal.strategy_name) setName(proposal.strategy_name);
    if (proposal.config) setConfig((current) => ({ ...current, ...proposal.config, advanced_rules: proposal.config?.advanced_rules ?? current.advanced_rules, risk: proposal.config?.risk ?? current.risk }));
    setAiVisible(false);
  };

  if (existing.isLoading) return <StatePanel description="正在读取服务器策略配置。" title="编辑策略" />;
  return <ScrollView contentContainerStyle={styles.content} style={styles.page}>
    <ScreenHeader actionLabel={mayManage ? "AI 草拟" : undefined} onAction={mayManage ? () => setAiVisible(true) : undefined} subtitle={strategyId ? "修改将由服务端再次校验" : "创建后仍需显式启动策略"} title={strategyId ? "编辑策略" : "新建策略"} />
    {!mayManage ? <StatePanel description={access.data?.status === "active" ? "当前角色不具备策略管理权限。" : "服务尚未激活，当前仅可查看。"} title="只读访问" /> : null}
    <SectionCard description="交易对会规范化为唯一的 *-USDT。" title="身份与市场范围"><View style={styles.field}><Text style={styles.label}>策略名称</Text><TextInput accessibilityLabel="策略名称" onChangeText={setName} style={styles.input} value={name} /></View><View style={styles.field}><Text style={styles.label}>说明</Text><TextInput accessibilityLabel="策略说明" multiline onChangeText={setDescription} style={[styles.input, styles.multiline]} value={description} /></View><NumberField label="初始资金（USDT）" onChange={(value) => patch("initial_capital_quote", value)} value={config.initial_capital_quote} /><View style={styles.symbols}>{config.symbols.map((symbol) => <Pressable accessibilityRole="button" key={symbol} onPress={() => patch("symbols", config.symbols.filter((item) => item !== symbol))} style={styles.symbol}><Text style={styles.symbolText}>{symbol.replace("-", "/")} ×</Text></Pressable>)}</View><View style={styles.addSymbol}><TextInput accessibilityLabel="添加交易对" autoCapitalize="characters" onChangeText={setSymbolInput} placeholder="BTC-USDT" placeholderTextColor={palette.textMuted} style={[styles.input, styles.flex]} value={symbolInput} /><Pressable accessibilityRole="button" onPress={addSymbol} style={styles.addButton}><Plus color={palette.canvas} size={18} /></Pressable></View><View style={styles.intervals}>{STRATEGY_INTERVALS.map((interval) => <Pressable accessibilityRole="button" key={interval} onPress={() => patch("interval", interval)} style={[styles.interval, config.interval === interval && styles.intervalActive]}><Text style={[styles.intervalText, config.interval === interval && styles.intervalTextActive]}>{interval}</Text></Pressable>)}</View></SectionCard>
    <SectionCard description={lockedTarget ? "策略运行中，必须先停止后才能修改执行目标。" : "Demo 仅允许当前工作区内经过验证的 OKX 现货连接。"} title="执行与额度"><View style={styles.environmentRow}>{(["paper", "okx_demo"] as const).map((environment) => <Pressable accessibilityRole="button" disabled={lockedTarget} key={environment} onPress={() => { patchExecution("environment", environment); if (environment === "okx_demo") patchRisk("leverage", 1); }} style={[styles.environment, config.execution.environment === environment && styles.environmentActive, lockedTarget && styles.disabled]}><Text style={styles.environmentTitle}>{environment === "paper" ? "纸面交易" : "OKX Demo"}</Text></Pressable>)}</View>{config.execution.environment === "okx_demo" ? <Pressable accessibilityRole="button" disabled={lockedTarget} onPress={() => setConnectionPickerVisible(true)} style={[styles.connectionButton, lockedTarget && styles.disabled]}><Text style={styles.label}>Demo 连接</Text><Text style={styles.connectionValue}>{demoConnections.find((item) => item.id === config.execution.sandbox_connection_id)?.label ?? "选择连接"}</Text></Pressable> : null}<NumberField label="单笔最大额度" onChange={(value) => patchExecution("max_order_quote_amount", value)} value={config.execution.max_order_quote_amount} /><NumberField label="每日最大额度" onChange={(value) => patchExecution("max_daily_quote_amount", value)} value={config.execution.max_daily_quote_amount} /><NumberField label="总最大额度" onChange={(value) => patchExecution("max_total_quote_amount", value)} value={config.execution.max_total_quote_amount} /></SectionCard>
    <SectionCard title="风险"><NumberField label="风险单笔金额" onChange={(value) => patchRisk("order_quote_amount", value)} value={config.risk.order_quote_amount} /><NumberField label="最大持仓数" onChange={(value) => patchRisk("max_positions", value)} value={config.risk.max_positions} /><NumberField label="杠杆" onChange={(value) => patchRisk("leverage", config.execution.environment === "okx_demo" ? 1 : value)} value={config.risk.leverage} /><NumberField label="止盈比例" onChange={(value) => patchRisk("take_profit_pct", value)} value={config.risk.take_profit_pct} /><NumberField label="止损比例" onChange={(value) => patchRisk("stop_loss_pct", value)} value={config.risk.stop_loss_pct} /></SectionCard>
    <SectionCard title="基础指标"><Toggle label="移动均线" onChange={(value) => patch("moving_average", { ...config.moving_average, enabled: value })} value={config.moving_average.enabled} /><NumberField label="短均线窗口" onChange={(value) => patch("moving_average", { ...config.moving_average, short_window: value })} value={config.moving_average.short_window} /><NumberField label="长均线窗口" onChange={(value) => patch("moving_average", { ...config.moving_average, long_window: value })} value={config.moving_average.long_window} /><Toggle label="RSI" onChange={(value) => patch("rsi", { ...config.rsi, enabled: value })} value={config.rsi.enabled} /><NumberField label="RSI 周期" onChange={(value) => patch("rsi", { ...config.rsi, period: value })} value={config.rsi.period} /><NumberField label="RSI 超卖" onChange={(value) => patch("rsi", { ...config.rsi, oversold: value })} value={config.rsi.oversold} /><NumberField label="RSI 超买" onChange={(value) => patch("rsi", { ...config.rsi, overbought: value })} value={config.rsi.overbought} /><Toggle label="布林带" onChange={(value) => patch("bollinger", { ...config.bollinger, enabled: value })} value={config.bollinger.enabled} /><NumberField label="布林带周期" onChange={(value) => patch("bollinger", { ...config.bollinger, period: value })} value={config.bollinger.period} /><NumberField label="布林带标准差" onChange={(value) => patch("bollinger", { ...config.bollinger, standard_deviations: value })} value={config.bollinger.standard_deviations} /><Toggle label="动量与 MACD" onChange={(value) => patch("momentum_macd", { ...config.momentum_macd, enabled: value })} value={config.momentum_macd.enabled} /><NumberField label="MACD 快线" onChange={(value) => patch("momentum_macd", { ...config.momentum_macd, macd_fast_window: value })} value={config.momentum_macd.macd_fast_window} /><NumberField label="MACD 慢线" onChange={(value) => patch("momentum_macd", { ...config.momentum_macd, macd_slow_window: value })} value={config.momentum_macd.macd_slow_window} /><NumberField label="MACD 信号线 / 动量周期" onChange={(value) => patch("momentum_macd", { ...config.momentum_macd, macd_signal_window: value, momentum_period: value })} value={config.momentum_macd.macd_signal_window} /></SectionCard>
    <SectionCard description="每张规则卡均使用服务端支持的周期、阈值、比较器与启用状态。" title="高级规则"><Toggle label="启用高级规则" onChange={(value) => patchAdvanced("enabled", value)} value={config.advanced_rules.enabled} /><View style={styles.environmentRow}>{(["all", "any"] as const).map((mode) => <Pressable accessibilityRole="button" key={mode} onPress={() => patchAdvanced("entry_confirmation_mode", mode)} style={[styles.interval, config.advanced_rules.entry_confirmation_mode === mode && styles.intervalActive]}><Text style={[styles.intervalText, config.advanced_rules.entry_confirmation_mode === mode && styles.intervalTextActive]}>入场 {mode}</Text></Pressable>)}</View><AdvancedCard config={config} name="moving_average" onChange={(value) => patchAdvanced("moving_average", value)} /><AdvancedCard config={config} name="macd" onChange={(value) => patchAdvanced("macd", value)} /><AdvancedCard config={config} name="bollinger" onChange={(value) => patchAdvanced("bollinger", value)} /><AdvancedCard config={config} name="rsi" onChange={(value) => patchAdvanced("rsi", value)} /><AdvancedCard config={config} name="momentum" onChange={(value) => patchAdvanced("momentum", value)} /><AdvancedCard config={config} name="brar" onChange={(value) => patchAdvanced("brar", value)} /></SectionCard>
    <SectionCard description="退出规则会使用独立的 all / any 确认模式。" title="高级规则退出确认"><View style={styles.environmentRow}>{(["all", "any"] as const).map((mode) => <Pressable accessibilityRole="button" key={mode} onPress={() => patchAdvanced("exit_confirmation_mode", mode)} style={[styles.interval, config.advanced_rules.exit_confirmation_mode === mode && styles.intervalActive]}><Text style={[styles.intervalText, config.advanced_rules.exit_confirmation_mode === mode && styles.intervalTextActive]}>退出 {mode}</Text></Pressable>)}</View></SectionCard>
    <Pressable accessibilityRole="button" disabled={!mayManage || create.isPending || update.isPending} onPress={() => void save()} style={[styles.save, (!mayManage || create.isPending || update.isPending) && styles.disabled]}><Save color={palette.canvas} size={19} /><Text style={styles.saveText}>显式保存策略</Text></Pressable>
    <BottomSheetSelector onClose={() => setConnectionPickerVisible(false)} onSelect={(id) => { patchExecution("sandbox_connection_id", id); setConnectionPickerVisible(false); }} options={demoConnections.map((item) => ({ description: "OKX Demo spot · 已验证", label: item.label, value: item.id }))} selectedValue={config.execution.sandbox_connection_id} title="选择 OKX Demo 连接" visible={connectionPickerVisible} />
    <Modal animationType="slide" onRequestClose={() => setAiVisible(false)} transparent visible={aiVisible}><View style={styles.modalBackdrop}><View style={styles.modal}><View style={styles.modalHead}><Text style={styles.modalTitle}>AI 草拟</Text><Pressable accessibilityRole="button" onPress={() => setAiVisible(false)} style={styles.iconButton}><X color={palette.text} size={20} /></Pressable></View><Text style={styles.modalCopy}>仅将返回的建议映射到同一草稿。不会自动保存、执行或修改策略。</Text><TextInput accessibilityLabel="策略自然语言描述" multiline onChangeText={setAiText} placeholder="输入 10 至 8,000 个字符的策略说明" placeholderTextColor={palette.textMuted} style={styles.aiInput} value={aiText} />{proposal ? <View style={styles.proposal}><Text style={styles.proposalTitle}>{proposal.summary ?? "已生成待审核草稿"}</Text>{(proposal.unresolved_items ?? []).map((item) => <Text key={item} style={styles.unresolved}>• {item}</Text>)}<Pressable accessibilityRole="button" onPress={applyProposal} style={styles.apply}><Text style={styles.applyText}>应用到草稿，继续审核</Text></Pressable></View> : <Pressable accessibilityRole="button" disabled={parse.isPending} onPress={() => void requestAiDraft()} style={[styles.apply, parse.isPending && styles.disabled]}><Bot color={palette.canvas} size={18} /><Text style={styles.applyText}>{parse.isPending ? "正在草拟…" : "生成待审核草稿"}</Text></Pressable>}</View></View></Modal>
  </ScrollView>;
}

function AdvancedCard({ config, name, onChange }: { config: StrategyConfig; name: "moving_average" | "macd" | "bollinger" | "rsi" | "momentum" | "brar"; onChange: (value: any) => void }) {
  const rule = config.advanced_rules[name] as unknown as Record<string, unknown>;
  const update = (key: string, value: unknown) => onChange({ ...rule, [key]: value });
  return <View style={styles.advancedCard}><Toggle label={`启用 ${name}`} onChange={(value) => update("enabled", value)} value={Boolean(rule.enabled)} /><View style={styles.intervals}>{STRATEGY_INTERVALS.map((interval) => <Pressable accessibilityRole="button" key={interval} onPress={() => update("interval", interval)} style={[styles.interval, rule.interval === interval && styles.intervalActive]}><Text style={[styles.intervalText, rule.interval === interval && styles.intervalTextActive]}>{interval}</Text></Pressable>)}</View>{["period", "fast_window", "slow_window", "signal_window", "standard_deviations", "entry_threshold", "exit_threshold"].filter((key) => key in rule).map((key) => <NumberField key={key} label={`${name}.${key}`} onChange={(value) => update(key, value)} value={typeof rule[key] === "number" ? rule[key] : undefined} />)}{["entry_comparator", "exit_comparator", "entry_cross", "entry_reference", "component"].filter((key) => key in rule).map((key) => <TextInput accessibilityLabel={`${name}.${key}`} key={key} onChangeText={(value) => update(key, value)} style={styles.input} value={String(rule[key] ?? "")} />)}{"exit_enabled" in rule ? <Toggle label={`${name}.exit_enabled`} onChange={(value) => update("exit_enabled", value)} value={Boolean(rule.exit_enabled)} /> : null}</View>;
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, field: { gap: spacing.xs }, label: { color: palette.textMuted, fontSize: 12, fontWeight: "700" }, input: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, fontSize: 15, minHeight: 46, paddingHorizontal: spacing.sm }, multiline: { minHeight: 88, paddingTop: spacing.sm, textAlignVertical: "top" }, toggle: { alignItems: "center", flexDirection: "row", justifyContent: "space-between", minHeight: 44 }, symbols: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs }, symbol: { backgroundColor: palette.primarySoft, borderRadius: radius.pill, minHeight: 36, justifyContent: "center", paddingHorizontal: spacing.sm }, symbolText: { color: palette.primary, fontSize: 12, fontWeight: "800" }, addSymbol: { alignItems: "center", flexDirection: "row", gap: spacing.xs }, flex: { flex: 1 }, addButton: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, height: 46, justifyContent: "center", width: 46 }, intervals: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs }, interval: { borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, minHeight: 40, justifyContent: "center", paddingHorizontal: spacing.sm }, intervalActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, intervalText: { color: palette.textMuted, fontSize: 12, fontWeight: "700" }, intervalTextActive: { color: palette.primary }, environmentRow: { flexDirection: "row", gap: spacing.sm }, environment: { borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flex: 1, minHeight: 52, justifyContent: "center", padding: spacing.sm }, environmentActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary }, environmentTitle: { color: palette.text, fontSize: 14, fontWeight: "800" }, connectionButton: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, gap: spacing.xs, minHeight: 52, justifyContent: "center", paddingHorizontal: spacing.sm }, connectionValue: { color: palette.primary, fontSize: 14, fontWeight: "800" }, advancedCard: { borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, gap: spacing.sm, padding: spacing.sm }, save: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.sm, justifyContent: "center", minHeight: 52 }, saveText: { color: palette.canvas, fontSize: 16, fontWeight: "800" }, disabled: { opacity: 0.45 }, modalBackdrop: { backgroundColor: "#00000099", flex: 1, justifyContent: "flex-end" }, modal: { backgroundColor: palette.surface, borderTopLeftRadius: radius.lg, borderTopRightRadius: radius.lg, gap: spacing.md, maxHeight: "88%", padding: spacing.lg }, modalHead: { alignItems: "center", flexDirection: "row", justifyContent: "space-between" }, modalTitle: { color: palette.text, fontSize: 20, fontWeight: "800" }, iconButton: { alignItems: "center", justifyContent: "center", minHeight: 44, minWidth: 44 }, modalCopy: { color: palette.textMuted, fontSize: 13, lineHeight: 20 }, aiInput: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, minHeight: 160, padding: spacing.sm, textAlignVertical: "top" }, proposal: { gap: spacing.sm }, proposalTitle: { color: palette.text, fontWeight: "800" }, unresolved: { color: palette.warning, fontSize: 13 }, apply: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.sm, justifyContent: "center", minHeight: 48, paddingHorizontal: spacing.md }, applyText: { color: palette.canvas, fontSize: 14, fontWeight: "800" },
});
