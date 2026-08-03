import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Modal, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { useNavigation, useRoute } from "@react-navigation/native";
import { Bot, Check, Plus, Save, Sparkles, X } from "lucide-react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { BottomSheetSelector, PrimaryButton, ScreenHeader, SectionCard, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import type { RuleStrategyTextImportProposal, StrategyConfig } from "../types";
import {
  createStrategyConfig,
  normalizeSymbols,
  STRATEGY_INTERVALS,
  validOkxDemoConnections,
  validateStrategyConfig,
} from "./strategy-form";

type RouteParams = { params?: { strategyId?: string } };
type AdvancedRuleName = "moving_average" | "macd" | "bollinger" | "rsi" | "momentum" | "brar";
type ChoiceOption<Value extends string> = { label: string; value: Value };

const ADVANCED_RULE_LABELS: Readonly<Record<AdvancedRuleName, string>> = {
  moving_average: "MA 价格规则",
  macd: "MACD 交叉规则",
  bollinger: "布林带规则",
  rsi: "RSI 阈值规则",
  momentum: "动量阈值规则",
  brar: "BRAR 阈值规则",
};

const ADVANCED_NUMBER_LABELS: Readonly<Record<string, string>> = {
  entry_threshold: "入场阈值",
  exit_threshold: "退出阈值",
  fast_window: "快速窗口",
  period: "周期",
  signal_window: "信号窗口",
  slow_window: "慢速窗口",
  standard_deviations: "标准差倍数",
};

function cloneConfig(config: StrategyConfig): StrategyConfig {
  return structuredClone(config);
}

function hasInvalidNumber(value: unknown): boolean {
  if (typeof value === "number") return !Number.isFinite(value);
  if (Array.isArray(value)) return value.some(hasInvalidNumber);
  if (value && typeof value === "object") return Object.values(value).some(hasInvalidNumber);
  return false;
}

function NumberField({
  disabled = false,
  label,
  onChange,
  value,
}: {
  disabled?: boolean;
  label: string;
  onChange: (value: number) => void;
  value: number | null | undefined;
}) {
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        accessibilityLabel={label}
        editable={!disabled}
        keyboardType="decimal-pad"
        onChangeText={(text) => {
          const next = text.trim() === "" ? 0 : Number(text);
          onChange(Number.isFinite(next) ? next : 0);
        }}
        style={[styles.input, disabled && styles.disabledInput]}
        value={value == null ? "" : String(value)}
      />
    </View>
  );
}

function TextField({
  disabled = false,
  label,
  multiline = false,
  onChangeText,
  placeholder,
  value,
}: {
  disabled?: boolean;
  label: string;
  multiline?: boolean;
  onChangeText: (value: string) => void;
  placeholder?: string;
  value: string;
}) {
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        accessibilityLabel={label}
        editable={!disabled}
        multiline={multiline}
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor={palette.textMuted}
        style={[styles.input, multiline && styles.multiline, disabled && styles.disabledInput]}
        value={value}
      />
    </View>
  );
}

function Toggle({
  disabled = false,
  label,
  onChange,
  value,
}: {
  disabled?: boolean;
  label: string;
  onChange: (value: boolean) => void;
  value: boolean;
}) {
  return (
    <Pressable
      accessibilityLabel={label}
      accessibilityRole="switch"
      accessibilityState={{ checked: value, disabled }}
      disabled={disabled}
      onPress={() => onChange(!value)}
      style={({ pressed }) => [styles.toggle, disabled && styles.disabled, pressed && !disabled && styles.pressed]}
    >
      <Text style={styles.toggleLabel}>{label}</Text>
      <View style={[styles.toggleTrack, value && styles.toggleTrackOn]}>
        <View style={[styles.toggleKnob, value && styles.toggleKnobOn]} />
      </View>
    </Pressable>
  );
}

function ChoicePills<Value extends string>({
  disabled = false,
  label,
  onChange,
  options,
  value,
}: {
  disabled?: boolean;
  label?: string;
  onChange: (value: Value) => void;
  options: readonly ChoiceOption<Value>[];
  value: Value;
}) {
  return (
    <View style={styles.choiceGroup}>
      {label ? <Text style={styles.label}>{label}</Text> : null}
      <View style={styles.choices}>
        {options.map((option) => {
          const selected = option.value === value;
          return (
            <Pressable
              accessibilityLabel={`${label ?? "选项"}：${option.label}`}
              accessibilityRole="button"
              disabled={disabled}
              key={option.value}
              onPress={() => onChange(option.value)}
              style={({ pressed }) => [styles.choice, selected && styles.choiceActive, disabled && styles.disabled, pressed && !disabled && styles.pressed]}
            >
              <Text style={[styles.choiceText, selected && styles.choiceTextActive]}>{option.label}</Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

function AdvancedCard({
  disabled,
  name,
  rule,
  onChange,
}: {
  disabled: boolean;
  name: AdvancedRuleName;
  rule: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
}) {
  const update = (key: string, value: unknown) => onChange({ ...rule, [key]: value });
  const hasKey = (key: string) => Object.prototype.hasOwnProperty.call(rule, key);
  const numericKeys = [
    "period",
    "fast_window",
    "slow_window",
    "signal_window",
    "standard_deviations",
    "entry_threshold",
    "exit_threshold",
  ].filter(hasKey);

  return (
    <View style={styles.advancedCard}>
      <Toggle disabled={disabled} label={`启用${ADVANCED_RULE_LABELS[name]}`} onChange={(value) => update("enabled", value)} value={Boolean(rule.enabled)} />
      <ChoicePills
        disabled={disabled}
        label="规则周期"
        onChange={(value) => update("interval", value)}
        options={STRATEGY_INTERVALS.map((value) => ({ label: value, value }))}
        value={(typeof rule.interval === "string" ? rule.interval : "15m") as StrategyConfig["interval"]}
      />
      {numericKeys.map((key) => (
        <NumberField
          disabled={disabled}
          key={key}
          label={ADVANCED_NUMBER_LABELS[key] ?? key}
          onChange={(value) => update(key, value)}
          value={typeof rule[key] === "number" ? rule[key] : 0}
        />
      ))}
      {hasKey("entry_comparator") ? <ChoicePills disabled={disabled} label="入场比较" onChange={(value) => update("entry_comparator", value)} options={[{ label: "高于或等于", value: "above" }, { label: "低于或等于", value: "below" }]} value={(rule.entry_comparator === "below" ? "below" : "above") as "above" | "below"} /> : null}
      {hasKey("entry_cross") ? <ChoicePills disabled={disabled} label="入场交叉" onChange={(value) => update("entry_cross", value)} options={[{ label: "金叉", value: "golden" }, { label: "死叉", value: "death" }]} value={(rule.entry_cross === "death" ? "death" : "golden") as "golden" | "death"} /> : null}
      {hasKey("entry_reference") ? <ChoicePills disabled={disabled} label="布林参考线" onChange={(value) => update("entry_reference", value)} options={[{ label: "上轨", value: "upper" }, { label: "中轨", value: "middle" }, { label: "下轨", value: "lower" }]} value={(["upper", "lower"].includes(String(rule.entry_reference)) ? rule.entry_reference : "middle") as "upper" | "middle" | "lower"} /> : null}
      {hasKey("component") ? <ChoicePills disabled={disabled} label="BRAR 分量" onChange={(value) => update("component", value)} options={[{ label: "AR", value: "ar" }, { label: "BR", value: "br" }]} value={(rule.component === "ar" ? "ar" : "br") as "ar" | "br"} /> : null}
      {hasKey("exit_enabled") ? <Toggle disabled={disabled} label="启用退出条件" onChange={(value) => update("exit_enabled", value)} value={Boolean(rule.exit_enabled)} /> : null}
      {hasKey("exit_comparator") && Boolean(rule.exit_enabled) ? <ChoicePills disabled={disabled} label="退出比较" onChange={(value) => update("exit_comparator", value)} options={[{ label: "高于或等于", value: "above" }, { label: "低于或等于", value: "below" }]} value={(rule.exit_comparator === "below" ? "below" : "above") as "above" | "below"} /> : null}
    </View>
  );
}

export default function StrategyEditorScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute() as RouteParams;
  const { session } = useSession();
  const queryClient = useQueryClient();
  const strategyId = route.params?.strategyId;
  const [name, setName] = useState("未命名策略");
  const [description, setDescription] = useState("");
  const [config, setConfig] = useState<StrategyConfig>(createStrategyConfig);
  const [symbolInput, setSymbolInput] = useState("");
  const [connectionPickerVisible, setConnectionPickerVisible] = useState(false);
  const [aiVisible, setAiVisible] = useState(false);
  const [aiText, setAiText] = useState("");
  const [proposal, setProposal] = useState<RuleStrategyTextImportProposal | null>(null);
  const [appliedProposal, setAppliedProposal] = useState<string | null>(null);
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const existing = useQuery({ queryKey: ["mobile", session?.tenantId, "strategy", strategyId], queryFn: () => api.strategy(strategyId ?? ""), enabled: Boolean(strategyId) });
  const connections = useQuery({ queryKey: ["mobile", session?.tenantId, "sandbox-connections"], queryFn: api.demoConnections, enabled: Boolean(session) });
  const parse = useMutation({ mutationFn: (strategyText: string) => api.parseStrategyText(strategyText) });
  const create = useMutation({
    mutationFn: (draft: { name: string; description?: string; config: StrategyConfig }) => api.createStrategy({ ...draft, initial_capital_quote: draft.config.initial_capital_quote }),
    onSuccess: async (saved) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategies"] }),
        queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategy", saved.strategy_id] }),
      ]);
    },
  });
  const update = useMutation({
    mutationFn: (draft: { name: string; description: string | null; config: StrategyConfig }) => api.updateStrategy(strategyId ?? "", draft),
    onSuccess: async (saved) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategies"] }),
        queryClient.invalidateQueries({ queryKey: ["mobile", session?.tenantId, "strategy", saved.strategy_id] }),
      ]);
    },
  });

  useEffect(() => {
    if (!existing.data) return;
    setName(existing.data.name);
    setDescription(existing.data.description ?? "");
    setConfig(cloneConfig(existing.data.config));
  }, [existing.data]);

  const demoConnections = useMemo(() => validOkxDemoConnections(connections.data ?? []), [connections.data]);
  const isArchived = Boolean(existing.data?.archived_at);
  const mayManage = canMutate(access.data, "strategy.manage");
  const mayEdit = mayManage && !isArchived;
  const lockedExecution = existing.data?.status === "running";
  const patch = <Key extends keyof StrategyConfig>(key: Key, value: StrategyConfig[Key]) => setConfig((current) => ({ ...current, [key]: value }));
  const patchExecution = <Key extends keyof StrategyConfig["execution"]>(key: Key, value: StrategyConfig["execution"][Key]) => setConfig((current) => ({ ...current, execution: { ...current.execution, [key]: value } }));
  const patchRisk = <Key extends keyof StrategyConfig["risk"]>(key: Key, value: StrategyConfig["risk"][Key]) => setConfig((current) => ({ ...current, risk: { ...current.risk, [key]: value } }));
  const patchAdvanced = <Key extends keyof StrategyConfig["advanced_rules"]>(key: Key, value: StrategyConfig["advanced_rules"][Key]) => setConfig((current) => ({ ...current, advanced_rules: { ...current.advanced_rules, [key]: value } }));
  const patchAdvancedRule = (ruleName: AdvancedRuleName, nextRule: Record<string, unknown>) => patchAdvanced(ruleName, nextRule as unknown as StrategyConfig["advanced_rules"][typeof ruleName]);

  const addSymbol = () => {
    const symbols = normalizeSymbols([...config.symbols, symbolInput]);
    if (symbols.length === config.symbols.length) return;
    patch("symbols", symbols);
    setSymbolInput("");
  };
  const save = async () => {
    if (!mayEdit) return;
    if (!name.trim()) {
      Alert.alert("缺少名称", "策略名称不能为空。\n");
      return;
    }
    const normalized = { ...config, symbols: normalizeSymbols(config.symbols) };
    if (hasInvalidNumber(normalized)) {
      Alert.alert("配置无效", "数值字段必须是有效数字。\n");
      return;
    }
    const validation = validateStrategyConfig(normalized, connections.data ?? []);
    if (validation) {
      Alert.alert("配置无效", validation);
      return;
    }
    const draft = { name: name.trim(), description: description.trim() || null, config: normalized };
    try {
      const saved = strategyId
        ? await update.mutateAsync(draft)
        : await create.mutateAsync({ ...draft, description: draft.description ?? undefined });
      navigation.replace("StrategyDetail", { strategyId: saved.strategy_id });
    } catch (error) {
      Alert.alert("保存失败", error instanceof Error ? error.message : "服务未完成本次保存。");
    }
  };
  const openAiDraft = () => {
    parse.reset();
    setAiVisible(true);
  };
  const requestAiDraft = async () => {
    const text = aiText.trim();
    if (text.length < 10 || text.length > 8_000) {
      Alert.alert("草拟长度无效", "策略描述需为 10 至 8,000 个字符。\n");
      return;
    }
    try {
      setProposal(await parse.mutateAsync(text));
    } catch {
      // Mutation error is rendered in the review sheet so the server message stays visible.
    }
  };
  const applyProposal = () => {
    if (!proposal) return;
    if (proposal.strategy_name) setName(proposal.strategy_name);
    setConfig((current) => ({
      ...current,
      interval: proposal.config.interval,
      advanced_rules: structuredClone(proposal.config.advanced_rules),
      risk: { ...current.risk, ...proposal.config.risk },
    }));
    setAppliedProposal(proposal.summary);
    setAiVisible(false);
  };

  if (existing.isLoading) return <StatePanel description="正在读取服务端策略配置。" title="正在打开策略编辑器" />;
  if (existing.isError) return <StatePanel actionLabel="重试" description={(existing.error as Error).message} onAction={() => void existing.refetch()} title="策略编辑器暂不可用" tone="error" />;

  return (
    <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled" style={styles.page}>
      <ScreenHeader subtitle={isArchived ? "该策略已归档，只能查看历史配置。" : strategyId ? "保存前会再次由服务端校验，修改不会自动启动策略。" : "创建后仍需显式启动策略，移动端不会自动执行。"} title={strategyId ? "编辑策略" : "新建策略"} />
      {!mayManage ? <StatePanel description={access.data?.status === "active" ? "当前角色不具备策略管理权限。" : "服务尚未激活，当前仅可查看。"} title="策略编辑受限" /> : null}
      {isArchived ? <StatePanel description="归档策略保留历史但不可修改、启动或恢复。" title="已归档，只读" /> : null}

      {mayEdit ? <View style={styles.aiBanner}>
        <View style={styles.aiIcon}><Sparkles color={palette.primary} size={22} /></View>
        <View style={styles.aiCopy}><Text style={styles.aiTitle}>用 AI 生成参数草案</Text><Text style={styles.aiText}>先解析，再逐项核对并显式应用到当前草稿；AI 不会保存、启动或执行策略。</Text></View>
        <PrimaryButton fullWidth={false} label="AI 草拟" leading={<Bot color={palette.canvas} size={18} />} onPress={openAiDraft} />
      </View> : null}
      {appliedProposal ? <View style={styles.appliedBanner}><Check color={palette.positive} size={20} /><View style={styles.appliedCopy}><Text style={styles.appliedTitle}>AI 草案已应用到当前草稿</Text><Text style={styles.appliedText}>“{appliedProposal}” 已写入本地草稿，尚未保存。请复核所有字段后点击“保存策略”。</Text></View><Pressable accessibilityLabel="再次查看 AI 草拟" accessibilityRole="button" onPress={openAiDraft} style={({ pressed }) => [styles.appliedAction, pressed && styles.pressed]}><Text style={styles.appliedActionText}>查看</Text></Pressable></View> : null}

      <SectionCard description="交易对会规范化为唯一的 *-USDT；不确定的 AI 字段会保留在待审核状态。" title="身份与市场范围">
        <TextField disabled={!mayEdit} label="策略名称" onChangeText={setName} value={name} />
        <TextField disabled={!mayEdit} label="策略说明" multiline onChangeText={setDescription} value={description} />
        <NumberField disabled={!mayEdit} label="初始资金（USDT）" onChange={(value) => patch("initial_capital_quote", value)} value={config.initial_capital_quote} />
        <ChoicePills disabled={!mayEdit} label="评估周期" onChange={(value) => patch("interval", value)} options={STRATEGY_INTERVALS.map((value) => ({ label: value, value }))} value={config.interval} />
        <ChoicePills disabled={!mayEdit} label="基础指标确认方式" onChange={(value) => patch("confirmation_mode", value)} options={[{ label: "全部满足", value: "all" }, { label: "任一满足", value: "any" }]} value={config.confirmation_mode} />
        <View style={styles.field}>
          <Text style={styles.label}>观察交易对</Text>
          <View style={styles.symbols}>
            {config.symbols.map((symbol) => <Pressable accessibilityLabel={`移除交易对 ${symbol}`} accessibilityRole="button" disabled={!mayEdit} key={symbol} onPress={() => patch("symbols", config.symbols.filter((item) => item !== symbol))} style={({ pressed }) => [styles.symbol, !mayEdit && styles.disabled, pressed && mayEdit && styles.pressed]}><Text style={styles.symbolText}>{symbol}</Text><X color={palette.primary} size={16} /></Pressable>)}
          </View>
          <View style={styles.addSymbolRow}><TextInput accessibilityLabel="新增交易对" editable={mayEdit} onChangeText={setSymbolInput} onSubmitEditing={addSymbol} placeholder="例如 BTC 或 BTC-USDT" placeholderTextColor={palette.textMuted} style={[styles.input, styles.symbolInput, !mayEdit && styles.disabledInput]} value={symbolInput} /><Pressable accessibilityLabel="新增交易对" accessibilityRole="button" disabled={!mayEdit} onPress={addSymbol} style={({ pressed }) => [styles.addSymbolButton, !mayEdit && styles.disabled, pressed && mayEdit && styles.pressed]}><Plus color={palette.canvas} size={18} /><Text style={styles.addSymbolText}>添加</Text></Pressable></View>
        </View>
      </SectionCard>

      <SectionCard description={lockedExecution ? "策略运行中。为保护已启动执行，执行环境和连接必须先停止后再修改。" : "Demo 仅允许当前工作区的已验证、未撤销 OKX 现货连接。"} title="执行与额度">
        <ChoicePills disabled={!mayEdit || Boolean(lockedExecution)} label="执行环境" onChange={(environment) => { patchExecution("environment", environment); if (environment === "okx_demo") patchRisk("leverage", 1); }} options={[{ label: "纸面交易", value: "paper" }, { label: "OKX Demo", value: "okx_demo" }]} value={config.execution.environment} />
        {config.execution.environment === "okx_demo" ? <>
          <Pressable accessibilityLabel="选择 OKX Demo 连接" accessibilityRole="button" disabled={!mayEdit || Boolean(lockedExecution)} onPress={() => setConnectionPickerVisible(true)} style={({ pressed }) => [styles.connectionButton, (!mayEdit || lockedExecution) && styles.disabled, pressed && mayEdit && !lockedExecution && styles.pressed]}><View style={styles.connectionCopy}><Text style={styles.label}>OKX Demo 连接</Text><Text numberOfLines={1} style={styles.connectionValue}>{demoConnections.find((connection) => connection.id === config.execution.sandbox_connection_id)?.label ?? "请选择已验证连接"}</Text></View><Text style={styles.connectionAction}>选择</Text></Pressable>
          {!demoConnections.length ? <Text style={styles.warning}>当前工作区没有可用的 OKX Demo 现货连接。保存前需先在“我的”中创建并验证连接。</Text> : null}
        </> : null}
        <NumberField disabled={!mayEdit} label="单笔执行额度（USDT）" onChange={(value) => patchExecution("max_order_quote_amount", value)} value={config.execution.max_order_quote_amount} />
        <NumberField disabled={!mayEdit} label="每日执行额度（USDT）" onChange={(value) => patchExecution("max_daily_quote_amount", value)} value={config.execution.max_daily_quote_amount} />
        <NumberField disabled={!mayEdit} label="总执行额度（USDT）" onChange={(value) => patchExecution("max_total_quote_amount", value)} value={config.execution.max_total_quote_amount} />
      </SectionCard>

      <SectionCard description="所有金额和仓位上限都会由服务端再次校验。" title="风险">
        <NumberField disabled={!mayEdit} label="风险单笔金额（USDT）" onChange={(value) => patchRisk("order_quote_amount", value)} value={config.risk.order_quote_amount} />
        <NumberField disabled={!mayEdit} label="最大持仓数" onChange={(value) => patchRisk("max_positions", value)} value={config.risk.max_positions} />
        <NumberField disabled={!mayEdit} label="杠杆" onChange={(value) => patchRisk("leverage", config.execution.environment === "okx_demo" ? 1 : value)} value={config.risk.leverage} />
        <NumberField disabled={!mayEdit} label="止盈比例" onChange={(value) => patchRisk("take_profit_pct", value)} value={config.risk.take_profit_pct} />
        <NumberField disabled={!mayEdit} label="止损比例" onChange={(value) => patchRisk("stop_loss_pct", value)} value={config.risk.stop_loss_pct} />
      </SectionCard>

      <SectionCard description="MA、RSI、MACD 等缩写保持与服务端配置一致。" title="基础指标">
        <Toggle disabled={!mayEdit} label="启用移动均线" onChange={(value) => patch("moving_average", { ...config.moving_average, enabled: value })} value={config.moving_average.enabled} />
        <NumberField disabled={!mayEdit} label="短期 MA 窗口" onChange={(value) => patch("moving_average", { ...config.moving_average, short_window: value })} value={config.moving_average.short_window} />
        <NumberField disabled={!mayEdit} label="长期 MA 窗口" onChange={(value) => patch("moving_average", { ...config.moving_average, long_window: value })} value={config.moving_average.long_window} />
        <Toggle disabled={!mayEdit} label="启用 RSI" onChange={(value) => patch("rsi", { ...config.rsi, enabled: value })} value={config.rsi.enabled} />
        <NumberField disabled={!mayEdit} label="RSI 周期" onChange={(value) => patch("rsi", { ...config.rsi, period: value })} value={config.rsi.period} />
        <NumberField disabled={!mayEdit} label="RSI 超卖阈值" onChange={(value) => patch("rsi", { ...config.rsi, oversold: value })} value={config.rsi.oversold} />
        <NumberField disabled={!mayEdit} label="RSI 超买阈值" onChange={(value) => patch("rsi", { ...config.rsi, overbought: value })} value={config.rsi.overbought} />
        <Toggle disabled={!mayEdit} label="启用布林带" onChange={(value) => patch("bollinger", { ...config.bollinger, enabled: value })} value={config.bollinger.enabled} />
        <NumberField disabled={!mayEdit} label="布林带周期" onChange={(value) => patch("bollinger", { ...config.bollinger, period: value })} value={config.bollinger.period} />
        <NumberField disabled={!mayEdit} label="布林带标准差倍数" onChange={(value) => patch("bollinger", { ...config.bollinger, standard_deviations: value })} value={config.bollinger.standard_deviations} />
        <Toggle disabled={!mayEdit} label="启用动量与 MACD" onChange={(value) => patch("momentum_macd", { ...config.momentum_macd, enabled: value })} value={config.momentum_macd.enabled} />
        <NumberField disabled={!mayEdit} label="动量周期" onChange={(value) => patch("momentum_macd", { ...config.momentum_macd, momentum_period: value })} value={config.momentum_macd.momentum_period} />
        <NumberField disabled={!mayEdit} label="MACD 快速窗口" onChange={(value) => patch("momentum_macd", { ...config.momentum_macd, macd_fast_window: value })} value={config.momentum_macd.macd_fast_window} />
        <NumberField disabled={!mayEdit} label="MACD 慢速窗口" onChange={(value) => patch("momentum_macd", { ...config.momentum_macd, macd_slow_window: value })} value={config.momentum_macd.macd_slow_window} />
        <NumberField disabled={!mayEdit} label="MACD 信号窗口" onChange={(value) => patch("momentum_macd", { ...config.momentum_macd, macd_signal_window: value })} value={config.momentum_macd.macd_signal_window} />
      </SectionCard>

      <SectionCard description="所有高级规则均使用服务端支持的周期、比较器、阈值与启用状态。" title="高级规则">
        <Toggle disabled={!mayEdit} label="启用高级规则" onChange={(value) => patchAdvanced("enabled", value)} value={config.advanced_rules.enabled} />
        <ChoicePills disabled={!mayEdit} label="入场确认方式" onChange={(value) => patchAdvanced("entry_confirmation_mode", value)} options={[{ label: "全部满足", value: "all" }, { label: "任一满足", value: "any" }]} value={config.advanced_rules.entry_confirmation_mode} />
        <ChoicePills disabled={!mayEdit} label="退出确认方式" onChange={(value) => patchAdvanced("exit_confirmation_mode", value)} options={[{ label: "全部满足", value: "all" }, { label: "任一满足", value: "any" }]} value={config.advanced_rules.exit_confirmation_mode} />
        {(["moving_average", "macd", "bollinger", "rsi", "momentum", "brar"] as const).map((ruleName) => <AdvancedCard disabled={!mayEdit} key={ruleName} name={ruleName} onChange={(value) => patchAdvancedRule(ruleName, value)} rule={config.advanced_rules[ruleName] as unknown as Record<string, unknown>} />)}
      </SectionCard>

      {mayEdit ? <PrimaryButton label="保存策略" leading={<Save color={palette.canvas} size={19} />} loading={create.isPending || update.isPending} onPress={() => void save()} /> : null}

      <BottomSheetSelector onClose={() => setConnectionPickerVisible(false)} onSelect={(connectionId) => { patchExecution("sandbox_connection_id", connectionId); setConnectionPickerVisible(false); }} options={demoConnections.map((connection) => ({ description: "OKX Demo 现货 · 已验证", label: connection.label, value: connection.id }))} selectedValue={config.execution.sandbox_connection_id} title="选择 OKX Demo 连接" visible={connectionPickerVisible} />

      <Modal animationType="slide" onRequestClose={() => setAiVisible(false)} transparent visible={aiVisible}>
        <View style={styles.modalBackdrop}>
          <View accessibilityViewIsModal style={styles.modal}>
            <View style={styles.modalHeader}>
              <View style={styles.modalHeading}><Bot color={palette.primary} size={22} /><View><Text style={styles.modalTitle}>AI 策略草拟</Text><Text style={styles.modalSubtitle}>解析 → 审核方案 → 显式应用 → 手动保存</Text></View></View>
              <Pressable accessibilityLabel="关闭 AI 草拟" accessibilityRole="button" onPress={() => setAiVisible(false)} style={({ pressed }) => [styles.closeButton, pressed && styles.pressed]}><X color={palette.text} size={20} /></Pressable>
            </View>
            <ScrollView contentContainerStyle={styles.modalContent} keyboardShouldPersistTaps="handled">
              <View style={styles.aiNotice}><Sparkles color={palette.warning} size={20} /><Text style={styles.aiNoticeText}>AI 仅把明确描述转换成可审核的参数建议。不会保存、启动、下单或修改服务器策略。</Text></View>
              <TextField label="策略自然语言描述" multiline onChangeText={(text) => { setAiText(text); setProposal(null); }} placeholder="输入 10 至 8,000 个字符，例如：日线趋势向上时，15m RSI 回落到阈值以下才入场……" value={aiText} />
              <Text style={styles.characterCount}>{aiText.trim().length} / 8,000 个字符</Text>
              <PrimaryButton disabled={aiText.trim().length < 10 || aiText.trim().length > 8_000} label="解析为待审核方案" leading={<Sparkles color={palette.canvas} size={18} />} loading={parse.isPending} onPress={() => void requestAiDraft()} />
              {parse.isError ? <StatePanel actionLabel="重试" description={(parse.error as Error).message} onAction={() => void requestAiDraft()} title="AI 草拟不可用" tone="error" /> : null}
              {proposal ? <View style={styles.proposal}>
                <View style={styles.proposalHead}><Check color={palette.positive} size={20} /><Text style={styles.proposalTitle}>待审核参数方案</Text></View>
                <Text style={styles.proposalSummary}>{proposal.summary}</Text>
                <View style={styles.proposalFields}><Text style={styles.proposalField}>策略名称：{proposal.strategy_name ?? "保留当前名称"}</Text><Text style={styles.proposalField}>评估周期：{proposal.config.interval}</Text><Text style={styles.proposalField}>将更新：高级规则与风险参数</Text><Text style={styles.proposalField}>不会更新：交易对、资金、执行环境与连接</Text></View>
                <View style={styles.unresolved}><Text style={styles.unresolvedTitle}>待人工确认</Text>{proposal.unresolved_items.length ? proposal.unresolved_items.map((item, index) => <Text key={`${item}-${index}`} style={styles.unresolvedItem}>• {item}</Text>) : <Text style={styles.unresolvedItem}>• 没有未明确项；仍请在保存前检查全部参数。</Text>}</View>
                <PrimaryButton label="显式应用到当前草稿" leading={<Check color={palette.canvas} size={18} />} onPress={applyProposal} />
                <Text style={styles.applyNote}>应用只会更新本地草稿。应用后仍需返回编辑器并点击“保存策略”。</Text>
              </View> : null}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  field: { gap: spacing.xs },
  label: { color: palette.textMuted, fontSize: 13, fontWeight: "800" },
  input: { backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, color: palette.text, fontSize: 16, minHeight: 48, paddingHorizontal: spacing.sm },
  multiline: { minHeight: 104, paddingTop: spacing.sm, textAlignVertical: "top" },
  disabledInput: { color: palette.textMuted, opacity: 0.65 },
  toggle: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", justifyContent: "space-between", minHeight: 48, paddingHorizontal: spacing.sm },
  toggleLabel: { color: palette.text, flex: 1, fontSize: 15, fontWeight: "800" },
  toggleTrack: { backgroundColor: palette.surfaceMuted, borderRadius: radius.pill, height: 28, justifyContent: "center", paddingHorizontal: 3, width: 50 },
  toggleTrackOn: { backgroundColor: palette.primarySoft },
  toggleKnob: { backgroundColor: palette.textMuted, borderRadius: radius.pill, height: 22, width: 22 },
  toggleKnobOn: { alignSelf: "flex-end", backgroundColor: palette.primary },
  choiceGroup: { gap: spacing.xs },
  choices: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  choice: { alignItems: "center", borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.sm },
  choiceActive: { backgroundColor: palette.primarySoft, borderColor: palette.primary },
  choiceText: { color: palette.textMuted, fontSize: 13, fontWeight: "800" },
  choiceTextActive: { color: palette.primary },
  symbols: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  symbol: { alignItems: "center", backgroundColor: palette.primarySoft, borderColor: palette.primary, borderRadius: radius.pill, borderWidth: 1, flexDirection: "row", gap: spacing.xxs, minHeight: 44, paddingHorizontal: spacing.sm },
  symbolText: { color: palette.primary, fontSize: 13, fontWeight: "900" },
  addSymbolRow: { flexDirection: "row", gap: spacing.xs },
  symbolInput: { flex: 1 },
  addSymbolButton: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xxs, justifyContent: "center", minHeight: 48, minWidth: 82, paddingHorizontal: spacing.sm },
  addSymbolText: { color: palette.canvas, fontSize: 14, fontWeight: "900" },
  connectionButton: { alignItems: "center", backgroundColor: palette.surfaceRaised, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.sm, justifyContent: "space-between", minHeight: 56, paddingHorizontal: spacing.sm },
  connectionCopy: { flex: 1, gap: spacing.xxs },
  connectionValue: { color: palette.text, fontSize: 15, fontWeight: "800" },
  connectionAction: { color: palette.primary, fontSize: 14, fontWeight: "900" },
  warning: { color: palette.warning, fontSize: 13, lineHeight: 20 },
  advancedCard: { backgroundColor: palette.surfaceRaised, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.sm },
  aiBanner: { alignItems: "center", backgroundColor: palette.primarySoft, borderColor: palette.primary, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, padding: spacing.sm },
  aiIcon: { alignItems: "center", justifyContent: "center", minHeight: 44, minWidth: 28 },
  aiCopy: { flex: 1, gap: spacing.xxs },
  aiTitle: { color: palette.text, fontSize: 16, fontWeight: "900" },
  aiText: { color: palette.textMuted, fontSize: 12, lineHeight: 18 },
  appliedBanner: { alignItems: "flex-start", backgroundColor: palette.positiveSoft, borderColor: palette.positive, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, padding: spacing.sm },
  appliedCopy: { flex: 1, gap: spacing.xxs },
  appliedTitle: { color: palette.positive, fontSize: 14, fontWeight: "900" },
  appliedText: { color: palette.text, fontSize: 12, lineHeight: 18 },
  appliedAction: { alignItems: "center", justifyContent: "center", minHeight: 44, minWidth: 44 },
  appliedActionText: { color: palette.primary, fontSize: 13, fontWeight: "900" },
  disabled: { opacity: 0.48 },
  pressed: { opacity: 0.76 },
  modalBackdrop: { backgroundColor: "rgba(0, 0, 0, 0.62)", flex: 1, justifyContent: "flex-end" },
  modal: { backgroundColor: palette.surface, borderColor: palette.border, borderTopLeftRadius: radius.lg, borderTopRightRadius: radius.lg, borderWidth: 1, maxHeight: "92%", paddingTop: spacing.md },
  modalHeader: { alignItems: "center", flexDirection: "row", gap: spacing.sm, paddingHorizontal: spacing.md, paddingBottom: spacing.sm },
  modalHeading: { alignItems: "center", flex: 1, flexDirection: "row", gap: spacing.xs },
  modalTitle: { color: palette.text, fontSize: 19, fontWeight: "900" },
  modalSubtitle: { color: palette.textMuted, fontSize: 12, marginTop: 2 },
  closeButton: { alignItems: "center", justifyContent: "center", minHeight: 44, minWidth: 44 },
  modalContent: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  aiNotice: { alignItems: "flex-start", backgroundColor: palette.warningSoft, borderRadius: radius.sm, flexDirection: "row", gap: spacing.sm, padding: spacing.sm },
  aiNoticeText: { color: palette.warning, flex: 1, fontSize: 13, fontWeight: "800", lineHeight: 19 },
  characterCount: { color: palette.textMuted, fontSize: 12, textAlign: "right" },
  proposal: { backgroundColor: palette.surfaceRaised, borderColor: palette.positive, borderRadius: radius.md, borderWidth: 1, gap: spacing.md, padding: spacing.md },
  proposalHead: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
  proposalTitle: { color: palette.positive, fontSize: 16, fontWeight: "900" },
  proposalSummary: { color: palette.text, fontSize: 14, lineHeight: 22 },
  proposalFields: { gap: spacing.xs },
  proposalField: { color: palette.textMuted, fontSize: 13, lineHeight: 19 },
  unresolved: { backgroundColor: palette.warningSoft, borderRadius: radius.sm, gap: spacing.xxs, padding: spacing.sm },
  unresolvedTitle: { color: palette.warning, fontSize: 13, fontWeight: "900" },
  unresolvedItem: { color: palette.text, fontSize: 13, lineHeight: 19 },
  applyNote: { color: palette.textMuted, fontSize: 12, lineHeight: 18, textAlign: "center" },
});
