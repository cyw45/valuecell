import { StyleSheet, Text, View } from "react-native";
import type { RuleStrategyCondition, RuleStrategyConditionState } from "../types";
import { palette, radius, spacing } from "../theme";
import {
  conditionCategoryLabel,
  conditionDetail,
  conditionFacts,
  conditionLabel,
  conditionStateLabel,
  conditionStateTone,
} from "../screens/strategy-presentation";

export type TradeDecisionCondition = {
  code?: string;
  label?: string | null;
  category?: string;
  state?: string;
  detail?: string;
  values?: Record<string, unknown>;
};

type TradeDecisionConditionsProps = {
  conditions?: readonly TradeDecisionCondition[] | null;
  side?: "buy" | "sell";
  missingText?: string;
};

const CONDITION_STATES: readonly RuleStrategyConditionState[] = [
  "triggered",
  "not_triggered",
  "blocked",
  "unavailable",
];

function normalizeState(value?: string): RuleStrategyConditionState {
  return CONDITION_STATES.includes(value as RuleStrategyConditionState)
    ? (value as RuleStrategyConditionState)
    : "unavailable";
}

function normalizeCategory(value?: string): RuleStrategyCondition["category"] {
  if (value === "indicator" || value === "exit" || value === "risk") {
    return value;
  }
  return "indicator";
}

function normalizeValue(value: unknown): number | string | boolean | null {
  if (value === null || typeof value === "number" || typeof value === "string" || typeof value === "boolean") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function normalizeCondition(condition: TradeDecisionCondition): RuleStrategyCondition {
  return {
    code: condition.code ?? "unknown",
    category: normalizeCategory(condition.category),
    state: normalizeState(condition.state),
    detail: condition.detail ?? "",
    values: Object.fromEntries(
      Object.entries(condition.values ?? {}).map(([key, value]) => [key, normalizeValue(value)]),
    ),
  };
}

function directionalConditions(
  conditions: readonly TradeDecisionCondition[],
  side?: "buy" | "sell",
): readonly TradeDecisionCondition[] {
  if (!side) return conditions;
  const prefix = side === "buy" ? "program.entry." : "program.exit.";
  const directional = conditions.filter((condition) => condition.code?.startsWith(prefix));
  return directional.length > 0 ? directional : conditions;
}

export function TradeDecisionConditions({
  conditions,
  side,
  missingText = "未找到该交易对应的持久化条件记录，未使用当前策略反推。",
}: TradeDecisionConditionsProps) {
  const visibleConditions = directionalConditions(conditions ?? [], side);
  if (visibleConditions.length === 0) {
    return <Text style={styles.missing}>{missingText}</Text>;
  }

  return (
    <View style={styles.root}>
      <View style={styles.headingRow}>
        <Text style={styles.heading}>策略依据</Text>
        <Text style={styles.caption}>{visibleConditions.length} 项服务端条件</Text>
      </View>
      <View style={styles.list}>
        {visibleConditions.map((rawCondition, index) => {
          const condition = normalizeCondition(rawCondition);
          const tone = conditionStateTone(condition.state);
          const color = tone === "positive" ? palette.positive : tone === "negative" ? palette.negative : tone === "warning" ? palette.warning : palette.textMuted;
          const soft = tone === "positive" ? palette.positiveSoft : tone === "negative" ? palette.negativeSoft : tone === "warning" ? palette.warningSoft : palette.surfaceRaised;
          const facts = conditionFacts(condition);
          return (
            <View key={`${condition.code}-${index}`} style={[styles.condition, { borderLeftColor: color }]}>
              <View style={styles.conditionHeader}>
                <View style={[styles.stateDot, { backgroundColor: color }]} />
                <Text numberOfLines={2} style={styles.label}>{rawCondition.label || conditionLabel(condition.code)}</Text>
                <Text style={[styles.state, { backgroundColor: soft, color }]}>{conditionStateLabel(condition.state)}</Text>
              </View>
              <Text style={styles.meta}>{conditionCategoryLabel(condition.category)} · {condition.code}</Text>
              <Text style={styles.detail}>{conditionDetail(condition)}</Text>
              {facts.length > 0 ? (
                <View style={styles.facts}>
                  {facts.map((fact) => <Text key={fact.label} style={styles.fact}>{fact.label} {fact.value}</Text>)}
                </View>
              ) : null}
            </View>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { backgroundColor: palette.surfaceMuted, borderRadius: radius.sm, gap: spacing.xs, padding: spacing.sm },
  headingRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
  heading: { color: palette.text, flex: 1, fontSize: 12, fontWeight: "900", letterSpacing: 0.2 },
  caption: { color: palette.textMuted, fontSize: 10, fontWeight: "700" },
  list: { gap: spacing.xs },
  condition: { backgroundColor: palette.surface, borderLeftWidth: 3, borderRadius: radius.sm, gap: 3, padding: spacing.xs },
  conditionHeader: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
  stateDot: { borderRadius: radius.pill, height: 7, width: 7 },
  label: { color: palette.text, flex: 1, fontSize: 12, fontWeight: "800" },
  state: { borderRadius: radius.pill, fontSize: 10, fontWeight: "900", overflow: "hidden", paddingHorizontal: 6, paddingVertical: 3 },
  meta: { color: palette.textMuted, fontSize: 10, fontWeight: "700" },
  detail: { color: palette.text, fontSize: 12, lineHeight: 18 },
  facts: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xxs },
  fact: { backgroundColor: palette.canvas, borderRadius: radius.sm, color: palette.textMuted, fontSize: 10, paddingHorizontal: 5, paddingVertical: 3 },
  missing: { color: palette.textMuted, fontSize: 12, lineHeight: 18 },
});
