import { StyleSheet, Text, View } from "react-native";
import type { RuleStrategyEvaluation } from "../types";
import { useTheme } from "../theme-context";
import { radius, spacing } from "../theme";
import {
  conditionCategoryLabel,
  conditionDetail,
  conditionFacts,
  conditionLabel,
  conditionStateLabel,
  conditionStateSummary,
  conditionStateTone,
  executionFunnelFacts,
  evaluationReason,
  strategyActionLabel,
  strategyActionTone,
  type StrategyPresentationTone,
} from "../screens/strategy-presentation";

export type StrategyEvaluationPanelProps = {
  evaluation: RuleStrategyEvaluation;
  compact?: boolean;
};

function toneColors(
  tone: StrategyPresentationTone,
  tokens: ReturnType<typeof useTheme>["tokens"],
): { color: string; soft: string } {
  if (tone === "positive") return { color: tokens.positive, soft: tokens.positiveSoft };
  if (tone === "negative") return { color: tokens.negative, soft: tokens.negativeSoft };
  if (tone === "warning") return { color: tokens.warning, soft: tokens.warningSoft };
  return { color: tokens.textMuted, soft: tokens.surfaceRaised };
}

export function StrategyEvaluationPanel({ evaluation, compact = false }: StrategyEvaluationPanelProps) {
  const { tokens } = useTheme();
  const decisionTone = strategyActionTone(evaluation.action);
  const decisionColors = toneColors(decisionTone, tokens);
  const funnel = executionFunnelFacts(evaluation);
  const styles = StyleSheet.create({
    root: { gap: spacing.md },
    decision: { backgroundColor: decisionColors.soft, borderColor: decisionColors.color, borderRadius: radius.md, borderWidth: 1, gap: spacing.xs, padding: spacing.sm },
    eyebrow: { color: tokens.textMuted, fontSize: 11, fontWeight: "800", letterSpacing: 0.5 },
    decisionRow: { alignItems: "center", flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
    action: { color: decisionColors.color, fontSize: 18, fontWeight: "900" },
    machineCode: { color: tokens.textMuted, fontSize: 11, fontWeight: "700" },
    reason: { color: tokens.text, fontSize: 14, lineHeight: 21 },
    funnel: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
    funnelItem: { backgroundColor: tokens.surfaceRaised, borderColor: tokens.border, borderRadius: radius.sm, borderWidth: 1, flexBasis: "47%", flexGrow: 1, gap: spacing.xxs, minWidth: 130, padding: spacing.sm },
    funnelLabel: { color: tokens.textMuted, fontSize: 11, fontWeight: "800" },
    funnelValue: { fontSize: 14, fontWeight: "800", lineHeight: 20 },
    funnelCaption: { color: tokens.textMuted, fontSize: 11, lineHeight: 16 },
    conditionsHeader: { gap: spacing.xxs },
    conditionsTitle: { color: tokens.text, fontSize: 16, fontWeight: "800" },
    conditionsSummary: { color: tokens.textMuted, fontSize: 12, lineHeight: 18 },
    conditions: { gap: spacing.xs },
    condition: { borderColor: tokens.border, borderRadius: radius.sm, borderWidth: 1, gap: spacing.xs, padding: spacing.sm },
    conditionHeader: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
    conditionTitle: { color: tokens.text, flex: 1, fontSize: 14, fontWeight: "800" },
    conditionMeta: { color: tokens.textMuted, fontSize: 10, fontWeight: "700" },
    stateBadge: { borderRadius: radius.pill, paddingHorizontal: spacing.xs, paddingVertical: spacing.xxs },
    stateText: { fontSize: 11, fontWeight: "900" },
    conditionDetail: { color: tokens.text, fontSize: 13, lineHeight: 20 },
    facts: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
    fact: { backgroundColor: tokens.canvas, borderRadius: radius.sm, gap: 2, minWidth: 104, paddingHorizontal: spacing.xs, paddingVertical: spacing.xs },
    factLabel: { color: tokens.textMuted, fontSize: 10, fontWeight: "700" },
    factValue: { color: tokens.text, fontSize: 12, fontWeight: "800" },
    empty: { color: tokens.textMuted, fontSize: 13, lineHeight: 20 },
  });

  return (
    <View style={styles.root}>
      <View style={styles.decision}>
        <Text style={styles.eyebrow}>本次服务端决策</Text>
        <View style={styles.decisionRow}>
          <Text style={styles.action}>{strategyActionLabel(evaluation.action)}</Text>
          <Text style={styles.machineCode}>action: {evaluation.action}</Text>
        </View>
        <Text style={styles.reason}>{evaluationReason(evaluation.reason_code, evaluation.reason)}</Text>
        <Text style={styles.machineCode}>reason_code: {evaluation.reason_code}</Text>
        {evaluation.status || evaluation.stage || evaluation.blocked_stage ? <Text style={styles.machineCode}>状态：{evaluation.status ?? "—"} · 阶段：{evaluation.stage ?? "—"}{evaluation.blocked_stage ? ` · 阻塞：${evaluation.blocked_stage}` : ""}</Text> : null}
      </View>

      {!compact ? (
        <View style={styles.funnel}>
          {funnel.map((fact) => {
            const colors = toneColors(fact.tone, tokens);
            return (
              <View key={fact.label} style={[styles.funnelItem, { borderColor: colors.color }]}>
                <Text style={styles.funnelLabel}>{fact.label}</Text>
                <Text style={[styles.funnelValue, { color: colors.color }]}>{fact.value}</Text>
                <Text style={styles.funnelCaption}>{fact.caption}</Text>
              </View>
            );
          })}
        </View>
      ) : null}

      <View style={styles.conditionsHeader}>
        <Text style={styles.conditionsTitle}>本次条件</Text>
        <Text style={styles.conditionsSummary}>{evaluation.condition_summary ? `通过 ${evaluation.condition_summary.matched}/${evaluation.condition_summary.total}，要求 ${evaluation.condition_summary.required} 项，${evaluation.condition_summary.available} 项数据可用` : conditionStateSummary(evaluation.conditions)}</Text>
      </View>
      {evaluation.conditions.length === 0 ? (
        <Text style={styles.empty}>本次服务端评估未返回条件记录。</Text>
      ) : (
        <View style={styles.conditions}>
          {evaluation.conditions.map((condition, index) => {
            const stateTone = conditionStateTone(condition.state);
            const stateColors = toneColors(stateTone, tokens);
            const facts = conditionFacts(condition);
            return (
              <View key={`${condition.code}-${index}`} style={[styles.condition, { borderColor: stateColors.color }]}>
                <View style={styles.conditionHeader}>
                  <Text numberOfLines={1} style={styles.conditionTitle}>{conditionLabel(condition.code)}</Text>
                  <View style={[styles.stateBadge, { backgroundColor: stateColors.soft }]}>
                    <Text style={[styles.stateText, { color: stateColors.color }]}>{conditionStateLabel(condition.state)}</Text>
                  </View>
                </View>
                <Text style={styles.conditionMeta}>{conditionCategoryLabel(condition.category)} · code: {condition.code}</Text>
                <Text style={styles.conditionDetail}>{conditionDetail(condition)}</Text>
                {facts.length ? (
                  <View style={styles.facts}>
                    {facts.map((fact) => (
                      <View key={fact.label} style={styles.fact}>
                        <Text style={styles.factLabel}>{fact.label}</Text>
                        <Text numberOfLines={1} style={styles.factValue}>{fact.value}</Text>
                      </View>
                    ))}
                  </View>
                ) : null}
              </View>
            );
          })}
        </View>
      )}
    </View>
  );
}
