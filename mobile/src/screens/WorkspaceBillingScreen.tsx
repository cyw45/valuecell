import { useQuery } from "@tanstack/react-query";
import { ScrollView, StyleSheet, Text, View } from "react-native";
import { api } from "../api";
import { canMutate } from "../access";
import { ScreenHeader, SectionCard, StatePanel } from "../components";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";
import { formatQuote } from "./workbench";

export default function WorkspaceBillingScreen() {
  const { session } = useSession();
  const access = useQuery({ queryKey: ["mobile", session?.tenantId, "access"], queryFn: api.access, enabled: Boolean(session) });
  const billing = useQuery({ queryKey: ["mobile", session?.tenantId, "tenant-billing"], queryFn: api.tenantBilling, enabled: Boolean(session) && canMutate(access.data, "billing.manage") });
  if (access.isLoading) return <StatePanel message="正在读取工作区访问权限。" state="loading" title="账单与合约" />;
  if (!canMutate(access.data, "billing.manage")) return <StatePanel message={access.data?.status === "active" ? "当前角色不具备账单查看权限。" : "服务未激活，账单访问已禁用。"} state="empty" title="账单与合约" />;
  if (!billing.data) return <StatePanel message="暂未收到当前租户的账单数据。" state="empty" title="账单与合约" />;
  const data = billing.data;
  return <ScrollView contentContainerStyle={styles.content} style={styles.page}><ScreenHeader subtitle={`${data.access.commercial_model ?? "未开通"} · ${data.access.status}`} title="账单与合约" /><SectionCard title="订阅">{data.subscriptions.length ? data.subscriptions.map((item) => <View key={item.id} style={styles.row}><Text style={styles.primary}>{item.status}</Text><Text style={styles.secondary}>{item.starts_at} 至 {item.ends_at}</Text>{item.note ? <Text style={styles.secondary}>{item.note}</Text> : null}</View>) : <Text style={styles.empty}>没有订阅记录。</Text>}</SectionCard><SectionCard title="企业合约">{data.agreement ? <View style={styles.row}><Text style={styles.primary}>{data.agreement.agreement_number} · {data.agreement.status}</Text><Text style={styles.secondary}>分成 {data.agreement.revenue_share_rate} · 结算周期 {data.agreement.settlement_cycle_days} 天</Text><Text style={styles.secondary}>高水位线 {data.agreement.high_water_mark_quote}</Text></View> : <Text style={styles.empty}>当前租户没有企业分成合约。</Text>}</SectionCard><SectionCard title="结算">{data.settlements.length ? data.settlements.map((item) => <View key={item.id} style={styles.row}><Text style={styles.primary}>{item.status} · {formatQuote(Number(item.amount_due_quote))}</Text><Text style={styles.secondary}>{item.period_started_at} 至 {item.period_ended_at}</Text></View>) : <Text style={styles.empty}>没有利润分成结算记录。</Text>}</SectionCard></ScrollView>;
}
const styles = StyleSheet.create({ page: { backgroundColor: palette.canvas, flex: 1 }, content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl }, row: { borderTopColor: palette.border, borderTopWidth: 1, gap: spacing.xs, paddingVertical: spacing.sm }, primary: { color: palette.text, fontSize: 14, fontWeight: "800" }, secondary: { color: palette.textMuted, fontSize: 12, lineHeight: 19 }, empty: { color: palette.textMuted, fontSize: 13 }, });
