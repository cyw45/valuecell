import { useMemo } from "react";
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
import { Activity, Database, RadioTower, RefreshCw, ShieldCheck } from "lucide-react-native";
import { api } from "../api";
import { palette, radius, spacing } from "../theme";
import type { WorldIntelligenceSnapshot } from "../types";

const FEEDS: ReadonlyArray<{ feed: string; label: string }> = [
  { feed: "cross_source_signals", label: "跨源信号" },
  { feed: "market_implications", label: "市场影响" },
  { feed: "risk_scores", label: "国家风险" },
  { feed: "thermal_escalations", label: "热度升级" },
];

function formatTimestamp(value: string | null | undefined) {
  if (!value) return "等待首次采集";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "时间未提供";
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function evidenceCount(snapshot: WorldIntelligenceSnapshot) {
  if (!snapshot.payload || typeof snapshot.payload !== "object") return "Evidence captured";
  const payload = snapshot.payload as Record<string, unknown>;
  for (const key of ["cards", "clusters", "signals", "countries", "scores"]) {
    if (Array.isArray(payload[key])) return `${payload[key].length} source records`;
  }
  return "Evidence captured";
}

export default function WorldMonitorScreen() {
  const status = useQuery({
    queryKey: ["mobile", "world-intelligence", "status"],
    queryFn: () => api.worldIntelligenceStatus(),
    refetchInterval: 60_000,
  });
  const snapshots = useQuery({
    queryKey: ["mobile", "world-intelligence", "snapshots", 12],
    queryFn: () => api.worldIntelligenceSnapshots({ limit: 12 }),
    refetchInterval: 60_000,
  });
  const feedStatus = useMemo(
    () =>
      FEEDS.map((definition) => ({
        ...definition,
        latestSnapshotAt:
          status.data?.feeds.find((item) => item.feed === definition.feed)
            ?.latest_snapshot_at ?? null,
      })),
    [status.data?.feeds],
  );
  const refreshing = status.isRefetching || snapshots.isRefetching;
  const loading = status.isLoading || snapshots.isLoading;
  const error = status.error ?? snapshots.error;

  const refresh = () => {
    void Promise.all([status.refetch(), snapshots.refetch()]);
  };

  return (
    <ScrollView
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl onRefresh={refresh} refreshing={refreshing} tintColor={palette.primary} />}
      style={styles.page}
    >
      <View style={styles.header}>
        <View style={styles.headerTitleRow}>
          <RadioTower color={palette.primary} size={22} />
          <Text style={styles.title}>全球情报</Text>
          <View style={styles.sourceBadge}>
            <ShieldCheck color={palette.primary} size={14} />
            <Text style={styles.sourceBadgeText}>来源可追溯</Text>
          </View>
        </View>
        <Text style={styles.subtitle}>ValueCell 已保存的 WorldMonitor 证据，用于研究与策略背景，不执行交易。</Text>
        {status.data ? (
          <Text style={styles.statusText}>
            {status.data.enabled ? "采集服务已启用 · 每 60 秒自动更新" : "采集服务当前未启用 · 仍可查看已保存证据"}
          </Text>
        ) : null}
      </View>

      <View style={styles.feedGrid}>
        {feedStatus.map((feed) => (
          <View key={feed.feed} style={styles.feedCard}>
            <Text style={styles.feedLabel}>{feed.label}</Text>
            <Text style={styles.feedTimestamp}>{formatTimestamp(feed.latestSnapshotAt)}</Text>
          </View>
        ))}
      </View>

      <View style={styles.sectionHeader}>
        <View style={styles.sectionTitleRow}>
          <Database color={palette.primary} size={18} />
          <Text style={styles.sectionTitle}>最新证据快照</Text>
        </View>
        <Pressable accessibilityLabel="刷新全球情报" accessibilityRole="button" onPress={refresh} style={styles.refreshButton}>
          <RefreshCw color={palette.primary} size={17} />
          <Text style={styles.refreshText}>{refreshing ? "刷新中" : "刷新"}</Text>
        </Pressable>
      </View>

      {loading && !snapshots.data ? (
        <View style={styles.statePanel}>
          <ActivityIndicator color={palette.primary} />
          <Text style={styles.mutedText}>正在加载已存储的情报证据…</Text>
        </View>
      ) : null}
      {error ? (
        <View style={styles.errorPanel}>
          <Text style={styles.errorTitle}>情报证据暂不可用</Text>
          <Text style={styles.errorText}>{error instanceof Error ? error.message : "无法加载全球情报。"}</Text>
          <Pressable accessibilityRole="button" onPress={refresh} style={styles.retryButton}>
            <RefreshCw color={palette.canvas} size={17} />
            <Text style={styles.retryText}>重试</Text>
          </Pressable>
        </View>
      ) : null}
      {!loading && !error && snapshots.data?.snapshots.length === 0 ? (
        <View style={styles.waitingPanel}>
          <Activity color={palette.textMuted} size={20} />
          <View style={styles.waitingCopy}>
            <Text style={styles.waitingTitle}>等待首个 WorldMonitor 采集周期</Text>
            <Text style={styles.mutedText}>证据快照到达后会保留来源类别、采集时间与记录数量。</Text>
          </View>
        </View>
      ) : null}
      {!error
        ? snapshots.data?.snapshots.map((snapshot) => {
            const feed = FEEDS.find((item) => item.feed === snapshot.feed);
            return (
              <View key={snapshot.id} style={styles.snapshotCard}>
                <View style={styles.snapshotMain}>
                  <Text style={styles.snapshotFeed}>{feed?.label ?? snapshot.feed}</Text>
                  <Text style={styles.snapshotEvidence}>{evidenceCount(snapshot)}</Text>
                  <Text style={styles.snapshotAttribution}>Source-attributed evidence · 快照 #{snapshot.id}</Text>
                </View>
                <Text style={styles.snapshotTime}>{formatTimestamp(snapshot.captured_at)}</Text>
              </View>
            );
          })
        : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  header: { gap: spacing.xs, paddingTop: spacing.sm },
  headerTitleRow: { alignItems: "center", flexDirection: "row", flexWrap: "wrap", gap: spacing.xs },
  title: { color: palette.text, fontSize: 27, fontWeight: "800", letterSpacing: -0.8 },
  subtitle: { color: palette.textMuted, fontSize: 13, lineHeight: 20 },
  statusText: { color: palette.primary, fontSize: 11, fontWeight: "800" },
  sourceBadge: { alignItems: "center", backgroundColor: palette.primarySoft, borderColor: palette.border, borderRadius: radius.pill, borderWidth: 1, flexDirection: "row", gap: 5, paddingHorizontal: 8, paddingVertical: 5 },
  sourceBadgeText: { color: palette.primary, fontSize: 10, fontWeight: "800" },
  feedGrid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  feedCard: { backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexGrow: 1, gap: 7, minWidth: "44%", padding: spacing.sm },
  feedLabel: { color: palette.text, fontSize: 13, fontWeight: "800" },
  feedTimestamp: { color: palette.textMuted, fontSize: 11, lineHeight: 17 },
  sectionHeader: { alignItems: "center", flexDirection: "row", justifyContent: "space-between", marginTop: spacing.xs },
  sectionTitleRow: { alignItems: "center", flexDirection: "row", gap: spacing.xs },
  sectionTitle: { color: palette.text, fontSize: 16, fontWeight: "800" },
  refreshButton: { alignItems: "center", borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: 6, justifyContent: "center", minHeight: 44, minWidth: 82, paddingHorizontal: spacing.sm },
  refreshText: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  statePanel: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, justifyContent: "center", minHeight: 180, padding: spacing.md },
  mutedText: { color: palette.textMuted, fontSize: 13, lineHeight: 20 },
  errorPanel: { alignItems: "flex-start", backgroundColor: palette.negativeSoft, borderColor: palette.negative, borderRadius: radius.md, borderWidth: 1, gap: spacing.sm, padding: spacing.md },
  errorTitle: { color: palette.negative, fontSize: 16, fontWeight: "800" },
  errorText: { color: palette.text, fontSize: 13, lineHeight: 20 },
  retryButton: { alignItems: "center", backgroundColor: palette.primary, borderRadius: radius.sm, flexDirection: "row", gap: spacing.xs, justifyContent: "center", minHeight: 44, paddingHorizontal: spacing.md },
  retryText: { color: palette.canvas, fontSize: 13, fontWeight: "800" },
  waitingPanel: { alignItems: "center", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderStyle: "dashed", borderWidth: 1, flexDirection: "row", gap: spacing.sm, padding: spacing.md },
  waitingCopy: { flex: 1, gap: 3 },
  waitingTitle: { color: palette.text, fontSize: 14, fontWeight: "800" },
  snapshotCard: { alignItems: "flex-start", backgroundColor: palette.surface, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, justifyContent: "space-between", padding: spacing.md },
  snapshotMain: { flex: 1, gap: 4 },
  snapshotFeed: { color: palette.text, fontSize: 14, fontWeight: "800" },
  snapshotEvidence: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  snapshotAttribution: { color: palette.textMuted, fontSize: 11, lineHeight: 17 },
  snapshotTime: { color: palette.textMuted, fontSize: 11, maxWidth: 96, textAlign: "right" },
});
