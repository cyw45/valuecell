import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { useNavigation, useRoute, type RouteProp } from "@react-navigation/native";
import { ChartCandlestick, Search } from "lucide-react-native";
import { api } from "../api";
import { SectionCard, StatePanel } from "../components";
import type { WorkbenchStackParamList } from "../navigation/types";
import { useSession } from "../session";
import { palette, radius, spacing } from "../theme";

type Route = RouteProp<WorkbenchStackParamList, "StrategySymbols">;

export default function StrategySymbolsScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<Route>();
  const { session } = useSession();
  const [filter, setFilter] = useState("");
  const strategy = useQuery({
    queryKey: ["mobile", session?.tenantId, "strategy", route.params.strategyId],
    queryFn: () => api.strategy(route.params.strategyId),
    enabled: Boolean(session && route.params.strategyId),
  });
  const symbols = useMemo(() => {
    const query = filter.trim().toUpperCase();
    const configured = strategy.data?.config.symbols ?? [];
    return query ? configured.filter((symbol) => symbol.includes(query)) : configured;
  }, [filter, strategy.data?.config.symbols]);

  if (strategy.isLoading) {
    return <StatePanel description="正在读取策略观察池。" title="观察标的" />;
  }
  if (strategy.isError || !strategy.data) {
    return <StatePanel actionLabel="重试" description={(strategy.error as Error)?.message ?? "无法读取策略观察池。"} onAction={() => void strategy.refetch()} title="观察标的暂不可用" tone="error" />;
  }

  return (
    <ScrollView contentContainerStyle={styles.content} style={styles.page}>
      <SectionCard description={`${strategy.data.name} · 共 ${strategy.data.config.symbols.length} 个策略观察标的`} title="全部观察标的">
        <View style={styles.searchBox}>
          <Search color={palette.textMuted} size={18} />
          <TextInput accessibilityLabel="搜索策略观察标的" autoCapitalize="characters" onChangeText={setFilter} placeholder="搜索 BTC 或 BTC-USDT" placeholderTextColor={palette.textMuted} style={styles.searchInput} value={filter} />
        </View>
        <View style={styles.list}>
          {symbols.map((symbol) => (
            <Pressable accessibilityLabel={`查看 ${symbol} 行情`} accessibilityRole="button" key={symbol} onPress={() => navigation.navigate("行情", { screen: "Market", params: { strategyId: route.params.strategyId, symbol } })} style={({ pressed }) => [styles.row, pressed && styles.pressed]}>
              <View style={styles.icon}><ChartCandlestick color={palette.primary} size={19} /></View>
              <View style={styles.copy}><Text style={styles.symbol}>{symbol.replace("-", "/")}</Text><Text style={styles.detail}>打开该标的的服务端行情与技术指标</Text></View>
              <Text style={styles.action}>查看</Text>
            </Pressable>
          ))}
        </View>
        {symbols.length === 0 ? <Text style={styles.empty}>没有匹配的观察标的。</Text> : null}
      </SectionCard>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { backgroundColor: palette.canvas, flex: 1 },
  content: { gap: spacing.md, padding: spacing.md, paddingBottom: spacing.xl },
  searchBox: { alignItems: "center", backgroundColor: palette.canvas, borderColor: palette.border, borderRadius: radius.sm, borderWidth: 1, flexDirection: "row", gap: spacing.xs, paddingHorizontal: spacing.sm },
  searchInput: { color: palette.text, flex: 1, fontSize: 14, height: 46 },
  list: { gap: spacing.xs },
  row: { alignItems: "center", backgroundColor: palette.surfaceMuted, borderColor: palette.border, borderRadius: radius.md, borderWidth: 1, flexDirection: "row", gap: spacing.sm, minHeight: 64, padding: spacing.sm },
  icon: { alignItems: "center", backgroundColor: palette.primarySoft, borderRadius: radius.sm, height: 38, justifyContent: "center", width: 38 },
  copy: { flex: 1, gap: 2 },
  symbol: { color: palette.text, fontSize: 15, fontWeight: "900" },
  detail: { color: palette.textMuted, fontSize: 12, lineHeight: 17 },
  action: { color: palette.primary, fontSize: 12, fontWeight: "800" },
  empty: { color: palette.textMuted, fontSize: 13, paddingVertical: spacing.lg, textAlign: "center" },
  pressed: { opacity: 0.76 },
});
