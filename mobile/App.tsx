import { useMemo } from "react";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";
import { DefaultTheme, NavigationContainer, type Theme } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator, type NativeStackNavigationOptions } from "@react-navigation/native-stack";
import { ChartCandlestick, LayoutDashboard, Settings2, SlidersHorizontal } from "lucide-react-native";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StatusBar } from "expo-status-bar";
import { I18nProvider, useI18n } from "./src/i18n";
import {
  type AccountStackParamList,
  type MarketStackParamList,
  type PostAuthTab,
  type StrategyStackParamList,
  type WorkbenchStackParamList,
  type WorkspaceTabParamList,
} from "./src/navigation/types";
import { PreferencesProvider } from "./src/preferences";
import { SessionProvider, useSession } from "./src/session";
import { spacing } from "./src/theme";
import { useTheme, ThemeProvider } from "./src/theme-context";
import AuthScreen from "./src/screens/AuthScreen";
import AccountScreen from "./src/screens/AccountScreen";
import ChangePasswordScreen from "./src/screens/ChangePasswordScreen";
import FundingPnlScreen from "./src/screens/FundingPnlScreen";
import ExecutionFactsScreen from "./src/screens/ExecutionFactsScreen";
import LiveExecutionScreen from "./src/screens/LiveExecutionScreen";
import MarketScreen from "./src/screens/MarketScreen";
import PlatformAdminScreen from "./src/screens/PlatformAdminScreen";
import PolymarketScreen from "./src/screens/PolymarketScreen";
import PreferencesScreen from "./src/screens/PreferencesScreen";
import SandboxConnectionDetailScreen from "./src/screens/SandboxConnectionDetailScreen";
import SandboxConnectionEditorScreen from "./src/screens/SandboxConnectionEditorScreen";
import SandboxConnectionsScreen from "./src/screens/SandboxConnectionsScreen";
import StrategyAdvisoryScreen from "./src/screens/StrategyAdvisoryScreen";
import StrategyDetailScreen from "./src/screens/StrategyDetailScreen";
import StrategyEditorScreen from "./src/screens/StrategyEditorScreen";
import StrategyListScreen from "./src/screens/StrategyListScreen";
import StrategyOverviewScreen from "./src/screens/StrategyOverviewScreen";
import StrategyPositionsScreen from "./src/screens/StrategyPositionsScreen";
import StrategyWorkbenchDetailScreen from "./src/screens/StrategyWorkbenchDetailScreen";
import StrategySymbolsScreen from "./src/screens/StrategySymbolsScreen";
import TradeLedgerScreen from "./src/screens/TradeLedgerScreen";
import WorkspaceAuditScreen from "./src/screens/WorkspaceAuditScreen";
import WorkspaceBillingScreen from "./src/screens/WorkspaceBillingScreen";
import WorkspaceMembersScreen from "./src/screens/WorkspaceMembersScreen";
import WorldMonitorScreen from "./src/screens/WorldMonitorScreen";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 15_000, retry: 1 } },
});

const Tab = createBottomTabNavigator<WorkspaceTabParamList>();
const WorkbenchStack = createNativeStackNavigator<WorkbenchStackParamList>();
const StrategyStack = createNativeStackNavigator<StrategyStackParamList>();
const MarketStack = createNativeStackNavigator<MarketStackParamList>();
const AccountStack = createNativeStackNavigator<AccountStackParamList>();

function useStackHeaderOptions(): NativeStackNavigationOptions {
  const { t } = useI18n();
  const { tokens } = useTheme();
  return useMemo<NativeStackNavigationOptions>(() => ({
    headerShown: false,
    headerBackTitle: t("common.back"),
    headerShadowVisible: false,
    headerStyle: { backgroundColor: tokens.surface },
    headerTintColor: tokens.text,
    headerTitleStyle: { color: tokens.text, fontSize: 17, fontWeight: "800" },
  }), [t, tokens]);
}

function WorkbenchNavigator() {
  const { t } = useI18n();
  const screenOptions = useStackHeaderOptions();
  return (
    <WorkbenchStack.Navigator screenOptions={screenOptions}>
      <WorkbenchStack.Screen component={StrategyOverviewScreen} name="StrategyOverview" />
      <WorkbenchStack.Screen component={StrategySymbolsScreen} name="StrategySymbols" options={{ headerShown: true, title: "观察标的" }} />
      <WorkbenchStack.Screen component={StrategyPositionsScreen} name="StrategyPositions" options={{ headerShown: true, title: "我的持仓" }} />
      <WorkbenchStack.Screen component={StrategyWorkbenchDetailScreen} name="StrategyWorkbenchDetail" options={{ headerShown: true, title: "策略详情" }} />
      <WorkbenchStack.Screen component={ExecutionFactsScreen} name="ExecutionFacts" options={{ headerShown: true, title: "执行详情" }} />
      <WorkbenchStack.Screen component={TradeLedgerScreen} name="TradeLedger" options={{ headerShown: true, title: t("navigation.tradeLedger") }} />
      <WorkbenchStack.Screen component={FundingPnlScreen} name="FundingPnl" options={{ headerShown: true, title: t("navigation.fundingPnl") }} />
    </WorkbenchStack.Navigator>
  );
}

function StrategyNavigator() {
  const { t } = useI18n();
  const screenOptions = useStackHeaderOptions();
  return (
    <StrategyStack.Navigator screenOptions={screenOptions}>
      <StrategyStack.Screen component={StrategyListScreen} name="StrategyList" />
      <StrategyStack.Screen component={StrategyDetailScreen} name="StrategyDetail" options={{ headerShown: true, title: t("navigation.strategyDetail") }} />
      <StrategyStack.Screen component={StrategyEditorScreen} name="StrategyEditor" options={{ headerShown: true, presentation: "modal", title: t("navigation.strategyEditor") }} />
      <StrategyStack.Screen component={StrategyAdvisoryScreen} name="StrategyAdvisory" options={{ headerShown: true, title: t("navigation.strategyAdvisory") }} />
    </StrategyStack.Navigator>
  );
}

function MarketNavigator() {
  const { t } = useI18n();
  const screenOptions = useStackHeaderOptions();
  return (
    <MarketStack.Navigator screenOptions={screenOptions}>
      <MarketStack.Screen component={MarketScreen} name="Market" />
      <MarketStack.Screen component={WorldMonitorScreen} name="WorldMonitor" options={{ headerShown: true, title: t("navigation.worldMonitor") }} />
      <MarketStack.Screen component={PolymarketScreen} name="Polymarket" options={{ headerShown: true, title: t("navigation.polymarket") }} />
    </MarketStack.Navigator>
  );
}

function AccountNavigator() {
  const { t } = useI18n();
  const screenOptions = useStackHeaderOptions();
  return (
    <AccountStack.Navigator screenOptions={screenOptions}>
      <AccountStack.Screen component={AccountScreen} name="Account" />
      <AccountStack.Screen component={PreferencesScreen} name="Preferences" options={{ headerShown: true, title: t("preferences.title") }} />
      <AccountStack.Screen component={ChangePasswordScreen} name="ChangePassword" options={{ headerShown: true, title: "修改密码" }} />
      <AccountStack.Screen component={SandboxConnectionsScreen} name="SandboxConnections" options={{ headerShown: true, title: t("navigation.sandboxConnections") }} />
      <AccountStack.Screen component={SandboxConnectionEditorScreen} name="SandboxConnectionEditor" options={{ headerShown: true, presentation: "modal", title: t("navigation.sandboxConnectionEditor") }} />
      <AccountStack.Screen component={SandboxConnectionDetailScreen} name="SandboxConnectionDetail" options={{ headerShown: true, title: t("navigation.sandboxConnectionDetail") }} />
      <AccountStack.Screen component={LiveExecutionScreen} name="LiveExecution" options={{ headerShown: true, title: t("navigation.liveExecution") }} />
      <AccountStack.Screen component={WorkspaceMembersScreen} name="WorkspaceMembers" options={{ headerShown: true, title: t("navigation.workspaceMembers") }} />
      <AccountStack.Screen component={WorkspaceBillingScreen} name="WorkspaceBilling" options={{ headerShown: true, title: t("navigation.workspaceBilling") }} />
      <AccountStack.Screen component={WorkspaceAuditScreen} name="WorkspaceAudit" options={{ headerShown: true, title: t("navigation.workspaceAudit") }} />
      <AccountStack.Screen component={PlatformAdminScreen} name="PlatformAdmin" options={{ headerShown: true, title: t("navigation.platformAdmin") }} />
    </AccountStack.Navigator>
  );
}

function WorkspaceNavigator({ initialRouteName }: { initialRouteName: PostAuthTab }) {
  const { tokens } = useTheme();
  const styles = useMemo(() => StyleSheet.create({
    tabBar: { backgroundColor: tokens.surface, borderTopColor: tokens.border, height: 72, paddingBottom: spacing.xs, paddingTop: 7 },
    tabLabel: { fontSize: 11, fontWeight: "700" },
  }), [tokens]);

  return (
    <Tab.Navigator
      initialRouteName={initialRouteName}
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: tokens.primary,
        tabBarInactiveTintColor: tokens.textMuted,
        tabBarLabelStyle: styles.tabLabel,
        tabBarStyle: styles.tabBar,
        tabBarIcon: ({ color, size }) => {
          const icons = { 工作台: LayoutDashboard, 策略: SlidersHorizontal, 行情: ChartCandlestick, 我的: Settings2 } as const;
          const Icon = icons[route.name as keyof typeof icons];
          return <Icon color={color} size={size} strokeWidth={2.1} />;
        },
      })}
    >
      <Tab.Screen component={WorkbenchNavigator} name="工作台" />
      <Tab.Screen component={StrategyNavigator} name="策略" />
      <Tab.Screen component={MarketNavigator} name="行情" />
      <Tab.Screen component={AccountNavigator} name="我的" />
    </Tab.Navigator>
  );
}

function AppContent() {
  const { t } = useI18n();
  const { tokens } = useTheme();
  const { postAuthTab, ready, session } = useSession();
  const styles = useMemo(() => StyleSheet.create({
    splash: { alignItems: "center", backgroundColor: tokens.canvas, flex: 1, gap: spacing.md, justifyContent: "center" },
    splashText: { color: tokens.textMuted, fontSize: 14 },
  }), [tokens]);

  if (!ready) {
    return <View style={styles.splash}><ActivityIndicator color={tokens.primary} size="large" /><Text style={styles.splashText}>{t("app.restoringSession")}</Text></View>;
  }
  return session ? <WorkspaceNavigator initialRouteName={postAuthTab} /> : <AuthScreen />;
}

function NavigationRoot() {
  const { isDark, tokens } = useTheme();
  const navigationTheme = useMemo<Theme>(() => ({
    ...DefaultTheme,
    dark: isDark,
    colors: {
      ...DefaultTheme.colors,
      primary: tokens.primary,
      background: tokens.canvas,
      card: tokens.surface,
      text: tokens.text,
      border: tokens.border,
      notification: tokens.negative,
    },
  }), [isDark, tokens]);

  return (
    <NavigationContainer theme={navigationTheme}>
      <StatusBar style={isDark ? "light" : "dark"} />
      <AppContent />
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SessionProvider>
        <PreferencesProvider>
          <I18nProvider>
            <ThemeProvider>
              <NavigationRoot />
            </ThemeProvider>
          </I18nProvider>
        </PreferencesProvider>
      </SessionProvider>
    </QueryClientProvider>
  );
}
