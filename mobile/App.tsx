import { ActivityIndicator, StyleSheet, Text, View } from "react-native";
import { DefaultTheme, NavigationContainer, type Theme } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { ChartCandlestick, LayoutDashboard, Settings2, SlidersHorizontal } from "lucide-react-native";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StatusBar } from "expo-status-bar";
import { SessionProvider, useSession } from "./src/session";
import { palette } from "./src/theme";
import AuthScreen from "./src/screens/AuthScreen";
import HomeScreen from "./src/screens/HomeScreen";
import MarketScreen from "./src/screens/MarketScreen";
import StrategiesScreen from "./src/screens/StrategiesScreen";
import AccountScreen from "./src/screens/AccountScreen";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 15_000, retry: 1 } },
});

const navigationTheme: Theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: palette.primary,
    background: palette.canvas,
    card: palette.surface,
    text: palette.text,
    border: palette.border,
    notification: palette.negative,
  },
};

const Tab = createBottomTabNavigator();

function WorkspaceNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: palette.primary,
        tabBarInactiveTintColor: palette.textMuted,
        tabBarStyle: styles.tabBar,
        tabBarLabelStyle: styles.tabLabel,
        tabBarIcon: ({ color, size }) => {
          const icons = { 工作台: LayoutDashboard, 策略: SlidersHorizontal, 行情: ChartCandlestick, 账户: Settings2 } as const;
          const Icon = icons[route.name as keyof typeof icons];
          return <Icon color={color} size={size} strokeWidth={2.1} />;
        },
      })}
    >
      <Tab.Screen component={HomeScreen} name="工作台" />
      <Tab.Screen component={StrategiesScreen} name="策略" />
      <Tab.Screen component={MarketScreen} name="行情" />
      <Tab.Screen component={AccountScreen} name="账户" />
    </Tab.Navigator>
  );
}

function AppContent() {
  const { ready, session } = useSession();
  if (!ready) {
    return <View style={styles.splash}><ActivityIndicator color={palette.primary} size="large" /><Text style={styles.splashText}>正在恢复安全会话</Text></View>;
  }
  return session ? <WorkspaceNavigator /> : <AuthScreen />;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SessionProvider>
        <NavigationContainer theme={navigationTheme}>
          <StatusBar style="light" />
          <AppContent />
        </NavigationContainer>
      </SessionProvider>
    </QueryClientProvider>
  );
}

const styles = StyleSheet.create({
  splash: { alignItems: "center", backgroundColor: palette.canvas, flex: 1, gap: 16, justifyContent: "center" },
  splashText: { color: palette.textMuted, fontSize: 14 },
  tabBar: { backgroundColor: palette.surface, borderTopColor: palette.border, height: 72, paddingBottom: 8, paddingTop: 7 },
  tabLabel: { fontSize: 11, fontWeight: "700" },
});
