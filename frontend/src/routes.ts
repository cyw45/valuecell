import {
  index,
  layout,
  prefix,
  type RouteConfig,
  route,
} from "@react-router/dev/routes";

export default [
  index("app/redirect-to-home.tsx"),

  route("/login", "app/login.tsx"),

  route("/dashboard", "app/dashboard.tsx"),
  route("/charts", "app/charts.tsx"),
  route("/strategies", "app/strategies/strategies.tsx"),
  route("/trades", "app/trades.tsx"),
  route("/funding", "app/funding.tsx"),
  route("/settings", "app/settings.tsx"),
  route("/settings/sandbox-exchanges", "app/settings/sandbox-exchanges.tsx"),
  route("/settings/live-execution", "app/settings/live-execution.tsx"),

  // Legacy routes remain reachable but are intentionally excluded from primary navigation.
  ...prefix("/home", [
    layout("app/home/_layout.tsx", [
      index("app/home/home.tsx"),
      route("/stock/:stockId", "app/home/stock.tsx"),
    ]),
  ]),
  route("/market", "app/market/agents.tsx"),
  route("/research/polymarket", "app/research/polymarket.tsx"),
  ...prefix("/agent", [
    route("/:agentName", "app/agent/chat.tsx"),
    route("/:agentName/config", "app/agent/config.tsx"),
  ]),
  ...prefix("/setting", [
    layout("app/setting/_layout.tsx", [
      index("app/setting/models.tsx"),
      route("/general", "app/setting/general.tsx"),
      route("/memory", "app/setting/memory.tsx"),
    ]),
  ]),
  route("/test", "app/test.tsx"),
] satisfies RouteConfig;
