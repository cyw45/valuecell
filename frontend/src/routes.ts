import {
  index,
  type RouteConfig,
  route,
} from "@react-router/dev/routes";

export default [
  index("app/redirect-to-home.tsx"),

  route("/login", "app/login.tsx"),

  route("/dashboard", "app/dashboard.tsx"),
  route("/charts", "app/charts.tsx"),
  route("/strategies", "app/strategies/strategies.tsx"),
  route("/strategies/advisory", "app/strategies/advisory.tsx"),
  route("/trades", "app/trades.tsx"),
  route("/funding", "app/funding.tsx"),
  route("/settings", "app/settings.tsx"),
  route("/settings/sandbox-exchanges", "app/settings/sandbox-exchanges.tsx"),
  route("/settings/live-execution", "app/settings/live-execution.tsx"),

  // Quant-focused routes. Legacy AI Agent/LLM setup pages are intentionally
  // not exposed in the deployed SaaS workspace.
  route("/research/polymarket", "app/research/polymarket.tsx"),
] satisfies RouteConfig;
