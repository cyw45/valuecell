import { index, type RouteConfig, route } from "@react-router/dev/routes";

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
  route("/workspace", "app/workspace.tsx"),
  route("/workspace/members", "app/workspace/members.tsx"),
  route("/workspace/billing", "app/workspace/billing.tsx"),
  route("/workspace/audit", "app/workspace/audit.tsx"),
  route("/platform", "app/platform-redirect.tsx"),
  route("/admin/tenants", "app/admin/tenants.tsx"),
  route("/admin/plans", "app/admin/plans.tsx"),
  route("/admin/contracts", "app/admin/contracts.tsx"),
  route("/admin/audit", "app/admin/audit.tsx"),

  // Quant-focused routes. Legacy AI Agent/LLM setup pages are intentionally
  // not exposed in the deployed SaaS workspace.
  route("/research/polymarket", "app/research/polymarket.tsx"),
  route("/research/world-intelligence", "app/research/world-monitor.tsx"),
] satisfies RouteConfig;
