const LOCAL_WORLDMONITOR_URL = "http://127.0.0.1:3001";

export function buildWorldMonitorDashboardUrl(configuredUrl?: string) {
  const url = new URL(configuredUrl || LOCAL_WORLDMONITOR_URL);
  const path = url.pathname.replace(/\/+$/, "");

  if (!path || path === "/") {
    url.pathname = "/dashboard";
  } else if (path.endsWith("/dashboard")) {
    url.pathname = path;
  } else {
    url.pathname = `${path}/dashboard`;
  }

  url.searchParams.set("lang", "zh");
  return url.toString();
}
