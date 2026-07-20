import { QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import { useEffect, useState } from "react";
import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  useLocation,
  useNavigate,
} from "react-router";
import "@/i18n";
import AppSidebar from "@/components/valuecell/app/app-sidebar";
import { useLanguage } from "@/store/settings-store";
import { useSaaSSession } from "@/store/system-store";
import { Toaster } from "./components/ui/sonner";
import { queryClient } from "./query-client";
import { isSaaSPublicRoute, SaaSGuardBoundary } from "./saas-guard";
import { SessionCacheBoundary } from "./session-cache-boundary";
import "./global.css";
import { SidebarProvider } from "./components/ui/sidebar";

export function Layout({ children }: { children: React.ReactNode }) {
  const language = useLanguage();
  const htmlLang =
    {
      en: "en",
      zh_CN: "zh-CN",
      zh_TW: "zh-TW",
      ja: "ja",
    }[language] ?? "en";

  return (
    <html lang={htmlLang} suppressHydrationWarning>
      <head>
        <meta charSet="UTF-8" />
        <link rel="icon" type="image/svg+xml" href="/logo.svg" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Value Cell</title>
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

import { AutoUpdateCheck } from "@/components/valuecell/app/auto-update-check";
import { BackendHealthCheck } from "@/components/valuecell/app/backend-health-check";
import { TrackerProvider } from "./provider/tracker-provider";

function SaaSGuard({ children }: { children: React.ReactNode }) {
  const { isLoggedIn } = useSaaSSession();
  const location = useLocation();
  const navigate = useNavigate();
  // Wait one tick for Zustand persist to rehydrate from localStorage before
  // making any redirect decision — prevents the login-flash on hard refresh.
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    const path = location.pathname;
    if (isLoggedIn || isSaaSPublicRoute(path)) {
      return;
    }
    navigate("/login", { replace: true, state: { from: path } });
  }, [hydrated, isLoggedIn, location.pathname, navigate]);

  const isPublicRoute = isSaaSPublicRoute(location.pathname);
  return (
    <SaaSGuardBoundary
      hydrated={hydrated}
      isLoggedIn={isLoggedIn}
      isPublicRoute={isPublicRoute}
    >
      {children}
    </SaaSGuardBoundary>
  );
}

function ApplicationShell() {
  const location = useLocation();
  const isPublicRoute = location.pathname === "/login";
  return (
    <SidebarProvider>
      <div className="fixed flex size-full overflow-hidden">
        {!isPublicRoute ? <AppSidebar /> : null}
        <main
          className="relative flex flex-1 overflow-hidden"
          id="main-content"
        >
          <Outlet />
        </main>
        <Toaster />
      </div>
    </SidebarProvider>
  );
}

function SessionCacheProvider({ children }: { children: React.ReactNode }) {
  const { userId, tenantId } = useSaaSSession();
  return (
    <SessionCacheBoundary boundary={`${userId}:${tenantId}`}>
      {children}
    </SessionCacheBoundary>
  );
}

export default function Root() {
  return (
    <QueryClientProvider client={queryClient}>
      <SessionCacheProvider>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          enableColorScheme
          storageKey="valuecell-theme"
        >
          <BackendHealthCheck>
            <TrackerProvider>
              <SaaSGuard>
                <ApplicationShell />
              </SaaSGuard>
            </TrackerProvider>
            <AutoUpdateCheck />
          </BackendHealthCheck>
        </ThemeProvider>
      </SessionCacheProvider>
    </QueryClientProvider>
  );
}
