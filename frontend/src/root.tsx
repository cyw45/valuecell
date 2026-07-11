import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import { Links, Meta, Outlet, Scripts, ScrollRestoration, useLocation, useNavigate } from "react-router";
import { useEffect, useState } from "react";
import "@/i18n";
import AppSidebar from "@/components/valuecell/app/app-sidebar";
import { useLanguage } from "@/store/settings-store";
import { useSaaSSession } from "@/store/system-store";
import { Toaster } from "./components/ui/sonner";
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

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // Global default 5 minutes fresh time
      gcTime: 30 * 60 * 1000, // Global default 30 minutes garbage collection time
      refetchOnWindowFocus: false, // Don't refetch on window focus by default
      retry: 1, // Default retry 1 times on failure
    },
    mutations: {
      retry: 1, // Default retry 1 time for mutations
    },
  },
});

import { AutoUpdateCheck } from "@/components/valuecell/app/auto-update-check";
import { BackendHealthCheck } from "@/components/valuecell/app/backend-health-check";
import { TrackerProvider } from "./provider/tracker-provider";
// Routes that are accessible without a SaaS session.
const PUBLIC_PATHS: readonly string[] = ["/login"];
// Legacy route prefixes that bypass the SaaS auth guard entirely.
const LEGACY_PREFIXES: readonly string[] = ["/home", "/market", "/agent", "/setting", "/research", "/test"];

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
    if (
      isLoggedIn ||
      PUBLIC_PATHS.includes(path) ||
      LEGACY_PREFIXES.some((prefix) => path.startsWith(prefix))
    ) {
      return;
    }
    navigate("/login", { replace: true, state: { from: path } });
  }, [hydrated, isLoggedIn, location.pathname, navigate]);

  return <>{children}</>;
}

export default function Root() {
  return (
    <QueryClientProvider client={queryClient}>
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
              <SidebarProvider>
                <div className="fixed flex size-full overflow-hidden">
                  <AppSidebar />
                  <main
                    className="relative flex flex-1 overflow-hidden"
                    id="main-content"
                  >
                    <Outlet />
                  </main>
                  <Toaster />
                </div>
              </SidebarProvider>
            </SaaSGuard>
          </TrackerProvider>
          <AutoUpdateCheck />
        </BackendHealthCheck>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
