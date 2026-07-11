import {
  BarChart3,
  CandlestickChart,
  CircleDollarSign,
  Landmark,
  LayoutDashboard,
  LogOut,
  Settings,
  SlidersHorizontal,
} from "lucide-react";
import { memo, type FC } from "react";
import { useTranslation } from "react-i18next";
import { NavLink, useLocation, useNavigate } from "react-router";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useSaaSSession } from "@/store/system-store";

const PRIMARY_NAVIGATION = [
  { labelKey: "saas.navigation.dashboard", to: "/dashboard", icon: LayoutDashboard },
  { labelKey: "saas.navigation.charts", to: "/charts", icon: CandlestickChart },
  { labelKey: "saas.navigation.strategies", to: "/strategies", icon: SlidersHorizontal },
  { labelKey: "saas.navigation.trades", to: "/trades", icon: BarChart3 },
  { labelKey: "saas.navigation.funding", to: "/funding", icon: Landmark },
] as const;

const AppSidebar: FC = () => {
  const { t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const { isLoggedIn, email, clearSystemInfo } = useSaaSSession();

  function handleSignOut() {
    clearSystemInfo();
    navigate("/login", { replace: true });
  }

  return (
    <aside className="flex w-16 shrink-0 flex-col border-r bg-card md:w-60" aria-label={t("saas.navigation.primary")}>
      <div className="flex h-16 items-center justify-center border-b px-3 md:justify-start">
        <NavLink
          to="/dashboard"
          className="flex items-center gap-2 rounded-md font-semibold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={t("saas.navigation.valueCellDashboard")}
        >
          <CircleDollarSign className="size-6 text-primary" aria-hidden="true" />
          <span className="hidden text-base md:inline">{t("saas.brand")}</span>
        </NavLink>
      </div>

      <nav className="flex flex-1 flex-col gap-1 p-2" aria-label={t("saas.navigation.workspace")}>
        {PRIMARY_NAVIGATION.map(({ labelKey, to, icon: Icon }) => {
          const label = t(labelKey);
          const isActive = location.pathname === to || location.pathname.startsWith(`${to}/`);
          const navItem = (
            <NavLink
              key={to}
              to={to}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                "flex h-10 items-center justify-center gap-3 rounded-md px-3 text-sm font-medium text-muted-foreground transition-colors outline-none hover:bg-accent hover:text-accent-foreground focus-visible:ring-2 focus-visible:ring-ring md:justify-start",
                isActive && "bg-accent text-accent-foreground",
              )}
            >
              <Icon className="size-4 shrink-0" aria-hidden="true" />
              <span className="hidden md:inline">{label}</span>
            </NavLink>
          );

          return (
            <Tooltip key={to}>
              <TooltipTrigger asChild>{navItem}</TooltipTrigger>
              <TooltipContent side="right" className="md:hidden">{label}</TooltipContent>
            </Tooltip>
          );
        })}
      </nav>

      <div className="border-t p-2">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            cn(
              "flex h-10 items-center justify-center gap-3 rounded-md px-3 text-sm font-medium text-muted-foreground transition-colors outline-none hover:bg-accent hover:text-accent-foreground focus-visible:ring-2 focus-visible:ring-ring md:justify-start",
              isActive && "bg-accent text-accent-foreground",
            )
          }
        >
          <Settings className="size-4 shrink-0" aria-hidden="true" />
          <span className="hidden md:inline">{t("saas.navigation.settings")}</span>
        </NavLink>
        <Badge variant="secondary" className="mt-3 hidden w-full justify-center text-[10px] font-medium md:flex">
          {t("saas.navigation.paperWorkspace")}
        </Badge>
        {isLoggedIn && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="mt-2 flex h-9 w-full items-center justify-center gap-3 px-3 text-sm font-medium text-muted-foreground md:justify-start"
                onClick={handleSignOut}
                aria-label={t("saas.navigation.signOut")}
              >
                <LogOut className="size-4 shrink-0" aria-hidden="true" />
                <span className="hidden truncate md:inline" title={email}>{email || t("saas.navigation.signOut")}</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right" className="md:hidden">{t("saas.navigation.signOut")}</TooltipContent>
          </Tooltip>
        )}
      </div>
    </aside>
  );
};

export default memo(AppSidebar);
