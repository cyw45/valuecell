import {
  BarChart3,
  Building2,
  CandlestickChart,
  ChevronLeft,
  ChevronRight,
  CircleDollarSign,
  ClipboardList,
  CreditCard,
  FileSignature,
  Landmark,
  LayoutDashboard,
  LogOut,
  RadioTower,
  Settings,
  SlidersHorizontal,
  Tags,
  UsersRound,
} from "lucide-react";
import { memo, type FC, useState } from "react";
import { useTranslation } from "react-i18next";
import { NavLink, useLocation, useNavigate } from "react-router";
import { useSaaSAccess } from "@/api/saas-control";
import { useSwitchWorkspace, useWorkspaces } from "@/api/saas-auth";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { useSaaSSession } from "@/store/system-store";

type NavigationItem = {
  label: string;
  to: string;
  icon: typeof LayoutDashboard;
};

const WORKBENCH_NAVIGATION: NavigationItem[] = [
  { label: "仪表盘", to: "/dashboard", icon: LayoutDashboard },
  { label: "策略配置", to: "/strategies", icon: SlidersHorizontal },
  { label: "订单与成交", to: "/trades", icon: BarChart3 },
  { label: "行情分析", to: "/charts", icon: CandlestickChart },
  { label: "全球情报", to: "/research/world-intelligence", icon: RadioTower },
  { label: "资金概览", to: "/funding", icon: Landmark },
];

const PLATFORM_NAVIGATION: NavigationItem[] = [
  { label: "租户账户", to: "/admin/tenants", icon: Building2 },
  { label: "订阅套餐", to: "/admin/plans", icon: Tags },
  { label: "企业合同", to: "/admin/contracts", icon: FileSignature },
  { label: "平台审计", to: "/admin/audit", icon: ClipboardList },
];

const AppSidebar: FC = () => {
  const { t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const { isLoggedIn, email, clearSystemInfo, setSaaSSession } =
    useSaaSSession();
  const access = useSaaSAccess();
  const workspaces = useWorkspaces(isLoggedIn);
  const switchWorkspace = useSwitchWorkspace();
  const [collapsed, setCollapsed] = useState(false);
  const tenantNavigation: NavigationItem[] =
    access.data?.tenant_type === "enterprise"
      ? [
          { label: "企业概览", to: "/workspace", icon: UsersRound },
          { label: "成员与角色", to: "/workspace/members", icon: UsersRound },
          { label: "资金账户", to: "/settings/live-execution", icon: Landmark },
          { label: "合同与结算", to: "/workspace/billing", icon: CreditCard },
          { label: "工作区审计", to: "/workspace/audit", icon: ClipboardList },
        ]
      : [
          { label: "账户概览", to: "/workspace", icon: UsersRound },
          { label: "订阅与结算", to: "/workspace/billing", icon: CreditCard },
          { label: "审计日志", to: "/workspace/audit", icon: ClipboardList },
        ];

  function handleSignOut() {
    clearSystemInfo();
    navigate("/login", { replace: true });
  }

  async function handleWorkspaceChange(tenantId: string) {
    try {
      const response = await switchWorkspace.mutateAsync(tenantId);
      const session = response.data;
      setSaaSSession({
        access_token: session.access_token,
        user_id: session.user_id,
        tenant_id: session.tenant_id,
        email: session.email,
      });
      window.location.assign("/workspace");
    } catch (error) {
      console.error("Failed to switch workspace", error);
    }
  }

  const selectedWorkspaceId = workspaces.data?.find(
    (workspace) => workspace.selected,
  )?.tenant_id;

  return (
    <aside
      className={cn(
        "relative hidden shrink-0 flex-col overflow-hidden border-r bg-card transition-[width] duration-200 ease-out lg:flex",
        collapsed ? "w-16" : "w-64",
      )}
      aria-label={t("saas.navigation.primary")}
    >
      <div className="flex h-16 items-center border-b px-3">
        <NavLink
          to="/dashboard"
          className="flex min-w-0 flex-1 items-center gap-2 rounded-md font-semibold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={t("saas.navigation.valueCellDashboard")}
        >
          <CircleDollarSign className="size-6 shrink-0 text-primary" />
          <span
            className={cn(
              "truncate text-base transition-opacity duration-150",
              collapsed && "pointer-events-none opacity-0",
            )}
          >
            {t("saas.brand")}
          </span>
        </NavLink>
        <Button
          aria-label={collapsed ? "展开侧边栏" : "收起侧边栏"}
          className="size-8 shrink-0"
          onClick={() => setCollapsed((value) => !value)}
          size="icon"
          variant="ghost"
        >
          {collapsed ? <ChevronRight className="size-4" /> : <ChevronLeft className="size-4" />}
        </Button>
      </div>

      {isLoggedIn && !collapsed && (workspaces.data?.length ?? 0) > 1 ? (
        <div className="border-b px-3 py-2.5">
          <Select
            disabled={switchWorkspace.isPending}
            onValueChange={handleWorkspaceChange}
            value={selectedWorkspaceId}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {workspaces.data?.map((workspace) => (
                <SelectItem key={workspace.tenant_id} value={workspace.tenant_id}>
                  {workspace.tenant_type === "enterprise" ? "企业 · " : "个人 · "}
                  {workspace.organization_name ?? workspace.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      ) : null}

      <nav
        className="flex flex-1 flex-col gap-6 overflow-y-auto px-2 py-4"
        aria-label={t("saas.navigation.workspace")}
      >
        <NavigationGroup
          collapsed={collapsed}
          items={WORKBENCH_NAVIGATION}
          label="交易工作台"
          pathname={location.pathname}
        />
        {isLoggedIn ? (
          <NavigationGroup
            collapsed={collapsed}
            items={tenantNavigation}
            label={
              access.data?.tenant_type === "enterprise"
                ? "企业管理"
                : "账户与服务"
            }
            pathname={location.pathname}
          />
        ) : null}
        {access.isSuccess && access.data?.is_platform_admin ? (
          <NavigationGroup
            collapsed={collapsed}
            items={PLATFORM_NAVIGATION}
            label="平台控制台"
            pathname={location.pathname}
          />
        ) : null}
      </nav>

      <div className="border-t p-2">
        <NavigationLink
          collapsed={collapsed}
          icon={Settings}
          label={t("saas.navigation.settings")}
          pathname={location.pathname}
          to="/settings"
        />
        {!collapsed ? (
          <Badge
            variant="secondary"
            className="mt-3 w-full justify-center text-[11px] font-medium"
          >
            {access.data?.is_platform_admin
              ? "平台管理员会话"
              : t("saas.navigation.paperWorkspace")}
          </Badge>
        ) : null}
        {isLoggedIn ? (
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 flex h-9 w-full items-center justify-center gap-3 px-3 text-sm font-medium text-muted-foreground"
            onClick={handleSignOut}
            aria-label={t("saas.navigation.signOut")}
            title={collapsed ? t("saas.navigation.signOut") : undefined}
          >
            <LogOut className="size-4 shrink-0" />
            <span
              className={cn(
                "min-w-0 truncate transition-opacity duration-150",
                collapsed && "pointer-events-none w-0 opacity-0",
              )}
              title={email}
            >
              {email || t("saas.navigation.signOut")}
            </span>
          </Button>
        ) : null}
      </div>
    </aside>
  );
};


function NavigationGroup({
  collapsed,
  items,
  label,
  pathname,
}: {
  collapsed: boolean;
  items: NavigationItem[];
  label: string;
  pathname: string;
}) {
  return (
    <section
      className={cn(
        "space-y-1.5",
        collapsed && "border-t border-border/60 pt-3 first:border-t-0 first:pt-0",
      )}
    >
      <div
        className={cn(
          "flex items-center gap-2 px-3 text-[11px] font-semibold tracking-[0.04em] text-foreground/70 transition-opacity duration-150",
          collapsed && "pointer-events-none h-0 overflow-hidden opacity-0",
        )}
      >
        <span className="size-1.5 rounded-full bg-sky-500/80" />
        {label}
      </div>
      {items.map((item) => (
        <NavigationLink
          collapsed={collapsed}
          icon={item.icon}
          key={item.to}
          label={item.label}
          pathname={pathname}
          to={item.to}
        />
      ))}
    </section>
  );
}

function NavigationLink({
  collapsed,
  icon: Icon,
  label,
  pathname,
  to,
}: {
  collapsed: boolean;
  icon: NavigationItem["icon"];
  label: string;
  pathname: string;
  to: string;
}) {
  const active = pathname === to || (to !== "/workspace" && pathname.startsWith(`${to}/`));
  return (
    <NavLink
      to={to}
      aria-current={active ? "page" : undefined}
      aria-label={collapsed ? label : undefined}
      title={collapsed ? label : undefined}
      className={cn(
        "group relative flex h-10 items-center justify-center gap-3 rounded-lg px-3 text-sm font-medium text-muted-foreground transition-[background-color,color,transform] duration-150 outline-none hover:bg-slate-950/[0.05] hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring dark:hover:bg-white/[0.07]",
        active && "bg-sky-500/10 text-sky-700 shadow-sm shadow-sky-500/5 dark:text-sky-300",
        active && "before:absolute before:inset-y-2 before:left-0 before:w-0.5 before:rounded-full before:bg-sky-500",
        !collapsed && "justify-start",
      )}
    >
      <Icon
        className={cn(
          "size-4 shrink-0 transition-transform duration-150 group-hover:scale-105",
          active && "text-sky-600 dark:text-sky-300",
        )}
      />
      <span
        className={cn(
          "truncate transition-opacity duration-150",
          collapsed && "pointer-events-none w-0 opacity-0",
        )}
      >
        {label}
      </span>
    </NavLink>
  );
}

export default memo(AppSidebar);
