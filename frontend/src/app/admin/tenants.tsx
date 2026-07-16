import { useMemo, useState } from "react";
import { Building2, Save, UsersRound } from "lucide-react";
import { toast } from "sonner";
import {
  useAdminTenants,
  useAdminPlans,
  useGrantSubscription,
  useSaaSAccess,
  useUpdateTenantProfile,
} from "@/api/saas-control";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function AdminTenantsPage() {
  const access = useSaaSAccess();
  const enabled = access.data?.is_platform_admin === true;
  const tenants = useAdminTenants(enabled);
  const plans = useAdminPlans(enabled);
  const updateProfile = useUpdateTenantProfile();
  const grantSubscription = useGrantSubscription();
  const [selectedId, setSelectedId] = useState("");
  const selected = useMemo(
    () =>
      tenants.data?.find((tenant) => tenant.id === selectedId) ??
      tenants.data?.[0],
    [selectedId, tenants.data],
  );
  const [tenantType, setTenantType] = useState<"personal" | "enterprise">(
    "personal",
  );
  const [organizationName, setOrganizationName] = useState("");
  const [planId, setPlanId] = useState("");
  const [subscriptionEndsAt, setSubscriptionEndsAt] = useState("");
  const [subscriptionNote, setSubscriptionNote] = useState("");

  if (!enabled)
    return (
      <main className="flex flex-1 items-center justify-center p-6">
        <Alert className="max-w-lg border-destructive/40">
          <AlertTitle>无平台管理权限</AlertTitle>
          <AlertDescription>
            只有部署配置指定的平台管理员可管理所有租户。
          </AlertDescription>
        </Alert>
      </main>
    );

  const loadProfile = () => {
    if (!selected) return;
    setTenantType(selected.tenant_type);
    setOrganizationName(selected.organization_name ?? "");
  };

  async function saveProfile() {
    if (!selected) return;
    try {
      await updateProfile.mutateAsync({
        tenant_id: selected.id,
        tenant_type: tenantType,
        organization_name: organizationName.trim() || undefined,
      });
      toast.success("租户类型已更新。");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "无法保存租户类型。",
      );
    }
  }

  async function grantAccess() {
    if (!selected || !planId || !subscriptionEndsAt) return;
    try {
      await grantSubscription.mutateAsync({
        tenant_id: selected.id,
        plan_id: planId,
        ends_at: new Date(subscriptionEndsAt).toISOString(),
        note: subscriptionNote.trim() || undefined,
      });
      toast.success("订阅已开通，用户重新登录后即可使用交易测试功能。");
      setSubscriptionNote("");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "无法开通订阅。");
    }
  }

  return (
    <main className="scroll-container flex flex-1 flex-col gap-5 p-5 lg:p-8">
      <header>
        <p className="text-sm font-medium text-sky-600 dark:text-sky-300">
          平台管理 / 租户
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">租户账户</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          个人账户与企业租户是不同的商业主体；企业租户可使用成员、资金账户、合同与结算能力。
        </p>
      </header>
      <section className="grid min-h-0 gap-4 xl:grid-cols-[minmax(360px,0.9fr)_minmax(0,1.1fr)]">
        <Card className="min-h-0">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UsersRound className="size-4" /> 租户列表
            </CardTitle>
            <CardDescription>
              选择一个租户查看并维护其商业身份。
            </CardDescription>
          </CardHeader>
          <CardContent className="max-h-[620px] overflow-auto p-0">
            {tenants.data?.map((tenant) => (
              <button
                className={`flex w-full items-center justify-between gap-3 border-b px-5 py-3 text-left transition-colors hover:bg-muted/60 ${selected?.id === tenant.id ? "bg-sky-500/10" : ""}`}
                key={tenant.id}
                onClick={() => {
                  setSelectedId(tenant.id);
                  setTenantType(tenant.tenant_type);
                  setOrganizationName(tenant.organization_name ?? "");
                }}
                type="button"
              >
                <span>
                  <span className="block font-medium text-sm">
                    {tenant.name}
                  </span>
                  <span className="mt-1 block text-xs text-muted-foreground">
                    {tenant.organization_name ?? "个人工作区"}
                  </span>
                </span>
                <span className="flex flex-col items-end gap-1">
                  <Badge variant="outline">
                    {tenant.tenant_type === "enterprise" ? "企业" : "个人"}
                  </Badge>
                  <Badge
                    variant={
                      tenant.access.status === "active"
                        ? "default"
                        : "secondary"
                    }
                  >
                    {tenant.access.status === "active" ? "已开通" : "待开通"}
                  </Badge>
                </span>
              </button>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Building2 className="size-4" /> 租户详情
            </CardTitle>
            <CardDescription>
              平台管理员可将历史个人工作区升级为企业租户；该操作会记录审计日志。
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-5">
            {selected ? (
              <>
                <div className="rounded-md border bg-muted/30 p-4">
                  <p className="font-medium">{selected.name}</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    ID：{selected.id}
                  </p>
                </div>
                <div className="grid gap-2">
                  <Label>账户类型</Label>
                  <Select
                    onValueChange={(value) =>
                      setTenantType(value as "personal" | "enterprise")
                    }
                    value={tenantType}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="personal">个人账户</SelectItem>
                      <SelectItem value="enterprise">企业租户</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                {tenantType === "enterprise" ? (
                  <div className="grid gap-2">
                    <Label htmlFor="organization-name">企业名称</Label>
                    <Input
                      id="organization-name"
                      onChange={(event) =>
                        setOrganizationName(event.target.value)
                      }
                      required
                      value={organizationName}
                    />
                  </div>
                ) : null}
                <div className="flex justify-end gap-2">
                  <Button onClick={loadProfile} type="button" variant="outline">
                    恢复
                  </Button>
                  <Button
                    disabled={
                      updateProfile.isPending ||
                      (tenantType === "enterprise" && !organizationName.trim())
                    }
                    onClick={saveProfile}
                    type="button"
                  >
                    <Save /> 保存租户身份
                  </Button>
                </div>
                <section className="grid gap-3 border-t pt-5">
                  <div>
                    <h2 className="font-medium">开通订阅</h2>
                    <p className="mt-1 text-sm text-muted-foreground">
                      开通后，工作区才能保存策略、连接模拟交易所并提交测试订单。
                    </p>
                  </div>
                  <div className="grid gap-2">
                    <Label>订阅套餐</Label>
                    <Select value={planId} onValueChange={setPlanId}>
                      <SelectTrigger>
                        <SelectValue placeholder="选择已启用的套餐" />
                      </SelectTrigger>
                      <SelectContent>
                        {plans.data
                          ?.filter((plan) => plan.active === "active")
                          .map((plan) => (
                            <SelectItem key={plan.id} value={plan.id}>
                              {plan.name}（{plan.duration_days} 天）
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="subscription-ends-at">到期时间</Label>
                    <Input
                      id="subscription-ends-at"
                      onChange={(event) => setSubscriptionEndsAt(event.target.value)}
                      required
                      type="datetime-local"
                      value={subscriptionEndsAt}
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="subscription-note">备注</Label>
                    <Input
                      id="subscription-note"
                      maxLength={1000}
                      onChange={(event) => setSubscriptionNote(event.target.value)}
                      value={subscriptionNote}
                    />
                  </div>
                  <Button
                    disabled={
                      grantSubscription.isPending || !planId || !subscriptionEndsAt
                    }
                    onClick={grantAccess}
                    type="button"
                  >
                    开通订阅
                  </Button>
                </section>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">请选择左侧租户。</p>
            )}
          </CardContent>
        </Card>
      </section>
    </main>
  );
}
