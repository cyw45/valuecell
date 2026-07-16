import { ArrowRight, Building2, CreditCard, UsersRound } from "lucide-react";
import { Link } from "react-router";
import { useSaaSAccess } from "@/api/saas-control";
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

export default function WorkspacePage() {
  const access = useSaaSAccess();
  const enterprise = access.data?.tenant_type === "enterprise";
  return (
    <main className="scroll-container flex flex-1 flex-col gap-6 p-5 lg:p-8">
      <header>
        <p className="text-sm font-medium text-sky-600 dark:text-sky-300">
          {enterprise ? "企业工作区" : "个人账户"}
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">账户概览</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          商业身份、服务状态和管理入口分开呈现，权限由服务端角色控制。
        </p>
      </header>
      {access.data?.status !== "active" ? (
        <Alert className="border-amber-500/40 bg-amber-500/5">
          <CreditCard className="text-amber-600" />
          <AlertTitle>工作区尚未开通</AlertTitle>
          <AlertDescription>
            当前不能创建策略、绑定交易账户或提交订单。请在“订阅与结算”页面查看开通状态。
          </AlertDescription>
        </Alert>
      ) : null}
      <section className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>账户类型</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant="outline">
              {enterprise ? "企业租户" : "个人账户"}
            </Badge>
            <p className="mt-3 text-sm text-muted-foreground">
              {access.data?.organization_name ?? "独立策略工作区"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>服务状态</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge
              variant={
                access.data?.status === "active" ? "default" : "secondary"
              }
            >
              {access.data?.status === "active" ? "已开通" : "待开通 / 已到期"}
            </Badge>
            <p className="mt-3 text-sm text-muted-foreground">
              {access.data?.commercial_model === "revenue_share"
                ? "企业利润分成"
                : "人工订阅"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>当前角色</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant="outline">{access.data?.role ?? "—"}</Badge>
            <p className="mt-3 text-sm text-muted-foreground">
              权限变化立即在服务端生效。
            </p>
          </CardContent>
        </Card>
      </section>
      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CreditCard className="size-4" /> 订阅与结算
            </CardTitle>
            <CardDescription>
              查看订阅到期、企业合同和利润分成结算记录。
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="outline">
              <Link to="/workspace/billing">
                打开订阅与结算 <ArrowRight />
              </Link>
            </Button>
          </CardContent>
        </Card>
        {enterprise ? (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UsersRound className="size-4" /> 组织成员
              </CardTitle>
              <CardDescription>
                邀请已注册用户并配置管理员、策略师、交易员或观察员角色。
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button asChild variant="outline">
                <Link to="/workspace/members">
                  打开成员管理 <ArrowRight />
                </Link>
              </Button>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Building2 className="size-4" /> 升级企业租户
              </CardTitle>
              <CardDescription>
                需要多人协作、多个资金账户或利润分成合同，请联系平台管理员升级。
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button asChild variant="outline">
                <Link to="/workspace/billing">
                  查看服务方案 <ArrowRight />
                </Link>
              </Button>
            </CardContent>
          </Card>
        )}
      </section>
    </main>
  );
}
