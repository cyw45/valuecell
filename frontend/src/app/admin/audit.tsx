import { ClipboardList } from "lucide-react";
import { useAdminAudit, useSaaSAccess } from "@/api/saas-control";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function AdminAuditPage() {
  const access = useSaaSAccess();
  const enabled = access.data?.is_platform_admin === true;
  const audit = useAdminAudit(enabled);
  if (!enabled)
    return (
      <main className="flex flex-1 items-center justify-center p-6">
        <Alert className="max-w-lg border-destructive/40">
          <AlertTitle>无平台管理权限</AlertTitle>
          <AlertDescription>全局审计日志只对平台管理员开放。</AlertDescription>
        </Alert>
      </main>
    );
  return (
    <main className="scroll-container flex flex-1 flex-col gap-5 p-5 lg:p-8">
      <header>
        <p className="text-sm font-medium text-sky-600 dark:text-sky-300">
          平台管理 / 审计
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">
          全局审计日志
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          套餐、订阅、合同、成员和实盘控制动作均以不可变事件保存。
        </p>
      </header>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ClipboardList className="size-4" /> 审计事件
          </CardTitle>
          <CardDescription>按最新发生时间排序。</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="divide-y">
            {audit.data?.map((event) => (
              <article
                className="grid gap-2 px-5 py-4 sm:grid-cols-[minmax(0,1fr)_auto]"
                key={event.id}
              >
                <div>
                  <p className="font-medium text-sm">{event.action}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {event.target_type} · {event.target_id} ·{" "}
                    {new Date(event.created_at).toLocaleString("zh-CN")}
                  </p>
                </div>
                <Badge className="w-fit self-start" variant="outline">
                  {event.outcome}
                </Badge>
              </article>
            )) ?? (
              <p className="p-5 text-sm text-muted-foreground">
                暂无审计事件。
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </main>
  );
}
