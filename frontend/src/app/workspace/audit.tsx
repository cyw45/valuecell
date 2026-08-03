import { ClipboardList } from "lucide-react";
import { useTenantAudit } from "@/api/saas-control";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function WorkspaceAuditPage() {
  const audit = useTenantAudit();
  return (
    <main className="scroll-container flex flex-1 flex-col gap-5 p-5 lg:p-8">
      <header>
        <p className="font-medium text-sky-600 text-sm dark:text-sky-300">
          账户与服务 / 审计
        </p>
        <h1 className="mt-1 font-semibold text-2xl tracking-tight">
          工作区审计日志
        </h1>
        <p className="mt-2 text-muted-foreground text-sm">
          成员、订阅、合同、交易连接和订单控制事件均可追溯。
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
                  <p className="mt-1 text-muted-foreground text-xs">
                    {event.target_type} · {event.target_id} ·{" "}
                    {new Date(event.created_at).toLocaleString("zh-CN")}
                  </p>
                </div>
                <Badge className="w-fit self-start" variant="outline">
                  {event.outcome}
                </Badge>
              </article>
            )) ?? (
              <p className="p-5 text-muted-foreground text-sm">
                暂无审计事件。
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </main>
  );
}
