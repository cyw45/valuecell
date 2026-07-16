import { CreditCard, FileClock } from "lucide-react";
import { useTenantBilling } from "@/api/saas-control";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function WorkspaceBillingPage() {
  const billing = useTenantBilling();
  const subscription = billing.data?.subscriptions[0];
  const agreement = billing.data?.agreement;
  return (
    <main className="scroll-container flex flex-1 flex-col gap-5 p-5 lg:p-8">
      <header>
        <p className="text-sm font-medium text-sky-600 dark:text-sky-300">
          账户与服务 / 订阅
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">
          订阅与结算
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          个人账户显示人工订阅；企业租户显示合同和高水位线分成结算。
        </p>
      </header>
      <section className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CreditCard className="size-4" /> 订阅服务
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Badge
              variant={
                billing.data?.access.status === "active"
                  ? "default"
                  : "secondary"
              }
            >
              {billing.data?.access.status === "active"
                ? "已开通"
                : "待开通 / 已到期"}
            </Badge>
            <p className="text-sm text-muted-foreground">
              到期时间：
              {subscription?.ends_at
                ? new Date(subscription.ends_at).toLocaleString("zh-CN")
                : "—"}
            </p>
            <p className="text-sm text-muted-foreground">
              开通方式：线下付款后由平台管理员人工开通。
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileClock className="size-4" /> 企业合同与结算
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {agreement ? (
              <>
                <p className="text-sm">
                  合同编号：{agreement.agreement_number}
                </p>
                <p className="text-sm text-muted-foreground">
                  分成比例：
                  {(Number(agreement.revenue_share_rate) * 100).toFixed(2)}%
                </p>
                <p className="text-sm text-muted-foreground">
                  高水位线：{agreement.high_water_mark_quote} USDT
                </p>
                <div className="divide-y rounded-md border">
                  {billing.data?.settlements.map((item) => (
                    <div
                      className="flex justify-between gap-3 px-3 py-2 text-sm"
                      key={item.id}
                    >
                      <span>
                        {new Date(item.period_ended_at).toLocaleDateString(
                          "zh-CN",
                        )}
                      </span>
                      <span>应收 {item.amount_due_quote} USDT</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                当前没有企业利润分成合同。
              </p>
            )}
          </CardContent>
        </Card>
      </section>
    </main>
  );
}
