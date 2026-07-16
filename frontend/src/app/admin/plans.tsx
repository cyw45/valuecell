import { type FormEvent, useState } from "react";
import { BadgeDollarSign, Plus } from "lucide-react";
import { toast } from "sonner";
import {
  useAdminPlans,
  useCreatePlan,
  useSaaSAccess,
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

export default function AdminPlansPage() {
  const access = useSaaSAccess();
  const enabled = access.data?.is_platform_admin === true;
  const plans = useAdminPlans(enabled);
  const createPlan = useCreatePlan();
  const [form, setForm] = useState({
    code: "individual-monthly",
    name: "个人月度版",
    duration: "30",
    price: "50000",
  });

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      await createPlan.mutateAsync({
        code: form.code,
        name: form.name,
        duration_days: Number(form.duration),
        price_cents: Number(form.price),
      });
      toast.success("套餐已创建。");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "无法创建套餐。");
    }
  }

  if (!enabled)
    return (
      <main className="flex flex-1 items-center justify-center p-6">
        <Alert className="max-w-lg border-destructive/40">
          <AlertTitle>无平台管理权限</AlertTitle>
          <AlertDescription>套餐仅可由平台管理员维护。</AlertDescription>
        </Alert>
      </main>
    );
  return (
    <main className="scroll-container flex flex-1 flex-col gap-5 p-5 lg:p-8">
      <header>
        <p className="text-sm font-medium text-sky-600 dark:text-sky-300">
          平台管理 / 套餐
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">订阅套餐</h1>
      </header>
      <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <Card>
          <CardHeader>
            <CardTitle>套餐列表</CardTitle>
            <CardDescription>
              面向个人账户的人工收款套餐。开通动作在租户详情中执行。
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="divide-y">
              {plans.data?.map((plan) => (
                <div
                  className="flex items-center justify-between gap-4 px-5 py-4"
                  key={plan.id}
                >
                  <div>
                    <p className="font-medium">{plan.name}</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {plan.code} · {plan.duration_days} 天 ·{" "}
                      {(plan.price_cents / 100).toLocaleString()}{" "}
                      {plan.currency}
                    </p>
                  </div>
                  <Badge
                    variant={plan.active === "active" ? "default" : "secondary"}
                  >
                    {plan.active}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BadgeDollarSign className="size-4" /> 新建套餐
            </CardTitle>
            <CardDescription>价格按分保存：500 元填写 50000。</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="grid gap-4" onSubmit={submit}>
              <Field
                label="套餐编码"
                value={form.code}
                onChange={(code) =>
                  setForm((current) => ({ ...current, code }))
                }
              />
              <Field
                label="显示名称"
                value={form.name}
                onChange={(name) =>
                  setForm((current) => ({ ...current, name }))
                }
              />
              <Field
                label="服务天数"
                type="number"
                value={form.duration}
                onChange={(duration) =>
                  setForm((current) => ({ ...current, duration }))
                }
              />
              <Field
                label="价格（分）"
                type="number"
                value={form.price}
                onChange={(price) =>
                  setForm((current) => ({ ...current, price }))
                }
              />
              <Button disabled={createPlan.isPending} type="submit">
                <Plus /> 创建套餐
              </Button>
            </form>
          </CardContent>
        </Card>
      </section>
    </main>
  );
}
function Field({
  label,
  value,
  onChange,
  type = "text",
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: "number" | "text";
}) {
  return (
    <div className="grid gap-1.5">
      <Label>{label}</Label>
      <Input
        min={type === "number" ? 0 : undefined}
        onChange={(event) => onChange(event.target.value)}
        required
        type={type}
        value={value}
      />
    </div>
  );
}
