import { type FormEvent, useState } from "react";
import { FileSignature } from "lucide-react";
import { toast } from "sonner";
import {
  useAdminTenants,
  useCreateEnterpriseAgreement,
  useSaaSAccess,
} from "@/api/saas-control";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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

export default function AdminContractsPage() {
  const access = useSaaSAccess();
  const enabled = access.data?.is_platform_admin === true;
  const tenants = useAdminTenants(enabled);
  const createAgreement = useCreateEnterpriseAgreement();
  const [form, setForm] = useState({
    tenantId: "",
    number: "",
    ratePercent: "10",
    cycleDays: "30",
  });

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      await createAgreement.mutateAsync({
        tenant_id: form.tenantId,
        agreement_number: form.number,
        revenue_share_rate: String(Number(form.ratePercent) / 100),
        settlement_cycle_days: Number(form.cycleDays),
        starts_at: new Date().toISOString(),
      });
      toast.success("企业利润分成合同已激活。");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "无法创建企业合同。",
      );
    }
  }

  if (!enabled)
    return (
      <main className="flex flex-1 items-center justify-center p-6">
        <Alert className="max-w-lg border-destructive/40">
          <AlertTitle>无平台管理权限</AlertTitle>
          <AlertDescription>
            合同与利润结算仅供平台管理员维护。
          </AlertDescription>
        </Alert>
      </main>
    );
  return (
    <main className="scroll-container flex flex-1 flex-col gap-5 p-5 lg:p-8">
      <header>
        <p className="text-sm font-medium text-sky-600 dark:text-sky-300">
          平台管理 / 合同
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">
          企业利润分成
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          合同激活后，企业租户获得工作区权限；分成使用资金账户对账与高水位线计算。
        </p>
      </header>
      <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <Card>
          <CardHeader>
            <CardTitle>结算口径</CardTitle>
            <CardDescription>
              每个资金账户每个周期独立结算，净入出金先调整高水位线。
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm text-muted-foreground">
            <p>可分成利润 = max(0, 期末权益 − 调整后高水位线)。</p>
            <p>应收分成 = 可分成利润 × 合同分成比例。</p>
            <p>
              结算接口只接收该合同租户拥有且未撤销的交易所资金账户，结果与审计记录不可修改。
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileSignature className="size-4" /> 新建企业合同
            </CardTitle>
            <CardDescription>
              请先在“租户”页把工作区设置为企业租户。
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form className="grid gap-4" onSubmit={submit}>
              <div className="grid gap-1.5">
                <Label>企业租户</Label>
                <Select
                  onValueChange={(tenantId) =>
                    setForm((current) => ({ ...current, tenantId }))
                  }
                  value={form.tenantId}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="选择企业租户" />
                  </SelectTrigger>
                  <SelectContent>
                    {tenants.data
                      ?.filter((tenant) => tenant.tenant_type === "enterprise")
                      .map((tenant) => (
                        <SelectItem key={tenant.id} value={tenant.id}>
                          {tenant.organization_name ?? tenant.name}
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </div>
              <Field
                label="合同编号"
                value={form.number}
                onChange={(number) =>
                  setForm((current) => ({ ...current, number }))
                }
              />
              <Field
                label="分成比例（%）"
                type="number"
                value={form.ratePercent}
                onChange={(ratePercent) =>
                  setForm((current) => ({ ...current, ratePercent }))
                }
              />
              <Field
                label="结算周期（天）"
                type="number"
                value={form.cycleDays}
                onChange={(cycleDays) =>
                  setForm((current) => ({ ...current, cycleDays }))
                }
              />
              <Button
                disabled={!form.tenantId || createAgreement.isPending}
                type="submit"
              >
                激活合同
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
