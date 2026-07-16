import { type FormEvent, useState } from "react";
import { toast } from "sonner";
import {
  useSaaSAccess,
  useSaveWorkspaceMember,
  useWorkspaceMembers,
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
import type { TenantRole } from "@/types/saas-control";

export default function WorkspaceMembersPage() {
  const access = useSaaSAccess();
  const enabled =
    access.data?.status === "active" &&
    access.data?.tenant_type === "enterprise";
  const members = useWorkspaceMembers(enabled);
  const saveMember = useSaveWorkspaceMember();
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<TenantRole>("viewer");
  const canManage =
    access.data?.role === "owner" || access.data?.role === "admin";
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      await saveMember.mutateAsync({ email: email.trim(), role });
      setEmail("");
      toast.success("成员权限已保存。");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "无法保存成员权限。",
      );
    }
  }
  if (access.data?.tenant_type !== "enterprise")
    return (
      <main className="flex flex-1 items-center justify-center p-6">
        <Alert className="max-w-lg">
          <AlertTitle>仅企业租户可使用成员管理</AlertTitle>
          <AlertDescription>
            个人账户如需多人协作，请联系平台管理员升级为企业租户。
          </AlertDescription>
        </Alert>
      </main>
    );
  return (
    <main className="scroll-container flex flex-1 flex-col gap-5 p-5 lg:p-8">
      <header>
        <p className="text-sm font-medium text-sky-600 dark:text-sky-300">
          企业管理 / 成员
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">
          成员与角色
        </h1>
      </header>
      <section className="grid gap-4 xl:grid-cols-[420px_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>添加或更新成员</CardTitle>
            <CardDescription>
              成员必须先注册平台账户；角色权限立即由后端生效。
            </CardDescription>
          </CardHeader>
          <CardContent>
            {canManage && enabled ? (
              <form className="grid gap-4" onSubmit={submit}>
                <div className="grid gap-1.5">
                  <Label>成员邮箱</Label>
                  <Input
                    onChange={(event) => setEmail(event.target.value)}
                    required
                    type="email"
                    value={email}
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label>角色</Label>
                  <Select
                    onValueChange={(value) => setRole(value as TenantRole)}
                    value={role}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="admin">管理员</SelectItem>
                      <SelectItem value="strategist">策略师</SelectItem>
                      <SelectItem value="trader">交易员</SelectItem>
                      <SelectItem value="viewer">观察员</SelectItem>
                      <SelectItem value="billing_manager">
                        账务管理员
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button disabled={saveMember.isPending} type="submit">
                  保存成员
                </Button>
              </form>
            ) : (
              <p className="text-sm text-muted-foreground">
                只有 owner 或 admin 可以修改成员。
              </p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>成员列表</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="divide-y">
              {members.data?.map((member) => (
                <div
                  className="flex items-center justify-between gap-3 px-5 py-3"
                  key={member.user_id}
                >
                  <span className="text-sm">{member.email}</span>
                  <Badge variant="outline">{member.role}</Badge>
                </div>
              )) ?? (
                <p className="p-5 text-sm text-muted-foreground">
                  暂无可显示成员。
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </section>
    </main>
  );
}
