import { useState } from "react";
import { toast } from "sonner";
import { useChangePassword } from "@/api/saas-auth";
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

export default function WorkspaceSecurityPage() {
  const changePassword = useChangePassword();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (newPassword.length < 12) {
      toast.error("新密码至少需要 12 个字符。");
      return;
    }
    if (newPassword !== confirmation) {
      toast.error("两次输入的新密码不一致。");
      return;
    }
    changePassword.mutate(
      { current_password: currentPassword, new_password: newPassword },
      {
        onSuccess: () => {
          setCurrentPassword("");
          setNewPassword("");
          setConfirmation("");
          toast.success("密码已更新。浏览器保存的旧密码请同步更新。");
        },
        onError: (reason: unknown) =>
          toast.error(reason instanceof Error ? reason.message : "密码修改失败。"),
      },
    );
  };

  return (
    <main className="scroll-container flex flex-1 justify-center p-5 lg:p-8">
      <Card className="h-fit w-full max-w-lg">
        <CardHeader>
          <CardTitle>账户安全</CardTitle>
          <CardDescription>
            验证当前密码后可更新密码。新密码至少 12 个字符。
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={submit}>
            <div className="space-y-1.5">
              <Label htmlFor="current-password">当前密码</Label>
              <Input
                autoComplete="current-password"
                id="current-password"
                onChange={(event) => setCurrentPassword(event.target.value)}
                required
                type="password"
                value={currentPassword}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="new-password">新密码</Label>
              <Input
                autoComplete="new-password"
                id="new-password"
                minLength={12}
                onChange={(event) => setNewPassword(event.target.value)}
                required
                type="password"
                value={newPassword}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="confirm-password">确认新密码</Label>
              <Input
                autoComplete="new-password"
                id="confirm-password"
                minLength={12}
                onChange={(event) => setConfirmation(event.target.value)}
                required
                type="password"
                value={confirmation}
              />
            </div>
            <Button className="w-full" disabled={changePassword.isPending} type="submit">
              {changePassword.isPending ? "正在更新…" : "更新密码"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </main>
  );
}
