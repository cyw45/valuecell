import { useState } from "react";
import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { useLogin, useRegister } from "@/api/saas-auth";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useSaaSSession } from "@/store/system-store";

export default function LoginPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { setSaaSSession } = useSaaSSession();
  const loginMutation = useLogin();
  const registerMutation = useRegister();

  const [loginForm, setLoginForm] = useState({ email: "", password: "" });
  const [registerForm, setRegisterForm] = useState({
    email: "",
    password: "",
    workspace_name: "",
  });

  function handleLoginSubmit(e: React.FormEvent) {
    e.preventDefault();
    loginMutation.mutate(loginForm, {
      onSuccess: (resp) => {
        const session = resp.data;
        if (!session?.access_token || !session.user_id || !session.tenant_id) {
          toast.error(t("saas.login.errors.incompleteAuthentication"));
          return;
        }
        setSaaSSession({
          access_token: session.access_token,
          user_id: session.user_id,
          tenant_id: session.tenant_id,
          email: session.email ?? loginForm.email.trim().toLowerCase(),
        });
        navigate("/dashboard", { replace: true });
      },
      onError: (err: unknown) => {
        const msg = err instanceof Error ? err.message : t("saas.login.errors.invalidCredentials");
        toast.error(msg);
      },
    });
  }

  function handleRegisterSubmit(e: React.FormEvent) {
    e.preventDefault();
    registerMutation.mutate(registerForm, {
      onSuccess: (resp) => {
        const session = resp.data;
        if (!session?.access_token || !session.user_id || !session.tenant_id) {
          toast.error(t("saas.login.errors.incompleteRegistration"));
          return;
        }
        setSaaSSession({
          access_token: session.access_token,
          user_id: session.user_id,
          tenant_id: session.tenant_id,
          email: session.email ?? registerForm.email.trim().toLowerCase(),
        });
        navigate("/dashboard", { replace: true });
      },
      onError: (err: unknown) => {
        const msg = err instanceof Error ? err.message : t("saas.login.errors.registrationFailed");
        toast.error(msg);
      },
    });
  }

  return (
    <div
      className="flex size-full items-center justify-center bg-muted/40"
      role="main"
      aria-label={t("saas.login.aria.authentication")}
    >
      <div className="w-full max-w-sm px-4">
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">{t("saas.brand")}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {t("saas.login.subtitle")}
          </p>
        </div>

        <Tabs defaultValue="login">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="login">{t("saas.login.signIn")}</TabsTrigger>
            <TabsTrigger value="register">{t("saas.login.createAccount")}</TabsTrigger>
          </TabsList>

          <TabsContent value="login">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">{t("saas.login.signIn")}</CardTitle>
                <CardDescription>
                  {t("saas.login.signInDescription")}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleLoginSubmit} className="space-y-4">
                  <div className="space-y-1.5">
                    <Label htmlFor="login-email">{t("saas.login.email")}</Label>
                    <Input
                      id="login-email"
                      type="email"
                      autoComplete="email"
                      required
                      value={loginForm.email}
                      onChange={(e) =>
                        setLoginForm((f) => ({ ...f, email: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="login-password">{t("saas.login.password")}</Label>
                    <Input
                      id="login-password"
                      type="password"
                      autoComplete="current-password"
                      required
                      value={loginForm.password}
                      onChange={(e) =>
                        setLoginForm((f) => ({
                          ...f,
                          password: e.target.value,
                        }))
                      }
                    />
                  </div>
                  <Button
                    type="submit"
                    className="w-full"
                    disabled={loginMutation.isPending}
                    aria-busy={loginMutation.isPending}
                  >
                    {loginMutation.isPending ? t("saas.login.signingIn") : t("saas.login.signIn")}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="register">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">{t("saas.login.createAccount")}</CardTitle>
                <CardDescription>
                  {t("saas.login.createAccountDescription")}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleRegisterSubmit} className="space-y-4">
                  <div className="space-y-1.5">
                    <Label htmlFor="reg-workspace">{t("saas.login.workspaceName")}</Label>
                    <Input
                      id="reg-workspace"
                      type="text"
                      autoComplete="organization"
                      required
                      minLength={2}
                      maxLength={100}
                      value={registerForm.workspace_name}
                      onChange={(e) =>
                        setRegisterForm((f) => ({
                          ...f,
                          workspace_name: e.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="reg-email">{t("saas.login.email")}</Label>
                    <Input
                      id="reg-email"
                      type="email"
                      autoComplete="email"
                      required
                      value={registerForm.email}
                      onChange={(e) =>
                        setRegisterForm((f) => ({
                          ...f,
                          email: e.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="reg-password">
                      {t("saas.login.password")} {" "}
                      <span className="text-xs text-muted-foreground">
                        {t("saas.login.minimumPasswordLength")}
                      </span>
                    </Label>
                    <Input
                      id="reg-password"
                      type="password"
                      autoComplete="new-password"
                      required
                      minLength={12}
                      value={registerForm.password}
                      onChange={(e) =>
                        setRegisterForm((f) => ({
                          ...f,
                          password: e.target.value,
                        }))
                      }
                    />
                  </div>
                  <Button
                    type="submit"
                    className="w-full"
                    disabled={registerMutation.isPending}
                    aria-busy={registerMutation.isPending}
                  >
                    {registerMutation.isPending
                      ? t("saas.login.creating")
                      : t("saas.login.createWorkspace")}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
