import { createContext, type PropsWithChildren, useCallback, useContext, useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  api,
  clearRememberedPassword,
  clearSession,
  loadRememberedEmail,
  loadRememberedPassword,
  loadSession,
  persistRememberedEmail,
  persistRememberedPassword,
  persistSession,
} from "./api";
import type { PostAuthTab } from "./navigation/types";
import type { SaaSAuthResponse, SaaSRegisterRequest, Session } from "./types";

type SessionContextValue = {
  authNotice: string | null;
  dismissAuthNotice: () => void;
  postAuthTab: PostAuthTab;
  ready: boolean;
  rememberedEmail: string;
  rememberedPassword: string;
  register: (request: SaaSRegisterRequest) => Promise<void>;
  replaceSession: (next: SaaSAuthResponse, postAuthTab: PostAuthTab) => Promise<void>;
  session: Session | null;
  signIn: (email: string, password: string, rememberPassword: boolean) => Promise<void>;
  signOut: () => Promise<void>;
  switchWorkspace: (tenantId: string) => Promise<void>;
};

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: PropsWithChildren) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const [postAuthTab, setPostAuthTab] = useState<PostAuthTab>("工作台");
  const [rememberedEmail, setRememberedEmail] = useState("");
  const [rememberedPassword, setRememberedPassword] = useState("");
  const [authNotice, setAuthNotice] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const endSession = useCallback(async (notice: string | null) => {
    await clearSession();
    await queryClient.cancelQueries({ queryKey: ["mobile"] });
    queryClient.removeQueries({ queryKey: ["mobile"] });
    setPostAuthTab("工作台");
    setSession(null);
    setAuthNotice(notice);
  }, [queryClient]);

  useEffect(() => {
    api.setUnauthorizedHandler(() => endSession("登录已过期，请重新登录。"));
    return () => api.setUnauthorizedHandler(null);
  }, [endSession]);
  useEffect(() => {
    void Promise.all([loadSession(), loadRememberedEmail(), loadRememberedPassword()])
      .then(([restoredSession, storedEmail, storedPassword]) => {
        setRememberedEmail(storedEmail ?? restoredSession?.email ?? "");
        setRememberedPassword(storedPassword ?? "");
        setSession(restoredSession);
      })
      .catch(() => setSession(null))
      .finally(() => setReady(true));
  }, []);

  const replaceSession = useCallback(async (next: SaaSAuthResponse, nextPostAuthTab: PostAuthTab) => {
    queryClient.removeQueries({ queryKey: ["mobile"] });
    const saved: Session = {
      accessToken: next.access_token,
      userId: next.user_id,
      tenantId: next.tenant_id,
      email: next.email,
    };
    await Promise.all([persistSession(saved), persistRememberedEmail(saved.email)]);
    setRememberedEmail(saved.email);
    setAuthNotice(null);
    setPostAuthTab(nextPostAuthTab);
    setSession(saved);
  }, [queryClient]);

  const signIn = useCallback(async (email: string, password: string, rememberPassword: boolean) => {
    await replaceSession(await api.login(email.trim(), password), "工作台");
    if (rememberPassword) {
      await persistRememberedPassword(password);
      setRememberedPassword(password);
    } else {
      await clearRememberedPassword();
      setRememberedPassword("");
    }
  }, [replaceSession]);

  const register = useCallback(async (request: SaaSRegisterRequest) => {
    await replaceSession(await api.register(request), "我的");
  }, [replaceSession]);

  const switchWorkspace = useCallback(async (tenantId: string) => {
    await replaceSession(await api.switchWorkspace(tenantId), postAuthTab);
  }, [postAuthTab, replaceSession]);

  const signOut = useCallback(async () => {
    await endSession(null);
  }, [endSession]);

  return (
    <SessionContext.Provider value={{ authNotice, dismissAuthNotice: () => setAuthNotice(null), postAuthTab, ready, rememberedEmail, rememberedPassword, register, replaceSession, session, signIn, signOut, switchWorkspace }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSession must be used inside SessionProvider");
  return value;
}
