import { createContext, type PropsWithChildren, useCallback, useContext, useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, clearSession, loadSession, persistSession } from "./api";
import type { PostAuthTab } from "./navigation/types";
import type { SaaSAuthResponse, SaaSRegisterRequest, Session } from "./types";

type SessionContextValue = {
  session: Session | null;
  ready: boolean;
  postAuthTab: PostAuthTab;
  replaceSession: (next: SaaSAuthResponse, postAuthTab: PostAuthTab) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  register: (request: SaaSRegisterRequest) => Promise<void>;
  switchWorkspace: (tenantId: string) => Promise<void>;
  signOut: () => Promise<void>;
};

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: PropsWithChildren) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const [postAuthTab, setPostAuthTab] = useState<PostAuthTab>("工作台");
  const queryClient = useQueryClient();

  useEffect(() => {
    void loadSession()
      .then(setSession)
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
    await persistSession(saved);
    setPostAuthTab(nextPostAuthTab);
    setSession(saved);
  }, [queryClient]);

  const signIn = useCallback(async (email: string, password: string) => {
    await replaceSession(await api.login(email.trim(), password), "工作台");
  }, [replaceSession]);

  const register = useCallback(async (request: SaaSRegisterRequest) => {
    await replaceSession(await api.register(request), "我的");
  }, [replaceSession]);

  const switchWorkspace = useCallback(async (tenantId: string) => {
    await replaceSession(await api.switchWorkspace(tenantId), postAuthTab);
  }, [postAuthTab, replaceSession]);

  const signOut = useCallback(async () => {
    await clearSession();
    queryClient.removeQueries({ queryKey: ["mobile"] });
    setPostAuthTab("工作台");
    setSession(null);
  }, [queryClient]);

  return (
    <SessionContext.Provider value={{ session, ready, postAuthTab, replaceSession, signIn, register, switchWorkspace, signOut }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSession must be used inside SessionProvider");
  return value;
}
