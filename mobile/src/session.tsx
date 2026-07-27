import { createContext, type PropsWithChildren, useCallback, useContext, useEffect, useState } from "react";
import { api, clearSession, loadSession, persistSession } from "./api";
import type { Session } from "./types";

type SessionContextValue = {
  session: Session | null;
  ready: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  switchWorkspace: (tenantId: string) => Promise<void>;
  signOut: () => Promise<void>;
};

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: PropsWithChildren) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    void loadSession().then((stored) => {
      setSession(stored);
      setReady(true);
    });
  }, []);

  const replaceSession = useCallback(async (next: {
    access_token: string;
    user_id: string;
    tenant_id: string;
    email: string;
  }) => {
    const saved: Session = {
      accessToken: next.access_token,
      userId: next.user_id,
      tenantId: next.tenant_id,
      email: next.email,
    };
    await persistSession(saved);
    setSession(saved);
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    await replaceSession(await api.login(email.trim(), password));
  }, [replaceSession]);

  const switchWorkspace = useCallback(async (tenantId: string) => {
    await replaceSession(await api.switchWorkspace(tenantId));
  }, [replaceSession]);

  const signOut = useCallback(async () => {
    await clearSession();
    setSession(null);
  }, []);

  return (
    <SessionContext.Provider value={{ session, ready, signIn, switchWorkspace, signOut }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSession must be used inside SessionProvider");
  return value;
}
