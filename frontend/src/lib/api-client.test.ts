import assert from "node:assert/strict";
import test from "node:test";
import { ApiClient } from "./api-client.ts";

const session = {
  access_token: "saas-token",
  refresh_token: "",
  id: "user-1",
  tenant_id: "tenant-a",
};

test("builds independent request headers without leaking a stale bearer token", async () => {
  const calls: RequestInit[] = [];
  const client = new ApiClient({
    getSession: () => session,
    fetch: async (_url, init) => {
      calls.push(init ?? {});
      return new Response(JSON.stringify({ ok: true }), {
        headers: { "content-type": "application/json" },
      });
    },
  });

  await client.get("/protected", { requiresAuth: true });
  session.access_token = "";
  await client.get("/public");

  assert.deepEqual(calls[0].headers, {
    "Content-Type": "application/json",
    Authorization: "Bearer saas-token",
  });
  assert.deepEqual(calls[1].headers, { "Content-Type": "application/json" });
});

test("clears an expired SaaS session without calling legacy refresh", async () => {
  let refreshCalls = 0;
  let clearCalls = 0;
  const saasSession = { ...session, access_token: "saas-token" };
  const client = new ApiClient({
    getSession: () => saasSession,
    clearSession: () => {
      clearCalls += 1;
    },
    refreshLegacySession: async () => {
      refreshCalls += 1;
    },
    fetch: async () => new Response("unauthorized", { status: 401 }),
  });

  await assert.rejects(client.get("/protected", { requiresAuth: true }));
  assert.equal(refreshCalls, 0);
  assert.equal(clearCalls, 1);
});

test("preserves legacy refresh behavior on a 401", async () => {
  let refreshCalls = 0;
  let clearCalls = 0;
  const legacySession = { ...session, refresh_token: "legacy-refresh" };
  const client = new ApiClient({
    getSession: () => legacySession,
    clearSession: () => {
      clearCalls += 1;
    },
    refreshLegacySession: async () => {
      refreshCalls += 1;
    },
    fetch: async () => new Response("unauthorized", { status: 401 }),
  });

  await assert.rejects(client.get("/protected", { requiresAuth: true }));
  assert.equal(refreshCalls, 1);
  assert.equal(clearCalls, 0);
});

test("a late SaaS 401 cannot clear a newer token session", async () => {
  const current = { ...session, access_token: "old-token", refresh_token: "" };
  let resolveResponse!: (response: Response) => void;
  let clearCalls = 0;
  const client = new ApiClient({
    getSession: () => current,
    clearSession: () => {
      clearCalls += 1;
    },
    fetch: () =>
      new Promise((resolve) => {
        resolveResponse = resolve;
      }),
  });

  const request = client.get("/protected", { requiresAuth: true });
  current.access_token = "new-token";
  resolveResponse(new Response("unauthorized", { status: 401 }));
  await assert.rejects(request);

  assert.equal(clearCalls, 0);
});

test("concurrent legacy 401 responses share one refresh flight", async () => {
  const current = {
    ...session,
    access_token: "legacy-token",
    refresh_token: "legacy-refresh",
  };
  let refreshCalls = 0;
  let finishRefresh!: () => void;
  const client = new ApiClient({
    getSession: () => current,
    refreshLegacySession: () => {
      refreshCalls += 1;
      return new Promise<void>((resolve) => {
        finishRefresh = resolve;
      });
    },
    fetch: async () => new Response("unauthorized", { status: 401 }),
  });

  const requests = [
    client.get("/first", { requiresAuth: true }),
    client.get("/second", { requiresAuth: true }),
  ];
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(refreshCalls, 1);
  finishRefresh();
  await Promise.allSettled(requests);
  assert.equal(refreshCalls, 1);
});

test("a legacy refresh result cannot overwrite a session that changed while it was in flight", async () => {
  const current = {
    ...session,
    access_token: "old-legacy-token",
    refresh_token: "old-legacy-refresh",
  };
  let finishRefresh!: (session: Partial<typeof current>) => void;
  let clearCalls = 0;
  const writes: Array<Partial<typeof current>> = [];
  const client = new ApiClient({
    getSession: () => current,
    refreshLegacySession: () =>
      new Promise((resolve) => {
        finishRefresh = resolve;
      }),
    setSession: (nextSession) => {
      writes.push(nextSession);
    },
    clearSession: () => {
      clearCalls += 1;
    },
    fetch: async () => new Response("unauthorized", { status: 401 }),
  });

  const request = client.get("/protected", { requiresAuth: true });
  await new Promise((resolve) => setTimeout(resolve, 0));
  Object.assign(current, {
    access_token: "new-token",
    refresh_token: "new-refresh",
    id: "user-2",
    tenant_id: "tenant-b",
  });
  finishRefresh({
    access_token: "refreshed-old-token",
    refresh_token: "refreshed-old-refresh",
  });
  await assert.rejects(request);

  assert.deepEqual(writes, []);
  assert.equal(clearCalls, 0);
});

test("a failed legacy refresh cannot clear a session that changed while it was in flight", async () => {
  const current = {
    ...session,
    access_token: "old-legacy-token",
    refresh_token: "old-legacy-refresh",
  };
  let failRefresh!: (error: Error) => void;
  let clearCalls = 0;
  const client = new ApiClient({
    getSession: () => current,
    refreshLegacySession: () =>
      new Promise((_resolve, reject) => {
        failRefresh = reject;
      }),
    clearSession: () => {
      clearCalls += 1;
    },
    notifyError: () => {},
    fetch: async () => new Response("unauthorized", { status: 401 }),
  });

  const request = client.get("/protected", { requiresAuth: true });
  await new Promise((resolve) => setTimeout(resolve, 0));
  Object.assign(current, {
    access_token: "new-token",
    refresh_token: "new-refresh",
    id: "user-2",
    tenant_id: "tenant-b",
  });
  failRefresh(new Error("old refresh failed"));
  await assert.rejects(request);

  assert.equal(clearCalls, 0);
});

test("different legacy sessions do not share a refresh flight", async () => {
  const current = {
    ...session,
    access_token: "first-token",
    refresh_token: "first-refresh",
  };
  const refreshes: Array<{
    session: typeof current;
    finish: () => void;
  }> = [];
  const client = new ApiClient({
    getSession: () => current,
    refreshLegacySession: (refreshSession) =>
      new Promise<void>((resolve) => {
        refreshes.push({ session: { ...refreshSession }, finish: resolve });
      }),
    fetch: async () => new Response("unauthorized", { status: 401 }),
  });

  const firstRequest = client.get("/first", { requiresAuth: true });
  await new Promise((resolve) => setTimeout(resolve, 0));
  Object.assign(current, {
    access_token: "second-token",
    refresh_token: "second-refresh",
    id: "user-2",
    tenant_id: "tenant-b",
  });
  const secondRequest = client.get("/second", { requiresAuth: true });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(refreshes.length, 2);
  assert.deepEqual(
    refreshes.map(({ session: refreshSession }) => refreshSession),
    [
      {
        access_token: "first-token",
        refresh_token: "first-refresh",
        id: "user-1",
        tenant_id: "tenant-a",
      },
      {
        access_token: "second-token",
        refresh_token: "second-refresh",
        id: "user-2",
        tenant_id: "tenant-b",
      },
    ],
  );
  for (const refresh of refreshes) refresh.finish();
  await Promise.allSettled([firstRequest, secondRequest]);
});

test("a late legacy 401 cannot refresh or clear a newer session", async () => {
  const current = {
    ...session,
    access_token: "old-legacy-token",
    refresh_token: "old-legacy-refresh",
  };
  let resolveResponse!: (response: Response) => void;
  let refreshCalls = 0;
  let clearCalls = 0;
  const client = new ApiClient({
    getSession: () => current,
    refreshLegacySession: async () => {
      refreshCalls += 1;
      throw new Error("must not refresh");
    },
    clearSession: () => {
      clearCalls += 1;
    },
    fetch: () =>
      new Promise((resolve) => {
        resolveResponse = resolve;
      }),
  });

  const request = client.get("/protected", { requiresAuth: true });
  Object.assign(current, {
    access_token: "new-saas-token",
    refresh_token: "",
    id: "user-2",
    tenant_id: "tenant-b",
  });
  resolveResponse(new Response("unauthorized", { status: 401 }));
  await assert.rejects(request);
  assert.equal(refreshCalls, 0);
  assert.equal(clearCalls, 0);
});
