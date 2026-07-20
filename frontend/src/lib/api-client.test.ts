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

test("does not call the legacy refresh route for a SaaS 401", async () => {
  let refreshCalls = 0;
  const client = new ApiClient({
    getSession: () => session,
    refreshLegacySession: async () => {
      refreshCalls += 1;
    },
    fetch: async () => new Response("unauthorized", { status: 401 }),
  });

  await assert.rejects(client.get("/protected", { requiresAuth: true }));
  assert.equal(refreshCalls, 0);
});
