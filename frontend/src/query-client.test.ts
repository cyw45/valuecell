import assert from "node:assert/strict";
import test from "node:test";
import { QueryClient } from "@tanstack/react-query";
import { createSessionCacheBoundary, queryClient } from "./query-client.ts";

test("a failed global mutation is attempted only once", async () => {
  let calls = 0;
  await assert.rejects(
    queryClient
      .getMutationCache()
      .build(queryClient, {
        mutationFn: async () => {
          calls += 1;
          throw new Error("nope");
        },
      })
      .execute(undefined),
  );
  assert.equal(calls, 1);
});

test("logout and subsequent login cannot reuse the previous session cache", () => {
  const client = new QueryClient();
  const updateSession = createSessionCacheBoundary(
    client,
    "user-1",
    "tenant-a",
  );
  client.setQueryData(["portfolio"], "private-data");
  updateSession("", "");
  assert.equal(client.getQueryData(["portfolio"]), undefined);

  client.setQueryData(["portfolio"], "anonymous-data");
  updateSession("user-1", "tenant-a");
  assert.equal(client.getQueryData(["portfolio"]), undefined);
});
