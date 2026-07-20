import assert from "node:assert/strict";
import test from "node:test";
import {
  QueryClient,
  QueryClientProvider,
  useQueryClient,
} from "@tanstack/react-query";
import { act, create } from "react-test-renderer";
import { SessionCacheBoundary } from "./session-cache-boundary.tsx";

(
  globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

function CachedValue({ renders }: { renders: Array<string | undefined> }) {
  const client = useQueryClient();
  const value = client.getQueryData<string>(["private"]);
  renders.push(value);
  return <>{value ?? "empty"}</>;
}

test("an identity change removes the old subtree and clears cache before rendering the new subtree", async () => {
  const client = new QueryClient();
  const renders: Array<string | undefined> = [];
  client.setQueryData(["private"], "user-a-private-data");

  let renderer!: ReturnType<typeof create>;
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>
        <SessionCacheBoundary boundary="user-a:tenant-a">
          <CachedValue renders={renders} />
        </SessionCacheBoundary>
      </QueryClientProvider>,
    );
  });
  assert.deepEqual(renders, ["user-a-private-data"]);

  await act(async () => {
    renderer.update(
      <QueryClientProvider client={client}>
        <SessionCacheBoundary boundary="user-b:tenant-b">
          <CachedValue renders={renders} />
        </SessionCacheBoundary>
      </QueryClientProvider>,
    );
  });

  assert.equal(renderer.toJSON(), "empty");
  assert.deepEqual(renders, ["user-a-private-data", undefined]);
});
