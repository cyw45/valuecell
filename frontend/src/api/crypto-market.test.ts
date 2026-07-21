import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("./crypto-market.ts", import.meta.url), "utf8");

test("market indicator query keeps previous data when its key changes", () => {
  assert.match(source, /import \{ keepPreviousData, useQuery \}/);
  assert.match(
    source,
    /useQuery\(\{[\s\S]*?queryKey: API_QUERY_KEYS\.CRYPTO_MARKET\.indicators[\s\S]*?placeholderData: keepPreviousData,/,
  );
});
