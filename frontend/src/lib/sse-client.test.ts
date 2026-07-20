import assert from "node:assert/strict";
import test from "node:test";
import { parseSSEData, SSEClient } from "./sse-client.ts";

async function collectEvent(data: string) {
  const received: unknown[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(`data: ${data}\n\n`));
          controller.close();
        },
      }),
      { status: 200 },
    );
  try {
    const closed = new Promise<void>((resolve) => {
      new SSEClient(
        { url: "/events" },
        { onData: (value) => received.push(value), onClose: resolve },
      ).connect();
    });
    await closed;
    return received;
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test("normal and best-effort JSON payloads are delivered to onData", async () => {
  assert.deepEqual(await collectEvent('{"kind":"complete"}'), [
    { kind: "complete" },
  ]);
  assert.deepEqual(
    await collectEvent('{"kind":"partial","text":"still streaming'),
    [{ kind: "partial", text: "still streaming" }],
  );
});

test("SSE parse failures log metadata without exposing raw event data", () => {
  const secret = "access_token=super-secret-value";
  const warnings: unknown[][] = [];

  const result = parseSSEData(`:${secret}`, (...args) => warnings.push(args));

  assert.equal(result, undefined);
  assert.equal(warnings.length, 1);
  const warning = JSON.stringify(warnings[0]);
  assert.match(warning, /Failed to parse SSE message/);
  assert.match(warning, /length/);
  assert.doesNotMatch(warning, new RegExp(secret));
  assert.doesNotMatch(warning, /super-secret-value/);
});
