import assert from "node:assert/strict";
import test from "node:test";
import { buildWorldMonitorDashboardUrl } from "./world-monitor-url.ts";

test("uses the dashboard route instead of the upstream landing page", () => {
  assert.equal(
    buildWorldMonitorDashboardUrl("https://wm.example.com"),
    "https://wm.example.com/dashboard?lang=zh",
  );
});

test("normalizes trailing slashes without duplicating dashboard", () => {
  assert.equal(
    buildWorldMonitorDashboardUrl("https://wm.example.com/dashboard/"),
    "https://wm.example.com/dashboard?lang=zh",
  );
});

test("preserves configured query parameters and enforces Chinese", () => {
  assert.equal(
    buildWorldMonitorDashboardUrl(
      "https://wm.example.com/base?embed=1&lang=en",
    ),
    "https://wm.example.com/base/dashboard?embed=1&lang=zh",
  );
});
