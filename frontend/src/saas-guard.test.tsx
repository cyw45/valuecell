import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import { isSaaSPublicRoute, SaaSGuardBoundary } from "./saas-guard.tsx";

test("public routes mount before hydration", () => {
  const html = renderToStaticMarkup(
    <SaaSGuardBoundary hydrated={false} isLoggedIn={false} isPublicRoute={true}>
      <div>login</div>
    </SaaSGuardBoundary>,
  );
  assert.match(html, /login/);
});

test("legacy matching is boundary safe and research polymarket is protected", () => {
  assert.equal(isSaaSPublicRoute("/market"), true);
  assert.equal(isSaaSPublicRoute("/market/BTC"), true);
  assert.equal(isSaaSPublicRoute("/marketplace"), false);
  assert.equal(isSaaSPublicRoute("/research/polymarket"), false);
});

test("protected children are not mounted before hydration", () => {
  let mounts = 0;
  function ProtectedChild() {
    mounts += 1;
    return <div>secret</div>;
  }

  const html = renderToStaticMarkup(
    <SaaSGuardBoundary hydrated={false} isLoggedIn={true} isPublicRoute={false}>
      <ProtectedChild />
    </SaaSGuardBoundary>,
  );

  assert.equal(mounts, 0);
  assert.equal(html, "");
});

test("protected children are not mounted for an unauthenticated session", () => {
  let mounts = 0;
  function ProtectedChild() {
    mounts += 1;
    return <div>secret</div>;
  }

  renderToStaticMarkup(
    <SaaSGuardBoundary hydrated={true} isLoggedIn={false} isPublicRoute={false}>
      <ProtectedChild />
    </SaaSGuardBoundary>,
  );
  assert.equal(mounts, 0);
});

test("authenticated and public routes may mount children after hydration", () => {
  const protectedHtml = renderToStaticMarkup(
    <SaaSGuardBoundary hydrated={true} isLoggedIn={true} isPublicRoute={false}>
      <div>protected</div>
    </SaaSGuardBoundary>,
  );
  const publicHtml = renderToStaticMarkup(
    <SaaSGuardBoundary hydrated={true} isLoggedIn={false} isPublicRoute={true}>
      <div>public</div>
    </SaaSGuardBoundary>,
  );

  assert.match(protectedHtml, /protected/);
  assert.match(publicHtml, /public/);
});
