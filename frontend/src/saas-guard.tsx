import type { ReactNode } from "react";

const PUBLIC_PATHS: readonly string[] = ["/login"];
const LEGACY_PREFIXES: readonly string[] = [
  "/home",
  "/market",
  "/agent",
  "/setting",
  "/research",
  "/test",
];

function matchesPathSegment(path: string, prefix: string) {
  return path === prefix || path.startsWith(`${prefix}/`);
}

export function isSaaSPublicRoute(path: string) {
  if (PUBLIC_PATHS.includes(path)) return true;
  if (matchesPathSegment(path, "/research/polymarket")) return false;
  return LEGACY_PREFIXES.some((prefix) => matchesPathSegment(path, prefix));
}

interface SaaSGuardBoundaryProps {
  children: ReactNode;
  hydrated: boolean;
  isLoggedIn: boolean;
  isPublicRoute: boolean;
}

export function SaaSGuardBoundary({
  children,
  hydrated,
  isLoggedIn,
  isPublicRoute,
}: SaaSGuardBoundaryProps) {
  if ((!hydrated && !isPublicRoute) || (!isLoggedIn && !isPublicRoute))
    return null;
  return <>{children}</>;
}
