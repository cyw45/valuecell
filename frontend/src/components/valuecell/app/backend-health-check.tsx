import type React from "react";
import { useBackendHealth } from "@/api/system";

/**
 * Keeps a lightweight backend reachability probe alive without blocking the
 * paper-trading workspace. Individual API panels report actionable failures.
 */
export function BackendHealthCheck({
  children,
}: {
  children: React.ReactNode;
}) {
  useBackendHealth();
  return <>{children}</>;
}
