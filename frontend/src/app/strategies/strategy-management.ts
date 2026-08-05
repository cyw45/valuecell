import type { RuleStrategy } from "@/types/rule-strategy";

export function strategyManagementActions({
  selectedStatus,
}: {
  selectedStatus: RuleStrategy["status"] | undefined;
}) {
  const isRunning = selectedStatus === "running";
  return {
    canSave: !isRunning,
    canStart: selectedStatus === "stopped",
    canStop: isRunning,
    canDelete: selectedStatus === "stopped",
  };
}
