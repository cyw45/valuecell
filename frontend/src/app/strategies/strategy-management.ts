import type { RuleStrategy } from "@/types/rule-strategy";

export function strategyManagementActions({
  selectedStatus,
  anotherRunning,
}: {
  selectedStatus: RuleStrategy["status"] | undefined;
  anotherRunning: boolean;
}) {
  const isRunning = selectedStatus === "running";
  return {
    canSave: !isRunning,
    canStart: selectedStatus === "stopped" && !anotherRunning,
    canStop: isRunning,
    canDelete: selectedStatus === "stopped",
  };
}
