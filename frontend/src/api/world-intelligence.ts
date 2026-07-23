import { useQuery } from "@tanstack/react-query";
import { type ApiResponse, apiClient } from "@/lib/api-client";

export type WorldIntelligenceFeedStatus = {
  feed: string;
  latest_snapshot_at: string | null;
};

export type WorldIntelligenceStatus = {
  enabled: boolean;
  feeds: WorldIntelligenceFeedStatus[];
};

export type WorldIntelligenceSnapshot = {
  id: number;
  feed: string;
  payload: unknown;
  captured_at: string;
};

export type WorldIntelligenceSnapshotList = {
  snapshots: WorldIntelligenceSnapshot[];
};

export function useWorldIntelligenceStatus() {
  return useQuery({
    queryKey: ["world-intelligence", "status"],
    queryFn: async () => {
      const response = await apiClient.get<
        ApiResponse<WorldIntelligenceStatus>
      >("/world-intelligence/status");
      return response.data;
    },
    refetchInterval: 60_000,
  });
}

export function useWorldIntelligenceSnapshots() {
  return useQuery({
    queryKey: ["world-intelligence", "snapshots"],
    queryFn: async () => {
      const response = await apiClient.get<
        ApiResponse<WorldIntelligenceSnapshotList>
      >("/world-intelligence/snapshots?limit=12");
      return response.data;
    },
    refetchInterval: 60_000,
  });
}
