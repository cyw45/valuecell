import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      gcTime: 30 * 60 * 1000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
    mutations: {
      retry: 0,
    },
  },
});

export function createSessionCacheBoundary(
  client: QueryClient,
  initialUserId: string,
  initialTenantId: string,
) {
  let previousBoundary = `${initialUserId}:${initialTenantId}`;
  return (userId: string, tenantId: string) => {
    const boundary = `${userId}:${tenantId}`;
    if (previousBoundary !== boundary) client.clear();
    previousBoundary = boundary;
  };
}
