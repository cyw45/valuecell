import { useQueryClient } from "@tanstack/react-query";
import { useLayoutEffect, useState } from "react";

export function SessionCacheBoundary({
  boundary,
  children,
}: {
  boundary: string;
  children: React.ReactNode;
}) {
  const client = useQueryClient();
  const [visibleBoundary, setVisibleBoundary] = useState(boundary);

  useLayoutEffect(() => {
    if (visibleBoundary === boundary) return;
    client.clear();
    setVisibleBoundary(boundary);
  }, [boundary, client, visibleBoundary]);

  if (visibleBoundary !== boundary) return null;
  return <>{children}</>;
}
