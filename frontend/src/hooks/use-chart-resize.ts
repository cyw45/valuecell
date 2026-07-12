import type { ECharts } from "echarts/core";
import { useEffect } from "react";

/**
 * Keep ECharts in sync with both window resizes and container visibility changes.
 * This matters for charts mounted inside hidden tabs: they often initialize at 0px
 * and need a resize call once the tab becomes visible.
 */
export function useChartResize(
  chartInstance: React.RefObject<ECharts | null>,
  containerRef?: React.RefObject<HTMLElement | null>,
  dependencies: React.DependencyList = [],
) {
  useEffect(() => {
    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener("resize", handleResize);

    const container = containerRef?.current;
    const observer =
      container && typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => {
            handleResize();
          })
        : null;

    if (observer && container) {
      observer.observe(container);
      queueMicrotask(handleResize);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      observer?.disconnect();
    };
  }, [chartInstance, containerRef, ...dependencies]);
}
