import { useId } from "react";
import { cn } from "@/lib/utils";

interface ThresholdGaugeProps {
  label: string;
  value: number | null;
  displayValue: string;
  description: string;
  thresholds: readonly [string, string, string];
  className?: string;
}

const clampPercent = (value: number): number => Math.min(Math.max(value, 0), 100);

export function ThresholdGauge({
  label,
  value,
  displayValue,
  description,
  thresholds,
  className,
}: ThresholdGaugeProps) {
  const gradientId = useId().replace(/:/g, "");
  const progress = value === null ? 0 : clampPercent(value);
  const accessibleValue = value === null ? "暂无数据" : displayValue;

  return (
    <article
      aria-label={`${label}：${accessibleValue}。${description}`}
      className={cn(
        "dashboard-panel rounded-lg border border-sky-500/15 bg-card/90 px-4 py-3",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="terminal-label">{label}</p>
          <p className="mt-1 text-muted-foreground text-xs">{description}</p>
        </div>
        <svg
          aria-hidden="true"
          className="h-[86px] w-[152px] shrink-0 overflow-visible text-foreground"
          viewBox="0 0 160 92"
        >
          <defs>
            <linearGradient gradientUnits="userSpaceOnUse" id={gradientId} x1="20" x2="140" y1="80" y2="80">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="22%" stopColor="#84cc16" />
              <stop offset="45%" stopColor="#facc15" />
              <stop offset="66%" stopColor="#fb923c" />
              <stop offset="84%" stopColor="#f97316" />
              <stop offset="100%" stopColor="#f43f5e" />
            </linearGradient>
          </defs>
          <path
            d="M 20 80 A 60 60 0 0 1 140 80"
            fill="none"
            className="text-muted-foreground/20"
            pathLength="100"
            stroke="currentColor"
            strokeLinecap="round"
            strokeWidth="12"
          />
          <path
            d="M 20 80 A 60 60 0 0 1 140 80"
            fill="none"
            pathLength="100"
            stroke={`url(#${gradientId})`}
            strokeDasharray={`${progress} 100`}
            strokeLinecap="round"
            strokeWidth="12"
          />
          <text
            fill="currentColor"
            fontSize="24"
            fontWeight="700"
            textAnchor="middle"
            x="80"
            y="70"
          >
            {displayValue}
          </text>
        </svg>
      </div>
      <div className="mt-1 grid grid-cols-3 text-[10px] text-muted-foreground tabular-nums">
        <span>{thresholds[0]}</span>
        <span className="text-center">{thresholds[1]}</span>
        <span className="text-right">{thresholds[2]}</span>
      </div>
    </article>
  );
}
