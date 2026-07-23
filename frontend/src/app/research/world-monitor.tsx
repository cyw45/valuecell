import {
  Activity,
  Database,
  RadioTower,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import {
  useWorldIntelligenceSnapshots,
  useWorldIntelligenceStatus,
  type WorldIntelligenceSnapshot,
} from "@/api/world-intelligence";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const FEED_LABELS: Record<string, string> = {
  cross_source_signals: "Cross-source signals",
  market_implications: "Market implications",
  risk_scores: "Country risk",
  thermal_escalations: "Thermal escalation",
};

function formatTimestamp(value: string | null) {
  if (!value) return "Awaiting first import";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function evidenceCount(snapshot: WorldIntelligenceSnapshot) {
  if (!snapshot.payload || typeof snapshot.payload !== "object") {
    return "Evidence captured";
  }
  const payload = snapshot.payload as Record<string, unknown>;
  const collections = ["cards", "clusters", "signals", "countries", "scores"];
  for (const key of collections) {
    if (Array.isArray(payload[key]))
      return `${payload[key].length} source records`;
  }
  return "Evidence captured";
}

export default function WorldMonitorPage() {
  const status = useWorldIntelligenceStatus();
  const snapshots = useWorldIntelligenceSnapshots();
  const refreshing = status.isFetching || snapshots.isFetching;

  return (
    <div className="scroll-container flex size-full flex-col bg-background">
      <header className="border-b px-4 py-4 sm:px-6">
        <div className="mx-auto flex max-w-[1440px] flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <RadioTower className="size-5 text-sky-600" />
              <h1 className="font-semibold text-xl">World Intelligence</h1>
              <Badge variant="outline">
                <ShieldCheck /> Source-attributed
              </Badge>
            </div>
            <p className="mt-1 text-muted-foreground text-sm">
              WorldMonitor evidence stored in ValueCell for research and
              strategy context.
            </p>
          </div>
          <Button
            aria-label="Refresh intelligence evidence"
            onClick={() => {
              void status.refetch();
              void snapshots.refetch();
            }}
            variant="outline"
          >
            <RefreshCw className={refreshing ? "animate-spin" : undefined} />
            Refresh
          </Button>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-[1440px] gap-4 p-4 sm:p-6">
        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {status.data?.feeds.map((feed) => (
            <Card className="rounded-lg py-0 shadow-none" key={feed.feed}>
              <CardHeader className="px-4 pt-4 pb-2">
                <CardTitle className="text-sm">
                  {FEED_LABELS[feed.feed] ?? feed.feed}
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 text-muted-foreground text-sm">
                {formatTimestamp(feed.latest_snapshot_at)}
              </CardContent>
            </Card>
          ))}
        </section>

        <section className="grid gap-3">
          <div className="flex items-center gap-2 font-medium text-sm">
            <Database className="size-4 text-sky-600" />
            Latest evidence snapshots
          </div>
          {snapshots.data?.snapshots.map((snapshot) => (
            <Card className="rounded-lg py-0 shadow-none" key={snapshot.id}>
              <CardContent className="flex flex-col gap-2 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <p className="font-medium">
                    {FEED_LABELS[snapshot.feed] ?? snapshot.feed}
                  </p>
                  <p className="mt-1 text-muted-foreground text-sm">
                    {evidenceCount(snapshot)}
                  </p>
                </div>
                <span className="shrink-0 text-muted-foreground text-xs">
                  {formatTimestamp(snapshot.captured_at)}
                </span>
              </CardContent>
            </Card>
          ))}
          {snapshots.isLoading ? (
            <p className="py-8 text-center text-muted-foreground text-sm">
              Loading intelligence evidence...
            </p>
          ) : null}
          {!snapshots.isLoading && snapshots.data?.snapshots.length === 0 ? (
            <Card className="rounded-lg border-dashed py-0 shadow-none">
              <CardContent className="flex items-center gap-3 px-4 py-5 text-muted-foreground text-sm">
                <Activity className="size-4" />
                Waiting for the first WorldMonitor collection cycle.
              </CardContent>
            </Card>
          ) : null}
        </section>
      </main>
    </div>
  );
}
