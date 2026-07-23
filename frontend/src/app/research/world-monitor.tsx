import {
  AlertTriangle,
  ExternalLink,
  RadioTower,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { useMemo, useState } from "react";
import {
  useWorldIntelligenceSnapshots,
  useWorldIntelligenceStatus,
  type WorldIntelligenceSnapshot,
} from "@/api/world-intelligence";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const FEED_LABELS: Record<string, string> = {
  cross_source_signals: "跨源事件",
  market_implications: "市场影响",
  risk_scores: "国家与地区风险",
  thermal_escalations: "热异常升级",
};

const WORLDMONITOR_DASHBOARD_URL = `${
  import.meta.env.VITE_WORLDMONITOR_URL || "http://127.0.0.1:3001"
}/?lang=zh`;

function formatTimestamp(value: string | null) {
  if (!value) return "等待首次采集";
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function levelStyle(level: string) {
  if (level === "严重" || level === "高") {
    return "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300";
  }
  if (level === "中等") {
    return "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300";
  }
  return "border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-300";
}

function SummaryCard({ snapshot }: { snapshot: WorldIntelligenceSnapshot }) {
  const summary = snapshot.summary_zh;
  return (
    <Card className="rounded-lg py-0 shadow-none">
      <CardHeader className="border-b px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <CardTitle className="text-base">{summary.title}</CardTitle>
            <p className="mt-1 text-muted-foreground text-xs">
              {FEED_LABELS[snapshot.feed] ?? snapshot.feed} ·{" "}
              {formatTimestamp(snapshot.captured_at)}
            </p>
          </div>
          <Badge className={levelStyle(summary.level)} variant="outline">
            {summary.level}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="grid gap-4 px-4 py-4">
        {summary.metrics.length > 0 ? (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {summary.metrics.map((metric) => (
              <div
                className="rounded-md border bg-muted/20 px-3 py-2"
                key={metric.label}
              >
                <p className="text-muted-foreground text-xs">{metric.label}</p>
                <p className="mt-1 font-semibold tabular-nums">
                  {metric.value}
                </p>
              </div>
            ))}
          </div>
        ) : null}
        {summary.highlights.length > 0 ? (
          <ul className="grid gap-2 text-sm">
            {summary.highlights.map((highlight) => (
              <li className="leading-6" key={highlight}>
                {highlight}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-muted-foreground text-sm">
            当前周期暂无可展示的有效线索。
          </p>
        )}
        {summary.data_notice ? (
          <div className="flex items-start gap-2 border-amber-500/30 border-t pt-3 text-amber-700 text-xs dark:text-amber-300">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span>{summary.data_notice}</span>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

export default function WorldMonitorPage() {
  const status = useWorldIntelligenceStatus();
  const snapshots = useWorldIntelligenceSnapshots();
  const [iframeKey, setIframeKey] = useState(0);
  const refreshing = status.isFetching || snapshots.isFetching;
  const latestSnapshots = useMemo(() => {
    const seen = new Set<string>();
    return (snapshots.data?.snapshots ?? []).filter((snapshot) => {
      if (seen.has(snapshot.feed)) return false;
      seen.add(snapshot.feed);
      return true;
    });
  }, [snapshots.data?.snapshots]);

  const refreshBrief = () => {
    void status.refetch();
    void snapshots.refetch();
  };

  return (
    <div className="scroll-container flex size-full flex-col bg-background">
      <header className="border-b px-4 py-4 sm:px-6">
        <div className="mx-auto flex max-w-[1600px] flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <RadioTower className="size-5 text-sky-600" />
              <h1 className="font-semibold text-xl">全球情报</h1>
              <Badge variant="outline">
                <ShieldCheck /> 来源可追溯
              </Badge>
            </div>
            <p className="mt-1 text-muted-foreground text-sm">
              ValueCell 中文研判与 WorldMonitor 原始仪表盘对照视图。
            </p>
          </div>
          <Button
            aria-label="刷新全球情报"
            onClick={refreshBrief}
            variant="outline"
          >
            <RefreshCw className={refreshing ? "animate-spin" : undefined} />
            刷新情报
          </Button>
        </div>
      </header>

      <Tabs
        className="mx-auto flex w-full max-w-[1600px] flex-1 flex-col gap-0"
        defaultValue="brief"
      >
        <div className="border-b px-4 py-3 sm:px-6">
          <TabsList>
            <TabsTrigger value="brief">中文研判</TabsTrigger>
            <TabsTrigger value="dashboard">原始仪表盘</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent className="m-0 grid gap-4 p-4 sm:p-6" value="brief">
          <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {status.data?.feeds.map((feed) => (
              <div className="rounded-md border px-3 py-3" key={feed.feed}>
                <p className="font-medium text-sm">
                  {FEED_LABELS[feed.feed] ?? feed.feed}
                </p>
                <p className="mt-1 text-muted-foreground text-xs">
                  {formatTimestamp(feed.latest_snapshot_at)}
                </p>
              </div>
            ))}
          </section>

          {snapshots.error ? (
            <Alert variant="destructive">
              <AlertTriangle />
              <AlertTitle>中文情报加载失败</AlertTitle>
              <AlertDescription>
                请刷新后重试，或切换到原始仪表盘查看实时数据。
              </AlertDescription>
            </Alert>
          ) : null}

          <section className="grid gap-3 lg:grid-cols-2">
            {latestSnapshots.map((snapshot) => (
              <SummaryCard key={snapshot.id} snapshot={snapshot} />
            ))}
          </section>

          {snapshots.isLoading ? (
            <p className="py-10 text-center text-muted-foreground text-sm">
              正在生成中文情报摘要...
            </p>
          ) : null}
          {!snapshots.isLoading && latestSnapshots.length === 0 ? (
            <Alert>
              <RadioTower />
              <AlertTitle>暂无情报快照</AlertTitle>
              <AlertDescription>
                正在等待 WorldMonitor 完成首次采集。
              </AlertDescription>
            </Alert>
          ) : null}
        </TabsContent>

        <TabsContent
          className="m-0 flex min-h-[720px] flex-1 flex-col p-4 sm:p-6"
          value="dashboard"
        >
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <p className="text-muted-foreground text-sm">
              原始数据与地图由已配置的 WorldMonitor 仪表盘提供。
            </p>
            <div className="flex items-center gap-2">
              <Button
                onClick={() => setIframeKey((value) => value + 1)}
                size="sm"
                variant="outline"
              >
                <RefreshCw /> 刷新仪表盘
              </Button>
              <Button asChild size="sm" variant="outline">
                <a
                  href={WORLDMONITOR_DASHBOARD_URL}
                  rel="noreferrer"
                  target="_blank"
                >
                  <ExternalLink /> 新窗口打开
                </a>
              </Button>
            </div>
          </div>
          <iframe
            allow="fullscreen"
            className="min-h-[680px] w-full flex-1 border"
            key={iframeKey}
            sandbox="allow-downloads allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts"
            src={WORLDMONITOR_DASHBOARD_URL}
            title="WorldMonitor 原始全球情报仪表盘"
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
