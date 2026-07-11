import {
  Activity,
  AlertTriangle,
  Clock3,
  Database,
  Gauge,
  Layers3,
  Play,
  RefreshCw,
  ShieldAlert,
  TestTube2,
} from "lucide-react";
import { type FormEvent, useEffect, useMemo, useState } from "react";
import {
  usePredictionMarketCatalog,
  usePredictionMarketSignal,
  usePredictionMarketSnapshot,
  usePredictionReplayPreview,
} from "@/api/prediction-market";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TimeUtils } from "@/lib/time";
import { numberFixed } from "@/lib/utils";
import type {
  PredictionMarketBookLevel,
  PredictionMarketCatalog,
  PredictionMarketFreshnessStatus,
  PredictionMarketSnapshot,
} from "@/types/prediction-market";

const formatProbability = (value?: string) =>
  value === undefined ? "--" : `${numberFixed(Number(value) * 100, 2)}%`;

const formatNumber = (value?: string, decimals = 4) =>
  value === undefined ? "--" : numberFixed(Number(value), decimals);

const formatTimestamp = (timestamp?: number) =>
  timestamp ? `${TimeUtils.formatUTC(timestamp)} local` : "Not available";

const freshnessStyle: Record<PredictionMarketFreshnessStatus, string> = {
  fresh: "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  delayed: "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  stale: "border-orange-500/30 bg-orange-500/10 text-orange-700 dark:text-orange-300",
  unavailable: "border-destructive/30 bg-destructive/10 text-destructive",
};

function SourceStrip({
  metadata,
}: {
  metadata?: PredictionMarketCatalog | PredictionMarketSnapshot;
}) {
  const freshness = metadata?.freshness_status ?? "unavailable";
  const age = metadata?.freshness_age_ms;

  return (
    <div className="grid gap-3 border-y bg-muted/30 px-4 py-3 text-xs sm:grid-cols-[auto_1fr_auto_auto] sm:items-center">
      <Badge className="w-fit" variant="outline">
        <Database /> {metadata?.source ?? "Public source"}
      </Badge>
      <span className="text-muted-foreground">
        Source timestamp: {formatTimestamp(metadata?.source_timestamp_ms)} | Observed: {formatTimestamp(metadata?.observed_at_ms)}
      </span>
      <span className="text-muted-foreground">
        Age: {age === undefined ? "--" : `${numberFixed(age / 1000, 1)}s`}
      </span>
      <Badge className={freshnessStyle[freshness]} variant="outline">
        <Clock3 /> {freshness}
      </Badge>
    </div>
  );
}

function BookSide({
  levels,
  side,
}: {
  levels: PredictionMarketBookLevel[];
  side: "bid" | "ask";
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>{side === "bid" ? "Bid probability" : "Ask probability"}</TableHead>
          <TableHead className="text-right">Contracts</TableHead>
          <TableHead className="text-right">Cumulative</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {levels.slice(0, 8).map((level, index) => {
          const cumulative = levels
            .slice(0, index + 1)
            .reduce((total, current) => total + Number(current.size), 0);
          return (
            <TableRow key={`${side}-${level.price}-${index}`}>
              <TableCell
                className={
                  side === "bid"
                    ? "font-medium text-emerald-700 dark:text-emerald-300"
                    : "font-medium text-rose-700 dark:text-rose-300"
                }
              >
                {formatProbability(level.price)}
              </TableCell>
              <TableCell className="text-right tabular-nums">{formatNumber(level.size, 2)}</TableCell>
              <TableCell className="text-right tabular-nums text-muted-foreground">
                {numberFixed(cumulative, 2)}
              </TableCell>
            </TableRow>
          );
        })}
        {levels.length === 0 ? (
          <TableRow>
            <TableCell className="py-8 text-center text-muted-foreground" colSpan={3}>
              No public {side}s available
            </TableCell>
          </TableRow>
        ) : null}
      </TableBody>
    </Table>
  );
}

export default function PolymarketResearch() {
  const catalog = usePredictionMarketCatalog();
  const [marketId, setMarketId] = useState<string>();
  const [outcome, setOutcome] = useState<string>();
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [size, setSize] = useState("100");
  const [latencyMs, setLatencyMs] = useState("250");
  const [history, setHistory] = useState<string[]>([]);
  const snapshot = usePredictionMarketSnapshot(marketId, outcome);
  const signal = usePredictionMarketSignal(marketId, outcome, history);
  const replay = usePredictionReplayPreview();

  const selectedMarket = useMemo(
    () => catalog.data?.markets.find((market) => market.market_id === marketId),
    [catalog.data?.markets, marketId],
  );
  const activeMarkets = useMemo(
    () => catalog.data?.markets.filter((market) => market.active && !market.closed) ?? [],
    [catalog.data?.markets],
  );

  useEffect(() => {
    if (marketId || activeMarkets.length === 0) return;
    const firstMarket = activeMarkets[0];
    setMarketId(firstMarket.market_id);
    setOutcome(firstMarket.outcomes[0]?.outcome);
  }, [activeMarkets, marketId]);

  useEffect(() => {
    const reference = snapshot.data?.book.microprice ?? snapshot.data?.book.midpoint;
    if (!reference) return;
    setHistory((current) => [...current, reference].slice(-32));
  }, [snapshot.data?.book.microprice, snapshot.data?.book.midpoint]);

  const runReplay = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!snapshot.data?.book || !detail?.source_timestamp_ms || !detail.observed_at_ms) return;
    replay.mutate({
      decision_time_ms: Date.now(),
      latency_ms: Number(latencyMs),
      order: {
        side,
        size: Number(size),
        max_levels: 8,
        extra_slippage_bps: 0,
      },
      snapshots: [
        {
          source_timestamp_ms: detail.source_timestamp_ms,
          observed_at_ms: detail.observed_at_ms,
          bids: detail.book.bids,
          asks: detail.book.asks,
        },
      ],
    });
  };

  const detail = signal.data ?? snapshot.data;
  const book = detail?.book;
  const loading = catalog.isLoading || snapshot.isLoading;
  const unavailable = Boolean(catalog.error || snapshot.error) || (!catalog.isLoading && activeMarkets.length === 0);

  return (
    <div className="scroll-container flex size-full flex-col bg-background">
      <div className="border-b px-4 py-4 sm:px-6">
        <div className="mx-auto flex max-w-[1600px] flex-col gap-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-xl font-semibold">Polymarket Research</h1>
                <Badge variant="outline"><TestTube2 /> Paper and simulated only</Badge>
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                Public Gamma catalog and CLOB order-book observations. No wallet, signing, account data, or live execution.
              </p>
            </div>
            <Button
              aria-label="Refresh public market data"
              variant="outline"
              onClick={() => {
                void catalog.refetch();
                void snapshot.refetch();
                void signal.refetch();
              }}
            >
              <RefreshCw /> Refresh public data
            </Button>
          </div>
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_220px]">
            <Select
              value={marketId}
              onValueChange={(nextMarketId) => {
                const nextMarket = activeMarkets.find((market) => market.market_id === nextMarketId);
                setMarketId(nextMarketId);
                setOutcome(nextMarket?.outcomes[0]?.outcome);
                setHistory([]);
              }}
            >
              <SelectTrigger aria-label="Market selector"><SelectValue placeholder="Select a public market" /></SelectTrigger>
              <SelectContent>
                {activeMarkets.map((market) => (
                  <SelectItem key={market.market_id} value={market.market_id}>{market.question}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={outcome} onValueChange={setOutcome} disabled={!selectedMarket}>
              <SelectTrigger aria-label="Outcome selector"><SelectValue placeholder="Select outcome" /></SelectTrigger>
              <SelectContent>
                {selectedMarket?.outcomes.map((item) => (
                  <SelectItem key={item.token_id} value={item.outcome}>{item.outcome}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      <SourceStrip metadata={detail ?? catalog.data} />

      <div className="mx-auto grid w-full max-w-[1600px] gap-4 p-4 sm:p-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <section className="flex min-w-0 flex-col gap-4">
          {unavailable ? (
            <Alert>
              <AlertTriangle />
              <AlertTitle>{activeMarkets.length === 0 ? "No active public markets are available" : "Public market data is unavailable"}</AlertTitle>
              <AlertDescription>
                The workstation retained no inferred prices. Check source freshness and retry; research and replay controls stay disabled until a public snapshot is returned.
              </AlertDescription>
            </Alert>
          ) : null}

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["Reference probability", formatProbability(book?.microprice ?? book?.midpoint), "Microprice when available"],
              ["Best bid", formatProbability(book?.best_bid), "Public CLOB bid"],
              ["Best ask", formatProbability(book?.best_ask), "Public CLOB ask"],
              ["Book condition", book?.health?.status ?? (loading ? "Loading" : "Unknown"), book?.health?.reason ?? "Validate before interpreting"],
            ].map(([label, value, hint]) => (
              <Card className="gap-2 rounded-lg py-4 shadow-none" key={label}>
                <CardContent className="px-4">
                  <p className="text-xs text-muted-foreground">{label}</p>
                  <p className="mt-1 font-semibold text-lg tabular-nums">{value}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <CardHeader className="border-b px-4 py-4 sm:px-5">
              <CardTitle className="flex items-center gap-2 text-base"><Layers3 /> Public order-book depth</CardTitle>
              <CardDescription>Displayed levels are observations, not executable quotes or liquidity guarantees.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-0 px-0 md:grid-cols-2 md:divide-x">
              <BookSide levels={book?.bids ?? []} side="bid" />
              <BookSide levels={book?.asks ?? []} side="ask" />
            </CardContent>
          </Card>
        </section>

        <aside className="flex flex-col gap-4">
          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <CardHeader className="border-b px-4 py-4">
              <CardTitle className="flex items-center gap-2 text-base"><Activity /> Research signal</CardTitle>
              <CardDescription>Probability movement summary from locally observed public snapshots.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 px-4 py-4 text-sm">
              <div className="grid grid-cols-2 gap-3">
                <div><p className="text-muted-foreground text-xs">Reference</p><p className="mt-1 font-medium tabular-nums">{formatProbability(detail?.signal?.reference_price)}</p></div>
                <div><p className="text-muted-foreground text-xs">Volatility</p><p className="mt-1 font-medium tabular-nums">{formatNumber(detail?.signal?.volatility, 6)}</p></div>
                <div><p className="text-muted-foreground text-xs">Method</p><p className="mt-1 font-medium">{detail?.signal?.reference_method ?? "Awaiting observations"}</p></div>
                <div><p className="text-muted-foreground text-xs">Observations</p><p className="mt-1 font-medium tabular-nums">{detail?.signal?.observation_count ?? history.length}</p></div>
              </div>
              <Alert>
                <Gauge />
                <AlertTitle>{detail?.signal?.volatility_status ?? "Caveat"}</AlertTitle>
                <AlertDescription>Signals describe observed probability variation only. They do not estimate outcome likelihood or performance.</AlertDescription>
              </Alert>
            </CardContent>
          </Card>

          <Card className="gap-0 rounded-lg py-0 shadow-none">
            <CardHeader className="border-b px-4 py-4">
              <CardTitle className="flex items-center gap-2 text-base"><Play /> Paper replay</CardTitle>
              <CardDescription>Deterministic simulation against the received public book; no live order is submitted.</CardDescription>
            </CardHeader>
            <CardContent className="px-4 py-4">
              <form className="grid gap-3" onSubmit={runReplay}>
                <div className="grid grid-cols-2 gap-3">
                  <Select value={side} onValueChange={(value) => setSide(value as "buy" | "sell")}>
                    <SelectTrigger aria-label="Paper replay side"><SelectValue /></SelectTrigger>
                    <SelectContent><SelectItem value="buy">Paper buy</SelectItem><SelectItem value="sell">Paper sell</SelectItem></SelectContent>
                  </Select>
                  <Input aria-label="Paper replay size" inputMode="decimal" min="0" onChange={(event) => setSize(event.target.value)} type="number" value={size} />
                </div>
                <Input aria-label="Assumed latency milliseconds" inputMode="numeric" min="0" onChange={(event) => setLatencyMs(event.target.value)} type="number" value={latencyMs} />
                <p className="text-xs leading-relaxed text-muted-foreground">Assumptions: {latencyMs || "0"} ms latency, visible frozen public book levels, no extra slippage, and unfilled remainder is cancelled. Results are simulated and not performance guarantees.</p>
                <Button disabled={!snapshot.data || replay.isPending || !size || Number(size) <= 0} type="submit">
                  <TestTube2 /> {replay.isPending ? "Simulating" : "Run paper replay"}
                </Button>
              </form>
              {replay.error ? <p className="mt-3 text-sm text-destructive">Simulation unavailable. No result was inferred.</p> : null}
              {replay.data ? (
                <div className="mt-4 grid gap-3 border-t pt-4 text-sm">
                  <div className="flex items-center justify-between"><span className="text-muted-foreground">Simulated VWAP</span><span className="font-medium tabular-nums">{replay.data.fill.vwap === null ? "--" : `${numberFixed(replay.data.fill.vwap * 100, 2)}%`}</span></div>
                  <div className="flex items-center justify-between"><span className="text-muted-foreground">Filled / requested</span><span className="font-medium tabular-nums">{numberFixed(replay.data.fill.filled_size, 2)} / {numberFixed(replay.data.fill.requested_size, 2)}</span></div>
                  <div className="flex items-center justify-between"><span className="text-muted-foreground">Unfilled remainder</span><span className="font-medium tabular-nums">{numberFixed(replay.data.fill.unfilled_size, 2)}</span></div>
                  <div className="flex items-center justify-between"><span className="text-muted-foreground">Marked P&L</span><span className="font-medium tabular-nums">{numberFixed(replay.data.mark_to_book.pnl, 4)} {replay.data.mark_to_book.currency}</span></div>
                  <Badge className="w-fit" variant="outline"><TestTube2 /> {replay.data.simulation_mode} | {Number(replay.data.fill.unfilled_size) > 0 ? "Partial fill" : "Full fill"}</Badge>
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Alert>
            <ShieldAlert />
            <AlertTitle>Research-only safeguards</AlertTitle>
            <AlertDescription>Public source data may be delayed, incomplete, stale, or unavailable. This screen cannot access a wallet and cannot place a live order.</AlertDescription>
          </Alert>
        </aside>
      </div>
    </div>
  );
}
