# Web OKX Demo Multi-Strategy Control Plane

## Goal

Make the Web application an understandable, evidence-driven control plane for
four strategies sharing one OKX Demo account. The first release covers the
backend execution contract and Web only; Mobile remains unchanged until Web
verification is complete.

## Non-negotiable boundaries

- `okx_demo` uses one tenant + credential shared wallet as the account
  authority. Wallet equity and balances are never copied into strategy PnL.
- Strategy allocation, positions, PnL, and explanations come from immutable
  strategy-owned reservations, intents, orders, fills, and valuation facts.
- `paper` remains a separate opt-in environment. FixedPaper tables are never a
  success criterion for OKX Demo execution.
- Missing, stale, partial, or unattributed facts render an explicit status and
  reason; they are never converted to zero or inferred from current config.
- No live execution is enabled by this work.

## Data flow

```text
fixed/configurable signal
  -> execution batch
  -> shared-wallet allocator reservation
  -> durable intent/outbox
  -> OKX Demo order
  -> reconciled venue fill
  -> strategy position/PnL projection
  -> shared account summary + unified trade facts
  -> Web dashboard/detail/trades views
```

The scheduler selects the path from the persisted strategy execution
environment. Demo strategies must never fall through to a Paper fill path.
Paper evaluation and Paper ledger code remains available only when the row is
explicitly `paper`.

## Web surfaces

### Dashboard

The first viewport shows one shared-wallet summary: total equity, available
quote, reserved quote, occupied notional, sync freshness, attribution status,
and wallet-to-strategy reconciliation delta. A strategy matrix lists all four
strategies with environment, running/stopped state, current batch, reserved
and occupied quote, utilization, realized/unrealized/net PnL, return rate,
and risk or execution blockers.

### Strategy detail

The detail view is scoped by strategy and current batch. It shows the strategy
environment, allocation, positions, PnL/return metrics, an equity curve, and
the latest sync/attribution state. Empty and unavailable states include the
server reason and last observed time.

### Trade facts

The unified trade table is filtered by environment, strategy, and batch. Each
row includes symbol/pair, action, lifecycle status, reservation/intent/order
identifiers, requested and filled amounts, quantity, average fill price,
fees, and timestamps. Expandable explanation details show every persisted
condition with actual value, comparator, threshold, and data time, followed by
the decision, risk result, execution path, and failure/block reason.

`submission_unknown` is displayed as “待远端对账（不可重提）” and remains
reserved until reconciliation.

## API contract

The existing shared account summary, strategy Demo execution, PnL curve, and
unified trade-facts endpoints remain the single Web data source. Any missing
fields required by the views are added to the service normalizers and shared
wire types before UI work. Pages do not call the exchange or derive business
facts locally.

## Verification gates

1. Contract tests prove Demo strategy creation persists `okx_demo`, Paper and
   Demo paths are isolated, and fixed signals carry symbol and batch facts.
2. Scheduler tests prove a Demo signal creates reservation/intent submission
   and never writes FixedPaper fills; Paper tests prove the inverse.
3. Web typecheck, lint, and production build pass.
4. Authenticated browser verification confirms wallet totals, strategy matrix,
   detail curve, and expanded trade explanations without overlap or fabricated
   values.

