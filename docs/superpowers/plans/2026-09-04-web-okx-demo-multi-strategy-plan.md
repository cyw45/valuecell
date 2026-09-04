# Web OKX Demo Multi-Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make four strategies execute through the shared OKX Demo path and make the Web UI expose wallet facts, strategy allocations, trade lifecycle, conditions, and PnL clearly.

**Architecture:** Persisted strategy environment selects exactly one execution path. `okx_demo` uses shared-account reservation, intent, order, fill, and projection facts; `paper` remains isolated in FixedPaper tables. Web reads normalized account summary, Demo execution, PnL curve, and unified trade facts without deriving business results locally.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, pytest, React, TanStack Query, TypeScript, Recharts/existing chart components, Bun/Vite.

**Spec:** `docs/superpowers/specs/2026-09-04-web-okx-demo-multi-strategy-design.md`

## Global Constraints

- Web first; do not modify Mobile in this plan.
- `okx_demo` and `paper` are separate evidence chains.
- Never use shared wallet balances as strategy PnL.
- Never resubmit `submission_unknown` orders.
- Missing or stale facts render explicit unavailable/blocking states.
- Live execution remains disabled.
- Use TDD: each behavior change starts with a failing test.

---

### Task 1: Freeze Demo environment at strategy creation

**Files:**
- Modify: `frontend/src/app/dashboard-strategy-management.tsx`
- Modify: `frontend/src/api/rule-strategy.ts`
- Modify: `python/valuecell/server/api/routers/rule_strategy.py`
- Modify: `python/valuecell/server/services/rule_strategy_service.py`
- Test: `python/valuecell/server/tests/test_rule_strategy_api.py`
- Test: `frontend/src/app/dashboard-strategy-management.test.tsx` or the existing component test location

**Interfaces:** Fixed strategy creation accepts an explicit environment and Demo credential. The response must persist and return `config.execution.environment = "okx_demo"`; ordinary configurable strategies retain explicit user selection.

- [ ] Write a backend test asserting fixed creation with `environment="okx_demo"` persists Demo config and shared-account scope.
- [ ] Run the focused test and verify it fails because the Web fixed creator currently sends `paper` and the default remains `paper`.
- [ ] Change the Web fixed creator to require/select the configured Demo credential and send `okx_demo`; reject creation when no valid Demo credential is available.
- [ ] Change backend validation/default handling so fixed Demo creation cannot silently fall back to Paper when the Web request is intended for the shared Demo flow.
- [ ] Add a regression assertion that Paper fixed creation remains explicit and isolated.
- [ ] Run backend focused tests and Web test/typecheck.

### Task 2: Connect fixed Demo scheduler to the canonical execution path

**Files:**
- Modify: `python/valuecell/server/services/strategy_scheduler.py`
- Modify: `python/valuecell/server/services/fixed_strategy_paper_service.py`
- Modify: `python/valuecell/server/services/fixed_strategy_paper_ledger.py` only if required for explicit Paper isolation
- Test: `python/valuecell/server/tests/test_strategy_scheduler.py`
- Test: `python/valuecell/server/tests/test_fixed_strategy_paper_service.py`

**Interfaces:** Demo signals produce shared allocator/intent/order facts; Paper signals may produce FixedPaper facts only when the strategy environment is `paper`. Both paths retain `symbol`, action, evaluation ID, and batch ID.

- [ ] Add a failing scheduler test proving a Demo `long_entry` invokes the shared execution boundary and does not create FixedPaper rows.
- [ ] Add a failing Paper test proving a Paper signal enters FixedPaper ledger with symbol and idempotent evaluation identity.
- [ ] Add `symbol` to the persisted fixed journal result and normalize `long_entry`, `short_entry`, and `exit` consistently.
- [ ] Ensure the Demo adapter maps only supported spot actions and records blocked short actions as explicit Demo execution facts.
- [ ] Ensure Paper ledger invocation is guarded by `environment == "paper"` and uses the durable batch/evaluation IDs.
- [ ] Run the focused scheduler, fixed service, and ledger tests; then run the full related backend test set.

### Task 3: Complete Demo strategy allocation and PnL read models

**Files:**
- Modify: `python/valuecell/server/services/multi_strategy_account_summary.py`
- Modify: `python/valuecell/server/services/rule_strategy_demo_execution_read_model.py`
- Modify: `python/valuecell/server/services/multi_strategy_trade_facts.py`
- Modify: `python/valuecell/server/api/routers/rule_strategy.py`
- Modify: `frontend/src/types/multi-strategy.ts`
- Modify: `frontend/src/types/rule-strategy-demo-execution.ts`
- Test: `python/valuecell/server/tests/test_multi_strategy_account_summary.py`
- Test: `python/valuecell/server/tests/test_multi_strategy_trade_facts.py`
- Test: `python/valuecell/server/tests/test_rule_strategy_demo_execution_read_model.py`

**Interfaces:** Summary and facts expose wallet authority, strategy allocations, lifecycle, conditions, quantities, prices, fees, PnL status, and curve points with stable unavailable/blocking reasons.

- [ ] Add failing contract tests for allocation return rate, current batch, reserved/occupied quote, and sync/attribution status.
- [ ] Add failing trade-fact tests for reservation ID, intent/order IDs, action mapping, actual/threshold condition values, price, quantity, fees, and submission_unknown status.
- [ ] Normalize Demo fills and strategy projections into the shared `UnifiedTradeFact` contract without importing Paper rows into Demo facts.
- [ ] Add missing strategy curve and return metrics to the API response using only confirmed strategy-owned fills and valid marks.
- [ ] Run focused backend contract tests and verify no Paper/Demo cross-contamination.

### Task 4: Rebuild Web dashboard around shared wallet and strategy matrix

**Files:**
- Modify: `frontend/src/app/dashboard.tsx`
- Modify: `frontend/src/api/rule-strategy.ts`
- Modify: `frontend/src/app/dashboard-demo-execution.ts`
- Modify: `frontend/src/types/multi-strategy.ts`
- Test: `frontend/src/app/dashboard-demo-execution.test.ts`
- Test: `frontend/src/app/dashboard-refresh.test.ts`

**Interfaces:** Dashboard consumes shared account summary plus selected strategy Demo execution and curve queries. It must show loading, empty, stale, blocked, unavailable, and partial states.

- [ ] Add failing presentation tests for wallet totals, sync freshness, attribution state, and strategy allocation rows.
- [ ] Add failing presentation tests ensuring unavailable PnL is labeled unavailable rather than rendered as zero.
- [ ] Render wallet facts as the primary summary and strategy allocations as a separate matrix.
- [ ] Render environment, status, current batch, reserved/occupied funds, utilization, realized/unrealized/net PnL, return rate, and blocker reason for each strategy.
- [ ] Add a readable wallet/strategy reconciliation section and preserve existing chart components.
- [ ] Run Web tests, typecheck, lint, and production build.

### Task 5: Rebuild Web strategy detail and trade facts views

**Files:**
- Modify: `frontend/src/app/strategies/strategies.tsx`
- Modify: `frontend/src/app/trades.tsx`
- Modify: `frontend/src/app/positions.tsx`
- Modify: `frontend/src/api/rule-strategy.ts`
- Test: `frontend/src/app/trades.test.tsx` or existing trade presentation tests

**Interfaces:** Detail and trade pages use strategy/batch/environment filters and expandable facts from the normalized API contract.

- [ ] Add failing tests for lifecycle labels, condition actual/threshold rendering, price/quantity/fee rendering, and submission_unknown copy.
- [ ] Add failing tests for strategy-specific curve, PnL amount, return rate, and unavailable states.
- [ ] Implement strategy detail sections for allocation, positions, metrics, curve, sync, and attribution.
- [ ] Implement trade filters and expandable explanation rows with IDs and execution lifecycle.
- [ ] Ensure no page fetches OKX directly and no page reconstructs PnL or conditions.
- [ ] Run Web tests, typecheck, lint, and build.

### Task 6: Authenticated Web verification and deployment evidence

**Files:**
- Modify: `PROJECT_CONTEXT.md` only if durable architecture/status facts change
- Test/QA: browser verification against the running Web and backend

- [ ] Run the complete backend focused regression suite and Web quality gates.
- [ ] Rebuild/recreate backend and frontend using the standard deployment script without changing runtime secrets.
- [ ] Confirm four strategy rows, shared wallet totals, strategy matrix, detail curve, and trade explanations through authenticated browser interaction.
- [ ] Confirm Demo rows show reservation/intent/order/fill lifecycle and Paper rows remain isolated.
- [ ] Record any external OKX limitation as a blocked verification item; do not label unverified remote execution as complete.

