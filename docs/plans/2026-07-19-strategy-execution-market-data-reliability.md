# Tenant-Scoped Strategy Continuity and Market Data Reliability Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make a same-account, same-workspace login on any browser reliably discover running strategies; make paper and OKX Demo execution state explicit and non-misleading; and make K-line retrieval observable and resilient through retries, cooldowns, and multi-provider fallback.

**Architecture:** Keep tenant isolation authoritative on the backend. Replace the browser-only active-strategy assumption with a tenant-aware server list and deterministic selection. Preserve the current guarded OKX Demo boundary, but distinguish paper execution from exchange submission/settlement rather than treating a submission as a paper fill. Enhance existing market-provider fallback with structured failures, error-aware retry/cooldown, and scheduler diagnostics.

**Tech Stack:** FastAPI, SQLAlchemy/PostgreSQL, React/TypeScript/TanStack Query, APScheduler, pytest, Docker Compose.

---

### Task 1: Cross-browser tenant strategy discovery

**Files:**
- Modify: `frontend/src/api/rule-strategy.ts`
- Modify: `frontend/src/hooks/use-active-rule-strategy.ts`
- Modify: `frontend/src/app/strategies/strategies.tsx`
- Modify: `frontend/src/app/dashboard.tsx`
- Modify: `frontend/src/app/trades.tsx`
- Modify: `frontend/src/app/funding.tsx`
- Modify: `frontend/src/app/strategies/advisory.tsx`
- Test: existing frontend test convention or new focused tests

**Behavior:** Fetch tenant-scoped `GET /api/v1/rule-strategies`; show a strategy picker/list; on a fresh browser automatically select the newest running strategy. Store local selection scoped by user and tenant; never directly read a global strategy localStorage key.

**Verification:** Same account + same tenant in a fresh browser sees all running strategies and may select one; tenant switch cannot reuse an old tenant ID.

### Task 2: Session and workspace boundary correctness

**Files:**
- Modify: `frontend/src/lib/api-client.ts`
- Modify: `frontend/src/store/system-store.ts`
- Modify as needed: SaaS auth API/hooks
- Test: frontend test convention

**Behavior:** Request headers are isolated per request; tenant-bound query keys include tenant identity; logout/workspace changes invalidate relevant selection/cache. SaaS 401 behavior must not invoke the legacy refresh route for SaaS sessions.

**Verification:** No stale bearer header leaks after logout; stale active strategy is not queried under a different tenant.

### Task 3: Explicit execution lifecycle and Demo order attribution

**Files:**
- Modify: `python/valuecell/server/db/models/sandbox_exchange_order.py`
- Modify: migrations/init schema pattern used by this repo
- Modify: `python/valuecell/server/services/sandbox_exchange_trading_service.py`
- Modify: `python/valuecell/server/services/strategy_scheduler.py`
- Modify: `python/valuecell/server/services/rule_strategy_service.py`
- Modify: relevant API schemas/routes and frontend data consumers
- Test: `python/valuecell/server/tests/test_sandbox_exchange_trading.py`, `test_strategy_scheduler.py`, new execution lifecycle tests

**Behavior:** Attribute Demo orders to strategy and evaluation. Do not settle a paper ledger fill for Demo until a confirmed exchange fill. Refresh and reconcile non-terminal orders; preserve paper and Demo account sources separately. A stop/switch must fence stale running ticks before order routing.

**Verification:** A rejected Demo order creates no synthetic strategy position; a confirmed fill is attributed to the originating strategy; switching execution target preserves rule configuration but does not mix paper and Demo balances/PnL.

### Task 4: K-line fetch reliability and scheduler observability

**Files:**
- Modify: `python/valuecell/server/services/crypto_market_service.py`
- Modify: `python/valuecell/server/services/strategy_scheduler.py`
- Modify: `python/valuecell/server/config/settings.py`
- Test: `python/valuecell/server/tests/test_crypto_market*.py`, `test_strategy_scheduler.py`

**Behavior:** Classify provider failures; use bounded retry with jitter for retryable failures; observe provider cooldowns and fall back across configured sources; retain partial successes; log/record explicit market-data unavailable reasons rather than generic empty candles. Apply provider health/cooldown to historical requests too, and use bounded provider concurrency.

**Verification:** Simulated OKX 403/timeout retries and falls back to next provider; one failed interval does not abort independent interval fetches; scheduler records reason and never trades on missing/stale candles beyond its configured freshness rule.

### Task 5: Regression, deployment and delivery

**Files:** only required docs/config tests

**Verification:** Run focused backend tests, frontend typecheck/build, Docker rebuild/recreate, HTTP health and strategy endpoints, container restart count, `git diff --check`, commit and push to `origin/main`.
