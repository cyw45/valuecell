# Strategy-history Excel export deployment

This guide deploys the authenticated strategy-history Excel export so that the server,
PC client, and mobile client use the same behavior as local development.

## Export contract

The backend exposes one tenant-scoped attachment endpoint:

```text
GET /api/v1/rule-strategies/{strategy_id}/export
```

Optional UTC calendar-date filters:

```text
from_date=YYYY-MM-DD
to_date=YYYY-MM-DD
```

- Both boundaries are inclusive.
- A single supplied date exports that day.
- Omitting both dates exports all persisted history for the strategy.
- A descending range returns `422`.
- The endpoint requires the existing `Authorization: Bearer <access token>` header.
  Never place an access token in the URL, a query parameter, proxy logs, or a mobile
  deep link.

The response is an `.xlsx` attachment with these sheets:

1. `导出说明`
2. `策略参数`
3. `成交明细`
4. `资金变化`
5. `执行明细`
6. `资金费`

Strategy parameters are always included. Journal-derived sheets respect the selected
range. API keys, secrets, passphrases, passwords, tokens, credentials, and signatures
are redacted before workbook generation.

## Backend deployment

The XLSX writer uses the Python standard library. No new Python package or database
migration is required.

1. Deploy the backend code and its existing dependency lockfile.
2. Keep `VALUECELL_DATABASE_URL` pointed at the same PostgreSQL database that holds
   `rule_strategies` and `rule_strategy_evaluation_journal`. Do not point a production
   API at a local/empty database.
3. Keep the production JWT configuration stable across every backend replica:

   ```env
   VALUECELL_JWT_SECRET=<stable-production-secret>
   VALUECELL_JWT_ISSUER=valuecell-saas
   ```

   Changing the JWT secret invalidates all existing sessions.
4. Build and restart the existing production service using its normal deployment
   command. For the repository Docker deployment:

   ```bash
   docker compose --env-file docker/runtime/.env \
     -f docker-compose.production.yml build backend
   docker compose --env-file docker/runtime/.env \
     -f docker-compose.production.yml up -d backend
   ```

5. Confirm health before client rollout:

   ```bash
   curl --fail http://127.0.0.1:18000/api/v1/healthz
   ```

## Reverse proxy and CORS

The proxy must forward `GET` requests, the `Authorization` request header, and these
response headers unchanged:

```text
Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
Content-Disposition: attachment; filename="strategy-history.xlsx"
```

Do not transform an XLSX response into JSON, cache it in a shared cache, or log its
body. Size proxy response buffers and upstream timeouts for the largest permitted
historical range; the workbook is generated from persisted journal history and can be
materially larger than a normal API response.

For PC browser development or a separately hosted PC frontend, configure the exact
origins required by the browser:

```env
CORS_ORIGINS=https://app.example.com,http://localhost:5173,http://127.0.0.1:8081
```

Use only the real origins for production. Native Android/iOS requests do not rely on
browser CORS, but still require the public HTTPS API endpoint.

## PC client release

Deploy the PC/frontend bundle after the backend route is reachable:

```bash
cd frontend
bun install --frozen-lockfile
bun run lint
bun run typecheck
bun run build
```

Verify the strategy dashboard/configuration export panel with an account that has
strategy-read permission. Test all-history, one-day, and date-range exports.

## Mobile release

The mobile export flow downloads the authenticated workbook into the cache and opens
the native system share sheet. It uses `expo-file-system` and `expo-sharing`; the
Expo config includes the `expo-sharing` plugin. A new APK/IPA is required—an over-the-
air JavaScript update alone is not sufficient for a binary that lacks these modules.

```bash
cd mobile
bun install --frozen-lockfile
bun run typecheck
bun run export:android
eas build --platform android --profile preview
```

Set only the public API base URL in the mobile environment:

```env
EXPO_PUBLIC_API_BASE_URL=https://api.example.com/api/v1
```

Do not add database URLs, JWT secrets, exchange credentials, or provider keys to the
mobile environment.

## Release verification

Use a strategy belonging to the test tenant; do not test by guessing another tenant's
strategy ID.

```bash
curl --fail --location \
  --header "Authorization: Bearer <short-lived-test-token>" \
  --output strategy-history.xlsx \
  "https://api.example.com/api/v1/rule-strategies/<strategy-id>/export?from_date=2026-01-10&to_date=2026-01-12"
unzip -t strategy-history.xlsx
```

Confirm the six named sheets exist, the selected UTC dates are inclusive, no rows from
another tenant appear, and exported parameters contain no secret values. Then verify
that PC downloads the attachment and Android/iOS opens the system share sheet.
