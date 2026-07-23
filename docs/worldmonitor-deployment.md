# WorldMonitor production integration

ValueCell and WorldMonitor run as separate services in one Docker Compose deployment.
WorldMonitor owns real-time source collection and Redis cache data. ValueCell owns
the PostgreSQL research record, user access, AI calls, and strategy context.

## Server layout

Keep the repositories as siblings on the server:

```text
/opt/valuecell
/opt/worldmonitor
```

The compose file uses `WORLDMONITOR_DIR=../worldmonitor` by default. Change this
only if the WorldMonitor checkout is elsewhere.

## Configure secrets

1. Copy `docker/runtime/.env.example` to `docker/runtime/.env` if it does not
   already exist. It holds existing ValueCell runtime settings and model-provider
   credentials.
2. Copy the variable names in `docker/runtime/worldmonitor.env.example` into
   `docker/runtime/.env`, then set the values on the server.
3. Generate `RELAY_SHARED_SECRET`, `REDIS_PASSWORD`, and `REDIS_TOKEN` with
   `openssl rand -hex 32`. They must be different values.
4. Add the provider credentials already obtained for FRED, EIA, NASA FIRMS,
   ACLED, AISStream, and Finnhub. Do not add flight-provider credentials.
5. Set `WORLDMONITOR_PUBLIC_URL` only when a reverse proxy exposes the optional
   WorldMonitor dashboard at a dedicated HTTPS hostname. It does not affect the
   backend integration.

## Deploy

From `/opt/valuecell`:

```bash
docker compose --env-file docker/runtime/.env -f docker-compose.production.yml build
docker compose --env-file docker/runtime/.env -f docker-compose.production.yml up -d
docker compose --env-file docker/runtime/.env -f docker-compose.production.yml ps
```

The services expose only loopback ports by default: ValueCell on `18080`, its API
on `18000`, and the optional WorldMonitor dashboard on `13000`. Put Caddy, nginx,
or an equivalent TLS reverse proxy in front of the two web ports. Do not expose
PostgreSQL, Redis, Redis REST, AIS relay, or the ValueCell API directly to the
public internet.

## Operational checks

```bash
curl http://127.0.0.1:18000/api/v1/healthz
curl http://127.0.0.1:18000/api/v1/world-intelligence/status
docker compose --env-file docker/runtime/.env -f docker-compose.production.yml logs -f worldmonitor-seeders backend
```

The `worldmonitor-seeders` container refreshes the full upstream collection set
every 15 minutes. The ValueCell backend imports risk scores, thermal escalations,
cross-source signals, and market implications every five minutes. It retains
deduplicated source snapshots in its own PostgreSQL database.

The integration does not make trading decisions automatically. It creates the
source-attributed evidence base for the existing ValueCell model provider and
strategy workflows.
