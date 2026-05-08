# ValueCell Ubuntu VPS Deployment

This deployment layout is intended for a Linux VPS where:

- the frontend is served as a static SPA container,
- the backend runs as a separate Python container,
- the SQLAlchemy database uses PostgreSQL,
- ValueCell local runtime state is persisted under `docker/runtime`.

## 1. Important Notes

### PostgreSQL support

The backend now supports PostgreSQL through `psycopg`.

Use a DSN like:

```text
postgresql+psycopg://USER:PASSWORD@HOST:5432/DB_NAME
```

### Runtime persistence behavior

When `VALUECELL_DATABASE_URL` is a PostgreSQL DSN:

- SQLAlchemy business tables use PostgreSQL
- conversation runtime store uses PostgreSQL
- conversation items use PostgreSQL
- task runtime store uses PostgreSQL

The following data still remains local-file based and should be persisted via volume mount:

- runtime `.env`
- LanceDB
- local knowledge cache

That is why `docker/runtime` is still mounted to `/root/.config/valuecell`.

## 2. Prepare Runtime Config

Copy the example runtime env:

```bash
cd /opt/valuecell
cp docker/runtime/.env.example docker/runtime/.env
```

Edit `docker/runtime/.env` and set at least:

- `VALUECELL_DATABASE_URL`
- `CORS_ORIGINS`
- `PRIMARY_PROVIDER`
- your model provider API keys

The backend now loads `docker/runtime/.env` first. This means:

- local development can use the same runtime env as VPS
- after upload to Ubuntu, you usually do not need to change code or rebuild config paths

## 3. Build and Start

Run:

```bash
docker compose -f docker/docker-compose.vps.yml up -d --build
```

This starts:

- backend on `127.0.0.1:18000`
- frontend on `127.0.0.1:18080`

They are intentionally bound to loopback so your host Nginx can proxy them safely.

## 4. Nginx Reverse Proxy

Use the host-level Nginx config template:

- `docker/nginx.valuecell-subdomain.conf.example`

Recommended routing:

- `https://valuecell.example.com/` -> frontend container
- `https://valuecell.example.com/api/v1/` -> backend container

Because the frontend is built with `VITE_API_BASE_URL=/api/v1`, no extra frontend rebuild is needed per domain.

## 5. Health Checks

Backend health endpoint:

```text
/api/v1/healthz
```

Frontend container health endpoint:

```text
/healthz
```

## 6. Upgrade Flow

```bash
git pull
docker compose -f docker/docker-compose.vps.yml up -d --build
```

Your runtime state remains in:

- `docker/runtime/.env`
- `docker/runtime/valuecell.db`
- `docker/runtime/lancedb`
- `docker/runtime/.knowledge`

## 7. Security Recommendation

Do not hardcode real database passwords into the repository.

If a database password has already been shared in chat or shell history, rotate it before going live.
