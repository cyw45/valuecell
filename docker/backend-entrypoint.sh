#!/bin/sh
set -eu

SYSTEM_DIR="/root/.config/valuecell"
mkdir -p "${SYSTEM_DIR}"

if [ ! -f "${SYSTEM_DIR}/.env" ] && [ -f /app/.env.example ]; then
  cp /app/.env.example "${SYSTEM_DIR}/.env"
fi

cd /app/python

uv run python -m valuecell.server.db.init_db

exec uv run uvicorn valuecell.server.main:app \
  --host 0.0.0.0 \
  --port "${API_PORT:-8000}"
