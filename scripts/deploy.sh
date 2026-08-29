#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="docker-compose.local.yml"
ENV_FILE="docker/runtime/.env"
BRANCH="main"
SERVICE_MODE="auto"
DRY_RUN=0
SKIP_TESTS=0
FORCE=0

usage() {
  cat <<'EOF'
Usage: scripts/deploy.sh [options]

Pull origin/main, classify the change, run the applicable gates, rebuild affected
ValueCell services, recreate only those services, and verify local/public health.

Options:
  --service auto|backend|frontend|all  Deployment scope (default: auto)
  --force                             Redeploy even when origin/main has no new commit
  --skip-tests                        Skip local test gates (emergency use only)
  --dry-run                           Pull/classify/check only; do not build or recreate
  -h, --help                          Show this help

Required environment:
  docker/runtime/.env must exist and must contain the Compose-required values.
EOF
}

log() { printf '[deploy] %s\n' "$*"; }
fatal() { printf '[deploy] ERROR: %s\n' "$*" >&2; exit 1; }
run() {
  if (( DRY_RUN )); then
    printf '[deploy] dry-run:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

while (($#)); do
  case "$1" in
    --service)
      (($# >= 2)) || fatal "--service requires auto, backend, frontend, or all"
      SERVICE_MODE="$2"
      shift 2
      ;;
    --force) FORCE=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fatal "unknown option: $1" ;;
  esac
done

case "$SERVICE_MODE" in
auto|backend|frontend|all) ;;
*) fatal "invalid --service: $SERVICE_MODE" ;;
esac

[[ -f "$ENV_FILE" ]] || fatal "missing $ENV_FILE"
[[ -f "$COMPOSE_FILE" ]] || fatal "missing $COMPOSE_FILE"
[[ "$(git branch --show-current)" == "$BRANCH" ]] || fatal "checkout must be on $BRANCH"
[[ -z "$(git status --porcelain)" ]] || fatal "worktree is dirty; commit or stash changes first"

BEFORE_SHA="$(git rev-parse HEAD)"
BEFORE_SHORT="$(git rev-parse --short HEAD)"
log "fetching origin/main from $BEFORE_SHORT"
git fetch origin main --prune
ORIGIN_SHA="$(git rev-parse origin/main)"

if [[ "$BEFORE_SHA" != "$ORIGIN_SHA" ]]; then
  git merge-base --is-ancestor HEAD origin/main || fatal "local main cannot fast-forward to origin/main"
  run git pull --ff-only origin main
fi

AFTER_SHA="$(git rev-parse HEAD)"
AFTER_SHORT="$(git rev-parse --short HEAD)"
if [[ "$BEFORE_SHA" == "$ORIGIN_SHA" && $FORCE -eq 0 ]]; then
  log "no new commits ($AFTER_SHORT); use --force to rebuild and recreate"
  curl --fail --silent --show-error http://127.0.0.1:18000/api/v1/healthz >/dev/null
  log "health check passed; nothing redeployed"
  exit 0
fi

COMPARE_SHA="$AFTER_SHA"
if (( DRY_RUN )); then
  COMPARE_SHA="$ORIGIN_SHA"
fi
mapfile -t CHANGED < <(git diff --name-only "$BEFORE_SHA" "$COMPARE_SHA")
((${#CHANGED[@]} > 0)) || [[ $FORCE -eq 1 ]] || fatal "no changed files and --force was not set"

has_match() {
  local path
  for path in "${CHANGED[@]}"; do
    [[ "$path" == $1 ]] && return 0
  done
  return 1
}

BACKEND_CHANGED=0
FRONTEND_CHANGED=0
CONFIG_CHANGED=0
for path in "${CHANGED[@]}"; do
  case "$path" in
    python/*|docker/DockerFile|docker/runtime/*) BACKEND_CHANGED=1 ;;
    frontend/*) FRONTEND_CHANGED=1 ;;
    docker-compose.local.yml|docker/frontend.Dockerfile) CONFIG_CHANGED=1 ;;
  esac
done

if [[ "$SERVICE_MODE" == backend ]]; then FRONTEND_CHANGED=0; BACKEND_CHANGED=1; fi
if [[ "$SERVICE_MODE" == frontend ]]; then BACKEND_CHANGED=0; FRONTEND_CHANGED=1; fi
if [[ "$SERVICE_MODE" == all ]]; then BACKEND_CHANGED=1; FRONTEND_CHANGED=1; fi
if (( CONFIG_CHANGED )); then BACKEND_CHANGED=1; FRONTEND_CHANGED=1; fi
if (( FORCE && ${#CHANGED[@]} == 0 )); then BACKEND_CHANGED=1; FRONTEND_CHANGED=1; fi

log "range $BEFORE_SHORT..$AFTER_SHORT"
printf '%s\n' "${CHANGED[@]:-<none>}" | sed 's/^/[deploy] changed: /'
log "scope backend=$BACKEND_CHANGED frontend=$FRONTEND_CHANGED"

if (( !SKIP_TESTS )); then
  if (( BACKEND_CHANGED )); then
    log "running backend gates"
    run uv run --project python ruff check python/valuecell/server
    run python -m compileall -q python/valuecell/server
    run uv run --project python pytest -q python/valuecell/server/tests/test_rule_strategy*.py
  fi
  if (( FRONTEND_CHANGED )); then
    log "running frontend gates"
    run bun run --cwd frontend typecheck
    run bun run --cwd frontend lint
    run bun run --cwd frontend build
  fi
else
  log "WARNING: local test gates skipped"
fi

run docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" config --quiet

if (( BACKEND_CHANGED )); then
  log "building backend"
  run docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build backend
fi
if (( FRONTEND_CHANGED )); then
  log "building frontend"
  run docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build frontend
fi

if (( !DRY_RUN )); then
  SERVICES=()
  (( BACKEND_CHANGED )) && SERVICES+=(backend)
  (( FRONTEND_CHANGED )) && SERVICES+=(frontend)
  if ((${#SERVICES[@]})); then
    log "recreating: ${SERVICES[*]}"
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --no-build --no-deps --force-recreate "${SERVICES[@]}"
  fi

  log "waiting for backend health"
  for _ in {1..60}; do
    if curl --fail --silent --show-error http://127.0.0.1:18000/api/v1/healthz >/dev/null; then break; fi
    sleep 1
  done
  curl --fail --silent --show-error http://127.0.0.1:18000/api/v1/healthz >/dev/null || fatal "backend health failed"
  curl --fail --silent --show-error -o /dev/null http://127.0.0.1:18080/ || fatal "frontend HTTP check failed"

  BACKEND_CID="$(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps -q backend)"
  [[ -n "$BACKEND_CID" ]] || fatal "backend container not found"
  docker inspect "$BACKEND_CID" --format 'backend={{.Id}} status={{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}}'
  docker logs --since 3m "$BACKEND_CID" 2>&1 | grep -Ei 'Traceback|FATAL|Application startup failed|scheduler initialization deferred' && fatal "startup error found in backend logs" || true
fi

git status --short --branch
if (( !DRY_RUN )); then
  [[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || fatal "local and origin/main differ after deploy"
fi
log "deployment completed at $AFTER_SHORT"
