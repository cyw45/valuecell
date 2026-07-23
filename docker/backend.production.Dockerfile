FROM ghcr.io/astral-sh/uv:0.7.8 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/python/.venv/bin:$PATH" \
    PYTHONPATH=/app/python

RUN useradd --create-home --uid 10001 valuecell

WORKDIR /app/python

COPY python/pyproject.toml python/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY python/valuecell ./valuecell
COPY python/configs ./configs
COPY python/apscheduler ./apscheduler

RUN chown -R valuecell:valuecell /app

USER valuecell

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "valuecell.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
