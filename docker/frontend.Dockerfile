FROM oven/bun:1 AS builder

WORKDIR /app/frontend

ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY frontend/package.json frontend/bun.lock frontend/react-router.config.ts frontend/vite.config.ts frontend/tsconfig.json frontend/tsconfig.app.json frontend/tsconfig.node.json ./
RUN bun install --frozen-lockfile

COPY frontend /app/frontend
RUN bun run build

FROM nginx:1.27-alpine

COPY docker/frontend-nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/frontend/build/client /usr/share/nginx/html

EXPOSE 80
