FROM node:22-alpine AS build

ENV BUN_INSTALL=/root/.bun \
    PATH=/root/.bun/bin:$PATH

RUN npm install -g bun@1.3.3

WORKDIR /app/frontend

COPY frontend/package.json frontend/bun.lock ./
RUN bun install --frozen-lockfile

COPY frontend ./

ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN bun run build

FROM nginx:1.27-alpine

COPY docker/frontend-nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/build/client /usr/share/nginx/html

EXPOSE 80
