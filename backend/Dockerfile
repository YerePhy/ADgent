# syntax=docker/dockerfile:1.6
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra providers

COPY config.yaml .
COPY logging.yaml .
COPY registry.json .
COPY backend/ backend/

CMD ["uv", "run", "python", "-m", "backend.app"]
