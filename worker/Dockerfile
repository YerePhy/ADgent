# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY worker/src/ worker/src/

RUN mkdir -p backend/src/backend pipelines/src/pipelines frontend/src/frontend \
    && touch backend/src/backend/__init__.py pipelines/src/pipelines/__init__.py frontend/src/frontend/__init__.py

RUN pip install --no-cache-dir '.[worker]'

CMD ["python", "-m", "worker"]
