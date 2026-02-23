# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY frontend/src/ frontend/src/

RUN mkdir -p backend/src/backend worker/src/worker pipelines/src/pipelines \
    && touch backend/src/backend/__init__.py worker/src/worker/__init__.py pipelines/src/pipelines/__init__.py

RUN pip install --no-cache-dir '.[frontend]'

CMD ["python", "-m", "frontend.app"]
