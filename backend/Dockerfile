# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e '.[providers]'

COPY config.yaml .
COPY backend/ backend/

CMD ["python", "-m", "backend.app"]
