# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY backend/ backend/

RUN pip install --no-cache-dir -e '.'

CMD ["python", "-m", "backend.main"]
