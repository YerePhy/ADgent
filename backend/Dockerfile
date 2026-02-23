# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

WORKDIR /app

# Copy dependency manifest first (better layer caching).
COPY pyproject.toml .

# Copy only this service's source.
COPY backend/src/ backend/src/

# Create stubs so hatch can resolve the package list.
RUN mkdir -p worker/src/worker pipelines/src/pipelines frontend/src/frontend \
    && touch worker/src/worker/__init__.py pipelines/src/pipelines/__init__.py frontend/src/frontend/__init__.py

RUN pip install --no-cache-dir -e '.[backend]'

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/backend/src"]
