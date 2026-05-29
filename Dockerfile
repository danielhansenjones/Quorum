# Dockerfile for the api service. Built by docker compose under the `api` profile.
# Phase 1 doesn't build this; first build lands in Phase 9 when the FastAPI surface
# is ready. Keeping the file present so `docker compose config` validates.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# Minimal apt for lxml/psycopg/torch wheel sanity. No build toolchain - we rely
# on prebuilt wheels for everything heavy.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# uv from the official image. Pinned digest can be added later for reproducibility.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Lockfile-first install; src/ copied after so a code-only change reuses the layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "quorum.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
