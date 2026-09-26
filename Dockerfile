# Dependencies are synced from the lockfile into their own layer before the
# source is copied, so code changes do not invalidate the dependency cache.
# The second sync installs the project itself (it is an installable package).
# PORT is provided by Cloud Run; 8080 is the local fallback.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8080
CMD ["sh", "-c", "uvicorn chatbot.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
