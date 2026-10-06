FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dependencies first (cached layer), from the lock file
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src

# Chromium + its system libraries for the Playwright loaders
RUN playwright install --with-deps chromium

COPY src/ /app/src/

# Default command
CMD ["python", "-m", "ingestion.pipeline"]
