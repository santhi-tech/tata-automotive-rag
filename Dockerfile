FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /bin/uv

WORKDIR /app

# Dependencies first (cached layer), from the lock file
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src

# Chromium + its system libraries for the Playwright loaders
RUN playwright install --with-deps chromium

COPY src/ /app/src/

# Create non-root user for security
RUN useradd -m -u 1000 tatauser && chown -R tatauser:tatauser /app
USER tatauser

# Default command
CMD ["python", "-m", "ingestion.pipeline"]
