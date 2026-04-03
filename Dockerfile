# Root Dockerfile — used by HuggingFace Spaces Docker deployment
# Build context is the repo root (all source files available).
# Delegates to the same logic as server/Dockerfile.

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

# HF_TOKEN passed at build time to pre-warm LLM cache (not stored in final image)
ARG HF_TOKEN=""

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy full repo into /app
COPY . /app

# Install from requirements file (keeps dependency list in one place)
RUN pip install --no-cache-dir -r server/requirements.txt

# Install the project itself (for package imports)
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir -e . 2>/dev/null || true

# Pre-warm LLM cache at build time if HF_TOKEN provided
RUN if [ -n "${HF_TOKEN}" ]; then \
        echo "Pre-warming LLM cache for seeds 0..99..." && \
        HF_TOKEN=${HF_TOKEN} \
        PYTHONPATH=/app \
        python -m server.preload_cache \
            --seeds 100 \
            --output /app/llm_sim_cache.json && \
        echo "LLM cache pre-warm complete."; \
    else \
        echo "HF_TOKEN not provided — skipping cache pre-warm."; \
        echo "{}" > /app/llm_sim_cache.json; \
    fi

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV LLM_SIMULATOR_ENABLED=true
ENV LLM_CACHE_PATH=/app/llm_sim_cache.json

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
    || exit 1

# HuggingFace Spaces uses port 7860 by default
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
