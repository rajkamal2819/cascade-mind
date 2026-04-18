# Stage 1: Builder — install dependencies using uv
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

# Optional: HF token for pre-warming the LLM simulator cache at build time.
# The token is NOT stored in the final image — only the cached responses JSON is.
ARG HF_TOKEN=""

WORKDIR /app

# Install system deps needed at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv 2>/dev/null || true; \
    fi

# Copy the entire environment directory
COPY . /app/env

WORKDIR /app/env

# Install Python deps (two-pass for better layer caching)
# Pass 1: install all dependencies without the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable 2>/dev/null || \
        pip install --no-cache-dir -r server/requirements.txt; \
    else \
        pip install --no-cache-dir \
            "openenv-core>=0.2.2" \
            "fastapi>=0.115.0" \
            "uvicorn[standard]>=0.24.0" \
            "pydantic>=2.0.0" \
            "networkx>=3.0" \
            "requests>=2.31.0" \
            "httpx>=0.27.0" \
            "websockets>=12.0" \
            "openai>=1.0.0" \
            "huggingface_hub>=0.24.0" \
            "jinja2==3.1.4"; \
    fi

# Pass 2: install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable 2>/dev/null || true; \
    fi

# Pre-warm LLM simulator cache (bakes responses into image — no LLM latency at runtime)
# Only runs if HF_TOKEN is provided at build time. Cache JSON is safe to include in image.
RUN if [ -n "${HF_TOKEN}" ]; then \
        echo "Pre-warming LLM cache for seeds 0..99..." && \
        HF_TOKEN=${HF_TOKEN} \
        PYTHONPATH=/app/env \
        python -m server.preload_cache \
            --seeds 100 \
            --output /app/env/llm_sim_cache.json && \
        echo "LLM cache pre-warm complete."; \
    else \
        echo "HF_TOKEN not provided — skipping LLM cache pre-warm. Will use fallback templates."; \
        touch /app/env/llm_sim_cache.json && echo "{}" > /app/env/llm_sim_cache.json; \
    fi

# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Also copy installed system packages (pip-installed in fallback path)
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin

# Environment
ENV PATH="/app/.venv/bin:/usr/local/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# LLM simulator settings
# LLM_SIMULATOR_ENABLED: set to "false" to disable LLM and use template fallbacks
ENV LLM_SIMULATOR_ENABLED=true
# LLM_CACHE_PATH: path to the pre-warmed cache JSON baked into the image
ENV LLM_CACHE_PATH=/app/env/llm_sim_cache.json
# HF_TOKEN: set at runtime via HuggingFace Space secrets (not baked into image)
# ENV HF_TOKEN=<set-via-runtime-secret>

# Expose port (7860 = HuggingFace Spaces standard)
EXPOSE 7860

# Health check — hits the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
    || exit 1

# Start the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860 --log-level info"]
