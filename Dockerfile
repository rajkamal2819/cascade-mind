# Root Dockerfile — used by the Phase-2 validator (from GitHub) and
# HuggingFace Spaces (via `openenv push`).
#
# Base image: the hackathon-provided openenv-base hosted on ghcr.io
# (GitHub Container Registry), which the validator can always reach.
# DO NOT switch to docker.io images — the validator cannot pull from Docker Hub.

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# System utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy full repo
COPY . /app

# Install Python dependencies from the single source of truth
RUN pip install --no-cache-dir -r server/requirements.txt

# Install the project package itself (best-effort)
RUN pip install --no-cache-dir -e . 2>/dev/null || true

# Empty LLM cache placeholder (runtime fills via HF_TOKEN secret)
RUN echo "{}" > /app/llm_sim_cache.json

# Environment
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV LLM_SIMULATOR_ENABLED=true
ENV LLM_CACHE_PATH=/app/llm_sim_cache.json

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
    || exit 1

# Default command — openenv push may override this to enable the web interface
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
