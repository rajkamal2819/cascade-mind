# Cascade-Mind — Project Context

> **Read this first before making any changes to the codebase.**

## What This Project Is

**Cascade-Mind** (v0.2.0) is an OpenEnv reinforcement-learning environment for the **Meta × PyTorch OpenEnv Hackathon × Scaler School of Technology**. An AI agent plays an on-call SRE engineer: a microservice has a breaking change, and the agent must trace the full blast radius across a **30-service dependency graph** using only **noisy, LLM-generated observations** (never raw graph data). Scored with F-beta (β=2) — recall weighted 4× over precision.

- **Author:** Rajkamal2819 (solo entry)
- **License:** Apache-2.0
- **Python:** ≥ 3.10
- **Live:** https://rajkamal2819-cascade-mind.hf.space
- **GitHub:** https://github.com/rajkamal2819/cascade-mind
- **HF Space:** https://huggingface.co/spaces/Rajkamal2819/cascade-mind

## Architecture — 3 Layers

```
Agent (inference.py)          →  GPT-4o-mini via OpenAI API
    ↕ WebSocket
LLM Simulator (llm_simulator.py)  →  Llama-3.1-8B via Cerebras
    ↕ queries
Ground Truth (graph_builder.py)   →  30-node networkx DiGraph
```

**Option A:** The agent NEVER sees raw graph data — only LLM-generated noisy text. Ground truth is revealed only on `submit`.

## File Map

| File | Role | Key Exports |
|------|------|-------------|
| `server/service_impact_environment.py` | **Core environment** — reset/step/submit, F-beta scorer | `ServiceImpactEnvironment` |
| `server/graph_builder.py` | **Dependency graph** — 30 services, seed perturbation, scenario picker | `build_service_graph()`, `get_affected_services()`, `get_scenario()` |
| `server/llm_simulator.py` | **LLM observation generator** — Llama-3.1-8B via Cerebras, 2-layer cache | `LLMSimulator`, `SimulatorCache` |
| `server/app.py` | **FastAPI server** — REST + WebSocket + MCP endpoints, Swagger UI | `app`, `create_app()` |
| `server/preload_cache.py` | **Cache pre-warmer** — bakes LLM responses for seeds 0–99 at build time | CLI script |
| `inference.py` | **Baseline agent** — sync OpenAI client, 4-phase SRE strategy, structured output | `run_episode()`, `main()` |
| `client.py` | **WebSocket client** — implements OpenEnv `EnvClient`, has `.sync()` method | `ServiceImpactEnv` |
| `models.py` | **Pydantic data models** — action/observation/state schemas | `ServiceImpactAction`, `ServiceImpactObservation`, `ServiceImpactState` |
| `openenv.yaml` | **OpenEnv manifest** — declares env name, runtime, port | — |
| `Dockerfile` | **Root Docker** — `ghcr.io/meta-pytorch/openenv-base`, port 7860 | — |
| `server/Dockerfile` | **Server Docker** — multi-stage with uv, pre-warms cache | — |
| `test_smoke.py` | **45 smoke tests** — all core logic, runs without LLM | — |
| `test_llm.py` | **Live LLM integration test** — needs HF_TOKEN | — |

## Critical Rules — DO NOT Break These

### 1. Score Clamping (Validator Requirement)
All task scores MUST be strictly in the **open interval (0, 1)** — never exactly `0.0` or `1.0`. The hackathon Phase-2 validator rejects scores at boundaries.
- `_handle_submit()` clamps to `max(0.001, min(0.999, fbeta))`
- `inference.py` clamps at `[END]` output: `max(0.001, min(0.999, raw_reward))`

### 2. Sync OpenAI Client (Checklist Requirement)
`inference.py` MUST use `from openai import OpenAI` (sync), NOT `AsyncOpenAI`. The validator checks this import literally.

### 3. Structured Output Markers (Validator Requirement)
`inference.py` must print these with `flush=True`:
```
[START] task=<name>
[STEP] step=<N> reward=<X>
[END] task=<name> score=<X> steps=<N>
```

### 4. Docker Base Image
Root `Dockerfile` MUST use `ghcr.io/meta-pytorch/openenv-base:latest` (GitHub Container Registry). The validator CANNOT reach Docker Hub (`docker.io`), so never use `python:*-slim` or similar.

### 5. Environment Variables Contract
```python
HF_TOKEN     = os.getenv("HF_TOKEN")                                          # REQUIRED, no default
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")         # has default
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")                       # has default
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://rajkamal2819-cascade-mind.hf.space")
```
`HF_TOKEN` must NEVER have a default or fallback. The validator injects it at runtime.

### 6. Option A — No Graph Leaks
`result=[]` on ALL query steps. Only `submit` populates `result` with ground truth. Never expose raw graph adjacency in observations.

### 7. Jinja2 Pin
`jinja2==3.1.4` is pinned everywhere (pyproject.toml, requirements.txt). openenv-core 0.2.x breaks with newer versions.

## Episode Lifecycle

```
reset(seed) → build graph → pick scenario (easy/medium/hard) → LLM incident alert
    ↓
step(query_dependents/query_dependencies) → noisy LLM registry output, costs 1 budget
step(query_runbook/query_changelog/query_monitoring) → FREE, no budget cost
    ↓
step(submit, affected_services=[...]) → F-beta(β=2) score, episode ends
```

## Reward Design

| Event | Reward |
|-------|--------|
| New service queried | `+0.05` |
| Re-query same service | `−0.05` |
| Free action (runbook/changelog/monitoring) | `None` |
| Budget exhaustion (no submit) | `−0.4` |
| Submit | F-beta(β=2) ∈ (0.001, 0.999) |
| Overclaiming (>60% services submitted) | Penalty applied |

## Noise Model by Difficulty

| Difficulty | Noise | Hallucinated Services | Dropped Services | Budget |
|------------|-------|-----------------------|------------------|--------|
| Easy | 15% | 0 | 0 | 13–15 |
| Medium | 40% | 1 | 1 | 10–12 |
| Hard | 70% | 2 | 2 | 9–10 |

## Tech Stack

- **Framework:** OpenEnv (`openenv-core ≥ 0.2.2`)
- **Server:** FastAPI + Uvicorn, port 7860
- **Graph:** NetworkX (30-node DiGraph, seed-perturbed)
- **LLM (observations):** Llama-3.1-8B via Cerebras HF Inference Providers
- **LLM (agent):** GPT-4o-mini via OpenAI API (configurable)
- **Models:** Pydantic v2
- **Caching:** 2-layer (memory dict + JSON file), thread-safe `threading.Lock`
- **Deploy:** HuggingFace Spaces (Docker SDK), `ghcr.io/meta-pytorch/openenv-base`
- **Protocol:** REST + WebSocket + MCP (JSON-RPC 2.0)

## Deployment

```bash
# Push to HF Space
HF_TOKEN=hf_<write_token> openenv push --repo-id Rajkamal2819/cascade-mind

# Space secrets (set via HF UI, not in code):
#   HF_TOKEN = hf_<read_token>
#   LLM_SIMULATOR_ENABLED = true
#   LLM_CACHE_PATH = /app/llm_sim_cache.json

# Verify
curl https://rajkamal2819-cascade-mind.hf.space/health
# → {"status":"healthy"}
```

## Testing

```bash
# Smoke tests (no LLM needed)
LLM_SIMULATOR_ENABLED=false python test_smoke.py

# Live LLM test (needs HF_TOKEN)
HF_TOKEN=hf_... python test_llm.py

# Local server
uvicorn server.app:app --port 8888
```

## Common Pitfalls (Learned the Hard Way)

1. **Never use `AsyncOpenAI`** in inference.py — validator checks literal import
2. **Never return score 0.0 or 1.0** — validator rejects boundary values
3. **Never use Docker Hub images** in root Dockerfile — validator can't reach `docker.io`
4. **Never give `HF_TOKEN` a default** — validator expects it unset until injected
5. **Always `flush=True`** on structured output prints — validator reads stdout in real-time
6. **`SPACE_ID` is a reserved HF variable** — never set it as a secret
7. **`openenv push` modifies Dockerfile** — it enables the web interface; check git diff after
8. **The `.sync()` method** on `ServiceImpactEnv` returns a sync context manager — use `with env.sync() as env:` not `async with`
