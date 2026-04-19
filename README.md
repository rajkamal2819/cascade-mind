---
title: Cascade Mind
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - sre
  - microservices
  - llm
  - reinforcement-learning
  - incident-response
  - networkx
base_path: /web
---

<div align="center">

# 🧠 cascade-mind

### AI-Powered Service Impact Analysis for On-Call SREs

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Rajkamal2819/cascade-mind)
[![API Docs](https://img.shields.io/badge/FastAPI-Docs-009688?logo=fastapi)](https://rajkamal2819-cascade-mind.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-7c3aed)](https://github.com/openenv/openenv)

**cascade-mind** is an [OpenEnv](https://github.com/openenv/openenv)-compatible reinforcement learning environment where an AI agent acts as an on-call SRE engineer. A microservice just had a breaking change — the agent must trace the blast radius across a 30-service dependency graph using the same noisy, LLM-generated tooling a real engineer would see.

[**🚀 Live Playground**](https://rajkamal2819-cascade-mind.hf.space) · [**📖 API Docs**](https://rajkamal2819-cascade-mind.hf.space/docs) · [**🗺️ Ground Truth Graph**](https://rajkamal2819-cascade-mind.hf.space/graph/ground-truth?seed=0&difficulty=easy)

</div>

---

## What Makes This Hard

Every observation — incident alerts, registry lookups, runbooks, monitoring snapshots — is generated live by **Llama-3.1-8B-Instruct via Cerebras**. The ground truth is a seeded `networkx` dependency graph that the agent **never sees directly**. It must triangulate the true blast radius from contradictory, noisy sources.

> **First OpenEnv environment to use an LLM as the world model.** Scripted observations can be memorized. Ours can't.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GROUND TRUTH LAYER                          │
│   networkx DiGraph · 30 services · seed-perturbed topology          │
│   55–80 edges · 10,000+ unique episodes per difficulty              │
│   nx.ancestors(G, service) → exact affected set  (scorer only)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  structured data — never exposed to agent
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LLM SIMULATOR LAYER                           │
│   Llama-3.1-8B-Instruct  ·  Cerebras  ·  HuggingFace Providers     │
│                                                                      │
│   reset()        →  PagerDuty-style P1 incident alert               │
│   query_*        →  Noisy registry / runbook / monitoring output     │
│   query_changelog →  PR / CHANGELOG entry for the changed service   │
│                                                                      │
│   Noise:  easy 0.15  ·  medium 0.40  ·  hard 0.70                  │
│   Cache:  2-layer  (in-memory MD5 + JSON disk, pre-warmed in CI)    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  LLM text — noisy, sometimes wrong
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           AGENT LAYER                               │
│   Sees only:  observation.message  (free-text LLM output)           │
│   result[] is always [] during queries  (must parse message)        │
│   Budget: 5 / 8 / 12 queries  ·  Goal: submit correct affected set  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### ▶ Interactive Playground

1. Open the [Live Space](https://rajkamal2819-cascade-mind.hf.space)
2. Click **Reset** — you'll receive a PagerDuty-style P1 alert
3. Select an **Action Type** and enter a **Service Name**, then click **Step**
4. When confident, choose `submit` and enter your affected services list
5. Your **F-beta(β=2) score** appears instantly

---

### 🐍 Python Client (WebSocket)

```python
import asyncio, json, websockets

async def run():
    async with websockets.connect("wss://rajkamal2819-cascade-mind.hf.space/ws") as ws:
        # 1. Start a new episode
        await ws.send(json.dumps({"type": "reset", "data": {"seed": 0}}))
        obs = json.loads(await ws.recv())
        print(obs["data"]["message"])           # P1 incident alert

        # 2. Free action — read the changelog
        await ws.send(json.dumps({"type": "step", "data": {
            "action_type": "query_changelog", "service_name": "catalog_service"
        }}))
        print(json.loads(await ws.recv())["data"]["message"])

        # 3. Budgeted query — trace dependents
        await ws.send(json.dumps({"type": "step", "data": {
            "action_type": "query_dependents", "service_name": "catalog_service"
        }}))
        print(json.loads(await ws.recv())["data"]["message"])

        # 4. Submit final answer
        await ws.send(json.dumps({"type": "step", "data": {
            "action_type": "submit",
            "affected_services": ["api_gateway", "cart_service", "web_backend"]
        }}))
        result = json.loads(await ws.recv())
        print("Score:", result["data"]["reward"])   # float ∈ [0.0, 1.0]

asyncio.run(run())
```

### Typed Client

```python
from cascade_mind import ServiceImpactEnv

async with ServiceImpactEnv(base_url="https://rajkamal2819-cascade-mind.hf.space") as env:
    obs, _ = await env.reset(seed=0)
    print(obs.message)   # incident alert text
```

---

### 💻 Run Locally

```bash
git clone https://github.com/rajkamal2819/cascade-mind
cd cascade-mind
pip install -e ".[dev]"

export HF_TOKEN=hf_your_token_here      # Cerebras access for Llama-3.1-8B

# With LLM observations (recommended)
LLM_SIMULATOR_ENABLED=true uvicorn cascade_mind.server.app:app --port 8000

# Fully offline — template fallbacks only
LLM_SIMULATOR_ENABLED=false uvicorn cascade_mind.server.app:app --port 8000
```

### Run Tests

```bash
# Offline unit tests (no LLM calls)
LLM_SIMULATOR_ENABLED=false pytest tests/ -v

# Live LLM integration test
HF_TOKEN=hf_... pytest tests/test_llm.py -v
```

---

### 🐳 Docker

```bash
# Build — pre-warms LLM cache for seeds 0–99 at build time
docker build \
  --build-arg HF_TOKEN=hf_your_token_here \
  -t cascade-mind .

# Run
docker run -p 8000:8000 \
  -e HF_TOKEN=hf_your_token_here \
  -e LLM_SIMULATOR_ENABLED=true \
  cascade-mind
```

The image bakes 600 LLM responses into `/tmp/llm_sim_cache.json` during build, giving **zero-latency** episode starts for seeds 0–99.

---

## Action Space

| Action | Cost | Description |
|---|:---:|---|
| `query_dependents` | −1 | "Which services call `X`?" → noisy registry CLI output |
| `query_dependencies` | −1 | "What does `X` depend on?" → noisy registry CLI output |
| `query_runbook` | free | Confluence-style runbook: ownership, SLOs, failure modes |
| `query_changelog` | free | Recent PR / deployment changelog for `X` |
| `query_monitoring` | free | Datadog-style snapshot: latency, error rate, dependency health |
| `query_service_health` | free | Real-time health status for `X` |
| `query_topology_diff` | free | Recent topology changes detected by the service mesh |
| `submit_hypothesis` | −1 | Test a partial hypothesis mid-episode |
| `submit` | terminal | Submit the final predicted affected set — ends the episode |

Budget by difficulty: **easy = 5 · medium = 8 · hard = 12**

### Action Schema

```json
{ "action_type": "query_dependents",  "service_name": "payment_service" }
{ "action_type": "query_runbook",     "service_name": "auth_service" }
{ "action_type": "submit",            "affected_services": ["cart_service", "order_service"] }
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `changed_service` | `str` | The service with a breaking change this episode |
| `message` | `str` | LLM-generated tool output (free text) |
| `result` | `List[str]` | Always `[]` during queries; populated on `submit` |
| `queries_remaining` | `int` | Budget remaining |
| `done` | `bool` | `true` after a terminal action |
| `reward` | `float \| None` | `None` during episode; F-beta score on `submit` |

**Example reset observation:**
```
[PagerDuty] INCIDENT INC-7841 | P1 | TRIGGERED
Service: catalog_service
Alert: Breaking API change detected — downstream consumers may be impacted.
Error rate elevated above SLO threshold.
Investigate all services that depend on catalog_service.

Recent changelog:
PR #142: Breaking change to catalog_service — downstream consumers will need
to update their integration.

Budget: 8 queries remaining.
```

---

## Reward Design

$$F_\beta(\beta=2) = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R} \quad \text{where } \beta = 2$$

β=2 weights **recall 4× over precision** — missing an affected service (silent outage propagation) is far worse than a false alarm.

| Condition | Modifier |
|---|:---:|
| Each new unique service queried | +0.05 |
| Re-querying the same service | −0.05 |
| Budget exhausted without submit | −0.40 |
| Overclaiming (> 60% of all 30 services) | −0.3 × oversubmit fraction |
| Terminal `submit` | F-beta(β=2) ∈ [0.0, 1.0] |

---

## Episode Variety

Each seed produces a unique microservice topology through deterministic graph perturbation:

- **Remove 4–9 redundant edges** (edges with structurally safe alternative paths)
- **Add 3–7 new edges** sampled from 27 plausible candidate dependencies

The `changed_service` is selected by **betweenness-centrality**, binned into difficulty tiers:

| Difficulty | Blast radius | Query budget | Registry noise |
|---|:---:|:---:|:---:|
| Easy | 1–6 services | 5 | 0.15 |
| Medium | 6–13 services | 8 | 0.40 |
| Hard | 13+ services | 12 | 0.70 |

`seed % 3` cycles difficulty, ensuring balanced evaluation across seeds.

---

## LLM Noise Model

| Difficulty | Hallucinated services | Dropped services |
|---|:---:|:---:|
| Easy | 0 | 0 |
| Medium | 1 | 1 |
| Hard | 2 | 2 |

At **Hard** difficulty, a registry query for a service with 8 real dependents may return 9 results — 7 correct + 2 hallucinated — while silently dropping 2 real ones. The agent must triangulate across runbooks, monitoring, and query directions.

---

## Microservice Graph

30 services across 5 tiers:

```
Tier 1 — Gateway:    api_gateway · mobile_backend · web_backend
Tier 2 — Business:   auth_service · user_service · order_service · cart_service
                     checkout_service · payment_service · billing_service
                     subscription_service · inventory_service · shipping_service
                     catalog_service · search_service · recommendation_service
                     review_service · notification_service
Tier 3 — Support:    email_service · sms_service · media_service · analytics_service
                     logging_service · audit_service · config_service · cache_service
Tier 4 — Data/Infra: database_service · message_queue · feature_flags · rate_limiter
```

---

## MCP Integration

The server exposes a **Model Context Protocol** endpoint at `/mcp`:

```bash
# List available tools
GET /mcp

# Call a tool (JSON-RPC 2.0)
POST /mcp
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "query_dependents",
    "arguments": { "service_name": "payment_service" }
  },
  "id": 1
}
```

Available MCP tools: `query_dependents` · `query_dependencies` · `query_runbook` · `query_changelog` · `query_monitoring` · `submit_impact_assessment`

---

## Baseline Agent

Run the included LLM agent across all three difficulty levels:

**Against the live HuggingFace Space:**

```bash
export SPACE_ID="Rajkamal2819/cascade-mind"
export HF_TOKEN=hf_...
export API_BASE_URL=https://api.cerebras.ai/v1
export MODEL_NAME=llama-3.3-70b

python scripts/inference.py
```

**Against a local server:**

```bash
# Terminal 1
export HF_TOKEN=hf_...
export LLM_SIMULATOR_ENABLED=true
uvicorn cascade_mind.server.app:app --port 8000

# Terminal 2
export ENV_BASE_URL=http://localhost:8000
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python scripts/inference.py
```

Output includes per-task F-beta scores and a `JSON_RESULTS: {"mean_f1": ...}` summary line for automated harnesses.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status": "healthy"}` |
| `GET` | `/schema` | JSON schemas for action, observation, state |
| `GET` | `/metadata` | Environment name, version, description |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute an action |
| `GET` | `/state` | Current episode state |
| `WS` | `/ws` | WebSocket session (primary protocol) |
| `GET` | `/mcp` | MCP tools manifest |
| `POST` | `/mcp` | MCP JSON-RPC 2.0 tool calls |
| `GET` | `/graph/ground-truth` | Interactive vis.js ground-truth graph |
| `GET` | `/docs` | Swagger UI |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | `""` | HuggingFace token for Cerebras/Llama-3.1-8B |
| `LLM_SIMULATOR_ENABLED` | `true` | `false` for fully offline template mode |
| `LLM_CACHE_PATH` | `/tmp/llm_sim_cache.json` | Path for LLM response cache |
| `OPENAI_API_KEY` | — | For the baseline inference agent |

---

## Repository Structure

```
cascade-mind/
├── cascade_mind/
│   ├── __init__.py                    # Public API re-exports
│   ├── models.py                      # Pydantic Action / Observation / State models
│   ├── client.py                      # Typed WebSocket client (ServiceImpactEnv)
│   └── server/
│       ├── app.py                     # FastAPI app · routes · MCP endpoint
│       ├── env/
│       │   ├── service_impact_environment.py   # Core reset() / step() logic
│       │   └── curriculum_scheduler.py         # Per-difficulty config
│       ├── graph/
│       │   ├── graph_builder.py       # 30-service seed-perturbed NetworkX DiGraph
│       │   └── mutation_engine.py     # Mid-episode topology mutations
│       ├── simulator/
│       │   ├── llm_simulator.py       # Llama-3.1-8B observation generator + cache
│       │   └── preload_cache.py       # CLI cache pre-warmer
│       ├── reward/
│       │   └── reward_orchestrator.py # F-beta(β=2) profiles + scoring
│       ├── trajectory/
│       │   ├── trajectory_logger.py   # JSONL episode writer
│       │   └── trajectory_auditor.py  # JSONL → strategy tags
│       └── ui/
│           └── playground.py          # Gradio 6 interactive playground
├── scripts/
│   ├── inference.py                   # Baseline LLM agent loop
│   └── benchmark.py                  # Multi-seed benchmarking harness
├── tests/
│   ├── test_smoke.py                  # Offline unit + integration tests
│   └── test_llm.py                    # Live LLM integration tests
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Package config (v0.2.0)
└── Dockerfile                         # Container image with pre-warmed cache
```

---

## Design Rationale

### Why `result=[]` during queries?
Forcing agents to parse free-text LLM output — not JSON arrays — requires genuine **text comprehension and cross-source synthesis**. An agent must extract service names from PagerDuty prose, reconcile noisy registry output against runbook tables, and distinguish signal from monitoring noise. This mirrors real SRE cognition.

### Why F-beta β=2?
In production incidents, missing a cascading service is catastrophically worse than a false alarm. F-beta β=2 formalizes this asymmetry: recall is weighted 4× over precision, reflecting the true SRE cost model.

### Why LLM-as-world-model?
Scripted observations can be memorized. LLM-generated observations vary in phrasing, ordering, and emphasis — creating a **generalization surface** even for agents that have "seen" the environment before. Noise injection (hallucinated/dropped services) is LLM-mediated, making pattern exploitation much harder.

---

## LLM Configuration

| Setting | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Provider | Cerebras via HuggingFace Inference Providers |
| Temperature | `0.0` (deterministic) |
| Seed | `42` (reproducible across runs) |
| Cache strategy | MD5-keyed · in-memory + JSON disk |
| Fallback | Template strings when `HF_TOKEN` is absent |

---

## References

- [OpenEnv Specification](https://github.com/openenv/openenv)
- [Meta × PyTorch OpenEnv Hackathon](https://huggingface.co/spaces/openenv/openenv-hackathon)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Cerebras Inference](https://cerebras.ai/inference)
- [F-score — Wikipedia](https://en.wikipedia.org/wiki/F-score)

---

<div align="center">

*Built for the Meta × PyTorch OpenEnv Hackathon · Service topology and incident scenarios are synthetic*

</div>
