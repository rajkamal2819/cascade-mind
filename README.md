---
title: cascade-mind
colorFrom: gray
colorTo: gray
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

### World-Modeling RL Environment for Causal Blast-Radius Analysis

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Rajkamal2819/cascade-mind)
[![API Docs](https://img.shields.io/badge/FastAPI-Docs-009688?logo=fastapi)](https://rajkamal2819-cascade-mind.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-7c3aed)](https://github.com/openenv/openenv)

**cascade-mind** is an [OpenEnv](https://github.com/openenv/openenv)-compatible reinforcement learning environment where an AI agent acts as an on-call SRE engineer. A microservice just had a breaking change — the agent must trace the blast radius across a 30-service dependency graph using only noisy, LLM-generated tool outputs.

**First OpenEnv environment to use an LLM as the world model.** Scripted observations can be memorized. Ours can't.

[**🚀 Live Playground**](https://rajkamal2819-cascade-mind.hf.space) · [**📖 API Docs**](https://rajkamal2819-cascade-mind.hf.space/docs) · [**🗺️ Ground Truth Graph**](https://rajkamal2819-cascade-mind.hf.space/graph/ground-truth?seed=0&difficulty=easy)

</div>

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
│   reset()         →  PagerDuty-style P1 incident alert              │
│   query_*         →  Noisy registry / runbook / monitoring output   │
│   query_changelog →  PR / CHANGELOG entry for the changed service   │
│                                                                      │
│   Noise:  easy 15%  ·  medium 40%  ·  hard 70%                     │
│   Cache:  2-layer  (in-memory MD5 + JSON disk, pre-warmed in CI)    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  LLM text — noisy, sometimes wrong
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    WORLD MODELING LAYER  (★ novel)                  │
│                                                                      │
│   BeliefTracker       — per-service confidence [0,1], updated each  │
│                         step, returned in every observation         │
│   ContradictionEngine — cross-tool disagreement detection;          │
│                         fires [CONTRADICTION DETECTED] alerts       │
│   GraphPrior          — session-level edge-frequency table;         │
│                         high-conf edges appear as hints at reset()  │
│   ProcessReward (PRM) — information_gain + delta-Fβ per step;      │
│                         dense reward signal throughout episode      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  structured uncertainty
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           AGENT LAYER                               │
│   Sees: observation.message + belief_state + contradiction_count    │
│   result[] is always [] during queries  (must parse free text)      │
│   Budget: 13–15 / 10–12 / 8–10 queries by difficulty               │
│   Goal: submit the correct affected set before budget runs out      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What Makes This Hard

- **result=[] always during queries.** The agent must parse meaning from free-text LLM prose — no JSON extraction.
- **Noise is LLM-mediated, not scripted.** At Hard difficulty, a registry lookup may return 7 correct + 2 hallucinated services while silently dropping 2 real ones. Phrasing, ordering, and emphasis vary across seeds.
- **The graph mutates mid-episode.** `MutationEngine` fires silently at scheduled steps on Medium/Hard, expanding the ground truth. The agent gets no explicit notification — it must detect the shift from observation changes.
- **4 rotating reward profiles per episode** (recall_heavy, balanced, precision_heavy, efficiency) prevent the agent from memorizing a single strategy.
- **10,000+ unique episodes per difficulty** via seed-perturbed topology. No memorization is possible.

---

## Quick Start

### Python Client (WebSocket)

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
        print("Score:", result["data"]["reward"])   # float ∈ (0.001, 0.999)

asyncio.run(run())
```

### Typed Client

```python
from cascade_mind import ServiceImpactEnv

async with ServiceImpactEnv(base_url="https://rajkamal2819-cascade-mind.hf.space") as env:
    obs, _ = await env.reset(seed=0)
    print(obs.message)        # incident alert text
    print(obs.belief_state)   # {"cart_service": 0.0, ...} — starts empty
```

---

### Run Locally

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

### Docker

```bash
docker build --build-arg HF_TOKEN=hf_your_token_here -t cascade-mind .
docker run -p 8000:8000 -e HF_TOKEN=hf_your_token_here -e LLM_SIMULATOR_ENABLED=true cascade-mind
```

### Baseline Agent

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://api.cerebras.ai/v1
export MODEL_NAME=llama-3.3-70b
python inference.py
```

---

## Action Space

| Action | Cost | Description |
|---|:---:|---|
| `query_dependents` | −1 | "Which services call X?" → noisy registry output |
| `query_dependencies` | −1 | "What does X depend on?" → noisy registry output |
| `query_runbook` | free (cap 2) | Confluence-style runbook: ownership, SLOs, failure modes |
| `query_changelog` | free (cap 2) | Recent PR / deployment changelog for X |
| `query_monitoring` | free (cap 3) | Datadog-style snapshot: latency, error rate, dependency health |
| `query_service_health` | free (uncapped) | Real-time health status and metadata for X |
| `query_topology_diff` | free (uncapped) | Topology changes since episode start (reveals mutations) |
| `submit_hypothesis` | −1 | Test a partial hypothesis mid-episode — returns partial F-beta, non-terminal |
| `submit` | terminal | Submit the final predicted affected set — ends the episode |

Budget by difficulty: **easy = 13–15 · medium = 10–12 · hard = 8–10** (randomised per seed)

### Action Schema

```json
{ "action_type": "query_dependents",  "service_name": "payment_service" }
{ "action_type": "query_runbook",     "service_name": "auth_service" }
{ "action_type": "submit_hypothesis", "affected_services": ["cart_service"], "confidence": 0.7 }
{ "action_type": "submit",            "affected_services": ["cart_service", "order_service"] }
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `changed_service` | `str` | The service with a breaking change this episode |
| `message` | `str` | LLM-generated tool output (free text) |
| `result` | `List[str]` | Always `[]` during queries; ground truth revealed on `submit` |
| `queries_remaining` | `int` | Budget remaining |
| `done` | `bool` | `true` after a terminal action |
| `reward` | `float \| None` | Step reward; F-beta score on `submit` |
| `belief_state` | `Dict[str, float]` | Per-service confidence [0,1] — updated each step |
| `information_gain` | `float \| None` | Belief-state Jaccard improvement from this step |
| `intermediate_fbeta` | `float \| None` | F-beta of current high-confidence set vs ground truth |
| `contradiction_count` | `int` | Cumulative cross-tool disagreements detected |
| `world_version` | `int` | Increments on each mutation event |
| `belief_drift` | `float \| None` | Confidence shift since last step (signals mutation) |
| `graph_prior` | `str \| None` | Session-level edge hints from prior episodes (reset only) |

**Example reset observation:**
```
[PagerDuty] INCIDENT INC-7841 | P1 | TRIGGERED
Service: catalog_service
Alert: Breaking API change detected — downstream consumers may be impacted.

Budget: 12 queries remaining.
Reward profile: balanced

Belief state: all services at 0.0 (no evidence yet)
```

---

## Reward Design

$$F_\beta(\beta=2) = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$$

β=2 weights **recall 4× over precision** — missing an affected service is far worse than a false alarm.

| Condition | Modifier |
|---|:---:|
| Each new unique service queried | +0.05 |
| Re-querying the same service | −0.05 |
| Budget exhausted without submit | −0.40 |
| Overclaiming (> 60% of all services) | −0.3 × oversubmit fraction |
| Terminal `submit` | F-beta(β=2) ∈ (0.001, 0.999) |

### Rotating Reward Profiles

To prevent reward hacking, the environment rotates across 4 profiles per seed band:

| Profile | β | Overclaim threshold | Strategy |
|---|:---:|:---:|---|
| `recall_heavy` | 2.5 | 70% | Be very inclusive |
| `balanced` | 1.5 | 60% | Balance coverage and precision |
| `precision_heavy` | 0.8 | 50% | Be selective, avoid false positives |
| `efficiency` | 2.0 | 65% | Finish fast — budget bonus for early submit |

---

## Episode Variety

Each seed produces a unique microservice topology through deterministic graph perturbation:

- **Remove 4–9 redundant edges** (edges with structurally safe alternative paths)
- **Add 3–7 new edges** sampled from 27 plausible candidate dependencies

The `changed_service` is selected by **betweenness-centrality**, binned into difficulty tiers:

| Difficulty | Blast radius | Query budget | Registry noise | Mutations |
|---|:---:|:---:|:---:|:---:|
| Easy | 1–6 services | 13–15 | 15% | None |
| Medium | 6–13 services | 10–12 | 40% | Step 5 (30% prob) |
| Hard | 13+ services | 8–10 | 70% | Steps 4 + 8 (always) |

---

## LLM Noise Model

| Difficulty | Hallucinated services | Dropped services | Drop rate | Inject rate |
|---|:---:|:---:|:---:|:---:|
| Easy | 0 | 0 | 5% | 10% |
| Medium | 1 | 1 | 15% | 20% |
| Hard | 2 | 2 | 25% | 30% |

At **Hard** difficulty, a registry query for a service with 8 real dependents may return 9 results — 7 correct + 2 hallucinated — while silently dropping 2 real ones. The agent must triangulate across runbooks, monitoring, and query directions.

---

## Two Domains

cascade-mind ships two causal graph domains. The World Modeling Layer, reward pipeline, and GRPO training interface are identical across both.

### Domain 1 — SRE / Microservices

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

A breaking API change fires a PagerDuty P1 alert. The agent must trace the blast radius through noisy registry CLI outputs, Confluence runbooks, and Datadog monitoring snapshots.

### Domain 2 — Supply Chain Disruption

30 nodes (raw-materials, factories, logistics, distributors, retail), 5 real-world incident archetypes:

| Archetype | Based on |
|---|---|
| Semiconductor fab fire | Renesas Naka factory fire, 2021 |
| Canal blockage | Ever Given / Suez Canal, 2021 |
| Demand shock | COVID-19 staples surge, 2020 |
| Fab concentration risk | TSMC Taiwan concentration |
| Maritime route disruption | Red Sea shipping attacks, 2023–24 |

Use the **Domain** dropdown in the [live playground](https://rajkamal2819-cascade-mind.hf.space) to switch between domains.

---

## GRPO Training

Every episode writes a JSONL trajectory. The export endpoint returns TRL-ready data:

```python
import requests
from datasets import load_dataset

data = requests.get(
    "https://rajkamal2819-cascade-mind.hf.space/export/grpo?min_reward=0.5"
).json()

dataset = load_dataset("json", data_files=data["path"])
# → ready for GRPOTrainer in HF TRL
```

---

## MCP Integration

The server exposes a **Model Context Protocol** endpoint at `/mcp`:

```bash
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
| `GET` | `/belief` | Current belief state |
| `GET` | `/prior` | Session graph prior |
| `GET` | `/contradictions` | Detected contradictions this episode |
| `GET` | `/export/grpo` | Export trajectories as JSONL (TRL-ready) |
| `GET` | `/graph/ground-truth` | Interactive vis.js ground-truth graph |
| `GET` | `/mcp` | MCP tools manifest |
| `POST` | `/mcp` | MCP JSON-RPC 2.0 tool calls |
| `GET` | `/docs` | Swagger UI |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token for Cerebras/Llama-3.1-8B (required) |
| `LLM_SIMULATOR_ENABLED` | `true` | `false` for fully offline template mode |
| `LLM_CACHE_PATH` | `/tmp/llm_sim_cache.json` | Path for LLM response cache |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint for inference agent |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier for inference agent |
| `ENV_BASE_URL` | `https://rajkamal2819-cascade-mind.hf.space` | Environment server URL |

---

## Repository Structure

```
cascade-mind/
├── inference.py                           # Baseline LLM agent (validator entry point)
├── cascade_mind/
│   ├── __init__.py                        # Public API re-exports
│   ├── models.py                          # Pydantic Action / Observation / State models
│   ├── client.py                          # Typed WebSocket client (ServiceImpactEnv)
│   └── server/
│       ├── app.py                         # FastAPI app · routes · MCP endpoint
│       ├── domain/
│       │   ├── domain_config.py           # DomainConfig plugin interface
│       │   ├── sre_domain.py              # SRE 30-node graph + metadata
│       │   └── supply_chain_domain.py     # Supply-chain 30-node graph + archetypes
│       ├── env/
│       │   ├── service_impact_environment.py  # Core reset() / step() logic (1141 lines)
│       │   ├── belief_tracker.py          # Per-service Bayesian confidence tracker
│       │   ├── contradiction_engine.py    # Cross-tool disagreement detection
│       │   ├── graph_prior.py             # Session-level edge frequency table
│       │   └── curriculum_scheduler.py   # Per-difficulty config (caps, noise, hints)
│       ├── graph/
│       │   ├── graph_builder.py           # 30-service seed-perturbed NetworkX DiGraph
│       │   └── mutation_engine.py         # Mid-episode topology mutations
│       ├── simulator/
│       │   ├── llm_simulator.py           # Llama-3.1-8B observation generator + cache
│       │   └── preload_cache.py           # CLI cache pre-warmer
│       ├── reward/
│       │   ├── reward_orchestrator.py     # 4 rotating F-beta profiles
│       │   └── process_reward.py          # Dense step-level PRM signals
│       ├── trajectory/
│       │   ├── trajectory_logger.py       # JSONL episode writer
│       │   └── trajectory_auditor.py      # Strategy tagging + audit reports
│       └── ui/
│           └── playground.py             # Gradio 6 interactive playground
├── scripts/
│   ├── inference.py                       # Same agent script (scripts/ copy)
│   └── benchmark.py                       # Multi-seed benchmarking harness
├── tests/
│   ├── test_smoke.py                      # Offline unit + integration tests
│   └── test_llm.py                        # Live LLM integration tests
├── notebooks/
│   └── grpo_sre_training_8b_final.ipynb  # GRPO training with TRL
├── openenv.yaml                           # OpenEnv manifest
├── pyproject.toml                         # Package config (v0.2.0)
└── Dockerfile                             # Container image (ghcr.io/meta-pytorch/openenv-base)
```

---

## Design Rationale

**Why `result=[]` during queries?**
Forcing agents to parse free-text LLM output requires genuine text comprehension and cross-source synthesis. This mirrors real SRE cognition — engineers don't get structured JSON from PagerDuty or Datadog.

**Why F-beta β=2?**
In production incidents, missing a cascading service is catastrophically worse than a false alarm. F-beta β=2 formalizes this asymmetry: recall is weighted 4× over precision.

**Why the World Modeling Layer?**
The BeliefTracker, ContradictionEngine, and GraphPrior give the agent explicit uncertainty signals — enabling hypothesis-driven investigation rather than blind BFS. These also produce dense intermediate reward signals (PRM) that make GRPO training significantly more stable than sparse terminal reward alone.

**Why LLM-as-world-model?**
Scripted observations can be memorized. LLM-generated observations vary in phrasing, ordering, and emphasis — creating a generalization surface even for agents that have seen the environment before.
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
