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

# cascade-mind — Service Impact Analysis

An AI agent plays the role of an on-call SRE engineer. A microservice just changed — the agent must trace the blast radius and identify every downstream service that will be impacted, using the same noisy, LLM-generated tool output a real engineer would see.

Every observation (incident alerts, registry lookups, runbooks, monitoring snapshots) is generated live by **Llama-3.1-8B via Cerebras**. The ground truth is a seeded `networkx` dependency graph of 15 microservices — the agent never sees it directly.

📖 **[Full API docs + WebSocket example](https://rajkamal2819-cascade-mind.hf.space/docs)**

---

## Try it in the Playground

1. Click **Reset** to start a new episode — you'll get a PagerDuty-style incident alert
2. Pick an **Action Type** from the dropdown and enter a **Service Name**
3. Click **Step** to see the LLM-generated tool output
4. When ready, select `submit` and enter your list of affected services in **Affected Services**
5. Click **Step** to submit — your F-beta score appears in the response

---

## Available Actions

| Action Type | What it returns |
|---|---|
| `query_dependents` | Services that directly call the queried service (may be noisy) |
| `query_dependencies` | Services the queried service depends on (may be noisy) |
| `query_runbook` | Internal runbook: ownership, SLOs, known failure modes |
| `query_changelog` | Recent PR / deployment changelog for the changed service |
| `query_monitoring` | Live monitoring snapshot: latency, error rate, dependency health |
| `submit` | Submit your final answer — ends the episode, returns score |

`query_dependents` and `query_dependencies` draw from a shared **query budget of 10**.  
The other three actions are always free.

---

## Connect from Python

```python
import asyncio, json, websockets

async def run():
    async with websockets.connect("wss://rajkamal2819-cascade-mind.hf.space/ws") as ws:
        # Start episode
        await ws.send(json.dumps({"type": "reset", "data": {"seed": 0}}))
        obs = json.loads(await ws.recv())
        print(obs["data"]["message"])          # incident alert

        # Read the changelog (free, no budget)
        await ws.send(json.dumps({"type": "step", "data": {
            "action_type": "query_changelog", "service_name": "catalog_service"
        }}))
        print(json.loads(await ws.recv())["data"]["message"])

        # Query dependents
        await ws.send(json.dumps({"type": "step", "data": {
            "action_type": "query_dependents", "service_name": "catalog_service"
        }}))
        print(json.loads(await ws.recv())["data"]["message"])

        # Submit answer
        await ws.send(json.dumps({"type": "step", "data": {
            "action_type": "submit",
            "affected_services": ["api_gateway", "cart_service", "web_backend"]
        }}))
        result = json.loads(await ws.recv())
        print("Score:", result["data"]["reward"])   # float in [0.0, 1.0]

asyncio.run(run())
```

Or use the typed Python client:

```python
from cascade_mind import ServiceImpactEnv

async with ServiceImpactEnv(base_url="https://rajkamal2819-cascade-mind.hf.space") as env:
    obs, _ = await env.reset(seed=0)
    print(obs.message)
```

---

## Scoring

Scored with **F-beta (β=2)** against the ground-truth dependency graph — recall is weighted twice as heavily as precision, because missing a real affected service causes more damage than a false alarm.

Difficulty levels:
- `easy` — clean registry output, 5 query budget
- `medium` — one service added/removed in registry output, 8 query budget  
- `hard` — two services added/removed, 12 query budget

---

## Run the Baseline Agent

```bash
git clone https://github.com/rajkamal2819/cascade-mind
cd cascade-mind

export HF_TOKEN=hf_...
export API_BASE_URL=https://api.cerebras.ai/v1
export MODEL_NAME=llama-3.3-70b

python inference.py
```

The script runs all three tasks (easy / medium / hard) against the live Space and prints per-task F-beta scores plus a final `JSON_RESULTS` summary.

---


## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GROUND TRUTH LAYER                          │
│   networkx DiGraph  (30 services, seed-perturbed topology per       │
│   episode — 55-80 edges, 10,000+ unique episodes)                   │
│   nx.ancestors(G, service)  →  exact affected set (scorer only)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Structured data (never exposed to agent)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LLM SIMULATOR LAYER                           │
│   Llama-3.1-8B  (Cerebras, via HuggingFace Inference Providers)     │
│                                                                      │
│   Incident alert   →  PagerDuty-style P1 alert text                 │
│   Registry query   →  Noisy service registry CLI output             │
│   Runbook          →  Confluence-style Markdown page                │
│   Monitoring       →  Datadog-style JSON (spike annotations)        │
│   Changelog        →  PR/CHANGELOG entry for the changed service    │
│                                                                      │
│   Noise levels: easy=0.15  medium=0.40  hard=0.70                   │
│   (hallucinated + dropped services increase with difficulty)         │
│   2-layer cache: in-memory + JSON disk (pre-warmed in Docker)        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ LLM text (noisy, sometimes wrong)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           AGENT LAYER                               │
│   Sees only: observation.message (LLM text)                         │
│   Never sees: result[] during queries (always [])   ← Option A      │
│   Budget: 5–12 queries depending on difficulty                      │
│   Goal: submit the correct set of affected services                 │
└─────────────────────────────────────────────────────────────────────┘
```

**This is the first OpenEnv environment to use an LLM as the observation generator.** Agents must reason under uncertainty — cross-referencing multiple noisy sources to reconstruct the true impact set.

---

## Why this domain?

1. **Real-world relevance**: Every software company with microservices faces this exact problem. A misconfiguration in `auth_service` can cascade to 15+ other services in ways that aren't obvious.

2. **Information asymmetry**: Real SREs don't have a perfect dependency graph. They have stale runbooks, unreliable service registries, and 3am PagerDuty alerts with incomplete details. This environment models that reality.

3. **Reasoning challenge**: The agent must decide *which tools to use*, *in what order*, and *how much to trust each source*. Registry output may list wrong dependents. Runbooks may reference services that were renamed. Monitoring may show noise spikes unrelated to the incident.

4. **LLM-as-world-model novelty**: Instead of scripted responses, every observation is generated by Llama-3.1-8B. This creates an evaluation surface that probes whether agents can extract signal from realistic, unstructured SRE text.

---

## Action Space

| Action | Budget Cost | Description |
|---|---|---|
| `query_dependents` | −1 query | "Which services call `X`?" → noisy registry output |
| `query_dependencies` | −1 query | "What does `X` call?" → noisy registry output |
| `query_runbook` | **Free** | Read the runbook for `X` (Confluence-style Markdown) |
| `query_changelog` | **Free** | Read recent PR/CHANGELOG for `X` |
| `query_monitoring` | **Free** | Read Datadog-style monitoring data for `X` |
| `submit` | Terminal | Submit final list of predicted affected services |

Budget varies by difficulty: **easy=5, medium=8, hard=12 queries**.
Free actions (runbook, changelog, monitoring) do not consume budget.

### Action Schema

```json
{ "action_type": "query_dependents", "service_name": "payment_service" }
{ "action_type": "query_runbook",    "service_name": "auth_service" }
{ "action_type": "submit",           "affected_services": ["cart_service", "order_service"] }
```

---

## Observation Space

Every observation contains:

| Field | Type | Description |
|---|---|---|
| `changed_service` | `str` | The service that had a breaking change this episode |
| `message` | `str` | **LLM-generated text** from the requested tool |
| `result` | `List[str]` | Always `[]` during queries (Option A — agents must parse `message`) |
| `queries_remaining` | `int` | Budget remaining |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float \| None` | `None` until terminal step |

On `submit`, `result` is populated with the ground-truth affected set (for postmortem).

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

$$F_\beta(\beta=2) = \frac{(1+4) \cdot P \cdot R}{4 \cdot P + R}$$

**β=2** weights recall 4× over precision — reflecting the SRE cost model where missing an affected service (outage propagates undetected) is much worse than a false alarm.

| Condition | Modifier |
|---|---|
| Each new unique service queried | +0.05 |
| Re-querying same service | −0.05 |
| Budget exhausted without submit | −0.40 |
| Overclaiming (submit > 60% of all services) | `−0.3 × oversubmit_fraction` |
| Terminal submit | F-beta(β=2) ∈ [0.0, 1.0] |

---

## Episode Variety: 10,000+ Unique Episodes

Each seed produces a different microservice topology through deterministic perturbation:
- **Remove 4–9 redundant edges** (edges with alternative paths — structurally safe to remove)
- **Add 3–7 new edges** from 27 candidate edges (plausible but non-obvious dependencies)

The `changed_service` is selected dynamically by **betweenness-centrality**, binned into difficulty tiers:

| Difficulty | Affected services | Queries budget |
|---|---|---|
| Easy | 1–6 | 5 |
| Medium | 6–13 | 8 |
| Hard | 13+ | 12 |

Seed cycling: `seed % 3` selects the difficulty tier, ensuring balanced evaluation.

---

## Microservice Graph

30 services across 3 tiers:

```
Tier 1 (Gateway):    api_gateway, mobile_backend, web_backend
Tier 2 (Business):   auth_service, user_service, order_service, cart_service,
                     checkout_service, payment_service, billing_service,
                     subscription_service, inventory_service, shipping_service,
                     catalog_service, search_service, recommendation_service,
                     review_service, notification_service
Tier 3 (Support):    email_service, sms_service, media_service, analytics_service,
                     logging_service, audit_service, config_service, cache_service,
                     database_service, message_queue, feature_flags, rate_limiter
```

---

## LLM Noise Model

The simulator injects realistic noise at each difficulty level:

| Difficulty | Noise Level | Hallucinated services | Dropped services |
|---|---|---|---|
| Easy | 0.15 | 0 | 0 |
| Medium | 0.40 | 1 | 1 |
| Hard | 0.70 | 2 | 2 |

*Hallucinated*: registry lists a service that doesn't actually depend on `X`.  
*Dropped*: registry omits a service that does actually depend on `X`.

This means at Hard difficulty, a registry query for a service with 8 dependents might show 9 results — 7 correct + 2 hallucinated — while silently dropping 2 real ones. The agent must triangulate from runbooks, monitoring, and multiple query directions.

---

## MCP Endpoint

The server exposes an **MCP (Model Context Protocol)** endpoint at `/mcp` for direct tool integration:

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

Available MCP tools: `query_dependents`, `query_dependencies`, `query_runbook`, `query_changelog`, `query_monitoring`, `submit_impact_assessment`

---

## Quick Start

### Run locally

```bash
# Clone and install
git clone https://github.com/your-org/service-impact-env
cd service_impact_env
pip install -e ".[dev]"

# Set credentials
export HF_TOKEN=hf_your_token_here    # for Llama-3.1-8B via Cerebras

# Start the environment server
LLM_SIMULATOR_ENABLED=true uvicorn server.app:app --port 8000

# In another terminal — run the baseline agent
OPENAI_API_KEY=sk-... python inference.py
```

### Without LLM (template fallbacks — fully offline)

```bash
LLM_SIMULATOR_ENABLED=false uvicorn server.app:app --port 8000
```

### Run tests

```bash
# 57 unit + integration tests (no LLM calls required)
LLM_SIMULATOR_ENABLED=false pytest test_smoke.py -v

# Live LLM test (requires HF_TOKEN with Cerebras access)
HF_TOKEN=hf_... pytest test_llm.py -v
```

---

## Docker

```bash
# Build with LLM cache pre-warmed (seeds 0-99 baked in)
docker build \
  --build-arg HF_TOKEN=hf_your_token_here \
  -t service_impact_env \
  server/

# Run
docker run -p 8000:8000 \
  -e HF_TOKEN=hf_your_token_here \
  -e LLM_SIMULATOR_ENABLED=true \
  service_impact_env
```

The Dockerfile pre-warms the LLM cache during build (`python -m server.preload_cache --seeds 100`), so the first 100 seed episodes serve from disk with zero LLM latency.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | `""` | HuggingFace token (Cerebras access for Llama-3.1-8B) |
| `LLM_SIMULATOR_ENABLED` | `true` | Set `false` for template-only mode |
| `LLM_CACHE_PATH` | `/tmp/llm_sim_cache.json` | Path for the LLM response cache |
| `OPENAI_API_KEY` | — | For baseline `inference.py` agent |

---

## Cache Pre-Warming

To generate LLM responses for seeds 0–N ahead of time:

```bash
HF_TOKEN=hf_... python -m server.preload_cache --seeds 100 \
  --output /tmp/llm_sim_cache.json
```

Takes ~5 minutes for 100 seeds (600 LLM calls). Mount the JSON file into Docker for instant episode start times across the first 100 seeds.

---

## File Structure

```
service_impact_env/
├── models.py                          # Pydantic Action/Observation/State models
├── inference.py                       # Baseline LLM agent loop (OpenAI)
├── test_smoke.py                      # 57 unit tests (no LLM required)
├── test_llm.py                        # Live LLM integration tests
├── openenv.yaml                       # OpenEnv environment manifest
├── pyproject.toml                     # Python package config
└── server/
    ├── app.py                         # FastAPI app + MCP endpoint (/mcp)
    ├── service_impact_environment.py  # Core environment logic (v0.2.0)
    ├── graph_builder.py               # Seed-perturbed networkx dependency graph
    ├── llm_simulator.py               # Llama-3.1-8B SRE tool simulator + cache
    ├── preload_cache.py               # CLI cache pre-warmer
    ├── requirements.txt               # Python dependencies
    └── Dockerfile                     # Container image with pre-warmed cache
```

---

## Key Design Decisions

### Why Option A (result=[] during queries)?

Option A forces agents to develop **text comprehension and information synthesis** skills. An agent that can only parse JSON arrays would achieve a high score trivially. Option A requires the agent to:

1. Extract service names from PagerDuty alert prose
2. Cross-reference registry output (which may list wrong services) against runbook tables
3. Identify which monitoring spikes are signal vs. noise
4. Decide *when to stop querying* and commit to an answer

This mirrors what LLM agents must do in real SRE workflows.

### Why F-beta β=2?

In production incidents, **missing an affected service is far worse than a false alarm**. An overcautious SRE who notifies 5 extra teams wastes some engineering time. An SRE who misses that `billing_service` is cascading causes a payment outage. F-beta β=2 formalizes this asymmetry.

### Why LLM-as-world-model?

Scripted observations (returning deterministic strings) can be memorized. LLM-generated observations vary with phrasing, ordering, and emphasis — creating a **generalization test** even for agents that have "seen" the environment before. The noise injection (hallucinated/dropped services) is itself LLM-mediated, making it harder to exploit.

---

## LLM Configuration

| Setting | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Provider | Cerebras (via HuggingFace Inference Providers) |
| Temperature | 0.0 (deterministic) |
| Seed | 42 (reproducible) |
| Cache | MD5-keyed, in-memory + JSON disk |
| Fallback | Template strings (if `HF_TOKEN` absent or LLM unavailable) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "healthy"}` |
| `GET` | `/schema` | JSON schemas for action, observation, state |
| `GET` | `/metadata` | Environment name, description, version |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute an action |
| `GET` | `/state` | Current episode state |
| `WS` | `/ws` | WebSocket session (primary protocol) |
| `GET` | `/mcp` | MCP tools manifest |
| `POST` | `/mcp` | MCP JSON-RPC 2.0 tool calls |
| `GET` | `/docs` | Swagger UI |

---

## Environment Manifest

```yaml
spec_version: 1
name: service_impact_env
type: space
runtime: fastapi
app: server.app:app
port: 7860
```

---

## Running the Baseline Agent

`inference.py` runs an LLM-driven agent across all three difficulty levels (easy / medium / hard)
and prints per-task F-beta scores plus a JSON summary.

**Against the live HuggingFace Space** (recommended):

```bash
export SPACE_ID="Rajkamal2819/cascade-mind"
export HF_TOKEN=hf_...                          # your HuggingFace read token
export API_BASE_URL=https://api.cerebras.ai/v1  # Cerebras via HF Inference Providers
export MODEL_NAME=llama-3.3-70b                 # or any OpenAI-compatible model

python inference.py
```

**Against a local server**:

```bash
# Terminal 1 — start the server
export HF_TOKEN=hf_...
export LLM_SIMULATOR_ENABLED=true
uvicorn server.app:app --port 8000

# Terminal 2 — run the agent
export ENV_BASE_URL=http://localhost:8000
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python inference.py
```

Expected output includes per-task scores and a final `JSON_RESULTS: {"mean_f1": ..., ...}` line
that can be parsed by automated evaluation harnesses.

---

## References

- [OpenEnv Specification](https://github.com/openenv/openenv)
- [Meta x PyTorch OpenEnv Hackathon](https://huggingface.co/spaces/openenv/openenv-hackathon)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Cerebras Inference](https://cerebras.ai/inference)
- [F-score (Wikipedia)](https://en.wikipedia.org/wiki/F-score)

---

*Built for the Meta x PyTorch OpenEnv Hackathon. Service topology and incident scenarios are synthetic.*
