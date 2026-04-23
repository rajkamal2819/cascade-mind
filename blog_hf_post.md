---
title: "Cascade-Mind: A World Modeling RL Framework for Causally-Dependent Knowledge Domains"
thumbnail: /blog/assets/cascade-mind/thumbnail.png
authors:
  - user: Rajkamal2819
tags:
  - reinforcement-learning
  - world-modeling
  - openenv
  - llm-agents
  - grpo
---

# Cascade-Mind: Training LLM Agents to Reason About Hidden Worlds

*An OpenEnv-compatible framework for building RL environments where an agent must
construct and maintain a causal model of a hidden dependency graph — using noisy,
contradictory tools — applicable to any professional domain where knowledge has
graph structure.*

---

## Why This Exists

Here is a problem every engineer has lived through.

A critical library gets compromised. A supplier factory shuts down. A microservice
silently breaks. In every case the question is identical: **which downstream entities
are affected — and how far does it cascade?**

The answer is never obvious. The dependency graph is hidden. The tools you have
(registry scanners, monitoring dashboards, vendor reports, runbooks) return
incomplete, stale, sometimes contradictory information. And you are racing the
clock.

This is not an SRE problem or a supply chain problem. It is a **world modeling
problem** — and it is the shape of the hardest real-world reasoning tasks that
exist.

Most RL benchmarks give agents a clean, complete world state and ask them to plan.
Cascade-Mind does the opposite. It hides the world, makes the tools lie, and
charges the agent for every query.

---

## What Cascade-Mind Is

Cascade-Mind is not a single RL environment. It is a **framework for generating
OpenEnv-compatible RL environments** for any domain where:

- Entities have **causal dependencies** (a knowledge graph exists)
- The ground truth is **partially observable** (tools return approximations)
- Queries have a **cost** (finite budget before forced submission)
- The outcome is **verifiable** (ground truth can be computed from the graph)

The core claim is simple:

> *OpenEnv is the universal interface for training LLMs on any task.
> Cascade-Mind is the universal framework for building environments on any domain
> that has a knowledge graph.*

Change the domain plugin → new causal graph, new incident archetypes, new tool
vocabulary. The agent interface, reward function, world modeling layer, and GRPO
training pipeline stay identical.

---

## Architecture

*[Architecture diagram — System Overview]*

The system has five major layers:

**Ground Truth Layer** — A seeded, perturbed NetworkX DiGraph. At `reset()`, the
changed entity and all downstream affected nodes are computed via `nx.ancestors()`
and frozen as a frozenset. The agent never sees this object. The scorer reads it
only at `submit`.

**LLM Simulator Layer** — Every tool output is a fresh Llama-3.1-8B-Instruct
generation via Cerebras. Registry lookups, runbook retrieval, monitoring snapshots,
topology diffs — all LLM-generated, all domain-appropriate, all noisy. Scripted
observations can be memorized. LLM-generated observations require genuine
comprehension. Noise scales from 15% on Easy to 90% on Ultra.

**World Modeling Layer** — The core novel contribution. Four components that no
other OpenEnv environment has, working together as a coherent layer:

- **BeliefTracker** — maintains a `Dict[str, float]` of per-entity confidence,
  updated after every tool call, returned in every observation. The agent literally
  sees its own uncertainty at each step.
- **ContradictionEngine** — cross-references outputs across tool types. When a
  registry says entity B is a dependent of A, but the runbook says B was deprecated,
  a structured `[CONTRADICTION DETECTED]` alert fires in the next observation.
- **GraphPrior** — a session-level edge-frequency table built across episodes.
  High-confidence edges from previous episodes appear as hints at `reset()`.
  Within-session meta-learning without gradient updates.
- **Process Reward Model (PRM)** — computes `information_gain` and
  `intermediate_fbeta` at every step, providing dense reward signal throughout
  the episode rather than only at terminal `submit`.

**Reward Layer** — Domain-agnostic F-beta scoring with β=2, weighting recall 4×
over precision. This encodes the real cost asymmetry present in every professional
domain: missing a cascading failure is always worse than a false alarm.

**Audit and Export Layer** — Every episode writes a JSONL trajectory with action
history, belief state snapshots, strategy tags, and reward components. The GRPO
export endpoint at `/export/grpo?min_reward=0.5` returns a TRL-ready dataset in
three lines of code.

*[Architecture diagram — Reward Pipeline]*

*[Architecture diagram — Episode Lifecycle]*

---

## The Core Abstraction

Every Cascade-Mind environment shares one loop:

```
┌─────────────────────────────────────────────────────────┐
│              GROUND TRUTH LAYER (hidden)                │
│  NetworkX DiGraph · seeded + perturbed per episode      │
│  ground_truth = frozenset(nx.ancestors(G, changed))     │
│  Agent never sees this — scorer reads it at submit only │
└───────────────────────┬─────────────────────────────────┘
                        │ hides the world
                        ▼
┌─────────────────────────────────────────────────────────┐
│           LLM SIMULATOR LAYER (noisy oracle)            │
│  Llama-3.1-8B · Cerebras · noise 15–90% by difficulty  │
│  reset()   → domain-specific incident framing           │
│  query_*   → noisy tool outputs (registry, runbook,     │
│              monitoring, topology diff, health check)   │
└───────────────────────┬─────────────────────────────────┘
                        │ free text — noisy, sometimes wrong
                        ▼
┌─────────────────────────────────────────────────────────┐
│              WORLD MODELING LAYER (★ new)               │
│  BeliefTracker · ContradictionEngine · GraphPrior · PRM │
│  belief_state returned in every observation             │
│  information_gain computed per paid query               │
│  contradiction_count escalates when tools conflict      │
└───────────────────────┬─────────────────────────────────┘
                        │ structured uncertainty
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    AGENT LAYER                          │
│  Sees: observation.message + belief_state               │
│  result[] is always [] during queries (parse prose)     │
│  Budget: 5 / 8 / 12 queries by difficulty tier         │
│  Goal: submit the correct affected set                  │
└─────────────────────────────────────────────────────────┘
```

The agent's task is the same regardless of domain: **reconstruct the hidden causal
graph from noisy observations and submit the correct affected set before the budget
runs out.**

---

## The Domain Plugin Interface

The `DomainConfig` dataclass is the plugin interface. Implement it, pass it to
`ServiceImpactEnvironment`, and you have a new RL environment:

```python
@dataclass(frozen=True)
class DomainConfig:
    name: str
    description: str
    nodes: list[str]                   # entities in the causal graph
    edges: list[tuple[str, str]]       # dependency relationships
    tool_labels: dict[str, str]        # generic actions → domain vocabulary
    incident_template: str             # prompt template for reset() alert
    node_metadata: dict[str, dict]     # tier, type, has_alt — per node
    incident_archetypes: tuple[str, ...]  # real-world framing variants
```

`tool_labels` maps the generic action vocabulary to domain language. The LLM
simulator generates contextually appropriate output for each domain, while the
agent interface remains unchanged.

---

## Two Domains Shipping Today

### Domain 1 — SRE / Microservices

30-service NetworkX DiGraph across 5 tiers:

```
Tier 1 — Gateway:     api_gateway · mobile_backend · web_backend
Tier 2 — Business:    auth_service · order_service · cart_service · payment_service
Tier 3 — Support:     email_service · analytics_service · logging_service
Tier 4 — Data:        database_service · message_queue · feature_flags
Tier 5 — Infra:       rate_limiter · cache · load_balancer
```

A breaking API change fires a PagerDuty-style P1 alert. The agent must trace the
blast radius through noisy registry CLI outputs, Confluence runbooks, Datadog
monitoring snapshots, and topology diffs.

**A concrete episode:**

```
RESET (seed 42, medium difficulty):
  payment_gateway has a breaking change deployed
  ground_truth = {cart_service, order_service, billing_worker} — SEALED

STEP 1 (free): query_runbook("payment_gateway")
  → "Downstream callers include cart_service and order_service.
     Retry policy: 3 attempts with exponential backoff."
  BeliefTracker: cart_service 0.55, order_service 0.50

STEP 2 (paid, −1 budget): query_dependents("payment_gateway")
  → "cart_service, order_service [WARN: checkout_proxy may also be
     affected — data from 48hr-stale registry snapshot]"
  BeliefTracker: cart_service 0.78, order_service 0.72

STEP 3 (free): query_monitoring("cart_service")
  → "Status: DEGRADED. Latency p99: 4200ms. Error rate: 12%"
  BeliefTracker: cart_service 0.91

MUTATION fires at step 4 (silent):
  billing_worker goes DOWN — ground_truth expands to include it
  Agent has no notification — must detect from tool outputs

STEP 4 (paid, −1 budget): get_service_health("billing_worker")
  → "503 UNAVAILABLE. Last healthy: 4 minutes ago."
  ContradictionEngine fires: runbook said billing_worker was stable
  BeliefTracker: billing_worker rises to 0.83

HYPOTHESIS (free): submit_hypothesis(["cart_service", "order_service",
                                       "billing_worker"], confidence=0.78)
  → Calibration reward computed. Episode continues.

SUBMIT: {cart_service, order_service, billing_worker}
  F-beta (β=2) = 0.94. Ground truth revealed. Episode ends.
```

### Domain 2 — Supply Chain Disruption

30-node supplier/factory/logistics/retail graph. Real-world incident archetypes
drawn from documented disruptions:

| Archetype | Based On |
|---|---|
| Semiconductor fab fire | Renesas Naka factory fire, 2021 |
| Canal blockage | Ever Given / Suez Canal, March 2021 |
| Demand shock | COVID-19 staples surge, 2020 |
| Fab concentration risk | TSMC Taiwan concentration |
| Maritime route disruption | Red Sea shipping attacks, 2023–24 |

The agent traces which distributors, manufacturers, and logistics nodes are
disrupted — same framework, same tool interface, same reward function, different
causal graph.

**Tool vocabulary mapping:**

| Generic action | SRE framing | Supply chain framing |
|---|---|---|
| `query_dependents` | "Which services call X?" | "Which factories source from X?" |
| `query_runbook` | Confluence runbook | Supplier SLA / contingency SOP |
| `query_monitoring` | Datadog snapshot | Inventory level / logistics health |
| `get_service_health` | Real-time health status | Node operational status |
| `submit_hypothesis` | Test partial service set | Test partial disruption set |

The graph engine, World Modeling Layer, reward pipeline, curriculum scheduler,
and GRPO training pipeline are identical across both domains.

---

## What Makes This Hard by Design

**Result arrays are always empty during queries.** The agent sees
`obs.result == []` for every query — it must parse meaning from free-text LLM
prose. This forces genuine comprehension and cross-source synthesis, not JSON
key extraction.

**Noise is LLM-mediated, not scripted.** At Hard difficulty, a registry lookup
for a node with 8 real dependents returns 9 results: 7 correct, 2 hallucinated,
2 silently dropped. The LLM varies phrasing, ordering, and emphasis across
seeds — pattern exploitation fails.

**The graph mutates mid-episode.** `MutationEngine` fires silently between
steps 3 and 5, removing an entity or adding a new dependency. The ground truth
expands. The agent gets no explicit notification — it must infer the topology
shift from tool output changes. `world_version` and `belief_drift` are returned
in the next observation.

**10,000+ unique episodes per difficulty.** Each seed removes 4–9 redundant
edges and adds 3–7 domain-valid edges. The episode space is effectively
unbounded — no memorization is possible.

---

## Reward Design

The reward function is domain-agnostic:

| Condition | Modifier |
|---|:---:|
| New unique node queried | +0.05 |
| Re-querying the same node | −0.05 |
| Budget exhausted without submit | −0.40 |
| Overclaiming (> 60% of all nodes) | −0.30 × oversubmit fraction |
| Terminal `submit` | F-beta(β=2) ∈ [0.001, 0.95] |

F-beta with β=2 weights recall 4× over precision — encoding the real asymmetry
present in every professional domain. Missing a cascading failure is always
worse than a false alarm.

The three-layer reward isolation ensures the agent cannot game individual
components:

- **Layer 1** — 7 sealed atomic functions (F-beta ORM, InformationGain PRM,
  QueryQualityScorer, Novelty, Efficiency, Calibration Brier, Self-correction)
- **Layer 2** — One-way aggregator with rotating weight configs (A–D) per
  seed band. Agent never sees component weights.
- **Layer 3** — Post-episode trajectory auditor. 7 heuristics tag episodes
  LEGITIMATE / SUSPICIOUS / ANOMALOUS. ANOMALOUS episodes are excluded from
  GRPO batches before training.

---

## Baseline Results (SRE Domain)

Baseline agent: Llama-3.3-70B via Cerebras, simple chain-of-thought — read
alert, query dependents, query dependencies, submit.

| Difficulty | Mean F-beta (β=2) | Recall | Precision |
|---|:---:|:---:|:---:|
| Easy | ~0.81 | ~0.89 | ~0.67 |
| Medium | ~0.61 | ~0.72 | ~0.48 |
| Hard | ~0.38 | ~0.51 | ~0.31 |

The Hard gap is the interesting region. The naive BFS strategy saturates at
Medium. Closing the Hard gap requires belief state tracking, hypothesis
management, and strategic query ordering — exactly what the World Modeling Layer
is designed to train.

**[GRPO training results — to be updated after on-site compute run, July 25th]**

---

## OpenEnv Compatibility + GRPO-Ready Trajectories

Standard WebSocket interface:

```python
from cascade_mind import ServiceImpactEnv

async with ServiceImpactEnv(
    base_url="https://rajkamal2819-cascade-mind.hf.space"
) as env:
    obs, _ = await env.reset(seed=42)
    print(obs.message)          # domain-specific incident alert
    print(obs.belief_state)     # {"cart_service": 0.0, ...} — starts empty
```

Every episode writes JSONL:

```json
{
  "episode_id": "ep_42_medium",
  "domain": "sre",
  "changed_service": "payment_gateway",
  "actions": ["query_runbook", "query_dependents", "query_monitoring",
              "get_service_health", "submit_hypothesis", "submit"],
  "final_score": 0.94,
  "belief_trajectory": [0.0, 0.55, 0.78, 0.91, 0.83, 0.83],
  "strategy_tags": ["breadth_first", "hypothesis_driven"]
}
```

GRPO export endpoint:

```python
import requests
from datasets import load_dataset

# Get all episodes where agent scored above 0.5
data = requests.get(
    "https://rajkamal2819-cascade-mind.hf.space/export/grpo?min_reward=0.5"
).json()

dataset = load_dataset("json", data_files=data["path"])
# → ready for GRPOTrainer in HF TRL
```

---

## The Curriculum

Four difficulty tiers, gated by rolling F-beta performance:

| Tier | Nodes | Noise | Budget | Mutation |
|---|:---:|:---:|:---:|:---:|
| Easy | 15 | 15% | 15 | None |
| Medium | 30 | 40% | 12 | 30% prob |
| Hard | 60 | 70% | 9 | 70% prob |
| Ultra | 100+ | 90% | 9 | Always |

Promotion threshold: rolling mean F-beta ≥ 0.75 over 5 episodes.
Demotion threshold: rolling mean F-beta ≤ 0.35 over 5 episodes.

This is adaptive curriculum — the environment escalates when the agent is ready,
recovers when it regresses, and never wastes training compute on problems the
agent has already solved.

---

## Adding Your Own Domain

To build a new Cascade-Mind environment for any knowledge domain:

1. **Define your causal graph** — nodes (entities) + edges (dependencies)
2. **Write `node_metadata`** — tier, type, alternative paths per node
3. **Add `incident_archetypes`** — real or synthetic framing variants
4. **Map your tool vocabulary** in `tool_labels`
5. **Pass your `DomainConfig`** to `ServiceImpactEnvironment`

The LLM simulator, World Modeling Layer, reward function, trajectory logger,
curriculum scheduler, and OpenEnv interface are inherited automatically.

Candidate domains this generalizes to directly: **network security** (attack
graph traversal), **clinical diagnosis** (symptom → disease causal chains),
**software vulnerability** (CVE transitive exposure), **legal discovery**
(entity dependency in document graphs), **financial contagion** (counterparty
exposure tracing).

---

## What We're Building Next

**Self-play scenario generation** — a ScenarioGenerator agent acts as Proposer,
designing maximally hard episodes. Proposer reward = 1 − Solver F-beta. The
environment generates its own curriculum.

**Full GRPO training results** — on-site compute run on July 25th. F-beta
before and after training, CoT-Pass@K, TQS distribution by difficulty.

**Multi-domain benchmark suite** — 1000-seed evaluation harness across SRE,
Supply Chain, and CVE domains with per-difficulty F-beta distributions and
a public HuggingFace leaderboard.

**DomainConfig SDK** — open-source tooling so any researcher can define a new
knowledge-graph domain in a single YAML file and receive a complete OpenEnv
training environment.

---

## Try It

**Interactive playground:**
🚀 [rajkamal2819-cascade-mind.hf.space](https://huggingface.co/spaces/Rajkamal2819/cascade-mind)

Use the **Domain** dropdown to switch between SRE and Supply Chain. Click
**New Episode**, work through the investigation using the available tools,
and submit your predicted affected set. The ground-truth graph reveals on
submission alongside your F-beta score and belief trajectory.

**Links:**
- 📖 [API Docs](https://rajkamal2819-cascade-mind.hf.space/docs)
- 🗺️ [Ground Truth Graph Viewer](https://rajkamal2819-cascade-mind.hf.space/graph/ground-truth?seed=0&difficulty=easy)
- 💻 [GitHub](https://github.com/rajkamal2819/cascade-mind)
- 📓 [GRPO Training Colab](#) ← add your Colab URL here

---

*Built for the Meta × PyTorch OpenEnv Hackathon Finale, Bangalore, July 2026.
Service topology and incident scenarios are synthetic. All LLM-generated
observations are produced by Llama-3.1-8B-Instruct via Cerebras HuggingFace
Inference Providers.*