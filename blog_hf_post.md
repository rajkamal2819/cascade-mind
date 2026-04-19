# Cascade-Mind: A World Modeling RL Framework for Causally-Dependent Knowledge Domains

*An OpenEnv-compatible framework for building RL environments where an agent must construct and maintain a causal model of a hidden dependency graph using noisy, contradictory tools — applicable to any professional domain where knowledge has graph structure.*

---

## The Gap This Fills

Most RL benchmarks for reasoning give the agent a **complete, clean world state** and ask it to plan. Real professional reasoning doesn't work like that.

An on-call engineer doesn't see the dependency graph. A supply-chain analyst doesn't see the full disruption tree. A security researcher doesn't see the attack surface. They all see the same thing: **noisy, contradictory tool outputs** from which they must reconstruct a hidden causal structure — under time pressure, with incomplete information.

**Cascade-Mind** is a framework for turning any domain with graph-structured knowledge into an RL training environment that captures this challenge.

---

## What Cascade-Mind Is

Cascade-Mind is not a single RL environment. It is a **generator of environments** — a framework with three layers and a plugin interface that lets you swap the underlying knowledge domain without touching the agent interface, reward function, or training pipeline.

> One OpenEnv environment. One `DomainConfig`. One `reset()` / `step()` / `score()` loop.
> Change the domain plugin → new environment, new causal graph, new incident archetypes.
> The framework stays identical.

This is the distinction between a benchmark and an infrastructure. Cascade-Mind is infrastructure.

---

## The Core Abstraction

Every Cascade-Mind environment is defined by three layers:

```
┌─────────────────────────────────────────────────────────┐
│                  GROUND TRUTH LAYER                     │
│  Hidden causal graph (NetworkX DiGraph)                 │
│  Seeded + perturbed per episode                         │
│  Agent never sees this — scorer only                    │
└───────────────────────┬─────────────────────────────────┘
                        │ hidden structure
                        ▼
┌─────────────────────────────────────────────────────────┐
│               LLM SIMULATOR LAYER                       │
│  Llama-3.1-8B-Instruct · Cerebras · HF Providers        │
│                                                         │
│  reset()  → domain-specific incident framing            │
│  query_*  → noisy tool outputs (registry, runbook,      │
│             monitoring, topology diff, health check)    │
│                                                         │
│  Noise: easy 0.15 · medium 0.40 · hard 0.70            │
└───────────────────────┬─────────────────────────────────┘
                        │ LLM text — noisy, sometimes wrong
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   AGENT LAYER                           │
│  Sees only: observation.message (free text)             │
│  result[] always [] during queries (must parse prose)   │
│  Budget: 5 / 8 / 12  ·  Goal: submit correct set        │
└─────────────────────────────────────────────────────────┘
```

The agent's task is the same regardless of domain: **reconstruct the hidden causal graph from noisy observations and submit the correct affected set before the budget runs out.**

---

## The Domain Plugin Interface

The `DomainConfig` dataclass is the plugin interface. Implement it, pass it to `ServiceImpactEnvironment`, and you have a new RL environment:

```python
@dataclass(frozen=True)
class DomainConfig:
    name: str
    description: str
    nodes: list[str]                          # entities in the graph
    edges: list[tuple[str, str]]              # causal dependencies
    tool_labels: dict[str, str]               # maps generic actions → domain vocabulary
    incident_template: str                    # prompt template for reset() alert
    node_metadata: dict[str, dict]            # tier, type, has_alt — per node
    incident_archetypes: tuple[str, ...]      # real-world framing variants
```

`tool_labels` is the key mechanism. It maps the generic action vocabulary (`query_dependents`, `query_runbook`, `submit`) to domain-specific language — so the LLM simulator generates contextually appropriate output for each domain, while the agent interface remains unchanged.

---

## Two Domains Shipping Today

### 🖥️ SRE / Microservices

30-service NetworkX DiGraph across 5 tiers (Gateway → Business → Support → Data → Infra). A breaking API change fires a PagerDuty-style P1 alert. The agent must trace the blast radius through noisy registry CLI outputs, Confluence runbooks, Datadog monitoring snapshots, and topology diffs.

**Used for GRPO training.** Every episode writes a JSONL trajectory — action history, reward, discovery timeline, strategy tags — ready for HF TRL.

```
Tier 1 — Gateway:    api_gateway · mobile_backend · web_backend
Tier 2 — Business:   auth_service · order_service · cart_service · payment_service ...
Tier 3 — Support:    email_service · analytics_service · logging_service ...
Tier 4 — Data/Infra: database_service · message_queue · feature_flags · rate_limiter
```

### 🚢 Supply-Chain Disruption

30-node supplier/factory/logistics/retail graph. Real-world incident archetypes drawn from documented disruptions:

| Archetype | Based On |
|---|---|
| Semiconductor fab fire | Renesas Naka factory fire, 2021 |
| Canal blockage | Ever Given / Suez Canal, March 2021 |
| Demand shock + panic buying | COVID-19 staples surge, 2020 |
| Geopolitical fab concentration | TSMC Taiwan concentration risk |
| Maritime route disruption | Red Sea / Houthi shipping attacks, 2023–24 |

The agent traces which distributors, manufacturers, and logistics nodes are disrupted — same framework, same tool interface, different causal graph and incident language.

---

## What Makes This Hard (By Design)

### LLM as World Model, Not Just Agent

This is the first OpenEnv framework where **the world itself speaks through an LLM**. Every tool output — every runbook, every registry response, every monitoring snapshot — is a fresh Llama-3.1-8B generation.

Scripted observations can be memorized. LLM-generated observations vary in phrasing, ordering, and emphasis. An agent that has seen seed 42 cannot reliably exploit that at seed 43. The generalization surface is real.

### Result Arrays Are Always Empty During Queries

```python
obs.result  # always [] during queries — agent must parse obs.message
```

This is intentional. Forcing agents to extract structured information from free-text LLM prose requires genuine comprehension and cross-source synthesis — not JSON key extraction. This mirrors real professional reasoning.

### Noise Is LLM-Mediated

At Hard difficulty, a registry lookup for a node with 8 real dependents returns 9 results: 7 correct, 2 hallucinated, 2 silently dropped. The agent must triangulate across tool directions, runbook tables, and monitoring data. Pattern exploitation fails here.

### 10,000+ Unique Episodes Per Difficulty

Each seed perturbs the graph deterministically:
- Remove 4–9 redundant edges (structurally safe to remove)
- Add 3–7 new edges from domain-valid candidates

`seed % 3` cycles difficulty for balanced evaluation. The episode space is effectively unbounded.

---

## Reward Design

The reward function is domain-agnostic. F-beta with β=2 weights recall 4× over precision — encoding the real cost asymmetry present in every professional domain: missing a cascading failure is always worse than a false alarm.

| Condition | Modifier |
|---|:---:|
| New unique node queried | +0.05 |
| Re-querying the same node | −0.05 |
| Budget exhausted without submit | −0.40 |
| Overclaiming (> 60% of all nodes) | −0.3 × oversubmit fraction |
| Terminal `submit` | F-beta(β=2) ∈ [0.0, 1.0] |

---

## OpenEnv Compatibility + GRPO-Ready Trajectories

Cascade-Mind is fully OpenEnv-compatible. The agent interface is standard WebSocket:

```python
from cascade_mind import ServiceImpactEnv

async with ServiceImpactEnv(base_url="https://rajkamal2819-cascade-mind.hf.space") as env:
    obs, _ = await env.reset(seed=42)
    print(obs.message)   # domain-specific incident alert
```

Every episode is logged as JSONL:

```json
{
  "episode_id": "ep_42_hard",
  "domain": "sre",
  "changed_service": "catalog_service",
  "actions": ["query_dependents", "query_runbook", "submit"],
  "final_score": 0.74,
  "strategy_tags": ["breadth_first", "runbook_heavy"]
}
```

These trajectories feed directly into GRPO via HF TRL. The SRE domain is the primary training domain — the supply-chain domain demonstrates the framework's generalization, but GRPO training runs on SRE episodes only, keeping the training signal clean and reproducible.

---

## The Action Space (Domain-Agnostic)

| Action | Cost | SRE Framing | Supply-Chain Framing |
|---|:---:|---|---|
| `query_dependents` | −1 | "Which services call X?" | "Which nodes depend on X?" |
| `query_dependencies` | −1 | "What does X depend on?" | "What does X source from?" |
| `query_runbook` | free | Confluence runbook | Supplier SLA / contingency doc |
| `query_changelog` | free | Recent PR / deploy log | Recent disruption log |
| `query_monitoring` | free | Datadog snapshot | Inventory / logistics health |
| `query_service_health` | free | Real-time health status | Node operational status |
| `query_topology_diff` | free | Recent topology changes | Recent network changes |
| `submit_hypothesis` | −1 | Test partial hypothesis | Test partial affected set |
| `submit` | terminal | Final affected services | Final disrupted nodes |

The vocabulary is domain-translated by `tool_labels` at the `DomainConfig` level. The agent sees generic action names. The LLM simulator receives domain-appropriate prompts.

---

## Baseline Results (SRE Domain)

Baseline agent: **Llama-3.3-70B via Cerebras**, simple chain-of-thought — read alert, query dependents, query dependencies, submit.

| Difficulty | Mean F-beta (β=2) | Recall | Precision |
|---|:---:|:---:|:---:|
| Easy | ~0.81 | ~0.89 | ~0.67 |
| Medium | ~0.61 | ~0.72 | ~0.48 |
| Hard | ~0.38 | ~0.51 | ~0.31 |

The Hard gap is the interesting region. The naive strategy saturates at Medium. Closing the Hard gap requires belief state tracking, hypothesis management, and strategic query ordering — which is exactly what we're building next.

---

## What We're Building Next

**Belief State Tracking** — explicit probability distributions over the affected set, updated with each query. The agent sees `belief_state: dict[str, float]` as part of every observation.

**Information Gain Metrics** — per-action IG scored against the belief posterior. Actions that resolve high-uncertainty nodes are rewarded with a process signal, not just terminal F-beta.

**Process Reward Model (PRM)** — dense mid-episode rewards for strategic query ordering, source reconciliation, and contradiction detection. This unlocks GRPO training on process quality, not just outcome.

**Contradiction Engine** — detect and surface when two tool outputs contradict each other. High contradiction count becomes a signal that the agent is in a noisy region of the graph.

**Graph Prior** — learned prior over likely dependency structures, updated from trajectory history. Starts uniform, converges toward domain-typical topologies over training.

The trajectory logger and auditor are already live. Every episode writes strategy tags (`breadth_first`, `runbook_heavy`, `hypothesis_driven`) derived from action sequence analysis — ready for the training pipeline the moment the PRM is wired in.

---

## Adding Your Own Domain

To build a new Cascade-Mind environment for your knowledge domain:

1. **Define your causal graph** — nodes (entities) + edges (dependencies)
2. **Write `node_metadata`** — tier, type, alternative paths per node
3. **Add `incident_archetypes`** — real or synthetic framing variants for `reset()`
4. **Map your tool vocabulary** in `tool_labels`
5. **Pass your `DomainConfig`** to `ServiceImpactEnvironment`

The LLM simulator, reward function, trajectory logger, and OpenEnv interface are inherited automatically. You get a fully functioning RL environment with noisy observations, F-beta scoring, and JSONL trajectory export — for whatever knowledge graph domain you care about.

Candidate domains this generalizes to: **network security** (attack graph traversal), **clinical diagnosis** (symptom → disease causal chains), **legal discovery** (entity dependency in document graphs), **financial contagion** (counterparty exposure tracing).

---

## Try It

**Interactive Playground** — [rajkamal2819-cascade-mind.hf.space](https://huggingface.co/spaces/Rajkamal2819/cascade-mind)

Use the **Domain** dropdown to switch between SRE and Supply-Chain. Click **New Episode**, work through the investigation, submit your affected set, see the ground-truth graph reveal and your F-beta score.

**Links:**
- 🚀 [Live Space](https://huggingface.co/spaces/Rajkamal2819/cascade-mind)
- 📖 [API Docs](https://rajkamal2819-cascade-mind.hf.space/docs)
- 🗺️ [Ground Truth Graph](https://rajkamal2819-cascade-mind.hf.space/graph/ground-truth?seed=0&difficulty=easy)
- 💻 [GitHub](https://github.com/rajkamal2819/cascade-mind)

---

*Built for the Meta × PyTorch OpenEnv Hackathon. Service topology and incident scenarios are synthetic. All LLM-generated observations are produced by Llama-3.1-8B-Instruct via Cerebras HuggingFace Inference Providers.*
