"""
server/app.py
-------------
FastAPI application entry point for service_impact_env.

Wires the ServiceImpactEnvironment into the OpenEnv HTTP/WebSocket server
using the create_app factory from openenv-core.

The server exposes:
  GET  /health     → {"status": "healthy"}
  GET  /schema     → action + observation + state JSON schemas
  GET  /metadata   → environment name, description, version
  GET  /state      → current episode state (HTTP mode only)
  POST /reset      → start new episode (HTTP mode only)
  POST /step       → execute action (HTTP mode only)
  WS   /ws         → persistent WebSocket session (primary protocol)

Usage:
  uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

# Dual-import: relative works when run from repo root (PYTHONPATH=src:envs),
#              bare imports work inside Docker (/app/env on PYTHONPATH).
try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server import create_app  # type: ignore

try:
    from ..models import ServiceImpactAction, ServiceImpactObservation
    from .env.service_impact_environment import ServiceImpactEnvironment
except ImportError:
    from cascade_mind.models import ServiceImpactAction, ServiceImpactObservation  # type: ignore
    from cascade_mind.server.env.service_impact_environment import ServiceImpactEnvironment  # type: ignore


# ─── Patch openenv-core web_interface for cascade-mind branding ───────────────
# Must happen BEFORE create_app() is called so the web interface picks up changes.
try:
    import openenv.core.env_server.web_interface as _wi

    # 1. Strip YAML frontmatter (--- ... ---) before the README is shown in the
    #    Gradio accordion — the raw README.md starts with HF Space config that
    #    renders as ugly plain text when Gradio displays it as Markdown.
    _orig_load_readme = _wi._load_readme_from_filesystem

    def _patched_load_readme(env_name):  # type: ignore[override]
        content = _orig_load_readme(env_name)
        if content and content.startswith("---"):
            end = content.find("\n---\n", 4)
            if end != -1:
                content = content[end + 5:].lstrip("\n")
        return content

    _wi._load_readme_from_filesystem = _patched_load_readme  # type: ignore[assignment]

    # 2. Replace the generic OpenEnv docs link in the Quick Start accordion with
    #    the cascade-mind Swagger / ReDoc link.
    _wi.DEFAULT_QUICK_START_MARKDOWN = _wi.DEFAULT_QUICK_START_MARKDOWN.replace(
        "For more information, see the [OpenEnv documentation](https://meta-pytorch.org/OpenEnv/).",
        "For more information, see the **[cascade-mind API docs](https://rajkamal2819-cascade-mind.hf.space/docs)**.",
    )
except Exception:
    pass  # Never crash the server over a cosmetic patch
# ─────────────────────────────────────────────────────────────────────────────


# Pass the CLASS (not an instance) so the server creates one environment
# per WebSocket session — required for concurrent session isolation.
# SUPPORTS_CONCURRENT_SESSIONS=True is set on the class, so max_concurrent_envs > 1 is safe.
app = create_app(
    ServiceImpactEnvironment,
    ServiceImpactAction,
    ServiceImpactObservation,
    env_name="service_impact_env",
    max_concurrent_envs=4,
)

# ---------------------------------------------------------------------------
# Override generic openenv-core metadata with cascade-mind–specific docs.
# FastAPI renders full Markdown in Swagger UI / ReDoc.
# ---------------------------------------------------------------------------
app.title   = "cascade-mind — Knowledge-Graph RL Environment"
app.version = "0.3.0"
app.description = """
**cascade-mind** is a domain-agnostic reinforcement learning environment for training agents
that reason over hidden dependency graphs with noisy tools and a limited query budget.

Built on **OpenEnv**, cascade-mind ships with two plug-and-play domains — SRE incident
response (30 microservices) and supply chain disruption (30 supplier nodes) — and a
`DomainConfig` plugin interface that lets you add any knowledge-graph domain in a single
file. Same engine, different world.

---

## What Makes This Environment Different

**World Modeling Layer** — every step updates three persistent signals the agent can read:

| Signal | What it captures |
|---|---|
| `belief_state` | Per-service confidence [0–1] — how likely is each node affected? |
| `graph_prior` | Session-level edge frequencies from prior episodes — institutional memory |
| `contradictions` | Tool outputs that conflict with each other — a sign of noise or topology drift |

**Composable Reward (OpenEnv Rubric)** — `CascadeMindRubric` is a `WeightedSum` of two rubrics:

- **FBetaRubric (80%)** — F-beta(β=2), recall-weighted. Getting the right services matters more than avoiding false positives.
- **BrierScoreRubric (20%)** — calibration reward. Agents are scored on *how confident* they were about each service, not just whether they got the answer right. An agent that's 90% confident on affected services and 10% confident on unaffected ones scores higher than one that's 100% confident on everything.

**Anti-Gaming Architecture** — three mechanisms prevent reward hacking:
- `MutationEngine` silently drifts graph topology mid-episode
- `CurriculumScheduler` gates harder difficulty behind F-beta thresholds
- `OverclaimGuard` penalises agents that flag more than 60% of the graph

---

## How an Episode Works

1. **Reset** — a scenario is seeded: one service/node has changed, an incident alert fires (generated by Llama-3.1-8B via Cerebras).
2. **Investigate** — use the tools below to explore the dependency graph. Registry queries cost budget; intelligence tools are free.
3. **Submit** — declare the full blast radius. The episode ends, the ground truth is revealed, and a composite F-beta + Brier score is returned.

Difficulty controls registry noise: `easy` → clean · `medium` → ±1 edge · `hard` → ±2 edges.

---

## Available Tools

| Tool | Budget cost | What it returns |
|---|---|---|
| `query_dependents` | 1 | Nodes that directly depend on the queried node |
| `query_dependencies` | 1 | Nodes the queried node depends on |
| `submit_hypothesis` | 1 | Partial F-beta score without ending the episode — useful for mid-episode course correction |
| `query_runbook` | Free (cap 2) | Internal runbook: ownership, SLOs, known failure modes |
| `query_changelog` | Free (cap 2) | Recent PR/deployment changelog for the changed node |
| `query_monitoring` | Free (cap 3) | Live monitoring snapshot: latency, error rate, dependency health |
| `query_topology_diff` | Free | All topology mutations injected since episode start |
| `query_service_health` | Free | Health summary for a node: tier, team, degree, investigation status |
| `submit` | — | Submit final blast radius — ends episode, triggers full scoring |

Budget is difficulty-adaptive (easy: 15 · medium: 10 · hard: 7). Re-querying the same node costs budget but returns the same response.

---

## Connecting via WebSocket

```python
import asyncio, json, websockets

async def run():
    async with websockets.connect("wss://rajkamal2819-cascade-mind.hf.space/ws") as ws:

        # Start a new episode
        await ws.send(json.dumps({"type": "reset", "data": {"seed": 42}}))
        obs = json.loads(await ws.recv())
        print(obs["data"]["message"])          # → incident alert + graph prior hint

        # Check the changelog (free)
        await ws.send(json.dumps({
            "type": "step",
            "data": {"action_type": "query_changelog", "service_name": "catalog_service"}
        }))
        print(json.loads(await ws.recv())["data"]["message"])

        # Query dependents (costs 1 budget)
        await ws.send(json.dumps({
            "type": "step",
            "data": {"action_type": "query_dependents", "service_name": "catalog_service"}
        }))
        result = json.loads(await ws.recv())
        print(result["data"]["belief_state"])   # → per-service confidence map

        # Submit final answer
        await ws.send(json.dumps({
            "type": "step",
            "data": {
                "action_type": "submit",
                "affected_services": ["api_gateway", "cart_service", "web_backend"]
            }
        }))
        result = json.loads(await ws.recv())
        print("Score:", result["data"]["reward"])   # → float in [0.001, 0.999]

asyncio.run(run())
```

---

## Python Client

```python
from cascade_mind import ServiceImpactEnv

async with ServiceImpactEnv(base_url="https://rajkamal2819-cascade-mind.hf.space") as env:
    obs, _ = await env.reset(seed=0)
    print(obs.message)          # incident alert
    print(obs.graph_prior)      # edge frequencies from prior episodes

    obs, reward, done, _, _ = await env.step(
        {"action_type": "query_dependents", "service_name": "catalog_service"}
    )
    print(obs.belief_state)     # updated per-service confidence
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | HuggingFace token — used to call Llama-3.1-8B via Cerebras |
| `LLM_SIMULATOR_ENABLED` | `true` | Set to `false` for deterministic template responses |
| `LLM_CACHE_PATH` | `/app/llm_sim_cache.json` | Pre-warmed response cache for faster startup |

---

## Resources

- [GitHub — rajkamal2819/cascade-mind](https://github.com/rajkamal2819/cascade-mind)
- [HuggingFace Space](https://huggingface.co/spaces/Rajkamal2819/cascade-mind)
- [OpenEnv framework](https://github.com/meta-pytorch/OpenEnv)
"""

app.openapi_tags = [
    {
        "name": "Environment Control",
        "description": "Start a new episode with `/reset`, then advance it step by step with `/step`. "
                       "For full episodes, prefer the **WebSocket at `/ws`** — HTTP endpoints are "
                       "stateless and do not share session context between calls.",
    },
    {
        "name": "State Management",
        "description": "Read the current episode state at any point: which service changed, "
                       "how many registry queries remain, which services have been explored, "
                       "and (after submission) the ground-truth affected set.",
    },
    {
        "name": "Environment Info",
        "description": "Metadata about the environment: name, version, supported difficulty levels, "
                       "and descriptions of all available action types.",
    },
    {
        "name": "Schema",
        "description": "JSON Schema definitions for `ServiceImpactAction`, "
                       "`ServiceImpactObservation`, and `ServiceImpactState` — useful for "
                       "building typed clients.",
    },
    {
        "name": "Health",
        "description": "Returns `{\"status\": \"healthy\"}` when the server is ready to accept connections.",
    },
]

# ---------------------------------------------------------------------------
# Ground-truth graph — must be registered before the Gradio catch-all mount
# ---------------------------------------------------------------------------
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse

@app.get("/graph/ground-truth", include_in_schema=False)
async def ground_truth_graph(seed: int = 0, difficulty: str = "easy"):
    """Serve a standalone interactive vis.js page of the full ground-truth graph."""
    try:
        from .ui.playground import build_ground_truth_html
    except ImportError:
        from cascade_mind.server.ui.playground import build_ground_truth_html  # type: ignore
    html = build_ground_truth_html(seed=seed, difficulty=difficulty)
    return HTMLResponse(content=html)


# ── World Modeling API routes (v3) ─────────────────────────────────────────

@app.get("/belief", tags=["world-model"])
async def get_belief_state():
    """Return the current belief state (per-service confidence) for the active episode."""
    env = _get_env()
    if env is None:
        return JSONResponse({"error": "No active environment"}, status_code=503)
    bt = getattr(env, "_belief_tracker", None)
    if bt is None:
        return JSONResponse({"belief_state": {}, "note": "BeliefTracker not available"})
    return JSONResponse({
        "belief_state": bt.belief,
        "world_version": getattr(env, "_world_version", 0),
    })


@app.get("/prior", tags=["world-model"])
async def get_graph_prior():
    """Return the session-level graph prior (edge confidence from previous episodes)."""
    env = _get_env()
    if env is None:
        return JSONResponse({"error": "No active environment"}, status_code=503)
    gp = getattr(env, "_graph_prior", None)
    if gp is None:
        return JSONResponse({"prior": {}, "episodes": 0, "note": "GraphPrior not available"})
    return JSONResponse({
        "prior": gp.get_prior(),
        "episodes": gp.episode_count,
        "top_10": gp.top_k(10),
    })


@app.get("/contradictions", tags=["world-model"])
async def get_contradictions():
    """Return all tool contradictions detected in the current episode."""
    env = _get_env()
    if env is None:
        return JSONResponse({"error": "No active environment"}, status_code=503)
    ce = getattr(env, "_contradiction_engine", None)
    if ce is None:
        return JSONResponse({"contradictions": [], "count": 0, "note": "ContradictionEngine not available"})
    return JSONResponse({
        "contradictions": ce.descriptions(),
        "count": ce.count,
    })


@app.get("/export/grpo", tags=["world-model"])
async def export_grpo(min_reward: float = 0.0, output: str = "/tmp/grpo_export.jsonl"):
    """Export completed trajectories as a GRPO training JSONL file."""
    import os
    try:
        from .trajectory.trajectory_auditor import TrajectoryAuditor
    except ImportError:
        from cascade_mind.server.trajectory.trajectory_auditor import TrajectoryAuditor  # type: ignore
    trajectory_dir = os.environ.get("TRAJECTORY_DIR", "/tmp/cascade_trajectories")
    auditor = TrajectoryAuditor(trajectory_dir)
    count = auditor.export_grpo_jsonl(output_path=output, min_reward=min_reward)
    return JSONResponse({
        "exported": count,
        "output_path": output,
        "min_reward": min_reward,
    })


def _get_env():
    """Retrieve the running ServiceImpactEnvironment instance from the app state."""
    return getattr(app.state, "env", None)

# ---------------------------------------------------------------------------
# Evict any default openenv /web or / Gradio route so ours takes priority
# ---------------------------------------------------------------------------
_evict_paths = {"/", "/web"}
app.routes[:] = [r for r in app.routes if not (hasattr(r, "path") and r.path in _evict_paths)]

# HF Spaces / openenv hits /web first — must be BEFORE the Gradio catch-all mount
@app.get("/web", include_in_schema=False)
async def web_redirect():
    return RedirectResponse(url="/")

@app.get("/web/", include_in_schema=False)
async def web_slash_redirect():
    return RedirectResponse(url="/")

# ---------------------------------------------------------------------------
# Playground — custom Gradio 6 interactive UI at /
# ---------------------------------------------------------------------------
try:
    import gradio as gr
    try:
        from .ui.playground import playground_blocks, PLAYGROUND_CSS, PLAYGROUND_THEME
    except ImportError:
        from cascade_mind.server.ui.playground import playground_blocks, PLAYGROUND_CSS, PLAYGROUND_THEME  # type: ignore
    app = gr.mount_gradio_app(
        app, playground_blocks, path="/",
        css=PLAYGROUND_CSS, theme=PLAYGROUND_THEME,
    )
except Exception as _pg_exc:
    import warnings as _w
    _w.warn(f"cascade-mind: playground mount failed -- {_pg_exc}", stacklevel=1)

# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) endpoint — RFC 003 compliance
# Exposes the environment as a tool-callable MCP server.
# GET  /mcp        → tools manifest (list all available tools)
# POST /mcp        → JSON-RPC 2.0 dispatcher (tools/list, tools/call)
# ---------------------------------------------------------------------------

_MCP_TOOLS = [
    {
        "name": "query_dependents",
        "description": "Find services that directly call/depend on a given service. Uses query budget.",
        "inputSchema": {
            "type": "object",
            "properties": {"service_name": {"type": "string", "description": "Service to query"}},
            "required": ["service_name"],
        },
    },
    {
        "name": "query_dependencies",
        "description": "Find services that a given service directly depends on. Uses query budget.",
        "inputSchema": {
            "type": "object",
            "properties": {"service_name": {"type": "string", "description": "Service to query"}},
            "required": ["service_name"],
        },
    },
    {
        "name": "query_runbook",
        "description": "Fetch the internal runbook for a service: ownership, SLOs, and known failure modes.",
        "inputSchema": {
            "type": "object",
            "properties": {"service_name": {"type": "string"}},
            "required": ["service_name"],
        },
    },
    {
        "name": "query_changelog",
        "description": "Fetch the recent PR and deployment changelog for the changed service.",
        "inputSchema": {
            "type": "object",
            "properties": {"service_name": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "query_monitoring",
        "description": "Fetch a live monitoring snapshot for a service: latency, error rate, and dependency health.",
        "inputSchema": {
            "type": "object",
            "properties": {"service_name": {"type": "string"}},
            "required": ["service_name"],
        },
    },
    {
        "name": "submit",
        "description": "Submit the final list of affected services. Ends the episode and returns an F-beta(β=2) score against the ground truth.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "affected_services": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All service names you believe are affected by the incident.",
                }
            },
            "required": ["affected_services"],
        },
    },
    {
        "name": "submit_hypothesis",
        "description": "Submit a hypothesis for partial scoring without ending the episode. Returns a delayed_reward (partial F-beta) so the agent can adjust its strategy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "affected_services": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Services you currently believe are affected.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Your confidence in this hypothesis (0.0-1.0).",
                },
            },
            "required": ["affected_services"],
        },
    },
    {
        "name": "query_topology_diff",
        "description": "Show all topology changes (mutations) that occurred during this episode. FREE action — no budget cost.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "query_service_health",
        "description": "Get health summary for a service: tier, team, degree, investigation status. FREE action — no budget cost.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "The service to get health info for.",
                },
            },
            "required": ["service_name"],
        },
    },
]


@app.get("/mcp")
async def mcp_manifest():
    """MCP tools manifest — lists all available environment tools."""
    return {
        "schema_version": "2024-11-05",
        "name": "cascade_mind",
        "version": "0.3.0",
        "description": (
            "Domain-agnostic RL environment for hidden dependency graph reasoning. "
            "Agents investigate blast radius with noisy tools, a query budget, and a "
            "composable OpenEnv rubric (F-beta + Brier calibration). "
            "Supports SRE and supply-chain domains out of the box via DomainConfig plugin."
        ),
        "tools": _MCP_TOOLS,
        "execution": "Use the WebSocket at /ws for stateful episode execution.",
    }


@app.post("/mcp")
async def mcp_rpc(request: Request):
    """JSON-RPC 2.0 endpoint for MCP protocol compliance.

    Supports:
      tools/list  → returns full tools manifest
      tools/call  → returns tool schema (actual execution via WebSocket)
    """
    try:
        body   = await request.json()
        method = body.get("method", "")
        req_id = body.get("id", 1)

        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "cascade_mind", "version": "0.3.0"},
                },
            })

        if method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0", "id": req_id,
                "result": {"tools": _MCP_TOOLS},
            })

        if method == "tools/call":
            params    = body.get("params", {})
            tool_name = params.get("name", "")
            tool = next((t for t in _MCP_TOOLS if t["name"] == tool_name), None)
            if tool is None:
                return JSONResponse({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
                })

            # ── Execute the tool against a real environment instance ──
            arguments = params.get("arguments", {})
            env = ServiceImpactEnvironment()

            # Auto-reset if no episode is active (MCP callers may not reset first)
            if not env._changed_service:
                seed = arguments.pop("seed", 42)
                env.reset(seed=seed)

            # Build action from MCP arguments
            action_data = {"action_type": tool_name, **arguments}
            try:
                action = ServiceImpactAction(**action_data)
            except Exception as exc:
                return JSONResponse({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32602, "message": f"Invalid params: {exc}"},
                })

            obs = env.step(action)
            return JSONResponse({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": obs.message},
                    ],
                    "isError": False,
                },
            })

        return JSONResponse({
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })

    except Exception as exc:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}},
            status_code=400,
        )


def main() -> None:
    """Entry point for `uv run --project . server`."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
