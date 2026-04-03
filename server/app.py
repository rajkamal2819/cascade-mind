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
    from .service_impact_environment import ServiceImpactEnvironment
except ImportError:
    from models import ServiceImpactAction, ServiceImpactObservation  # type: ignore
    from server.service_impact_environment import ServiceImpactEnvironment  # type: ignore


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
app.title   = "cascade-mind — Service Impact Analysis"
app.version = "0.2.0"
app.description = """
## What is this?

**cascade-mind** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible
reinforcement-learning environment for the **Meta × PyTorch OpenEnv Hackathon**.

An agent acts as an SRE on-call engineer: given an **incident alert** about a changed
microservice, it must identify every downstream service that will be affected — using
only the same noisy, LLM-generated tool output a real engineer would see.

The ground truth is a seeded `networkx` dependency graph of **15 microservices**
across 3 teams (commerce / platform / data).  All tool output (PagerDuty alerts,
registry lookups, runbooks, monitoring snapshots) is generated in real-time by
**Llama-3.1-8B-Instruct via Cerebras**, with calibrated noise on medium/hard difficulty.

---

## Action Space

| `action_type` | Cost | Description |
|---|---|---|
| `query_dependents` | −1 budget | Which services call this service? |
| `query_dependencies` | −1 budget | Which services does this service depend on? |
| `query_runbook` | **FREE** | Confluence-style runbook for a service |
| `query_changelog` | **FREE** | PR/changelog for the incident's changed service |
| `query_monitoring` | **FREE** | Datadog-style monitoring snapshot for a service |
| `submit_affected` | ends episode | Submit your final list of affected services |

**Query budget: 10** — once exhausted, only free actions and `submit_affected` remain.
Re-querying the same service costs −0.05 reward as a penalty.

---

## Scoring

$$
\\text{reward} = F_{\\beta=2}(\\text{predicted}, \\text{ground truth}) - 0.1 \\times |\\text{over-claims}|
$$

- $F_{\\beta=2}$ weights **recall twice as much as precision** (missing an affected service
  is worse than a false positive)
- Score ∈ **[0.0, 1.0]**
- Difficulty levels: `easy` (no noise) · `medium` (±1 service in registry output) ·
  `hard` (±2 services in registry output)

---

## Primary Protocol — WebSocket `/ws`

The WebSocket session is **stateful**: `reset` and all `step` calls share a single
episode context.  HTTP `/reset` + `/step` are also available but each HTTP call is
**stateless** — use WebSocket for full episodes.

```python
import asyncio, json, websockets

async def run_episode():
    async with websockets.connect(
        "wss://rajkamal2819-cascade-mind.hf.space/ws"
    ) as ws:

        # 1 — start episode
        await ws.send(json.dumps({"type": "reset", "data": {"seed": 42}}))
        obs = json.loads(await ws.recv())
        print("ALERT:", obs["data"]["message"])

        # 2 — free intel: read changelog
        await ws.send(json.dumps({
            "type": "step",
            "data": {"action_type": "query_changelog", "service_name": "catalog_service"}
        }))
        print(json.loads(await ws.recv())["data"]["message"])

        # 3 — query dependents (costs 1 budget)
        await ws.send(json.dumps({
            "type": "step",
            "data": {"action_type": "query_dependents", "service_name": "catalog_service"}
        }))
        print(json.loads(await ws.recv())["data"]["message"])

        # 4 — submit answer
        await ws.send(json.dumps({
            "type": "step",
            "data": {
                "action_type": "submit_affected",
                "affected_services": ["api_gateway", "cart_service", "web_backend"]
            }
        }))
        result = json.loads(await ws.recv())
        print("Score:", result["data"]["reward"])

asyncio.run(run_episode())
```

---

## Quick-start with the Python client

```bash
pip install cascade-mind   # or: pip install git+https://github.com/rajkamal2819/cascade-mind
```

```python
from cascade_mind import ServiceImpactEnv

async with ServiceImpactEnv(base_url="https://rajkamal2819-cascade-mind.hf.space") as env:
    obs, _ = await env.reset(seed=0)
    print(obs.message)   # → PagerDuty-style incident alert

    obs, reward, done, _, _ = await env.step(
        {"action_type": "query_dependents", "service_name": "catalog_service"}
    )
    print(obs.message)   # → LLM-generated registry output (possibly noisy)
```

---

## Environment Variables (Space secrets)

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `HF_TOKEN` | ✅ | — | HuggingFace token for Llama-3.1-8B via Cerebras |
| `LLM_SIMULATOR_ENABLED` | — | `true` | Set `false` to use fast template fallbacks |
| `LLM_CACHE_PATH` | — | `/app/llm_sim_cache.json` | Path to pre-warmed response cache |

---

## Links

- 📦 **GitHub**: [rajkamal2819/cascade-mind](https://github.com/rajkamal2819/cascade-mind)
- 🤗 **HF Space**: [Rajkamal2819/cascade-mind](https://huggingface.co/spaces/Rajkamal2819/cascade-mind)
- 📖 **OpenEnv docs**: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
"""

app.openapi_tags = [
    {
        "name": "Environment Control",
        "description": "Start a new episode (`/reset`) or advance it by one step (`/step`). "
                       "**Prefer the WebSocket at `/ws`** for full episodes — HTTP endpoints "
                       "do not share session state between separate requests.",
    },
    {
        "name": "State Management",
        "description": "Inspect the current episode state: changed service, budget remaining, "
                       "services queried so far, ground-truth graph (revealed post-submit).",
    },
    {
        "name": "Environment Info",
        "description": "Environment name, version, action/observation schemas, difficulty levels.",
    },
    {
        "name": "Schema",
        "description": "JSON Schema definitions for `ServiceImpactAction`, "
                       "`ServiceImpactObservation`, and `ServiceImpactState`.",
    },
    {
        "name": "Health",
        "description": "Liveness probe. Returns `{\"status\": \"healthy\"}` when the server "
                       "and LLM simulator are ready.",
    },
]

# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) endpoint — RFC 003 compliance
# Exposes the environment as a tool-callable MCP server.
# GET  /mcp        → tools manifest (list all available tools)
# POST /mcp        → JSON-RPC 2.0 dispatcher (tools/list, tools/call)
# ---------------------------------------------------------------------------
from fastapi import Request
from fastapi.responses import JSONResponse

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
        "description": "Fetch internal Confluence-style runbook for a service. FREE — no budget cost.",
        "inputSchema": {
            "type": "object",
            "properties": {"service_name": {"type": "string"}},
            "required": ["service_name"],
        },
    },
    {
        "name": "query_changelog",
        "description": "Fetch PR/changelog for the changed service. FREE — no budget cost.",
        "inputSchema": {
            "type": "object",
            "properties": {"service_name": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "query_monitoring",
        "description": "Fetch Datadog-style monitoring snapshot for a service. FREE — no budget cost.",
        "inputSchema": {
            "type": "object",
            "properties": {"service_name": {"type": "string"}},
            "required": ["service_name"],
        },
    },
    {
        "name": "submit_affected",
        "description": "Submit final list of affected services. Ends episode, returns F-beta score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "affected_services": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of all affected service names",
                }
            },
            "required": ["affected_services"],
        },
    },
]


@app.get("/mcp")
async def mcp_manifest():
    """MCP tools manifest — lists all available environment tools."""
    return {
        "schema_version": "2024-11-05",
        "name": "service_impact_env",
        "version": "0.2.0",
        "description": (
            "Cross-service impact analysis environment. Agents identify downstream "
            "services affected by a microservice change using LLM-simulated SRE tool output."
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
                    "serverInfo": {"name": "service_impact_env", "version": "0.2.0"},
                },
            })

        if method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0", "id": req_id,
                "result": {"tools": _MCP_TOOLS},
            })

        if method == "tools/call":
            tool_name = body.get("params", {}).get("name", "")
            tool = next((t for t in _MCP_TOOLS if t["name"] == tool_name), None)
            if tool is None:
                return JSONResponse({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
                })
            return JSONResponse({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": (
                            f"Tool '{tool_name}' is available. "
                            f"Execute via WebSocket at /ws using action_type='{tool_name}' "
                            f"(or 'submit' for submit_affected)."
                        ),
                    }]
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
