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
