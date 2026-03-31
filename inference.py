"""
inference.py
------------
Baseline inference script for service_impact_env.

Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks
(easy, medium, hard) and reports reproducible F1 scores.

Environment variables required:
  OPENAI_API_KEY   — API key (also used as HF_TOKEN if set)
  API_BASE_URL     — API endpoint (default: https://api.openai.com/v1)
  MODEL_NAME       — model to use (default: gpt-4o-mini)
  ENV_BASE_URL     — running environment URL (default: http://localhost:8000)

Usage:
  export OPENAI_API_KEY=sk-...
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  export ENV_BASE_URL=http://localhost:8000
  python inference.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Imports — dual-path for in-repo vs installed package
# ---------------------------------------------------------------------------
try:
    from openai import AsyncOpenAI
except ImportError:
    raise SystemExit(
        "openai package not found. Install with: pip install openai"
    )

try:
    from service_impact_env import ServiceImpactAction, ServiceImpactEnv
except ImportError:
    try:
        from models import ServiceImpactAction  # type: ignore
        from client import ServiceImpactEnv  # type: ignore
    except ImportError:
        raise SystemExit(
            "Cannot import ServiceImpactAction/ServiceImpactEnv. "
            "Run from the service_impact_env directory or install the package."
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY      = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# Task seeds: 0 → easy, 1 → medium, 2 → hard  (seed % 3 determines difficulty)
TASK_SEEDS = {
    "easy":   0,
    "medium": 1,
    "hard":   2,
}

MAX_STEPS_PER_EPISODE = 30

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert SRE (Site Reliability Engineer) specializing in microservice
dependency analysis and incident impact assessment.

CONTEXT:
A microservice has just been changed (API update, schema migration, config change).
You must identify ALL downstream services that will be AFFECTED by this change.
A service X is "affected" if it (directly or transitively) depends on the changed service.

IMPORTANT: You are operating in a realistic SRE environment.
- Responses from service registries are NOISY — they may omit real services or include stale entries.
- Runbooks and changelogs may have outdated information or use confusing jargon.
- You must reason critically from imperfect information, just like a real SRE.
- Cross-reference multiple sources to build confidence in your answer.

AVAILABLE ACTIONS (respond with exactly one JSON object per turn):

BUDGET-CONSUMING actions (each deducts 1 from your query budget):
1. Find services that CALL a specific service (upstream dependents):
   {"action_type": "query_dependents", "service_name": "<service>"}

2. Find what a service DEPENDS ON (its downstream dependencies):
   {"action_type": "query_dependencies", "service_name": "<service>"}

FREE actions (no budget cost — use freely):
3. Read internal runbook/documentation for a service:
   {"action_type": "query_runbook", "service_name": "<service>"}

4. Read the changelog/PR for the changed service:
   {"action_type": "query_changelog", "service_name": "<any service>"}

5. View monitoring dashboard for a service (Datadog-style metrics):
   {"action_type": "query_monitoring", "service_name": "<service>"}

TERMINAL action:
6. Submit your final answer when confident (ends the episode):
   {"action_type": "submit", "affected_services": ["svc_a", "svc_b", ...]}

STRATEGY:
1. Start with FREE actions on the changed service:
   - query_changelog to understand WHAT changed and WHY
   - query_runbook on the changed service for its known consumers
   - query_monitoring to see which services are actively calling it

2. Use budget actions for BFS traversal:
   - query_dependents on the changed service → find direct callers
   - query_dependents on each caller → find their callers
   - Continue until no new services are discovered

3. Cross-reference noisy results:
   - If registry says service X calls the changed service, verify with query_runbook(X)
   - If a service appears in runbook but not registry, include it (registry may be stale)

4. Submit when confident OR budget is low (< 3 queries remaining):
   - Submit with ALL services you have evidence for
   - Missing a real affected service (false negative) is worse than a false positive

TEXT PARSING: The response text may list services like "- auth_service (team: platform)"
or "→ cart_service (status: active)". Extract the service names from the text.

IMPORTANT:
- result[] field will be EMPTY for all query actions — information is in message text only.
- result[] is only populated on submit (reveals the ground truth for review).
- Respond with ONLY a single valid JSON object. No explanation, no markdown.
- The "affected_services" list should NOT include the changed service itself.
- If budget < 3, SUBMIT immediately with everything you've found.
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from model output."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # First bare JSON object
    m = re.search(r"(\{[^{}]*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def format_score_bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.3f}"


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    client: AsyncOpenAI,
    env_url: str,
    seed: int,
    task_name: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one complete episode and return results dict."""

    async with ServiceImpactEnv(base_url=env_url) as env:
        # ── Reset ──────────────────────────────────────────────────────
        reset_result = await env.reset(seed=seed)
        obs = reset_result.observation

        if verbose:
            print(f"\n{'─'*65}")
            print(f"  TASK: {task_name.upper():<8}  |  seed={seed}")
            print(f"  Changed service : {obs.changed_service}")
            print(f"  Budget          : {obs.queries_remaining} queries")
            print(f"{'─'*65}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"The service '{obs.changed_service}' has just been changed.\n"
                    f"Find ALL downstream services that are affected by this change.\n"
                    f"Budget: {obs.queries_remaining} queries available.\n\n"
                    f"Begin exploring now."
                ),
            },
        ]

        result = reset_result
        steps_taken = 0
        queries_used = 0
        parse_errors = 0
        start_time = time.time()

        # ── Agent loop ─────────────────────────────────────────────────
        for step_num in range(MAX_STEPS_PER_EPISODE):
            if result.done:
                break

            # LLM call
            try:
                completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256,
                )
                raw = completion.choices[0].message.content or ""
            except Exception as e:
                if verbose:
                    print(f"  [Step {step_num+1}] LLM error: {e}")
                break

            messages.append({"role": "assistant", "content": raw})

            if verbose:
                print(f"\n  [Step {step_num+1}] Agent → {raw.strip()[:120]}")

            # Parse action
            action_data = extract_json(raw)
            if action_data is None:
                parse_errors += 1
                err_msg = (
                    "ERROR: Could not parse JSON. "
                    "Respond with ONLY a raw JSON object like: "
                    '{"action_type": "query_dependents", "service_name": "auth_service"}'
                )
                messages.append({"role": "user", "content": err_msg})
                if parse_errors >= 3:
                    # Force submit if too many parse errors
                    action_data = {"action_type": "submit", "affected_services": []}
                else:
                    continue

            try:
                action = ServiceImpactAction(**action_data)
            except Exception as e:
                parse_errors += 1
                messages.append({
                    "role": "user",
                    "content": f"ERROR: Invalid action — {e}. Check field names.",
                })
                continue

            # Execute in environment
            result = await env.step(action)
            obs = result.observation
            steps_taken += 1

            if action.action_type != "submit":
                queries_used += 1

            if verbose:
                msg_preview = obs.message[:150]
                print(f"  [Step {step_num+1}] Env   → {msg_preview}")
                if not result.done:
                    print(f"             Queries left: {obs.queries_remaining}")

            # Feed back to model
            if not result.done:
                feedback = (
                    f"Result: {obs.message}\n"
                    f"Queries remaining: {obs.queries_remaining}"
                )
            else:
                feedback = f"Episode ended. {obs.message}"

            messages.append({"role": "user", "content": feedback})

        elapsed = time.time() - start_time
        reward = result.reward if result.reward is not None else 0.0

        if verbose:
            print(f"\n  ┌─ RESULT {'─'*50}")
            print(f"  │  Task       : {task_name}")
            print(f"  │  Reward (F1): {format_score_bar(reward)}")
            print(f"  │  Steps      : {steps_taken}  |  Queries: {queries_used}")
            print(f"  │  Time       : {elapsed:.1f}s")
            print(f"  └{'─'*58}")

        return {
            "task": task_name,
            "seed": seed,
            "reward": reward,
            "steps": steps_taken,
            "queries_used": queries_used,
            "elapsed_s": round(elapsed, 2),
            "parse_errors": parse_errors,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 65)
    print("  service_impact_env — Baseline Inference Script")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  API Base  : {API_BASE_URL}")
    print(f"  Env URL   : {ENV_BASE_URL}")
    print("=" * 65)

    if not API_KEY:
        raise SystemExit(
            "ERROR: OPENAI_API_KEY (or HF_TOKEN) environment variable not set."
        )

    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    results: List[Dict[str, Any]] = []

    for task_name, seed in TASK_SEEDS.items():
        result = await run_episode(
            client=client,
            env_url=ENV_BASE_URL,
            seed=seed,
            task_name=task_name,
            verbose=True,
        )
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  FINAL SCORES")
    print(f"{'═'*65}")
    total_reward = 0.0
    for r in results:
        bar = format_score_bar(r["reward"])
        flag = "✓" if r["reward"] >= 0.5 else "✗"
        print(f"  {flag}  {r['task']:<8}  {bar}  ({r['elapsed_s']}s)")
        total_reward += r["reward"]

    mean_reward = total_reward / len(results)
    print(f"{'─'*65}")
    print(f"     MEAN F1  {format_score_bar(mean_reward)}")
    print(f"{'═'*65}\n")

    # Machine-readable output for CI / automated validation
    print("JSON_RESULTS:", json.dumps({
        "model": MODEL_NAME,
        "mean_f1": round(mean_reward, 4),
        "tasks": results,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
