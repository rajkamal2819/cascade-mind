"""
inference.py
------------
Baseline inference script for service_impact_env.

Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks
(easy, medium, hard) and reports reproducible F1 scores.

Environment variables:
  HF_TOKEN       — HuggingFace / API key (required)
  API_BASE_URL   — LLM API endpoint (default: https://api.openai.com/v1)
  MODEL_NAME     — model identifier (default: gpt-4o-mini)
  ENV_BASE_URL   — environment server URL
                   (default: https://rajkamal2819-cascade-mind.hf.space)

Usage against the live HuggingFace Space:
  export HF_TOKEN=hf_...
  export API_BASE_URL=https://api.cerebras.ai/v1
  export MODEL_NAME=llama-3.3-70b
  python inference.py

Usage against a local server:
  export HF_TOKEN=hf_...
  export ENV_BASE_URL=http://localhost:8000
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  python inference.py
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Imports — dual-path for in-repo vs installed package
# ---------------------------------------------------------------------------
from openai import OpenAI

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
# Configuration — matches the required inference.py variable contract:
#   HF_TOKEN     : no default (must be set by caller)
#   API_BASE_URL : has default  (can be overridden)
#   MODEL_NAME   : has default  (can be overridden)
# ---------------------------------------------------------------------------
HF_TOKEN     = os.getenv("HF_TOKEN")                              # no default — required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
# ENV_BASE_URL defaults to the live HF Space so `python inference.py` works out-of-the-box
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://rajkamal2819-cascade-mind.hf.space")

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

THE ENVIRONMENT IS NOISY — treat it like real production:
- Service registry responses MAY silently DROP real dependents or ADD fake ones.
- Runbooks may list stale or renamed services.
- A service appearing in MULTIPLE sources (registry + runbook + monitoring) is very likely real.
- A service appearing in ONLY ONE noisy source may be hallucinated — still include it if budget allows.
- err on the side of OVER-INCLUSION: missing a real service is 4× worse than a false positive (F-beta β=2).

AVAILABLE ACTIONS (respond with exactly one JSON object per turn):

BUDGET-CONSUMING actions (each deducts 1 from your query budget):
1. Find services that CALL a specific service (upstream dependents):
   {"action_type": "query_dependents", "service_name": "<service>"}

2. Find what a service DEPENDS ON (its downstream dependencies):
   {"action_type": "query_dependencies", "service_name": "<service>"}

FREE actions (ZERO budget cost — use aggressively):
3. Read internal runbook/documentation for a service:
   {"action_type": "query_runbook", "service_name": "<service>"}

4. Read the changelog/PR for the changed service:
   {"action_type": "query_changelog", "service_name": "<any service>"}

5. View Datadog-style monitoring dashboard for a service:
   {"action_type": "query_monitoring", "service_name": "<service>"}

TERMINAL action:
6. Submit your final answer (ends the episode):
   {"action_type": "submit", "affected_services": ["svc_a", "svc_b", ...]}

═══════════════════════════════════════════════════
OPTIMAL STRATEGY — follow this EXACTLY:
═══════════════════════════════════════════════════

PHASE 1 — Free intel (no budget cost, always do all 3):
  a. query_changelog(changed_service)  → understand what broke
  b. query_runbook(changed_service)    → get known consumers list
  c. query_monitoring(changed_service) → see who is actively calling it

  → After Phase 1: extract ALL service names mentioned. Add to "candidates" set.

PHASE 2 — BFS with budget (use query_dependents as primary tool):
  For each service in candidates (starting with changed_service):
    - query_dependents(svc) → adds new callers to candidates
    - If new callers found, add them to candidates for next iteration
    - Stop when: no new services found OR budget ≤ 3

  IMPORTANT: query_dependents is better than query_dependencies for finding blast radius.
  Only use query_dependencies when you need to verify a service's role.

PHASE 3 — Free verification (for uncertain services, zero budget):
  For any service seen in ONLY ONE source and you're unsure:
    - query_runbook(uncertain_svc) → confirms or denies from a second source
    - query_monitoring(uncertain_svc) → check if it's actively receiving traffic from changed svc

PHASE 4 — Submit:
  - If budget ≤ 3: SUBMIT IMMEDIATELY with everything found so far
  - Include ALL services with ANY evidence from ANY source
  - Do NOT include the changed_service itself in affected_services
  - Do NOT include services only mentioned as infrastructure (e.g. database_service, cache_service)
    UNLESS they were returned by query_dependents

TEXT PARSING RULES:
  Service names always use underscore format: "auth_service", "cart_service", "payment_service"
  Extract names from patterns like:
    "- auth_service (team: platform)"
    "→ cart_service (status: active)"
    "* order_service"
    "| payment_service | active |"
  Ignore: service names with hyphens (fake), names not ending in _service/_backend/_gateway

SCORING REMINDER:
  F-beta(β=2) weights RECALL 4× over precision.
  Submitting 2 extra false positives costs 0.02 points.
  Missing 2 real services costs 0.08+ points.
  → Be inclusive. Submit everything you have reasonable evidence for.

CRITICAL:
- result[] is ALWAYS [] during queries — only message text has information.
- result[] is only populated AFTER submit (shows ground truth for postmortem).
- Respond with ONLY a single valid JSON object. No explanation, no markdown fences.
- The "affected_services" list must NOT include the changed service itself.
- NEVER re-query a service you already queried (costs -0.05 penalty).
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

def run_episode(
    client: OpenAI,
    env_url: str,
    seed: int,
    task_name: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one complete episode and return results dict."""

    with ServiceImpactEnv(base_url=env_url).sync() as env:
        # ── Reset ──────────────────────────────────────────────────────
        reset_result = env.reset(seed=seed)
        obs = reset_result.observation

        # ── Structured output: START marker (required by validator) ────
        print(f"[START] task={task_name}", flush=True)

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
                    f"INCIDENT: Service '{obs.changed_service}' has a breaking change.\n"
                    f"Budget: {obs.queries_remaining} queries available.\n\n"
                    f"INITIAL ALERT:\n{obs.message}\n\n"
                    f"Start Phase 1 now: query_changelog → query_runbook → query_monitoring "
                    f"(all free). Then BFS with query_dependents."
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
                completion = client.chat.completions.create(
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
            result = env.step(action)
            obs = result.observation
            steps_taken += 1

            if action.action_type != "submit":
                queries_used += 1

            # ── Structured output: STEP marker ────────────────────────
            _step_reward = obs.reward if obs.reward is not None else 0.0
            print(f"[STEP] step={steps_taken} reward={_step_reward}", flush=True)

            if verbose:
                msg_preview = obs.message[:150]
                print(f"  [Step {step_num+1}] Env   → {msg_preview}")
                if not result.done:
                    print(f"             Queries left: {obs.queries_remaining}")

            # Feed back to model
            if not result.done:
                feedback = (
                    f"[{action.action_type.upper()}] result:\n"
                    f"{obs.message}\n\n"
                    f"Queries remaining: {obs.queries_remaining}\n"
                    f"Step reward: {obs.reward}\n"
                    f"Tip: Extract all service_name patterns from the text above, "
                    f"then decide the next action."
                )
            else:
                feedback = f"Episode ended. {obs.message}"

            messages.append({"role": "user", "content": feedback})

        elapsed = time.time() - start_time
        raw_reward = result.reward if result.reward is not None else 0.0
        # Clamp to open interval (0, 1) — validator rejects 0.0 and 1.0
        reward = max(0.001, min(0.999, raw_reward))

        # ── Structured output: END marker (required by validator) ─────
        print(f"[END] task={task_name} score={reward} steps={steps_taken}", flush=True)

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

def main() -> None:
    print("=" * 65)
    print("  service_impact_env — Baseline Inference Script")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  API Base  : {API_BASE_URL}")
    print(f"  Env URL   : {ENV_BASE_URL}")
    print("=" * 65)

    if not HF_TOKEN:
        raise SystemExit(
            "ERROR: HF_TOKEN environment variable not set."
        )

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    results: List[Dict[str, Any]] = []

    for task_name, seed in TASK_SEEDS.items():
        result = run_episode(
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
    main()
