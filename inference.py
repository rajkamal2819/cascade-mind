"""
inference.py
------------
Cascade-Mind v2 inference script — hypothesis-driven SRE agent.

Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks
(easy, medium, hard) and reports reproducible F-beta scores.

v2 features:
  - Reward-profile-adaptive strategy (parses profile from reset message)
  - submit_hypothesis for mid-investigation calibration
  - Handles [TOPOLOGY ALERT] mutations mid-episode
  - Free-action cap awareness (runbook=2, changelog=2, monitoring=3)
  - Structured [START]/[STEP]/[END] markers (required by validator)

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
        try:
            from cascade_mind.models import ServiceImpactAction  # type: ignore
            from cascade_mind.client import ServiceImpactEnv  # type: ignore
        except ImportError:
            raise SystemExit(
                "Cannot import ServiceImpactAction/ServiceImpactEnv. "
                "Run from the repo root or install the package with: pip install -e ."
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

═══════════════════════════════════════════════════
AVAILABLE ACTIONS (respond with exactly one JSON object per turn):
═══════════════════════════════════════════════════

BUDGET-CONSUMING actions (each deducts 1 from your query budget):
1. Find services that CALL a specific service (upstream dependents):
   {"action_type": "query_dependents", "service_name": "<service>"}

2. Find what a service DEPENDS ON (its downstream dependencies):
   {"action_type": "query_dependencies", "service_name": "<service>"}

FREE actions (ZERO budget cost — but have PER-EPISODE CAPS):
3. Read internal runbook/documentation (cap: 2 uses total):
   {"action_type": "query_runbook", "service_name": "<service>"}

4. Read changelog/PR for the changed service (cap: 2 uses total):
   {"action_type": "query_changelog", "service_name": "<any service>"}

5. View Datadog-style monitoring dashboard (cap: 3 uses total):
   {"action_type": "query_monitoring", "service_name": "<service>"}

   ⚠️ Exceeding a free-action cap costs 1 query from budget!

FREE uncapped tools:
6. Check for topology changes during this episode:
   {"action_type": "query_topology_diff"}

7. Get health/metadata summary for a service:
   {"action_type": "query_service_health", "service_name": "<service>"}

HYPOTHESIS CHECK (costs 1 query — use once mid-investigation to calibrate):
8. Submit a partial hypothesis to get an F-beta score without ending the episode:
   {"action_type": "submit_hypothesis", "affected_services": ["svc_a", ...], "confidence": 0.7}
   → Returns partial F-beta score. Max 3 hypotheses per episode.

TERMINAL action:
9. Submit your final answer (ends the episode):
   {"action_type": "submit", "affected_services": ["svc_a", "svc_b", ...]}

═══════════════════════════════════════════════════
REWARD PROFILES (adapts per episode — check reset message):
═══════════════════════════════════════════════════
- recall_heavy   → β=2.5, overclaim ok (70%), mild penalty → Be VERY inclusive
- balanced       → β=1.5, moderate overclaim (60%), medium penalty → Balance coverage & precision
- precision_heavy → β=0.8, strict overclaim (50%), heavy penalty → Be selective, avoid FPs
- efficiency     → β=2.0, moderate overclaim, budget bonus → Finish fast, save queries

Adapt your strategy:
- recall_heavy/efficiency → include everything with any evidence
- precision_heavy → only include services with 2+ sources of evidence
- balanced → include services with 1+ source, skip very uncertain ones

═══════════════════════════════════════════════════
TOPOLOGY MUTATIONS (medium/hard difficulties):
═══════════════════════════════════════════════════
Mid-episode, you may receive a [TOPOLOGY ALERT] indicating the service graph
has changed (new dependencies added or old ones removed via failover/deprecation).
When you see this:
- Re-investigate affected areas — the blast radius has changed
- Services previously not affected may now be affected (and vice versa)
- submit_hypothesis can help you recalibrate after a mutation

═══════════════════════════════════════════════════
OPTIMAL STRATEGY — follow this EXACTLY:
═══════════════════════════════════════════════════

PHASE 1 — Free intel (use all 3 free actions on the changed service):
  a. query_changelog(changed_service) → understand what broke
  b. query_runbook(changed_service)   → get known consumers list
  c. query_monitoring(changed_service) → see who is actively calling it
  → Extract ALL service names mentioned. Build initial "candidates" set.

PHASE 2 — BFS with budget (query_dependents as primary tool):
  For each candidate (starting with changed_service):
    - query_dependents(svc) → adds new callers to candidates
    - If new callers found, add them for next iteration
    - Stop when: no new services found OR budget ≤ 4

PHASE 3 — Hypothesis check (optional, costs 1 query):
  After Phase 2, if you have ≥ 5 candidates and budget > 4:
    - submit_hypothesis with current candidates
    - Use the partial F-beta score to decide if you need more investigation

PHASE 4 — Free verification (for uncertain services):
  For services seen in ONLY ONE source:
    - query_monitoring(uncertain_svc) → check for active traffic
  (Stay within caps: 2 runbooks, 2 changelogs, 3 monitoring total)

PHASE 5 — Submit:
  - If budget ≤ 3 or no new services found: SUBMIT IMMEDIATELY
  - For recall_heavy: include ALL candidates
  - For precision_heavy: include only multi-source candidates
  - Do NOT include the changed_service itself
  - Do NOT include infrastructure services UNLESS query_dependents returned them

TEXT PARSING RULES:
  Service names always use underscore format: "auth_service", "cart_service"
  Extract names from patterns like:
    "- auth_service (team: platform)"
    "→ cart_service (status: active)"
  Ignore: names with hyphens (fake), names not ending in _service/_backend/_gateway

SCORING REMINDER:
  F-beta weights RECALL more than precision (β≥1 usually).
  Missing real services is much worse than including false positives.
  → Be inclusive. Submit everything with reasonable evidence.

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

        # ── Parse reward profile from reset message ───────────────────
        reward_profile = "balanced"  # default
        profile_match = re.search(r"Reward profile:\s*(\w+)", obs.message)
        if profile_match:
            reward_profile = profile_match.group(1)

        # ── Profile-adaptive strategy hints ───────────────────────────
        profile_hints = {
            "recall_heavy": (
                "STRATEGY: recall_heavy profile — MAXIMIZE RECALL. "
                "Include every service with ANY evidence. "
                "False positives are cheap; missing services is very expensive (β=2.5)."
            ),
            "balanced": (
                "STRATEGY: balanced profile — moderate approach. "
                "Include services with at least one source of evidence. "
                "Skip only very uncertain services (β=1.5)."
            ),
            "precision_heavy": (
                "STRATEGY: precision_heavy profile — BE SELECTIVE. "
                "Only include services confirmed by 2+ sources. "
                "Overclaiming is heavily penalized. Prefer fewer, confident predictions (β=0.8)."
            ),
            "efficiency": (
                "STRATEGY: efficiency profile — FINISH FAST. "
                "Budget bonus for early submission. "
                "Do thorough free intel, limited BFS, then submit quickly (β=2.0)."
            ),
        }
        strategy_hint = profile_hints.get(reward_profile, profile_hints["balanced"])

        # ── Structured output: START marker (required by validator) ────
        print(f"[START] task={task_name}", flush=True)

        if verbose:
            print(f"\n{'─'*65}")
            print(f"  TASK: {task_name.upper():<8}  |  seed={seed}")
            print(f"  Changed service : {obs.changed_service}")
            print(f"  Budget          : {obs.queries_remaining} queries")
            print(f"  Reward profile  : {reward_profile}")
            print(f"{'─'*65}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"INCIDENT: Service '{obs.changed_service}' has a breaking change.\n"
                    f"Budget: {obs.queries_remaining} queries available.\n"
                    f"Reward profile: {reward_profile}\n\n"
                    f"{strategy_hint}\n\n"
                    f"INITIAL ALERT:\n{obs.message}\n\n"
                    f"Start Phase 1 now: query_changelog → query_runbook → query_monitoring "
                    f"(all free, but capped at 2/2/3 uses total). Then BFS with query_dependents."
                ),
            },
        ]

        result = reset_result
        steps_taken = 0
        queries_used = 0
        parse_errors = 0
        hypothesis_used = False
        mutation_seen = False
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

            if action.action_type not in ("submit", "query_runbook", "query_changelog", "query_monitoring"):
                queries_used += 1

            if action.action_type == "submit_hypothesis":
                hypothesis_used = True

            # Detect topology mutations
            if "[TOPOLOGY ALERT]" in obs.message:
                mutation_seen = True

            # ── Structured output: STEP marker ────────────────────────
            _step_reward = obs.reward if obs.reward is not None else 0.0
            print(f"[STEP] step={steps_taken} reward={_step_reward}", flush=True)

            if verbose:
                msg_preview = obs.message[:150]
                print(f"  [Step {step_num+1}] Env   → {msg_preview}")
                if not result.done:
                    print(f"             Queries left: {obs.queries_remaining}")
                if mutation_seen:
                    print(f"             ⚡ TOPOLOGY MUTATION DETECTED")

            # Feed back to model
            if not result.done:
                # Build adaptive feedback based on action type
                mutation_note = ""
                if "[TOPOLOGY ALERT]" in obs.message:
                    mutation_note = (
                        "\n\n⚡ TOPOLOGY CHANGE DETECTED! The service graph has been modified. "
                        "Previously unaffected services may now be in the blast radius. "
                        "Consider re-investigating or using submit_hypothesis to recalibrate."
                    )

                hypothesis_note = ""
                if action.action_type == "submit_hypothesis" and obs.delayed_reward is not None:
                    score = obs.delayed_reward
                    if score >= 0.8:
                        hypothesis_note = f"\n\nYour hypothesis scored {score:.3f} — GOOD. Consider submitting soon."
                    elif score >= 0.5:
                        hypothesis_note = f"\n\nYour hypothesis scored {score:.3f} — MODERATE. Try adding more services."
                    else:
                        hypothesis_note = f"\n\nYour hypothesis scored {score:.3f} — LOW. You're missing key services. Keep investigating."

                feedback = (
                    f"[{action.action_type.upper()}] result:\n"
                    f"{obs.message}\n\n"
                    f"Queries remaining: {obs.queries_remaining}\n"
                    f"Step reward: {obs.reward}"
                    f"{mutation_note}"
                    f"{hypothesis_note}\n"
                    f"Extract all service_name patterns from the text above, "
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
            print(f"  │  Profile    : {reward_profile}")
            print(f"  │  Reward (Fβ): {format_score_bar(reward)}")
            print(f"  │  Steps      : {steps_taken}  |  Queries: {queries_used}")
            print(f"  │  Hypothesis : {'used' if hypothesis_used else 'not used'}")
            print(f"  │  Mutations  : {'detected' if mutation_seen else 'none'}")
            print(f"  │  Time       : {elapsed:.1f}s")
            print(f"  └{'─'*58}")

        return {
            "task": task_name,
            "seed": seed,
            "reward": reward,
            "reward_profile": reward_profile,
            "steps": steps_taken,
            "queries_used": queries_used,
            "hypothesis_used": hypothesis_used,
            "mutation_seen": mutation_seen,
            "elapsed_s": round(elapsed, 2),
            "parse_errors": parse_errors,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  cascade-mind v2 — Hypothesis-Driven SRE Agent")
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
        profile_tag = f"[{r.get('reward_profile', '?')[:4]}]"
        print(f"  {flag}  {r['task']:<8}  {bar}  {profile_tag}  ({r['elapsed_s']}s)")
        total_reward += r["reward"]

    mean_reward = total_reward / len(results)
    print(f"{'─'*65}")
    print(f"     MEAN Fβ  {format_score_bar(mean_reward)}")
    print(f"{'═'*65}\n")

    # Machine-readable output for CI / automated validation
    print("JSON_RESULTS:", json.dumps({
        "model": MODEL_NAME,
        "version": "cascade-mind-v2",
        "mean_fbeta": round(mean_reward, 4),
        "tasks": results,
    }, indent=2))


if __name__ == "__main__":
    main()
