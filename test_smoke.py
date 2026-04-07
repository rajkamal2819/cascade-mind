"""
test_smoke.py
-------------
Smoke tests for service_impact_env v0.2.0.

Tests validate:
  - Seed-perturbed graph produces unique topologies
  - get_scenario() returns valid scenarios across all difficulties
  - LLM simulator falls back gracefully (no HF_TOKEN needed for smoke test)
  - Option A: result=[] for all query actions (agents must reason from message text)
  - New free actions: query_runbook, query_changelog, query_monitoring
  - F-beta (β=2) reward on submit
  - Overclaiming penalty
  - Budget exhaustion reward = -0.4
  - Step rewards: +0.05 new query, -0.05 re-query
"""
import os
import sys

# Disable LLM for smoke tests — exercises fallback templates (no HF_TOKEN needed)
os.environ["LLM_SIMULATOR_ENABLED"] = "false"

sys.path.insert(0, os.path.dirname(__file__))

from models import ServiceImpactAction
from server.service_impact_environment import ServiceImpactEnvironment
from server.graph_builder import (
    build_service_graph,
    get_affected_services,
    get_all_services,
    get_scenario,
    DIFFICULTY_ORDER,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {label}")
    else:
        msg = f"  {FAIL} FAILED: {label}" + (f" — {detail}" if detail else "")
        print(msg)
        _failures.append(msg)


# ── 1. Graph perturbation check ──────────────────────────────────────────────
print("\n=== Graph Perturbation ===")

graphs = {}
for seed in [0, 1, 2, 100, 999]:
    G = build_service_graph(seed=seed)
    graphs[seed] = G
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    check(
        f"seed={seed}: {n_nodes} nodes, {n_edges} edges",
        n_nodes == 30 and 55 <= n_edges <= 80,
        f"nodes={n_nodes} edges={n_edges}",
    )

# Different seeds should produce different graphs
edges_0   = set(graphs[0].edges())
edges_100 = set(graphs[100].edges())
check(
    "seed=0 and seed=100 produce different topologies",
    edges_0 != edges_100,
    "graphs are identical!",
)

# ── 2. Scenario selection check ──────────────────────────────────────────────
print("\n=== Scenario Selection ===")

for seed in range(6):
    G = build_service_graph(seed=seed)
    scenario = get_scenario(G, seed)
    difficulty = DIFFICULTY_ORDER[seed % 3]
    svc = scenario["changed_service"]
    n_aff = len(get_affected_services(G, svc))
    check(
        f"seed={seed} ({difficulty}): changed={svc}, affected={n_aff}, budget={scenario['max_queries']}",
        scenario["difficulty"] == difficulty and svc in G.nodes() and n_aff >= 1,
        f"difficulty mismatch or svc not in graph",
    )

# ── 3. Environment: Option A + new actions ───────────────────────────────────
print("\n=== Environment: Option A & New Actions ===")

env = ServiceImpactEnvironment()

for seed, label in [(0, "EASY"), (3, "MEDIUM"), (6, "HARD")]:
    print(f"\n  --- Task: {label} (seed={seed}) ---")
    obs = env.reset(seed=seed)

    check(
        f"[{label}] reset: result=[] (Option A)",
        obs.result == [],
        f"result={obs.result}",
    )
    check(
        f"[{label}] reset: message non-empty with task info",
        len(obs.message) > 30 and obs.changed_service in obs.message,
        f"message={obs.message[:80]}",
    )
    check(
        f"[{label}] reset: queries_remaining = {obs.queries_remaining}",
        obs.queries_remaining >= 8,
    )

    changed_svc = obs.changed_service

    # ── query_dependents (budget action) ─────────────────────────────────
    step1 = env.step(ServiceImpactAction(
        action_type="query_dependents",
        service_name=changed_svc,
    ))
    check(
        f"[{label}] query_dependents: result=[] (Option A)",
        step1.result == [],
        f"result={step1.result}",
    )
    check(
        f"[{label}] query_dependents: message contains text",
        len(step1.message) > 20,
    )
    check(
        f"[{label}] query_dependents: step reward=+0.05",
        step1.reward == 0.05,
        f"reward={step1.reward}",
    )

    # ── Re-query penalty ──────────────────────────────────────────────────
    step2 = env.step(ServiceImpactAction(
        action_type="query_dependents",
        service_name=changed_svc,  # same service — should penalise
    ))
    check(
        f"[{label}] re-query: step reward=-0.05",
        step2.reward == -0.05,
        f"reward={step2.reward}",
    )

    # ── Free actions (no budget deduction) ───────────────────────────────
    budget_before_free = step2.queries_remaining

    runbook_obs = env.step(ServiceImpactAction(
        action_type="query_runbook",
        service_name=changed_svc,
    ))
    check(
        f"[{label}] query_runbook: FREE (budget unchanged)",
        runbook_obs.queries_remaining == budget_before_free,
        f"before={budget_before_free}, after={runbook_obs.queries_remaining}",
    )
    check(
        f"[{label}] query_runbook: message has content",
        len(runbook_obs.message) > 50,
    )
    check(
        f"[{label}] query_runbook: reward=None (no signal)",
        runbook_obs.reward is None,
        f"reward={runbook_obs.reward}",
    )

    changelog_obs = env.step(ServiceImpactAction(
        action_type="query_changelog",
        service_name=changed_svc,
    ))
    check(
        f"[{label}] query_changelog: FREE (budget unchanged)",
        changelog_obs.queries_remaining == budget_before_free,
    )

    monitoring_obs = env.step(ServiceImpactAction(
        action_type="query_monitoring",
        service_name=changed_svc,
    ))
    check(
        f"[{label}] query_monitoring: FREE (budget unchanged)",
        monitoring_obs.queries_remaining == budget_before_free,
    )

    # ── Perfect submit ────────────────────────────────────────────────────
    G = build_service_graph(seed=seed)
    correct = sorted(get_affected_services(G, changed_svc))
    final = env.step(ServiceImpactAction(
        action_type="submit",
        affected_services=correct,
    ))
    check(
        f"[{label}] perfect submit: done=True",
        final.done,
    )
    check(
        f"[{label}] perfect submit: result reveals ground truth",
        sorted(final.result) == correct,
        f"result={sorted(final.result)}, expected={correct}",
    )
    check(
        f"[{label}] perfect submit: reward == 0.999 (F-beta β=2, clamped to open interval)",
        final.reward is not None and final.reward == 0.999,
        f"reward={final.reward}",
    )

    # ── Empty submit ──────────────────────────────────────────────────────
    env.reset(seed=seed)
    bad = env.step(ServiceImpactAction(action_type="submit", affected_services=[]))
    check(
        f"[{label}] empty submit: reward=0.001 (clamped to open interval)",
        bad.reward == 0.001,
        f"reward={bad.reward}",
    )

# ── 4. Overclaiming penalty ──────────────────────────────────────────────────
print("\n=== Overclaiming Penalty ===")

env.reset(seed=0)
G = build_service_graph(seed=0)
all_svcs = get_all_services(G)
# Submit all 30 services — should trigger overclaiming penalty
massive_submit = env.step(ServiceImpactAction(
    action_type="submit",
    affected_services=all_svcs,
))
check(
    "Submitting all 30 services triggers overclaiming penalty (reward < 0.7)",
    massive_submit.reward is not None and massive_submit.reward < 0.7,
    f"reward={massive_submit.reward}",
)

# ── 5. Budget exhaustion reward = -0.4 ──────────────────────────────────────
print("\n=== Budget Exhaustion Reward ===")

obs = env.reset(seed=2)  # hard task, budget=8-10
budget = obs.queries_remaining
# Burn all queries without submitting — capture the final step response
last_step = None
for i in range(budget):
    last_step = env.step(ServiceImpactAction(
        action_type="query_dependents" if i % 2 == 0 else "query_dependencies",
        service_name=obs.changed_service,
    ))
# The step that exhausts the budget returns done=True, reward=-0.4
check(
    "Budget exhaustion: done=True",
    last_step.done,
    f"done={last_step.done}",
)
check(
    "Budget exhaustion: reward=-0.4",
    last_step.reward == -0.4,
    f"reward={last_step.reward}",
)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if _failures:
    print(f"\033[91mFAILED — {len(_failures)} test(s) failed:\033[0m")
    for f in _failures:
        print(f"  {f}")
    sys.exit(1)
else:
    print(f"\033[92mAll tests passed!\033[0m")
