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
    SERVICES,
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

# ── 7. submit_hypothesis (v2) ────────────────────────────────────────────────
print("\n=== submit_hypothesis (v2) ===")

obs = env.reset(seed=0)
changed_svc = obs.changed_service
G = build_service_graph(seed=0)
correct = sorted(get_affected_services(G, changed_svc))

# Submit a partial hypothesis — should return delayed_reward, NOT end episode
partial = correct[:len(correct)//2] if len(correct) > 1 else correct
h1 = env.step(ServiceImpactAction(
    action_type="submit_hypothesis",
    affected_services=partial,
    confidence=0.5,
))
check(
    "Hypothesis: done=False (episode continues)",
    not h1.done,
    f"done={h1.done}",
)
check(
    "Hypothesis: delayed_reward is float > 0",
    h1.delayed_reward is not None and h1.delayed_reward > 0,
    f"delayed_reward={h1.delayed_reward}",
)
check(
    "Hypothesis: result=[] (ground truth hidden)",
    h1.result == [],
    f"result={h1.result}",
)
check(
    "Hypothesis: costs 1 query",
    h1.queries_remaining == obs.queries_remaining - 1,
    f"remaining={h1.queries_remaining}, expected={obs.queries_remaining - 1}",
)
check(
    "Hypothesis: message contains HYPOTHESIS",
    "HYPOTHESIS" in h1.message,
    f"message={h1.message[:80]}",
)

# Perfect hypothesis should get high partial score
h2 = env.step(ServiceImpactAction(
    action_type="submit_hypothesis",
    affected_services=correct,
    confidence=0.95,
))
check(
    "Perfect hypothesis: delayed_reward == 0.999",
    h2.delayed_reward is not None and h2.delayed_reward == 0.999,
    f"delayed_reward={h2.delayed_reward}",
)

# State tracks hypothesis_count
check(
    "State tracks hypothesis_count=2",
    env.state.hypothesis_count == 2,
    f"hypothesis_count={env.state.hypothesis_count}",
)
check(
    "State tracks last_hypothesis_score",
    env.state.last_hypothesis_score == 0.999,
    f"last_hypothesis_score={env.state.last_hypothesis_score}",
)

# Can still submit after hypothesis
final = env.step(ServiceImpactAction(action_type="submit", affected_services=correct))
check(
    "Submit after hypothesis: done=True",
    final.done,
)
check(
    "Submit after hypothesis: reward == 0.999",
    final.reward is not None and final.reward == 0.999,
    f"reward={final.reward}",
)

# ── 8. Trajectory logging (v2) ───────────────────────────────────────────────
print("\n=== Trajectory Logging (v2) ===")

import tempfile, json
from server.trajectory_logger import TrajectoryLogger

with tempfile.TemporaryDirectory() as tmpdir:
    tlogger = TrajectoryLogger(tmpdir)
    tlogger.log_reset(seed=99, changed_service="test_svc", difficulty="easy", max_queries=15)
    tlogger.log_step(seed=99, step_num=1, action_type="query_dependents",
                     service_name="test_svc", reward=0.05, queries_remaining=14,
                     message="test message")
    tlogger.log_hypothesis(seed=99, step_num=2, predicted=["a", "b"],
                          confidence=0.6, partial_score=0.5)
    tlogger.log_submit(seed=99, step_num=3, predicted=["a", "b", "c"],
                      correct=["a", "b", "c"], reward=0.999, precision=1.0, recall=1.0)
    tlogger.log_episode(seed=99, summary={"reward": 0.999})

    records = tlogger.read_episode(99)
    check(
        "Trajectory: 5 records written",
        len(records) == 5,
        f"got {len(records)} records",
    )
    check(
        "Trajectory: events are reset,step,hypothesis,submit,episode",
        [r["event"] for r in records] == ["reset", "step", "hypothesis", "submit", "episode"],
        f"events={[r['event'] for r in records]}",
    )
    check(
        "Trajectory: hypothesis has partial_score in extra",
        records[2]["extra"]["confidence"] == 0.6,
        f"extra={records[2]['extra']}",
    )

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("\n=== Free-Action Caps (v2) ===")

# Use seed=1 (medium) — caps: runbook=2, changelog=2, monitoring=3
env = ServiceImpactEnvironment()
obs = env.reset(seed=1)
changed_svc = obs.changed_service
budget_start = obs.queries_remaining

# First 2 runbook calls should be free (medium runbook_cap=2)
r1 = env.step(ServiceImpactAction(action_type="query_runbook", service_name=changed_svc))
r2 = env.step(ServiceImpactAction(action_type="query_runbook", service_name=changed_svc))
check(
    "Runbook call 1: budget unchanged",
    r1.queries_remaining == budget_start,
    f"remaining={r1.queries_remaining}, expected={budget_start}",
)
check(
    "Runbook call 2: budget unchanged",
    r2.queries_remaining == budget_start,
    f"remaining={r2.queries_remaining}, expected={budget_start}",
)

# 3rd runbook call should deduct 1 from budget (exceeds cap=2)
r3 = env.step(ServiceImpactAction(action_type="query_runbook", service_name=changed_svc))
check(
    "Runbook call 3 (over cap): budget deducted by 1",
    r3.queries_remaining == budget_start - 1,
    f"remaining={r3.queries_remaining}, expected={budget_start - 1}",
)
check(
    "Runbook call 3: cap warning in message",
    "cap exceeded" in r3.message.lower(),
    f"message={r3.message[:120]}",
)

# Monitoring: 3 free, 4th costs budget (medium monitoring_cap=3)
env = ServiceImpactEnvironment()
obs = env.reset(seed=1)
budget_start = obs.queries_remaining
for i in range(3):
    m = env.step(ServiceImpactAction(action_type="query_monitoring", service_name=obs.changed_service))
check(
    "Monitoring call 3: budget unchanged (cap=3)",
    m.queries_remaining == budget_start,
    f"remaining={m.queries_remaining}, expected={budget_start}",
)
m4 = env.step(ServiceImpactAction(action_type="query_monitoring", service_name=obs.changed_service))
check(
    "Monitoring call 4 (over cap): budget deducted by 1",
    m4.queries_remaining == budget_start - 1,
    f"remaining={m4.queries_remaining}, expected={budget_start - 1}",
)

# Changelog: 2 free, 3rd costs budget (medium changelog_cap=2)
env = ServiceImpactEnvironment()
obs = env.reset(seed=1)
budget_start = obs.queries_remaining
for i in range(2):
    c = env.step(ServiceImpactAction(action_type="query_changelog", service_name=obs.changed_service))
check(
    "Changelog call 2: budget unchanged (cap=2)",
    c.queries_remaining == budget_start,
    f"remaining={c.queries_remaining}, expected={budget_start}",
)
c3 = env.step(ServiceImpactAction(action_type="query_changelog", service_name=obs.changed_service))
check(
    "Changelog call 3 (over cap): budget deducted by 1",
    c3.queries_remaining == budget_start - 1,
    f"remaining={c3.queries_remaining}, expected={budget_start - 1}",
)

# State counters track uses
state = env.state
check(
    "State tracks changelog_uses=3",
    state.changelog_uses == 3,
    f"changelog_uses={state.changelog_uses}",
)

# ── 9. RewardOrchestrator (v2) ───────────────────────────────────────────────
print("\n=== RewardOrchestrator (v2) ===")

from server.reward_orchestrator import RewardOrchestrator, PROFILES

orch = RewardOrchestrator()

# Profile selection is deterministic from seed
p0 = orch.get_profile(0)
p1 = orch.get_profile(1)
p2 = orch.get_profile(2)
p3 = orch.get_profile(3)
check(
    "Profile rotation: seed 0-3 give 4 different profiles",
    len({p0.name, p1.name, p2.name, p3.name}) == 4,
    f"profiles={[p0.name, p1.name, p2.name, p3.name]}",
)
check(
    "Profile same seed = same profile",
    orch.get_profile(0).name == orch.get_profile(4).name,
    f"seed0={orch.get_profile(0).name}, seed4={orch.get_profile(4).name}",
)

# Perfect prediction should give 0.999 on any profile
for name, profile in PROFILES.items():
    score = orch.compute(
        predicted={"a", "b", "c"},
        correct={"a", "b", "c"},
        all_services_count=30,
        queries_used=5,
        max_queries=15,
        profile=profile,
    )
    check(
        f"Profile '{name}': perfect prediction → 0.999",
        score["reward"] == 0.999,
        f"reward={score['reward']}",
    )

# Empty prediction gives 0.001 on any profile
for name, profile in PROFILES.items():
    score = orch.compute(
        predicted=set(),
        correct={"a", "b", "c"},
        all_services_count=30,
        profile=profile,
    )
    check(
        f"Profile '{name}': empty prediction → 0.001",
        score["reward"] == 0.001,
        f"reward={score['reward']}",
    )

# Precision-heavy profile should penalize overclaiming harder
recall_score = orch.compute(
    predicted=set(SERVICES[:20]),
    correct=set(SERVICES[:5]),
    all_services_count=30,
    profile=PROFILES["recall_heavy"],
)
prec_score = orch.compute(
    predicted=set(SERVICES[:20]),
    correct=set(SERVICES[:5]),
    all_services_count=30,
    profile=PROFILES["precision_heavy"],
)
check(
    "Precision-heavy penalizes overclaiming more than recall-heavy",
    prec_score["reward"] < recall_score["reward"],
    f"precision_heavy={prec_score['reward']}, recall_heavy={recall_score['reward']}",
)

# Environment uses reward profile from state
obs = env.reset(seed=0)
check(
    "Env state has reward_profile",
    hasattr(env.state, 'reward_profile') and env.state.reward_profile in PROFILES,
    f"reward_profile={getattr(env.state, 'reward_profile', 'MISSING')}",
)
check(
    "Reset message mentions reward profile",
    "Reward profile" in obs.message or "reward_profile" in obs.message,
    f"message tail={obs.message[-200:]}",
)

# ── Section 10: MutationEngine ────────────────────────────────────────────────
print("\n─── Section 10: MutationEngine ───")

from server.mutation_engine import MutationEngine, DEFAULT_SCHEDULES, MutationConfig

# Easy difficulty → no mutations
me_easy = MutationEngine(seed=42, difficulty="easy")
G_easy = build_service_graph(42)
_, ev = me_easy.maybe_mutate(G_easy, 5)
check("Easy difficulty: no mutation at step 5", ev is None, f"event={ev}")

# Medium difficulty → mutation at step 5 only
me_med = MutationEngine(seed=42, difficulty="medium")
G_med = build_service_graph(42)
_, ev1 = me_med.maybe_mutate(G_med, 1)
check("Medium: no mutation at step 1", ev1 is None, "")
_, ev5 = me_med.maybe_mutate(G_med, 5)
check("Medium: mutation at step 5", ev5 is not None, f"event={ev5}")
check("Medium: mutation has [TOPOLOGY ALERT]", "[TOPOLOGY ALERT]" in ev5.description, ev5.description[:80])
_, ev5b = me_med.maybe_mutate(G_med, 5)
check("Medium: no double-trigger at step 5", ev5b is None, f"event={ev5b}")
check("Medium: 1 total mutation applied", me_med.total_mutations == 1, f"total={me_med.total_mutations}")

# Hard difficulty → mutations at steps 4 and 8
me_hard = MutationEngine(seed=42, difficulty="hard")
G_hard = build_service_graph(42)
_, ev4 = me_hard.maybe_mutate(G_hard, 4)
check("Hard: mutation at step 4", ev4 is not None, f"event={ev4}")
_, ev6 = me_hard.maybe_mutate(G_hard, 6)
check("Hard: no mutation at step 6", ev6 is None, "")
_, ev8 = me_hard.maybe_mutate(G_hard, 8)
check("Hard: mutation at step 8", ev8 is not None, f"event={ev8}")
check("Hard: 2 total mutations", me_hard.total_mutations == 2, f"total={me_hard.total_mutations}")

# Mutation event types
check(
    "Mutation event has valid type",
    ev4.mutation_type in ("add_edge", "remove_edge"),
    f"type={ev4.mutation_type}",
)
check(
    "Mutation event edges list exists",
    isinstance(ev4.edges, list),
    f"edges={ev4.edges}",
)

# End-to-end: hard env mutation appears in observation messages
mutation_found = False
for test_seed in range(100):
    env_t = ServiceImpactEnvironment()
    obs_r = env_t.reset(seed=test_seed)
    if env_t._task_difficulty == "hard" and env_t._mutation_engine is not None:
        # Do 4 queries to reach step 4 where hard triggers
        svc_list = sorted(env_t._graph.nodes())
        for i in range(4):
            query_svc = svc_list[i % len(svc_list)]
            obs_q = env_t.step(ServiceImpactAction(
                action_type="query_dependents",
                service_name=query_svc,
                affected_services=[],
            ))
            if "[TOPOLOGY ALERT]" in obs_q.message:
                mutation_found = True
                break
        if mutation_found:
            break

check(
    "Hard env: [TOPOLOGY ALERT] appears in query obs",
    mutation_found,
    "scanned 100 seeds for hard difficulty",
)

# Ground truth recomputation: after mutation, correct_affected may differ from original
if mutation_found:
    check(
        "Hard env: ground truth is a non-empty set after mutation",
        len(env_t._correct_affected) > 0,
        f"correct_affected size={len(env_t._correct_affected)}",
    )

# ── Section 11: CurriculumScheduler ──────────────────────────────────────────
print("\n─── Section 11: CurriculumScheduler ───")

from server.curriculum_scheduler import CurriculumScheduler, CURRICULUM_CONFIGS

cs = CurriculumScheduler()

# Easy config has lower noise and more generous caps
easy_cfg = cs.get_config("easy")
check("Easy: low drop probability", easy_cfg.drop_probability <= 0.10, f"drop={easy_cfg.drop_probability}")
check("Easy: runbook cap >= 2", easy_cfg.runbook_cap >= 2, f"cap={easy_cfg.runbook_cap}")
check("Easy: max_hypotheses == 3", easy_cfg.max_hypotheses == 3, f"max_hyp={easy_cfg.max_hypotheses}")

# Hard config has higher noise and tighter caps
hard_cfg = cs.get_config("hard")
check("Hard: high drop probability", hard_cfg.drop_probability >= 0.20, f"drop={hard_cfg.drop_probability}")
check("Hard: max_hypotheses == 2", hard_cfg.max_hypotheses == 2, f"max_hyp={hard_cfg.max_hypotheses}")

# Medium is between easy and hard
med_cfg = cs.get_config("medium")
check(
    "Medium: drop probability between easy and hard",
    easy_cfg.drop_probability <= med_cfg.drop_probability <= hard_cfg.drop_probability,
    f"easy={easy_cfg.drop_probability}, med={med_cfg.drop_probability}, hard={hard_cfg.drop_probability}",
)

# Unknown difficulty falls back to medium
unk_cfg = cs.get_config("unknown")
check("Unknown difficulty: falls back to medium", unk_cfg.difficulty == "medium", f"got={unk_cfg.difficulty}")

# Hint text varies by difficulty
easy_hint = cs.get_hint_text(easy_cfg, "test_service", 3)
check("Easy hint: contains 'small' or 'Hint'", "small" in easy_hint.lower() or "hint" in easy_hint.lower(), easy_hint[:50])
hard_hint = cs.get_hint_text(hard_cfg, "test_service", 20)
check("Hard hint: contains 'hard' or 'extensive'", "hard" in hard_hint.lower() or "extensive" in hard_hint.lower(), hard_hint[:50])

# Environment integration: easy env has runbook_cap=3 (from curriculum)
env_easy = ServiceImpactEnvironment()
obs_easy = env_easy.reset(seed=0)  # seed=0 → easy
check(
    "Easy env: runbook cap from curriculum",
    env_easy.FREE_ACTION_CAPS["query_runbook"] == easy_cfg.runbook_cap,
    f"env_cap={env_easy.FREE_ACTION_CAPS['query_runbook']}, curriculum={easy_cfg.runbook_cap}",
)

# Environment integration: hard env has max_hypotheses=2
env_hard_cs = ServiceImpactEnvironment()
obs_hard_cs = env_hard_cs.reset(seed=2)  # seed=2 → hard
# Test by submitting 3 hypotheses — 3rd should have extra penalty
all_svcs = sorted(env_hard_cs._graph.nodes())
for i in range(2):
    env_hard_cs.step(ServiceImpactAction(
        action_type="submit_hypothesis",
        affected_services=all_svcs[:3],
        confidence=0.5,
    ))
h3 = env_hard_cs.step(ServiceImpactAction(
    action_type="submit_hypothesis",
    affected_services=all_svcs[:3],
    confidence=0.5,
))
check(
    "Hard env: 3rd hypothesis over cap (max=2) shows warning",
    "cap exceeded" in h3.message.lower() or "⚠️" in h3.message,
    f"msg={h3.message[:100]}",
)

# Reset message includes curriculum hint
check(
    "Easy env: reset message has curriculum hint",
    "small" in obs_easy.message.lower() or "1-6" in obs_easy.message,
    f"message contains curriculum hint",
)

# ── Section 12: New MCP Tools (query_topology_diff, query_service_health) ─────
print("\n─── Section 12: New MCP Tools ───")

env_tools = ServiceImpactEnvironment()
obs_t = env_tools.reset(seed=0)

# query_service_health: free action, returns health info
h_obs = env_tools.step(ServiceImpactAction(
    action_type="query_service_health",
    service_name=obs_t.changed_service,
    affected_services=[],
))
check(
    "query_service_health: message has SERVICE HEALTH",
    "SERVICE HEALTH" in h_obs.message,
    f"message={h_obs.message[:80]}",
)
check(
    "query_service_health: budget unchanged (free)",
    h_obs.queries_remaining == obs_t.queries_remaining,
    f"remaining={h_obs.queries_remaining}",
)
check(
    "query_service_health: shows team/tier info",
    "Team:" in h_obs.message and "Tier:" in h_obs.message,
    f"message={h_obs.message[:120]}",
)

# query_service_health: unknown service
h_unk = env_tools.step(ServiceImpactAction(
    action_type="query_service_health",
    service_name="nonexistent_service",
    affected_services=[],
))
check(
    "query_service_health: unknown service handled",
    "Unknown" in h_unk.message or "unknown" in h_unk.message,
    f"message={h_unk.message[:80]}",
)

# query_topology_diff: no mutations for easy
td_obs = env_tools.step(ServiceImpactAction(
    action_type="query_topology_diff",
    service_name="",
    affected_services=[],
))
check(
    "query_topology_diff: message has TOPOLOGY DIFF",
    "TOPOLOGY DIFF" in td_obs.message,
    f"message={td_obs.message[:80]}",
)
check(
    "query_topology_diff: no changes for easy",
    "No topology changes" in td_obs.message,
    f"message={td_obs.message[:80]}",
)
check(
    "query_topology_diff: budget unchanged (free)",
    td_obs.queries_remaining == obs_t.queries_remaining,
    f"remaining={td_obs.queries_remaining}",
)

# query_topology_diff with mutations: find a hard seed, step past mutation threshold
for test_seed in range(100):
    env_td = ServiceImpactEnvironment()
    obs_td = env_td.reset(seed=test_seed)
    if env_td._task_difficulty == "hard":
        svc_list = sorted(env_td._graph.nodes())
        # Step 4 times to trigger hard mutation at step 4
        for i in range(4):
            env_td.step(ServiceImpactAction(
                action_type="query_dependents",
                service_name=svc_list[i % len(svc_list)],
                affected_services=[],
            ))
        td_hard = env_td.step(ServiceImpactAction(
            action_type="query_topology_diff",
            service_name="",
            affected_services=[],
        ))
        check(
            "query_topology_diff: shows mutations after hard step 4",
            "mutation" in td_hard.message.lower() or "Step" in td_hard.message,
            f"message={td_hard.message[:120]}",
        )
        break

# ── Section 13: TrajectoryAuditor ─────────────────────────────────────────────
print("\n─── Section 13: TrajectoryAuditor ───")

from server.trajectory_auditor import TrajectoryAuditor, AuditReport

auditor = TrajectoryAuditor()

# Audit an episode that was logged (seed=42 from section 8)
report = auditor.audit_episode(42)
if report:
    check("Auditor: report has seed", report.seed == 42, f"seed={report.seed}")
    check("Auditor: report has difficulty", report.difficulty != "unknown", f"difficulty={report.difficulty}")
    check("Auditor: report has strategy", report.strategy != "unknown", f"strategy={report.strategy}")
    check("Auditor: hypothesis_scores is list", isinstance(report.hypothesis_scores, list), "")
else:
    # No log exists — just check the class works
    check("Auditor: returns None for missing seed", report is None, "")
    # Create a fresh episode to audit
    check("Auditor: AuditReport constructible",
          AuditReport(seed=0).seed == 0, "")

# Summary works even with no/few logs
summary = auditor.summary()
check(
    "Auditor: summary returns dict with 'episodes' key",
    isinstance(summary, dict) and "episodes" in summary,
    f"keys={list(summary.keys())}",
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
