#!/usr/bin/env python3
"""
benchmark.py
-------------
Local benchmark for cascade-mind v2.

Runs the environment through a range of seeds and produces aggregated
performance statistics without requiring an LLM. Uses a simple heuristic
agent (BFS from changed service) to establish a performance baseline.

Usage:
  python benchmark.py                   # run 30 seeds with heuristic agent
  python benchmark.py --seeds 100       # run 100 seeds
  python benchmark.py --audit           # also run trajectory auditor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# Disable LLM simulator for benchmarking
os.environ.setdefault("LLM_SIMULATOR_ENABLED", "false")

# Imports
try:
    from server.service_impact_environment import ServiceImpactEnvironment
    from models import ServiceImpactAction
    from server.graph_builder import SERVICES
except ImportError:
    sys.exit("Run from the cascade-mind root directory.")

try:
    from server.trajectory_auditor import TrajectoryAuditor
except ImportError:
    TrajectoryAuditor = None


def heuristic_agent(env: ServiceImpactEnvironment, seed: int, verbose: bool = False) -> Dict[str, Any]:
    """Simple heuristic agent: free intel → BFS → submit.

    This establishes a baseline that any LLM agent should beat.
    """
    obs = env.reset(seed=seed)
    changed = obs.changed_service
    budget = obs.queries_remaining
    difficulty = env._task_difficulty

    candidates = set()
    steps = 0
    start = time.time()

    # Phase 1: Free intel (always do all 3)
    for action_type in ["query_changelog", "query_runbook", "query_monitoring"]:
        obs = env.step(ServiceImpactAction(
            action_type=action_type,
            service_name=changed,
            affected_services=[],
        ))
        steps += 1
        # Extract service names from message
        for svc in SERVICES:
            if svc in obs.message and svc != changed:
                candidates.add(svc)

    # Phase 2: BFS with query_dependents
    queried = set()
    to_query = [changed]  # start from changed service
    while to_query and env._queries_used < env._max_queries - 2:  # leave buffer for submit
        svc = to_query.pop(0)
        if svc in queried:
            continue
        queried.add(svc)

        obs = env.step(ServiceImpactAction(
            action_type="query_dependents",
            service_name=svc,
            affected_services=[],
        ))
        steps += 1

        # Extract new service names
        for s in SERVICES:
            if s in obs.message and s != changed and s not in queried:
                candidates.add(s)
                if s not in to_query:
                    to_query.append(s)

        if obs.done:
            break

    # Phase 3: Submit
    if not env._episode_ended:
        obs = env.step(ServiceImpactAction(
            action_type="submit",
            affected_services=sorted(candidates),
        ))
        steps += 1

    elapsed = time.time() - start
    reward = obs.reward if obs.reward is not None else 0.0
    reward = max(0.001, min(0.999, reward))

    if verbose:
        print(f"  seed={seed:4d}  difficulty={difficulty:<6s}  "
              f"reward={reward:.3f}  candidates={len(candidates):2d}  "
              f"steps={steps:2d}  time={elapsed:.2f}s")

    return {
        "seed": seed,
        "difficulty": difficulty,
        "reward": reward,
        "candidates": len(candidates),
        "steps": steps,
        "elapsed_s": round(elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="cascade-mind v2 benchmark")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds to run")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--audit", action="store_true", help="Run trajectory auditor")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-seed results")
    args = parser.parse_args()

    print(f"{'='*65}")
    print(f"  cascade-mind v2 — Heuristic Agent Benchmark")
    print(f"  Seeds: {args.start_seed} to {args.start_seed + args.seeds - 1}")
    print(f"{'='*65}")

    results: List[Dict[str, Any]] = []
    by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}

    start_total = time.time()
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        env = ServiceImpactEnvironment()
        result = heuristic_agent(env, seed, verbose=args.verbose)
        results.append(result)
        by_difficulty[result["difficulty"]].append(result["reward"])

    total_elapsed = time.time() - start_total

    # Summary
    print(f"\n{'─'*65}")
    all_rewards = [r["reward"] for r in results]
    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    for diff in ["easy", "medium", "hard"]:
        scores = by_difficulty[diff]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {diff:<8s}  n={len(scores):3d}  mean={avg:.3f}  "
                  f"min={min(scores):.3f}  max={max(scores):.3f}")
        else:
            print(f"  {diff:<8s}  n=  0")

    print(f"{'─'*65}")
    print(f"  OVERALL   n={len(results):3d}  mean={mean_reward:.3f}  time={total_elapsed:.1f}s")
    print(f"{'='*65}")

    # Audit
    if args.audit and TrajectoryAuditor is not None:
        print(f"\n{'─'*65}")
        print("  Trajectory Audit")
        print(f"{'─'*65}")
        auditor = TrajectoryAuditor()
        summary = auditor.summary()
        print(f"  Episodes audited: {summary.get('episodes', 0)}")
        print(f"  Mean reward:      {summary.get('mean_reward', 0):.3f}")
        print(f"  Budget util:      {summary.get('mean_budget_utilization', 0):.1%}")
        print(f"  Strategies:       {json.dumps(summary.get('strategy_distribution', {}))}")
        print(f"  Hypothesis used:  {summary.get('hypothesis_usage', 0)}")

    # JSON output
    print(f"\nJSON_BENCHMARK: {json.dumps({'mean_reward': round(mean_reward, 4), 'seeds': args.seeds, 'by_difficulty': {d: round(sum(s)/len(s), 4) if s else 0 for d, s in by_difficulty.items()}})}")


if __name__ == "__main__":
    main()
