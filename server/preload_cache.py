"""
server/preload_cache.py
-----------------------
CLI script to pre-warm the LLM simulator cache for seeds 0..N-1.

Bakes all LLM responses into a JSON file that gets copied into the Docker
image — eliminating all LLM latency during actual evaluation runs.

Usage:
    # Pre-warm 100 seeds (default)
    python -m server.preload_cache

    # Pre-warm 50 seeds to a specific path
    python -m server.preload_cache --seeds 50 --output /app/env/llm_sim_cache.json

    # Check what's already cached
    python -m server.preload_cache --status

Requires:
    HF_TOKEN env var with "Make calls to Inference Providers" permission.
    pip install huggingface_hub
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-warm the LLM simulator cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds",  type=int,  default=100,
                        help="Number of seeds to pre-warm (0..N-1)")
    parser.add_argument("--output", type=str,
                        default=os.environ.get("LLM_CACHE_PATH", "/tmp/llm_sim_cache.json"),
                        help="Cache output path")
    parser.add_argument("--status", action="store_true",
                        help="Show cache stats and exit")
    args = parser.parse_args()

    # ── Dual-import for in-repo vs installed ─────────────────────────────
    try:
        from .llm_simulator import LLMSimulator, SimulatorCache
        from .graph_builder import (
            build_service_graph, get_scenario,
            get_direct_dependents, get_direct_dependencies,
            SERVICE_METADATA,
        )
    except ImportError:
        try:
            from server.llm_simulator import LLMSimulator, SimulatorCache  # type: ignore
            from server.graph_builder import (  # type: ignore
                build_service_graph, get_scenario,
                get_direct_dependents, get_direct_dependencies,
                SERVICE_METADATA,
            )
        except ImportError:
            from llm_simulator import LLMSimulator, SimulatorCache  # type: ignore
            from graph_builder import (  # type: ignore
                build_service_graph, get_scenario,
                get_direct_dependents, get_direct_dependencies,
                SERVICE_METADATA,
            )

    # ── Cache status check ────────────────────────────────────────────────
    cache = SimulatorCache(cache_path=args.output)
    if args.status:
        print(f"Cache path : {args.output}")
        print(f"Entries    : {len(cache)}")
        return

    # ── Validate HF_TOKEN ─────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.error(
            "HF_TOKEN not set — cannot call LLM. "
            "Set HF_TOKEN with 'Make calls to Inference Providers' scope."
        )
        sys.exit(1)

    simulator = LLMSimulator(hf_token=hf_token, cache=cache, enabled=True)
    if not simulator.is_active:
        logger.error(
            "LLMSimulator not active. Check HF_TOKEN and huggingface_hub install."
        )
        sys.exit(1)

    # ── Pre-warm loop ──────────────────────────────────────────────────────
    logger.info(
        "Pre-warming LLM cache for seeds 0..%d → %s",
        args.seeds - 1, args.output
    )

    start = time.time()
    calls_made = 0
    calls_cached = 0

    for seed in range(args.seeds):
        G = build_service_graph(seed=seed)
        scenario    = get_scenario(G, seed)
        svc         = scenario["changed_service"]
        difficulty  = scenario["difficulty"]
        meta        = SERVICE_METADATA.get(svc, {})
        dependents  = get_direct_dependents(G, svc)
        dependencies = get_direct_dependencies(G, svc)
        all_svcs    = sorted(G.nodes())

        logger.info(
            "[seed=%3d  %-6s] changed=%s  affected=%d",
            seed, difficulty, svc, scenario.get("n_affected", 0),
        )

        tasks = [
            # (generator_fn, args_dict, description)
            (
                simulator.generate_incident_context,
                dict(seed=seed, changed_service=svc,
                     team=meta.get("team", "platform"),
                     language=meta.get("language", "python"),
                     tier=meta.get("tier", 2), difficulty=difficulty),
                "incident_context",
            ),
            (
                simulator.simulate_registry_query,
                dict(seed=seed, action_type="query_dependents",
                     service_name=svc, true_result=dependents,
                     all_services=all_svcs, team=meta.get("team", "platform"),
                     tier=meta.get("tier", 2), difficulty=difficulty),
                "registry_dependents",
            ),
            (
                simulator.simulate_registry_query,
                dict(seed=seed, action_type="query_dependencies",
                     service_name=svc, true_result=dependencies,
                     all_services=all_svcs, team=meta.get("team", "platform"),
                     tier=meta.get("tier", 2), difficulty=difficulty),
                "registry_dependencies",
            ),
            (
                simulator.generate_runbook,
                dict(seed=seed, service_name=svc, dependents=dependents,
                     dependencies=dependencies, team=meta.get("team", "platform"),
                     tier=meta.get("tier", 2), difficulty=difficulty),
                "runbook",
            ),
            (
                simulator.generate_monitoring,
                dict(seed=seed, service_name=svc, dependents=dependents,
                     dependencies=dependencies, difficulty=difficulty),
                "monitoring",
            ),
            (
                simulator.generate_changelog,
                dict(seed=seed, changed_service=svc,
                     team=meta.get("team", "platform"), difficulty=difficulty),
                "changelog",
            ),
        ]

        for fn, kwargs, label in tasks:
            try:
                fn(**kwargs)
                calls_made += 1
            except Exception as exc:
                logger.warning("  [seed=%d %s] failed: %s", seed, label, exc)
            else:
                logger.debug("  ✓ %s", label)

        # Progress heartbeat every 10 seeds
        if (seed + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (seed + 1) / elapsed
            remaining = (args.seeds - seed - 1) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d seeds  |  %.1fs elapsed  |  ~%.0fs remaining",
                seed + 1, args.seeds, elapsed, remaining,
            )

    elapsed = time.time() - start
    logger.info(
        "Cache pre-warm complete! %d entries in %s  (%.1fs total)",
        len(cache), args.output, elapsed,
    )


if __name__ == "__main__":
    main()
