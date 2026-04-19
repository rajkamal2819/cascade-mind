"""
server/mutation_engine.py
--------------------------
MutationEngine — mid-episode graph topology mutations for cascade-mind v2.

Simulates real-world SRE scenarios where the service topology changes during
an incident investigation (e.g., failovers, auto-scaling, dependency updates).

Design:
  - Mutations occur at configurable step thresholds (e.g., step 5, 10)
  - Each mutation adds 1-2 edges or removes 1 edge, seed-deterministic
  - The agent receives a [TOPOLOGY ALERT] message when a mutation occurs
  - Ground truth is recomputed lazily at submit time from the current graph

Difficulty controls mutation frequency:
  easy   → 0 mutations (stable topology)
  medium → 1 mutation  (at step 5)
  hard   → 2 mutations (at steps 4 and 8)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


@dataclass
class MutationEvent:
    """Record of a single graph mutation."""
    step_num: int
    mutation_type: str   # "add_edge" | "remove_edge"
    edges: List[Tuple[str, str]]
    description: str


@dataclass
class MutationConfig:
    """Per-difficulty mutation schedule."""
    thresholds: List[int]    # step numbers where mutations trigger
    max_add: int = 2         # max edges to add per mutation
    max_remove: int = 1      # max edges to remove per mutation


# Default mutation schedules per difficulty
DEFAULT_SCHEDULES: Dict[str, MutationConfig] = {
    "easy":   MutationConfig(thresholds=[]),           # no mutations
    "medium": MutationConfig(thresholds=[5]),           # 1 mutation at step 5
    "hard":   MutationConfig(thresholds=[4, 8]),        # 2 mutations
}


class MutationEngine:
    """Applies seed-deterministic graph mutations at configured step thresholds.

    Usage:
        engine = MutationEngine(seed=42, difficulty="hard")
        mutated, event = engine.maybe_mutate(graph, step_num=5)
        if event:
            print(event.description)  # "[TOPOLOGY ALERT] ..."
    """

    def __init__(
        self,
        seed: int,
        difficulty: str = "easy",
        schedules: Optional[Dict[str, MutationConfig]] = None,
    ) -> None:
        self._seed = seed
        self._difficulty = difficulty
        self._schedules = schedules or DEFAULT_SCHEDULES
        self._config = self._schedules.get(difficulty, MutationConfig(thresholds=[]))
        self._rng = random.Random(seed + 77777)  # separate RNG for mutations
        self._applied: List[MutationEvent] = []  # history of applied mutations
        self._triggered_steps: Set[int] = set()   # prevent double-triggering

    def maybe_mutate(
        self,
        graph: nx.DiGraph,
        step_num: int,
    ) -> Tuple[nx.DiGraph, Optional[MutationEvent]]:
        """Check if a mutation should occur at this step; if so, apply it.

        Returns:
            (graph, event) — event is None if no mutation was triggered.
            The graph is mutated in place AND returned for convenience.
        """
        if step_num not in self._config.thresholds:
            return graph, None

        if step_num in self._triggered_steps:
            return graph, None

        self._triggered_steps.add(step_num)

        # Decide mutation type: 60% add edges, 40% remove edges
        if self._rng.random() < 0.6:
            event = self._add_edges(graph, step_num)
        else:
            event = self._remove_edges(graph, step_num)

        self._applied.append(event)
        return graph, event

    @property
    def mutations_applied(self) -> List[MutationEvent]:
        """Return history of all mutations applied this episode."""
        return list(self._applied)

    @property
    def total_mutations(self) -> int:
        return len(self._applied)

    # ── Mutation operations ───────────────────────────────────────────────

    def _add_edges(self, G: nx.DiGraph, step_num: int) -> MutationEvent:
        """Add 1-2 plausible edges that don't create cycles."""
        all_nodes = sorted(G.nodes())
        n_add = self._rng.randint(1, self._config.max_add)
        added = []

        for _ in range(n_add * 3):  # try up to 3× to find valid edges
            if len(added) >= n_add:
                break
            u = self._rng.choice(all_nodes)
            v = self._rng.choice(all_nodes)
            if u == v or G.has_edge(u, v):
                continue
            # Don't create cycles
            try:
                if nx.has_path(G, v, u):
                    continue
            except nx.NetworkXError:
                continue
            G.add_edge(u, v)
            added.append((u, v))

        if not added:
            # Fallback: just report no-op
            return MutationEvent(
                step_num=step_num,
                mutation_type="add_edge",
                edges=[],
                description="[TOPOLOGY ALERT] A topology change was detected but no new dependencies were added.",
            )

        edge_desc = ", ".join(f"{u} → {v}" for u, v in added)
        return MutationEvent(
            step_num=step_num,
            mutation_type="add_edge",
            edges=added,
            description=(
                f"[TOPOLOGY ALERT] New service dependencies detected: {edge_desc}. "
                f"This may change the blast radius. Consider re-investigating affected services."
            ),
        )

    def _remove_edges(self, G: nx.DiGraph, step_num: int) -> MutationEvent:
        """Remove 1 edge that has an alternative path (safe removal)."""
        candidates = []
        for u, v in list(G.edges()):
            G_tmp = G.copy()
            G_tmp.remove_edge(u, v)
            try:
                if nx.has_path(G_tmp, u, v):
                    candidates.append((u, v))
            except nx.NetworkXError:
                pass

        if not candidates:
            return MutationEvent(
                step_num=step_num,
                mutation_type="remove_edge",
                edges=[],
                description="[TOPOLOGY ALERT] A failover was detected but the topology remains unchanged.",
            )

        n_remove = min(self._config.max_remove, len(candidates))
        to_remove = self._rng.sample(sorted(candidates), n_remove)
        G.remove_edges_from(to_remove)

        edge_desc = ", ".join(f"{u} → {v}" for u, v in to_remove)
        return MutationEvent(
            step_num=step_num,
            mutation_type="remove_edge",
            edges=to_remove,
            description=(
                f"[TOPOLOGY ALERT] Service dependency removed (failover/deprecation): {edge_desc}. "
                f"The blast radius may have changed."
            ),
        )
