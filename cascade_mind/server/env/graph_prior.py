"""graph_prior.py — Session-scoped edge frequency table.

After every completed episode the environment calls GraphPrior.update() with
the seed service and the set of queried edges.  Across a session the prior
converges to a frequency table so that the agent (and any reinforcement
learning loop) can warm-start with a prior over likely causal edges.

Design
------
* Keys: "source→target"
* Values: hit-count / total-episodes  (float in [0.0, 1.0])
* Stored in memory for the life of the server process; NOT persisted to disk
  (persistence is handled by TrajectoryLogger if needed).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, FrozenSet, Iterable


class GraphPrior:
    """Session-level edge-frequency tracker."""

    def __init__(self) -> None:
        self._hit_count: Counter = Counter()
        self._episode_count: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        seed_service: str,
        true_affected: FrozenSet[str],
    ) -> None:
        """Record a completed episode.

        Args:
            seed_service: The changed service for that episode.
            true_affected: Ground-truth affected service set (from graph BFS).
        """
        self._episode_count += 1
        for target in true_affected:
            key = f"{seed_service}→{target}"
            self._hit_count[key] += 1

    def get_prior(self) -> Dict[str, float]:
        """Return edge → hit-frequency dict.  Empty if no episodes yet."""
        if self._episode_count == 0:
            return {}
        return {
            k: v / self._episode_count
            for k, v in self._hit_count.items()
        }

    def top_k(self, k: int = 10) -> Dict[str, float]:
        """Return the k highest-confidence edges."""
        prior = self.get_prior()
        sorted_items = sorted(prior.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:k])

    def to_observation_text(self, k: int = 8) -> str:
        """Render a human-readable prior summary for injection into prompts."""
        top = self.top_k(k)
        if not top:
            return "(no session prior available yet)"
        lines = [
            f"  {edge}: {freq:.0%} of prior episodes"
            for edge, freq in top.items()
        ]
        return "Session graph prior (top edges):\n" + "\n".join(lines)

    @property
    def episode_count(self) -> int:
        return self._episode_count
