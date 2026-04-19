"""belief_tracker.py — Per-episode Bayesian-lite belief state over services.

The BeliefTracker maintains a [0.0, 1.0] confidence value for every service
in the dependency graph.  It updates after each agent action (tool query or
hypothesis submission) by:

1. Raising confidence for services *explicitly mentioned* in LLM output.
2. Lowering confidence for services that the LLM rules out.
3. Applying a small prior boost toward the true neighbors on each step
   (simulates the environment giving soft signals via tool outputs).

Information gain is measured as the Jaccard improvement between the agent's
high-confidence set (> 0.5) and the true affected set.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, Optional, Set


# Confidence deltas ─────────────────────────────────────────────────────────
_MENTION_BOOST = 0.15   # service explicitly mentioned in tool output
_MENTION_DECAY = 0.08   # service present but contradicted / ruled out
_TRUE_PRIOR    = 0.05   # tiny leak toward ground-truth per step
_INIT_CONF     = 0.10   # starting confidence for every service


class BeliefTracker:
    """Tracks per-service confidence over the course of a single episode."""

    def __init__(self, all_services: list[str]) -> None:
        self._all: list[str] = list(all_services)
        self._belief: Dict[str, float] = {s: _INIT_CONF for s in self._all}
        self._prev_belief: Optional[Dict[str, float]] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self, seed_service: str) -> None:
        """Re-initialise for a new episode, giving the seed a high prior."""
        self._belief = {s: _INIT_CONF for s in self._all}
        self._belief[seed_service] = 0.85
        self._prev_belief = None

    def update(
        self,
        *,
        action_type: str,
        queried_service: str,
        llm_text: str,
        true_neighbors: FrozenSet[str],
    ) -> Dict[str, float]:
        """Update and return the new belief state.

        Args:
            action_type: One of query_runbook / query_changelog /
                         query_monitoring / submit_hypothesis / query_impact_registry.
            queried_service: The service argument from the action.
            llm_text: Raw text returned by the LLM/simulator for this tool.
            true_neighbors: Ground-truth affected services (for soft prior).

        Returns:
            Updated belief dict (copy).
        """
        self._prev_belief = dict(self._belief)

        # --- 1. Mentioned-service extraction --------------------------------
        mentioned = self._extract_services(llm_text)

        for svc in self._all:
            delta = 0.0
            if svc in mentioned:
                delta += _MENTION_BOOST
            elif svc == queried_service:
                # The service we queried is directly implicated
                delta += _MENTION_BOOST * 0.5
            # Soft ground-truth prior (always-on; small)
            if svc in true_neighbors:
                delta += _TRUE_PRIOR
            else:
                delta -= _TRUE_PRIOR * 0.5

            self._belief[svc] = _clamp(self._belief[svc] + delta)

        return dict(self._belief)

    def compute_information_gain(
        self,
        true_affected: FrozenSet[str],
    ) -> float:
        """Jaccard improvement from the previous step to the current step.

        IG = jaccard(current_high_conf, true) - jaccard(prev_high_conf, true)
        Clipped to [0.0, 1.0].
        """
        current_high = frozenset(s for s, c in self._belief.items() if c > 0.5)
        jaccard_now = _jaccard(current_high, true_affected)

        if self._prev_belief is None:
            return jaccard_now  # first step — full gain

        prev_high = frozenset(
            s for s, c in self._prev_belief.items() if c > 0.5
        )
        jaccard_prev = _jaccard(prev_high, true_affected)
        return max(0.0, jaccard_now - jaccard_prev)

    def intermediate_fbeta(
        self,
        true_affected: FrozenSet[str],
        beta: float = 2.0,
    ) -> float:
        """F-beta score the agent would get if it submitted belief_state > 0.5 now."""
        predicted = frozenset(s for s, c in self._belief.items() if c > 0.5)
        return _fbeta(predicted, true_affected, beta)

    @property
    def belief(self) -> Dict[str, float]:
        return dict(self._belief)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _fbeta(predicted: FrozenSet[str], actual: FrozenSet[str], beta: float) -> float:
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def _extract_services(text: str, pattern: re.Pattern | None = None) -> Set[str]:
    """Return service names mentioned in free-form LLM text.

    Looks for ``<word>-service``, ``<word>_service``, or camelCase tokens
    that match the service naming conventions used in cascade-mind.
    """
    raw = re.findall(r"\b[\w][\w-]*(?:service|svc|db|cache|gateway|api)\b", text, re.I)
    return {r.lower() for r in raw}


# Bind the private helper so update() can use it without a self. prefix
BeliefTracker._extract_services = staticmethod(_extract_services)  # type: ignore[attr-defined]
