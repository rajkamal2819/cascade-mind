"""contradiction_engine.py — Detects when tool outputs disagree about a service.

A contradiction occurs when two different tools queried for the *same* service
return conflicting sets of downstream dependencies within the same episode.
Contradictions are important training signals: they indicate ambiguous or
stale knowledge in the underlying registry.

Detection heuristic
-------------------
* Each tool type (runbook / changelog / monitoring / registry) maintains a
  set of *services mentioned* for each queried service.
* If tool B mentions service X about service S, but tool A *did not* mention
  service X despite being queried for S, we record a potential contradiction.
* We use a threshold: the symmetric-difference fraction > `threshold` triggers
  a contradiction event.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


_CONTRADICTION_THRESHOLD = 0.4   # fraction of sym-diff vs union to fire


@dataclass
class ContradictionEvent:
    queried_service: str
    tool_a: str
    tool_b: str
    services_in_a: frozenset
    services_in_b: frozenset
    score: float   # sym-diff / union

    def to_text(self) -> str:
        diff_ab = self.services_in_a - self.services_in_b
        diff_ba = self.services_in_b - self.services_in_a
        parts = []
        if diff_ab:
            parts.append(f"{self.tool_a} mentions {sorted(diff_ab)} but {self.tool_b} does not")
        if diff_ba:
            parts.append(f"{self.tool_b} mentions {sorted(diff_ba)} but {self.tool_a} does not")
        body = "; ".join(parts) if parts else "outputs differ"
        return (
            f"CONTRADICTION [{self.queried_service}]: {body} "
            f"(score={self.score:.2f})"
        )


class ContradictionEngine:
    """Accumulates tool outputs and fires when two tools disagree."""

    def __init__(self, threshold: float = _CONTRADICTION_THRESHOLD) -> None:
        self._threshold = threshold
        # {queried_service: {tool_type: frozenset(mentioned_services)}}
        self._records: Dict[str, Dict[str, frozenset]] = {}
        self._events: list[ContradictionEvent] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._records.clear()
        self._events.clear()

    def check(
        self,
        *,
        action_type: str,
        queried_service: str,
        llm_text: str,
    ) -> Optional[ContradictionEvent]:
        """Record this tool output and return a ContradictionEvent if a new
        contradiction is detected, else None."""
        mentioned = _extract_services(llm_text)
        frozen = frozenset(mentioned)

        if queried_service not in self._records:
            self._records[queried_service] = {}

        tool_results = self._records[queried_service]

        # Compare against every previously-seen tool for this service
        event: Optional[ContradictionEvent] = None
        for prev_tool, prev_set in tool_results.items():
            if prev_tool == action_type:
                continue
            score = _sym_diff_score(prev_set, frozen)
            if score > self._threshold:
                ev = ContradictionEvent(
                    queried_service=queried_service,
                    tool_a=prev_tool,
                    tool_b=action_type,
                    services_in_a=prev_set,
                    services_in_b=frozen,
                    score=score,
                )
                self._events.append(ev)
                event = ev  # return the most recent one

        # Store this tool's result (overwrite if repeated)
        tool_results[action_type] = frozen
        return event

    @property
    def events(self) -> list[ContradictionEvent]:
        return list(self._events)

    @property
    def count(self) -> int:
        return len(self._events)

    def descriptions(self) -> list[str]:
        return [e.to_text() for e in self._events]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _sym_diff_score(a: frozenset, b: frozenset) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a.symmetric_difference(b)) / len(union)


def _extract_services(text: str) -> Set[str]:
    raw = re.findall(r"\b[\w][\w-]*(?:service|svc|db|cache|gateway|api)\b", text, re.I)
    return {r.lower() for r in raw}
