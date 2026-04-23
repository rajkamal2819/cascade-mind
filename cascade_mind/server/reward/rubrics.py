"""
server/reward/rubrics.py
------------------------
OpenEnv Rubric integration for cascade-mind.

Wraps the existing reward pipeline into composable openenv-core Rubric objects
so judges can inspect, extend, and compose reward components via the standard
framework API (env.rubric.named_rubrics(), env.rubric.get_rubric("fbeta"), etc.).

Rubric hierarchy:
    CascadeMindRubric          ← top-level (WeightedSum pattern)
    ├── fbeta   : FBetaRubric        — F-beta(β) + overclaim penalty + budget bonus
    └── calibration : BrierScoreRubric  — belief-state calibration (Brier score)

Usage in environment:
    rubric = CascadeMindRubric(orchestrator)
    rubric.set_submit_context(predicted, correct, all_services, ...)
    reward = rubric(action, observation)   # → float in (0.001, 0.999)
    breakdown = rubric.last_breakdown      # → dict with full diagnostics
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

try:
    from openenv.core.rubrics.base import Rubric
except ImportError:
    # Fallback stub so the module loads without openenv-core
    class Rubric:  # type: ignore[no-redef]
        def __init__(self):
            object.__setattr__(self, "_rubric_children", {})
        def __setattr__(self, name, value):
            if isinstance(value, Rubric):
                self._rubric_children[name] = value
            object.__setattr__(self, name, value)
        def __call__(self, action, observation):
            return self.forward(action, observation)
        def forward(self, action, observation) -> float:
            raise NotImplementedError
        def named_rubrics(self, prefix=""):
            for name, child in self._rubric_children.items():
                full = f"{prefix}.{name}" if prefix else name
                yield full, child
                yield from child.named_rubrics(full)
        def reset(self): pass


class FBetaRubric(Rubric):
    """Terminal F-beta reward rubric.

    Wraps RewardOrchestrator.compute() to expose the existing rotating-profile
    F-beta scoring as a composable Rubric. Context must be set via set_context()
    before each call.

    Stores full diagnostic breakdown in self.last_breakdown after each call.
    """

    def __init__(self, orchestrator: Any = None) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._context: Dict[str, Any] = {}
        self.last_breakdown: Dict[str, Any] = {}

    def set_context(
        self,
        predicted: Set[str],
        correct: Set[str],
        all_services_count: int,
        queries_used: int,
        max_queries: int,
        profile: Any,
    ) -> None:
        self._context = {
            "predicted": predicted,
            "correct": correct,
            "all_services_count": all_services_count,
            "queries_used": queries_used,
            "max_queries": max_queries,
            "profile": profile,
        }

    def forward(self, action: Any, observation: Any) -> float:
        if not self._context or self._orchestrator is None:
            return 0.001
        data = self._orchestrator.compute(**self._context)
        self.last_breakdown = data
        # RewardOrchestrator already clamps to (0.001, 0.999)
        return data["reward"]


class BrierScoreRubric(Rubric):
    """Belief-state calibration rubric using Brier score.

    Rewards agents for having accurate confidence estimates across all services,
    not just for identifying the correct affected set. An agent that is 90%
    confident on truly affected services and 10% confident on safe services
    scores higher than one that is uniformly uncertain.

    Brier score = 1 - mean((confidence_i - true_label_i)^2)
    Range: [0, 1] where 1 = perfect calibration.

    Context must be set via set_context() before each call.
    """

    def __init__(self) -> None:
        super().__init__()
        self._context: Dict[str, Any] = {}
        self.last_brier: float = 0.5

    def set_context(
        self,
        belief_state: Dict[str, float],
        correct: Set[str],
        all_services: List[str],
    ) -> None:
        self._context = {
            "belief_state": belief_state,
            "correct": correct,
            "all_services": all_services,
        }

    def forward(self, action: Any, observation: Any) -> float:
        belief_state = self._context.get("belief_state", {})
        correct = self._context.get("correct", set())
        all_services = self._context.get("all_services", [])

        if not belief_state or not all_services:
            return 0.5  # neutral if no belief state available

        squared_errors = []
        for svc in all_services:
            confidence = belief_state.get(svc, 0.0)
            true_label = 1.0 if svc in correct else 0.0
            squared_errors.append((confidence - true_label) ** 2)

        brier_error = sum(squared_errors) / len(squared_errors)
        brier_score = 1.0 - brier_error  # higher = better calibrated
        self.last_brier = round(max(0.0, min(1.0, brier_score)), 4)
        return self.last_brier


class CascadeMindRubric(Rubric):
    """Top-level cascade-mind reward rubric.

    Composes FBetaRubric (task correctness) and BrierScoreRubric (belief
    calibration) into a weighted blend. Child rubrics are auto-registered
    via openenv-core's Rubric framework, enabling introspection:

        for name, r in env.rubric.named_rubrics():
            print(name, r.last_score)
        # → fbeta <FBetaRubric>
        # → calibration <BrierScoreRubric>

    Weights: 80% F-beta (task) + 20% Brier (calibration).
    Final reward is clamped to (0.001, 0.999) for validator compliance.
    """

    FBETA_WEIGHT = 0.80
    BRIER_WEIGHT = 0.20

    def __init__(self, orchestrator: Any = None) -> None:
        super().__init__()
        # Assigning Rubric instances as attributes auto-registers them as children
        self.fbeta = FBetaRubric(orchestrator)
        self.calibration = BrierScoreRubric()
        self.last_breakdown: Dict[str, Any] = {}

    def set_submit_context(
        self,
        predicted: Set[str],
        correct: Set[str],
        all_services: List[str],
        all_services_count: int,
        queries_used: int,
        max_queries: int,
        profile: Any,
        belief_state: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set all context needed for one submit scoring call."""
        self.fbeta.set_context(
            predicted=predicted,
            correct=correct,
            all_services_count=all_services_count,
            queries_used=queries_used,
            max_queries=max_queries,
            profile=profile,
        )
        self.calibration.set_context(
            belief_state=belief_state or {},
            correct=correct,
            all_services=all_services,
        )

    def forward(self, action: Any, observation: Any) -> float:
        fbeta_score = self.fbeta(action, observation)
        brier_score = self.calibration(action, observation)

        combined = self.FBETA_WEIGHT * fbeta_score + self.BRIER_WEIGHT * brier_score
        reward = round(max(0.001, min(0.999, combined)), 4)

        # Store full breakdown so environment can build the episode message
        self.last_breakdown = {
            **self.fbeta.last_breakdown,
            "brier_score": brier_score,
            "fbeta_component": fbeta_score,
            "brier_component": brier_score,
            "fbeta_weight": self.FBETA_WEIGHT,
            "brier_weight": self.BRIER_WEIGHT,
            "reward": reward,
        }
        return reward

    def reset(self) -> None:
        """Reset per-episode state on env.reset()."""
        self.fbeta._context = {}
        self.fbeta.last_breakdown = {}
        self.calibration._context = {}
        self.calibration.last_brier = 0.5
        self.last_breakdown = {}
