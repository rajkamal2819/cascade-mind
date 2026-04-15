"""
server/reward_orchestrator.py
------------------------------
RewardOrchestrator — rotating reward profiles for cascade-mind v2.

Each episode is assigned a deterministic reward profile based on the seed.
This prevents agents from overfitting to a single scoring regime and teaches
them to adapt their investigation strategy based on the active profile.

Profiles:
  recall_heavy   — β=2.5, low overclaim penalty   → cast a wide net
  balanced       — β=1.5, moderate penalty         → balanced investigation
  precision_heavy— β=0.8, high overclaim penalty   → be selective
  efficiency     — β=2.0, budget bonus/penalty     → submit fast, use fewer queries

All profiles clamp final reward to (0.001, 0.999) for validator compliance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class RewardProfile:
    """Configuration for one reward scoring regime."""
    name: str
    beta: float                         # F-beta parameter (higher = more recall weight)
    overclaim_threshold: float = 0.6    # fraction of all services above which penalty kicks in
    overclaim_penalty_max: float = 0.3  # max deduction for extreme overclaiming
    budget_bonus_weight: float = 0.0    # bonus/penalty for budget efficiency (0 = disabled)
    description: str = ""


# ── Predefined profiles ──────────────────────────────────────────────────────

PROFILES: Dict[str, RewardProfile] = {
    "recall_heavy": RewardProfile(
        name="recall_heavy",
        beta=2.5,
        overclaim_threshold=0.7,
        overclaim_penalty_max=0.15,
        budget_bonus_weight=0.0,
        description=(
            "Recall-weighted (β=2.5). Missing a service is very costly. "
            "Overclaim penalty only triggers above 70% submission. "
            "Strategy: cast a wide net, include everything with any evidence."
        ),
    ),
    "balanced": RewardProfile(
        name="balanced",
        beta=1.5,
        overclaim_threshold=0.6,
        overclaim_penalty_max=0.25,
        budget_bonus_weight=0.0,
        description=(
            "Balanced scoring (β=1.5). Moderate weight on both precision and recall. "
            "Strategy: gather evidence from multiple sources before including a service."
        ),
    ),
    "precision_heavy": RewardProfile(
        name="precision_heavy",
        beta=0.8,
        overclaim_threshold=0.5,
        overclaim_penalty_max=0.4,
        budget_bonus_weight=0.0,
        description=(
            "Precision-weighted (β=0.8). False positives are very costly. "
            "Overclaim penalty triggers above 50% submission with heavy deduction. "
            "Strategy: only submit services with 2+ corroborating sources."
        ),
    ),
    "efficiency": RewardProfile(
        name="efficiency",
        beta=2.0,
        overclaim_threshold=0.6,
        overclaim_penalty_max=0.3,
        budget_bonus_weight=0.15,
        description=(
            "Efficiency-weighted (β=2.0 + budget bonus). "
            "Bonus for using fewer queries, penalty for exhausting budget. "
            "Strategy: submit early when confident, don't waste queries on low-value targets."
        ),
    ),
}

PROFILE_ORDER = ["recall_heavy", "balanced", "precision_heavy", "efficiency"]


class RewardOrchestrator:
    """Selects and applies reward profiles per episode.

    Usage:
        orch = RewardOrchestrator()
        profile = orch.get_profile(seed=42)
        reward = orch.compute(
            predicted={"auth_service", "cart_service"},
            correct={"auth_service", "cart_service", "order_service"},
            all_services=30,
            queries_used=5,
            max_queries=12,
            profile=profile,
        )
    """

    def __init__(self, profiles: Optional[Dict[str, RewardProfile]] = None) -> None:
        self._profiles = profiles or PROFILES
        self._order = list(self._profiles.keys())

    def get_profile(self, seed: int) -> RewardProfile:
        """Deterministically select a reward profile from the seed."""
        idx = seed % len(self._order)
        return self._profiles[self._order[idx]]

    def compute(
        self,
        predicted: Set[str],
        correct: Set[str],
        all_services_count: int,
        queries_used: int = 0,
        max_queries: int = 15,
        profile: Optional[RewardProfile] = None,
    ) -> Dict[str, float]:
        """Compute reward using the given profile.

        Returns dict with:
          reward      — final clamped score
          fbeta       — raw F-beta before penalties
          precision   — TP / (TP + FP)
          recall      — TP / (TP + FN)
          overclaim_penalty — deduction for overclaiming
          budget_bonus      — bonus/penalty for budget efficiency
        """
        if profile is None:
            profile = PROFILES["recall_heavy"]  # default fallback

        tp = len(predicted & correct)
        fp = len(predicted - correct)
        fn = len(correct   - predicted)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta      = profile.beta
        fbeta     = (
            (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
            if (beta ** 2 * precision + recall) > 0
            else 0.0
        )

        # Overclaiming penalty
        overclaim_penalty = 0.0
        n_pred = len(predicted)
        if all_services_count > 0 and n_pred / all_services_count > profile.overclaim_threshold:
            oversubmit_frac = min(
                1.0,
                (n_pred / all_services_count - profile.overclaim_threshold)
                / (1.0 - profile.overclaim_threshold),
            )
            overclaim_penalty = profile.overclaim_penalty_max * oversubmit_frac

        # Budget efficiency bonus (only for "efficiency" profile)
        # Only applies if the agent actually found something (fbeta > 0)
        budget_bonus = 0.0
        if profile.budget_bonus_weight > 0 and max_queries > 0 and fbeta > 0:
            # Bonus range: +0.15 (used 0 queries) to -0.15 (exhausted budget)
            efficiency_ratio = 1.0 - (queries_used / max_queries)
            budget_bonus = profile.budget_bonus_weight * (2 * efficiency_ratio - 1)

        raw_reward = fbeta - overclaim_penalty + budget_bonus

        # Clamp to open interval (0, 1) — validator compliance
        reward = round(max(0.001, min(0.999, raw_reward)), 4)

        return {
            "reward": reward,
            "fbeta": round(fbeta, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "overclaim_penalty": round(overclaim_penalty, 4),
            "budget_bonus": round(budget_bonus, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "profile": profile.name,
        }

    @property
    def available_profiles(self) -> List[str]:
        """Return list of profile names."""
        return list(self._profiles.keys())
