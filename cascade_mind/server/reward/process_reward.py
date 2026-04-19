"""process_reward.py — Step-level intermediate reward signals.

The standard OpenEnv reward is episode-terminal (F-beta at submission).
The Process Reward Model (PRM) supplements this with dense, per-step signals
so that GRPO / PPO training can credit useful intermediate actions.

Two signals
-----------
1. Information-gain bonus  — reward for reducing uncertainty this step.
2. Delta-F-beta bonus      — reward for improving the intermediate F-beta score.

Both are scaled to keep them well below the terminal reward (which is ≤ 1.0).
"""

from __future__ import annotations

from typing import FrozenSet


# Scaling factors — keep step rewards small relative to terminal reward
_IG_SCALE    = 0.10   # max +0.10 per step for pure information gain
_DFBETA_SCALE = 0.05   # max +0.05 per step for delta-F-beta improvement
_CONTRADICTION_PENALTY = 0.03  # applied per newly-detected contradiction


def compute_intermediate_fbeta(
    high_conf_set: FrozenSet[str],
    true_affected: FrozenSet[str],
    beta: float = 2.0,
) -> float:
    """F-beta(β) of the agent's current high-confidence prediction.

    Args:
        high_conf_set: Services where belief_state > 0.5.
        true_affected: Ground-truth affected services.
        beta: F-beta parameter (default 2 = recall-heavy).

    Returns:
        Float in [0.0, 1.0].
    """
    tp = len(high_conf_set & true_affected)
    fp = len(high_conf_set - true_affected)
    fn = len(true_affected - high_conf_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def compute_step_reward(
    information_gain: float,
    prev_fbeta: float,
    current_fbeta: float,
    new_contradictions: int = 0,
) -> float:
    """Combine IG and delta-F-beta into a single step reward.

    Args:
        information_gain: Output of BeliefTracker.compute_information_gain().
        prev_fbeta: intermediate_fbeta from the previous step (0.0 if first).
        current_fbeta: intermediate_fbeta for this step.
        new_contradictions: Number of NEW contradictions fired this step.

    Returns:
        Step reward in approximately [-0.1, 0.15].
    """
    ig_bonus     = information_gain * _IG_SCALE
    delta_fbeta  = max(0.0, current_fbeta - prev_fbeta)
    fbeta_bonus  = delta_fbeta * _DFBETA_SCALE
    penalty      = new_contradictions * _CONTRADICTION_PENALTY
    return ig_bonus + fbeta_bonus - penalty
