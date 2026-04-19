"""
server/curriculum_scheduler.py
-------------------------------
CurriculumScheduler — difficulty-adaptive environment parameters for cascade-mind v2.

Controls per-difficulty configuration that shapes the agent's experience:
  - Noise injection rates (how often registry queries drop/add services)
  - Hint quality (how much info the reset message reveals)
  - Free-action caps (can be adjusted per difficulty)
  - Maximum hypothesis count
  - Visible service count (how many services appear in enumeration hints)

Design:
  - Easy:   low noise, generous hints, more budget, 3 hypotheses
  - Medium: moderate noise, standard hints, moderate budget, 3 hypotheses
  - Hard:   high noise, minimal hints, tight budget, 2 hypotheses, mutations

The scheduler is queried at reset() time and provides a CurriculumConfig
that the environment uses to parameterize the episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class CurriculumConfig:
    """Per-episode configuration derived from difficulty level."""

    difficulty: str

    # Noise: probability of dropping a real dependent from registry response
    drop_probability: float = 0.0
    # Noise: probability of injecting a fake dependent into registry response
    inject_probability: float = 0.0
    # Max fake services injected per query
    max_inject_per_query: int = 0

    # How many services to mention in the hint (0 = no enumeration hint)
    hint_visible_services: int = 0
    # Whether to reveal difficulty in the reset message
    reveal_difficulty: bool = True
    # Whether to reveal the exact number of affected services as a range
    reveal_affected_range: bool = False

    # Free-action caps (override per difficulty if desired)
    runbook_cap: int = 2
    changelog_cap: int = 2
    monitoring_cap: int = 3

    # Max hypotheses allowed
    max_hypotheses: int = 3

    # Budget multiplier (applied to base max_queries from scenario)
    budget_multiplier: float = 1.0


# Pre-defined difficulty configurations
CURRICULUM_CONFIGS: Dict[str, CurriculumConfig] = {
    "easy": CurriculumConfig(
        difficulty="easy",
        drop_probability=0.05,        # 5% chance to drop a real dependent
        inject_probability=0.10,      # 10% chance to inject a fake
        max_inject_per_query=1,
        hint_visible_services=12,     # Show 12 of 30 services as "known services"
        reveal_difficulty=True,
        reveal_affected_range=True,   # "Expected: 1-6 affected services"
        runbook_cap=3,                # More generous caps for easy
        changelog_cap=2,
        monitoring_cap=4,
        max_hypotheses=3,
        budget_multiplier=1.0,
    ),
    "medium": CurriculumConfig(
        difficulty="medium",
        drop_probability=0.15,        # 15% drop rate
        inject_probability=0.20,      # 20% inject rate
        max_inject_per_query=2,
        hint_visible_services=20,     # Show 20 of 30 services
        reveal_difficulty=True,
        reveal_affected_range=False,
        runbook_cap=2,
        changelog_cap=2,
        monitoring_cap=3,
        max_hypotheses=3,
        budget_multiplier=1.0,
    ),
    "hard": CurriculumConfig(
        difficulty="hard",
        drop_probability=0.25,        # 25% drop rate — significant info loss
        inject_probability=0.30,      # 30% inject rate — lots of fake services
        max_inject_per_query=3,
        hint_visible_services=30,     # Show all 30 (but noise makes it hard)
        reveal_difficulty=True,
        reveal_affected_range=False,
        runbook_cap=2,
        changelog_cap=2,
        monitoring_cap=3,
        max_hypotheses=2,             # Fewer hypothesis checks available
        budget_multiplier=1.0,
    ),
}


class CurriculumScheduler:
    """Provides difficulty-adaptive configuration for each episode.

    Usage:
        scheduler = CurriculumScheduler()
        config = scheduler.get_config("hard")
        # Use config.drop_probability, config.hint_visible_services, etc.
    """

    def __init__(
        self,
        configs: Optional[Dict[str, CurriculumConfig]] = None,
    ) -> None:
        self._configs = configs or CURRICULUM_CONFIGS

    def get_config(self, difficulty: str) -> CurriculumConfig:
        """Return the CurriculumConfig for the given difficulty.

        Falls back to medium if difficulty is unknown.
        """
        return self._configs.get(difficulty, self._configs["medium"])

    def get_hint_text(self, config: CurriculumConfig, changed_service: str, n_affected: int) -> str:
        """Generate difficulty-appropriate hint text for the reset message.

        Easy: reveals affected range and explicit investigation strategy.
        Medium: hints at scope but no numbers.
        Hard: minimal guidance.
        """
        lines = []

        if config.difficulty == "easy":
            lines.append(f"💡 Hint: The blast radius is small (estimated 1-6 services affected).")
            lines.append(f"Start with free actions on '{changed_service}', then BFS outward.")
            lines.append(f"You have generous free-action caps: runbook={config.runbook_cap}, monitoring={config.monitoring_cap}.")
        elif config.difficulty == "medium":
            lines.append(f"⚠️ This is a medium-difficulty incident. The blast radius is moderate.")
            lines.append(f"Use free actions first, then targeted BFS. Watch for noisy data.")
            lines.append(f"Topology mutations may occur mid-episode — watch for [TOPOLOGY ALERT].")
        else:  # hard
            lines.append(f"🔴 This is a hard incident. The blast radius is extensive.")
            lines.append(f"Registry data is noisy — cross-reference multiple sources.")
            lines.append(f"Expect topology mutations. Budget is tight. Hypotheses limited to {config.max_hypotheses}.")
            lines.append(f"Consider using submit_hypothesis to calibrate mid-investigation.")

        return "\n".join(lines)

    @property
    def available_difficulties(self) -> list:
        return sorted(self._configs.keys())
