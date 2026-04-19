"""
models.py
---------
Pydantic Action / Observation / State models for service_impact_env.

Action space:
  - query_dependents   → "who calls service X?"
  - query_dependencies → "what does service X call?"
  - submit             → final answer: list of affected services

Observation space:
  - changed_service     : the service that was modified this episode
  - result              : list of services returned by the current query
  - queries_remaining   : how many queries are left in the budget
  - message             : human-readable description of what happened
  - done / reward       : inherited from Observation base

State (internal bookkeeping, exposed via /state):
  - episode_id, step_count   : inherited
  - changed_service          : which service triggered this episode
  - queries_used             : running count of queries consumed
  - max_queries              : total budget for this episode
  - correct_affected         : ground-truth affected services (revealed at end)
  - predicted_affected       : agent's submitted answer
  - task_difficulty          : "easy" | "medium" | "hard"
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

# OpenEnv base classes -------------------------------------------------------
# These provide: done, reward, metadata on Observation
#                episode_id, step_count on State
#                metadata on Action
try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except ImportError:
    # Fallback for local dev without openenv-core installed yet
    from pydantic import BaseModel, ConfigDict
    from typing import Any, Dict

    class Action(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
        done: bool = Field(default=False)
        reward: float | int | bool | None = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)
        episode_id: Optional[str] = Field(default=None)
        step_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ServiceImpactAction(Action):
    """An action the agent can take in a service impact analysis episode.

    Three action types:
      query_dependents   – discover which services call `service_name`
      query_dependencies – discover what `service_name` depends on
      submit             – end the episode with a final list of affected services
    """

    action_type: Literal[
        "query_dependents",    # find who calls a service (uses budget)
        "query_dependencies",  # find what a service depends on (uses budget)
        "query_runbook",       # fetch Confluence-style runbook (FREE — no budget)
        "query_changelog",     # fetch PR/changelog for the changed service (FREE)
        "query_monitoring",    # fetch Datadog-style monitoring snapshot (FREE)
        "query_topology_diff", # show graph changes since episode start (FREE)
        "query_service_health",# fetch health status summary for a service (FREE)
        "submit_hypothesis",   # partial check — returns delayed_reward, does NOT end episode
        "submit",              # final answer — ends the episode
    ] = Field(
        ...,
        description=(
            "Type of action. Budget-consuming: 'query_dependents', 'query_dependencies'. "
            "Free (no budget cost): 'query_runbook', 'query_changelog', 'query_monitoring', "
            "'query_topology_diff', 'query_service_health'. "
            "Hypothesis check: 'submit_hypothesis' (returns partial F-beta, episode continues). "
            "Terminal: 'submit' (ends episode, scored with F-beta β=2)."
        ),
    )
    service_name: Optional[str] = Field(
        default=None,
        description="The service to query. Required for query_* actions.",
    )
    affected_services: Optional[List[str]] = Field(
        default=None,
        description=(
            "Your final answer: list of ALL services affected by the change. "
            "Required when action_type='submit' or 'submit_hypothesis'."
        ),
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Agent's confidence in the hypothesis (0.0-1.0). "
            "Used with 'submit_hypothesis' to track agent calibration."
        ),
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ServiceImpactObservation(Observation):
    """What the agent sees after each step.

    Fields:
      changed_service   – the service that has been changed this episode
      result            – list of services returned by the last query
      queries_remaining – how many queries the agent still has
      message           – plain-English description of what just happened
      done              – True when the episode is over (inherited)
      reward            – F1 score [0.0–1.0] on submit, else None (inherited)
    """

    changed_service: str = Field(
        default="",
        description="The service that triggered this impact-analysis episode.",
    )
    result: List[str] = Field(
        default_factory=list,
        description="Services returned by the last query action.",
    )
    queries_remaining: int = Field(
        default=0,
        ge=0,
        description="Number of query actions remaining in this episode.",
    )
    message: str = Field(
        default="",
        description="Human-readable description of the current step result.",
    )
    delayed_reward: Optional[float] = Field(
        default=None,
        description=(
            "Partial F-beta score from submit_hypothesis. "
            "None for all other actions. Does not end the episode."
        ),
    )

    # ── World Modeling fields (v3) ──────────────────────────────────────
    belief_state: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Per-service confidence [0.0–1.0] representing the agent's current "
            "internal world model. Updated after every query based on mentions, "
            "contradictions, and hypothesis feedback."
        ),
    )
    information_gain: Optional[float] = Field(
        default=None,
        description=(
            "Entropy reduction this step: Jaccard improvement of the agent's known "
            "set vs the true affected set. 0.0 = no new information, 1.0 = perfect step."
        ),
    )
    intermediate_fbeta: Optional[float] = Field(
        default=None,
        description=(
            "F-beta(β=2) score the agent would receive if it submitted right now "
            "using all services with belief_state > 0.5. Updated every step."
        ),
    )
    world_version: int = Field(
        default=0,
        description=(
            "Increments each time a MutationEngine event fires. "
            "Agent should re-investigate when world_version increases."
        ),
    )
    belief_drift: Optional[float] = Field(
        default=None,
        description=(
            "Jaccard distance between the previous and current correct_affected set. "
            "Non-zero only after a mutation — indicates how stale the agent's world model is."
        ),
    )
    contradiction_count: int = Field(
        default=0,
        description="Running count of tool-output contradictions detected this episode.",
    )
    graph_prior: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Session-level edge confidence estimates from previous episodes. "
            "Keys are 'service_a→service_b', values are [0.0–1.0] frequency. "
            "Provided at reset() if prior episodes exist in this session."
        ),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ServiceImpactState(State):
    """Internal episode state exposed via /state endpoint.

    Agents can read this to understand the current episode context.
    Ground-truth `correct_affected` is revealed only after `submit`.
    """

    changed_service: str = Field(default="", description="Service modified in this episode.")
    queries_used: int = Field(default=0, ge=0, description="Queries consumed so far.")
    max_queries: int = Field(default=15, gt=0, description="Total query budget.")
    correct_affected: List[str] = Field(
        default_factory=list,
        description="Ground-truth affected services (revealed after episode ends).",
    )
    predicted_affected: List[str] = Field(
        default_factory=list,
        description="Agent's submitted answer (populated after submit action).",
    )
    task_difficulty: str = Field(
        default="easy",
        description="Difficulty tier: 'easy' | 'medium' | 'hard'.",
    )
    episode_ended: bool = Field(default=False, description="True once the episode is complete.")

    # Free-action usage counters (v2: caps prevent unlimited free exploration)
    runbook_uses: int = Field(default=0, ge=0, description="Times query_runbook used this episode.")
    changelog_uses: int = Field(default=0, ge=0, description="Times query_changelog used this episode.")
    monitoring_uses: int = Field(default=0, ge=0, description="Times query_monitoring used this episode.")
    hypothesis_count: int = Field(default=0, ge=0, description="Times submit_hypothesis used this episode.")
    last_hypothesis_score: Optional[float] = Field(default=None, description="Partial F-beta from last hypothesis.")
    reward_profile: str = Field(
        default="recall_heavy",
        description="Active reward profile for this episode: recall_heavy | balanced | precision_heavy | efficiency.",
    )
    # ── World Modeling state (v3) ─────────────────────────────────────────
    graph_prior: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Session-level edge confidence from prior episodes. "
            "Keys are 'service_a→service_b', values are [0.0–1.0] hit frequency."
        ),
    )
    contradictions: List[str] = Field(
        default_factory=list,
        description="Human-readable descriptions of tool contradictions detected this episode.",
    )
    world_version: int = Field(
        default=0,
        description="Current graph mutation version; increments on each MutationEngine event.",
    )