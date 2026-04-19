"""
server/service_impact_environment.py
--------------------------------------
Core environment logic for service_impact_env.

LLM Simulator Architecture (Option A):
  - Ground truth (networkx) is NEVER exposed in result[] during queries.
  - Agents see only LLM-generated SRE tool output in observation.message.
  - result[] is populated ONLY on submit (reveals ground truth for postmortem).
  - Llama-3.1-8B via Cerebras (HF Inference Providers) generates realistic,
    noisy PagerDuty alerts, registry lookups, runbooks, and monitoring output.
  - Graceful degradation: if HF_TOKEN absent → template string fallback.

Episode lifecycle:
  1. reset(seed) → build seed-perturbed graph, pick scenario, generate incident
  2. step(query) → return LLM-generated registry/runbook/monitoring (result=[])
  3. step(submit) → score F-beta(β=2) vs ground truth, reveal correct in result[]

Reward design:
  - query step (new svc)   : +0.05
  - query step (re-query)  : -0.05
  - free action (runbook…) : None
  - budget exhaustion      : -0.4  (penalizes not submitting)
  - submit                 : F-beta(β=2) ∈ [0.0, 1.0] minus overclaiming penalty
"""
from __future__ import annotations

import os
import random
from typing import Any, List, Optional, Set
from uuid import uuid4

# OpenEnv base ---------------------------------------------------------------
try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    from abc import ABC, abstractmethod
    class Environment(ABC):  # type: ignore[no-redef]
        def _reset_rubric(self): pass
        @abstractmethod
        def reset(self, seed=None, episode_id=None, **kwargs): ...
        @abstractmethod
        def step(self, action, timeout_s=None, **kwargs): ...
        @property
        @abstractmethod
        def state(self): ...

# Dual-import for in-repo vs Docker ------------------------------------------
try:
    from ...models import ServiceImpactAction, ServiceImpactObservation, ServiceImpactState
except ImportError:
    from cascade_mind.models import ServiceImpactAction, ServiceImpactObservation, ServiceImpactState  # type: ignore

try:
    from ..graph.graph_builder import (
        build_service_graph,
        get_affected_services,
        get_direct_dependents,
        get_direct_dependencies,
        get_all_services,
        get_scenario,
        SERVICE_METADATA,
    )
except ImportError:
    from cascade_mind.server.graph.graph_builder import (  # type: ignore
        build_service_graph,
        get_affected_services,
        get_direct_dependents,
        get_direct_dependencies,
        get_all_services,
        get_scenario,
        SERVICE_METADATA,
    )

try:
    from ..simulator.llm_simulator import LLMSimulator, SimulatorCache
except ImportError:
    from cascade_mind.server.simulator.llm_simulator import LLMSimulator, SimulatorCache  # type: ignore

try:
    from ..trajectory.trajectory_logger import TrajectoryLogger
except ImportError:
    try:
        from cascade_mind.server.trajectory.trajectory_logger import TrajectoryLogger  # type: ignore
    except ImportError:
        TrajectoryLogger = None  # type: ignore

try:
    from ..reward.reward_orchestrator import RewardOrchestrator
except ImportError:
    try:
        from cascade_mind.server.reward.reward_orchestrator import RewardOrchestrator  # type: ignore
    except ImportError:
        RewardOrchestrator = None  # type: ignore

try:
    from ..graph.mutation_engine import MutationEngine
except ImportError:
    try:
        from cascade_mind.server.graph.mutation_engine import MutationEngine  # type: ignore
    except ImportError:
        MutationEngine = None  # type: ignore

try:
    from .curriculum_scheduler import CurriculumScheduler
except ImportError:
    try:
        from cascade_mind.server.env.curriculum_scheduler import CurriculumScheduler  # type: ignore
    except ImportError:
        CurriculumScheduler = None  # type: ignore

try:
    from .belief_tracker import BeliefTracker
except ImportError:
    try:
        from cascade_mind.server.env.belief_tracker import BeliefTracker  # type: ignore
    except ImportError:
        BeliefTracker = None  # type: ignore

try:
    from .contradiction_engine import ContradictionEngine
except ImportError:
    try:
        from cascade_mind.server.env.contradiction_engine import ContradictionEngine  # type: ignore
    except ImportError:
        ContradictionEngine = None  # type: ignore

try:
    from .graph_prior import GraphPrior
except ImportError:
    try:
        from cascade_mind.server.env.graph_prior import GraphPrior  # type: ignore
    except ImportError:
        GraphPrior = None  # type: ignore

try:
    from ..reward.process_reward import compute_intermediate_fbeta, compute_step_reward
except ImportError:
    try:
        from cascade_mind.server.reward.process_reward import compute_intermediate_fbeta, compute_step_reward  # type: ignore
    except ImportError:
        compute_intermediate_fbeta = None  # type: ignore
        compute_step_reward = None  # type: ignore

import networkx as nx


class ServiceImpactEnvironment(
    Environment  # type: ignore[misc]
):
    """Cross-service impact analysis environment with LLM-simulated observations.

    Agents observe realistic, noisy SRE tool output generated by Llama-3.1-8B
    (PagerDuty alerts, service registry responses, runbooks, monitoring data).
    Scoring is always against the deterministic networkx ground truth graph.

    Key properties:
    - 10,000+ unique episodes via seed-perturbed graph topology
    - Option A: agents must reason from noisy text — result[] is always [] during queries
    - F-beta (β=2) reward: recall weighted 4× over precision (SRE cost model)
    - Overclaiming penalty for submitting > 60% of all services
    - Free actions: query_runbook, query_changelog, query_monitoring (no budget cost)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    name:        str = "service_impact_env"
    description: str = (
        "Cross-service impact analysis environment. "
        "Agents identify downstream services affected by a microservice change. "
        "Observations are LLM-generated noisy SRE tool output (Llama-3.1-8B). "
        "Reward = F-beta(β=2) of predicted vs ground-truth affected set."
    )
    version: str = "0.2.0"
    author:  str = "Rajkamal2819"

    def get_metadata(self):
        """Return environment metadata for the /metadata endpoint."""
        try:
            from openenv.core.env_server.types import EnvironmentMetadata
        except ImportError:
            from openenv.core.env_server.interfaces import EnvironmentMetadata  # type: ignore
        return EnvironmentMetadata(
            name=self.name,
            version=self.version,
            description=(
                "cascade-mind simulates an SRE incident-response workflow. "
                "An agent investigates a production change in one microservice and must "
                "identify every downstream service that could be affected \u2014 using realistic, "
                "LLM-generated tool outputs (alerts, service registry, runbooks, monitoring). "
                "Observations are generated by Llama-3.1-8B via Cerebras. "
                "Scored with F-beta(\u03b2=2) against the ground-truth networkx dependency graph."
            ),
            author=self.author,
            documentation_url="https://huggingface.co/spaces/Rajkamal2819/cascade-mind",
        )

    def __init__(self) -> None:
        super().__init__()

        # LLM simulator (optional — graceful fallback if HF_TOKEN not set)
        llm_enabled = os.environ.get("LLM_SIMULATOR_ENABLED", "true").lower() == "true"
        cache_path  = os.environ.get("LLM_CACHE_PATH", "/tmp/llm_sim_cache.json")
        self._simulator = LLMSimulator(
            hf_token=os.environ.get("HF_TOKEN", ""),
            cache=SimulatorCache(cache_path=cache_path),
            enabled=llm_enabled,
        )

        # Free-action caps (per episode)
        self.FREE_ACTION_CAPS = {"query_runbook": 2, "query_changelog": 2, "query_monitoring": 3}

        # Trajectory logger (v2: logs every step for auditing)
        trajectory_dir = os.environ.get("TRAJECTORY_DIR", "/tmp/cascade_trajectories")
        if TrajectoryLogger is not None:
            self._trajectory_logger = TrajectoryLogger(trajectory_dir)
        else:
            self._trajectory_logger = None

        # Reward orchestrator (v2: rotating reward profiles)
        if RewardOrchestrator is not None:
            self._reward_orchestrator = RewardOrchestrator()
        else:
            self._reward_orchestrator = None
        self._reward_profile = None
        self._mutation_engine = None  # initialized per-episode in reset()
        self._pending_mutation_alert = ""  # set each step by mutation check

        # Curriculum scheduler (v2: difficulty-adaptive parameters)
        if CurriculumScheduler is not None:
            self._curriculum = CurriculumScheduler()
        else:
            self._curriculum = None
        self._curriculum_config = None  # set per-episode in reset()

        # ── World Modeling components (v3) ──────────────────────────────
        # Instantiated once; reset per-episode
        self._belief_tracker = BeliefTracker(all_services=[]) if BeliefTracker else None
        self._contradiction_engine = ContradictionEngine() if ContradictionEngine else None
        self._graph_prior = GraphPrior() if GraphPrior else None
        self._world_version: int = 0
        self._prev_correct_affected: Set[str] = set()
        self._prev_intermediate_fbeta: float = 0.0

        # Per-episode mutable state (initialised by reset())
        self._graph:            nx.DiGraph  = nx.DiGraph()
        self._all_services:     List[str]   = []
        self._changed_service:  str         = ""
        self._correct_affected: Set[str]    = set()
        self._queries_used:     int         = 0
        self._max_queries:      int         = 15
        self._task_difficulty:  str         = "easy"
        self._queried_services: Set[str]    = set()
        self._episode_ended:    bool        = False
        self._current_seed:     int         = 0
        self._incident_context              = None
        self._free_action_uses: dict        = {"query_runbook": 0, "query_changelog": 0, "query_monitoring": 0}
        self._state = ServiceImpactState()

    # ── OpenEnv required interface ────────────────────────────────────────

    def reset(
        self,
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs:   Any,
    ) -> ServiceImpactObservation:
        """Start a new episode.

        Each seed produces a unique graph topology + changed_service selection.
        seed % 3 determines difficulty (0=easy, 1=medium, 2=hard).
        """
        self._reset_rubric()

        if seed is None:
            seed = random.randint(0, 100_000)
        self._current_seed = seed

        # Build episode-specific perturbed graph
        self._graph       = build_service_graph(seed=seed)
        self._all_services = get_all_services(self._graph)

        # Select scenario dynamically from seed + graph
        scenario               = get_scenario(self._graph, seed)
        self._task_difficulty  = scenario["difficulty"]
        self._changed_service  = scenario["changed_service"]
        self._max_queries      = scenario["max_queries"]
        # Store initial affected set (may change via mutations; recomputed at submit)
        self._correct_affected = get_affected_services(self._graph, self._changed_service)

        # MutationEngine (v2: mid-episode topology changes)
        if MutationEngine is not None:
            self._mutation_engine = MutationEngine(
                seed=seed,
                difficulty=self._task_difficulty,
            )
        else:
            self._mutation_engine = None

        # Curriculum config (v2: difficulty-adaptive parameters)
        if self._curriculum is not None:
            self._curriculum_config = self._curriculum.get_config(self._task_difficulty)
            # Override free-action caps from curriculum
            self.FREE_ACTION_CAPS = {
                "query_runbook":    self._curriculum_config.runbook_cap,
                "query_changelog":  self._curriculum_config.changelog_cap,
                "query_monitoring": self._curriculum_config.monitoring_cap,
            }
        else:
            self._curriculum_config = None
            self.FREE_ACTION_CAPS = {"query_runbook": 2, "query_changelog": 2, "query_monitoring": 3}

        # Reset tracking
        self._queries_used    = 0
        self._queried_services = set()
        self._episode_ended   = False
        self._incident_context = None
        self._free_action_uses = {"query_runbook": 0, "query_changelog": 0, "query_monitoring": 0}

        # ── World Modeling reset (v3) ────────────────────────────────────
        # Update graph prior with PREVIOUS episode's ground truth (if any)
        if self._graph_prior and self._prev_correct_affected:
            self._graph_prior.update(
                seed_service=self._changed_service,
                true_affected=frozenset(self._prev_correct_affected),
            )
        self._world_version = 0
        self._prev_correct_affected = set()
        self._prev_intermediate_fbeta = 0.0
        if self._belief_tracker is not None:
            self._belief_tracker = BeliefTracker(all_services=self._all_services)  # type: ignore[arg-type]
            self._belief_tracker.reset(self._changed_service)
        if self._contradiction_engine is not None:
            self._contradiction_engine.reset()

        # Log reset event
        if self._trajectory_logger:
            self._trajectory_logger.log_reset(
                seed=seed,
                changed_service=self._changed_service,
                difficulty=self._task_difficulty,
                max_queries=self._max_queries,
            )

        # Select reward profile for this episode (v2: rotating profiles)
        if self._reward_orchestrator:
            self._reward_profile = self._reward_orchestrator.get_profile(seed)
        else:
            self._reward_profile = None
        profile_name = self._reward_profile.name if self._reward_profile else "recall_heavy"

        self._state = ServiceImpactState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            changed_service=self._changed_service,
            queries_used=0,
            max_queries=self._max_queries,
            correct_affected=[],   # hidden until episode ends
            predicted_affected=[],
            task_difficulty=self._task_difficulty,
            episode_ended=False,
            reward_profile=profile_name,
            graph_prior=self._graph_prior.get_prior() if self._graph_prior else None,
            contradictions=[],
            world_version=0,
        )

        # Generate LLM incident context (cached per seed — fast on cache hit)
        meta = SERVICE_METADATA.get(self._changed_service, {})
        incident_block = ""
        if self._simulator.is_active:
            try:
                self._incident_context = self._simulator.generate_incident_context(
                    seed=seed,
                    changed_service=self._changed_service,
                    team=meta.get("team", "platform"),
                    language=meta.get("language", "python"),
                    tier=meta.get("tier", 2),
                    difficulty=self._task_difficulty,
                )
                incident_block = (
                    f"\n\n{'═'*52}\n"
                    f"INCIDENT ALERT\n{self._incident_context.alert_text}\n\n"
                    f"CHANGE LOG\n{self._incident_context.changelog_text}\n"
                    f"{'═'*52}\n"
                )
            except Exception:
                pass   # graceful degradation — episode continues without LLM context

        n_total = len(self._all_services)
        profile_desc = ""
        if self._reward_profile:
            profile_desc = (
                f"\nReward profile: {self._reward_profile.name} — "
                f"{self._reward_profile.description}"
            )

        # Graph prior hint (v3: session-level warm-start)
        prior_hint = ""
        if self._graph_prior and self._graph_prior.episode_count > 0:
            prior_hint = "\n\n" + self._graph_prior.to_observation_text(k=6) + "\n"

        # Curriculum hints (v2: difficulty-adaptive guidance)
        curriculum_hint = ""
        if self._curriculum is not None and self._curriculum_config is not None:
            n_affected = len(self._correct_affected)
            curriculum_hint = "\n" + self._curriculum.get_hint_text(
                self._curriculum_config, self._changed_service, n_affected
            )

        return ServiceImpactObservation(
            changed_service=self._changed_service,
            result=[],
            queries_remaining=self._max_queries,
            message=(
                f"[{self._task_difficulty.upper()} TASK] "
                f"Service '{self._changed_service}' has a breaking change. "
                f"Identify ALL downstream services that will be affected."
                f"{incident_block}"
                f"System has {n_total} total services. "
                f"Query budget: {self._max_queries}.\n"
                f"Hint: {scenario['hint']}"
                f"{curriculum_hint}\n"
                f"{prior_hint}"
                f"Budget actions: query_dependents, query_dependencies. "
                f"Free actions: query_runbook, query_changelog, query_monitoring. "
                f"Hypothesis check: submit_hypothesis (costs 1 query). "
                f"Finish with: submit."
                f"{profile_desc}"
            ),
            done=False,
            reward=None,
            graph_prior=self._graph_prior.get_prior() if self._graph_prior else None,
        )

    def step(
        self,
        action:    ServiceImpactAction,
        timeout_s: Optional[float] = None,
        **kwargs:  Any,
    ) -> ServiceImpactObservation:
        """Execute one action and return the resulting observation.

        Reward semantics:
          - Registry query (new service) : +0.05  — intermediate exploration signal
          - Registry query (re-query)    : -0.05  — intermediate penalty signal
          - Free action (runbook…)       : None   — no reward signal
          - Budget exhaustion            : -0.4   — terminal, episode ends
          - submit                       : F-beta(β=2) in [0.0, 1.0]  ← **evaluation metric**

        Note: step rewards (+0.05 / -0.05) are pedagogical signals only and are NOT
        summed into the terminal score. The reward returned on the 'submit' action
        is the definitive F-beta(β=2) score used for evaluation and leaderboard ranking.
        """
        self._state.step_count += 1

        # Guard: reset() not called yet
        if not self._changed_service:
            return ServiceImpactObservation(
                changed_service="",
                result=[],
                queries_remaining=0,
                message="No active episode. Call reset() to start a new episode.",
                done=False,
                reward=None,
            )

        # Guard: episode already over
        if self._episode_ended:
            return ServiceImpactObservation(
                changed_service=self._changed_service,
                result=[],
                queries_remaining=0,
                message="Episode has already ended. Call reset() to start a new episode.",
                done=True,
                reward=None,
            )

        # ── MUTATION CHECK (v2: mid-episode topology changes) ─────────────
        self._pending_mutation_alert = ""
        if self._mutation_engine:
            _, event = self._mutation_engine.maybe_mutate(
                self._graph, self._state.step_count
            )
            if event:
                # Recompute ground truth from mutated graph
                self._correct_affected = get_affected_services(
                    self._graph, self._changed_service
                )
                self._pending_mutation_alert = f"\n\n{event.description}\n"

        # ── SUBMIT ────────────────────────────────────────────────────────
        if action.action_type == "submit":
            return self._handle_submit(action)

        # ── HYPOTHESIS CHECK (non-terminal partial score) ─────────────────
        if action.action_type == "submit_hypothesis":
            return self._handle_hypothesis(action)

        # ── FREE ACTIONS (no budget deduction) ────────────────────────────
        if action.action_type in ("query_runbook", "query_changelog", "query_monitoring"):
            return self._handle_free_action(action)

        # ── NEW FREE TOOLS (v2: topology diff + service health) ───────────
        if action.action_type == "query_topology_diff":
            return self._handle_topology_diff(action)
        if action.action_type == "query_service_health":
            return self._handle_service_health(action)

        # ── BUDGET GUARD ──────────────────────────────────────────────────
        if self._queries_used >= self._max_queries:
            self._episode_ended    = True
            self._state.episode_ended = True
            return ServiceImpactObservation(
                changed_service=self._changed_service,
                result=[],
                queries_remaining=0,
                message=(
                    "Query budget exhausted without a submission. "
                    "Episode ended — reward = -0.4. "
                    "Submit before your budget runs out next time."
                ),
                done=True,
                reward=-0.4,
            )

        # ── VALIDATE SERVICE NAME ─────────────────────────────────────────
        svc = action.service_name
        if not svc or svc not in self._graph:
            sample = self._all_services[:6]
            return ServiceImpactObservation(
                changed_service=self._changed_service,
                result=[],
                queries_remaining=self._max_queries - self._queries_used,
                message=(
                    f"Unknown service '{svc}'. "
                    f"Valid services include: {sample}. "
                    f"No query consumed."
                ),
                done=False,
                reward=None,
            )

        # ── DETERMINE STEP REWARD ─────────────────────────────────────────
        is_requery   = svc in self._queried_services
        step_reward  = -0.05 if is_requery else 0.05

        # Execute ground-truth query (internal only — never returned to agent)
        if action.action_type == "query_dependents":
            true_result   = get_direct_dependents(self._graph, svc)
            direction_lbl = f"direct callers of '{svc}'"
        else:  # query_dependencies
            true_result   = get_direct_dependencies(self._graph, svc)
            direction_lbl = f"direct dependencies of '{svc}'"

        # Deduct budget
        self._queries_used            += 1
        self._state.queries_used       = self._queries_used
        queries_remaining              = self._max_queries - self._queries_used
        self._queried_services.add(svc)

        # Generate LLM-simulated registry response (Option A: agent sees only this)
        meta     = SERVICE_METADATA.get(svc, {})
        llm_text = self._get_registry_response(action.action_type, svc, true_result, meta)

        # Auto-terminate on budget exhaustion
        done = queries_remaining == 0
        if done:
            self._episode_ended       = True
            self._state.episode_ended = True

        penalty_note = " [⚠️ Re-query penalty: -0.05]" if is_requery else ""
        budget_note  = (
            "\n[Budget exhausted — episode ends. Submit earlier next time.]"
            if done else ""
        )

        obs_message = (
            f"[{action.action_type.upper()}] {direction_lbl}\n"
            f"{llm_text}"
            f"{penalty_note}"
            f"\nQueries remaining: {queries_remaining}{budget_note}"
            f"{self._pending_mutation_alert}"
        )

        # ── World Modeling updates (v3) ───────────────────────────────────
        new_belief: dict | None = None
        ig: float | None = None
        ifbeta: float | None = None
        belief_drift: float | None = None
        contradiction_count: int = 0
        step_process_reward: float | None = None

        true_affected_frozen = frozenset(self._correct_affected)

        # World version bump on mutation
        if self._pending_mutation_alert:
            self._world_version += 1
            prev_set = self._prev_correct_affected
            if prev_set:
                drift_union = prev_set | self._correct_affected
                belief_drift = (
                    len(prev_set.symmetric_difference(self._correct_affected)) / len(drift_union)
                    if drift_union else 0.0
                )
            self._prev_correct_affected = set(self._correct_affected)

        if self._belief_tracker is not None:
            new_belief = self._belief_tracker.update(
                action_type=action.action_type,
                queried_service=svc,
                llm_text=llm_text,
                true_neighbors=true_affected_frozen,
            )
            ig = self._belief_tracker.compute_information_gain(true_affected_frozen)
            ifbeta = self._belief_tracker.intermediate_fbeta(true_affected_frozen)

        if self._contradiction_engine is not None:
            contradiction_event = self._contradiction_engine.check(
                action_type=action.action_type,
                queried_service=svc,
                llm_text=llm_text,
            )
            contradiction_count = self._contradiction_engine.count
            if contradiction_event:
                obs_message += f"\n\n[CONTRADICTION DETECTED] {contradiction_event.to_text()}"
            # Sync to state
            self._state.contradictions = self._contradiction_engine.descriptions()

        # Process reward (step-level dense signal)
        if compute_step_reward is not None and ig is not None and ifbeta is not None:
            step_process_reward = compute_step_reward(
                information_gain=ig,
                prev_fbeta=self._prev_intermediate_fbeta,
                current_fbeta=ifbeta,
                new_contradictions=1 if (self._contradiction_engine and contradiction_count > (self._contradiction_engine.count - 1)) else 0,
            )
            self._prev_intermediate_fbeta = ifbeta

        # Update world_version in state
        self._state.world_version = self._world_version

        # Log step to trajectory
        if self._trajectory_logger:
            self._trajectory_logger.log_step(
                seed=self._current_seed,
                step_num=self._state.step_count,
                action_type=action.action_type,
                service_name=svc,
                reward=-0.4 if done else step_reward,
                queries_remaining=queries_remaining,
                message=obs_message,
            )

        return ServiceImpactObservation(
            changed_service=self._changed_service,
            result=[],          # Option A: never expose graph structure during queries
            queries_remaining=queries_remaining,
            message=obs_message,
            done=done,
            reward=-0.4 if done else step_reward,
            belief_state=new_belief,
            information_gain=ig,
            intermediate_fbeta=ifbeta,
            world_version=self._world_version,
            belief_drift=belief_drift,
            contradiction_count=contradiction_count,
        )

    @property
    def state(self) -> ServiceImpactState:
        return self._state

    # ── Private helpers ───────────────────────────────────────────────────

    def _handle_free_action(
        self, action: ServiceImpactAction
    ) -> ServiceImpactObservation:
        """Handle free actions: query_runbook, query_changelog, query_monitoring.

        Free-action caps (v2): Each free action type has a per-episode usage limit.
        Exceeding the cap deducts 0.5 from the query budget instead of being free.
        Caps: runbook=2, changelog=2, monitoring=3.
        """
        svc  = action.service_name or self._changed_service
        meta = SERVICE_METADATA.get(svc, {})

        # ── Free-action cap enforcement ───────────────────────────────
        action_key = action.action_type
        cap = self.FREE_ACTION_CAPS.get(action_key, 99)
        self._free_action_uses[action_key] = self._free_action_uses.get(action_key, 0) + 1
        uses = self._free_action_uses[action_key]
        is_over_cap = uses > cap

        budget_penalty = 0
        cap_note = ""
        if is_over_cap:
            # Overage costs 0.5 from query budget (fractional — rounded up on next budget query)
            budget_penalty = 1  # deduct 1 full query as penalty
            self._queries_used += budget_penalty
            self._state.queries_used = self._queries_used
            cap_note = (
                f"\n⚠️  {action_key} cap exceeded ({uses}/{cap} uses). "
                f"Budget penalty: -1 query. Queries remaining: {self._max_queries - self._queries_used}."
            )
            # Check if budget is now exhausted
            if self._queries_used >= self._max_queries:
                self._episode_ended = True
                self._state.episode_ended = True
                return ServiceImpactObservation(
                    changed_service=self._changed_service,
                    result=[],
                    queries_remaining=0,
                    message=(
                        f"Free-action cap exceeded and query budget exhausted. "
                        f"Episode ended — reward = -0.4."
                    ),
                    done=True,
                    reward=-0.4,
                )

        # Update state counters
        if action_key == "query_runbook":
            self._state.runbook_uses = uses
        elif action_key == "query_changelog":
            self._state.changelog_uses = uses
        elif action_key == "query_monitoring":
            self._state.monitoring_uses = uses

        try:
            if action.action_type == "query_runbook":
                dependents   = get_direct_dependents(self._graph, svc)
                dependencies = get_direct_dependencies(self._graph, svc)
                content = self._simulator.generate_runbook(
                    seed=self._current_seed, service_name=svc,
                    dependents=dependents, dependencies=dependencies,
                    team=meta.get("team", "platform"), tier=meta.get("tier", 2),
                    difficulty=self._task_difficulty,
                )
                label = "RUNBOOK"

            elif action.action_type == "query_changelog":
                # Changelog is always about the episode's changed_service
                svc_for_change = self._changed_service
                meta_c = SERVICE_METADATA.get(svc_for_change, {})
                content = self._simulator.generate_changelog(
                    seed=self._current_seed,
                    changed_service=svc_for_change,
                    team=meta_c.get("team", "platform"),
                    difficulty=self._task_difficulty,
                )
                svc   = svc_for_change
                label = "CHANGELOG"

            else:  # query_monitoring
                dependents   = get_direct_dependents(self._graph, svc)
                dependencies = get_direct_dependencies(self._graph, svc)
                content = self._simulator.generate_monitoring(
                    seed=self._current_seed, service_name=svc,
                    dependents=dependents, dependencies=dependencies,
                    difficulty=self._task_difficulty,
                )
                label = "MONITORING"

        except Exception as exc:
            content = f"[{action.action_type} unavailable: {exc}]"
            label   = action.action_type.upper()

        remaining = self._max_queries - self._queries_used
        budget_msg = "(free action — budget unchanged)" if not is_over_cap else "(cap exceeded — budget deducted)"
        return ServiceImpactObservation(
            changed_service=self._changed_service,
            result=[],
            queries_remaining=remaining,
            message=(
                f"[{label}] {svc}\n"
                f"{content}\n"
                f"Queries remaining: {remaining} {budget_msg}"
                f"{cap_note}"
                f"{self._pending_mutation_alert}"
            ),
            done=False,
            reward=None,
        )

    def _handle_topology_diff(
        self, action: ServiceImpactAction
    ) -> ServiceImpactObservation:
        """Show topology changes since episode start (FREE action).

        Reports mutations that have occurred during this episode.
        Useful for agents to understand how the graph has changed.
        """
        mutations = []
        if self._mutation_engine:
            mutations = self._mutation_engine.mutations_applied

        if not mutations:
            content = (
                "[TOPOLOGY DIFF] No topology changes detected since episode start.\n"
                "The service dependency graph remains in its original state."
            )
        else:
            lines = [f"[TOPOLOGY DIFF] {len(mutations)} mutation(s) detected this episode:"]
            for i, m in enumerate(mutations, 1):
                edge_desc = ", ".join(f"{u} → {v}" for u, v in m.edges) if m.edges else "no edges changed"
                lines.append(f"  {i}. Step {m.step_num}: {m.mutation_type} — {edge_desc}")
            lines.append("\nThe blast radius may have changed. Consider re-investigating.")
            content = "\n".join(lines)

        remaining = self._max_queries - self._queries_used
        return ServiceImpactObservation(
            changed_service=self._changed_service,
            result=[],
            queries_remaining=remaining,
            message=f"{content}\nQueries remaining: {remaining} (free action — budget unchanged)",
            done=False,
            reward=None,
        )

    def _handle_service_health(
        self, action: ServiceImpactAction
    ) -> ServiceImpactObservation:
        """Provide a health summary for a service (FREE action).

        Returns aggregated info: tier, team, in/out degree, whether it's
        in the current investigation scope.
        """
        svc = action.service_name or self._changed_service
        meta = SERVICE_METADATA.get(svc, {})

        if svc not in self._graph:
            remaining = self._max_queries - self._queries_used
            return ServiceImpactObservation(
                changed_service=self._changed_service,
                result=[],
                queries_remaining=remaining,
                message=f"[SERVICE HEALTH] Unknown service '{svc}'.\nQueries remaining: {remaining}",
                done=False,
                reward=None,
            )

        in_degree = self._graph.in_degree(svc)
        out_degree = self._graph.out_degree(svc)
        tier = meta.get("tier", "unknown")
        team = meta.get("team", "unknown")
        language = meta.get("language", "unknown")

        # Indicate if this service was already queried
        queried_status = "already queried" if svc in self._queried_services else "not yet queried"

        content = (
            f"[SERVICE HEALTH] {svc}\n"
            f"  Team: {team} | Tier: {tier} | Language: {language}\n"
            f"  Incoming dependencies (callers): {in_degree}\n"
            f"  Outgoing dependencies (calls): {out_degree}\n"
            f"  Investigation status: {queried_status}\n"
            f"  Changed service: {'YES — this is the incident source' if svc == self._changed_service else 'no'}"
        )

        remaining = self._max_queries - self._queries_used
        return ServiceImpactObservation(
            changed_service=self._changed_service,
            result=[],
            queries_remaining=remaining,
            message=f"{content}\nQueries remaining: {remaining} (free action — budget unchanged)",
            done=False,
            reward=None,
        )

    def _handle_hypothesis(
        self, action: ServiceImpactAction
    ) -> ServiceImpactObservation:
        """Score a hypothesis without ending the episode.

        Returns a delayed_reward (partial F-beta) that the agent can use to
        calibrate its investigation strategy. Costs 1 query from budget.
        Max hypotheses per episode is set by CurriculumScheduler (default 3).
        """
        # Curriculum-adaptive hypothesis cap
        if self._curriculum_config is not None:
            MAX_HYPOTHESES = self._curriculum_config.max_hypotheses
        else:
            MAX_HYPOTHESES = 3
        self._state.hypothesis_count = getattr(self._state, 'hypothesis_count', 0) + 1
        hypothesis_num = self._state.hypothesis_count

        # Budget cost: each hypothesis check costs 1 query
        self._queries_used += 1
        self._state.queries_used = self._queries_used
        queries_remaining = self._max_queries - self._queries_used

        predicted = set(action.affected_services or [])
        correct   = self._correct_affected

        # Compute partial F-beta (same formula as final submit)
        tp = len(predicted & correct)
        fp = len(predicted - correct)
        fn = len(correct   - predicted)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta      = 2.0
        fbeta     = (
            (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
            if (beta ** 2 * precision + recall) > 0
            else 0.0
        )
        partial_score = round(max(0.001, min(0.999, fbeta)), 4)

        self._state.last_hypothesis_score = partial_score

        # Cap enforcement
        cap_note = ""
        if hypothesis_num > MAX_HYPOTHESES:
            cap_note = (
                f"\n⚠️  Hypothesis cap exceeded ({hypothesis_num}/{MAX_HYPOTHESES}). "
                f"Extra budget cost applied."
            )
            # Extra penalty: double budget cost for over-cap hypotheses
            self._queries_used += 1
            self._state.queries_used = self._queries_used
            queries_remaining = self._max_queries - self._queries_used

        # Log hypothesis via trajectory logger if available
        if hasattr(self, '_trajectory_logger') and self._trajectory_logger:
            self._trajectory_logger.log_hypothesis(
                seed=self._current_seed,
                step_num=self._state.step_count,
                predicted=sorted(predicted),
                confidence=action.confidence,
                partial_score=partial_score,
            )

        # Check if budget exhausted
        done = queries_remaining <= 0
        if done:
            self._episode_ended = True
            self._state.episode_ended = True

        confidence_str = f" (confidence={action.confidence:.2f})" if action.confidence is not None else ""

        return ServiceImpactObservation(
            changed_service=self._changed_service,
            result=[],  # Never reveal ground truth on hypothesis
            queries_remaining=max(0, queries_remaining),
            message=(
                f"[HYPOTHESIS #{hypothesis_num}]{confidence_str}\n"
                f"Partial F-beta(β=2) = {partial_score:.3f} | "
                f"Precision={precision:.3f} | Recall={recall:.3f} | "
                f"TP={tp} FP={fp} FN={fn} | "
                f"Predicted {len(predicted)} services.\n"
                f"Queries remaining: {max(0, queries_remaining)} (hypothesis cost: 1 query)"
                f"{cap_note}"
                f"{self._pending_mutation_alert}"
            ),
            done=done,
            reward=-0.4 if done else None,
            delayed_reward=partial_score,
        )

    def _handle_submit(
        self, action: ServiceImpactAction
    ) -> ServiceImpactObservation:
        """Score the agent's prediction using the active reward profile and end the episode."""
        self._episode_ended       = True
        self._state.episode_ended = True

        predicted = set(action.affected_services or [])
        correct   = self._correct_affected

        # Use RewardOrchestrator if available, else fall back to hardcoded β=2
        if self._reward_orchestrator and self._reward_profile:
            score_data = self._reward_orchestrator.compute(
                predicted=predicted,
                correct=correct,
                all_services_count=len(self._all_services),
                queries_used=self._queries_used,
                max_queries=self._max_queries,
                profile=self._reward_profile,
            )
            reward    = score_data["reward"]
            precision = score_data["precision"]
            recall    = score_data["recall"]
            tp, fp, fn = score_data["tp"], score_data["fp"], score_data["fn"]
        else:
            # Fallback: original hardcoded β=2 scoring
            tp = len(predicted & correct)
            fp = len(predicted - correct)
            fn = len(correct   - predicted)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            beta      = 2.0
            fbeta     = (
                (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
                if (beta ** 2 * precision + recall) > 0
                else 0.0
            )
            n_total    = len(self._all_services)
            n_pred     = len(predicted)
            if n_total > 0 and n_pred / n_total > 0.6:
                oversubmit_frac = min(1.0, (n_pred / n_total - 0.6) / 0.4)
                fbeta = max(0.0, fbeta - 0.3 * oversubmit_frac)
            reward = round(max(0.001, min(0.999, fbeta)), 4)

        self._state.predicted_affected = sorted(predicted)
        self._state.correct_affected   = sorted(correct)   # reveal ground truth

        missed          = sorted(correct   - predicted)
        false_positives = sorted(predicted - correct)

        # Log submit and episode summary to trajectory
        if self._trajectory_logger:
            self._trajectory_logger.log_submit(
                seed=self._current_seed,
                step_num=self._state.step_count,
                predicted=sorted(predicted),
                correct=sorted(correct),
                reward=reward,
                precision=precision,
                recall=recall,
            )
            self._trajectory_logger.log_episode(
                seed=self._current_seed,
                summary={
                    "difficulty": self._task_difficulty,
                    "reward": reward,
                    "queries_used": self._queries_used,
                    "max_queries": self._max_queries,
                    "n_correct": len(correct),
                    "n_predicted": len(predicted),
                    "tp": tp, "fp": fp, "fn": fn,
                    "hypothesis_count": getattr(self._state, 'hypothesis_count', 0),
                },
            )

        return ServiceImpactObservation(
            changed_service=self._changed_service,
            result=sorted(correct),           # ground truth revealed on submit
            queries_remaining=self._max_queries - self._queries_used,
            message=(
                f"Episode complete! "
                f"F-beta(β=2)={reward:.3f} | Precision={precision:.3f} | "
                f"Recall={recall:.3f} | TP={tp} FP={fp} FN={fn} | "
                f"Queries used: {self._queries_used}/{self._max_queries} | "
                f"Ground truth ({len(correct)} services): {sorted(correct)} | "
                f"Missed: {missed} | False positives: {false_positives}"
            ),
            done=True,
            reward=reward,
        )

    def _get_registry_response(
        self, action_type: str, svc: str, true_result: List[str], meta: dict
    ) -> str:
        """Get LLM-simulated registry response with graceful fallback."""
        direction = "dependents" if action_type == "query_dependents" else "dependencies"
        try:
            return self._simulator.simulate_registry_query(
                seed=self._current_seed,
                action_type=action_type,
                service_name=svc,
                true_result=true_result,
                all_services=self._all_services,
                team=meta.get("team", "platform"),
                tier=meta.get("tier", 2),
                difficulty=self._task_difficulty,
            )
        except Exception:
            return self._simulator.fallback_registry(
                svc, true_result, direction, self._task_difficulty
            )
