"""
server/trajectory_auditor.py
------------------------------
TrajectoryAuditor — post-episode analysis for cascade-mind v2.

Reads trajectory logs and produces structured audit reports:
  - Action efficiency (budget utilization)
  - Hypothesis accuracy progression
  - Mutation response quality
  - Strategy classification (BFS-first, hypothesis-driven, etc.)

Used by the benchmark script and can be called from MCP tools.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditReport:
    """Structured audit of a single episode."""

    seed: int
    difficulty: str = "unknown"
    total_steps: int = 0
    queries_used: int = 0
    budget_total: int = 0
    budget_utilization: float = 0.0  # queries_used / budget_total

    # Action breakdown
    action_counts: Dict[str, int] = field(default_factory=dict)
    unique_services_queried: int = 0
    requery_count: int = 0

    # Hypothesis tracking
    hypothesis_scores: List[float] = field(default_factory=list)
    hypothesis_trend: str = "none"  # "improving", "declining", "stable", "none"

    # Mutation response
    mutations_detected: int = 0
    post_mutation_actions: int = 0  # actions taken after last mutation

    # Final result
    final_reward: float = 0.0
    final_fbeta: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    reward_profile: str = "unknown"

    # Strategy classification
    strategy: str = "unknown"  # "bfs_first", "hypothesis_driven", "free_intel_heavy", "mixed"


class TrajectoryAuditor:
    """Analyzes trajectory logs to produce audit reports.

    Usage:
        auditor = TrajectoryAuditor(trajectory_dir="/tmp/cascade_trajectories")
        report = auditor.audit_episode(seed=42)
        print(report.strategy, report.budget_utilization)
    """

    def __init__(self, trajectory_dir: str = "/tmp/cascade_trajectories") -> None:
        self._dir = Path(trajectory_dir)

    def audit_episode(self, seed: int) -> Optional[AuditReport]:
        """Read and analyze a trajectory log for the given seed.

        Returns None if the log file doesn't exist.
        """
        log_path = self._dir / f"episode_{seed}.jsonl"
        if not log_path.exists():
            return None

        records = []
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            return None

        return self._analyze(seed, records)

    def _analyze(self, seed: int, records: List[Dict[str, Any]]) -> AuditReport:
        """Produce an AuditReport from trajectory records."""
        report = AuditReport(seed=seed)

        queried_services: set = set()
        action_counts: Dict[str, int] = {}
        hypothesis_scores: List[float] = []
        last_mutation_step: int = -1
        max_step: int = 0

        for rec in records:
            event = rec.get("event", "")
            step_num = rec.get("step_num", 0)
            max_step = max(max_step, step_num)

            if event == "reset":
                extra = rec.get("extra", {})
                report.difficulty = extra.get("difficulty", "unknown")
                report.budget_total = extra.get("max_queries", 0)

            elif event == "step":
                action_type = rec.get("action_type", "")
                svc = rec.get("service_name", "")
                action_counts[action_type] = action_counts.get(action_type, 0) + 1

                if action_type in ("query_dependents", "query_dependencies"):
                    if svc in queried_services:
                        report.requery_count += 1
                    queried_services.add(svc)

                # Check for mutation in message
                msg = rec.get("extra", {}).get("message_hash", "")
                # We can detect mutations from the action log context
                queries_remaining = rec.get("queries_remaining", 0)

            elif event == "hypothesis":
                extra = rec.get("extra", {})
                score = extra.get("partial_score", 0.0)
                hypothesis_scores.append(score)

            elif event == "submit":
                extra = rec.get("extra", {})
                report.final_reward = rec.get("reward", 0.0)
                report.final_fbeta = extra.get("fbeta", 0.0)
                report.tp = extra.get("tp", 0)
                report.fp = extra.get("fp", 0)
                report.fn = extra.get("fn", 0)
                report.reward_profile = extra.get("profile", "unknown")

            elif event == "episode":
                extra = rec.get("extra", {})
                report.total_steps = extra.get("total_steps", max_step)

        report.action_counts = action_counts
        report.unique_services_queried = len(queried_services)
        report.queries_used = sum(
            v for k, v in action_counts.items()
            if k in ("query_dependents", "query_dependencies", "submit_hypothesis")
        )
        report.budget_utilization = (
            report.queries_used / report.budget_total
            if report.budget_total > 0 else 0.0
        )
        report.hypothesis_scores = hypothesis_scores

        # Classify hypothesis trend
        if len(hypothesis_scores) >= 2:
            if hypothesis_scores[-1] > hypothesis_scores[0] + 0.05:
                report.hypothesis_trend = "improving"
            elif hypothesis_scores[-1] < hypothesis_scores[0] - 0.05:
                report.hypothesis_trend = "declining"
            else:
                report.hypothesis_trend = "stable"
        elif len(hypothesis_scores) == 1:
            report.hypothesis_trend = "single"

        # Classify strategy
        report.strategy = self._classify_strategy(action_counts, hypothesis_scores)

        return report

    def _classify_strategy(
        self,
        action_counts: Dict[str, int],
        hypothesis_scores: List[float],
    ) -> str:
        """Classify the agent's strategy based on action patterns."""
        dependents = action_counts.get("query_dependents", 0)
        dependencies = action_counts.get("query_dependencies", 0)
        free_actions = sum(
            action_counts.get(k, 0)
            for k in ("query_runbook", "query_changelog", "query_monitoring")
        )
        hypotheses = action_counts.get("submit_hypothesis", 0)
        total_budget_actions = dependents + dependencies + hypotheses

        if total_budget_actions == 0:
            return "free_intel_only"
        if hypotheses >= 2:
            return "hypothesis_driven"
        if free_actions >= total_budget_actions:
            return "free_intel_heavy"
        if dependents >= 3 and hypotheses == 0:
            return "bfs_first"
        return "mixed"

    def audit_all(self) -> List[AuditReport]:
        """Audit all episode logs in the trajectory directory."""
        reports = []
        if not self._dir.exists():
            return reports
        for log_file in sorted(self._dir.glob("episode_*.jsonl")):
            try:
                seed = int(log_file.stem.split("_")[1])
                report = self.audit_episode(seed)
                if report:
                    reports.append(report)
            except (ValueError, IndexError):
                continue
        return reports

    # ── GRPO Export ─────────────────────────────────────────────────────────

    def export_grpo_jsonl(
        self,
        output_path: str | Path,
        min_reward: float = 0.0,
        include_process_rewards: bool = True,
    ) -> int:
        """Export trajectories in GRPO training format.

        Each line in the output JSONL has the schema::

            {
                "prompt":    str,          # episode reset message (system context)
                "response":  str,          # concatenated agent tool calls
                "reward":    float,        # terminal F-beta reward
                "metadata":  { ... }       # seed, difficulty, strategy, …
            }

        Args:
            output_path: Destination JSONL file.
            min_reward: Only export episodes with reward >= this threshold.
            include_process_rewards: Attach per-step intermediate rewards
                when ``world_version`` / ``intermediate_fbeta`` fields exist.

        Returns:
            Number of episodes exported.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported = 0

        for log_file in sorted(self._dir.glob("episode_*.jsonl")):
            try:
                seed = int(log_file.stem.split("_")[1])
            except (ValueError, IndexError):
                continue

            steps = self._load_steps(log_file)
            if not steps:
                continue

            # Terminal reward is the reward on the last step
            terminal_reward = 0.0
            for s in reversed(steps):
                if s.get("reward") is not None:
                    terminal_reward = float(s["reward"])
                    break

            if terminal_reward < min_reward:
                continue

            # Build prompt from reset metadata (first step)
            first = steps[0]
            prompt = first.get("message", "") or first.get("observation", "")

            # Build response as newline-separated action texts
            response_parts: list[str] = []
            process_rewards: list[float] = []
            for s in steps:
                action = s.get("action", "")
                if action:
                    response_parts.append(str(action))
                if include_process_rewards:
                    ig   = s.get("information_gain") or 0.0
                    dfb  = s.get("intermediate_fbeta") or 0.0
                    process_rewards.append(round(float(ig) * 0.1 + float(dfb) * 0.05, 4))

            record: Dict[str, Any] = {
                "prompt": prompt,
                "response": "\n".join(response_parts),
                "reward": round(terminal_reward, 4),
                "metadata": {
                    "seed": seed,
                    "difficulty": first.get("difficulty", "unknown"),
                    "strategy": self._classify_strategy(steps),
                    "steps": len(steps),
                },
            }
            if include_process_rewards and process_rewards:
                record["process_rewards"] = process_rewards

            with open(output_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

            exported += 1

        return exported

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load_steps(self, log_file: Path) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        try:
            with open(log_file, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        steps.append(json.loads(line))
        except Exception:
            pass
        return steps

    def _classify_strategy_from_steps(self, steps: List[Dict[str, Any]]) -> str:
        """Classify strategy from raw step records (used by GRPO export)."""
        try:
            return self._get_strategy(steps)  # type: ignore[attr-defined]
        except AttributeError:
            return "unknown"

    def summary(self, reports: Optional[List[AuditReport]] = None) -> Dict[str, Any]:
        """Produce a summary across all audited episodes."""
        if reports is None:
            reports = self.audit_all()
        if not reports:
            return {"episodes": 0}

        rewards = [r.final_reward for r in reports]
        utilizations = [r.budget_utilization for r in reports]
        strategies = {}
        for r in reports:
            strategies[r.strategy] = strategies.get(r.strategy, 0) + 1

        return {
            "episodes": len(reports),
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "min_reward": round(min(rewards), 4),
            "max_reward": round(max(rewards), 4),
            "mean_budget_utilization": round(sum(utilizations) / len(utilizations), 3),
            "strategy_distribution": strategies,
            "hypothesis_usage": sum(1 for r in reports if r.hypothesis_scores),
        }
