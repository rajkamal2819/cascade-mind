"""
server/trajectory_logger.py
----------------------------
JSONL-per-episode trajectory logger for cascade-mind v2.

Logs every step and episode summary to disk so the TrajectoryAuditor (P6)
can compute per-episode metrics like:
  - steps-to-first-correct-hypothesis
  - budget efficiency  (queries_used / max_queries)
  - mutation recovery time

File layout:
  {trajectory_dir}/episode_{seed}.jsonl   — one JSON line per event

Event types:
  reset     – episode start (seed, changed_service, difficulty, max_queries)
  step      – per-step log (action_type, service_name, reward, queries_remaining)
  hypothesis – submit_hypothesis event (predicted, confidence, partial_score)
  submit    – final submit (predicted, correct, reward, precision, recall)
  episode   – summary record written at episode end
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """One line in the JSONL trajectory file."""
    event: str                          # "reset" | "step" | "hypothesis" | "submit" | "episode"
    timestamp: float = 0.0
    seed: int = 0
    step_num: int = 0
    action_type: str = ""
    service_name: str = ""
    reward: Optional[float] = None
    queries_remaining: int = 0
    message_hash: str = ""              # MD5 of message text (keep files small)
    extra: Dict[str, Any] = field(default_factory=dict)


class TrajectoryLogger:
    """Writes JSONL trajectory files for each episode.

    Usage:
        logger = TrajectoryLogger("/app/trajectories")
        logger.log_reset(seed=42, changed="catalog_service", difficulty="medium", max_q=12)
        logger.log_step(seed=42, step=1, action_type="query_dependents",
                        svc="catalog_service", reward=0.05, remaining=11, message="...")
        logger.log_episode(seed=42, summary={...})
    """

    def __init__(self, trajectory_dir: str = "/app/trajectories") -> None:
        self._dir = Path(trajectory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._start_times: Dict[int, float] = {}  # seed → episode start time

    # ── Public API ────────────────────────────────────────────────────────

    def log_reset(
        self,
        seed: int,
        changed_service: str,
        difficulty: str,
        max_queries: int,
    ) -> None:
        """Log episode start."""
        self._start_times[seed] = time.time()
        self._write(seed, StepRecord(
            event="reset",
            timestamp=time.time(),
            seed=seed,
            extra={
                "changed_service": changed_service,
                "difficulty": difficulty,
                "max_queries": max_queries,
            },
        ))

    def log_step(
        self,
        seed: int,
        step_num: int,
        action_type: str,
        service_name: str,
        reward: Optional[float],
        queries_remaining: int,
        message: str,
    ) -> None:
        """Log a single step (query or free action)."""
        self._write(seed, StepRecord(
            event="step",
            timestamp=time.time(),
            seed=seed,
            step_num=step_num,
            action_type=action_type,
            service_name=service_name,
            reward=reward,
            queries_remaining=queries_remaining,
            message_hash=hashlib.md5(message.encode()).hexdigest(),
        ))

    def log_hypothesis(
        self,
        seed: int,
        step_num: int,
        predicted: List[str],
        confidence: Optional[float],
        partial_score: float,
    ) -> None:
        """Log a submit_hypothesis event."""
        self._write(seed, StepRecord(
            event="hypothesis",
            timestamp=time.time(),
            seed=seed,
            step_num=step_num,
            reward=partial_score,
            extra={
                "predicted": sorted(predicted),
                "confidence": confidence,
                "n_predicted": len(predicted),
            },
        ))

    def log_submit(
        self,
        seed: int,
        step_num: int,
        predicted: List[str],
        correct: List[str],
        reward: float,
        precision: float,
        recall: float,
    ) -> None:
        """Log the final submit."""
        self._write(seed, StepRecord(
            event="submit",
            timestamp=time.time(),
            seed=seed,
            step_num=step_num,
            reward=reward,
            extra={
                "predicted": sorted(predicted),
                "correct": sorted(correct),
                "n_predicted": len(predicted),
                "n_correct": len(correct),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            },
        ))

    def log_episode(
        self,
        seed: int,
        summary: Dict[str, Any],
    ) -> None:
        """Log episode summary (final record in the JSONL file)."""
        elapsed = time.time() - self._start_times.get(seed, time.time())
        self._write(seed, StepRecord(
            event="episode",
            timestamp=time.time(),
            seed=seed,
            extra={**summary, "elapsed_s": round(elapsed, 2)},
        ))

    # ── Internal ──────────────────────────────────────────────────────────

    def _write(self, seed: int, record: StepRecord) -> None:
        """Append one JSON line to the episode file."""
        path = self._dir / f"episode_{seed}.jsonl"
        try:
            with open(path, "a") as f:
                f.write(json.dumps(asdict(record), default=str) + "\n")
        except OSError as exc:
            logger.warning("Trajectory write failed (seed=%d): %s", seed, exc)

    def read_episode(self, seed: int) -> List[Dict[str, Any]]:
        """Read all records for an episode (for auditing)."""
        path = self._dir / f"episode_{seed}.jsonl"
        if not path.exists():
            return []
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
