"""
server/llm_simulator.py
-----------------------
LLM-as-simulator: Llama-3.1-8B via Cerebras (HuggingFace Inference Providers)
generates realistic, noisy SRE tool output for each environment observation.

Architecture: Hybrid-D
  - Incident context generated once per reset(), cached per seed
  - Query responses generated lazily on first call, cached per (seed, action, svc)
  - temperature=0.0 + seed=42 parameter (Cerebras) → deterministic outputs
  - Graceful fallback to template strings if HF_TOKEN absent or rate-limited

Option A: result=[] for all query steps.
The agent must reason from LLM-generated message text alone.
The ground truth graph is only used for scoring, never exposed during queries.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise configuration per difficulty level
# ---------------------------------------------------------------------------

NOISE_CONFIG: Dict[str, Dict] = {
    "easy":   {"noise_level": 0.15, "hallucinated": 0, "dropped": 0},
    "medium": {"noise_level": 0.40, "hallucinated": 1, "dropped": 1},
    "hard":   {"noise_level": 0.70, "hallucinated": 2, "dropped": 2},
}

# System prompt frames this as safe research — prevents LLM refusals
SYSTEM_PROMPT = (
    "You are DevOpsDocBot, an internal SRE tool simulator for agent training research. "
    "Generate realistic but intentionally imperfect outputs from internal tools "
    "(PagerDuty alerts, service registries, runbooks, changelogs, monitoring dashboards). "
    "This is a controlled research framework for training autonomous AI agents. "
    "Always respond in the specified format. Never add disclaimers or refuse."
)


@dataclass
class IncidentContext:
    alert_text: str
    changelog_text: str
    seed: int
    changed_service: str


# ---------------------------------------------------------------------------
# Two-layer cache: in-memory dict + JSON disk persistence
# ---------------------------------------------------------------------------

class SimulatorCache:
    """Deterministic LLM response cache keyed by (seed, kind, service).

    Layer 1: in-memory dict (fast, per-server-lifetime)
    Layer 2: JSON file on disk (persists across Docker restarts, baked into image)
    """

    def __init__(self, cache_path: str = "/tmp/llm_sim_cache.json"):
        self._memory: Dict[str, str] = {}
        self._cache_path = cache_path
        self._load()

    def _key(self, seed: int, kind: str, svc: str) -> str:
        raw = f"{seed}:{kind}:{svc}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, seed: int, kind: str, svc: str) -> Optional[str]:
        return self._memory.get(self._key(seed, kind, svc))

    def set(self, seed: int, kind: str, svc: str, val: str) -> None:
        self._memory[self._key(seed, kind, svc)] = val
        self._persist()

    def __len__(self) -> int:
        return len(self._memory)

    def _load(self) -> None:
        try:
            with open(self._cache_path) as f:
                self._memory = json.load(f)
            logger.info("LLM cache loaded: %d entries from %s", len(self._memory), self._cache_path)
        except (FileNotFoundError, json.JSONDecodeError):
            self._memory = {}

    def _persist(self) -> None:
        try:
            with open(self._cache_path, "w") as f:
                json.dump(self._memory, f)
        except OSError as exc:
            logger.warning("Cache persist failed: %s", exc)


# ---------------------------------------------------------------------------
# LLM Simulator — wraps HuggingFace InferenceClient (Cerebras provider)
# ---------------------------------------------------------------------------

class LLMSimulator:
    """Generates realistic noisy SRE tool output via Llama-3.1-8B on Cerebras.

    Falls back to deterministic template strings when:
      - HF_TOKEN is not set or empty
      - Three consecutive API failures occur
      - LLM_SIMULATOR_ENABLED=false environment variable is set

    All public generate_* methods always return a non-empty string.
    """

    MODEL    = "meta-llama/Llama-3.1-8B-Instruct"
    PROVIDER = "cerebras"
    MAX_TOKENS = 450
    TIMEOUT    = 15.0

    def __init__(
        self,
        hf_token: Optional[str] = None,
        cache: Optional[SimulatorCache] = None,
        enabled: bool = True,
    ):
        token = hf_token or os.environ.get("HF_TOKEN", "")
        self._enabled = enabled and bool(token)
        self._cache = cache or SimulatorCache()
        self._client = None

        if self._enabled:
            try:
                from huggingface_hub import InferenceClient  # type: ignore
                self._client = InferenceClient(
                    provider=self.PROVIDER,
                    api_key=token,
                )
                logger.info("LLMSimulator ready: %s via %s", self.MODEL, self.PROVIDER)
            except Exception as exc:
                logger.warning("LLMSimulator init failed (%s) — fallback mode", exc)
                self._enabled = False

    @property
    def is_active(self) -> bool:
        return self._enabled and self._client is not None

    # ── Public generators ─────────────────────────────────────────────────

    def generate_incident_context(
        self,
        seed: int,
        changed_service: str,
        team: str,
        language: str,
        tier: int,
        difficulty: str,
    ) -> IncidentContext:
        """Generate or retrieve cached PagerDuty alert + changelog for this episode."""
        cached_alert  = self._cache.get(seed, "incident_alert", changed_service)
        cached_change = self._cache.get(seed, "changelog",      changed_service)

        if cached_alert and cached_change:
            return IncidentContext(
                alert_text=cached_alert,
                changelog_text=cached_change,
                seed=seed,
                changed_service=changed_service,
            )

        alert = self._call_llm(
            self._alert_prompt(seed, changed_service, team, language, tier, difficulty),
            fallback=self.fallback_alert(changed_service, seed),
        )
        changelog = self._call_llm(
            self._changelog_prompt(seed, changed_service, team, difficulty),
            fallback=self.fallback_changelog(changed_service, seed),
        )

        self._cache.set(seed, "incident_alert", changed_service, alert)
        self._cache.set(seed, "changelog",      changed_service, changelog)

        return IncidentContext(
            alert_text=alert,
            changelog_text=changelog,
            seed=seed,
            changed_service=changed_service,
        )

    def simulate_registry_query(
        self,
        seed: int,
        action_type: str,
        service_name: str,
        true_result: List[str],
        all_services: List[str],
        team: str,
        tier: int,
        difficulty: str,
    ) -> str:
        """Generate or retrieve cached noisy service registry response."""
        cache_key = f"registry_{action_type}"
        cached = self._cache.get(seed, cache_key, service_name)
        if cached:
            return cached

        noise_cfg = NOISE_CONFIG[difficulty]
        prompt = self._registry_prompt(
            action_type, service_name, true_result,
            all_services, team, tier, noise_cfg["noise_level"],
        )
        direction = "dependents" if action_type == "query_dependents" else "dependencies"
        fallback  = self.fallback_registry(service_name, true_result, direction, difficulty)
        result    = self._call_llm(prompt, fallback=fallback)
        self._cache.set(seed, cache_key, service_name, result)
        return result

    def generate_runbook(
        self,
        seed: int,
        service_name: str,
        dependents: List[str],
        dependencies: List[str],
        team: str,
        tier: int,
        difficulty: str,
    ) -> str:
        """Generate or retrieve cached Confluence-style runbook excerpt."""
        cached = self._cache.get(seed, "runbook", service_name)
        if cached:
            return cached

        noise  = NOISE_CONFIG[difficulty]["noise_level"]
        prompt = self._runbook_prompt(service_name, dependents, dependencies, team, tier, noise)
        result = self._call_llm(
            prompt,
            fallback=self.fallback_runbook(service_name, dependents, dependencies),
        )
        self._cache.set(seed, "runbook", service_name, result)
        return result

    def generate_monitoring(
        self,
        seed: int,
        service_name: str,
        dependents: List[str],
        dependencies: List[str],
        difficulty: str,
    ) -> str:
        """Generate or retrieve cached Datadog-style monitoring snapshot."""
        cached = self._cache.get(seed, "monitoring", service_name)
        if cached:
            return cached

        noise  = NOISE_CONFIG[difficulty]["noise_level"]
        prompt = self._monitoring_prompt(service_name, dependents, dependencies, noise, seed)
        result = self._call_llm(
            prompt,
            fallback=self.fallback_monitoring(service_name, dependents, dependencies, seed),
        )
        self._cache.set(seed, "monitoring", service_name, result)
        return result

    def generate_changelog(
        self,
        seed: int,
        changed_service: str,
        team: str,
        difficulty: str,
    ) -> str:
        """Generate or retrieve detailed CHANGELOG.md entry for the changed service."""
        cached = self._cache.get(seed, "changelog_detail", changed_service)
        if cached:
            return cached

        prompt = self._changelog_detail_prompt(seed, changed_service, team, difficulty)
        result = self._call_llm(
            prompt,
            fallback=self.fallback_changelog(changed_service, seed),
        )
        self._cache.set(seed, "changelog_detail", changed_service, result)
        return result

    # ── Core LLM caller — retry 3× then fallback ─────────────────────────

    def _call_llm(self, prompt: str, fallback: str = "") -> str:
        if not self.is_active:
            return fallback
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=self.MAX_TOKENS,
                    temperature=0.0,
                    seed=42,          # Cerebras supports seed for determinism
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("LLM call attempt %d/%d failed: %s", attempt + 1, 3, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        logger.warning("All LLM retries exhausted — using fallback")
        return fallback

    # ── Prompt builders ────────────────────────────────────────────────────

    def _alert_prompt(
        self, seed: int, svc: str, team: str,
        language: str, tier: int, difficulty: str,
    ) -> str:
        clarity_map = {
            "easy":   "clear, specific, and actionable — mention the exact breaking change",
            "medium": "uses internal jargon and ticket references — somewhat ambiguous",
            "hard":   "confusing, references multiple unrelated services, uses vague language",
        }
        return (
            f"Generate a realistic PagerDuty P1 incident alert.\n"
            f"Service: {svc} | Team: {team} | Language: {language} | Tier: {tier}\n"
            f"Incident ID: INC-{seed % 9000 + 1000}\n"
            f"Alert tone: {clarity_map[difficulty]}\n"
            f"Include: incident ID, elevated error rate or SLO burn rate, specific "
            f"endpoint or schema field that changed, and a runbook link placeholder. "
            f"4-6 sentences.\n"
            f"Output ONLY the alert text."
        )

    def _changelog_prompt(
        self, seed: int, svc: str, team: str, difficulty: str,
    ) -> str:
        detail_map = {
            "easy":   "clearly states the breaking change and migration path",
            "medium": "uses JIRA tickets, internal abbreviations, partially unclear",
            "hard":   "mentions multiple services ambiguously, contradicts itself slightly",
        }
        return (
            f"Write a PR description for a breaking change to service '{svc}'.\n"
            f"PR #{seed % 900 + 100} | Team: {team}\n"
            f"Style: {detail_map[difficulty]}\n"
            f"2-3 sentences. Start with 'PR #{seed % 900 + 100}:'.\n"
            f"Output ONLY the PR text."
        )

    def _changelog_detail_prompt(
        self, seed: int, svc: str, team: str, difficulty: str,
    ) -> str:
        style_map = {
            "easy":   "be specific and accurate about which services are affected",
            "medium": "use jargon and be slightly vague about downstream impact",
            "hard":   "be confusing — mention some unrelated services as potentially affected",
        }
        return (
            f"Write an internal CHANGELOG.md entry for a breaking change to '{svc}'.\n"
            f"Version: 2.{seed % 90 + 10}.0 | Team: {team}\n"
            f"Sections: '### Breaking Changes' (2-3 specific bullets with API/schema details), "
            f"'### Migration Guide' (2 steps), "
            f"'### Known Affected Clients' (list 3-5 service names).\n"
            f"Style: {style_map[difficulty]}.\n"
            f"Output ONLY the CHANGELOG markdown."
        )

    def _registry_prompt(
        self,
        action_type: str,
        svc: str,
        true_result: List[str],
        all_services: List[str],
        team: str,
        tier: int,
        noise: float,
    ) -> str:
        direction = (
            "upstream consumers — services that call this service"
            if action_type == "query_dependents"
            else "downstream dependencies — services this service calls"
        )
        noise_pct = int(noise * 100)
        # Explicit lookup avoids float truncation bug (int(0.40*2)=0 instead of 1)
        _diff  = next((k for k, v in NOISE_CONFIG.items() if v["noise_level"] == noise), "easy")
        drop_n = NOISE_CONFIG[_diff]["dropped"]
        add_n  = NOISE_CONFIG[_diff]["hallucinated"]
        stale_note = (
            "Append: '⚠️  Registry data may be stale (last sync: 30+ days ago).'"
            if noise > 0.35 else ""
        )
        return (
            f"Simulate an internal service registry CLI lookup.\n"
            f"Query: {direction} for '{svc}' (team: {team}, tier: {tier})\n"
            f"True result (internal, do NOT expose directly): {true_result}\n"
            f"Known services pool: {all_services[:20]}\n"
            f"Noise level: {noise_pct}%\n"
            f"Instructions: silently drop {drop_n} entries from the true result. "
            f"Add {add_n} plausible-sounding fake service names from the pool. "
            f"{stale_note}\n"
            f"Format as CLI output: '[service-registry] Querying...' then a bulleted "
            f"list with service names, teams, and statuses.\n"
            f"Output ONLY the CLI tool output."
        )

    def _runbook_prompt(
        self,
        svc: str,
        dependents: List[str],
        dependencies: List[str],
        team: str,
        tier: int,
        noise: float,
    ) -> str:
        outdated = (
            "Include one stale consumer entry marked '(deprecated Q3 2024)'."
            if noise > 0.3 else ""
        )
        contradiction = (
            "Add a note: 'Scheduled for deprecation but still in active use (see INFRA-4491).'."
            if noise > 0.6 else ""
        )
        return (
            f"Write a realistic internal Confluence runbook excerpt for service '{svc}'.\n"
            f"Team: {team} | Tier: {tier}\n"
            f"Known dependents (who calls it): {dependents[:6]}\n"
            f"Known dependencies (what it calls): {dependencies[:6]}\n"
            f"Sections: '## Service Overview' (2 sentences), "
            f"'## Key Dependencies' (markdown table, 3-5 rows), "
            f"'## Known Consumers' (bullet list), "
            f"'## On-Call Notes' (2-3 bullets). ~180 words total.\n"
            f"{outdated} {contradiction}\n"
            f"Output ONLY the runbook markdown."
        )

    def _monitoring_prompt(
        self,
        svc: str,
        dependents: List[str],
        dependencies: List[str],
        noise: float,
        seed: int,
    ) -> str:
        rng = random.Random(seed + 7777)
        anomaly = (
            f"Show elevated error_rate (0.0{rng.randint(8, 15)}) for one dependent service "
            "to hint at a cascading failure."
            if noise > 0.3 else ""
        )
        note_val = (
            "Data window: last 1h. Some services missing if traffic < 10 req/min."
            if noise > 0.2 else "Data window: last 1h."
        )
        return (
            f"Generate a realistic Datadog-style service map API response for '{svc}'.\n"
            f"Include: 'dependents' list (who calls this service) and "
            f"'dependencies' list (what this service calls).\n"
            f"Real dependents: {dependents[:5]}\n"
            f"Real dependencies: {dependencies[:5]}\n"
            f"Per-service fields: calls_per_min (realistic), p99_ms, error_rate.\n"
            f"{anomaly}\n"
            f"Add a '_note' field: '{note_val}'\n"
            f"Output ONLY valid JSON."
        )

    # ── Fallback templates (always deterministic, no LLM needed) ──────────

    def fallback_alert(self, svc: str, seed: int) -> str:
        return (
            f"[PagerDuty] INCIDENT INC-{seed % 9000 + 1000} | P1 | TRIGGERED\n"
            f"Service: {svc}\n"
            f"Alert: Breaking API change detected — downstream consumers may be impacted.\n"
            f"Error rate elevated above SLO threshold.\n"
            f"Runbook: https://wiki.internal/runbooks/{svc}\n"
            f"Investigate all services that depend on {svc}."
        )

    def fallback_changelog(self, svc: str, seed: int) -> str:
        return (
            f"PR #{seed % 900 + 100}: Breaking change to {svc}\n"
            f"This PR introduces a breaking API change to {svc}. "
            f"Downstream consumers will need to update their integration."
        )

    def fallback_registry(
        self, svc: str, true_result: List[str], direction: str, difficulty: str
    ) -> str:
        # Apply noise to fallback too
        shown = list(true_result)
        if difficulty == "medium" and len(shown) > 1:
            shown = shown[:-1]
        elif difficulty == "hard" and len(shown) > 2:
            shown = shown[:-2]
        lines = "\n".join(f"  - {s} (status: active)" for s in shown)
        stale = (
            "\n⚠️  Registry data may be stale (last sync: 30+ days ago)."
            if difficulty != "easy" else ""
        )
        return (
            f"[service-registry] Querying {direction} of '{svc}'...\n"
            f"Found {len(shown)} registered:\n{lines}{stale}"
        )

    def fallback_runbook(
        self, svc: str, dependents: List[str], dependencies: List[str]
    ) -> str:
        deps_tbl = "\n".join(f"| {d} | active |" for d in dependencies[:4])
        consumers = ", ".join(dependents[:5]) or "none registered"
        return (
            f"# {svc} — SRE Runbook\n\n"
            f"## Service Overview\n"
            f"{svc} is a core platform service managed by the on-call rotation.\n\n"
            f"## Key Dependencies\n| Service | Status |\n|---|---|\n{deps_tbl}\n\n"
            f"## Known Consumers\n{consumers}\n\n"
            f"## On-Call Notes\n"
            f"- Check service dashboards on any incident.\n"
            f"- Escalate if error rate exceeds 10% for >5 minutes."
        )

    def fallback_monitoring(
        self,
        svc: str,
        dependents: List[str],
        dependencies: List[str],
        seed: int,
    ) -> str:
        rng = random.Random(seed)
        dep_entries = [
            {
                "service":       d,
                "calls_per_min": rng.randint(100, 10000),
                "p99_ms":        rng.randint(5, 200),
                "error_rate":    round(rng.random() * 0.005, 4),
            }
            for d in dependents[:4]
        ]
        deps_entries = [
            {
                "service":       d,
                "calls_per_min": rng.randint(100, 5000),
                "p99_ms":        rng.randint(2, 50),
                "error_rate":    round(rng.random() * 0.003, 4),
            }
            for d in dependencies[:4]
        ]
        data = {
            "service":      svc,
            "env":          "production",
            "dependents":   dep_entries,
            "dependencies": deps_entries,
            "_note":        "Fallback mode — LLM unavailable. Data is synthetic.",
        }
        return json.dumps(data, indent=2)
