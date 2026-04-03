"""
client.py
---------
WebSocket client for service_impact_env.

Implements the EnvClient interface from openenv-core.
Agents use this class to connect to a running environment server,
either locally or on a Hugging Face Space.

Usage (async):
    async with ServiceImpactEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(seed=0)          # seed 0 → easy task
        print(result.observation.changed_service)
        result = await env.step(ServiceImpactAction(
            action_type="query_dependents",
            service_name="email_service"
        ))
        print(result.observation.result)

Usage (sync):
    with ServiceImpactEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(seed=0)
        result = env.step(ServiceImpactAction(
            action_type="submit",
            affected_services=["notification_service", "order_service"]
        ))
        print(result.reward)  # F1 score

From HuggingFace Space (point ENV_BASE_URL at the live Space):
    async with ServiceImpactEnv(base_url="https://rajkamal2819-cascade-mind.hf.space") as env:
        result = await env.reset(seed=0)
"""
from __future__ import annotations

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except ImportError:
    # Stub for local dev without openenv-core installed
    from dataclasses import dataclass
    from typing import Generic, Optional, TypeVar
    ObsT = TypeVar("ObsT")

    @dataclass
    class StepResult(Generic[ObsT]):  # type: ignore[no-redef]
        observation: ObsT
        reward: Optional[float]
        done: bool

    class EnvClient:  # type: ignore[no-redef]
        def __init__(self, base_url: str = "http://localhost:8000"): ...
        def _step_payload(self, action): ...
        def _parse_result(self, payload): ...
        def _parse_state(self, payload): ...

try:
    from .models import ServiceImpactAction, ServiceImpactObservation, ServiceImpactState
except ImportError:
    from models import ServiceImpactAction, ServiceImpactObservation, ServiceImpactState  # type: ignore


class ServiceImpactEnv(
    EnvClient  # type: ignore[misc]
):
    """Async/sync WebSocket client for the Service Impact Analysis environment.

    Connects to a running ServiceImpactEnvironment server and provides
    the standard OpenEnv client interface: reset(), step(), state().
    """

    def _step_payload(self, action: ServiceImpactAction) -> dict:
        """Serialize the Action model to a flat dict for the /ws step message.

        The server's WSStepMessage.data field receives this dict directly
        and deserializes it into ServiceImpactAction — so field names must
        match exactly (no nesting under an "action" key).
        """
        payload: dict = {"action_type": action.action_type}
        if action.service_name is not None:
            payload["service_name"] = action.service_name
        if action.affected_services is not None:
            payload["affected_services"] = action.affected_services
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: dict) -> "StepResult[ServiceImpactObservation]":
        """Parse the server's WebSocket observation response.

        Server response shape:
            {
                "observation": { ...ServiceImpactObservation fields... },
                "reward": float | null,
                "done": bool
            }
        """
        obs_data = payload.get("observation", {})

        # Build observation — handle both nested and flat payloads
        obs = ServiceImpactObservation(
            changed_service=obs_data.get("changed_service", ""),
            result=obs_data.get("result", []),
            queries_remaining=obs_data.get("queries_remaining", 0),
            message=obs_data.get("message", ""),
            done=obs_data.get("done", payload.get("done", False)),
            reward=obs_data.get("reward", payload.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> ServiceImpactState:
        """Parse the server's /state WebSocket response."""
        return ServiceImpactState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            changed_service=payload.get("changed_service", ""),
            queries_used=payload.get("queries_used", 0),
            max_queries=payload.get("max_queries", 15),
            correct_affected=payload.get("correct_affected", []),
            predicted_affected=payload.get("predicted_affected", []),
            task_difficulty=payload.get("task_difficulty", "easy"),
            episode_ended=payload.get("episode_ended", False),
        )
