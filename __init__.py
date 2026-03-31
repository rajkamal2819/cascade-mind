"""
service_impact_env
------------------
Cross-service impact analysis environment for OpenEnv.

Public API:
    ServiceImpactAction      — agent action model
    ServiceImpactObservation — environment observation model
    ServiceImpactState       — internal episode state model
    ServiceImpactEnv         — WebSocket client (use this to connect to the server)

Quick start:
    import asyncio
    from service_impact_env import ServiceImpactEnv, ServiceImpactAction

    async def main():
        async with ServiceImpactEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(seed=0)   # seed 0 → easy task
            print(result.observation.changed_service)
            result = await env.step(ServiceImpactAction(
                action_type="query_dependents",
                service_name=result.observation.changed_service
            ))
            print(result.observation.result)

    asyncio.run(main())
"""
from .models import ServiceImpactAction, ServiceImpactObservation, ServiceImpactState
from .client import ServiceImpactEnv

__version__ = "0.1.0"
__all__ = [
    "ServiceImpactAction",
    "ServiceImpactObservation",
    "ServiceImpactState",
    "ServiceImpactEnv",
]
