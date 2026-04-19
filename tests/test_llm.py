"""Quick LLM simulator smoke test — run with HF_TOKEN set."""
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# HF_TOKEN must be set in environment: export HF_TOKEN=hf_...

from cascade_mind.server.simulator.llm_simulator import LLMSimulator, SimulatorCache

cache = SimulatorCache(cache_path=tempfile.mktemp(suffix=".json"))
sim   = LLMSimulator(hf_token=os.environ["HF_TOKEN"], cache=cache, enabled=True)
print(f"is_active : {sim.is_active}")
assert sim.is_active, "LLM not active — check HF_TOKEN"

# --- Incident alert -----------------------------------------------------------
ctx = sim.generate_incident_context(
    seed=42, changed_service="catalog_service",
    team="commerce", language="python", tier=2, difficulty="easy",
)
print("\n=== INCIDENT ALERT (easy, Llama-3.1-8B via Cerebras) ===")
print(ctx.alert_text)

print("\n=== CHANGELOG ===")
print(ctx.changelog_text)

# --- Registry query (noisy, medium difficulty) --------------------------------
reg = sim.simulate_registry_query(
    seed=42, action_type="query_dependents",
    service_name="catalog_service",
    true_result=["api_gateway", "cart_service", "web_backend", "search_service", "review_service"],
    all_services=["api_gateway", "cart_service", "web_backend", "search_service",
                  "recommendation_service", "review_service", "order_service"],
    team="commerce", tier=2, difficulty="medium",
)
print("\n=== SERVICE REGISTRY (noisy, medium) ===")
print(reg)

# --- Runbook ------------------------------------------------------------------
rb = sim.generate_runbook(
    seed=42, service_name="catalog_service",
    dependents=["api_gateway", "cart_service", "search_service"],
    dependencies=["database_service", "cache_service", "media_service"],
    team="commerce", tier=2, difficulty="medium",
)
print("\n=== RUNBOOK (medium) ===")
print(rb[:500])

print(f"\nCache entries: {len(cache)}")
print("\nAll LLM simulator tests passed!")
