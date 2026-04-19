"""
server/graph_builder.py
------------------------
Builds the seed-perturbed synthetic microservice dependency knowledge graph.

Each episode seed produces a unique topology via deterministic edge perturbation:
  - 4-9 redundant edges removed (edges with alternative paths — safe to remove)
  - 3-7 plausible candidate edges added from CANDIDATE_EDGES

Graph convention:
  Edge A → B  means  "A depends on B"
  (i.e. if B changes, A is potentially affected)

Key networkx queries:
  nx.ancestors(G, service)   → all affected upstream services (scoring)
  G.predecessors(service)    → immediate callers (query_dependents)
  G.successors(service)      → immediate dependencies (query_dependencies)

get_scenario(G, seed) selects changed_service dynamically by betweenness-
centrality binned by difficulty (seed % 3), giving 10,000+ unique episodes.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

# Difficulty ordering (seed % 3 → index)
DIFFICULTY_ORDER = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Service catalogue (30 services)
# ---------------------------------------------------------------------------

SERVICES: List[str] = [
    # Tier 1 — Gateway / Frontend
    "api_gateway",
    "mobile_backend",
    "web_backend",
    # Tier 2 — Business Logic
    "auth_service",
    "user_service",
    "order_service",
    "cart_service",
    "checkout_service",
    "payment_service",
    "billing_service",
    "subscription_service",
    "inventory_service",
    "shipping_service",
    "catalog_service",
    "search_service",
    "recommendation_service",
    "review_service",
    "notification_service",
    # Tier 3 — Support
    "email_service",
    "sms_service",
    "media_service",
    "cdn_service",
    # Tier 4 — Data / ML
    "analytics_service",
    "reporting_service",
    "ml_service",
    "cache_service",
    "database_service",
    # Tier 5 — Infrastructure
    "config_service",
    "logging_service",
    "metrics_service",
]

# Service metadata for richer observations
SERVICE_METADATA: Dict[str, Dict] = {
    "api_gateway":          {"team": "platform",  "tier": 1, "language": "go"},
    "mobile_backend":       {"team": "platform",  "tier": 1, "language": "node"},
    "web_backend":          {"team": "platform",  "tier": 1, "language": "python"},
    "auth_service":         {"team": "platform",  "tier": 2, "language": "go"},
    "user_service":         {"team": "platform",  "tier": 2, "language": "python"},
    "order_service":        {"team": "commerce",  "tier": 2, "language": "java"},
    "cart_service":         {"team": "commerce",  "tier": 2, "language": "python"},
    "checkout_service":     {"team": "commerce",  "tier": 2, "language": "java"},
    "payment_service":      {"team": "commerce",  "tier": 2, "language": "java"},
    "billing_service":      {"team": "commerce",  "tier": 2, "language": "python"},
    "subscription_service": {"team": "commerce",  "tier": 2, "language": "python"},
    "inventory_service":    {"team": "commerce",  "tier": 2, "language": "go"},
    "shipping_service":     {"team": "commerce",  "tier": 2, "language": "python"},
    "catalog_service":      {"team": "commerce",  "tier": 2, "language": "python"},
    "search_service":       {"team": "commerce",  "tier": 2, "language": "java"},
    "recommendation_service": {"team": "ml",      "tier": 2, "language": "python"},
    "review_service":       {"team": "commerce",  "tier": 2, "language": "python"},
    "notification_service": {"team": "platform",  "tier": 3, "language": "node"},
    "email_service":        {"team": "platform",  "tier": 3, "language": "python"},
    "sms_service":          {"team": "platform",  "tier": 3, "language": "python"},
    "media_service":        {"team": "platform",  "tier": 3, "language": "go"},
    "cdn_service":          {"team": "infra",     "tier": 3, "language": "go"},
    "analytics_service":    {"team": "data",      "tier": 4, "language": "python"},
    "reporting_service":    {"team": "data",      "tier": 4, "language": "python"},
    "ml_service":           {"team": "ml",        "tier": 4, "language": "python"},
    "cache_service":        {"team": "infra",     "tier": 4, "language": "go"},
    "database_service":     {"team": "infra",     "tier": 4, "language": "go"},
    "config_service":       {"team": "infra",     "tier": 5, "language": "go"},
    "logging_service":      {"team": "infra",     "tier": 5, "language": "python"},
    "metrics_service":      {"team": "infra",     "tier": 5, "language": "go"},
}

# ---------------------------------------------------------------------------
# Static base dependency edges  (A → B = "A depends on B")
# ---------------------------------------------------------------------------

BASE_EDGES: List[Tuple[str, str]] = [
    # Tier 1 → Tier 2
    ("api_gateway",          "auth_service"),
    ("api_gateway",          "user_service"),
    ("api_gateway",          "catalog_service"),
    ("api_gateway",          "cart_service"),
    ("api_gateway",          "order_service"),
    ("api_gateway",          "search_service"),
    ("mobile_backend",       "auth_service"),
    ("mobile_backend",       "user_service"),
    ("mobile_backend",       "catalog_service"),
    ("mobile_backend",       "notification_service"),
    ("web_backend",          "auth_service"),
    ("web_backend",          "user_service"),
    ("web_backend",          "catalog_service"),
    ("web_backend",          "media_service"),
    # Auth + User
    ("auth_service",         "user_service"),
    ("auth_service",         "cache_service"),
    ("user_service",         "database_service"),
    ("user_service",         "cache_service"),
    # Commerce flow
    ("cart_service",         "catalog_service"),
    ("cart_service",         "user_service"),
    ("cart_service",         "recommendation_service"),
    ("cart_service",         "cache_service"),
    ("checkout_service",     "cart_service"),
    ("checkout_service",     "payment_service"),
    ("checkout_service",     "shipping_service"),
    ("checkout_service",     "inventory_service"),
    ("order_service",        "checkout_service"),
    ("order_service",        "inventory_service"),
    ("order_service",        "notification_service"),
    ("order_service",        "user_service"),
    ("payment_service",      "billing_service"),
    ("payment_service",      "notification_service"),
    ("billing_service",      "subscription_service"),
    ("billing_service",      "database_service"),
    ("subscription_service", "database_service"),
    ("inventory_service",    "database_service"),
    ("inventory_service",    "cache_service"),
    # Shipping + Catalog
    ("shipping_service",     "notification_service"),
    ("shipping_service",     "database_service"),
    ("catalog_service",      "database_service"),
    ("catalog_service",      "cache_service"),
    ("catalog_service",      "media_service"),
    # Search + Recommendation
    ("search_service",       "catalog_service"),
    ("search_service",       "ml_service"),
    ("search_service",       "cache_service"),
    ("recommendation_service", "ml_service"),
    ("recommendation_service", "catalog_service"),
    ("recommendation_service", "analytics_service"),
    ("recommendation_service", "cache_service"),
    # Reviews
    ("review_service",       "user_service"),
    ("review_service",       "catalog_service"),
    ("review_service",       "database_service"),
    # Notifications
    ("notification_service", "email_service"),
    ("notification_service", "sms_service"),
    # Media
    ("media_service",        "cdn_service"),
    ("media_service",        "cache_service"),
    # Analytics + Reporting + ML
    ("analytics_service",    "database_service"),
    ("analytics_service",    "cache_service"),
    ("reporting_service",    "analytics_service"),
    ("reporting_service",    "database_service"),
    ("ml_service",           "database_service"),
    ("ml_service",           "cache_service"),
    # Infra
    ("config_service",       "database_service"),
    ("logging_service",      "database_service"),
    ("metrics_service",      "database_service"),
    ("metrics_service",      "cache_service"),
]

# ---------------------------------------------------------------------------
# Candidate edges for seed-based perturbation
# Plausible connections not in the base graph — used to add variety per seed
# ---------------------------------------------------------------------------

CANDIDATE_EDGES: List[Tuple[str, str]] = [
    ("api_gateway",            "recommendation_service"),
    ("api_gateway",            "review_service"),
    ("mobile_backend",         "cart_service"),
    ("mobile_backend",         "search_service"),
    ("web_backend",            "search_service"),
    ("web_backend",            "review_service"),
    ("web_backend",            "recommendation_service"),
    ("order_service",          "billing_service"),
    ("order_service",          "shipping_service"),
    ("cart_service",           "inventory_service"),
    ("checkout_service",       "billing_service"),
    ("checkout_service",       "notification_service"),
    ("search_service",         "recommendation_service"),
    ("analytics_service",      "ml_service"),
    ("reporting_service",      "ml_service"),
    ("reporting_service",      "cache_service"),
    ("subscription_service",   "cache_service"),
    ("payment_service",        "database_service"),
    ("billing_service",        "cache_service"),
    ("user_service",           "config_service"),
    ("auth_service",           "config_service"),
    ("notification_service",   "logging_service"),
    ("media_service",          "database_service"),
    ("metrics_service",        "logging_service"),
    ("review_service",         "cache_service"),
    ("inventory_service",      "config_service"),
    ("shipping_service",       "config_service"),
]

# Fallback static scenarios (used when dynamic selection finds no suitable service)
_STATIC_SCENARIOS: Dict[str, Dict] = {
    "easy":   {"changed_service": "email_service",   "max_queries": 15},
    "medium": {"changed_service": "catalog_service", "max_queries": 12},
    "hard":   {"changed_service": "cache_service",   "max_queries": 10},
}


# ---------------------------------------------------------------------------
# Graph builder — seed-perturbed topology
# ---------------------------------------------------------------------------

def build_service_graph(seed: int = 42) -> nx.DiGraph:
    """Return a seed-perturbed DiGraph of service dependencies.

    Each unique seed produces a slightly different topology:
      - 4-9 redundant edges removed (edges that have an alternative path)
      - 3-7 plausible new edges added from CANDIDATE_EDGES

    This creates 10,000+ unique episodes from a single base architecture.

    Args:
        seed: Controls exactly which edges are removed/added.
              Same seed → identical graph (reproducible episodes).

    Returns:
        nx.DiGraph with 30 nodes and ~60-72 directed edges.
    """
    rng = random.Random(seed)

    G = nx.DiGraph()
    for svc in SERVICES:
        G.add_node(svc, **SERVICE_METADATA.get(svc, {}))
    G.add_edges_from(BASE_EDGES)

    # Remove redundant edges (those with an alternative path — safe to remove)
    removable = _find_redundant_edges(G)
    if removable:
        n_remove = rng.randint(4, min(9, len(removable)))
        to_remove = rng.sample(sorted(removable), n_remove)  # sorted for determinism
        G.remove_edges_from(to_remove)

    # Add candidate edges that don't create cycles
    available = [
        (u, v) for u, v in CANDIDATE_EDGES
        if not G.has_edge(u, v) and not _would_create_cycle(G, u, v)
    ]
    if available:
        n_add = rng.randint(3, min(7, len(available)))
        to_add = rng.sample(sorted(available), n_add)   # sorted for determinism
        G.add_edges_from(to_add)

    return G


def _find_redundant_edges(G: nx.DiGraph) -> List[Tuple[str, str]]:
    """Return edges (u, v) where an alternative path u→v exists without that edge.

    Removing a redundant edge keeps the graph connected for u→v traversals.
    """
    redundant = []
    for u, v in list(G.edges()):
        G_tmp = G.copy()
        G_tmp.remove_edge(u, v)
        try:
            if nx.has_path(G_tmp, u, v):
                redundant.append((u, v))
        except nx.NetworkXError:
            pass
    return redundant


def _would_create_cycle(G: nx.DiGraph, u: str, v: str) -> bool:
    """Return True if adding edge u→v would create a directed cycle."""
    try:
        return nx.has_path(G, v, u)
    except nx.NetworkXError:
        return False


# ---------------------------------------------------------------------------
# Scenario selector — dynamic changed_service from betweenness centrality
# ---------------------------------------------------------------------------

def get_scenario(G: nx.DiGraph, seed: int) -> Dict:
    """Select episode scenario from the seeded graph.

    Difficulty is determined by seed % 3:
      0 → easy   (low-betweenness node, few affected services: 1-6)
      1 → medium (mid-betweenness node, moderate affected: 6-14)
      2 → hard   (high-betweenness node, many affected: 14+)

    The changed_service is selected deterministically from the seed using
    betweenness centrality, so the same seed always produces the same episode.

    Returns:
        Dict with: changed_service, max_queries, difficulty, description,
                   hint, n_affected
    """
    difficulty_idx = seed % 3
    difficulty     = DIFFICULTY_ORDER[difficulty_idx]
    rng            = random.Random(seed + 9999)  # separate seed for service selection

    # Bin services by how many ancestors they have (= how many are affected)
    affected_counts = {svc: len(nx.ancestors(G, svc)) for svc in G.nodes()}

    easy_candidates   = sorted(s for s, c in affected_counts.items() if 1 <= c <= 6)
    medium_candidates = sorted(s for s, c in affected_counts.items() if 6 < c <= 13)
    hard_candidates   = sorted(s for s, c in affected_counts.items() if c > 13)

    # Non-empty fallbacks
    all_sorted = sorted(affected_counts, key=lambda s: affected_counts[s])
    if not easy_candidates:
        easy_candidates = all_sorted[:5]
    if not medium_candidates:
        mid = len(all_sorted) // 2
        medium_candidates = all_sorted[mid - 3: mid + 3]
    if not hard_candidates:
        hard_candidates = all_sorted[-5:]

    if difficulty == "easy":
        candidates  = easy_candidates
        max_queries = rng.choice([13, 14, 15])
        hint_tmpl   = "Start by querying who calls '{svc}' directly."
    elif difficulty == "medium":
        candidates  = medium_candidates
        max_queries = rng.choice([10, 11, 12])
        hint_tmpl   = "'{svc}' is shared by several services — explore broadly."
    else:  # hard
        candidates  = hard_candidates
        max_queries = rng.choice([8, 9, 10])
        hint_tmpl   = "'{svc}' is a critical shared dependency — map the full blast radius."

    static = _STATIC_SCENARIOS[difficulty]
    if candidates:
        changed_service = rng.choice(candidates)
    else:
        changed_service = static["changed_service"]
        max_queries     = static["max_queries"]

    n_affected = affected_counts.get(changed_service, 0)
    return {
        "changed_service": changed_service,
        "max_queries":     max_queries,
        "difficulty":      difficulty,
        "description": (
            f"{changed_service} has a breaking change. "
            f"Budget: {max_queries} queries. Identify all affected services."
        ),
        "hint":       hint_tmpl.format(svc=changed_service),
        "n_affected": n_affected,
    }


# ---------------------------------------------------------------------------
# Public query API (unchanged interface)
# ---------------------------------------------------------------------------


def get_affected_services(G: nx.DiGraph, changed_service: str) -> Set[str]:
    """Return ALL services affected by a change to `changed_service`.

    A service X is "affected" if it has a directed path TO `changed_service`
    — meaning X (transitively) depends on `changed_service`.

    Uses nx.ancestors(G, node) which returns all nodes that can reach `node`
    following directed edges.

    Args:
        G: The service dependency graph.
        changed_service: The service that was modified.

    Returns:
        Set of service names that are affected (not including the changed
        service itself).
    """
    if changed_service not in G:
        return set()
    return nx.ancestors(G, changed_service)


def get_direct_dependents(G: nx.DiGraph, service: str) -> List[str]:
    """Return services that DIRECTLY depend on (call) `service`.

    In graph terms: predecessors of `service` (nodes with edge → service).
    Corresponds to action_type='query_dependents'.
    """
    if service not in G:
        return []
    return sorted(G.predecessors(service))


def get_direct_dependencies(G: nx.DiGraph, service: str) -> List[str]:
    """Return services that `service` DIRECTLY depends on (calls).

    In graph terms: successors of `service` (nodes service has edge to).
    Corresponds to action_type='query_dependencies'.
    """
    if service not in G:
        return []
    return sorted(G.successors(service))


def get_all_services(G: nx.DiGraph) -> List[str]:
    """Return sorted list of all service names in the graph."""
    return sorted(G.nodes())


def get_service_metadata(G: nx.DiGraph, service: str) -> Optional[Dict]:
    """Return metadata dict for a service, or None if not found."""
    if service not in G:
        return None
    return dict(G.nodes[service])
