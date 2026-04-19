"""SRE / Microservices domain — 30-node dependency graph.

This mirrors the existing cascade-mind graph topology (same service names and
edges as service_impact_environment.py) but expressed as a DomainConfig so
the domain abstraction layer can reference it.
"""

from .domain_config import DomainConfig

_NODES: tuple[str, ...] = (
    "api-gateway",
    "auth-service",
    "user-service",
    "order-service",
    "payment-service",
    "inventory-service",
    "notification-service",
    "shipping-service",
    "analytics-service",
    "search-service",
    "recommendation-service",
    "cart-service",
    "product-service",
    "review-service",
    "media-service",
    "cdn-service",
    "cache-service",
    "db-primary",
    "db-replica",
    "message-queue",
    "event-bus",
    "config-service",
    "secrets-service",
    "logging-service",
    "monitoring-service",
    "alerting-service",
    "rate-limiter",
    "load-balancer",
    "service-mesh",
    "identity-provider",
)

# Directed edges — same topology as the environment graph builder
_EDGES: tuple[tuple[str, str], ...] = (
    ("api-gateway",          "auth-service"),
    ("api-gateway",          "rate-limiter"),
    ("api-gateway",          "load-balancer"),
    ("auth-service",         "identity-provider"),
    ("auth-service",         "secrets-service"),
    ("auth-service",         "cache-service"),
    ("user-service",         "db-primary"),
    ("user-service",         "cache-service"),
    ("user-service",         "notification-service"),
    ("order-service",        "payment-service"),
    ("order-service",        "inventory-service"),
    ("order-service",        "shipping-service"),
    ("order-service",        "notification-service"),
    ("order-service",        "message-queue"),
    ("payment-service",      "db-primary"),
    ("payment-service",      "notification-service"),
    ("payment-service",      "analytics-service"),
    ("inventory-service",    "db-primary"),
    ("inventory-service",    "cache-service"),
    ("inventory-service",    "product-service"),
    ("shipping-service",     "notification-service"),
    ("shipping-service",     "analytics-service"),
    ("search-service",       "cache-service"),
    ("search-service",       "product-service"),
    ("search-service",       "recommendation-service"),
    ("recommendation-service", "cache-service"),
    ("recommendation-service", "analytics-service"),
    ("cart-service",         "cache-service"),
    ("cart-service",         "order-service"),
    ("product-service",      "db-replica"),
    ("product-service",      "media-service"),
    ("review-service",       "db-primary"),
    ("review-service",       "notification-service"),
    ("media-service",        "cdn-service"),
    ("analytics-service",    "message-queue"),
    ("analytics-service",    "db-replica"),
    ("message-queue",        "event-bus"),
    ("event-bus",            "logging-service"),
    ("event-bus",            "monitoring-service"),
    ("monitoring-service",   "alerting-service"),
    ("config-service",       "service-mesh"),
    ("logging-service",      "db-replica"),
    ("db-primary",           "db-replica"),
    ("load-balancer",        "service-mesh"),
    ("service-mesh",         "config-service"),
)

SRE_DOMAIN = DomainConfig(
    name="sre",
    display_name="SRE / Microservices",
    nodes=_NODES,
    edges=_EDGES,
    node_type_label="service",
    edge_type_label="calls",
    task_description=(
        "A service was changed/deployed. Identify all downstream microservices "
        "that may be impacted via transitive dependency chains."
    ),
)
