"""Supply-chain domain — 30-node supplier / manufacturer / logistics graph.

Same causal-graph mechanics as the SRE domain but applied to supply-chain
disruptions.  A "changed node" represents a supplier failure, port closure,
or logistics hub disruption.  The agent must identify all downstream nodes
that will be affected.
"""

from .domain_config import DomainConfig

_NODES: tuple[str, ...] = (
    # Tier-0  Raw material suppliers
    "raw-materials-asia",
    "raw-materials-europe",
    "raw-materials-americas",
    # Tier-1  Component manufacturers
    "chipmaker-primary",
    "chipmaker-secondary",
    "pcb-manufacturer",
    "battery-manufacturer",
    "display-manufacturer",
    "mechanical-parts-supplier",
    # Tier-2  Sub-assembly
    "electronics-assembler",
    "battery-assembler",
    "chassis-assembler",
    # Tier-3  Final assembly
    "factory-primary",
    "factory-secondary",
    # Logistics / warehousing
    "port-east",
    "port-west",
    "air-freight-hub",
    "ground-logistics-primary",
    "ground-logistics-secondary",
    "regional-warehouse-na",
    "regional-warehouse-eu",
    "regional-warehouse-apac",
    # Distribution
    "wholesale-distributor",
    "retail-distributor",
    "ecommerce-fulfillment",
    # Quality / compliance
    "quality-control",
    "customs-broker",
    # Finance
    "trade-finance",
    "insurance-provider",
    # End customers
    "enterprise-customer",
)

_EDGES: tuple[tuple[str, str], ...] = (
    # Raw → Component
    ("raw-materials-asia",     "chipmaker-primary"),
    ("raw-materials-asia",     "battery-manufacturer"),
    ("raw-materials-europe",   "pcb-manufacturer"),
    ("raw-materials-europe",   "mechanical-parts-supplier"),
    ("raw-materials-americas", "chipmaker-secondary"),
    ("raw-materials-americas", "display-manufacturer"),
    # Component → Sub-assembly
    ("chipmaker-primary",      "electronics-assembler"),
    ("chipmaker-secondary",    "electronics-assembler"),
    ("pcb-manufacturer",       "electronics-assembler"),
    ("battery-manufacturer",   "battery-assembler"),
    ("display-manufacturer",   "chassis-assembler"),
    ("mechanical-parts-supplier", "chassis-assembler"),
    # Sub-assembly → Factory
    ("electronics-assembler",  "factory-primary"),
    ("battery-assembler",      "factory-primary"),
    ("chassis-assembler",      "factory-primary"),
    ("electronics-assembler",  "factory-secondary"),
    ("battery-assembler",      "factory-secondary"),
    # Quality / customs gate
    ("factory-primary",        "quality-control"),
    ("factory-secondary",      "quality-control"),
    ("quality-control",        "port-east"),
    ("quality-control",        "air-freight-hub"),
    ("port-east",              "customs-broker"),
    ("port-west",              "customs-broker"),
    ("air-freight-hub",        "customs-broker"),
    # Logistics → Warehousing
    ("customs-broker",         "ground-logistics-primary"),
    ("customs-broker",         "ground-logistics-secondary"),
    ("ground-logistics-primary",  "regional-warehouse-na"),
    ("ground-logistics-primary",  "regional-warehouse-eu"),
    ("ground-logistics-secondary","regional-warehouse-apac"),
    # Warehouse → Distribution
    ("regional-warehouse-na",  "wholesale-distributor"),
    ("regional-warehouse-na",  "ecommerce-fulfillment"),
    ("regional-warehouse-eu",  "retail-distributor"),
    ("regional-warehouse-apac","ecommerce-fulfillment"),
    # Finance / insurance
    ("factory-primary",        "trade-finance"),
    ("factory-secondary",      "trade-finance"),
    ("trade-finance",          "insurance-provider"),
    # End customer
    ("wholesale-distributor",  "enterprise-customer"),
    ("retail-distributor",     "enterprise-customer"),
    ("ecommerce-fulfillment",  "enterprise-customer"),
    # port-west path
    ("factory-primary",        "port-west"),
)

# Tier metadata for each node — used by LLM fallback templates
_NODE_METADATA: dict[str, dict] = {
    # Tier 0 — Raw materials
    "raw-materials-asia":       {"tier": 0, "type": "raw_supplier",   "has_alt": True},
    "raw-materials-europe":     {"tier": 0, "type": "raw_supplier",   "has_alt": True},
    "raw-materials-americas":   {"tier": 0, "type": "raw_supplier",   "has_alt": True},
    # Tier 1 — Component manufacturers
    "chipmaker-primary":        {"tier": 1, "type": "manufacturer",   "has_alt": True},
    "chipmaker-secondary":      {"tier": 1, "type": "manufacturer",   "has_alt": False},
    "pcb-manufacturer":         {"tier": 1, "type": "manufacturer",   "has_alt": True},
    "battery-manufacturer":     {"tier": 1, "type": "manufacturer",   "has_alt": True},
    "display-manufacturer":     {"tier": 1, "type": "manufacturer",   "has_alt": False},
    "mechanical-parts-supplier":{"tier": 1, "type": "parts_supplier", "has_alt": True},
    # Tier 2 — Sub-assembly
    "electronics-assembler":    {"tier": 2, "type": "assembler",      "has_alt": False},
    "battery-assembler":        {"tier": 2, "type": "assembler",      "has_alt": True},
    "chassis-assembler":        {"tier": 2, "type": "assembler",      "has_alt": True},
    # Tier 3 — Final assembly
    "factory-primary":          {"tier": 3, "type": "factory",        "has_alt": True},
    "factory-secondary":        {"tier": 3, "type": "factory",        "has_alt": False},
    # Tier 4 — Logistics / warehousing
    "port-east":                {"tier": 4, "type": "logistics",      "has_alt": True},
    "port-west":                {"tier": 4, "type": "logistics",      "has_alt": True},
    "air-freight-hub":          {"tier": 4, "type": "logistics",      "has_alt": True},
    "ground-logistics-primary": {"tier": 4, "type": "logistics",      "has_alt": True},
    "ground-logistics-secondary":{"tier": 4, "type": "logistics",     "has_alt": False},
    "regional-warehouse-na":    {"tier": 4, "type": "warehouse",      "has_alt": True},
    "regional-warehouse-eu":    {"tier": 4, "type": "warehouse",      "has_alt": True},
    "regional-warehouse-apac":  {"tier": 4, "type": "warehouse",      "has_alt": False},
    # Tier 5 — Distribution
    "wholesale-distributor":    {"tier": 5, "type": "distributor",    "has_alt": True},
    "retail-distributor":       {"tier": 5, "type": "distributor",    "has_alt": True},
    "ecommerce-fulfillment":    {"tier": 5, "type": "distributor",    "has_alt": True},
    # Compliance / finance
    "quality-control":          {"tier": 4, "type": "compliance",     "has_alt": False},
    "customs-broker":           {"tier": 4, "type": "compliance",     "has_alt": True},
    "trade-finance":            {"tier": 5, "type": "finance",        "has_alt": True},
    "insurance-provider":       {"tier": 5, "type": "finance",        "has_alt": True},
    "enterprise-customer":      {"tier": 6, "type": "customer",       "has_alt": False},
}

# Real-world disruption archetypes — shown in the incident alert at reset()
_INCIDENT_ARCHETYPES: tuple[str, ...] = (
    "Renesas Naka factory fire (2021) — one Tier-1 chip plant halted Toyota's 14 assembly lines globally.",
    "Ever Given / Suez blockage (2021) — 6-day canal closure cascaded into shortages across 50 product categories.",
    "COVID office-vs-home supply split (2020) — two separate toilet-paper supply chains; home demand spiked but home factories couldn't scale.",
    "Taiwan Strait tension (2024) — single-source TSMC dependency exposed; no alt fab for leading-edge nodes.",
    "Red Sea Houthi attacks (2024) — Maersk rerouted around Cape of Good Hope, adding 10-14 days to EU-Asia transit.",
)

SUPPLY_CHAIN_DOMAIN = DomainConfig(
    name="supply_chain",
    display_name="Supply-Chain Disruption",
    nodes=_NODES,
    edges=_EDGES,
    node_type_label="node",
    edge_type_label="supplies",
    task_description=(
        "A supply-chain node (supplier / factory / logistics hub) has been disrupted. "
        "Identify all downstream nodes that will be affected via transitive supply relationships."
    ),
    tool_labels={
        "query_runbook":         "Query SOP / Runbook",
        "query_changelog":       "Query Disruption Log",
        "query_monitoring":      "Query Shipment Monitoring",
        "query_impact_registry": "Query Impact Registry",
        "submit_hypothesis":     "Submit Impact Hypothesis",
    },
    node_metadata=_NODE_METADATA,
    incident_archetypes=_INCIDENT_ARCHETYPES,
)
