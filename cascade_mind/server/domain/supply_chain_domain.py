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
        "query_changelog":       "Query Change Log",
        "query_monitoring":      "Query Shipment Monitoring",
        "query_impact_registry": "Query Impact Registry",
        "submit_hypothesis":     "Submit Impact Hypothesis",
    },
)
