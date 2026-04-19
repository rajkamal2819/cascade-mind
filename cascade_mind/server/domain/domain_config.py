"""DomainConfig — Dataclass describing a causal graph domain.

Each domain supplies:
* A list of node names (services / suppliers / …).
* A list of directed edges as (source, target) tuples.
* Human-readable labels used in playground UI and prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainConfig:
    name: str                                    # e.g. "sre", "supply_chain"
    display_name: str                            # e.g. "SRE / Microservices"
    nodes: tuple[str, ...]                       # ordered node names
    edges: tuple[tuple[str, str], ...]           # (source, target) directed edges
    node_type_label: str = "service"             # e.g. "service", "supplier", "node"
    edge_type_label: str = "calls"               # e.g. "calls", "supplies", "depends on"
    task_description: str = (
        "Identify all downstream nodes affected by the changed node."
    )
    tool_labels: dict[str, str] = field(
        default_factory=lambda: {
            "query_runbook":         "Query Runbook",
            "query_changelog":       "Query Changelog",
            "query_monitoring":      "Query Monitoring",
            "query_impact_registry": "Query Impact Registry",
            "submit_hypothesis":     "Submit Hypothesis",
        }
    )
    # Per-node metadata: {node_name: {"tier": int, "type": str, "has_alt": bool}}
    node_metadata: dict[str, dict] = field(default_factory=dict)
    # Real-world incident archetypes injected into the reset observation
    incident_archetypes: tuple[str, ...] = field(default_factory=tuple)
