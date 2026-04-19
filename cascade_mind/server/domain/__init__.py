"""Domain abstraction layer for cascade-mind.

Allows swapping the underlying causal graph (SRE, supply-chain, …)
without changing any environment or reward logic.
"""

from .domain_config import DomainConfig
from .sre_domain import SRE_DOMAIN
from .supply_chain_domain import SUPPLY_CHAIN_DOMAIN

DOMAINS: dict[str, DomainConfig] = {
    "sre": SRE_DOMAIN,
    "supply_chain": SUPPLY_CHAIN_DOMAIN,
}

__all__ = ["DomainConfig", "DOMAINS", "SRE_DOMAIN", "SUPPLY_CHAIN_DOMAIN"]
