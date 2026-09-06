"""Stage 10 - entity / network intelligence.

A deterministic correction layer. It groups records into candidate entities
and reports how much it knows, without any model, learned parameter or
probability. Original columns are never modified; the entity structure is
attached alongside them.

The rule that governs it: **better unknown than wrong merge.**
"""

from src.stage10.entity_graph import build_entity_graph, graph_summary
from src.stage10.vendor_entity import (
    VENDOR_COLUMNS,
    normalise_vendor_name,
    resolve_vendor_entities,
    vendor_summary,
)
from src.stage10.work_entity import (
    ENTITY_COLUMNS,
    EntityResolutionError,
    entity_summary,
    name_similarity,
    resolve_work_entities,
)

__all__ = [
    "ENTITY_COLUMNS",
    "VENDOR_COLUMNS",
    "EntityResolutionError",
    "build_entity_graph",
    "entity_summary",
    "graph_summary",
    "name_similarity",
    "normalise_vendor_name",
    "resolve_vendor_entities",
    "resolve_work_entities",
    "vendor_summary",
]
