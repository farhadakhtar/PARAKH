"""Stage 10 - minimal entity graph.

Adjacency between resolved entities. Deliberately small: this is a lookup
structure, not network science. No centrality, no community detection, no
embedding - those belong to a later stage and would need entity resolution to
be trustworthy first, which is exactly what Stage 10 is establishing.

What it is for
--------------
Once works and vendors are entities rather than raw strings, three questions
become answerable without re-deriving anything:

* which works did this vendor touch
* which vendors touched this work
* how concentrated is a district or an agency

Why degree counts distinct counterparties
-----------------------------------------
A vendor with ten records for one work has degree one, not ten. Counting rows
would make repeated paperwork look like market presence, and every
concentration signal built on it would be measuring filing volume.

Edges inherit their endpoints' uncertainty
------------------------------------------
An edge into a degenerate or conflicting entity is not evidence of a
relationship - it is evidence about one record whose grouping is unresolved.
Each vendor node therefore carries the confidence mix of the works it
connects, so a caller can tell a real network from an artefact of bad
grouping.
"""

from __future__ import annotations

from typing import Any, Dict, Set

import pandas as pd

from src.core.logger import get_logger

LOGGER = get_logger(__name__)


def build_entity_graph(frame: pd.DataFrame) -> Dict[str, Any]:
    """Build vendor/work/agency/district adjacency from resolved entities.

    Args:
        frame: Output of :func:`resolve_work_entities` passed through
            :func:`resolve_vendor_entities`. Must carry ``work_entity_id``
            and ``vendor_entity_id``.

    Returns:
        A nested mapping with ``vendors``, ``works``, ``agencies`` and
        ``districts``. Every list is sorted, so the structure is
        deterministic and can be compared across runs.

    Raises:
        KeyError: If either entity column is absent.
    """
    for required in ("work_entity_id", "vendor_entity_id"):
        if required not in frame.columns:
            raise KeyError(f"entity graph requires a {required!r} column")

    if frame.empty:
        return {
            "vendors": {}, "works": {}, "agencies": {}, "districts": {},
            "n_edges": 0,
        }

    agency_column = next(
        (c for c in ("implementing_agency", "agency") if c in frame.columns), None
    )

    vendors: Dict[str, Dict[str, Set[str]]] = {}
    works: Dict[str, Dict[str, Set[str]]] = {}
    agencies: Dict[str, Set[str]] = {}
    districts: Dict[str, Set[str]] = {}

    for _, row in frame.iterrows():
        vendor = str(row["vendor_entity_id"])
        work = str(row["work_entity_id"])

        v = vendors.setdefault(
            vendor, {"works": set(), "agencies": set(), "districts": set()}
        )
        w = works.setdefault(
            work, {"vendors": set(), "agencies": set(), "districts": set()}
        )
        v["works"].add(work)
        w["vendors"].add(vendor)

        if agency_column:
            agency = row.get(agency_column)
            if pd.notna(agency) and str(agency).strip():
                agency = str(agency).strip()
                v["agencies"].add(agency)
                w["agencies"].add(agency)
                agencies.setdefault(agency, set()).add(work)

        district = row.get("district")
        if pd.notna(district) and str(district).strip():
            district = str(district).strip()
            v["districts"].add(district)
            w["districts"].add(district)
            districts.setdefault(district, set()).add(work)

    # Confidence mix per vendor node, so an edge into an unresolved entity is
    # visibly different from an edge into a settled one.
    work_confidence = (
        frame.groupby("work_entity_id")["entity_confidence"].first().to_dict()
        if "entity_confidence" in frame.columns
        else {}
    )

    graph: Dict[str, Any] = {
        "vendors": {
            vendor_id: {
                "connected_works": sorted(data["works"]),
                "connected_agencies": sorted(data["agencies"]),
                "connected_districts": sorted(data["districts"]),
                # Degree counts distinct works, never rows.
                "degree": len(data["works"]),
                "work_confidence_mix": _mix(data["works"], work_confidence),
            }
            for vendor_id, data in sorted(vendors.items())
        },
        "works": {
            work_id: {
                "connected_vendors": sorted(data["vendors"]),
                "connected_agencies": sorted(data["agencies"]),
                "connected_districts": sorted(data["districts"]),
                "degree": len(data["vendors"]),
            }
            for work_id, data in sorted(works.items())
        },
        "agencies": {
            name: {"connected_works": sorted(ws), "degree": len(ws)}
            for name, ws in sorted(agencies.items())
        },
        "districts": {
            name: {"connected_works": sorted(ws), "degree": len(ws)}
            for name, ws in sorted(districts.items())
        },
    }
    graph["n_edges"] = sum(node["degree"] for node in graph["vendors"].values())

    LOGGER.info(
        "Stage 10 graph: %d vendor node(s), %d work node(s), %d edge(s)",
        len(graph["vendors"]),
        len(graph["works"]),
        graph["n_edges"],
    )
    return graph


def _mix(work_ids: Set[str], confidence: Dict[str, str]) -> Dict[str, int]:
    """Confidence histogram over the works a node touches."""
    counts: Dict[str, int] = {}
    for work_id in work_ids:
        level = str(confidence.get(work_id, "UNKNOWN"))
        counts[level] = counts.get(level, 0) + 1
    return dict(sorted(counts.items()))


def graph_summary(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Headline counts, with the caveat that makes them readable."""
    vendors = graph.get("vendors", {})
    if not vendors:
        return {"n_vendors": 0, "n_works": 0, "n_edges": 0}
    degrees = sorted((n["degree"] for n in vendors.values()), reverse=True)
    return {
        "n_vendors": len(vendors),
        "n_works": len(graph.get("works", {})),
        "n_edges": graph.get("n_edges", 0),
        "max_vendor_degree": degrees[0],
        "median_vendor_degree": degrees[len(degrees) // 2],
        "single_work_vendors": sum(1 for d in degrees if d == 1),
        "_note": (
            "Degree counts distinct works, not rows. A high "
            "single-work-vendor count usually means vendor names are "
            "unresolvable rather than that the market is fragmented - check "
            "vendor_summary before reading it as market structure."
        ),
    }
