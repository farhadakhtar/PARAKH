"""Resolving what a *work* means when several records claim to be it.

``work_id`` is not unique. Stage 1 injects duplicate identifiers as a data
defect, and on the reference corpus 200 records share one across 100 groups -
**56 of which are routed to different actions**. A consumer joining on
``work_id`` silently fans out; a dashboard grouping by it double-counts.

This module aggregates without deciding, under three rules:

**Maximum, never mean.** Averaging the risk of two records that share an id
would invent a number describing neither, and would let a high-risk record be
diluted by a clean one. The maximum is the only aggregate that cannot hide a
finding.

**Conflict is a fact, not a resolution.** Where records disagree, the
disagreement is reported. This module does not pick a winner; picking one
would assert that the records describe the same work, which is exactly what a
duplicated identifier makes unknowable.

**Supporting records are always carried.** An aggregate a reader cannot open
is an aggregate they have to trust blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ACTION_CLASSES,
    STAGE65_VERSION,
    WORK_ID_AMBIGUITY_WARNING,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Columns of the resolution table.
WORK_RESOLUTION_COLUMNS: Tuple[str, ...] = (
    "work_id",
    "final_action",
    "conflict",
    "total_records",
    "max_risk_score",
    "max_confidence",
    "min_confidence",
    "action_distribution",
    "supporting_records",
)


def resolve_works(
    frame: pd.DataFrame,
    action_column: str = "action_class",
    risk_column: str = "risk_score",
    confidence_column: str = "confidence",
) -> pd.DataFrame:
    """Aggregate records to works, surfacing contradictions.

    Args:
        frame: Corpus frame with Stage 6 output. Read only.
        action_column: Column carrying the routed action.
        risk_column: Column carrying the risk score.
        confidence_column: Column carrying Stage 2 confidence.

    Returns:
        One row per distinct ``work_id``, ordered by id so the table is
        deterministic. ``final_action`` is the most severe action present when
        records agree, and the most severe present when they do not - the
        conflict flag, not the action, carries the disagreement.

    Raises:
        KeyError: If ``work_id`` or ``action_column`` is absent.
    """
    for required in ("work_id", action_column):
        if required not in frame.columns:
            raise KeyError(f"work resolution requires a {required!r} column")

    # Severity order: the most consequential action a work attracted. Taking
    # the most severe rather than the most common means a single escalation
    # among ten monitored records is never voted away by majority.
    severity = {name: position for position, name in enumerate(ACTION_CLASSES)}

    working = pd.DataFrame(
        {
            "work_id": frame["work_id"].astype("object").to_numpy(),
            "action": frame[action_column].astype("object").to_numpy(),
            "risk": pd.to_numeric(frame.get(risk_column), errors="coerce").to_numpy()
            if risk_column in frame.columns
            else np.full(len(frame), np.nan),
            "confidence": pd.to_numeric(
                frame.get(confidence_column), errors="coerce"
            ).to_numpy()
            if confidence_column in frame.columns
            else np.full(len(frame), np.nan),
        },
        index=frame.index,
    )

    rows: List[Dict[str, Any]] = []
    for work_id, group in working.groupby("work_id", sort=True):
        actions = list(group["action"])
        counts = group["action"].value_counts()
        risks = [value for value in group["risk"] if np.isfinite(value)]
        confidences = [value for value in group["confidence"] if np.isfinite(value)]
        rows.append(
            {
                "work_id": work_id,
                # Most severe, not most common. See above.
                "final_action": min(actions, key=lambda name: severity.get(name, 99)),
                "conflict": bool(len(set(actions)) > 1),
                "total_records": int(len(group)),
                # Maximum, never mean.
                "max_risk_score": max(risks) if risks else None,
                "max_confidence": max(confidences) if confidences else None,
                "min_confidence": min(confidences) if confidences else None,
                "action_distribution": {
                    str(k): int(v) for k, v in sorted(counts.items())
                },
                "supporting_records": [
                    {
                        "record_id": label,
                        "action": str(row.action),
                        "risk_score": float(row.risk) if np.isfinite(row.risk) else None,
                    }
                    for label, row in group.iterrows()
                ],
            }
        )

    table = pd.DataFrame(rows, columns=list(WORK_RESOLUTION_COLUMNS))
    n_conflict = int(table["conflict"].sum()) if len(table) else 0
    if n_conflict:
        LOGGER.warning(
            "%d work_id group(s) hold records routed to different actions. %s",
            n_conflict,
            WORK_ID_AMBIGUITY_WARNING,
        )
    LOGGER.info(
        "Resolved %d record(s) into %d work(s); %d in conflict.",
        len(frame),
        len(table),
        n_conflict,
    )
    return table


def conflicting_work_ids(resolution: pd.DataFrame) -> frozenset:
    """The work ids whose records disagree. Empty when the table is empty."""
    if not len(resolution):
        return frozenset()
    return frozenset(resolution.loc[resolution["conflict"], "work_id"])


def work_conflict_summary(resolution: pd.DataFrame) -> Dict[str, Any]:
    """Corpus-level view of the contradictions, for attachment to outputs."""
    if not len(resolution):
        return {"n_works": 0, "n_conflicting": 0, "_note": WORK_ID_AMBIGUITY_WARNING}
    conflicting = resolution.loc[resolution["conflict"]]
    return {
        "stage65_version": STAGE65_VERSION,
        "n_works": int(len(resolution)),
        "n_conflicting": int(len(conflicting)),
        "n_records_in_conflict": int(conflicting["total_records"].sum())
        if len(conflicting)
        else 0,
        "_note": WORK_ID_AMBIGUITY_WARNING,
        "_aggregation": (
            "Risk is aggregated by MAXIMUM and final_action by SEVERITY, never "
            "by mean or majority: a single escalation among many monitored "
            "records must not be averaged or voted away."
        ),
        "conflicting_work_ids": sorted(str(value) for value in conflicting["work_id"]),
    }
