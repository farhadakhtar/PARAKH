"""Structured explanation *inputs* for Stage 4 (brief step 9).

Stage 3 assembles the evidence; Stage 4 writes the verdict. This module
therefore returns a structured record - cluster context, peer norms, deviations
with their definedness reasons, and the Stage 2 gating signals - and stops short
of narrating a conclusion.

The one thing it does say in words is *why a deviation could not be computed*,
because that is a statement about the data rather than about the record's
riskiness, and it is exactly what a remediation queue needs.

Like Stage 2's ``explain_confidence``, this reads stored outputs and recomputes
nothing: an explanation derived from a fresh computation could disagree with the
values it claims to explain.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.constants import MISSING_STRATUM, NOISE_CLUSTER_ID
from src.core.logger import get_logger
from src.stage3.deviations import DEVIATION_SPECS

LOGGER = get_logger(__name__)

#: Human-readable gloss for each undefined reason.
REASON_TEXT: Dict[str, str] = {
    "defined": "measured against the peer norm",
    "feature_missing": "the record does not carry this value",
    "cell_unstable": "the peer cell is too small or built on unclustered records",
    "no_peer_norm": "the peer group has too few high-confidence members to "
    "define a norm",
    "zero_dispersion": "every high-confidence peer reports an identical value, "
    "so no scale exists to measure against",
}


def _scalar(value: Any, digits: int = 4) -> Optional[float]:
    """Round a value, mapping non-finite to ``None`` for clean JSON."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(number) else round(number, digits)


def build_explanation_inputs(
    records: pd.DataFrame,
    row: Any,
    cluster_labels: Optional[Dict[int, str]] = None,
    peer_cell_keys: Optional[Dict[int, Any]] = None,
    cell_stats: Optional[pd.DataFrame] = None,
    cluster_stats: Optional[pd.DataFrame] = None,
    specs: Sequence[Any] = DEVIATION_SPECS,
) -> Dict[str, Any]:
    """Assemble everything Stage 4 needs to explain one record.

    Args:
        records: Corpus frame with Stage 2 and Stage 3 columns attached.
        row: Index label of the record.
        cluster_labels: ``cluster_id -> top-term label``.
        peer_cell_keys: ``peer_cell_id -> (cluster, stratum)``.
        cell_stats: Per-cell peer statistics.
        cluster_stats: Per-cluster peer statistics.
        specs: Deviation specifications to report.

    Returns:
        A JSON-serialisable dict with ``context``, ``peer_norms``,
        ``deviations``, ``duplicates`` and ``confidence`` sections.

    Raises:
        KeyError: If ``row`` is not in the frame.
        ValueError: If the Stage 3 columns are absent.
    """
    if "peer_cell_id" not in records.columns:
        raise ValueError(
            "Stage 3 outputs are absent from the frame. Run "
            "SemanticLayer().run(corpus) before requesting explanation inputs."
        )
    if row not in records.index:
        raise KeyError(f"row {row!r} is not in the frame index")

    record = records.loc[row]
    cluster_id = int(record["cluster_id"])
    cell_id = int(record["peer_cell_id"])
    labels = cluster_labels or {}

    context: Dict[str, Any] = {
        "cluster_id": cluster_id,
        "cluster_label": labels.get(cluster_id, "unclustered")
        if cluster_id != NOISE_CLUSTER_ID
        else "unclustered",
        "cluster_size": int(record.get("cluster_size", 0)),
        "is_noise": bool(record.get("cluster_is_noise", False)),
        "cost_stratum": int(record["cost_stratum"]),
        "peer_cell_id": cell_id,
        "peer_cell_size": int(record["peer_cell_size"]),
        "peer_cell_stable": bool(record["peer_cell_stable"]),
        "contributed_to_peer_norm": bool(record.get("peer_reference", False)),
    }
    if peer_cell_keys and cell_id in peer_cell_keys:
        cluster_key, stratum_key = peer_cell_keys[cell_id]
        context["peer_cell_key"] = [int(cluster_key), int(stratum_key)]
    if context["cost_stratum"] == MISSING_STRATUM:
        context["stratum_note"] = "no usable sanctioned amount, so no cost band"

    peer_norms: Dict[str, Any] = {}
    for source, frame_stats, key in (
        ("cell", cell_stats, cell_id),
        ("cluster", cluster_stats, cluster_id),
    ):
        if frame_stats is None or key not in frame_stats.index:
            continue
        stats_row = frame_stats.loc[key]
        entry: Dict[str, Any] = {"n_reference": int(stats_row.get("n_reference", 0))}
        for column in frame_stats.columns:
            if column.endswith(("_median", "_mad")):
                entry[column] = _scalar(stats_row[column])
        peer_norms[source] = entry

    deviations: Dict[str, Any] = {}
    for name, feature, level in specs:
        if name not in records.columns:
            continue
        reason = str(record.get(f"{name}_reason", "feature_missing"))
        deviations[name] = {
            "feature": feature,
            "level": level,
            "value": _scalar(record[name]),
            "record_value": _scalar(record.get(feature)),
            "defined": reason == "defined",
            "reason": reason,
            "reason_text": REASON_TEXT.get(reason, reason),
        }

    duplicates = {
        "duplicate_score": _scalar(record.get("duplicate_score")),
        "duplicate_flag": bool(record.get("duplicate_flag", False)),
        "duplicate_group_id": int(record.get("duplicate_group_id", -1)),
    }

    confidence = {
        "confidence": _scalar(record.get("confidence")),
        "completeness": _scalar(record.get("completeness")),
        "temporal": _scalar(record.get("temporal")),
        "reconciliation": _scalar(record.get("reconciliation")),
        "critical_missing_count": int(record.get("critical_missing_count", 0)),
        "critical_deficit": _scalar(record.get("critical_deficit")),
        "cluster_penalty_factor": _scalar(record.get("cluster_penalty_factor")),
        "temporal_hard_fail": bool(record.get("temporal_hard_fail", False)),
        "reconciliation_branch": str(record.get("reconciliation_branch", "unknown")),
        "lifecycle_state": str(record.get("lifecycle_state", "unknown")),
        "spend_ratio": _scalar(record.get("spend_ratio")),
    }

    return {
        "row": row,
        "work_id": record.get("work_id"),
        "stage": "stage3.structure",
        "note": (
            "Structural evidence only. Stage 3 does not score or classify "
            "anomalies; Stage 4 owns that decision."
        ),
        "context": context,
        "peer_norms": peer_norms,
        "deviations": deviations,
        "duplicates": duplicates,
        "confidence": confidence,
    }
