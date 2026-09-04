"""Per-record narrative, built only from values that exist.

Two rules govern every sentence produced here:

**Never assert a signal that was not measured.** If ``z_duration`` is NaN the
text says the duration could not be compared and why - it does not say the
duration was normal. "No temporal anomaly" and "no temporal evidence" are
different claims, and conflating them is how a reviewer ends up trusting a
record nobody could check.

**Always say what the confidence permits.** A record with a large deviation and
low confidence must read as a data problem, not a fraud hypothesis, because
that is what it is until the evidence is repaired.

The text is generated from the already-computed columns. It recomputes nothing,
so an explanation can never disagree with the decision it explains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.constants import ANOMALY_TYPES, COST_SCOPES
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Why a deviation was unavailable, in plain words. Keyed by Stage 3's reason.
REASON_PHRASES: Mapping[str, str] = {
    "feature_missing": "the record does not carry the underlying value",
    "cell_unstable": "its peer cell is too small to compare against",
    "no_peer_norm": "its peer group has too few high-confidence members",
    "zero_dispersion": "every peer reports an identical value, leaving no scale",
    "cluster_noise": "its work name could not be matched to any work type",
}

#: Human phrasing for each anomaly type.
TYPE_PHRASES: Mapping[str, str] = {
    "cost_outlier": "cost is unusual for its peer group",
    "overspend_anomaly": "spending runs high against comparable works",
    "underspend_anomaly": "a completed work reports unusually low spending",
    "temporal_outlier": "the project duration is unusual for its peer group",
    "duplicate_suspect": "a near-duplicate work exists nearby in the same district",
    "low_confidence": "the record's own evidence is too weak to rely on",
    "insufficient_context": "there is no usable peer comparison",
}

_SIGNAL_LABEL: Mapping[str, str] = {
    "z_cost": "cost",
    "z_spend": "spending",
    "z_duration": "duration",
}


def _fmt(value: Any, digits: int = 2) -> Optional[str]:
    """Format a number, or return None when it is not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return f"{number:.{digits}f}"


def explain_record(row: Mapping[str, Any]) -> str:
    """Compose the explanation for one already-decided record.

    Args:
        row: A mapping of the Stage 4 output columns for a single record,
            plus the Stage 3 reason columns.

    Returns:
        A single paragraph naming the decision, the evidence behind it, the
        confidence that governs it, and - explicitly - whatever could not be
        measured.
    """
    decision = str(row.get("decision_class", "MONITOR"))
    confidence = _fmt(row.get("confidence"))
    confidence_flag = str(row.get("confidence_flag", "high"))
    types = [name for name in ANOMALY_TYPES if bool(row.get(f"type_{name}", False))]

    parts: List[str] = []

    # --- what was measured -------------------------------------------------
    measured: List[str] = []
    for column, label in _SIGNAL_LABEL.items():
        formatted = _fmt(row.get(column))
        if formatted is None:
            continue
        if column == "z_cost":
            scope = str(row.get("cost_scope", "none"))
            where = (
                "within its cost band"
                if scope == "cell"
                else "against its whole work type"
            )
            measured.append(f"{label} z={formatted} {where}")
        else:
            measured.append(f"{label} z={formatted}")

    # --- what could not be measured, and why -------------------------------
    unmeasured: List[str] = []
    for column, reason_column, label in (
        ("z_cost", "deviation_cell_cost_reason", "cost"),
        ("z_spend", "deviation_spend_ratio_reason", "spending"),
        ("z_duration", "deviation_duration_reason", "duration"),
    ):
        if _fmt(row.get(column)) is not None:
            continue
        reason = str(row.get(reason_column, "feature_missing"))
        unmeasured.append(f"{label} ({REASON_PHRASES.get(reason, reason)})")

    # --- lead sentence, by decision ----------------------------------------
    if decision == "REMEDIATE":
        lead = (
            f"Routed to REMEDIATE: confidence is {confidence or 'unavailable'}, "
            "below the gate, so no finding here can be treated as a fraud "
            "signal until the record's evidence is repaired"
        )
        if measured:
            lead += f". Deviations were still measured ({'; '.join(measured)}) "
            lead += "and are retained for the remediation queue"
        parts.append(lead + ".")
    elif decision == "INSUFFICIENT_CONTEXT":
        parts.append(
            "Routed to INSUFFICIENT_CONTEXT: no peer comparison was possible, so "
            "this record is neither cleared nor flagged - it is unassessed."
        )
    elif decision == "INVESTIGATE":
        parts.append(
            f"Routed to INVESTIGATE: {'; '.join(measured)}, with confidence "
            f"{confidence or 'unavailable'} ({confidence_flag})."
        )
    else:
        detail = f" ({'; '.join(measured)})" if measured else ""
        parts.append(
            f"Routed to MONITOR: measurable signals sit within normal range for "
            f"its peer group{detail}, at confidence {confidence or 'unavailable'}."
        )

    # --- named findings ----------------------------------------------------
    findings = [
        TYPE_PHRASES[name]
        for name in types
        if name not in {"low_confidence", "insufficient_context"}
    ]
    if findings:
        parts.append("Findings: " + "; ".join(findings) + ".")

    # --- peer context ------------------------------------------------------
    label = row.get("cluster_label")
    if isinstance(label, str) and label and label != "unclustered":
        stable = bool(row.get("peer_cell_stable", False))
        parts.append(
            f"Compared against works of type '{label}'"
            + (
                f", peer cell of {int(row.get('peer_cell_size', 0))} records."
                if stable
                else ", but its peer cell is not stable enough to rely on."
            )
        )
    elif not bool(row.get("cluster_has_norm", False)):
        parts.append(
            "Its work name could not be matched to a work type, so no peer "
            "norm exists for it."
        )

    # --- what is unknown ---------------------------------------------------
    if unmeasured:
        parts.append(
            "Not assessed: " + "; ".join(unmeasured) + " - absence of a signal "
            "here means it could not be measured, not that it was normal."
        )

    # --- duplicate, always secondary ---------------------------------------
    if bool(row.get("duplicate_flag", False)):
        score = _fmt(row.get("duplicate_score"))
        parts.append(
            f"A near-duplicate was found (similarity {score}); this is "
            "supporting evidence only and does not drive the decision."
        )

    severity = _fmt(row.get("severity_score"), digits=3)
    parts.append(
        f"Severity {severity} over {int(row.get('valid_signal_count', 0))} "
        "usable signal(s)."
        if severity is not None
        else "Severity is undefined because no signal was usable."
    )

    return " ".join(parts)


def build_explanations(frame: pd.DataFrame) -> pd.Series:
    """Generate the explanation for every record.

    Args:
        frame: The assembled Stage 4 frame, joined with the Stage 3 reason
            columns and cluster context.

    Returns:
        Object Series of explanation text, aligned to ``frame.index``.
    """
    if len(frame) == 0:
        return pd.Series([], dtype="object", index=frame.index, name="explanation_text")
    columns = list(frame.columns)
    texts = [
        explain_record(dict(zip(columns, values)))
        for values in frame.itertuples(index=False, name=None)
    ]
    LOGGER.info("Generated %d explanation(s).", len(texts))
    return pd.Series(texts, index=frame.index, dtype="object", name="explanation_text")
