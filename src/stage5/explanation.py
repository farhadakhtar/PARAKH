"""Per-record risk narrative that reconstructs its own arithmetic.

The hard requirement here is unusual and worth stating plainly: the text must
**show the multiplication**. A reader who does not trust the number should be
able to check it from the sentence, without opening the code::

    risk 0.083 = signal 0.142 x quality 0.643 x stability 0.910

That constraint does more than aid transparency. It makes a whole class of bug
impossible to hide: if the explanation ever disagreed with the stored score,
the arithmetic in the sentence would not close, and anyone reading it would
see that immediately.

Nothing is recomputed. Every number is read from the columns Stage 5 already
wrote, so an explanation cannot drift from the value it explains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.constants import ANOMALY_TYPES, R_HIGH, R_LOW
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Why a record has no risk score, in plain words.
UNDEFINED_PHRASES: Mapping[str, str] = {
    "severity_undefined": (
        "no severity could be computed for it, so there is nothing to weigh"
    ),
    "confidence_below_gate": (
        "its own evidence falls below the confidence gate, so any score would "
        "be a claim about data quality rather than about the work"
    ),
    "no_cluster_norm": (
        "its work type carries no peer norm, so there is no baseline to be "
        "risky against"
    ),
}

#: What each anomaly type contributes, in plain words.
TYPE_PHRASES: Mapping[str, str] = {
    "cost_outlier": "cost out of line with its peer group",
    "overspend_anomaly": "spending high against comparable works",
    "underspend_anomaly": "a completed work reporting unusually low spending",
    "temporal_outlier": "an unusual project duration",
    "duplicate_suspect": "a near-duplicate work nearby",
    "low_confidence": "weak underlying evidence",
    "insufficient_context": "no usable peer comparison",
}

#: Anomaly types that describe the EVIDENCE rather than the work. They are
#: reported but never presented as contributing to signal strength.
_EVIDENCE_TYPES = ("low_confidence", "insufficient_context")


def _types_of(row: Mapping[str, Any]) -> List[str]:
    """The anomaly types on a record, from whichever form is present.

    ``anomaly_types`` is the Stage 4 contract column and is always attached;
    the per-type booleans live only on the Stage 4 result frame. Reading the
    booleans alone silently produced an empty list on a real corpus, which made
    the narrative contradict its own arithmetic - it reported boosts from
    findings while claiming there were none.
    """
    listed = row.get("anomaly_types")
    if isinstance(listed, (list, tuple)):
        return [name for name in ANOMALY_TYPES if name in listed]
    return [name for name in ANOMALY_TYPES if bool(row.get(f"type_{name}", False))]


def _fmt(value: Any, digits: int = 3) -> Optional[str]:
    """Format a number, or return None when it is not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return f"{number:.{digits}f}"


def explain_risk(row: Mapping[str, Any]) -> str:
    """Explain one record's risk score, reconstructing it arithmetically.

    Args:
        row: A mapping of the Stage 5 output columns for one record, plus the
            Stage 4 context columns used for the narrative.

    Returns:
        A paragraph naming the band, showing the multiplication that produced
        the score, attributing each factor, and - where the score is absent -
        saying exactly why.
    """
    parts: List[str] = []

    defined = bool(row.get("risk_defined", False))
    score = _fmt(row.get("risk_score"))
    strength = _fmt(row.get("risk_signal_strength"))
    quality = _fmt(row.get("risk_data_quality"))
    uncertainty_value = row.get("risk_uncertainty")
    uncertainty = _fmt(uncertainty_value)
    stability = _fmt(
        1.0 - float(uncertainty_value)
        if isinstance(uncertainty_value, (int, float, np.floating))
        and np.isfinite(float(uncertainty_value))
        else np.nan
    )

    types = _types_of(row)
    findings = [TYPE_PHRASES[name] for name in types if name not in _EVIDENCE_TYPES]

    # --- undefined: say why, and refuse to imply safety --------------------
    if not defined or score is None:
        reason = str(row.get("risk_defined_reason", "severity_undefined"))
        parts.append(
            "No risk score: "
            + UNDEFINED_PHRASES.get(reason, reason)
            + ". This record is unassessed, not cleared - the absence of a "
            "score means nobody could measure it, not that it was found safe."
        )
        if findings:
            parts.append(
                "Signals were still observed and are retained: "
                + "; ".join(findings)
                + "."
            )
        confidence = _fmt(row.get("confidence"), digits=2)
        if reason == "confidence_below_gate" and confidence is not None:
            parts.append(
                f"Confidence is {confidence}. Repairing the record's own "
                "evidence is the prerequisite to scoring it at all."
            )
        return " ".join(parts)

    # --- defined: band, then the arithmetic --------------------------------
    band = str(row.get("risk_flag", "low_risk")).replace("_", " ")
    parts.append(
        f"Risk {score} ({band}), composed as "
        f"signal {strength} x data quality {quality} x stability {stability}."
    )

    # --- factor 1: what is wrong -------------------------------------------
    severity = _fmt(row.get("severity_score"))
    if findings:
        lead = f"Signal strength {strength} rests on " + "; ".join(findings)
    else:
        lead = (
            f"Signal strength {strength} rests on deviations that were measured "
            "but earned no named finding"
        )
    if severity is not None:
        lead += f", starting from a Stage 4 severity of {severity}"

    extras: List[str] = []
    extreme = float(row.get("risk_extreme", 0.0) or 0.0)
    if extreme >= 1.0:
        extras.append("an extreme-magnitude deviation")
    elif extreme > 0.0:
        extras.append("a high-magnitude deviation")
    # Breadth only counts as a reason when there is genuinely more than one
    # finding; a single anomaly type is already named above.
    if len(findings) > 1:
        extras.append(f"{len(findings)} distinct findings at once")
    if extras:
        lead += ", raised by " + " and ".join(extras)
    parts.append(lead + ".")

    # --- factor 2: whether it can be trusted -------------------------------
    confidence = _fmt(row.get("confidence"), digits=2)
    quality_bits: List[str] = []
    if confidence is not None:
        quality_bits.append(f"confidence {confidence}")
    deficit_factor = row.get("risk_deficit_factor")
    if deficit_factor is not None and _fmt(deficit_factor) is not None:
        if float(deficit_factor) < 0.999:
            quality_bits.append(
                f"critical fields missing (factor {_fmt(deficit_factor)})"
            )
    floor = row.get("risk_component_floor")
    if floor is not None and _fmt(floor) is not None and float(floor) < 0.999:
        quality_bits.append(f"weakest Stage 2 component {_fmt(floor)}")
    if bool(row.get("temporal_hard_fail", False)):
        quality_bits.append("an impossible date ordering")
    if quality_bits:
        parts.append(
            f"Data quality {quality} follows from " + ", ".join(quality_bits) + "."
        )

    # --- factor 3: how stable --------------------------------------------
    unstable: List[str] = []
    if not bool(row.get("peer_cell_stable", True)):
        unstable.append("its peer cell is too small to rely on")
    coverage = row.get("valid_signal_count")
    if coverage is not None and _fmt(coverage) is not None and int(coverage) < 3:
        unstable.append(
            f"only {int(coverage)} of 3 possible peer comparisons were available"
        )
    if unstable:
        parts.append(
            f"Uncertainty {uncertainty} because " + " and ".join(unstable) + "."
        )
    else:
        parts.append(
            f"Uncertainty {uncertainty}: the comparison rests on the full set "
            "of peer signals."
        )

    # --- why the score is lower than the signal ----------------------------
    try:
        strength_value = float(row.get("risk_signal_strength"))
        score_value = float(row.get("risk_score"))
        if np.isfinite(strength_value) and strength_value > 0 and score_value < strength_value:
            shrink = 100.0 * (1.0 - score_value / strength_value)
            parts.append(
                f"The score sits {shrink:.0f}% below the raw signal because risk "
                "is conditional on evidence: a finding is only as actionable as "
                "the record that carries it."
            )
    except (TypeError, ValueError):
        pass

    # --- duplicate stays secondary ----------------------------------------
    if bool(row.get("duplicate_flag", False)):
        parts.append(
            "A near-duplicate was found; it contributes at most a tenth of the "
            "signal and never carries a case by itself."
        )

    parts.append("This is an estimate of risk under uncertainty, not a finding of fraud.")
    return " ".join(parts)


def build_risk_explanations(frame: pd.DataFrame) -> pd.Series:
    """Generate the risk explanation for every record.

    Args:
        frame: The assembled Stage 5 frame joined with its Stage 2-4 context.

    Returns:
        Object Series aligned to ``frame.index``.
    """
    if len(frame) == 0:
        return pd.Series([], dtype="object", index=frame.index, name="risk_explanation")
    columns = list(frame.columns)
    texts = [
        explain_risk(dict(zip(columns, values)))
        for values in frame.itertuples(index=False, name=None)
    ]
    LOGGER.info("Generated %d risk explanation(s).", len(texts))
    return pd.Series(texts, index=frame.index, dtype="object", name="risk_explanation")
