"""The transparency layer: seven measured risks, made impossible to miss.

Every field here is **derived** from a decision already made. None can change
one. The read-only audit measured seven ways this system can be misread, and
this module answers each of them by annotation rather than correction:

===  ============================================================  ===========
R1   18 of 419 escalations carry no named finding, 4 at P0         reason_flag
R2   ``action_spec`` merges 291 P0 referrals with 128 P1 reviews    truth_class
R3   P1 mixes 3,402 data-quality with 128 audit escalations         semantic_type
R4   the "never escalate unscored" guarantee depends on two         metadata
     thresholds happening to be equal
R5   200 records share a ``work_id``; 56 groups get different       work summary
     actions
R6   a reviewer cannot tell "no issue" from "cannot assess"         clarity_flag
R7   0.5 reads as a probability and is not one                      calibration
===  ============================================================  ===========

The rule that makes this safe: an annotation may **only** restate or qualify
what upstream already decided. If a field here could ever change a routing
outcome, it would belong in Stage 6, not in a transparency layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ACTION_CLASSES,
    ACTION_GROUPS,
    ACTION_SPEC_LOSSY_NOTE,
    ACTION_TO_GROUP,
    ACTION_TO_SEMANTIC_TYPE,
    ESCALATION_REASON_STATUSES,
    ESCALATION_UNEXPLAINED_WARNING,
    ACTION_SPEC_LOSSY_WARNING,
    CALIBRATION_WARNING,
    CONFIDENCE_GATE_THRESHOLD,
    ESCALATING_ACTIONS,
    CONFIG_DEPENDENCY_WARNING,
    DECISION_CLARITY_FLAGS,
    EXPLAINED_REASON_DETAIL,
    M1_CORRECTION_LABEL,
    MIN_CONFIDENCE_FOR_RISK,
    PRIORITY_EXECUTION,
    PRIORITY_LEVELS,
    PRIORITY_SEMANTIC_TYPES,
    R_HIGH,
    R_LOW,
    REASON_FLAGS,
    SPEC_ACTION_ALIAS,
    STAGE7_VERSION,
    UNEXPLAINED_REASON_DETAIL,
    WORK_ID_AMBIGUITY_WARNING,
    Z_INVESTIGATE_THRESHOLD,
    Z_TYPE_THRESHOLD,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: The annotation columns, in a fixed order. All derived, all deterministic.
ANNOTATION_COLUMNS: Tuple[str, ...] = (
    "stage7_reason_flag",
    "stage7_reason_detail",
    "action_truth_class",
    "action_interpretation_warning",
    "priority_semantic_type",
    "decision_clarity_flag",
    "calibration_warning",
    "stage7_explanation",
    # --- added by the surgical correction pass ---------------------------
    # Named by the correction specification. Each duplicates a fact an
    # existing column already carries, in the shape the specification asks
    # for; the pairs are asserted equal on every run rather than assumed.
    "escalation_reason_status",   # R1, scoped to escalations
    "escalation_reason_warning",  # R1, names the Stage 4 root cause
    "action_group",               # R3, lowercase partition
    "action_spec_lossy",          # R4, boolean rather than a message
)

#: Actions whose spec alias loses information. Computed from the alias table
#: rather than hard-coded, so a future one-to-one mapping stops warning by
#: itself instead of lying about a loss that no longer exists.
_LOSSY_ACTIONS: Tuple[str, ...] = tuple(
    sorted(
        action
        for action, alias in SPEC_ACTION_ALIAS.items()
        if sum(1 for other in SPEC_ACTION_ALIAS.values() if other == alias) > 1
    )
)


def _is_unexplained(findings: Any) -> bool:
    """Whether the record's only basis is an unnamed deviation."""
    return isinstance(findings, (list, tuple)) and M1_CORRECTION_LABEL in findings


def build_annotations(payloads: pd.Series) -> pd.DataFrame:
    """Derive every annotation from the payloads alone.

    Reads only ``explanation_payload``, which Stage 6 emits as canonical JSON
    and which Stage 7 treats as the sole source of truth. The human
    explanation string is never consulted here - it is display only, and the
    Stage 6 audit proved its delimiters are ambiguous.

    Args:
        payloads: Decoded Stage 6 payloads, from
            :func:`~src.stage7.interface.decode_payloads`.

    Returns:
        A frame of :data:`ANNOTATION_COLUMNS`, aligned to ``payloads.index``.
    """
    index = payloads.index
    rows: List[Dict[str, Any]] = []

    for payload in payloads:
        action = str(payload["action"])
        priority = str(payload["priority"])
        findings = list(payload.get("findings") or [])
        risk = payload.get("risk_score")
        risk_flag = str(payload.get("risk_flag", ""))
        decision = str(payload.get("decision_class", ""))

        unexplained = _is_unexplained(findings)
        unscored = risk is None
        is_escalation = action in ESCALATING_ACTIONS

        # R1 - is there a named basis for this record's treatment?
        reason_flag = "UNEXPLAINED_DEVIATION" if unexplained else "EXPLAINED"

        # R6 - can a reviewer act on this as presented? The two failure modes
        # are mutually exclusive by construction: `unexplained_deviation` is
        # only ever added to an escalated record, and every escalated record
        # is scored, so it can never also be data-limited.
        if unexplained:
            clarity = "AMBIGUOUS"
        elif unscored:
            clarity = "DATA_LIMITED"
        else:
            clarity = "CLEAR"

        rows.append(
            {
                "stage7_reason_flag": reason_flag,
                "stage7_reason_detail": (
                    UNEXPLAINED_REASON_DETAIL
                    if unexplained
                    else EXPLAINED_REASON_DETAIL
                ),
                # R2 - the canonical, non-lossy action.
                "action_truth_class": action,
                "action_interpretation_warning": (
                    ACTION_SPEC_LOSSY_WARNING if action in _LOSSY_ACTIONS else None
                ),
                # R3 - what this priority actually means.
                "priority_semantic_type": ACTION_TO_SEMANTIC_TYPE[action],
                "decision_clarity_flag": clarity,
                # R7 - attached to every record, not just the report.
                "calibration_warning": CALIBRATION_WARNING,
                # R1 - scoped to escalations. A monitored record with no
                # findings is correct, not unexplained, so it reads
                # "explained" rather than implying something is missing.
                "escalation_reason_status": (
                    "unexplained_upstream"
                    if (is_escalation and unexplained)
                    else "explained"
                ),
                "escalation_reason_warning": (
                    ESCALATION_UNEXPLAINED_WARNING
                    if (is_escalation and unexplained)
                    else None
                ),
                # R3 - the same partition as priority_semantic_type, in the
                # case the specification asked for.
                "action_group": ACTION_TO_GROUP[action],
                # R4 - boolean, so a consumer can filter rather than parse.
                "action_spec_lossy": action in _LOSSY_ACTIONS,
                "stage7_explanation": build_stage7_explanation(
                    action=action,
                    priority=priority,
                    findings=findings,
                    risk=risk,
                    risk_flag=risk_flag,
                    decision=decision,
                    unexplained=unexplained,
                ),
            }
        )

    frame = pd.DataFrame(rows, index=index, columns=list(ANNOTATION_COLUMNS))
    LOGGER.info(
        "Annotated %d record(s): %s",
        len(frame),
        frame["decision_clarity_flag"].value_counts().to_dict(),
    )
    return frame


def build_stage7_explanation(
    action: str,
    priority: str,
    findings: Sequence[str],
    risk: Optional[float],
    risk_flag: str,
    decision: str,
    unexplained: bool,
) -> str:
    """Write the sentence a reviewer should read first.

    Clearer than Stage 6's in three specific ways, each answering a measured
    complaint:

    * **Escalation versus review.** Stage 6 says ``ESCALATE_IMMEDIATE`` and
      ``ESCALATE_REVIEW``; this says what each *asks a person to do*, and how
      soon.
    * **Data problem versus real anomaly.** An unscored record is described as
      unassessable, never as clean. That distinction is the system's central
      claim and it must survive to the reader.
    * **Measured versus undefined.** A risk of 0.0 and no risk at all are
      rendered differently, and a present score is qualified as uncalibrated.

    It cannot contradict the data because every input is a payload field and
    nothing is recomputed.

    Args:
        action: The canonical action class.
        priority: The priority level.
        findings: Named anomaly types, possibly including the M1 label.
        risk: The risk score, or None where unscored.
        risk_flag: Stage 5's band.
        decision: Stage 4's decision class.
        unexplained: Whether the only basis is an unnamed deviation.

    Returns:
        A short paragraph, safe to show to a non-technical reviewer.
    """
    parts: List[str] = []
    sla = PRIORITY_EXECUTION.get(priority, {}).get("sla_hours")

    # --- what is being asked, and how urgently ----------------------------
    if action == "ESCALATE_IMMEDIATE":
        parts.append(
            f"ACT NOW ({priority}): assign this record to an investigator "
            f"within {sla} hours. The system considers it the strongest class "
            "of lead it produces."
        )
    elif action == "ESCALATE_REVIEW":
        parts.append(
            f"REVIEW ({priority}): a person should validate this record within "
            f"{sla} hours. This is a request for human judgement, not an "
            "urgent fraud referral."
        )
    elif action == "REQUEST_CORRECTION":
        parts.append(
            f"CORRECT THE RECORD ({priority}): the problem here is the data, "
            "not established wrongdoing. Send it back to the office that filed "
            "it. Nothing about the work itself has been concluded."
        )
    elif action == "DATA_QUALITY_REVIEW":
        parts.append(
            f"CANNOT ASSESS ({priority}): this record could not be evaluated. "
            "It is NOT a clean record - it is an unassessed one, and treating "
            "it as cleared would be a mistake."
        )
    else:
        parts.append(
            f"MONITOR ONLY ({priority}): nothing about this record stood out "
            "against comparable works. No one is assigned and no clock runs."
        )

    # --- what was found ---------------------------------------------------
    if unexplained:
        parts.append(
            "No named anomaly category was assigned. The escalation rests on a "
            "statistical deviation that upstream declined to label, so a "
            "reviewer must characterise it manually before acting on it."
        )
    elif findings:
        parts.append("Findings: " + ", ".join(str(item) for item in findings) + ".")
    else:
        parts.append("No anomaly category was recorded for this record.")

    # --- what the number does and does not mean ---------------------------
    if risk is None:
        parts.append(
            "There is no risk score. This means the record could not be "
            "measured, NOT that it was measured and found safe."
        )
    else:
        parts.append(
            f"Risk {risk:.3f} ({risk_flag}) on an uncalibrated scale - a "
            "position in an ordering, not a probability."
        )

    parts.append(f"Upstream decision: {decision}.")
    return " ".join(parts)


def build_system_metadata(n_records: int) -> Dict[str, Any]:
    """The configuration these outputs depend on (R4).

    Returned with every result rather than logged, because the fragility it
    documents is invisible in the records themselves: two runs can differ only
    in configuration and produce entirely different queues.
    """
    return {
        "stage7_version": STAGE7_VERSION,
        "n_records": n_records,
        "_warning": CONFIG_DEPENDENCY_WARNING,
        "_calibration": CALIBRATION_WARNING,
        "thresholds": {
            "min_confidence_stage4_gate": CONFIDENCE_GATE_THRESHOLD,
            "min_confidence_stage5_gate": MIN_CONFIDENCE_FOR_RISK,
            "gates_aligned": CONFIDENCE_GATE_THRESHOLD == MIN_CONFIDENCE_FOR_RISK,
            "r_high": R_HIGH,
            "r_low": R_LOW,
            "z_type_threshold": Z_TYPE_THRESHOLD,
            "z_investigate_threshold": Z_INVESTIGATE_THRESHOLD,
        },
        "_gate_note": (
            "gates_aligned must be true. The guarantee that an unscored record "
            "is never escalated holds only while the Stage 4 and Stage 5 "
            "confidence gates are equal; a configured divergence breaks it "
            "without changing any constant."
        ),
        "priority_execution": {
            level: dict(PRIORITY_EXECUTION[level]) for level in PRIORITY_LEVELS
        },
    }


def build_transparency_metrics(
    annotations: pd.DataFrame, payloads: pd.Series
) -> Dict[str, Any]:
    """The limitations, as percentages (R1, R3, R6, R7).

    Published rather than derivable: a consumer should not have to compute a
    system's weaknesses for themselves.
    """
    total = len(annotations)
    if total == 0:
        return {"n_records": 0, "_note": "no records"}

    unexplained = (annotations["stage7_reason_flag"] == "UNEXPLAINED_DEVIATION")
    escalating = annotations["priority_semantic_type"] == "ESCALATION"
    unscored = annotations["decision_clarity_flag"] == "DATA_LIMITED"

    def _pct(mask: pd.Series, denominator: int) -> float:
        return round(100.0 * int(mask.sum()) / denominator, 4) if denominator else 0.0

    n_escalations = int(escalating.sum())
    return {
        "n_records": total,
        "_note": (
            "These are the system's known limitations, stated as measured "
            "rates. They are published so that nobody has to discover them."
        ),
        "pct_insufficient_data": _pct(unscored, total),
        "pct_unexplained_deviation": _pct(unexplained, total),
        "pct_escalations_without_named_anomaly": _pct(
            unexplained & escalating, n_escalations
        ),
        "n_escalations_without_named_anomaly": int((unexplained & escalating).sum()),
        "n_escalations": n_escalations,
        "by_priority_semantic_type": {
            name: int((annotations["priority_semantic_type"] == name).sum())
            for name in PRIORITY_SEMANTIC_TYPES
        },
        "pct_by_priority_semantic_type": {
            name: _pct(annotations["priority_semantic_type"] == name, total)
            for name in PRIORITY_SEMANTIC_TYPES
        },
        "by_decision_clarity": {
            name: int((annotations["decision_clarity_flag"] == name).sum())
            for name in DECISION_CLARITY_FLAGS
        },
        "n_lossy_action_alias": int(
            annotations["action_interpretation_warning"].notna().sum()
        ),
        # R1/R3/R4 in the specified shapes.
        "n_unexplained_upstream": int(
            (annotations["escalation_reason_status"] == "unexplained_upstream").sum()
        ),
        "by_action_group": {
            name: int((annotations["action_group"] == name).sum())
            for name in ACTION_GROUPS
        },
        "n_action_spec_lossy": int(annotations["action_spec_lossy"].sum()),
        "_action_spec_note": ACTION_SPEC_LOSSY_NOTE,
    }


def build_work_level_summary(
    frame: pd.DataFrame, payloads: pd.Series, annotations: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate by ``work_id``, worst case first (R5).

    ``work_id`` is a business key that repeats: 200 records share one across
    100 groups on the reference corpus, and **56 of those groups receive
    different actions**. A consumer joining on it silently fans out.

    Risk is aggregated by **maximum**, never by mean. Averaging two records of
    a duplicated work would invent a number that describes neither of them,
    and would let a high-risk record be diluted by a clean twin. The full
    distribution is carried alongside so the maximum can never stand alone.

    Args:
        frame: The corpus frame, read only.
        payloads: Decoded payloads.
        annotations: Output of :func:`build_annotations`.

    Returns:
        One row per distinct ``work_id``, or an empty frame when the column is
        absent.
    """
    if "work_id" not in frame.columns:
        LOGGER.warning("no work_id column; work-level summary is empty")
        return pd.DataFrame(
            columns=[
                "work_id", "total_records", "max_risk_score", "escalation_count",
                "dominant_action_class", "action_distribution",
                "anomaly_distribution", "has_conflicting_actions",
            ]
        )

    working = pd.DataFrame(
        {
            "work_id": frame["work_id"].astype("object").to_numpy(),
            "action": [str(p["action"]) for p in payloads],
            "risk": [p.get("risk_score") for p in payloads],
            "findings": [list(p.get("findings") or []) for p in payloads],
            "semantic": annotations["priority_semantic_type"].to_numpy(),
        },
        index=frame.index,
    )

    rows: List[Dict[str, Any]] = []
    for work_id, group in working.groupby("work_id", sort=True):
        scored = [value for value in group["risk"] if value is not None]
        counts = group["action"].value_counts()
        anomalies: Dict[str, int] = {}
        for findings in group["findings"]:
            for name in findings:
                anomalies[name] = anomalies.get(name, 0) + 1
        rows.append(
            {
                "work_id": work_id,
                "total_records": int(len(group)),
                # Maximum, never mean. See the docstring.
                "max_risk_score": max(scored) if scored else None,
                "n_records_scored": len(scored),
                "escalation_count": int((group["semantic"] == "ESCALATION").sum()),
                # Ties broken by name so the result is deterministic.
                "dominant_action_class": sorted(
                    counts.index, key=lambda name: (-int(counts[name]), name)
                )[0],
                "action_distribution": {
                    str(k): int(v) for k, v in sorted(counts.items())
                },
                "anomaly_distribution": dict(sorted(anomalies.items())),
                # The fact a consumer most needs and is least likely to check.
                "has_conflicting_actions": bool(counts.size > 1),
            }
        )

    summary = pd.DataFrame(rows)
    conflicting = int(summary["has_conflicting_actions"].sum()) if len(summary) else 0
    if conflicting:
        LOGGER.warning(
            "%d work_id group(s) contain records routed to DIFFERENT actions. "
            "%s",
            conflicting,
            WORK_ID_AMBIGUITY_WARNING,
        )
    return summary
