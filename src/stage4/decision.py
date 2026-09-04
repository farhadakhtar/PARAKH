"""Severity composition and confidence-gated routing.

Severity is a *summary*, not a verdict
--------------------------------------
It is a weighted mean of the valid signals, renormalised per record so a
missing signal neither penalises nor credits the record. It never overrides the
anomaly types, and it is ``NaN`` - never 0 - when nothing was measurable,
because a record with no usable signal has an *unknown* severity, not a low one.

Confidence gates the decision, never the value
----------------------------------------------
A low-confidence record keeps its deviations and its severity at full
magnitude. What it cannot do is escalate. README sec.8: "The system never emits
a fraud hypothesis on low-confidence evidence." The routing enforces that as a
precedence rule rather than a weighting, so no combination of large deviations
can add up to an accusation on evidence that cannot be defended.

The threshold matches ``PEER_STAT_MIN_CONFIDENCE`` deliberately: a record that
was not trusted to *shape* a peer norm is not trusted to be *accused* by one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    CLUSTER_NOISE_REASON,
    CONFIDENCE_GATE_THRESHOLD,
    DECISION_CLASSES,
    FEATURE_MISSING_REASON,
    PEER_NORM_ABSENT_REASONS,
    SEVERITY_DEFINED_REASONS,
    SEVERITY_WEIGHTS,
    STAGE4_VERSION,
    Z_INVESTIGATE_THRESHOLD,
    Z_SEVERITY_SCALE,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SeverityResult:
    """Per-record severity and the contribution of each signal to it."""

    score: pd.Series
    components: pd.DataFrame
    weights: Mapping[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        defined = self.score.notna()
        return {
            "weights": dict(self.weights),
            "defined_pct": round(100.0 * float(defined.mean()), 4)
            if len(self.score)
            else 0.0,
            "median": round(float(self.score[defined].median()), 6)
            if defined.any()
            else None,
            "p95": round(float(self.score[defined].quantile(0.95)), 6)
            if defined.any()
            else None,
            "max": round(float(self.score[defined].max()), 6)
            if defined.any()
            else None,
            **self.diagnostics,
        }


def _bounded(values: np.ndarray, scale: float) -> np.ndarray:
    """Map |z| into [0,1], preserving order and leaving NaN as NaN.

    Bounded so one enormous deviation cannot swamp the mean, monotone so
    ordering within the tail survives. The raw ``z`` is reported alongside and
    is never modified - this transform exists only for composition.
    """
    with np.errstate(invalid="ignore"):
        return np.minimum(np.abs(values) / float(scale), 1.0)


def compute_severity(
    signals: pd.DataFrame,
    weights: Mapping[str, float] = SEVERITY_WEIGHTS,
    scale: float = Z_SEVERITY_SCALE,
) -> SeverityResult:
    """Compose a bounded severity from the valid signals only.

    Args:
        signals: Output of :func:`~src.stage4.anomaly.build_signals`.
        weights: Per-signal weights, renormalised over the valid signals of
            each record.
        scale: |z| at which a signal contributes its full weight.

    Returns:
        A :class:`SeverityResult`. ``score`` is ``NaN`` for any record with no
        valid CORE signal - deliberately, so that "nothing could be measured"
        is never reported as "nothing was wrong", and so the duplicate signal
        can never be the sole basis for a severity.
    """
    index = signals.index
    contributions = pd.DataFrame(index=index)
    contributions["cost"] = _bounded(signals["z_cost"].to_numpy(dtype="float64"), scale)
    contributions["spend"] = _bounded(
        signals["z_spend"].to_numpy(dtype="float64"), scale
    )
    contributions["duration"] = _bounded(
        signals["z_duration"].to_numpy(dtype="float64"), scale
    )
    values_core = contributions[["cost", "spend", "duration"]].to_numpy(
        dtype="float64"
    )

    # The duplicate score is already in [0,1] and is only counted when the
    # detector actually flagged the record; an unflagged score is background
    # similarity, not evidence.
    #
    # It is ALSO withheld when no core deviation is valid. The brief is
    # explicit that the duplicate signal is never a primary anomaly, and a
    # record whose severity came from the duplicate alone would have exactly
    # that: a weak supporting signal driving the whole number. Such a record
    # keeps its duplicate_suspect type and its duplicate_score - it simply
    # has no severity, which is the honest answer.
    has_core = np.isfinite(values_core).any(axis=1)
    duplicate = signals["duplicate_score"].to_numpy(dtype="float64", na_value=np.nan)
    flagged = signals["duplicate_flag"].to_numpy(dtype=bool)
    contributions["duplicate"] = np.where(flagged & has_core, duplicate, np.nan)

    weight_row = np.asarray(
        [float(weights.get(name, 0.0)) for name in contributions.columns],
        dtype="float64",
    )
    values = contributions.to_numpy(dtype="float64")
    valid = np.isfinite(values)

    effective = valid * weight_row[None, :]
    total = effective.sum(axis=1)
    no_signal = total <= 0.0

    weighted = np.where(valid, values, 0.0) * effective
    score = np.divide(
        weighted.sum(axis=1),
        np.where(no_signal, 1.0, total),
        out=np.zeros(len(index), dtype="float64"),
        where=~no_signal,
    )
    # NaN, not 0: an unmeasurable record has unknown severity, not low severity.
    score = np.where(no_signal, np.nan, score)

    assert not np.isinf(score).any(), "severity produced an infinite value"
    finite = np.isfinite(score)
    assert bool(((score[finite] >= 0.0) & (score[finite] <= 1.0)).all()), (
        "severity escaped [0,1]"
    )

    LOGGER.info(
        "Severity composed for %d record(s); %d undefined (no valid signal).",
        len(index),
        int(no_signal.sum()),
    )

    return SeverityResult(
        score=pd.Series(score, index=index, dtype="float64", name="severity_score"),
        components=contributions,
        weights=dict(weights),
        diagnostics={"n_undefined": int(no_signal.sum())},
    )


@dataclass(frozen=True)
class SeverityDefinedness:
    """Whether each record has a severity, and - when it does not - why.

    Undefined severity was already the behaviour; what was missing was the
    record saying so. A consumer reading only ``severity_score`` sees a NaN and
    must guess whether the record was unmeasurable, noise, or simply absent
    from the peer structure. Those have different owners and different fixes.
    """

    defined: pd.Series
    reason: pd.Series
    #: Records where the stated rule and the computed severity disagreed. Must
    #: be 0 on any frame produced by Stage 3; see :func:`severity_definedness`.
    rule_divergence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "defined_pct": round(100.0 * float(self.defined.mean()), 4)
            if len(self.defined)
            else 0.0,
            "by_reason": {
                str(k): int(v) for k, v in self.reason.value_counts().items()
            },
            "rule_divergence": self.rule_divergence,
        }


def severity_definedness(
    frame: pd.DataFrame, signals: pd.DataFrame, score: pd.Series
) -> SeverityDefinedness:
    """Explain the definedness of an already-computed severity.

    This function is **descriptive**. ``defined`` is read off the severity that
    :func:`compute_severity` already produced, so adding it cannot change an
    existing output and the invariant "not defined implies NaN" holds by
    construction rather than by enforcement.

    The stated rule - a severity exists only where a core deviation is valid
    *and* the work type has a peer norm - is *checked* against that, not applied
    to it. Applying it would mean blanking a severity that had already been
    computed, which would change an existing output. Where the two disagree the
    divergence is counted, logged and reported, never silently resolved.

    The norm conjunct is structurally redundant on any frame Stage 3 produces:
    ``cluster_has_norm`` is False only when no cluster-wide median exists for
    any metric, and a peer cell is a subset of its cluster, so no cell deviation
    can be defined either. It is checked anyway - relying on a coincidence
    between two modules is how invariants rot - and a non-zero divergence means
    that structural relationship has broken upstream.

    Args:
        frame: Corpus frame with Stage 3 reasons and ``cluster_has_norm``.
        signals: Output of :func:`~src.stage4.anomaly.build_signals`.
        score: The severity produced by :func:`compute_severity`.

    Returns:
        A :class:`SeverityDefinedness` aligned to ``frame.index``.
    """
    index = frame.index
    n_records = len(index)
    defined = score.notna().to_numpy()

    has_norm = frame["cluster_has_norm"].fillna(False).to_numpy(dtype=bool)
    has_core = signals["valid_signal_count"].to_numpy() > 0
    expected = has_core & has_norm

    divergent = int((defined != expected).sum())
    if divergent:
        LOGGER.warning(
            "severity definedness diverges from the stated rule on %d record(s): "
            "a severity exists where the work type reports no peer norm. Stage 3 "
            "cannot produce this, so the input is synthetic or upstream has "
            "changed. The computed severity is left exactly as it was; the "
            "divergence is reported rather than resolved.",
            divergent,
        )

    # One reason per record, in precedence order. Noise outranks the rest: a
    # record with no work type has no peer structure to be missing from.
    reason = np.full(n_records, "no_valid_deviation", dtype=object)

    reason_columns = [
        f"deviation_{name}_reason"
        for name in ("cell_cost", "cluster_cost", "spend_ratio", "duration")
        if f"deviation_{name}_reason" in frame.columns
    ]
    if reason_columns:
        values = frame[reason_columns].astype("object").to_numpy()
        undefined_cell = values != "defined"
        # "every deviation this record lacks, it lacks for the same cause"
        all_missing = (
            (values == FEATURE_MISSING_REASON) | ~undefined_cell
        ).all(axis=1) & undefined_cell.any(axis=1)
        any_no_norm = np.isin(values, PEER_NORM_ABSENT_REASONS).any(axis=1)
        reason[any_no_norm] = "no_peer_norm"
        reason[all_missing] = "insufficient_features"

    reason[~has_norm] = CLUSTER_NOISE_REASON
    reason[defined] = "ok"

    unknown = set(np.unique(reason)) - set(SEVERITY_DEFINED_REASONS)
    assert not unknown, f"undeclared severity reason(s): {sorted(unknown)}"

    LOGGER.info(
        "Severity definedness: %d of %d record(s) defined; %s",
        int(defined.sum()),
        n_records,
        {k: int(v) for k, v in pd.Series(reason[~defined]).value_counts().items()},
    )

    return SeverityDefinedness(
        defined=pd.Series(defined, index=index, dtype=bool, name="severity_defined"),
        reason=pd.Series(
            reason, index=index, dtype="object", name="severity_defined_reason"
        ),
        rule_divergence=divergent,
    )


@dataclass(frozen=True)
class DecisionResult:
    """Per-record provisional triage class and the rule that produced it."""

    decision_class: pd.Series
    #: Which precedence rule fired, for audit.
    decision_reason: pd.Series
    confidence_flag: pd.Series
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "decision_class": self.decision_class.value_counts().to_dict(),
            "decision_reason": self.decision_reason.value_counts().to_dict(),
            "low_confidence_pct": round(
                100.0 * float((self.confidence_flag == "low").mean()), 4
            )
            if len(self.confidence_flag)
            else 0.0,
            **self.diagnostics,
        }


def route(
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    types: pd.DataFrame,
    confidence_threshold: float = CONFIDENCE_GATE_THRESHOLD,
    investigate_threshold: float = Z_INVESTIGATE_THRESHOLD,
) -> DecisionResult:
    """Route each record, confidence first.

    Precedence, highest first. It is a precedence chain rather than a score so
    that no accumulation of deviations can ever outvote the confidence gate:

    1. ``confidence < threshold`` -> **REMEDIATE**. The evidence must be fixed
       before anything can be concluded from it. This fires regardless of how
       extreme the deviations are, which is the entire point.
    2. no valid signal -> **INSUFFICIENT_CONTEXT**. Nothing was measurable, and
       saying so is more honest than calling it clear.
    3. any |z| at or above ``investigate_threshold`` -> **INVESTIGATE**.
       Reachable only with sufficient confidence, by construction of rule 1.
    4. otherwise -> **MONITOR**.

    Args:
        frame: Corpus frame with Stage 2 and Stage 3 outputs.
        signals: Output of :func:`~src.stage4.anomaly.build_signals`.
        types: Output of :func:`~src.stage4.anomaly.classify_types`.
        confidence_threshold: The gate.
        investigate_threshold: |z| required to escalate.

    Returns:
        A :class:`DecisionResult` aligned to ``frame.index``.
    """
    index = frame.index
    n_records = len(index)

    confidence = frame["confidence"].to_numpy(dtype="float64", na_value=0.0)
    low_confidence = confidence < float(confidence_threshold)
    no_signal = signals["valid_signal_count"].to_numpy() == 0

    with np.errstate(invalid="ignore"):
        escalating = np.zeros(n_records, dtype=bool)
        for name in ("z_cost", "z_spend", "z_duration"):
            values = signals[name].to_numpy(dtype="float64")
            escalating |= np.abs(values) >= float(investigate_threshold)

    decision = np.full(n_records, "MONITOR", dtype=object)
    reason = np.full(n_records, "no_escalating_deviation", dtype=object)

    decision[escalating] = "INVESTIGATE"
    reason[escalating] = "deviation_at_or_above_investigate_threshold"

    decision[no_signal] = "INSUFFICIENT_CONTEXT"
    reason[no_signal] = "no_valid_signal"

    # Applied last so it wins outright: a low-confidence record can never be
    # escalated, whatever its deviations look like.
    decision[low_confidence] = "REMEDIATE"
    reason[low_confidence] = "confidence_below_gate"

    assert set(np.unique(decision)) <= set(DECISION_CLASSES)
    escalated = decision == "INVESTIGATE"
    assert not bool((escalated & low_confidence).any()), (
        "a low-confidence record was escalated to INVESTIGATE"
    )

    LOGGER.info(
        "Routing over %d record(s): %s",
        n_records,
        {k: int(v) for k, v in pd.Series(decision).value_counts().items()},
    )

    return DecisionResult(
        decision_class=pd.Series(decision, index=index, dtype="object", name="decision_class"),
        decision_reason=pd.Series(reason, index=index, dtype="object", name="decision_reason"),
        confidence_flag=pd.Series(
            np.where(low_confidence, "low", "high"),
            index=index,
            dtype="object",
            name="confidence_flag",
        ),
        diagnostics={
            "stage4_version": STAGE4_VERSION,
            "confidence_threshold": float(confidence_threshold),
            "investigate_threshold": float(investigate_threshold),
            "n_escalated": int(escalated.sum()),
        },
    )
