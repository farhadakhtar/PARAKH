"""Signal validation, core anomaly signals, and type classification.

Stage 4 recomputes nothing. It reads Stage 3's deviations and Stage 2's
confidence breakdown and decides what they *mean*.

The rule that governs every line here
-------------------------------------
**Undefined is not normal, and missing is not safe.**

Stage 3 emits a deviation only when it means something, and records a reason
whenever it does not. Stage 4 must honour that: an absent deviation is an
absence of evidence, never evidence of absence. Substituting zero would say
"this record sits exactly at its peer median", which is the single most
dangerous thing this system could assert about a record it could not measure.

That is why ``valid_signal_count`` exists, why ``severity_score`` is ``NaN``
rather than 0 when nothing is measurable, and why ``insufficient_context`` is a
first-class finding rather than a quiet gap.

Confidence controls interpretation, not value
---------------------------------------------
A low-confidence record keeps its deviations at full magnitude. What changes is
what those deviations are permitted to *mean*: it can be routed to REMEDIATE,
never escalated to INVESTIGATE. Suppressing the value instead would destroy the
information a remediator needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ANOMALY_TYPES,
    CONFIDENCE_GATE_THRESHOLD,
    CORE_SIGNALS,
    COST_SCOPES,
    LIFECYCLE_PRE_COMPLETION_STATES,
    LIFECYCLE_TERMINAL_STATES,
    STAGE4_VERSION,
    Z_TYPE_THRESHOLD,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Stage 3 columns Stage 4 depends on. Absence is an error, never a default.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "confidence",
    "lifecycle_state",
    "peer_cell_stable",
    "cluster_has_norm",
    "deviation_cell_cost",
    "deviation_cell_cost_reason",
    "deviation_cluster_cost",
    "deviation_cluster_cost_reason",
    "deviation_spend_ratio",
    "deviation_spend_ratio_reason",
    "deviation_duration",
    "deviation_duration_reason",
    "duplicate_score",
    "duplicate_flag",
)

#: The reason Stage 3 records when a deviation IS usable.
DEFINED_REASON: str = "defined"


class Stage4InputError(ValueError):
    """Raised when the Stage 2/3 contract is not satisfied."""


@dataclass(frozen=True)
class SignalValidation:
    """Which deviation signals are usable, and how many per record."""

    usable: pd.DataFrame
    valid_signal_count: pd.Series
    #: Rows where the value and its reason disagreed with each other.
    contract_violations: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "usable_pct": {
                name: round(100.0 * float(self.usable[name].mean()), 4)
                for name in self.usable.columns
            },
            "valid_signal_count": self.valid_signal_count.value_counts()
            .sort_index()
            .to_dict(),
            "contract_violations": self.contract_violations,
            **self.diagnostics,
        }


def require_contract(frame: pd.DataFrame) -> None:
    """Fail loudly when the upstream contract is incomplete.

    Args:
        frame: The corpus frame Stage 4 is asked to interpret.

    Raises:
        Stage4InputError: If any required column is absent. Stage 4 must not
            invent a default for a signal it was never given - that is exactly
            the fabrication the philosophy forbids.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise Stage4InputError(
            f"Stage 4 requires Stage 2 and Stage 3 outputs; missing {missing!r}. "
            "Run attach_confidence(corpus) then attach_structure(corpus) first."
        )


def validate_signals(frame: pd.DataFrame) -> SignalValidation:
    """Decide which deviation signals may be used, per record.

    A signal is usable only when its value is finite **and** Stage 3 recorded
    the reason ``defined``. Both are checked rather than one trusted: if they
    ever disagree the signal is dropped and the disagreement counted, because a
    contract violation upstream must not become a silent anomaly downstream.

    Args:
        frame: Corpus frame carrying Stage 3's deviations and reasons.

    Returns:
        A :class:`SignalValidation`. ``valid_signal_count`` counts only the
        deviation-derived signals - the duplicate score is supporting evidence
        and is excluded, so a record with no peer comparison cannot use it to
        escape the ``insufficient_context`` finding.
    """
    require_contract(frame)
    index = frame.index
    usable = pd.DataFrame(index=index)
    violations = 0
    per_reason: Dict[str, Dict[str, int]] = {}

    for name in (
        "deviation_cell_cost",
        "deviation_cluster_cost",
        "deviation_spend_ratio",
        "deviation_duration",
    ):
        value_ok = np.isfinite(frame[name].to_numpy(dtype="float64", na_value=np.nan))
        reason_ok = (frame[f"{name}_reason"].astype("object") == DEFINED_REASON).to_numpy()
        disagreement = int((value_ok != reason_ok).sum())
        if disagreement:
            violations += disagreement
            LOGGER.warning(
                "%s: value and reason disagree on %d record(s); signal dropped "
                "for those rows.",
                name,
                disagreement,
            )
        usable[name] = value_ok & reason_ok
        per_reason[name] = {
            str(k): int(v)
            for k, v in frame[f"{name}_reason"].value_counts().items()
        }

    # Cost counts once: cell and cluster are two views of one signal.
    cost_usable = usable["deviation_cell_cost"] | usable["deviation_cluster_cost"]
    valid_signal_count = (
        cost_usable.astype("int64")
        + usable["deviation_spend_ratio"].astype("int64")
        + usable["deviation_duration"].astype("int64")
    ).rename("valid_signal_count")

    LOGGER.info(
        "Signal validation over %d record(s): %d with no usable deviation "
        "signal, %d contract violation(s).",
        len(index),
        int((valid_signal_count == 0).sum()),
        violations,
    )

    return SignalValidation(
        usable=usable,
        valid_signal_count=valid_signal_count,
        contract_violations=violations,
        diagnostics={"reason_counts": per_reason, "n_core_signals": len(CORE_SIGNALS)},
    )


@dataclass(frozen=True)
class AnomalySignals:
    """The assembled signals, before any decision is taken."""

    frame: pd.DataFrame
    validation: SignalValidation
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        table = self.frame
        return {
            "stage4_version": STAGE4_VERSION,
            "cost_scope": table["cost_scope"].value_counts().to_dict(),
            "z_defined_pct": {
                name: round(100.0 * float(table[name].notna().mean()), 4)
                for name in ("z_cost", "z_spend", "z_duration")
            },
            "validation": self.validation.to_dict(),
            **self.diagnostics,
        }


def build_signals(frame: pd.DataFrame) -> AnomalySignals:
    """Assemble the four core anomaly signals.

    Cost prefers the cell-level deviation and falls back to the cluster-level
    one, recording which was used in ``cost_scope``. The fallback matters: the
    cell view is conservative by construction, so a record whose stratum is too
    thin still gets compared against its work type rather than nothing at all.

    Spend and duration are carried through at full magnitude. Lifecycle does
    **not** alter ``z_spend`` - it alters what ``z_spend`` is allowed to mean,
    which is decided in :func:`classify_types`.

    Args:
        frame: Corpus frame with Stage 2 and Stage 3 outputs.

    Returns:
        An :class:`AnomalySignals` whose frame is aligned to ``frame.index``.
    """
    validation = validate_signals(frame)
    usable = validation.usable
    index = frame.index
    table = pd.DataFrame(index=index)

    cell = frame["deviation_cell_cost"].to_numpy(dtype="float64", na_value=np.nan)
    cluster = frame["deviation_cluster_cost"].to_numpy(dtype="float64", na_value=np.nan)
    cell_ok = usable["deviation_cell_cost"].to_numpy()
    cluster_ok = usable["deviation_cluster_cost"].to_numpy()

    z_cost = np.where(cell_ok, cell, np.where(cluster_ok, cluster, np.nan))
    scope = np.full(len(index), "none", dtype=object)
    scope[cluster_ok] = "cluster"
    scope[cell_ok] = "cell"  # cell wins where both exist

    table["z_cost"] = z_cost
    table["cost_scope"] = pd.Series(scope, index=index, dtype="object")
    table["z_spend"] = np.where(
        usable["deviation_spend_ratio"].to_numpy(),
        frame["deviation_spend_ratio"].to_numpy(dtype="float64", na_value=np.nan),
        np.nan,
    )
    table["z_duration"] = np.where(
        usable["deviation_duration"].to_numpy(),
        frame["deviation_duration"].to_numpy(dtype="float64", na_value=np.nan),
        np.nan,
    )
    table["valid_signal_count"] = validation.valid_signal_count
    table["duplicate_score"] = frame["duplicate_score"].astype("float64")
    table["duplicate_flag"] = frame["duplicate_flag"].fillna(False).astype(bool)

    assert not np.isinf(table["z_cost"].to_numpy(dtype="float64")).any()
    assert set(table["cost_scope"].unique()) <= set(COST_SCOPES)

    LOGGER.info(
        "Signals assembled: cost defined for %.2f%% (cell %.2f%%, cluster "
        "fallback %.2f%%), spend %.2f%%, duration %.2f%%.",
        100.0 * float(table["z_cost"].notna().mean()),
        100.0 * float((table["cost_scope"] == "cell").mean()),
        100.0 * float((table["cost_scope"] == "cluster").mean()),
        100.0 * float(table["z_spend"].notna().mean()),
        100.0 * float(table["z_duration"].notna().mean()),
    )

    return AnomalySignals(frame=table, validation=validation)


def _lifecycle_masks(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Terminal and pre-completion masks from Stage 2's lifecycle label."""
    state = frame["lifecycle_state"].astype("object")
    terminal = state.isin(list(LIFECYCLE_TERMINAL_STATES)).to_numpy()
    pre_completion = state.isin(list(LIFECYCLE_PRE_COMPLETION_STATES)).to_numpy()
    return terminal, pre_completion


def classify_types(
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    z_threshold: float = Z_TYPE_THRESHOLD,
    confidence_threshold: float = CONFIDENCE_GATE_THRESHOLD,
) -> pd.DataFrame:
    """Assign every applicable anomaly type to each record.

    Types are not mutually exclusive and severity never overrides them: a
    record can be a cost outlier, a duplicate suspect and low confidence at
    once, and each is reported.

    The lifecycle gate is the one place a signal's *meaning* depends on
    something other than its magnitude. A pre-completion work with low spend is
    behaving exactly as it should, so ``underspend_anomaly`` is withheld -
    while ``z_spend`` itself is left untouched, because destroying the value
    would deny a reviewer the evidence. Overspend is never excused by
    lifecycle, matching Stage 2's rule that spending past a sanction is a
    control failure at any stage.

    Args:
        frame: Corpus frame with Stage 2 and Stage 3 outputs.
        signals: Output of :func:`build_signals`.
        z_threshold: |z| at which a deviation earns a type.
        confidence_threshold: Confidence below which the record is flagged.

    Returns:
        A boolean frame, one column per entry in ``ANOMALY_TYPES``.
    """
    index = frame.index
    types = pd.DataFrame(False, index=index, columns=list(ANOMALY_TYPES))

    z_cost = signals["z_cost"].to_numpy(dtype="float64")
    z_spend = signals["z_spend"].to_numpy(dtype="float64")
    z_duration = signals["z_duration"].to_numpy(dtype="float64")
    terminal, pre_completion = _lifecycle_masks(frame)

    # np.abs of NaN is NaN and every comparison against NaN is False, so an
    # undefined signal can never raise a type. That is the intended behaviour
    # and it is asserted in the tests rather than left to inference.
    with np.errstate(invalid="ignore"):
        types["cost_outlier"] = np.abs(z_cost) >= z_threshold
        types["overspend_anomaly"] = z_spend >= z_threshold
        types["underspend_anomaly"] = (z_spend <= -z_threshold) & terminal
        types["temporal_outlier"] = np.abs(z_duration) >= z_threshold

    types["duplicate_suspect"] = signals["duplicate_flag"].to_numpy(dtype=bool)
    types["low_confidence"] = (
        frame["confidence"].to_numpy(dtype="float64", na_value=0.0)
        < float(confidence_threshold)
    )
    # Three separate ways to have no trustworthy context, all of them worth
    # saying out loud rather than letting the record pass as unremarkable:
    # nothing measurable, no stable cell, or no norm for its work type at all.
    types["insufficient_context"] = (
        (signals["valid_signal_count"].to_numpy() == 0)
        | ~frame["peer_cell_stable"].fillna(False).to_numpy(dtype=bool)
        | ~frame["cluster_has_norm"].fillna(False).to_numpy(dtype=bool)
    )

    suppressed = int(((z_spend <= -z_threshold) & pre_completion).sum())
    if suppressed:
        LOGGER.info(
            "Lifecycle gate withheld underspend_anomaly on %d pre-completion "
            "record(s); z_spend preserved.",
            suppressed,
        )

    return types
