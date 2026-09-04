"""The three risk components: what is wrong, whether we can trust it, how stable.

Stage 5 never collapses these into one number and then forgets them. The score
is a product of the three, and the three are always reported alongside it,
because "risk 0.08" is unusable on its own: it could mean a clean record, a
filthy record nobody can read, or a serious finding on a corpus with no peers.
Those need three different responses.

Nothing here recomputes a Stage 2, 3 or 4 quantity. Every input is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ANOMALY_TYPES,
    MIN_CONFIDENCE_FOR_RISK,
    RISK_BREADTH_SATURATION,
    RISK_BREADTH_TYPES,
    RISK_BREADTH_WEIGHT,
    RISK_CRITICAL_DEFICIT_DECAY,
    RISK_DUPLICATE_WEIGHT,
    RISK_EXTREME_WEIGHT,
    RISK_LOW_CONFIDENCE_PENALTY,
    RISK_TEMPORAL_HARD_FAIL_QUALITY,
    RISK_UNCERTAINTY_COVERAGE_WEIGHT,
    RISK_UNCERTAINTY_NO_NORM,
    RISK_UNCERTAINTY_NO_SEVERITY,
    RISK_UNCERTAINTY_UNREACHABLE_DUPLICATE,
    RISK_UNCERTAINTY_UNSTABLE_CELL,
    RISK_UNCERTAINTY_NO_SEVERITY as _SATURATE,
    STAGE5_VERSION,
    Z_EXTREME_THRESHOLD,
    Z_HIGH_THRESHOLD,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Columns Stage 5 requires. Split by producing stage so a failure names the
#: stage that has to be re-run, not just a column.
REQUIRED_STAGE2: Tuple[str, ...] = (
    "confidence",
    "completeness",
    "temporal",
    "reconciliation",
    "completeness_defined",
    "temporal_defined",
    "reconciliation_defined",
    "critical_deficit",
    "cluster_penalty_factor",
    "temporal_hard_fail",
    "lifecycle_state",
)
REQUIRED_STAGE3: Tuple[str, ...] = (
    "cluster_id",
    "cluster_has_norm",
    "peer_cell_stable",
    "duplicate_flag",
    "duplicate_score",
)
REQUIRED_STAGE4: Tuple[str, ...] = (
    "severity_score",
    "severity_defined",
    "severity_defined_reason",
    "anomaly_types",
    "anomaly_count",
    "valid_signal_count",
    "z_cost",
    "z_spend",
    "z_duration",
)

#: Measured only when Stage 4's duplicate diagnostics ran. Absence means "not
#: measured", never "measured and fine".
OPTIONAL_COLUMNS: Tuple[str, ...] = ("duplicate_reachable",)

#: How many deviation comparisons a record could have had.
MAX_SIGNAL_COVERAGE: int = 3


class Stage5InputError(RuntimeError):
    """Raised when the Stage 2-4 contract is incomplete."""


def require_contract(frame: pd.DataFrame) -> None:
    """Fail loudly when an upstream column is absent.

    Args:
        frame: The corpus frame Stage 5 is asked to score.

    Raises:
        Stage5InputError: If any required column is missing, naming the stage
            responsible so the fix is unambiguous.
    """
    missing: Dict[str, List[str]] = {}
    for stage, columns in (
        ("stage2", REQUIRED_STAGE2),
        ("stage3", REQUIRED_STAGE3),
        ("stage4", REQUIRED_STAGE4),
    ):
        absent = [name for name in columns if name not in frame.columns]
        if absent:
            missing[stage] = absent
    if missing:
        raise Stage5InputError(
            f"Stage 5 requires complete Stage 2-4 output; missing {missing!r}. "
            "Run attach_confidence, attach_structure and attach_anomalies first."
        )


def _column(frame: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    """Read a numeric column as float64, NaN-preserving."""
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(
        dtype="float64", na_value=default
    )


def _bool_column(frame: pd.DataFrame, name: str, default: bool = False) -> np.ndarray:
    """Read a boolean column, treating absence of a value as ``default``."""
    if name not in frame.columns:
        return np.full(len(frame), default, dtype=bool)
    return frame[name].fillna(default).to_numpy(dtype=bool)


def _type_matrix(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    """One boolean array per anomaly type, from columns or from the list."""
    matrix: Dict[str, np.ndarray] = {}
    for name in ANOMALY_TYPES:
        column = f"type_{name}"
        if column in frame.columns:
            matrix[name] = frame[column].to_numpy(dtype=bool)
        elif "anomaly_types" in frame.columns:
            matrix[name] = np.asarray(
                [name in (types or []) for types in frame["anomaly_types"]], dtype=bool
            )
        else:
            matrix[name] = np.zeros(len(frame), dtype=bool)
    return matrix


# ---------------------------------------------------------------------------
# Step 1 - signal strength
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalStrength:
    """What is wrong, and how broadly."""

    value: pd.Series
    breadth: pd.Series
    extreme: pd.Series
    duplicate: pd.Series
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        defined = self.value.notna()
        return {
            "defined_pct": round(100.0 * float(defined.mean()), 4)
            if len(self.value)
            else 0.0,
            "median": round(float(self.value[defined].median()), 6)
            if defined.any()
            else None,
            "p95": round(float(self.value[defined].quantile(0.95)), 6)
            if defined.any()
            else None,
            **self.diagnostics,
        }


def compute_signal_strength(
    frame: pd.DataFrame,
    breadth_weight: float = RISK_BREADTH_WEIGHT,
    extreme_weight: float = RISK_EXTREME_WEIGHT,
    duplicate_weight: float = RISK_DUPLICATE_WEIGHT,
    saturation: int = RISK_BREADTH_SATURATION,
) -> SignalStrength:
    """Compose how strong the case against a record is.

    Severity is the base; breadth, extreme magnitude and a flagged duplicate
    fill the **remaining headroom** above it::

        strength = base + (1 - base) * (w_b * breadth + w_e * extreme + w_d * dup)

    That shape is chosen rather than a weighted sum for three reasons: it stays
    inside [0,1] without clipping, it is strictly increasing in severity so the
    Stage 4 ordering is never inverted, and a boost can never *lower* a score.
    It also bounds the duplicate contribution at ``(1 - base) * 0.10 <= 0.10``,
    which is the cap the design requires - the duplicate detector's measured
    recall is about 1%, so its absence means almost nothing and its presence
    must never carry a case by itself.

    ``low_confidence`` is deliberately excluded from breadth. It describes the
    evidence, not the work, and it is already priced into the data-quality
    term; counting it here would charge a record twice for one defect.

    Args:
        frame: Corpus frame with Stage 4 output.
        breadth_weight: Headroom share for anomaly breadth.
        extreme_weight: Headroom share for extreme magnitude.
        duplicate_weight: Headroom share for a flagged duplicate; ``<= 0.10``.
        saturation: Anomaly count at which breadth is full.

    Returns:
        A :class:`SignalStrength`; ``value`` is NaN wherever severity is.

    Raises:
        ValueError: If the weights are negative, could exceed the headroom, or
            the duplicate weight breaches its cap.
    """
    if min(breadth_weight, extreme_weight, duplicate_weight) < 0.0:
        raise ValueError("signal-strength weights must be non-negative")
    if breadth_weight + extreme_weight + duplicate_weight > 1.0:
        raise ValueError(
            "signal-strength weights must not exceed the available headroom"
        )
    if duplicate_weight > RISK_DUPLICATE_WEIGHT:
        raise ValueError(
            f"duplicate weight {duplicate_weight} exceeds the {RISK_DUPLICATE_WEIGHT} "
            "cap; the duplicate signal may support a case, never make one"
        )
    if saturation < 1:
        raise ValueError("breadth saturation must be at least 1")

    index = frame.index
    severity = _column(frame, "severity_score")
    defined = _bool_column(frame, "severity_defined")

    types = _type_matrix(frame)
    counted = np.zeros(len(frame), dtype="int64")
    for name in RISK_BREADTH_TYPES:
        counted += types[name].astype("int64")
    breadth = np.minimum(counted / float(saturation), 1.0)

    # Extreme bucket. Stage 4's severity saturates at |z| = Z_SEVERITY_SCALE,
    # so a z of 30 and a z of 6 arrive here identical. Stage 3's own extreme
    # and high thresholds restore the part of that ordering that matters.
    #
    # np.nanmax would warn on an all-undefined row, and invariant 5 forbids a
    # RuntimeWarning anywhere in Stage 5. Substituting -inf for NaN before the
    # max is equivalent and silent: a row with no defined z yields -inf, which
    # falls below both thresholds and scores 0 - the correct answer, since an
    # unmeasured deviation is not an extreme one.
    stacked = np.vstack(
        [np.abs(_column(frame, name)) for name in ("z_cost", "z_spend", "z_duration")]
    )
    peak = np.max(np.where(np.isnan(stacked), -np.inf, stacked), axis=0)
    extreme = np.where(
        peak >= Z_EXTREME_THRESHOLD,
        1.0,
        np.where(peak >= Z_HIGH_THRESHOLD, 0.5, 0.0),
    )

    duplicate = _bool_column(frame, "duplicate_flag").astype("float64")

    boost = (
        breadth_weight * breadth
        + extreme_weight * extreme
        + duplicate_weight * duplicate
    )
    base = np.where(defined, severity, np.nan)
    with np.errstate(invalid="ignore"):
        value = base + (1.0 - base) * boost
    value = np.where(defined & np.isfinite(base), value, np.nan)

    finite = np.isfinite(value)
    assert bool(((value[finite] >= 0.0) & (value[finite] <= 1.0)).all()), (
        "signal strength escaped [0,1]"
    )
    assert bool((value[finite] + 1e-12 >= base[finite]).all()), (
        "a boost lowered a signal strength below its own severity"
    )

    LOGGER.info(
        "Signal strength over %d record(s): %d defined, %d with an extreme "
        "deviation, %d with a flagged duplicate.",
        len(frame),
        int(finite.sum()),
        int((extreme > 0).sum()),
        int(duplicate.sum()),
    )

    return SignalStrength(
        value=pd.Series(value, index=index, dtype="float64", name="risk_signal_strength"),
        breadth=pd.Series(breadth, index=index, dtype="float64", name="risk_breadth"),
        extreme=pd.Series(extreme, index=index, dtype="float64", name="risk_extreme"),
        duplicate=pd.Series(duplicate, index=index, dtype="float64", name="risk_duplicate"),
        diagnostics={
            "n_extreme": int((extreme >= 1.0).sum()),
            "n_high": int((extreme == 0.5).sum()),
            "n_duplicate": int(duplicate.sum()),
            "weights": {
                "breadth": breadth_weight,
                "extreme": extreme_weight,
                "duplicate": duplicate_weight,
            },
        },
    )


# ---------------------------------------------------------------------------
# Step 2 - data quality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataQuality:
    """Whether the record can carry a finding at all."""

    value: pd.Series
    component_floor: pd.Series
    deficit_factor: pd.Series
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "median": round(float(self.value.median()), 6) if len(self.value) else None,
            "p05": round(float(self.value.quantile(0.05)), 6) if len(self.value) else None,
            **self.diagnostics,
        }


def compute_data_quality(
    frame: pd.DataFrame,
    min_confidence: float = MIN_CONFIDENCE_FOR_RISK,
    deficit_decay: float = RISK_CRITICAL_DEFICIT_DECAY,
    hard_fail_quality: float = RISK_TEMPORAL_HARD_FAIL_QUALITY,
    low_confidence_penalty: float = RISK_LOW_CONFIDENCE_PENALTY,
) -> DataQuality:
    """Compose how far the record's own evidence can be trusted.

    ``confidence`` is the base, multiplied by three modifiers::

        quality = confidence
                  * min(defined Stage 2 components)     non-compensatory floor
                  * exp(-k * critical_deficit)          missing critical fields
                  * cluster_penalty_factor              Stage 2's own penalty

    then floored at ``hard_fail_quality`` where the dates are internally
    impossible, and multiplied by ``low_confidence_penalty`` below the gate.

    The component floor is a minimum rather than a mean on purpose: Stage 2's
    whole philosophy is that a zero component dominates, and averaging would let
    a perfect completeness score paper over a broken reconciliation.

    **Known double count, accepted.** ``cluster_penalty_factor`` and
    ``critical_deficit`` are both already reflected inside Stage 2's
    ``completeness``, so applying them again charges the same defect twice. The
    design names all three as inputs, and the direction is conservative - it
    lowers risk on poor records, never raises it - so it is applied as specified
    and stated here rather than silently dropped.

    Args:
        frame: Corpus frame with the Stage 2 breakdown.
        min_confidence: The gate below which the extra penalty applies.
        deficit_decay: ``k`` in the exponential decay on critical deficit.
        hard_fail_quality: Ceiling under an impossible date ordering.
        low_confidence_penalty: Multiplier below the gate.

    Returns:
        A :class:`DataQuality` in [0,1] with no NaN - Stage 2 defines
        ``confidence`` for every record, so quality is always measurable.

    Raises:
        ValueError: If a parameter is outside its meaningful range.
    """
    if deficit_decay < 0.0:
        raise ValueError("deficit decay must be non-negative")
    if not 0.0 <= hard_fail_quality <= 1.0:
        raise ValueError("hard-fail quality must lie in [0,1]")
    if not 0.0 <= low_confidence_penalty <= 1.0:
        raise ValueError("low-confidence penalty must lie in [0,1]")

    index = frame.index
    confidence = np.clip(_column(frame, "confidence", 0.0), 0.0, 1.0)

    # Non-compensatory floor over the components Stage 2 could actually
    # measure. An undefined component is skipped, never treated as 1.0 -
    # that would be the vacuous-perfection bug Stage 2 was built to avoid.
    floor = np.ones(len(frame), dtype="float64")
    any_defined = np.zeros(len(frame), dtype=bool)
    for name in ("completeness", "temporal", "reconciliation"):
        values = np.clip(_column(frame, name, 1.0), 0.0, 1.0)
        defined = _bool_column(frame, f"{name}_defined")
        floor = np.where(defined, np.minimum(floor, values), floor)
        any_defined |= defined
    # No component measurable at all: the floor asserts nothing, so it is 1.0
    # and `confidence` alone carries the term. Stage 2 already priced this.
    floor = np.where(any_defined, floor, 1.0)

    deficit = np.maximum(_column(frame, "critical_deficit", 0.0), 0.0)
    deficit_factor = np.exp(-float(deficit_decay) * deficit)

    penalty = np.clip(_column(frame, "cluster_penalty_factor", 1.0), 0.0, 1.0)

    quality = confidence * floor * deficit_factor * penalty

    hard_fail = _bool_column(frame, "temporal_hard_fail")
    quality = np.where(hard_fail, np.minimum(quality, hard_fail_quality), quality)

    below_gate = confidence < float(min_confidence)
    quality = np.where(below_gate, quality * float(low_confidence_penalty), quality)

    quality = np.clip(quality, 0.0, 1.0)
    assert np.isfinite(quality).all(), "data quality produced a non-finite value"

    LOGGER.info(
        "Data quality over %d record(s): median %.4f; %d below the confidence "
        "gate, %d with an impossible date ordering.",
        len(frame),
        float(np.median(quality)) if len(frame) else 0.0,
        int(below_gate.sum()),
        int(hard_fail.sum()),
    )

    return DataQuality(
        value=pd.Series(quality, index=index, dtype="float64", name="risk_data_quality"),
        component_floor=pd.Series(floor, index=index, dtype="float64", name="risk_component_floor"),
        deficit_factor=pd.Series(
            deficit_factor, index=index, dtype="float64", name="risk_deficit_factor"
        ),
        diagnostics={
            "n_below_gate": int(below_gate.sum()),
            "n_temporal_hard_fail": int(hard_fail.sum()),
            "deficit_decay": float(deficit_decay),
            "_double_count": (
                "critical_deficit and cluster_penalty_factor are also inside "
                "Stage 2 completeness; applied per the design, conservative"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Step 3 - uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Uncertainty:
    """How stable the judgement is. Higher means less stable."""

    value: pd.Series
    contributions: pd.DataFrame
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "median": round(float(self.value.median()), 6) if len(self.value) else None,
            "p95": round(float(self.value.quantile(0.95)), 6) if len(self.value) else None,
            "n_saturated": int((self.value >= 1.0).sum()),
            **self.diagnostics,
        }


def compute_uncertainty(
    frame: pd.DataFrame,
    no_severity: float = RISK_UNCERTAINTY_NO_SEVERITY,
    no_norm: float = RISK_UNCERTAINTY_NO_NORM,
    unstable_cell: float = RISK_UNCERTAINTY_UNSTABLE_CELL,
    coverage_weight: float = RISK_UNCERTAINTY_COVERAGE_WEIGHT,
    unreachable_duplicate: float = RISK_UNCERTAINTY_UNREACHABLE_DUPLICATE,
) -> Uncertainty:
    """Compose how unstable the judgement about a record is.

    Contributions are additive and then clipped into [0,1]. Additive rather
    than multiplicative because these are independent ways of not knowing, and
    they should accumulate: a record with no norm *and* one signal *and* an
    unstable cell is worse off than one with any single defect.

    An undefined severity saturates the term on its own. If the central
    quantity could not be computed, there is nothing to be uncertain *around*,
    and the risk score is gated off anyway.

    Args:
        frame: Corpus frame with Stage 3 and Stage 4 output.
        no_severity: Contribution when severity is undefined.
        no_norm: Contribution when the work type carries no norm.
        unstable_cell: Contribution when the peer cell is too small.
        coverage_weight: Weight on the fraction of missing comparisons.
        unreachable_duplicate: Contribution when a flagged duplicate was not
            reachable. Provably impossible while Stage 4 holds; see below.

    Returns:
        An :class:`Uncertainty` in [0,1] with no NaN.

    Raises:
        ValueError: If any contribution is negative.
    """
    weights = (no_severity, no_norm, unstable_cell, coverage_weight, unreachable_duplicate)
    if min(weights) < 0.0:
        raise ValueError("uncertainty contributions must be non-negative")

    index = frame.index
    contributions = pd.DataFrame(index=index)

    severity_defined = _bool_column(frame, "severity_defined")
    contributions["no_severity"] = np.where(severity_defined, 0.0, float(no_severity))

    has_norm = _bool_column(frame, "cluster_has_norm")
    contributions["no_norm"] = np.where(has_norm, 0.0, float(no_norm))

    stable = _bool_column(frame, "peer_cell_stable")
    contributions["unstable_cell"] = np.where(stable, 0.0, float(unstable_cell))

    coverage = np.clip(
        _column(frame, "valid_signal_count", 0.0) / float(MAX_SIGNAL_COVERAGE), 0.0, 1.0
    )
    contributions["coverage"] = float(coverage_weight) * (1.0 - coverage)

    # Provably empty while Stage 4 holds: the temporal decay lies in [0,1], so a
    # blended score at or above the detection threshold implies a cosine at or
    # above the (lower) reachability cut. Computed anyway, and counted, because
    # the day it stops being empty is the day Stage 3 and Stage 4 disagree - and
    # a risk score should say so rather than quietly absorb the contradiction.
    flagged = _bool_column(frame, "duplicate_flag")
    if "duplicate_reachable" in frame.columns:
        unreachable = flagged & ~_bool_column(frame, "duplicate_reachable")
    else:
        # Not measured. Absence of the measurement is not evidence of a
        # contradiction, so the term stays silent rather than guessing.
        unreachable = np.zeros(len(frame), dtype=bool)
    contributions["unreachable_duplicate"] = np.where(
        unreachable, float(unreachable_duplicate), 0.0
    )

    value = np.clip(contributions.to_numpy(dtype="float64").sum(axis=1), 0.0, 1.0)
    assert np.isfinite(value).all(), "uncertainty produced a non-finite value"
    assert bool((value[~severity_defined] >= 1.0).all()), (
        "an undefined severity did not saturate the uncertainty"
    )

    n_unreachable = int(unreachable.sum())
    if n_unreachable:
        LOGGER.warning(
            "%d record(s) are flagged duplicates that the detector could not "
            "have reached. This is impossible while Stage 3 and Stage 4 agree; "
            "treat the duplicate signal as unreliable on this corpus.",
            n_unreachable,
        )

    LOGGER.info(
        "Uncertainty over %d record(s): median %.4f, %d saturated.",
        len(frame),
        float(np.median(value)) if len(frame) else 0.0,
        int((value >= 1.0).sum()),
    )

    return Uncertainty(
        value=pd.Series(value, index=index, dtype="float64", name="risk_uncertainty"),
        contributions=contributions,
        diagnostics={
            "n_unreachable_duplicate": n_unreachable,
            "duplicate_reachability_measured": "duplicate_reachable" in frame.columns,
            "weights": {
                "no_severity": no_severity,
                "no_norm": no_norm,
                "unstable_cell": unstable_cell,
                "coverage": coverage_weight,
                "unreachable_duplicate": unreachable_duplicate,
            },
        },
    )
