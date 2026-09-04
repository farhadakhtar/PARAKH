"""C(r) - log-space aggregation of the three confidence components.

    C(r) = exp( w1 log C_comp + w2 log C_temp + w3 log C_recon ),  sum w_i = 1

This is a **weighted geometric mean**, and every property that makes Stage 2
worth building comes from that choice rather than from the components:

**Non-compensatory.** No component can buy back another's failure. Under an
arithmetic mean a record with ``C_temp = 0`` but perfect completeness and
reconciliation scores 0.67 - comfortably above any plausible theta_C - and the
system would emit a fraud hypothesis about a record whose dates are physically
impossible. That single substitution defeats the entire premise, which is why
"arithmetic mean" is a listed fail condition rather than a preference.

**Zero dominance.** ``prod C_i^{w_i} = 0`` whenever any ``C_i = 0``. The system
refuses, rather than degrades, on unevidenced records.

**Log space is not a convenience.** With three small components the direct
product underflows to a flat zero and destroys the *ranking among
low-confidence records* - exactly the population the REMEDIATE queue has to
prioritise. Summing logs preserves that ordering down to ~1e-308.

Safe log handling
-----------------
``log(0)`` is never evaluated. Zero (or non-finite) components are detected
first and substituted with 1.0 inside the logarithm; the affected rows are then
forced to 0.0 by an explicit mask. This is stricter than relying on
``exp(-inf) = 0``, because that identity fails in the one case that matters:
``w_i = 0`` paired with ``C_i = 0`` gives ``0 * -inf = NaN``, which would
propagate silently to the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.core.constants import (
    CLUSTER_PENALTY_ALLOWANCE,
    IMPLAUSIBLE_AMOUNT_THRESHOLD,
    RECON_IMPLAUSIBLE_MAGNITUDE_CREDIT,
    RECON_OVERSPEND_TOLERANCE,
    RECON_PRE_COMPLETION_STATUSES,
    RECON_TERMINAL_STATUSES,
    RECON_UNKNOWN_STATUS_GAMMA_SCALE,
    STATUS_FIELD,
    CLUSTER_PENALTY_DELTA,
    COMPLETENESS_CREDIT,
    COMPLETENESS_WEIGHT_MODE,
    CRITICAL_FIELDS,
    FIELD_CRITICALITY,
    RECON_MODE,
    RECON_NON_POSITIVE_SANCTION_CREDIT,
    RECON_UNDERSPEND_FLOOR,
    RECON_UNDERSPEND_GAMMA,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_HISTOGRAM_BINS,
    CONFIDENCE_LOW_THRESHOLD,
    CONFIDENCE_WEIGHTS,
    ENTROPY_NORMALIZATION,
    MIN_FIELD_COVERAGE,
    ORDERED_DATE_PAIRS,
    RECON_BOTH_NULL_CREDIT,
    RECON_EPSILON,
    RECON_LAMBDA,
    RECON_NON_FINITE_CREDIT,
    RECON_NORMALIZATION,
    RECON_ONE_SIDED_CREDIT,
    RECONCILIATION_PAIR,
    STAGE2_VERSION,
    TEMPORAL_HARD_FAIL_ON_FUTURE,
    TEMPORAL_KAPPA_PER_DAY,
    TEMPORAL_MISSING_PAIR_CREDIT,
    WEIGHT_SUM_TOLERANCE,
)
from src.core.logger import get_logger
from src.stage1.schema import SCHEMA, Schema
from src.stage2.completeness import (
    CompletenessResult,
    FieldWeights,
    compute_completeness_result,
)
from src.stage2.reconciliation import ReconciliationResult, compute_reconciliation_result
from src.stage2.temporal import TemporalResult, compute_temporal_result
from src.utils.helpers import safe_percentage, write_json

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from src.stage1.corpus import Corpus

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

CONFIDENCE_COLUMN = "confidence"
COMPONENT_COLUMNS: Tuple[str, str, str] = ("completeness", "temporal", "reconciliation")

#: The complete Stage 2 -> Stage 3 column contract, in a fixed order.
#:
#: Every column here is written by :func:`attach_confidence`, row-aligned to
#: the corpus index, deterministic, and JSON/CSV serialisable. Stage 3 may
#: rely on all of them existing. Columns are only ever ADDED to this tuple;
#: removing one is a breaking change to the downstream contract.
BREAKDOWN_COLUMNS: Tuple[str, ...] = (
    # --- scores -----------------------------------------------------------
    "confidence",
    "completeness",
    "temporal",
    "reconciliation",
    # --- which components were measurable (the defined-component mask) ----
    "completeness_defined",
    "temporal_defined",
    "reconciliation_defined",
    "n_components_used",
    # --- completeness evidence --------------------------------------------
    "n_valid_fields",
    "critical_missing_count",
    "critical_deficit",
    "cluster_penalty_factor",
    # --- temporal evidence -------------------------------------------------
    "temporal_pairs_evaluated",
    "temporal_hard_fail",
    # --- reconciliation evidence -------------------------------------------
    "reconciliation_branch",
    "lifecycle_state",
    "spend_ratio",
)

#: Verdict bands used only for human-readable explanation. They carry no
#: scoring weight and are NOT routing thresholds - theta_C is Stage 6's
#: business, and is a calibration parameter, not a constant.
EXPLANATION_BANDS: Tuple[Tuple[float, str], ...] = (
    (0.0, "refused"),
    (0.2, "low"),
    (0.8, "moderate"),
    (1.01, "high"),
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceConfig:
    """All Stage 2 calibration parameters in one immutable object.

    Every field here is a calibration parameter, not a constant of nature. The
    README is explicit that PARAKH is non-operational until these are estimated
    against real data; the defaults reproduce Stage2.md exactly so the scores
    are structurally correct while remaining, in its words, "operationally
    meaningless" until calibrated.
    """

    weights: Tuple[float, float, float] = CONFIDENCE_WEIGHTS

    # completeness
    completeness_credit: Mapping[str, float] = field(
        default_factory=lambda: dict(COMPLETENESS_CREDIT)
    )
    completeness_fields: Optional[Tuple[str, ...]] = None
    entropy_normalization: str = ENTROPY_NORMALIZATION
    min_field_coverage: float = MIN_FIELD_COVERAGE
    #: v2: how v_f is formed. See the completeness module docstring for why
    #: criticality replaced the v1 surprisal weighting.
    completeness_weight_mode: str = COMPLETENESS_WEIGHT_MODE
    field_criticality: Mapping[str, float] = field(
        default_factory=lambda: dict(FIELD_CRITICALITY)
    )
    critical_fields: Tuple[str, ...] = CRITICAL_FIELDS
    cluster_delta: float = CLUSTER_PENALTY_DELTA
    cluster_allowance: float = CLUSTER_PENALTY_ALLOWANCE

    # temporal
    kappa: float = TEMPORAL_KAPPA_PER_DAY
    missing_pair_credit: float = TEMPORAL_MISSING_PAIR_CREDIT
    hard_fail_on_future: bool = TEMPORAL_HARD_FAIL_ON_FUTURE
    ordered_date_pairs: Tuple[Tuple[str, str], ...] = ORDERED_DATE_PAIRS

    # reconciliation
    lam: float = RECON_LAMBDA
    epsilon: float = RECON_EPSILON
    one_sided_credit: float = RECON_ONE_SIDED_CREDIT
    both_null_credit: float = RECON_BOTH_NULL_CREDIT
    non_finite_credit: float = RECON_NON_FINITE_CREDIT
    recon_normalization: str = RECON_NORMALIZATION
    reconciliation_pair: Tuple[str, str] = RECONCILIATION_PAIR
    #: v2: plausibility scoring rather than the v1 equality test.
    recon_mode: str = RECON_MODE
    underspend_floor: float = RECON_UNDERSPEND_FLOOR
    underspend_gamma: float = RECON_UNDERSPEND_GAMMA
    non_positive_sanction_credit: float = RECON_NON_POSITIVE_SANCTION_CREDIT
    #: Final corrections (audit response).
    overspend_tolerance: float = RECON_OVERSPEND_TOLERANCE
    implausible_credit: float = RECON_IMPLAUSIBLE_MAGNITUDE_CREDIT
    implausible_threshold: float = IMPLAUSIBLE_AMOUNT_THRESHOLD
    status_field: str = STATUS_FIELD
    pre_completion_statuses: Tuple[str, ...] = RECON_PRE_COMPLETION_STATUSES
    terminal_statuses: Tuple[str, ...] = RECON_TERMINAL_STATUSES
    unknown_status_gamma_scale: float = RECON_UNKNOWN_STATUS_GAMMA_SCALE

    #: Exclude a component from a record's geometric mean when it could not be
    #: measured for that record, renormalising the remaining weights.
    #:
    #: With this off, an unmeasurable component scores 1.0 and a wholly empty
    #: record - no dates to disorder, no amounts to disagree - collects two
    #: vacuous perfect scores and lands around C = 0.63. Turning it on is what
    #: makes "all missing -> C ~ 0" true rather than aspirational.
    drop_undefined_components: bool = True

    # reporting
    low_threshold: float = CONFIDENCE_LOW_THRESHOLD
    high_threshold: float = CONFIDENCE_HIGH_THRESHOLD
    histogram_bins: int = CONFIDENCE_HISTOGRAM_BINS

    def __post_init__(self) -> None:
        """Validate the configuration at construction, not at scoring time."""
        if len(self.weights) != 3:
            raise ValueError(f"weights must have 3 entries, got {len(self.weights)}")
        if any(w < 0.0 for w in self.weights):
            raise ValueError(f"weights must be non-negative, got {self.weights}")
        total = float(sum(self.weights))
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"weights must sum to 1, got {total}")
        for reason, value in self.completeness_credit.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"completeness credit for {reason!r} must lie in [0,1], got {value}"
                )
        if not 0.0 <= self.low_threshold <= self.high_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= low <= high <= 1, got "
                f"({self.low_threshold}, {self.high_threshold})"
            )
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be >= 1")
        if self.cluster_delta < 0.0:
            raise ValueError(
                f"cluster_delta must be non-negative, got {self.cluster_delta}"
            )
        if self.cluster_allowance < 0.0:
            raise ValueError(
                "cluster_allowance must be non-negative, got "
                f"{self.cluster_allowance}"
            )
        for name, value in self.field_criticality.items():
            if float(value) < 0.0:
                raise ValueError(
                    f"criticality for {name!r} must be non-negative, got {value}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable echo of the configuration."""
        return {
            "stage2_version": STAGE2_VERSION,
            "weights": {
                "completeness": self.weights[0],
                "temporal": self.weights[1],
                "reconciliation": self.weights[2],
            },
            "completeness": {
                "credit": dict(self.completeness_credit),
                "entropy_normalization": self.entropy_normalization,
                "min_field_coverage": self.min_field_coverage,
                "weight_mode": self.completeness_weight_mode,
                "criticality": dict(self.field_criticality),
                "critical_fields": list(self.critical_fields),
                "cluster_delta": self.cluster_delta,
                "cluster_allowance": self.cluster_allowance,
                "fields": list(self.completeness_fields)
                if self.completeness_fields
                else None,
            },
            "temporal": {
                "kappa_per_day": self.kappa,
                "missing_pair_credit": self.missing_pair_credit,
                "hard_fail_on_future": self.hard_fail_on_future,
                "ordered_date_pairs": [list(p) for p in self.ordered_date_pairs],
            },
            "reconciliation": {
                "lambda": self.lam,
                "epsilon": self.epsilon,
                "one_sided_credit": self.one_sided_credit,
                "both_null_credit": self.both_null_credit,
                "non_finite_credit": self.non_finite_credit,
                "normalization": self.recon_normalization,
                "pair": list(self.reconciliation_pair),
                "mode": self.recon_mode,
                "underspend_floor": self.underspend_floor,
                "underspend_gamma": self.underspend_gamma,
                "non_positive_sanction_credit": self.non_positive_sanction_credit,
                "overspend_tolerance": self.overspend_tolerance,
                "implausible_credit": self.implausible_credit,
                "implausible_threshold": self.implausible_threshold,
                "status_field": self.status_field,
                "pre_completion_statuses": list(self.pre_completion_statuses),
                "terminal_statuses": list(self.terminal_statuses),
                "unknown_status_gamma_scale": self.unknown_status_gamma_scale,
            },
            "aggregation": {
                "drop_undefined_components": self.drop_undefined_components,
            },
            "reporting": {
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
                "histogram_bins": self.histogram_bins,
            },
        }


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def log_space_geometric_mean(
    components: Sequence[np.ndarray],
    weights: Sequence[float],
    defined: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Weighted geometric mean, evaluated as a sum of logarithms.

    Args:
        components: One array per component, all the same length, each in
            [0, 1].
        weights: Non-negative weights, one per component, summing to 1.
        defined: Optional boolean mask per component saying, per record,
            whether that component was measurable at all. Undefined components
            are dropped and the remaining weights renormalised for that row.
            Defaults to everything defined.

    Returns:
        Float array in [0, 1]. Guaranteed free of NaN and inf.

    Raises:
        ValueError: On a length mismatch or malformed weights.

    Note:
        **Undefined is not the same as perfect.** A record with no dates has
        nothing to check temporally; scoring it ``C_temp = 1`` asserts perfect
        coherence on zero evidence, and lets a wholly empty record acquire a
        respectable confidence. Stage2.md sec.5.4 says to "ignore component"
        when there is nothing to compare, and in a weighted geometric mean
        *ignoring* means dropping the term and renormalising - which is what
        this does. A record with no defined component at all scores 0.

    Note:
        ``log(0)`` is never evaluated. Rows where a **defined** component is
        zero or non-finite are identified first and forced to ``0.0``
        afterwards, while the logarithm sees a substituted ``1.0``. Relying on
        ``exp(-inf) = 0`` instead would break for a zero-weighted zero
        component, where ``0 * -inf`` is NaN.
    """
    if len(components) != len(weights):
        raise ValueError(f"{len(components)} components but {len(weights)} weights")
    if not components:
        raise ValueError("at least one component is required")

    arrays = [np.asarray(component, dtype="float64") for component in components]
    length = len(arrays[0])
    for array in arrays:
        if len(array) != length:
            raise ValueError("all components must have the same length")

    weight_array = np.asarray(weights, dtype="float64")
    if np.any(weight_array < 0.0):
        raise ValueError(f"weights must be non-negative, got {weights}")
    total = float(weight_array.sum())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"weights must sum to 1, got {total}")

    if length == 0:
        return np.asarray([], dtype="float64")

    if defined is None:
        masks = [np.ones(length, dtype=bool) for _ in arrays]
    else:
        if len(defined) != len(arrays):
            raise ValueError("defined must supply one mask per component")
        masks = [np.asarray(mask, dtype=bool) for mask in defined]
        for mask in masks:
            if len(mask) != length:
                raise ValueError("defined masks must match the component length")

    # Per-row effective weights, renormalised over the defined components.
    effective = np.vstack(
        [weight * mask.astype("float64") for weight, mask in zip(weight_array, masks)]
    )
    row_total = effective.sum(axis=0)
    no_evidence = row_total <= 0.0
    normalised = effective / np.where(no_evidence, 1.0, row_total)

    # Zero dominance, evaluated only over components that were measurable.
    annihilate = np.zeros(length, dtype=bool)
    for array, mask in zip(arrays, masks):
        annihilate |= mask & (~np.isfinite(array) | (array <= 0.0))

    log_sum = np.zeros(length, dtype="float64")
    for array, row_weight in zip(arrays, normalised):
        safe = np.where(np.isfinite(array) & (array > 0.0), array, 1.0)
        log_sum += row_weight * np.log(safe)

    scores = np.exp(log_sum)
    scores = np.where(annihilate | no_evidence, 0.0, scores)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class ConfidenceReport(BaseModel):
    """Corpus-level confidence summary (Stage2.md sec.10)."""

    model_config = ConfigDict(frozen=True)

    # --- sec.10.1 mandated keys -----------------------------------------
    mean_confidence: float = 0.0
    low_confidence_pct: float = 0.0
    high_confidence_pct: float = 0.0

    # --- distribution ----------------------------------------------------
    stage2_version: str = STAGE2_VERSION
    n_records: int = 0
    median_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    std_confidence: float = 0.0
    zero_confidence_pct: float = 0.0
    histogram: Dict[str, int] = Field(default_factory=dict)

    # --- components -------------------------------------------------------
    component_summary: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    #: Share of records for which each component was not measurable and was
    #: therefore dropped from the geometric mean rather than scored 1.0.
    components_dropped_pct: Dict[str, float] = Field(default_factory=dict)
    completeness_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    temporal_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    reconciliation_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict form, suitable for ``json.dump``."""
        return self.model_dump()

    def prd_view(self) -> Dict[str, Any]:
        """Just the three keys Stage2.md sec.10.1 mandates."""
        return {
            "mean_confidence": self.mean_confidence,
            "low_confidence_pct": self.low_confidence_pct,
            "high_confidence_pct": self.high_confidence_pct,
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceResult:
    """Per-record confidence, its component breakdown, and the corpus report."""

    scores: pd.Series
    completeness: CompletenessResult
    temporal: TemporalResult
    reconciliation: ReconciliationResult
    report: ConfidenceReport
    config: ConfidenceConfig

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def field_weights(self) -> FieldWeights:
        """The frozen ``v_f`` weights used for ``C_comp``."""
        return self.completeness.weights

    @property
    def breakdown(self) -> pd.DataFrame:
        """Per-record confidence, components and full evidence base.

        Columns are exactly :data:`BREAKDOWN_COLUMNS`, in that order. This is
        the same set :func:`attach_confidence` writes onto the corpus, so a
        Stage 3 consumer sees identical signals whichever it reads.
        """
        completeness_defined = self.completeness.defined
        temporal_defined = self.temporal.defined
        reconciliation_defined = self.reconciliation.defined
        return pd.DataFrame(
            {
                CONFIDENCE_COLUMN: self.scores,
                "completeness": self.completeness.scores,
                "temporal": self.temporal.scores,
                "reconciliation": self.reconciliation.scores,
                "completeness_defined": completeness_defined,
                "temporal_defined": temporal_defined,
                "reconciliation_defined": reconciliation_defined,
                "n_components_used": (
                    completeness_defined.astype("int64")
                    + temporal_defined.astype("int64")
                    + reconciliation_defined.astype("int64")
                ),
                "n_valid_fields": self.completeness.n_valid_fields,
                "critical_missing_count": self.completeness.critical_missing_count,
                "critical_deficit": self.completeness.critical_deficit,
                "cluster_penalty_factor": self.completeness.cluster_factor,
                "temporal_pairs_evaluated": self.temporal.pairs_evaluated,
                "temporal_hard_fail": self.temporal.hard_fail,
                "reconciliation_branch": self.reconciliation.branch,
                "lifecycle_state": self.reconciliation.lifecycle,
                "spend_ratio": self.reconciliation.ratio,
            },
            index=self.scores.index,
        ).loc[:, list(BREAKDOWN_COLUMNS)]

    def to_records(self) -> List[Dict[str, Any]]:
        """Per-record payload in the exact shape of Stage2.md sec.4."""
        return [
            {
                "confidence": float(confidence),
                "components": {
                    "completeness": float(comp),
                    "temporal": float(temp),
                    "reconciliation": float(recon),
                },
            }
            for confidence, comp, temp, recon in zip(
                self.scores.to_numpy(),
                self.completeness.scores.to_numpy(),
                self.temporal.scores.to_numpy(),
                self.reconciliation.scores.to_numpy(),
            )
        ]

    def explain(self, row: Any) -> Dict[str, Any]:
        """Explain one record, using this result's own weights.

        Reads :attr:`breakdown`; recomputes nothing. See
        :func:`explain_confidence`.

        Args:
            row: Index label of the record to explain.

        Returns:
            A JSON-serialisable explanation dict.
        """
        return explain_confidence(self.breakdown, row, weights=self.config.weights)

    def save_reports(self, output_dir: PathLike) -> Dict[str, Path]:
        """Write the confidence report and field weights as JSON."""
        directory = Path(output_dir)
        written = {
            "confidence_report": write_json(
                self.report.to_dict(), directory / "stage2_confidence_report.json"
            ),
            "field_weights": write_json(
                self.field_weights.to_dict(), directory / "stage2_field_weights.json"
            ),
        }
        LOGGER.info("Wrote %d Stage 2 report(s) to %s", len(written), directory)
        return written


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ConfidenceModel:
    """The Stage 2 confidence engine (Stage2.md sec.6.2).

    Example:
        >>> model = ConfidenceModel()                      # doctest: +SKIP
        >>> result = model.score(corpus)                   # doctest: +SKIP
        >>> corpus.records["confidence"].mean()            # doctest: +SKIP
    """

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        config: Optional[ConfidenceConfig] = None,
        schema: Schema = SCHEMA,
    ) -> None:
        """Build a model.

        Args:
            weights: ``(w_comp, w_temp, w_recon)``. Ignored when ``config`` is
                supplied.
            config: Full calibration configuration.
            schema: Schema supplying the default field basis.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if config is not None:
            self.config = config
        elif weights is not None:
            self.config = ConfidenceConfig(weights=tuple(float(w) for w in weights))
        else:
            self.config = ConfidenceConfig()
        self.schema = schema

    def __repr__(self) -> str:
        return f"<ConfidenceModel weights={self.config.weights} {STAGE2_VERSION}>"

    # -- component computation -------------------------------------------

    def _frame_of(self, source: Union["Corpus", pd.DataFrame]) -> pd.DataFrame:
        """Accept either a Corpus or a bare frame."""
        if isinstance(source, pd.DataFrame):
            return source
        records = getattr(source, "records", None)
        if isinstance(records, pd.DataFrame):
            return records
        raise TypeError(
            f"Expected a Corpus or DataFrame, got {type(source).__name__}"
        )

    def score(
        self,
        source: Union["Corpus", pd.DataFrame],
        field_weights: Optional[FieldWeights] = None,
    ) -> ConfidenceResult:
        """Score every record in a corpus.

        Args:
            source: A :class:`~src.stage1.corpus.Corpus` or its ``records``
                frame.
            field_weights: Frozen weights from an earlier corpus. Supply these
                to keep scores comparable across batches - ``v_f`` is estimated
                from the corpus, so recomputing it on a different corpus shifts
                every record's ``C_comp``.

        Returns:
            A :class:`ConfidenceResult`. Row count, order and index are
            preserved exactly.
        """
        frame = self._frame_of(source)
        config = self.config

        completeness = compute_completeness_result(
            frame,
            weights=field_weights,
            fields=config.completeness_fields or tuple(self.schema.names),
            credit=config.completeness_credit,
            normalization=config.entropy_normalization,
            min_coverage=config.min_field_coverage,
            schema=self.schema,
            weight_mode=config.completeness_weight_mode,
            criticality=config.field_criticality,
            critical_fields=config.critical_fields,
            cluster_delta=config.cluster_delta,
            cluster_allowance=config.cluster_allowance,
        )
        temporal = compute_temporal_result(
            frame,
            kappa=config.kappa,
            missing_pair_credit=config.missing_pair_credit,
            hard_fail_on_future=config.hard_fail_on_future,
            pairs=config.ordered_date_pairs,
            schema=self.schema,
        )
        reconciliation = compute_reconciliation_result(
            frame,
            lam=config.lam,
            epsilon=config.epsilon,
            one_sided_credit=config.one_sided_credit,
            both_null_credit=config.both_null_credit,
            non_finite_credit=config.non_finite_credit,
            normalization=config.recon_normalization,
            pair=config.reconciliation_pair,
            mode=config.recon_mode,
            underspend_floor=config.underspend_floor,
            underspend_gamma=config.underspend_gamma,
            non_positive_sanction_credit=config.non_positive_sanction_credit,
            overspend_tolerance=config.overspend_tolerance,
            implausible_credit=config.implausible_credit,
            implausible_threshold=config.implausible_threshold,
            status_field=config.status_field,
            pre_completion_statuses=config.pre_completion_statuses,
            terminal_statuses=config.terminal_statuses,
            unknown_status_gamma_scale=config.unknown_status_gamma_scale,
        )

        scores_array = log_space_geometric_mean(
            [
                completeness.scores.to_numpy(dtype="float64"),
                temporal.scores.to_numpy(dtype="float64"),
                reconciliation.scores.to_numpy(dtype="float64"),
            ],
            config.weights,
            defined=[
                completeness.defined.to_numpy(dtype=bool),
                temporal.defined.to_numpy(dtype=bool),
                reconciliation.defined.to_numpy(dtype=bool),
            ]
            if config.drop_undefined_components
            else None,
        )

        # Post-conditions. These are assertions, not validation: a violation
        # here is a defect in this module, and the fail conditions for Stage 2
        # name NaN/inf output explicitly.
        assert np.all(np.isfinite(scores_array)), "confidence contains NaN or inf"
        assert np.all((scores_array >= 0.0) & (scores_array <= 1.0)), (
            "confidence escaped [0,1]"
        )

        scores = pd.Series(
            scores_array, index=frame.index, dtype="float64", name=CONFIDENCE_COLUMN
        )
        report = self._build_report(scores, completeness, temporal, reconciliation)

        LOGGER.info(
            "Scored %d record(s): mean C=%.4f, median=%.4f, C<%.1f=%.2f%%, "
            "C>%.1f=%.2f%%, exact zeros=%.2f%%.",
            len(scores),
            report.mean_confidence,
            report.median_confidence,
            config.low_threshold,
            report.low_confidence_pct,
            config.high_threshold,
            report.high_confidence_pct,
            report.zero_confidence_pct,
        )

        return ConfidenceResult(
            scores=scores,
            completeness=completeness,
            temporal=temporal,
            reconciliation=reconciliation,
            report=report,
            config=config,
        )

    # -- reporting ---------------------------------------------------------

    def _build_report(
        self,
        scores: pd.Series,
        completeness: CompletenessResult,
        temporal: TemporalResult,
        reconciliation: ReconciliationResult,
    ) -> ConfidenceReport:
        """Assemble the corpus-level report."""
        config = self.config
        n = len(scores)
        values = scores.to_numpy(dtype="float64")

        def summarise(series: pd.Series) -> Dict[str, float]:
            if not len(series):
                return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "zero_pct": 0.0}
            array = series.to_numpy(dtype="float64")
            return {
                "mean": round(float(array.mean()), 6),
                "median": round(float(np.median(array)), 6),
                "min": round(float(array.min()), 6),
                "max": round(float(array.max()), 6),
                "zero_pct": safe_percentage(int((array <= 0.0).sum()), len(array)),
            }

        histogram: Dict[str, int] = {}
        if n:
            edges = np.linspace(0.0, 1.0, config.histogram_bins + 1)
            counts, _ = np.histogram(values, bins=edges)
            for position, count in enumerate(counts):
                histogram[f"[{edges[position]:.1f},{edges[position + 1]:.1f})"] = int(count)

        return ConfidenceReport(
            mean_confidence=round(float(values.mean()), 6) if n else 0.0,
            low_confidence_pct=safe_percentage(
                int((values < config.low_threshold).sum()), n
            ),
            high_confidence_pct=safe_percentage(
                int((values > config.high_threshold).sum()), n
            ),
            stage2_version=STAGE2_VERSION,
            n_records=n,
            median_confidence=round(float(np.median(values)), 6) if n else 0.0,
            min_confidence=round(float(values.min()), 6) if n else 0.0,
            max_confidence=round(float(values.max()), 6) if n else 0.0,
            std_confidence=round(float(values.std(ddof=0)), 6) if n else 0.0,
            zero_confidence_pct=safe_percentage(int((values <= 0.0).sum()), n),
            histogram=histogram,
            component_summary={
                "completeness": summarise(completeness.scores),
                "temporal": summarise(temporal.scores),
                "reconciliation": summarise(reconciliation.scores),
            },
            components_dropped_pct={
                "temporal": safe_percentage(
                    int((~temporal.defined).sum()), n
                ),
                "reconciliation": safe_percentage(
                    int((~reconciliation.defined).sum()), n
                ),
                "completeness_no_evidence": safe_percentage(
                    int(completeness.no_evidence.sum()), n
                ),
            },
            completeness_diagnostics=completeness.to_dict(),
            temporal_diagnostics=temporal.to_dict(),
            reconciliation_diagnostics=reconciliation.to_dict(),
            config=config.to_dict(),
        )


# ---------------------------------------------------------------------------
# Functional API and integration
# ---------------------------------------------------------------------------


def compute_confidence(
    frame: pd.DataFrame, config: Optional[ConfidenceConfig] = None, **kwargs: Any
) -> pd.Series:
    """Compute ``C(r)`` for every record.

    Args:
        frame: Scoring frame (``corpus.records``).
        config: Calibration configuration; defaults to Stage2.md's values.
        **kwargs: Forwarded to :meth:`ConfidenceModel.score`.

    Returns:
        Float Series in [0, 1], sharing the frame's index and row order.
    """
    return ConfidenceModel(config=config).score(frame, **kwargs).scores


def attach_confidence(
    corpus: "Corpus",
    result: Optional[ConfidenceResult] = None,
    config: Optional[ConfidenceConfig] = None,
    with_components: bool = True,
) -> ConfidenceResult:
    """Attach confidence scores onto a corpus in place (Stage2.md sec.7).

    ``Corpus.records`` returns a live reference to the underlying frame, so
    Stage 1 needs no modification to support this - which matters, because
    Stage 1 is locked.

    Args:
        corpus: The corpus to annotate.
        result: A previously computed result. Computed here when omitted.
        config: Configuration used when ``result`` is omitted.
        with_components: Attach the full :data:`BREAKDOWN_COLUMNS` contract,
            not just the scalar score. Stage 3 needs this; leaving it off
            gives a score with no way to explain it.

    Returns:
        The :class:`ConfidenceResult` that was attached.

    Raises:
        ValueError: If the scores do not align with the corpus index. Silent
            misalignment would attach one record's confidence to another, so
            this is checked rather than trusted.
    """
    frame = corpus.records
    scored = result if result is not None else ConfidenceModel(config=config).score(corpus)

    if len(scored.scores) != len(frame):
        raise ValueError(
            f"Score length {len(scored.scores)} does not match corpus length "
            f"{len(frame)}"
        )
    if not scored.scores.index.equals(frame.index):
        raise ValueError("Score index does not match the corpus index")

    breakdown = scored.breakdown
    if not breakdown.index.equals(frame.index):
        raise ValueError("Breakdown index does not match the corpus index")

    frame[CONFIDENCE_COLUMN] = scored.scores
    if with_components:
        for column in BREAKDOWN_COLUMNS:
            if column == CONFIDENCE_COLUMN:
                continue
            frame[column] = breakdown[column]

    LOGGER.info(
        "Attached confidence and %d breakdown signal(s) to %d record(s); "
        "corpus row order unchanged.",
        len(BREAKDOWN_COLUMNS) - 1 if with_components else 0,
        len(frame),
    )
    return scored


# ---------------------------------------------------------------------------
# Explanation contract
# ---------------------------------------------------------------------------


def _verdict(confidence: float) -> str:
    """Band a confidence value for human reading."""
    if confidence <= 0.0:
        return "refused"
    for threshold, label in EXPLANATION_BANDS:
        if confidence < threshold:
            return label
    return EXPLANATION_BANDS[-1][1]


def explain_confidence(
    records: pd.DataFrame,
    row: Any,
    weights: Tuple[float, float, float] = CONFIDENCE_WEIGHTS,
) -> Dict[str, Any]:
    """Explain one record's confidence from stored outputs alone.

    **This function recomputes nothing.** It reads the columns
    :func:`attach_confidence` wrote and reports what they say. The only
    arithmetic performed is attribution bookkeeping over already-stored scores -
    converting each component score to its penalty in nats so the drivers can be
    ranked - and it never invokes a scorer. If the breakdown columns are absent
    it raises rather than silently scoring, because an explanation derived from
    a fresh computation could disagree with the stored score and would be worse
    than no explanation at all.

    Args:
        records: A corpus frame that has been through
            :func:`attach_confidence` with ``with_components=True``.
        row: Index label of the record to explain.
        weights: The ``(w_comp, w_temp, w_recon)`` used when scoring. Prefer
            :meth:`ConfidenceResult.explain`, which supplies its own.

    Returns:
        A JSON-serialisable dict answering "why is this confidence what it is",
        with per-component scores, effective weights after definedness
        renormalisation, penalty attribution, the evidence behind each
        component, and ordered human-readable reasons.

    Raises:
        KeyError: If ``row`` is not in the frame.
        ValueError: If the breakdown columns are missing.
    """
    absent = [name for name in BREAKDOWN_COLUMNS if name not in records.columns]
    if absent:
        raise ValueError(
            "explain_confidence reads stored outputs and never recomputes; "
            f"the frame is missing {absent!r}. Run attach_confidence(corpus) "
            "with with_components=True first."
        )
    if row not in records.index:
        raise KeyError(f"row {row!r} is not in the frame index")

    record = records.loc[row]
    confidence = float(record[CONFIDENCE_COLUMN])

    defined = {
        "completeness": bool(record["completeness_defined"]),
        "temporal": bool(record["temporal_defined"]),
        "reconciliation": bool(record["reconciliation_defined"]),
    }
    weight_of = dict(zip(COMPONENT_COLUMNS, (float(w) for w in weights)))
    live_total = sum(weight_of[name] for name in COMPONENT_COLUMNS if defined[name])

    evidence: Dict[str, Dict[str, Any]] = {
        "completeness": {
            "valid_fields": int(record["n_valid_fields"]),
            "critical_missing_count": int(record["critical_missing_count"]),
            "critical_deficit": round(float(record["critical_deficit"]), 4),
            "cluster_penalty_factor": round(float(record["cluster_penalty_factor"]), 4),
        },
        "temporal": {
            "pairs_evaluated": int(record["temporal_pairs_evaluated"]),
            "hard_fail": bool(record["temporal_hard_fail"]),
        },
        "reconciliation": {
            "branch": str(record["reconciliation_branch"]),
            "lifecycle_state": str(record["lifecycle_state"]),
            "spend_ratio": None
            if pd.isna(record["spend_ratio"])
            else round(float(record["spend_ratio"]), 4),
        },
    }

    components: Dict[str, Dict[str, Any]] = {}
    penalties: Dict[str, float] = {}
    for name in COMPONENT_COLUMNS:
        score = float(record[name])
        effective = (weight_of[name] / live_total) if (defined[name] and live_total) else 0.0
        if not defined[name]:
            penalty: Optional[float] = None
        elif score <= 0.0:
            penalty = None  # a refusal, not a finite penalty
        else:
            penalty = -float(np.log(score)) * effective
            penalties[name] = penalty
        components[name] = {
            "score": round(score, 6),
            "weight": round(weight_of[name], 6),
            "effective_weight": round(effective, 6),
            "defined": defined[name],
            "refused": bool(defined[name] and score <= 0.0),
            "penalty_nats": None if penalty is None else round(penalty, 6),
            "evidence": evidence[name],
        }

    total_penalty = sum(penalties.values())
    for name, penalty in penalties.items():
        components[name]["share_of_penalty"] = (
            round(penalty / total_penalty, 4) if total_penalty > 0 else 0.0
        )

    refusals = [name for name in COMPONENT_COLUMNS if components[name]["refused"]]
    if refusals:
        primary_driver: Optional[str] = refusals[0]
    elif penalties:
        primary_driver = max(penalties, key=lambda key: penalties[key])
    else:
        primary_driver = None

    return {
        "row": row,
        "work_id": record.get("work_id"),
        "confidence": round(confidence, 6),
        "verdict": _verdict(confidence),
        "aggregation": {
            "method": "weighted geometric mean, evaluated in log space",
            "weights": {name: round(weight_of[name], 6) for name in COMPONENT_COLUMNS},
            "components_used": int(record["n_components_used"]),
            "undefined_components_dropped": [
                name for name in COMPONENT_COLUMNS if not defined[name]
            ],
        },
        "components": components,
        "primary_driver": primary_driver,
        "reasons": _reasons(record, components, defined, refusals),
        "summary": _summary(confidence, primary_driver, refusals),
    }


def _reasons(
    record: pd.Series,
    components: Dict[str, Dict[str, Any]],
    defined: Dict[str, bool],
    refusals: List[str],
) -> List[str]:
    """Render stored state as ordered human-readable reasons.

    Every line is a description of a value already in the record. Nothing here
    consults a scorer or re-derives a component.
    """
    reasons: List[str] = []

    if bool(record["temporal_hard_fail"]):
        reasons.append(
            "REFUSED: temporal coherence failed hard - a milestone date is "
            "unreadable or predates the scheme, so the timeline cannot be "
            "reasoned about."
        )
    branch = str(record["reconciliation_branch"])
    if branch == "non_finite":
        reasons.append(
            "REFUSED: an amount is non-finite, so financial plausibility cannot "
            "be assessed."
        )
    elif branch == "implausible_magnitude":
        reasons.append(
            "REFUSED: an amount exceeds the implausibility threshold and is a "
            "data-entry accident rather than a figure."
        )
    elif branch == "non_positive_sanction":
        reasons.append(
            "Sanctioned amount is not positive, so there is no budget to have "
            "executed against."
        )
    elif branch == "one_null":
        reasons.append(
            "One of the two amounts is absent; the comparison cannot be made, "
            "so a partial-information penalty applies."
        )
    elif branch == "both_null":
        reasons.append(
            "Neither amount is recorded, so reconciliation was dropped from the "
            "mean rather than scored - the absence is priced by completeness."
        )

    ratio = record["spend_ratio"]
    lifecycle = str(record["lifecycle_state"])
    if not pd.isna(ratio):
        value = float(ratio)
        if value > 1.0:
            reasons.append(
                f"Expenditure is {value:.2f}x the sanctioned amount; overspend "
                "is penalised at every lifecycle stage."
            )
        elif lifecycle == "pre_completion":
            reasons.append(
                f"Expenditure is {value:.2f}x sanction, but the work is "
                "pre-completion, so low spend is not penalised."
            )
        elif value < 0.2:
            reasons.append(
                f"Expenditure is only {value:.2f}x sanction on a "
                f"{lifecycle} work, which the money does not support."
            )

    missing = int(record["critical_missing_count"])
    if missing:
        reasons.append(
            f"{missing} critical field(s) - dates and amounts - lack usable "
            "evidence."
        )
    cluster = float(record["cluster_penalty_factor"])
    if cluster < 1.0:
        reasons.append(
            f"Critical losses cluster: an extra x{cluster:.2f} decay applies "
            "because evidence loss is super-additive."
        )
    if int(record["temporal_pairs_evaluated"]) == 0 and not defined["temporal"]:
        reasons.append(
            "No milestone pair could be checked, so temporal coherence was "
            "dropped from the mean rather than scored as perfect."
        )

    if not reasons:
        reasons.append(
            "No defect detected: every component was measurable and scored at "
            "or near its maximum."
        )
    return reasons


def _summary(
    confidence: float, primary_driver: Optional[str], refusals: List[str]
) -> str:
    """One-sentence verdict."""
    if refusals:
        return (
            f"Confidence refused (0.0): {refusals[0]} could not be evidenced at "
            "all, and no other component can compensate for it."
        )
    verdict = _verdict(confidence)
    if primary_driver is None:
        return f"Confidence {confidence:.3f} ({verdict}); no component was penalised."
    return (
        f"Confidence {confidence:.3f} ({verdict}); {primary_driver} is the "
        "largest single reason it is not higher."
    )


def confidence_summary_frame(result: ConfidenceResult) -> pd.DataFrame:
    """Tabulate the confidence and component distributions for printing."""
    rows: List[Dict[str, Any]] = []
    series_map: Iterable[Tuple[str, pd.Series]] = (
        ("confidence", result.scores),
        ("completeness", result.completeness.scores),
        ("temporal", result.temporal.scores),
        ("reconciliation", result.reconciliation.scores),
    )
    for name, series in series_map:
        if not len(series):
            rows.append({"metric": name})
            continue
        array = series.to_numpy(dtype="float64")
        rows.append(
            {
                "metric": name,
                "mean": round(float(array.mean()), 4),
                "median": round(float(np.median(array)), 4),
                "std": round(float(array.std(ddof=0)), 4),
                "min": round(float(array.min()), 4),
                "p05": round(float(np.percentile(array, 5)), 4),
                "p95": round(float(np.percentile(array, 95)), 4),
                "max": round(float(array.max()), 4),
                "pct_zero": safe_percentage(int((array <= 0.0).sum()), len(array)),
                "pct_lt_0.2": safe_percentage(int((array < 0.2).sum()), len(array)),
                "pct_gt_0.8": safe_percentage(int((array > 0.8).sum()), len(array)),
            }
        )
    return pd.DataFrame(rows).set_index("metric")
