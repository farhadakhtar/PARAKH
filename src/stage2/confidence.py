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
    COMPLETENESS_CREDIT,
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
        """Per-record confidence with its components and evidence base."""
        return pd.DataFrame(
            {
                CONFIDENCE_COLUMN: self.scores,
                "completeness": self.completeness.scores,
                "temporal": self.temporal.scores,
                "reconciliation": self.reconciliation.scores,
                "temporal_pairs_evaluated": self.temporal.pairs_evaluated,
                "temporal_hard_fail": self.temporal.hard_fail,
                "temporal_defined": self.temporal.defined,
                "reconciliation_branch": self.reconciliation.branch,
                "reconciliation_defined": self.reconciliation.defined,
                "n_valid_fields": self.completeness.n_valid_fields,
            },
            index=self.scores.index,
        )

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
        with_components: Also attach the three component columns and the
            temporal evidence base.

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

    frame[CONFIDENCE_COLUMN] = scored.scores
    if with_components:
        frame["completeness"] = scored.completeness.scores
        frame["temporal"] = scored.temporal.scores
        frame["reconciliation"] = scored.reconciliation.scores
        frame["temporal_pairs_evaluated"] = scored.temporal.pairs_evaluated

    LOGGER.info(
        "Attached confidence to %d record(s); corpus row order unchanged.", len(frame)
    )
    return scored


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
