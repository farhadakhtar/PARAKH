"""C_comp - evidentiary completeness (Stage2.md sec.5.2, refined in v2).

    C_comp(r) = [ sum_f v_f * kappa(rho_f(r)) / sum_f v_f ] * cluster(r)
    v_f       = criticality_f * H_value(f)
    cluster(r)= exp(-delta * max(0, m - 1)),  m = sum over critical f of
                                                 (1 - kappa(rho_f))

What changed, and why
---------------------
v1 weighted fields by ``(1 - H_null(f)) * H_value(f)``, a surprisal argument:
an absence is informative only if absences are rare. Measured on the 20k
corpus, that produced a badly distorted basis:

    work_id (never null, proves nothing)   18.11% of all weight
    all three dates + both amounts         30.56% between them
    vendor_name (missing 26% of the time)   2.12%

and an algebraic floor of 0.3449 = 0.1811 (work_id) + 0.20 * 0.8189 (the
missing-credit), against an observed range of [0.5150, 1.0] with sd 0.0670.
C_comp contributed only 1.77% of the variance of log C: it was very nearly a
constant.

Two faults, one fix each:

* **The identifier dominated.** Stage 1 guarantees ``work_id`` is never null,
  so whatever weight it holds is an identical constant added to every record -
  pure range compression, zero discriminating power. Criticality puts it at
  0.05, alongside the other descriptive fields.
* **The evidentiary spine was starved.** ``(1 - H_null)`` systematically
  down-weighted exactly the fields most likely to be absent. Criticality states
  the domain judgement directly: dates and money are what evidence a public
  work, and they now hold 72% of the basis.

On the artifact-invariance question
-----------------------------------
v1 defended ``(1 - H_null)`` as preventing the score from encoding
administrative capacity. That defence was misapplied. README sec.9 places the
artifact-invariance guarantee on **R**, not **C**, and a low-confidence record
routes to REMEDIATE rather than INVESTIGATE. Confidence is *supposed* to track
documentation quality - that is what it measures - and suppressing it solved a
problem the routing layer already solves, at the cost of making the component
nearly inert. ``H_value`` is retained because it correctly zeroes constant,
non-informative fields and keeps the degenerate-corpus fallback working; v1's
weighting remains available verbatim via ``weight_mode="entropy"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    CLUSTER_PENALTY_ALLOWANCE,
    CLUSTER_PENALTY_DELTA,
    COMPLETENESS_CREDIT,
    COMPLETENESS_REQUIRE_EVIDENCE,
    COMPLETENESS_WEIGHT_MODE,
    COMPLETENESS_WEIGHT_MODES,
    CRITICAL_FIELDS,
    FIELD_CRITICALITY,
    ENTROPY_NORMALIZATION,
    ENTROPY_NORMALIZATIONS,
    MIN_FIELD_COVERAGE,
)
from src.core.logger import get_logger
from src.stage1.schema import SCHEMA, NullReason, Schema, null_reason_column

LOGGER = get_logger(__name__)

#: Category order used by Stage 1's null-reason columns.
REASON_ORDER: Tuple[str, ...] = tuple(reason.value for reason in NullReason)


# ---------------------------------------------------------------------------
# Entropy primitives
# ---------------------------------------------------------------------------


def bernoulli_entropy(p: float) -> float:
    """Binary Shannon entropy in bits, natively bounded to [0, 1].

    Args:
        p: Probability of the positive outcome.

    Returns:
        ``-p log2 p - (1-p) log2(1-p)``, and 0.0 at the degenerate endpoints
        where the limit is 0 but the expression is undefined.
    """
    if not math.isfinite(p) or p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def value_entropy(
    values: pd.Series, normalization: str = ENTROPY_NORMALIZATION
) -> Tuple[float, int, float]:
    """Normalised Shannon entropy of a field's observed values.

    Args:
        values: The **present** values of one field (nulls already removed).
        normalization: ``"cardinality"`` divides by ``log2(k)`` and is
            scale-invariant; ``"sample"`` divides by ``log2(n)`` and is more
            discriminative but corpus-size dependent.

    Returns:
        ``(normalised_entropy, n_distinct, raw_entropy_bits)``. A constant or
        empty field returns ``(0.0, k, 0.0)`` - it carries no information, so
        its presence proves nothing.

    Raises:
        ValueError: If ``normalization`` is not a known mode.
    """
    if normalization not in ENTROPY_NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {ENTROPY_NORMALIZATIONS}, "
            f"got {normalization!r}"
        )

    n_present = int(len(values))
    if n_present == 0:
        return 0.0, 0, 0.0

    counts = values.value_counts(dropna=True)
    k = int(len(counts))
    if k <= 1:
        return 0.0, k, 0.0

    probabilities = counts.to_numpy(dtype="float64")
    probabilities /= probabilities.sum()
    raw_bits = float(-(probabilities * np.log2(probabilities)).sum())

    denominator = (
        math.log2(k) if normalization == "cardinality" else math.log2(max(n_present, 2))
    )
    if denominator <= 0.0:
        return 0.0, k, raw_bits
    return float(min(max(raw_bits / denominator, 0.0), 1.0)), k, raw_bits


# ---------------------------------------------------------------------------
# Field weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldWeights:
    """Corpus-level field weights ``v_f``, frozen for auditability.

    ``v_f`` is estimated from the corpus, so ``C_comp(r)`` is deterministic
    *given a fixed corpus* but shifts if the corpus changes. Two audits of the
    same record would then disagree. These weights are therefore emitted with
    every score set and must be persisted alongside them.
    """

    weights: Mapping[str, float]
    h_null: Mapping[str, float]
    h_value: Mapping[str, float]
    coverage: Mapping[str, float]
    n_distinct: Mapping[str, int]
    normalization: str
    criticality: Mapping[str, float] = field(default_factory=dict)
    weight_mode: str = COMPLETENESS_WEIGHT_MODE
    excluded_fields: Tuple[str, ...] = ()
    degenerate: bool = False
    n_records: int = 0

    @property
    def total(self) -> float:
        """Sum of all weights - the denominator of ``C_comp``."""
        return float(sum(self.weights.values()))

    @property
    def shares(self) -> Dict[str, float]:
        """Each field's fraction of the total weight."""
        total = self.total
        if total <= 0.0:
            return {name: 0.0 for name in self.weights}
        return {name: value / total for name, value in self.weights.items()}

    @property
    def structural_floor(self) -> float:
        """Lowest ``C_comp`` reachable if every non-guaranteed field fails.

        Fields that are never null contribute an identical constant to every
        record, so they raise the floor without discriminating between any two
        records. Reporting this makes the compression visible rather than
        hidden.
        """
        total = self.total
        if total <= 0.0:
            return 0.0
        guaranteed = sum(
            weight
            for name, weight in self.weights.items()
            if self.coverage.get(name, 0.0) >= 1.0
        )
        return float(guaranteed / total)

    def as_array(self, fields: Sequence[str]) -> np.ndarray:
        """Weights as a float array aligned to ``fields``."""
        return np.asarray([self.weights.get(name, 0.0) for name in fields], dtype="float64")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable description."""
        return {
            "normalization": self.normalization,
            "weight_mode": self.weight_mode,
            "degenerate": self.degenerate,
            "excluded_fields": list(self.excluded_fields),
            "total_weight": round(self.total, 6),
            "structural_floor": round(self.structural_floor, 6),
            "n_records": self.n_records,
            "per_field": {
                name: {
                    "v": round(self.weights[name], 6),
                    "share": round(self.shares[name], 6),
                    "h_null": round(self.h_null.get(name, 0.0), 6),
                    "h_value": round(self.h_value.get(name, 0.0), 6),
                    "coverage": round(self.coverage.get(name, 0.0), 6),
                    "criticality": round(self.criticality.get(name, 0.0), 6),
                    "n_distinct": int(self.n_distinct.get(name, 0)),
                }
                for name in self.weights
            },
        }


def resolve_reasons(
    frame: pd.DataFrame, fields: Sequence[str]
) -> Tuple[pd.DataFrame, bool]:
    """Recover the per-cell null reason for each field.

    Prefers Stage 1's ``null_reason__*`` columns. When they are absent - for
    instance on a bare frame that never went through a ``Corpus`` - it degrades
    to deriving reasons from ``isna()``, in which case every null is reported as
    ``missing`` because the finer taxonomy simply is not recoverable.

    Args:
        frame: Scoring frame.
        fields: Fields to resolve.

    Returns:
        ``(reasons, derived)`` where ``reasons`` holds one string column per
        field and ``derived`` says whether the fallback was used.
    """
    have_all = all(null_reason_column(name) in frame.columns for name in fields)
    if have_all:
        return (
            pd.DataFrame(
                {name: frame[null_reason_column(name)] for name in fields},
                index=frame.index,
            ),
            False,
        )

    LOGGER.warning(
        "null_reason__* columns are absent; deriving reasons from isna(). "
        "The missing/placeholder/unparseable distinction is NOT recoverable "
        "this way and every null will be scored as 'missing'."
    )
    derived = {}
    for name in fields:
        null_mask = frame[name].isna() if name in frame.columns else pd.Series(
            True, index=frame.index
        )
        derived[name] = np.where(
            null_mask.to_numpy(), NullReason.MISSING.value, NullReason.PRESENT.value
        )
    return pd.DataFrame(derived, index=frame.index), True


def compute_field_weights(
    frame: pd.DataFrame,
    fields: Optional[Sequence[str]] = None,
    normalization: str = ENTROPY_NORMALIZATION,
    min_coverage: float = MIN_FIELD_COVERAGE,
    schema: Schema = SCHEMA,
    weight_mode: str = COMPLETENESS_WEIGHT_MODE,
    criticality: Mapping[str, float] = FIELD_CRITICALITY,
    reasons: Optional[pd.DataFrame] = None,
) -> FieldWeights:
    """Estimate ``v_f = (1 - H_null(f)) * H_value(f)`` across the corpus.

    Args:
        frame: Scoring frame (``corpus.records``).
        fields: Field basis ``F``. Defaults to every schema field.
        normalization: How ``H_value`` is scaled into [0, 1].
        min_coverage: Fields present in a smaller share of records are dropped.
        schema: Schema supplying the default field basis.
        weight_mode: ``"criticality"`` (default) uses
            ``criticality_f * H_value(f)``; ``"entropy"`` reproduces v1's
            ``(1 - H_null(f)) * H_value(f)``; ``"hybrid"`` multiplies all
            three factors.
        criticality: Per-field domain weight, used by the criticality and
            hybrid modes.
        reasons: Precomputed null-reason frame, to avoid a redundant pass.

    Returns:
        Frozen :class:`FieldWeights`.

    Note:
        If every weight comes out zero - a corpus where no field varies, which
        is the mandated "identical values (no variance)" edge case - the
        weighting **falls back to uniform** and sets ``degenerate=True``.
        Without that fallback a corpus of identical *perfect* records would
        compute 0/0 and score zero confidence, which is plainly wrong. Entropy
        weighting is a refinement; when it degenerates, fall back rather than
        emit a meaningless number.
    """
    if weight_mode not in COMPLETENESS_WEIGHT_MODES:
        raise ValueError(
            f"weight_mode must be one of {COMPLETENESS_WEIGHT_MODES}, "
            f"got {weight_mode!r}"
        )
    basis = list(fields) if fields is not None else list(schema.names)
    n_records = len(frame)
    if reasons is None:
        reasons, _ = resolve_reasons(frame, basis)

    weights: Dict[str, float] = {}
    h_null: Dict[str, float] = {}
    h_value: Dict[str, float] = {}
    coverage: Dict[str, float] = {}
    n_distinct: Dict[str, int] = {}
    excluded: list[str] = []

    for name in basis:
        present_mask = (
            reasons[name].astype("object") == NullReason.PRESENT.value
        ).to_numpy()
        present_share = float(present_mask.mean()) if n_records else 0.0
        coverage[name] = present_share

        entropy_null = bernoulli_entropy(1.0 - present_share)
        h_null[name] = entropy_null

        values = frame[name].loc[present_mask] if name in frame.columns else pd.Series(
            [], dtype="object"
        )
        entropy_value, distinct, _ = value_entropy(values, normalization=normalization)
        h_value[name] = entropy_value
        n_distinct[name] = distinct

        if present_share < min_coverage:
            excluded.append(name)
            weights[name] = 0.0
            continue

        # v1 used (1 - H_null) alone, which down-weighted precisely the
        # fields most likely to be absent: the three dates and two amounts
        # held 30.56% of total weight while work_id - never null, and so
        # incapable of discriminating between any two records - held
        # 18.11%. Criticality states the same judgement directly instead of
        # inferring a proxy for it from the null pattern.
        crit = float(criticality.get(name, 0.0))
        if weight_mode == "criticality":
            weights[name] = crit * entropy_value
        elif weight_mode == "hybrid":
            weights[name] = crit * (1.0 - entropy_null) * entropy_value
        else:  # "entropy" - v1 behaviour, retained and reproducible
            weights[name] = (1.0 - entropy_null) * entropy_value

    degenerate = sum(weights.values()) <= 0.0
    if degenerate:
        LOGGER.warning(
            "All entropy weights are zero (no field varies across %d record(s)); "
            "falling back to uniform field weights.",
            n_records,
        )
        weights = {name: 1.0 for name in basis}

    if excluded:
        LOGGER.info(
            "Excluded %d field(s) from the completeness basis for coverage < %.2f%%: %s",
            len(excluded),
            100 * min_coverage,
            excluded,
        )

    return FieldWeights(
        weights=weights,
        h_null=h_null,
        h_value=h_value,
        coverage=coverage,
        n_distinct=n_distinct,
        normalization=normalization,
        criticality={name: float(criticality.get(name, 0.0)) for name in basis},
        weight_mode=weight_mode,
        excluded_fields=tuple(excluded),
        degenerate=degenerate,
        n_records=n_records,
    )


# ---------------------------------------------------------------------------
# Credit matrix and scoring
# ---------------------------------------------------------------------------


def credit_matrix(
    frame: pd.DataFrame,
    fields: Sequence[str],
    credit: Mapping[str, float] = COMPLETENESS_CREDIT,
    reasons: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Map every cell's null reason onto its evidentiary credit.

    Uses the categorical codes of Stage 1's null-reason columns, so the lookup
    is a single array index rather than a per-cell Python call.

    Args:
        frame: Scoring frame.
        fields: Field basis, defining column order of the result.
        credit: ``NullReason.value -> credit`` mapping.
        reasons: Precomputed null-reason frame. Supplying it avoids a
            redundant second pass when the caller already has one.

    Returns:
        ``(n_records, len(fields))`` float array of credits in [0, 1].
    """
    if reasons is None:
        reasons, _ = resolve_reasons(frame, fields)
    n_records = len(frame)
    matrix = np.zeros((n_records, len(fields)), dtype="float64")
    if n_records == 0:
        return matrix

    for position, name in enumerate(fields):
        column = reasons[name]
        if isinstance(column.dtype, pd.CategoricalDtype):
            lookup = np.asarray(
                [float(credit.get(str(c), 0.0)) for c in column.cat.categories],
                dtype="float64",
            )
            codes = column.cat.codes.to_numpy()
            # code -1 marks a cell with no recorded reason; treat as missing.
            values = np.where(
                codes >= 0,
                lookup[np.clip(codes, 0, max(len(lookup) - 1, 0))],
                float(credit.get(NullReason.MISSING.value, 0.0)),
            )
        else:
            values = (
                column.astype("object")
                .map(lambda reason: float(credit.get(str(reason), 0.0)))
                .to_numpy(dtype="float64")
            )
        matrix[:, position] = values

    return matrix


@dataclass(frozen=True)
class CompletenessResult:
    """Per-record completeness plus the weights that produced it."""

    scores: pd.Series
    weights: FieldWeights
    fields: Tuple[str, ...]
    credit: Mapping[str, float] = field(default_factory=lambda: dict(COMPLETENESS_CREDIT))
    reasons_derived: bool = False
    n_valid_fields: pd.Series = field(default_factory=lambda: pd.Series(dtype="int64"))
    #: Rows forced to zero because no field carried any evidence at all.
    no_evidence: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    #: Fractional count of critical fields lacking usable evidence.
    critical_deficit: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="float64")
    )
    #: Multiplicative cluster penalty applied to the weighted average.
    cluster_factor: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="float64")
    )
    #: Integer count of critical fields lacking usable evidence.
    #:
    #: Reported alongside :attr:`critical_deficit` rather than instead of it.
    #: The deficit is FRACTIONAL - it sums ``1 - kappa`` so the
    #: missing/placeholder/unparseable ordering flows into the cluster term -
    #: and is what the formula uses. This count is the human-readable
    #: companion, and the two deliberately disagree: three placeholders are a
    #: count of 3 but a deficit of 2.76.
    critical_missing_count: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="int64")
    )

    @property
    def defined(self) -> pd.Series:
        """C_comp is always defined: the field basis exists for every record."""
        return pd.Series(True, index=self.scores.index, dtype=bool)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable diagnostics."""
        scores = self.scores
        return {
            "credit": dict(self.credit),
            "fields": list(self.fields),
            "reasons_derived_from_isna": self.reasons_derived,
            "weights": self.weights.to_dict(),
            "mean": round(float(scores.mean()), 6) if len(scores) else 0.0,
            "min": round(float(scores.min()), 6) if len(scores) else 0.0,
            "max": round(float(scores.max()), 6) if len(scores) else 0.0,
        }


def compute_completeness_result(
    frame: pd.DataFrame,
    weights: Optional[FieldWeights] = None,
    fields: Optional[Sequence[str]] = None,
    credit: Mapping[str, float] = COMPLETENESS_CREDIT,
    normalization: str = ENTROPY_NORMALIZATION,
    min_coverage: float = MIN_FIELD_COVERAGE,
    require_evidence: bool = COMPLETENESS_REQUIRE_EVIDENCE,
    schema: Schema = SCHEMA,
    weight_mode: str = COMPLETENESS_WEIGHT_MODE,
    criticality: Mapping[str, float] = FIELD_CRITICALITY,
    critical_fields: Sequence[str] = CRITICAL_FIELDS,
    cluster_delta: float = CLUSTER_PENALTY_DELTA,
    cluster_allowance: float = CLUSTER_PENALTY_ALLOWANCE,
) -> CompletenessResult:
    """Compute ``C_comp`` for every record, with full diagnostics.

    Args:
        frame: Scoring frame (``corpus.records``).
        weights: Pre-computed weights. Pass frozen weights to score a new batch
            against an earlier corpus, keeping scores comparable over time.
        fields: Field basis ``F``.
        credit: Null-reason credit vector ``kappa``.
        normalization: ``H_value`` normalisation mode.
        min_coverage: Coverage floor for including a field.
        require_evidence: Force ``C_comp = 0`` when a record has no present
            field at all. See
            :data:`~src.core.constants.COMPLETENESS_REQUIRE_EVIDENCE`.
        schema: Schema supplying the default basis.
        weight_mode: How ``v_f`` is formed.
        criticality: Per-field domain weight.
        critical_fields: Fields counted by the cluster penalty.
        cluster_delta: Cluster decay rate.
        cluster_allowance: Critical deficit tolerated before the cluster
            penalty engages.

    Returns:
        :class:`CompletenessResult` whose ``scores`` share the frame's index.
    """
    basis = tuple(fields) if fields is not None else tuple(schema.names)
    # Single null-reason pass, shared by the weight estimator and the credit
    # matrix. Both previously derived their own copy of the same frame.
    reasons, derived = resolve_reasons(frame, basis)
    resolved_weights = weights or compute_field_weights(
        frame,
        reasons=reasons,
        fields=basis,
        normalization=normalization,
        min_coverage=min_coverage,
        schema=schema,
        weight_mode=weight_mode,
        criticality=criticality,
    )

    weight_vector = resolved_weights.as_array(basis)
    total_weight = float(weight_vector.sum())
    credits = credit_matrix(frame, basis, credit=credit, reasons=reasons)

    no_evidence = pd.Series(False, index=frame.index, dtype=bool)
    critical_deficit = pd.Series(0.0, index=frame.index, dtype="float64")
    cluster_factor = pd.Series(1.0, index=frame.index, dtype="float64")
    critical_missing_count = pd.Series(0, index=frame.index, dtype="int64")

    if len(frame) == 0:
        scores = pd.Series([], dtype="float64", index=frame.index)
        counts = pd.Series([], dtype="int64", index=frame.index)
    elif total_weight <= 0.0:
        # Unreachable in practice: compute_field_weights already falls back to
        # uniform weights. Kept as a guard for externally supplied weights.
        LOGGER.error("Total field weight is zero; C_comp is undefined -> 0.0")
        scores = pd.Series(0.0, index=frame.index, dtype="float64")
        counts = pd.Series(0, index=frame.index, dtype="int64")
    else:
        raw = credits @ weight_vector / total_weight
        present_credit = float(credit.get(NullReason.PRESENT.value, 1.0))
        present_matrix = credits >= present_credit
        counts = pd.Series(
            present_matrix.sum(axis=1), index=frame.index, dtype="int64"
        )

        # --- cluster penalty ------------------------------------------------
        # Evidence loss is super-additive. Losing one milestone date is a gap;
        # losing all three dates and both amounts destroys the record's ability
        # to be cross-checked at all, and a linear weighted average cannot
        # express that. The deficit is FRACTIONAL - built from the same credit
        # vector as the numerator - so the missing < placeholder < unparseable
        # ordering flows into the cluster term as well.
        critical_positions = [
            index for index, name in enumerate(basis) if name in set(critical_fields)
        ]
        if critical_positions:
            critical_missing_count = pd.Series(
                (~present_matrix[:, critical_positions]).sum(axis=1),
                index=frame.index,
                dtype="int64",
            )
        if critical_positions and cluster_delta > 0.0:
            deficit = (1.0 - credits[:, critical_positions]).sum(axis=1)
            critical_deficit = pd.Series(
                deficit, index=frame.index, dtype="float64"
            )
            factor = np.exp(
                -cluster_delta * np.maximum(0.0, deficit - cluster_allowance)
            )
            cluster_factor = pd.Series(
                np.clip(factor, 0.0, 1.0), index=frame.index, dtype="float64"
            )
            raw = raw * cluster_factor.to_numpy()

        if require_evidence:
            # No present field anywhere in the basis -> no evidence base -> the
            # residual credits are a fraction of nothing. Refuse.
            blank = ~present_matrix.any(axis=1)
            no_evidence = pd.Series(blank, index=frame.index, dtype=bool)
            raw = np.where(blank, 0.0, raw)
            if int(blank.sum()):
                LOGGER.info(
                    "%d record(s) carry no present field at all; C_comp forced to 0.",
                    int(blank.sum()),
                )
        scores = pd.Series(
            np.clip(raw, 0.0, 1.0), index=frame.index, dtype="float64", name="completeness"
        )

    return CompletenessResult(
        scores=scores,
        weights=resolved_weights,
        fields=basis,
        credit=dict(credit),
        reasons_derived=derived,
        n_valid_fields=counts,
        no_evidence=no_evidence,
        critical_deficit=critical_deficit,
        cluster_factor=cluster_factor,
        critical_missing_count=critical_missing_count,
    )


def compute_completeness(
    frame: pd.DataFrame,
    weights: Optional[FieldWeights] = None,
    **kwargs: Any,
) -> pd.Series:
    """Compute ``C_comp`` for every record.

    Args:
        frame: Scoring frame (``corpus.records``).
        weights: Optional pre-computed, frozen field weights.
        **kwargs: Forwarded to :func:`compute_completeness_result`.

    Returns:
        Float Series in [0, 1], sharing the frame's index and row order.
    """
    return compute_completeness_result(frame, weights=weights, **kwargs).scores
