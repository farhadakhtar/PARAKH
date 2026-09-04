"""C_comp - evidentiary completeness (Stage2.md sec.5.2).

    C_comp(r) = sum_f v_f * kappa(rho_f(r)) / sum_f v_f
    v_f       = (1 - H_null(f)) * H_value(f)

What this actually measures
---------------------------
Not "how many fields are filled in". It measures **surprisal-weighted**
incompleteness: is this record unusually incomplete *relative to the corpus's
own filing habits*?

The ``(1 - H_null)`` term is the device that makes that true, and it is the
subtlest thing in Stage 2. A field that is null 26% of the time has a
high-entropy - unpredictable - fill pattern, so its absence from any one record
tells you almost nothing: absence is simply how that register behaves. A field
that is normally always filled being absent *is* a strong signal.

This is deliberate, and it is the README's central thesis applied to the
confidence layer. A completeness score that punished records for lacking a
field nobody in that district ever fills would be encoding **administrative
capacity**, which is precisely the artifact PARAKH exists not to learn.

The corollary is that C_comp has a narrow dynamic range with a structural floor
well above zero, so theta_C must be calibrated against the empirical
distribution and never against an absolute intuition such as "0.5 means half
the data is there". :meth:`FieldWeights.structural_floor` reports that floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    COMPLETENESS_CREDIT,
    COMPLETENESS_REQUIRE_EVIDENCE,
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
) -> FieldWeights:
    """Estimate ``v_f = (1 - H_null(f)) * H_value(f)`` across the corpus.

    Args:
        frame: Scoring frame (``corpus.records``).
        fields: Field basis ``F``. Defaults to every schema field.
        normalization: How ``H_value`` is scaled into [0, 1].
        min_coverage: Fields present in a smaller share of records are dropped.
        schema: Schema supplying the default field basis.

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
    basis = list(fields) if fields is not None else list(schema.names)
    n_records = len(frame)
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
) -> np.ndarray:
    """Map every cell's null reason onto its evidentiary credit.

    Uses the categorical codes of Stage 1's null-reason columns, so the lookup
    is a single array index rather than a per-cell Python call.

    Args:
        frame: Scoring frame.
        fields: Field basis, defining column order of the result.
        credit: ``NullReason.value -> credit`` mapping.

    Returns:
        ``(n_records, len(fields))`` float array of credits in [0, 1].
    """
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

    Returns:
        :class:`CompletenessResult` whose ``scores`` share the frame's index.
    """
    basis = tuple(fields) if fields is not None else tuple(schema.names)
    resolved_weights = weights or compute_field_weights(
        frame,
        fields=basis,
        normalization=normalization,
        min_coverage=min_coverage,
        schema=schema,
    )
    _, derived = resolve_reasons(frame, basis)

    weight_vector = resolved_weights.as_array(basis)
    total_weight = float(weight_vector.sum())
    credits = credit_matrix(frame, basis, credit=credit)

    no_evidence = pd.Series(False, index=frame.index, dtype=bool)

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
