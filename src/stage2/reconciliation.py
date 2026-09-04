"""C_recon - cross-assertion reconciliation (Stage2.md sec.5.4).

    C_recon(r) = exp(-lambda * |x1 - x2| / (|x1| + |x2| + eps))

Measures whether a record's two independently-reported money figures agree.
Symmetric normalisation is what makes the statistic usable: the disagreement
ratio is bounded in [0, 1], scale-free (rupees or lakhs give the same answer),
well-defined at zero, and sign-safe. C_recon is therefore bounded in
[exp(-lambda), 1] and can never be driven to zero by arithmetic alone - only by
the explicit non-finite hard fail below.

Two honest caveats, neither of which is a bug in this module:

**Semantics.** ``sanction_amount`` is a budget and ``amount_spent`` is an
outcome. They are not two measurements of one quantity, so a legitimate
underspend is charged here as though it were a data contradiction. At
lambda = 2.0 a 30% underspend costs roughly 30% of C_recon. The schema exposes
no genuine second source, so the spec's pair is implemented as written; lambda
belongs in the calibration set, not at a default.

**The one-sided branch.** Exactly one amount present fires on roughly 28% of a
realistically dirty corpus and caps those records at 0.2^(1/3) = 0.585 whatever
else is true of them. Stage2.md words the value as "e.g. 0.2" - a suggestion,
not a derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    RECON_BOTH_NULL_CREDIT,
    RECON_EPSILON,
    RECON_LAMBDA,
    RECON_NON_FINITE_CREDIT,
    RECON_NORMALIZATION,
    RECON_NORMALIZATIONS,
    RECON_ONE_SIDED_CREDIT,
    RECONCILIATION_PAIR,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ReconciliationResult:
    """Per-record reconciliation plus which branch each record took."""

    scores: pd.Series
    #: Per-record branch label: both_present / one_null / both_null / non_finite.
    branch: pd.Series
    #: Normalised disagreement, defined only on the both_present branch.
    ratio: pd.Series
    diagnostics: Dict[str, Any]

    @property
    def defined(self) -> pd.Series:
        """Whether a reconciliation could be attempted for each record.

        Undefined only on the ``both_null`` branch, where neither amount is
        asserted so nothing can agree or disagree. Stage2.md sec.5.4 says to
        "ignore component" in that case - and in a weighted geometric mean,
        *ignoring* a component means dropping it and renormalising the weights,
        not substituting 1.0, which would instead assert perfect agreement on
        no data.
        """
        return (self.branch != "both_null").astype(bool)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable diagnostics."""
        scores = self.scores
        n = len(scores)
        counts = self.branch.value_counts().to_dict() if n else {}
        return {
            "mean": round(float(scores.mean()), 6) if n else 0.0,
            "min": round(float(scores.min()), 6) if n else 0.0,
            "branch_counts": {str(k): int(v) for k, v in counts.items()},
            "branch_pct": {
                str(k): round(100.0 * int(v) / n, 4) for k, v in counts.items()
            }
            if n
            else {},
            "mean_ratio_both_present": round(float(self.ratio.mean(skipna=True)), 6)
            if n and self.ratio.notna().any()
            else 0.0,
            **self.diagnostics,
        }


def compute_reconciliation_result(
    frame: pd.DataFrame,
    lam: float = RECON_LAMBDA,
    epsilon: float = RECON_EPSILON,
    one_sided_credit: float = RECON_ONE_SIDED_CREDIT,
    both_null_credit: float = RECON_BOTH_NULL_CREDIT,
    non_finite_credit: float = RECON_NON_FINITE_CREDIT,
    normalization: str = RECON_NORMALIZATION,
    pair: Tuple[str, str] = RECONCILIATION_PAIR,
) -> ReconciliationResult:
    """Compute ``C_recon`` for every record, with full diagnostics.

    Four mutually exclusive branches:

    * **both non-finite-free and present** - ``exp(-lambda * ratio)``.
    * **exactly one null** - ``one_sided_credit``. One side of the comparison
      is simply unavailable; the assertion cannot be checked at all.
    * **both null** - ``both_null_credit`` (1.0). Nothing is asserted, so
      nothing can contradict. ``C_comp`` prices the absence.
    * **either value non-finite** - ``non_finite_credit`` (0.0), explicitly.
      This branch is not optional: with ``x1 = inf`` the symmetric ratio
      evaluates ``inf/inf = NaN``, which would propagate silently through the
      log-sum and poison the final score rather than lowering it.

    Args:
        frame: Scoring frame (``corpus.records``).
        lam: Disagreement penalty rate.
        epsilon: Denominator stabiliser; makes 0-vs-0 well defined.
        one_sided_credit: Score when exactly one amount is present.
        both_null_credit: Score when neither amount is present.
        non_finite_credit: Score when either amount is inf.
        normalization: ``"symmetric"`` (default, per Stage2.md sec.5.4 and
            README) or ``"max"``. See
            :data:`~src.core.constants.RECON_NORMALIZATION` for why symmetric
            is the default - ``max`` is not sign-safe against the negative
            amounts Stage 1 deliberately injects.
        pair: The two columns to reconcile.

    Returns:
        :class:`ReconciliationResult` whose ``scores`` share the frame's index.

    Raises:
        ValueError: On a negative ``lambda``, a non-positive ``epsilon``, an
            out-of-range credit, an unknown normalisation, or a missing column.
    """
    if lam < 0.0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if normalization not in RECON_NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {RECON_NORMALIZATIONS}, got {normalization!r}"
        )
    for label, value in (
        ("one_sided_credit", one_sided_credit),
        ("both_null_credit", both_null_credit),
        ("non_finite_credit", non_finite_credit),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must lie in [0,1], got {value}")

    left_name, right_name = pair
    missing_columns = [name for name in pair if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"Reconciliation columns absent from frame: {missing_columns}")

    index = frame.index
    n_records = len(frame)
    if n_records == 0:
        return ReconciliationResult(
            scores=pd.Series([], dtype="float64", index=index, name="reconciliation"),
            branch=pd.Series([], dtype="object", index=index),
            ratio=pd.Series([], dtype="float64", index=index),
            diagnostics={"lambda": lam, "normalization": normalization},
        )

    x1 = frame[left_name].to_numpy(dtype="float64", na_value=np.nan)
    x2 = frame[right_name].to_numpy(dtype="float64", na_value=np.nan)

    null_1 = np.isnan(x1)
    null_2 = np.isnan(x2)
    both_null = null_1 & null_2
    one_null = null_1 ^ null_2
    both_present = ~null_1 & ~null_2
    non_finite = both_present & (~np.isfinite(x1) | ~np.isfinite(x2))
    comparable = both_present & ~non_finite

    # Evaluate the ratio only where it is defined. Substituting neutral values
    # elsewhere keeps inf/inf and 0/0 out of the arithmetic entirely, so no
    # warning is raised and no NaN can be produced to mask later.
    safe_1 = np.where(comparable, x1, 0.0)
    safe_2 = np.where(comparable, x2, 0.0)
    numerator = np.abs(safe_1 - safe_2)
    if normalization == "symmetric":
        denominator = np.abs(safe_1) + np.abs(safe_2) + epsilon
    else:  # "max"
        denominator = np.maximum(np.maximum(safe_1, safe_2), epsilon)

    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros(n_records, dtype="float64"),
        where=denominator > 0.0,
    )
    ratio = np.where(comparable, ratio, 0.0)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio = np.maximum(ratio, 0.0)

    scores = np.full(n_records, both_null_credit, dtype="float64")
    scores = np.where(one_null, one_sided_credit, scores)
    scores = np.where(non_finite, non_finite_credit, scores)
    scores = np.where(comparable, np.exp(-lam * ratio), scores)
    scores = np.clip(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    branch = np.full(n_records, "both_present", dtype=object)
    branch[one_null] = "one_null"
    branch[both_null] = "both_null"
    branch[non_finite] = "non_finite"

    if int(non_finite.sum()):
        LOGGER.info(
            "%d record(s) carry a non-finite amount; C_recon forced to %.2f.",
            int(non_finite.sum()),
            non_finite_credit,
        )

    ratio_series = pd.Series(
        np.where(comparable, ratio, np.nan), index=index, dtype="float64"
    )

    return ReconciliationResult(
        scores=pd.Series(scores, index=index, dtype="float64", name="reconciliation"),
        branch=pd.Series(branch, index=index, dtype="object"),
        ratio=ratio_series,
        diagnostics={
            "lambda": lam,
            "epsilon": epsilon,
            "normalization": normalization,
            "one_sided_credit": one_sided_credit,
            "both_null_credit": both_null_credit,
            "non_finite_credit": non_finite_credit,
            "pair": list(pair),
            "theoretical_floor": round(float(np.exp(-lam)), 6),
        },
    )


def compute_reconciliation(frame: pd.DataFrame, **kwargs: Any) -> pd.Series:
    """Compute ``C_recon`` for every record.

    Args:
        frame: Scoring frame (``corpus.records``).
        **kwargs: Forwarded to :func:`compute_reconciliation_result`.

    Returns:
        Float Series in [0, 1], sharing the frame's index and row order.
    """
    return compute_reconciliation_result(frame, **kwargs).scores
