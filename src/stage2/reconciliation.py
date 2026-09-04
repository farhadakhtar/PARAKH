"""C_recon - financial plausibility (Stage2.md sec.5.4, refined in v2).

    r       = spent / (sanction + eps)
    C_recon = exp(-lambda * max(0, r - (1 + tau)))      # overspend
            * exp(-gamma * g(status) * max(0, theta_u - r))   # underspend

    tau = 0.05 overspend tolerance;  g(status) gates the underspend penalty
    on lifecycle stage: 0.0 pre-completion, 1.0 terminal, 0.5 unknown.

What changed, and why
---------------------
v1 implemented reconciliation as an EQUALITY test,
``exp(-lambda |x1 - x2| / (|x1| + |x2| + eps))``. That presumes the two amounts
are independent measurements of one quantity. They are not: ``sanction_amount``
is a budget and ``amount_spent`` is an outcome, and spending less than
sanctioned is what correct budget execution looks like.

Measured on the 20k corpus, the error was not marginal:

    0.2 <= r <= 1.0    74.57% of comparable records   mean C_recon 0.8875
    r > 1.0            24.81%                          penalised identically
    r < 0.2             0.62%                          penalised identically

The component charged three quarters of the corpus for behaving normally, and
being symmetric it could not tell an overspend from an underspend of the same
size - discarding the one direction that actually carries signal.

v2 scores PLAUSIBILITY instead, asymmetrically:

* **Overspend** (r > 1) decays at ``lambda``. Spending past the sanction is a
  control failure requiring a revision that should itself be on record.
* **Underspend** is free down to ``theta_u`` (0.2), then decays at ``gamma``.
  A work reported against a sanction while having consumed under a fifth of it
  asserts something the money does not support.

The v1 behaviour remains available verbatim via ``mode="agreement"``, so the
two are comparable on one corpus and the change is reversible.

Lifecycle awareness
-------------------
The ratio's meaning depends on lifecycle stage, and the component now reads
``status`` to account for it. A *proposed* work legitimately has
``amount_spent`` at or near zero; charging it the underspend penalty was
penalising a record for being normal, and was the largest reality-alignment
error in the component.

Only the underspend side is gated. Overspend is penalised at every stage,
because spending past a sanction is a control failure requiring a revision that
should itself be on record.

Garbage is refused rather than discounted: a non-finite amount, or one beyond
``IMPLAUSIBLE_AMOUNT_THRESHOLD``, scores 0.0 and so annihilates the geometric
mean. A moderate score on an unreadable number would let garbage survive
aggregation.

What remains uncalibrated
-------------------------
``lambda``, ``gamma``, ``tau``, ``theta_u`` and the unknown-status scale are
defensible defaults, not estimates. Per README the system is non-operational
until they are fitted to real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    IMPLAUSIBLE_AMOUNT_THRESHOLD,
    RECON_BOTH_NULL_CREDIT,
    RECON_IMPLAUSIBLE_MAGNITUDE_CREDIT,
    RECON_OVERSPEND_TOLERANCE,
    RECON_PRE_COMPLETION_STATUSES,
    RECON_TERMINAL_STATUSES,
    RECON_UNKNOWN_STATUS_GAMMA_SCALE,
    STATUS_FIELD,
    RECON_MODE,
    RECON_MODES,
    RECON_NON_POSITIVE_SANCTION_CREDIT,
    RECON_UNDERSPEND_FLOOR,
    RECON_UNDERSPEND_GAMMA,
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
    #: Under mode='plausibility', the budget execution rate spent/sanction;
    #: under mode='agreement', the v1 normalised disagreement. NaN wherever
    #: the statistic was not computable.
    ratio: pd.Series
    diagnostics: Dict[str, Any]
    #: Lifecycle class driving the underspend gate: pre_completion,
    #: terminal, or unknown.
    lifecycle: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="object")
    )

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
            "mean_ratio": round(float(self.ratio.mean(skipna=True)), 6)
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
    mode: str = RECON_MODE,
    underspend_floor: float = RECON_UNDERSPEND_FLOOR,
    underspend_gamma: float = RECON_UNDERSPEND_GAMMA,
    non_positive_sanction_credit: float = RECON_NON_POSITIVE_SANCTION_CREDIT,
    overspend_tolerance: float = RECON_OVERSPEND_TOLERANCE,
    implausible_credit: float = RECON_IMPLAUSIBLE_MAGNITUDE_CREDIT,
    implausible_threshold: float = IMPLAUSIBLE_AMOUNT_THRESHOLD,
    status_field: str = STATUS_FIELD,
    pre_completion_statuses: Tuple[str, ...] = RECON_PRE_COMPLETION_STATUSES,
    terminal_statuses: Tuple[str, ...] = RECON_TERMINAL_STATUSES,
    unknown_status_gamma_scale: float = RECON_UNKNOWN_STATUS_GAMMA_SCALE,
) -> ReconciliationResult:
    """Compute ``C_recon`` for every record, with full diagnostics.

    Five mutually exclusive branches:

    * **both present, sanction > 0** - the plausibility score above, or the v1
      agreement score under ``mode="agreement"``.
    * **exactly one null** - ``one_sided_credit`` (0.7). One side of the
      comparison is unavailable so the assertion cannot be checked. This is a
      partial-information penalty, not a verdict: at v1's 0.2 it fired on 28%
      of the corpus and manufactured a 4,255-record spike in the output
      distribution.
    * **both null** - ``both_null_credit`` (1.0), and marked *undefined* so the
      aggregator drops it rather than reading it as perfect agreement.
    * **either value non-finite** - ``non_finite_credit`` (0.0). Not optional:
      ``inf/inf`` evaluates to NaN, which would propagate silently through the
      log-sum and poison the score rather than lowering it. Refusal rather
      than discount: garbage must not be able to produce moderate confidence.
    * **either value beyond the implausibility threshold** -
      ``implausible_credit`` (0.0). Finite, so it survives the check above,
      but a 1e300 sanction is a data-entry accident, not a number.
    * **sanction <= 0** - ``non_positive_sanction_credit`` (0.25). There is no
      budget to have executed against, so the ratio is meaningless. This is a
      deliberate behaviour change: under v1's equality reading,
      ``sanction = spent = 0`` scored 1.0.

    Args:
        frame: Scoring frame (``corpus.records``).
        lam: Overspend decay rate, applied beyond ``overspend_tolerance``.
            Under ``mode="agreement"`` it is the legacy disagreement rate.
        epsilon: Denominator stabiliser; makes 0-vs-0 well defined.
        one_sided_credit: Score when exactly one amount is present.
        both_null_credit: Score when neither amount is present.
        non_finite_credit: Score when either amount is inf.
        normalization: Denominator form, used by ``mode="agreement"`` only.
            ``"symmetric"`` (default) or ``"max"``; ``max`` is not sign-safe
            against the negative amounts Stage 1 deliberately injects.
        pair: The two columns to reconcile.
        mode: ``"plausibility"`` (default) or ``"agreement"`` (the v1 test).
        underspend_floor: Ratio below which underspend starts to be penalised.
        underspend_gamma: Underspend decay rate.
        non_positive_sanction_credit: Score when the sanctioned amount is
            not positive.
        overspend_tolerance: Fraction above the sanction absorbed before the
            overspend penalty engages.
        implausible_credit: Score for an amount beyond
            ``implausible_threshold``.
        implausible_threshold: Magnitude above which an amount is treated as
            a data-entry accident rather than a number.
        status_field: Column carrying the lifecycle stage.
        pre_completion_statuses: Statuses exempt from the underspend penalty.
        terminal_statuses: Statuses given the full underspend penalty.
        unknown_status_gamma_scale: Multiplier on gamma when the lifecycle
            stage cannot be determined.

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
    if mode not in RECON_MODES:
        raise ValueError(f"mode must be one of {RECON_MODES}, got {mode!r}")
    if underspend_gamma < 0.0:
        raise ValueError(
            f"underspend_gamma must be non-negative, got {underspend_gamma}"
        )
    if not 0.0 <= underspend_floor <= 1.0:
        raise ValueError(
            f"underspend_floor must lie in [0,1], got {underspend_floor}"
        )
    if overspend_tolerance < 0.0:
        raise ValueError(
            f"overspend_tolerance must be non-negative, got {overspend_tolerance}"
        )
    if not 0.0 <= unknown_status_gamma_scale <= 1.0:
        raise ValueError(
            "unknown_status_gamma_scale must lie in [0,1], got "
            f"{unknown_status_gamma_scale}"
        )
    for label, value in (
        ("one_sided_credit", one_sided_credit),
        ("both_null_credit", both_null_credit),
        ("non_finite_credit", non_finite_credit),
        ("non_positive_sanction_credit", non_positive_sanction_credit),
        ("implausible_credit", implausible_credit),
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
            lifecycle=pd.Series([], dtype="object", index=index),
            diagnostics={
                "lambda": lam,
                "normalization": normalization,
                "mode": mode,
            },
        )

    x1 = frame[left_name].to_numpy(dtype="float64", na_value=np.nan)
    x2 = frame[right_name].to_numpy(dtype="float64", na_value=np.nan)

    null_1 = np.isnan(x1)
    null_2 = np.isnan(x2)
    both_null = null_1 & null_2
    one_null = null_1 ^ null_2
    both_present = ~null_1 & ~null_2
    non_finite = both_present & (~np.isfinite(x1) | ~np.isfinite(x2))
    # CORRECTION (audit finding 3): an amount beyond the implausibility
    # threshold is finite, so it slips past the check above, but a 1e300
    # sanction is a data-entry accident rather than a number. Stage 1
    # already flags it VALUE_IMPLAUSIBLE_MAGNITUDE; refuse it on the same
    # terms as an infinity instead of letting it produce a ratio.
    finite_pair = both_present & ~non_finite
    implausible = finite_pair & (
        (np.abs(np.where(finite_pair, x1, 0.0)) > implausible_threshold)
        | (np.abs(np.where(finite_pair, x2, 0.0)) > implausible_threshold)
    )
    comparable = finite_pair & ~implausible

    if mode == "plausibility":
        # A non-positive sanction makes the ratio meaningless: there is no
        # budget to have executed against.
        non_positive = comparable & (x1 <= 0.0)
        comparable = comparable & ~non_positive
    else:
        non_positive = np.zeros(n_records, dtype=bool)

    # --- lifecycle classification (audit finding 1) ------------------------
    # Read from Stage 1's cleaned, lowercased status column. Absent column,
    # null, placeholder, unparseable or an unrecognised value all fall to
    # "unknown", which earns the mild penalty rather than the full one.
    if status_field in frame.columns:
        status_values = frame[status_field].astype("object")
    else:
        LOGGER.warning(
            "Column %r absent; every record treated as unknown lifecycle stage "
            "and given the mild underspend penalty.",
            status_field,
        )
        status_values = pd.Series([None] * n_records, index=index, dtype="object")

    pre_completion_mask = status_values.isin(list(pre_completion_statuses)).to_numpy()
    terminal_mask = status_values.isin(list(terminal_statuses)).to_numpy()
    lifecycle = np.full(n_records, "unknown", dtype=object)
    lifecycle[pre_completion_mask] = "pre_completion"
    lifecycle[terminal_mask] = "terminal"

    # Evaluate only where the statistic is defined. Substituting neutral values
    # elsewhere keeps inf/inf and 0/0 out of the arithmetic entirely, so no
    # warning is raised and no NaN can be produced and masked later.
    safe_1 = np.where(comparable, x1, 1.0)
    safe_2 = np.where(comparable, x2, 1.0)

    if mode == "plausibility":
        # ratio = spent / sanction. Not a disagreement measure - a budget
        # execution rate, read as: did this work consume a plausible share of
        # what it was granted?
        ratio = np.divide(
            safe_2,
            safe_1 + epsilon,
            out=np.zeros(n_records, dtype="float64"),
            where=(safe_1 + epsilon) > 0.0,
        )
        ratio = np.where(comparable, ratio, np.nan)

        filled = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

        # CORRECTION (audit finding 2): absorb a tolerance band before charging
        # overspend. Rounding, minor price variation and final-bill adjustments
        # routinely put a work a percent or two over sanction; penalising from
        # the first rupee treated ordinary accounting noise as a control
        # failure.
        overspend = np.exp(
            -lam * np.maximum(0.0, filled - (1.0 + overspend_tolerance))
        )

        # CORRECTION (audit finding 1): gate the underspend penalty on
        # lifecycle stage. A proposed work with near-zero expenditure is
        # behaving exactly as it should, and charging it was the single largest
        # reality-alignment error in the component. Overspend is NOT gated -
        # spending past the sanction is a control failure at any stage.
        gamma_scale = np.where(
            pre_completion_mask,
            0.0,
            np.where(terminal_mask, 1.0, unknown_status_gamma_scale),
        )
        underspend = np.exp(
            -underspend_gamma
            * gamma_scale
            * np.maximum(0.0, underspend_floor - filled)
        )
        computed = overspend * underspend
    else:  # "agreement" - the v1 symmetric equality test, kept reproducible
        numerator = np.abs(safe_1 - safe_2)
        if normalization == "symmetric":
            denominator = np.abs(safe_1) + np.abs(safe_2) + epsilon
        else:  # "max"
            denominator = np.maximum(np.maximum(safe_1, safe_2), epsilon)
        disagreement = np.divide(
            numerator,
            denominator,
            out=np.zeros(n_records, dtype="float64"),
            where=denominator > 0.0,
        )
        disagreement = np.maximum(
            np.nan_to_num(disagreement, nan=0.0, posinf=0.0, neginf=0.0), 0.0
        )
        ratio = np.where(comparable, disagreement, np.nan)
        computed = np.exp(-lam * np.where(comparable, disagreement, 0.0))

    scores = np.full(n_records, both_null_credit, dtype="float64")
    scores = np.where(one_null, one_sided_credit, scores)
    scores = np.where(non_finite, non_finite_credit, scores)
    scores = np.where(implausible, implausible_credit, scores)
    scores = np.where(non_positive, non_positive_sanction_credit, scores)
    scores = np.where(comparable, computed, scores)
    scores = np.clip(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    branch = np.full(n_records, "both_present", dtype=object)
    branch[one_null] = "one_null"
    branch[both_null] = "both_null"
    branch[non_finite] = "non_finite"
    branch[implausible] = "implausible_magnitude"
    branch[non_positive] = "non_positive_sanction"

    refused = int(non_finite.sum()) + int(implausible.sum())
    if refused:
        LOGGER.info(
            "%d record(s) carry an unusable amount (%d non-finite, %d beyond "
            "the implausibility threshold); C_recon refused.",
            refused,
            int(non_finite.sum()),
            int(implausible.sum()),
        )

    ratio_series = pd.Series(
        np.where(comparable, ratio, np.nan), index=index, dtype="float64"
    )

    return ReconciliationResult(
        scores=pd.Series(scores, index=index, dtype="float64", name="reconciliation"),
        branch=pd.Series(branch, index=index, dtype="object"),
        ratio=ratio_series,
        lifecycle=pd.Series(lifecycle, index=index, dtype="object"),
        diagnostics={
            "mode": mode,
            "lambda": lam,
            "underspend_floor": underspend_floor,
            "underspend_gamma": underspend_gamma,
            "overspend_tolerance": overspend_tolerance,
            "unknown_status_gamma_scale": unknown_status_gamma_scale,
            "implausible_threshold": implausible_threshold,
            "lifecycle_counts": {
                "pre_completion": int(pre_completion_mask.sum()),
                "terminal": int(terminal_mask.sum()),
                "unknown": int(
                    n_records
                    - int(pre_completion_mask.sum())
                    - int(terminal_mask.sum())
                ),
            },
            "non_positive_sanction_credit": non_positive_sanction_credit,
            "epsilon": epsilon,
            "normalization": normalization,
            "one_sided_credit": one_sided_credit,
            "both_null_credit": both_null_credit,
            "non_finite_credit": non_finite_credit,
            "pair": list(pair),
            "theoretical_floor": round(
                float(np.exp(-underspend_gamma * underspend_floor))
                if mode == "plausibility"
                else float(np.exp(-lam)),
                6,
            ),
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
