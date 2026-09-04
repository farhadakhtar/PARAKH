"""C_temp - temporal coherence (Stage2.md sec.5.3).

    C_temp(r) = 1[not HardFail(r)] * prod_{(a,b) in O} phi(t_a, t_b)

                 | 1                          if t_b >= t_a
    phi(t_a,t_b) = exp(-kappa |t_b - t_a|)    if t_b <  t_a
                 | mu                          if either date is absent

This is the only component that tests a record **against itself**. Completeness
asks whether there is evidence; reconciliation asks whether two numbers agree;
temporal coherence asks whether the story is internally possible. A work cannot
be approved before it was proposed. When a record claims it was, no individual
field is wrong - the narrative is.

The penalty is deliberately **asymmetric**: correct ordering earns no reward,
only violations are priced, and the price decays exponentially in the size of
the inversion. A one-day clerical slip is nearly free; a 400-day inversion is
almost fatal. Rewarding correct ordering would let a record with two valid
orderings out-score one with a single valid ordering and one absent date, which
would smuggle a completeness signal into the coherence component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ORDERED_DATE_PAIRS,
    REFERENCE_DATE,
    SCHEME_START_DATE,
    TEMPORAL_HARD_FAIL_ON_FUTURE,
    TEMPORAL_KAPPA_PER_DAY,
    TEMPORAL_MISSING_PAIR_CREDIT,
)
from src.core.logger import get_logger
from src.stage1.schema import SCHEMA, NullReason, Schema, null_reason_column
from src.stage2.completeness import resolve_reasons

LOGGER = get_logger(__name__)

_ONE_DAY = np.timedelta64(1, "D")


@dataclass(frozen=True)
class TemporalResult:
    """Per-record temporal coherence plus the evidence base behind it."""

    scores: pd.Series
    #: Number of milestone pairs where BOTH dates were present and comparable.
    pairs_evaluated: pd.Series
    #: Rows forced to zero by an impossible date.
    hard_fail: pd.Series
    #: Rows carrying at least one ordering inversion.
    violated: pd.Series
    n_pairs: int
    diagnostics: Dict[str, Any]

    @property
    def defined(self) -> pd.Series:
        """Whether temporal coherence was actually measurable for each record.

        Defined when at least one milestone pair could be compared, or when a
        hard fail fired - an impossible date is positive evidence of
        incoherence, not an absence of evidence.

        Undefined rows are *excluded* from the geometric mean rather than
        scored 1.0. Scoring them 1.0 would assert perfect coherence about a
        record with no dates, which is how a wholly empty record can otherwise
        acquire a respectable confidence score.
        """
        return (self.pairs_evaluated > 0) | self.hard_fail

    @property
    def coverage(self) -> pd.Series:
        """Share of milestone pairs that could actually be checked.

        A record scoring ``C_temp = 1`` with ``coverage = 0`` is not coherent -
        it is unexamined. This series is what keeps those two cases apart.
        """
        if self.n_pairs <= 0:
            return pd.Series(0.0, index=self.scores.index, dtype="float64")
        return (self.pairs_evaluated / self.n_pairs).astype("float64")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable diagnostics."""
        scores = self.scores
        n = len(scores)
        return {
            "n_pairs": self.n_pairs,
            "mean": round(float(scores.mean()), 6) if n else 0.0,
            "hard_fail_pct": round(100.0 * float(self.hard_fail.mean()), 4) if n else 0.0,
            "violated_pct": round(100.0 * float(self.violated.mean()), 4) if n else 0.0,
            "perfect_pct": round(100.0 * float((scores >= 1.0).mean()), 4) if n else 0.0,
            "zero_evidence_pct": round(
                100.0 * float((self.pairs_evaluated == 0).mean()), 4
            )
            if n
            else 0.0,
            **self.diagnostics,
        }


def compute_temporal_result(
    frame: pd.DataFrame,
    kappa: float = TEMPORAL_KAPPA_PER_DAY,
    missing_pair_credit: float = TEMPORAL_MISSING_PAIR_CREDIT,
    hard_fail_on_future: bool = TEMPORAL_HARD_FAIL_ON_FUTURE,
    pairs: Tuple[Tuple[str, str], ...] = ORDERED_DATE_PAIRS,
    schema: Schema = SCHEMA,
) -> TemporalResult:
    """Compute ``C_temp`` for every record, with full diagnostics.

    The five cases, made explicit:

    1. **Both dates present, ordered** - factor 1.
    2. **Both present, inverted** - factor ``exp(-kappa * days)``.
    3. **Either date absent** - factor ``missing_pair_credit`` (1.0 by
       default). Absence is a completeness defect and ``C_comp`` already prices
       it; charging it again here would double-bill one defect across two
       components that must stay orthogonal.
    4. **Any date unparseable** - hard fail to 0. A date that cannot be read
       makes temporal reasoning impossible, not merely incomplete. This is why
       Stage 1's three-way null taxonomy matters: an unparseable date and an
       absent date look identical as ``NaT`` and must not be treated alike.
    5. **Any date before the scheme start** - hard fail to 0. Impossible epoch.

    Args:
        frame: Scoring frame (``corpus.records``).
        kappa: Decay rate **per day**. The parameter is dimensional; see
            :data:`~src.core.constants.TEMPORAL_KAPPA_PER_DAY`.
        missing_pair_credit: Factor for case 3.
        hard_fail_on_future: Also fail dates after ``REFERENCE_DATE``.
        pairs: Ordered milestone pairs. Imported from Stage 1 by default so the
            two stages cannot disagree about what "out of order" means.
        schema: Schema supplying the date fields.

    Returns:
        :class:`TemporalResult` whose ``scores`` share the frame's index.

    Raises:
        ValueError: If ``kappa`` is negative or the credits leave [0, 1].
    """
    if kappa < 0.0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")
    if not 0.0 <= missing_pair_credit <= 1.0:
        raise ValueError(
            f"missing_pair_credit must lie in [0,1], got {missing_pair_credit}"
        )

    index = frame.index
    n_records = len(frame)
    date_fields = tuple(schema.date_fields)

    if n_records == 0:
        empty_f = pd.Series([], dtype="float64", index=index)
        empty_i = pd.Series([], dtype="int64", index=index)
        empty_b = pd.Series([], dtype=bool, index=index)
        return TemporalResult(
            scores=empty_f.rename("temporal"),
            pairs_evaluated=empty_i,
            hard_fail=empty_b,
            violated=empty_b,
            n_pairs=len(pairs),
            diagnostics={"kappa_per_day": kappa, "pair_violations": {}},
        )

    factor = np.ones(n_records, dtype="float64")
    pairs_evaluated = np.zeros(n_records, dtype="int64")
    violated_any = np.zeros(n_records, dtype=bool)
    pair_violations: Dict[str, int] = {}

    for earlier, later in pairs:
        if earlier not in frame.columns or later not in frame.columns:
            LOGGER.warning(
                "Milestone pair (%s, %s) is not present in the frame; skipped.",
                earlier,
                later,
            )
            continue

        left = frame[earlier]
        right = frame[later]
        both_present = (left.notna() & right.notna()).to_numpy()

        # Signed gap in days. NaN wherever either date is absent; filled with 0
        # before exp() so no NaN is ever produced, then masked out by
        # `both_present` regardless.
        delta = ((right - left) / _ONE_DAY).to_numpy(dtype="float64")
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        inverted = both_present & (delta < 0.0)
        penalty = np.where(inverted, np.exp(-kappa * np.abs(delta)), 1.0)
        factor *= np.where(both_present, penalty, missing_pair_credit)

        pairs_evaluated += both_present.astype("int64")
        violated_any |= inverted
        pair_violations[f"{later}<{earlier}"] = int(inverted.sum())

    # --- hard fails ------------------------------------------------------
    reasons, _ = resolve_reasons(frame, date_fields)
    scheme_start = pd.Timestamp(SCHEME_START_DATE)
    reference = pd.Timestamp(REFERENCE_DATE)

    hard_fail = np.zeros(n_records, dtype=bool)
    n_unparseable = 0
    n_pre_scheme = 0
    n_future = 0

    for name in date_fields:
        unparseable = (
            reasons[name].astype("object") == NullReason.UNPARSEABLE.value
        ).to_numpy()
        n_unparseable += int(unparseable.sum())
        hard_fail |= unparseable

        if name not in frame.columns:
            continue
        column = frame[name]
        present = column.notna()

        before_scheme = (present & (column < scheme_start)).to_numpy()
        n_pre_scheme += int(before_scheme.sum())
        hard_fail |= before_scheme

        if hard_fail_on_future:
            after_reference = (present & (column > reference)).to_numpy()
            n_future += int(after_reference.sum())
            hard_fail |= after_reference

    factor = np.where(hard_fail, 0.0, factor)
    factor = np.clip(np.nan_to_num(factor, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    diagnostics: Dict[str, Any] = {
        "kappa_per_day": kappa,
        "missing_pair_credit": missing_pair_credit,
        "hard_fail_on_future": hard_fail_on_future,
        "pair_violations": pair_violations,
        "unparseable_date_cells": n_unparseable,
        "pre_scheme_date_cells": n_pre_scheme,
        "future_date_cells": n_future,
    }

    return TemporalResult(
        scores=pd.Series(factor, index=index, dtype="float64", name="temporal"),
        pairs_evaluated=pd.Series(pairs_evaluated, index=index, dtype="int64"),
        hard_fail=pd.Series(hard_fail, index=index, dtype=bool),
        violated=pd.Series(violated_any, index=index, dtype=bool),
        n_pairs=len(pairs),
        diagnostics=diagnostics,
    )


def compute_temporal(frame: pd.DataFrame, **kwargs: Any) -> pd.Series:
    """Compute ``C_temp`` for every record.

    Args:
        frame: Scoring frame (``corpus.records``).
        **kwargs: Forwarded to :func:`compute_temporal_result`.

    Returns:
        Float Series in [0, 1], sharing the frame's index and row order.
    """
    return compute_temporal_result(frame, **kwargs).scores
