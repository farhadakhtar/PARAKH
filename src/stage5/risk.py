"""Risk composition, gating and banding.

The score
---------
::

    risk_score = signal_strength x data_quality x (1 - uncertainty)

A product, not a sum, and that is the whole argument. A sum lets a strong
anomaly compensate for unreadable data, which is exactly the inference this
system exists to refuse. A product collapses toward zero the moment any one
factor is weak, so a serious-looking finding on a record nobody can verify
scores low - not because the finding is dismissed, but because *risk* is a
claim about what is worth acting on, and acting on unverifiable evidence is
not worth it. That record is a remediation case, and Stage 4 already routes it
as one.

The gate
--------
A risk score exists only where severity is defined, confidence clears the gate,
and the work type has a peer norm. Otherwise the score is **NaN with a stated
reason** - never 0, which would read as "checked, safe".

What this is not
----------------
No record is labelled fraud here. The bands are descriptive; Stage 6 routes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    MIN_CONFIDENCE_FOR_RISK,
    R_HIGH,
    R_LOW,
    RISK_FLAGS,
    RISK_UNDEFINED_REASONS,
    STAGE5_VERSION,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RiskResult:
    """The score, whether it exists, and why not when it does not."""

    score: pd.Series
    defined: pd.Series
    reason: pd.Series
    flag: pd.Series
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        defined = self.defined.to_numpy(dtype=bool)
        present = self.score[defined]
        return {
            "defined_pct": round(100.0 * float(defined.mean()), 4)
            if len(self.score)
            else 0.0,
            "undefined_by_reason": {
                str(k): int(v)
                for k, v in self.reason[~defined].value_counts().items()
            },
            "flags": {
                name: int((self.flag == name).sum()) for name in RISK_FLAGS
            },
            "median": round(float(present.median()), 6) if len(present) else None,
            "p95": round(float(present.quantile(0.95)), 6) if len(present) else None,
            "p99": round(float(present.quantile(0.99)), 6) if len(present) else None,
            "max": round(float(present.max()), 6) if len(present) else None,
            **self.diagnostics,
        }


def compute_risk(
    frame: pd.DataFrame,
    signal_strength: pd.Series,
    data_quality: pd.Series,
    uncertainty: pd.Series,
    min_confidence: float = MIN_CONFIDENCE_FOR_RISK,
    r_high: float = R_HIGH,
    r_low: float = R_LOW,
) -> RiskResult:
    """Compose, gate and band the risk score.

    The gate has three conjuncts, applied in a fixed precedence so the reason a
    record has no score is the *first* thing wrong with it rather than an
    arbitrary one:

    1. severity undefined -> ``severity_undefined``
    2. confidence below the gate -> ``confidence_below_gate``
    3. work type carries no norm -> ``no_cluster_norm``

    The third conjunct is structurally redundant on any frame Stage 3 produces
    - no norm implies no defined deviation implies no severity, so conjunct 1
    fires first - and it is kept as defence in depth. The diagnostics report
    how often it actually binds, which on the reference corpus is never.

    Args:
        frame: Corpus frame with Stage 2-4 output.
        signal_strength: Step 1 output.
        data_quality: Step 2 output.
        uncertainty: Step 3 output.
        min_confidence: The confidence gate.
        r_high: At or above this, ``high_risk``.
        r_low: At or above this and below ``r_high``, ``moderate_risk``.

    Returns:
        A :class:`RiskResult` aligned to ``frame.index``.

    Raises:
        ValueError: If the bands are not ordered inside [0,1].
    """
    if not 0.0 <= r_low <= r_high <= 1.0:
        raise ValueError(
            f"risk bands must satisfy 0 <= r_low ({r_low}) <= r_high ({r_high}) <= 1"
        )
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must lie in [0,1]")

    index = frame.index
    n_records = len(index)

    severity_defined = frame["severity_defined"].fillna(False).to_numpy(dtype=bool)
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").to_numpy(
        dtype="float64", na_value=0.0
    )
    has_norm = frame["cluster_has_norm"].fillna(False).to_numpy(dtype=bool)

    confidence_ok = confidence >= float(min_confidence)
    gated = severity_defined & confidence_ok & has_norm

    reason = np.full(n_records, "ok", dtype=object)
    # Reverse precedence order: the earliest failure is written last and wins.
    reason[~has_norm] = "no_cluster_norm"
    reason[~confidence_ok] = "confidence_below_gate"
    reason[~severity_defined] = "severity_undefined"
    reason[gated] = "ok"

    strength = signal_strength.to_numpy(dtype="float64")
    quality = data_quality.to_numpy(dtype="float64")
    stability = 1.0 - uncertainty.to_numpy(dtype="float64")

    with np.errstate(invalid="ignore"):
        product = strength * quality * stability
    # NaN, never 0: an ungated record has an UNKNOWN risk, not a low one.
    score = np.where(gated, product, np.nan)

    finite = np.isfinite(score)
    assert not np.isinf(score).any(), "risk produced an infinite value"
    assert bool(((score[finite] >= 0.0) & (score[finite] <= 1.0)).all()), (
        "risk escaped [0,1]"
    )
    assert bool(finite[gated].all()), (
        "a gated record produced a non-finite risk; a component was NaN where "
        "the gate said it should be defined"
    )

    flag = np.full(n_records, "insufficient_data", dtype=object)
    flag[gated & (score < float(r_low))] = "low_risk"
    flag[gated & (score >= float(r_low)) & (score < float(r_high))] = "moderate_risk"
    flag[gated & (score >= float(r_high))] = "high_risk"

    # --- invariants that make the score safe to act on --------------------
    high = flag == "high_risk"
    assert not bool((high & ~confidence_ok).any()), (
        "a record below the confidence gate was banded high_risk"
    )
    assert not bool((high & ~severity_defined).any()), (
        "a record with no severity was banded high_risk"
    )
    assert bool((flag[~gated] == "insufficient_data").all()), (
        "an ungated record received a risk band"
    )
    assert set(np.unique(flag)) <= set(RISK_FLAGS)
    assert set(np.unique(reason)) <= set(RISK_UNDEFINED_REASONS)

    n_norm_binds = int((severity_defined & confidence_ok & ~has_norm).sum())

    LOGGER.info(
        "Risk over %d record(s): %d defined (%.2f%%); bands %s",
        n_records,
        int(gated.sum()),
        100.0 * float(gated.mean()) if n_records else 0.0,
        {k: int(v) for k, v in pd.Series(flag).value_counts().items()},
    )
    if n_norm_binds:
        LOGGER.warning(
            "The cluster_has_norm conjunct bound on %d record(s); Stage 3 "
            "emitted a defined severity for a work type with no norm.",
            n_norm_binds,
        )

    return RiskResult(
        score=pd.Series(score, index=index, dtype="float64", name="risk_score"),
        defined=pd.Series(gated, index=index, dtype=bool, name="risk_defined"),
        reason=pd.Series(reason, index=index, dtype="object", name="risk_defined_reason"),
        flag=pd.Series(flag, index=index, dtype="object", name="risk_flag"),
        diagnostics={
            "stage5_version": STAGE5_VERSION,
            "min_confidence": float(min_confidence),
            "r_high": float(r_high),
            "r_low": float(r_low),
            "n_norm_conjunct_binds": n_norm_binds,
            "_note": (
                "Bands are descriptive, not decisions. No record is labelled "
                "fraud; Stage 6 owns routing."
            ),
        },
    )
