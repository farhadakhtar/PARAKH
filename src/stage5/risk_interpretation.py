"""Making an uncalibrated score readable without pretending it is calibrated.

The problem this solves is a reading problem, not a modelling one. ``risk =
0.53`` invites a reader to think "about half", and the number means nothing of
the sort: the scale is a product of three bounded factors, its observed maximum
is 0.73, and nothing about it has been fitted to an outcome.

A **percentile** is the strongest honest statement available. It says where a
record sits among the records that could be scored, which is a fact about this
corpus rather than a claim about the world. So ``0.53`` becomes *"Top 2.1% of
scored records, uncalibrated scale"* - useful for triage, and impossible to
mistake for a probability.

Two decisions worth stating:

**The population is scored records only.** Ranking a record with no risk
against records that have one would manufacture exactly the false certainty
this system refuses. Unscored records get no percentile and no band.

**This changes no score.** ``risk_score`` is read, never rewritten. The
percentile is an additional column, and the band is derived from the
percentile rather than from a cut on the raw scale - a fixed cut on an
uncalibrated axis is arbitrary twice over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    CALIBRATION_WARNING,
    RISK_BAND_FLOORS,
    RISK_BANDS,
    RISK_PERCENTILE_POPULATION,
    STAGE5_VERSION,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Columns this module adds. Additive; nothing existing is touched.
INTERPRETATION_COLUMNS: Tuple[str, ...] = (
    "risk_percentile",
    "risk_band",
)


def band_for_percentile(percentile: Optional[float]) -> Optional[str]:
    """The band a percentile falls into, or None when unscored.

    Args:
        percentile: Percentile rank in [0, 100], or None.

    Returns:
        A member of :data:`RISK_BANDS`, or None where there is no percentile.
        None rather than ``"LOW"``: an unscored record is not a low-risk one,
        and the distinction is the point of the whole system.
    """
    if percentile is None or not np.isfinite(percentile):
        return None
    for name, floor in RISK_BAND_FLOORS:
        if percentile >= floor:
            return name
    return RISK_BANDS[-1]


def compute_risk_interpretation(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a percentile and a band to every scored record.

    Args:
        frame: A frame carrying ``risk_score`` and ``risk_defined``.

    Returns:
        A frame of :data:`INTERPRETATION_COLUMNS` aligned to ``frame.index``.
        Both are NaN/None for unscored records.

    Raises:
        KeyError: If ``risk_score`` is absent.
    """
    if "risk_score" not in frame.columns:
        raise KeyError("risk_interpretation requires a risk_score column")

    scores = pd.to_numeric(frame["risk_score"], errors="coerce")
    if "risk_defined" in frame.columns:
        defined = frame["risk_defined"].fillna(False).to_numpy(dtype=bool)
    else:
        defined = scores.notna().to_numpy()

    # Ranked over scored records only; unscored rows stay NaN throughout.
    # `average` for ties so two identical risks receive one percentile - an
    # arbitrary tiebreak would put one record above another on no evidence.
    ranked = scores.where(pd.Series(defined, index=frame.index))
    percentile = ranked.rank(pct=True, method="average") * 100.0

    bands = [
        band_for_percentile(value if np.isfinite(value) else None)
        for value in percentile.to_numpy(dtype="float64", na_value=np.nan)
    ]

    result = pd.DataFrame(
        {
            "risk_percentile": percentile.round(4),
            "risk_band": pd.Series(bands, index=frame.index, dtype="object"),
        },
        index=frame.index,
    )

    n_scored = int(defined.sum())
    assert result.loc[~pd.Series(defined, index=frame.index), "risk_band"].isna().all(), (
        "an unscored record was given a risk band"
    )
    LOGGER.info(
        "Risk interpretation over %d scored record(s): %s",
        n_scored,
        result["risk_band"].value_counts().to_dict(),
    )
    return result


def describe_risk(
    risk_score: Optional[float],
    percentile: Optional[float],
    band: Optional[str],
) -> str:
    """One phrase a reviewer can read without misreading it.

    Replaces ``"risk = 0.53"`` with ``"risk 0.530 (Top 2.1% of scored records,
    uncalibrated scale)"`` - the same number, with the two facts that stop it
    being mistaken for a probability: where it sits, and that the scale is not
    calibrated.
    """
    if risk_score is None or not np.isfinite(float(risk_score)):
        return (
            "No risk score: this record could not be measured. That is not the "
            "same as being measured and found safe."
        )
    if percentile is None or not np.isfinite(float(percentile)):
        return f"risk {float(risk_score):.3f} (uncalibrated scale)"
    top = 100.0 - float(percentile)
    return (
        f"risk {float(risk_score):.3f} (Top {top:.1f}% of scored records, "
        f"band {band}, uncalibrated scale)"
    )


def interpretation_report(interpretation: pd.DataFrame) -> Dict[str, Any]:
    """Corpus-level summary of the bands. Descriptive only."""
    total = len(interpretation)
    scored = int(interpretation["risk_percentile"].notna().sum())
    return {
        "stage5_version": STAGE5_VERSION,
        "n_records": total,
        "n_scored": scored,
        "population": RISK_PERCENTILE_POPULATION,
        "_note": (
            "Percentiles rank a record among SCORED records only. An unscored "
            "record has no percentile and no band, because ranking it against "
            "measured records would invent a comparison that was never made."
        ),
        "calibration_warning": CALIBRATION_WARNING,
        "bands": {
            name: int((interpretation["risk_band"] == name).sum())
            for name in RISK_BANDS
        },
        "n_unbanded": int(interpretation["risk_band"].isna().sum()),
    }
