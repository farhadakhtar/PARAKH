"""Stage 5 calibration - descriptive only.

Same rule as Stage 4's: nothing measured here may become a threshold. Reading a
quantile back into the bands would let the corpus define its own risk appetite,
so a corpus with systematic fraud would calibrate that fraud into "normal".

The correlations are the interesting part. Risk is *meant* to track severity and
*meant* to be attenuated by low confidence; if those relationships do not appear
in the data, the composition is not doing what it claims.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.constants import (
    ANOMALY_TYPES,
    CALIBRATION_QUANTILES,
    MIN_CONFIDENCE_FOR_RISK,
    R_HIGH,
    R_LOW,
    RISK_FLAGS,
    RISK_UNDEFINED_REASONS,
    STAGE5_VERSION,
)
from src.core.logger import get_logger
from src.stage4.calibration import describe_defined

LOGGER = get_logger(__name__)

#: Component columns whose distributions explain the score's shape.
COMPONENT_COLUMNS = (
    "risk_signal_strength",
    "risk_data_quality",
    "risk_uncertainty",
)


def _spearman(left: pd.Series, right: pd.Series) -> Optional[float]:
    """Rank correlation over rows where both are defined.

    Rank rather than Pearson: risk is a bounded, heavily skewed product, and a
    linear coefficient on that would mostly measure the skew. Monotone
    association is the property actually being claimed.
    """
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3:
        return None
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return round(float(value), 6) if pd.notna(value) else None


def compute_stage5_calibration_report(
    frame: pd.DataFrame, quantiles: Sequence[float] = CALIBRATION_QUANTILES
) -> Dict[str, Any]:
    """Measure the risk distribution and the relationships behind it.

    Args:
        frame: A corpus frame carrying Stage 2-5 output.
        quantiles: Quantiles to report.

    Returns:
        A JSON-serialisable report. Descriptive only; defines no threshold.
    """
    total = int(len(frame))
    if "risk_score" not in frame.columns:
        return {
            "stage5_version": STAGE5_VERSION,
            "n_records": total,
            "unavailable": "Stage 5 has not been attached",
        }

    defined = frame["risk_defined"].to_numpy(dtype=bool)
    score = frame["risk_score"]

    report: Dict[str, Any] = {
        "stage5_version": STAGE5_VERSION,
        "n_records": total,
        "_note": (
            "Descriptive only. No value here is or becomes a threshold. R_HIGH "
            "and R_LOW are judgements fixed before this was measured; if they "
            "select uncomfortably, the honest response is to argue the number "
            "in the open, not to slide it to fit."
        ),
        "bands_in_force": {
            "r_high": R_HIGH,
            "r_low": R_LOW,
            "min_confidence_for_risk": MIN_CONFIDENCE_FOR_RISK,
            "_status": "judgements, not estimates - none is fitted to data",
        },
        "risk_score": describe_defined(score, quantiles),
        "undefined": {
            "count": int((~defined).sum()),
            "pct": round(100.0 * float((~defined).mean()), 4) if total else 0.0,
            "by_reason": {
                str(reason): int(count)
                for reason, count in frame.loc[~defined, "risk_defined_reason"]
                .value_counts()
                .items()
            }
            if "risk_defined_reason" in frame.columns
            else {},
        },
        "flags": {
            name: {
                "count": int((frame["risk_flag"] == name).sum()),
                "pct": round(
                    100.0 * float((frame["risk_flag"] == name).mean()), 4
                )
                if total
                else 0.0,
            }
            for name in RISK_FLAGS
        }
        if "risk_flag" in frame.columns
        else {},
    }

    report["components"] = {
        name: describe_defined(frame[name], quantiles)
        for name in COMPONENT_COLUMNS
        if name in frame.columns
    }

    # --- the relationships the composition claims to have -----------------
    report["correlations"] = {
        "_method": "spearman rank, over records where both values are defined",
        "_note": (
            "risk should track severity and should be attenuated by low "
            "confidence. These are the numbers that say whether it does."
        ),
        "risk_vs_severity": _spearman(score, frame["severity_score"])
        if "severity_score" in frame.columns
        else None,
        "risk_vs_confidence": _spearman(score, frame["confidence"])
        if "confidence" in frame.columns
        else None,
        "risk_vs_signal_strength": _spearman(score, frame["risk_signal_strength"])
        if "risk_signal_strength" in frame.columns
        else None,
        "risk_vs_data_quality": _spearman(score, frame["risk_data_quality"])
        if "risk_data_quality" in frame.columns
        else None,
        "risk_vs_uncertainty": _spearman(score, frame["risk_uncertainty"])
        if "risk_uncertainty" in frame.columns
        else None,
    }

    # --- how much the gate and the product actually attenuate -------------
    if "risk_signal_strength" in frame.columns:
        pair = frame.loc[defined, ["risk_score", "risk_signal_strength"]].dropna()
        positive = pair[pair["risk_signal_strength"] > 0]
        report["attenuation"] = {
            "_note": (
                "How far the product pulls the score below the raw signal. A "
                "large number is the design working, not a defect: risk is "
                "conditional on evidence."
            ),
            "median_ratio": round(
                float((positive["risk_score"] / positive["risk_signal_strength"]).median()),
                6,
            )
            if len(positive)
            else None,
            "n_compared": int(len(positive)),
        }

    # --- per anomaly type --------------------------------------------------
    breakdown: Dict[str, Any] = {}
    for name in ANOMALY_TYPES:
        column = f"type_{name}"
        if column in frame.columns:
            mask = frame[column].to_numpy(dtype=bool)
        elif "anomaly_types" in frame.columns:
            mask = np.asarray(
                [name in (types or []) for types in frame["anomaly_types"]], dtype=bool
            )
        else:
            continue
        subset = score[mask]
        breakdown[name] = {
            "count": int(mask.sum()),
            "risk_defined": int((defined & mask).sum()),
            "median_risk": round(float(subset.median()), 6)
            if subset.notna().any()
            else None,
            "p95_risk": round(float(subset.quantile(0.95)), 6)
            if subset.notna().any()
            else None,
        }
    report["by_anomaly_type"] = breakdown

    LOGGER.info(
        "Stage 5 calibration report computed over %d record(s); descriptive only.",
        total,
    )
    return report
