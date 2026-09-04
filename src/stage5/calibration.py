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
    CALIBRATION_STATUS_BANNER,
    CONTRIBUTION_FLAG_THRESHOLD_PCT,
    MIN_CONFIDENCE_FOR_RISK,
    R_HIGH,
    R_LOW,
    RISK_FLAGS,
    RISK_NOT_A_THRESHOLD_NOTE,
    STAGE5_VERSION,
    UNCERTAINTY_COMPONENT_CLASS,
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


def compute_contribution_analysis(
    frame: pd.DataFrame, flag_threshold: float = CONTRIBUTION_FLAG_THRESHOLD_PCT
) -> Dict[str, Any]:
    """Attribute the score's spread to its three factors.

    In log space the product becomes a sum, so ``log risk = log S + log Q +
    log(1-U)`` and the factors can be attributed properly. Two measures are
    reported, because they answer different questions and can disagree sharply:

    * **variance share** - each factor's own variance over the total. Ignores
      covariance, so it understates a factor that moves together with the
      others.
    * **covariance share** - ``cov(x, log risk) / var(log risk)``. These sum to
      exactly 100% by the bilinearity of covariance, and this is the honest
      attribution of the spread actually observed.

    On the reference corpus the stability factor reads 0.92% by variance share
    and 1.75% by covariance share. Neither number is a reason to remove it: a
    small contribution to spread is not the same as no effect on decisions, and
    the decisive test is whether dropping the factor changes a band. It changes
    294 of them.

    Args:
        frame: A frame carrying the Stage 5 score and its three components.
        flag_threshold: Percentage below which a factor is flagged for review.

    Returns:
        A JSON-serialisable attribution, or a note if the columns are absent.
    """
    needed = ("risk_score", *COMPONENT_COLUMNS)
    if any(name not in frame.columns for name in needed):
        return {"unavailable": "Stage 5 components are not present"}

    defined = frame["risk_defined"].to_numpy(dtype=bool)
    if int(defined.sum()) < 3:
        return {"unavailable": "too few scored records to attribute"}

    raw = {
        "signal_strength": frame.loc[defined, "risk_signal_strength"].to_numpy(
            dtype="float64"
        ),
        "data_quality": frame.loc[defined, "risk_data_quality"].to_numpy(
            dtype="float64"
        ),
        "stability": 1.0
        - frame.loc[defined, "risk_uncertainty"].to_numpy(dtype="float64"),
    }

    # A factor of exactly 0 has no logarithm, and clipping it to a floor breaks
    # the identity the whole attribution rests on: log risk would no longer
    # equal the sum of the component logs, and the shares would not sum to
    # 100%. Such records are excluded and counted rather than approximated.
    positive = np.ones(int(defined.sum()), dtype=bool)
    for values in raw.values():
        positive &= values > 0.0
    n_excluded = int((~positive).sum())
    if int(positive.sum()) < 3:
        return {
            "unavailable": "too few strictly positive records to attribute",
            "n_excluded_zero_factor": n_excluded,
        }

    factors = {name: np.log(values[positive]) for name, values in raw.items()}
    # The target is the SUM of the component logs, which is exactly log risk
    # wherever every factor is positive. Using it directly makes the shares sum
    # to 100% by the bilinearity of covariance, with no residual to explain.
    log_risk = sum(factors.values())
    risk_variance = float(np.var(log_risk))
    total_variance = float(sum(np.var(values) for values in factors.values()))
    if total_variance <= 0.0 or risk_variance <= 0.0:
        return {"unavailable": "no variance to attribute"}

    analysis: Dict[str, Any] = {
        "_method": (
            "log-space attribution; the product is a sum there. Covariance "
            "share is the honest one and sums to 100%."
        ),
        "_flag_threshold_pct": flag_threshold,
        "n_scored": int(defined.sum()),
        "n_attributed": int(positive.sum()),
        "n_excluded_zero_factor": n_excluded,
        "factors": {},
        "flagged": [],
    }
    for name, values in factors.items():
        variance_share = (
            100.0 * float(np.var(values)) / total_variance if total_variance > 0 else 0.0
        )
        # ddof=0 to match np.var above. numpy's cov defaults to ddof=1, and
        # mixing the two scales every share by n/(n-1) - enough to break the
        # sum-to-100% identity that makes this attribution trustworthy.
        covariance_share = (
            100.0 * float(np.cov(values, log_risk, ddof=0)[0, 1]) / risk_variance
            if risk_variance > 0
            else 0.0
        )
        analysis["factors"][name] = {
            "variance_share_pct": round(variance_share, 4),
            "covariance_share_pct": round(covariance_share, 4),
        }
        if covariance_share < flag_threshold:
            analysis["flagged"].append(name)

    if analysis["flagged"]:
        analysis["_flag_note"] = (
            f"{analysis['flagged']} contribute below {flag_threshold}% of the "
            "score's spread. FLAGGED FOR REVIEW, NOT FOR REMOVAL: a factor can "
            "move few records far. Removal requires proving no decision changes."
        )

    # The decisive test the flag alone cannot answer.
    without_stability = (
        frame.loc[defined, "risk_signal_strength"].to_numpy(dtype="float64")
        * frame.loc[defined, "risk_data_quality"].to_numpy(dtype="float64")
    )
    actual = frame.loc[defined, "risk_score"].to_numpy(dtype="float64")

    def _band(values: np.ndarray) -> np.ndarray:
        return np.where(
            values >= R_HIGH, "high", np.where(values >= R_LOW, "moderate", "low")
        )

    changed = int((_band(actual) != _band(without_stability)).sum())
    analysis["stability_removal_test"] = {
        "_note": (
            "What would happen if the stability factor were dropped from the "
            "product. This, not the variance share, is what decides whether it "
            "is dead logic."
        ),
        "bands_changed": changed,
        "bands_changed_pct": round(100.0 * changed / max(int(defined.sum()), 1), 4),
        "verdict": "load-bearing" if changed else "redundant",
    }
    return analysis


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
            "_status": CALIBRATION_STATUS_BANNER,
            "unavailable": "Stage 5 has not been attached",
        }

    defined = frame["risk_defined"].to_numpy(dtype=bool)
    score = frame["risk_score"]

    report: Dict[str, Any] = {
        "stage5_version": STAGE5_VERSION,
        "n_records": total,
        "_status": CALIBRATION_STATUS_BANNER,
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
        "risk_score": {
            **describe_defined(score, quantiles),
            "_not_a_threshold": RISK_NOT_A_THRESHOLD_NOTE,
        },
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

    # --- AUDIT TASK 4: which uncertainty terms actually do anything -------
    #
    # Reported on the SCORED subset specifically, because a term that fires
    # often across the corpus can still be structurally dead inside the score:
    # the gate excludes exactly the records that would have triggered it.
    scored = frame.loc[defined]
    contributions = {
        "no_severity": "severity_defined",
        "no_norm": "cluster_has_norm",
        "unstable_cell": "peer_cell_stable",
    }
    liveness: Dict[str, Any] = {
        "_note": (
            "A term firing 0% on the scored subset is redundant with the gate, "
            "not merely rare. It is retained as an invariant guard and its "
            "deadness is published rather than assumed."
        )
    }
    for name, column in contributions.items():
        if column in frame.columns:
            fired = ~frame[column].fillna(False).to_numpy(dtype=bool)
            fired_scored = (
                ~scored[column].fillna(False).to_numpy(dtype=bool)
                if len(scored)
                else np.zeros(0, dtype=bool)
            )
            liveness[name] = {
                "records_affected": int(fired.sum()),
                "corpus_pct": round(100.0 * float(fired.mean()), 4) if total else 0.0,
                "records_affected_scored": int(fired_scored.sum()),
                "scored_pct": round(100.0 * float(fired_scored.mean()), 4)
                if len(scored)
                else 0.0,
                "dead_in_score": bool(len(scored) and not fired_scored.any()),
                "class": UNCERTAINTY_COMPONENT_CLASS.get(name, "unknown"),
            }
    if "risk_uncertainty" in frame.columns and len(scored):
        liveness["uncertainty_range_on_scored"] = {
            "min": round(float(scored["risk_uncertainty"].min()), 6),
            "max": round(float(scored["risk_uncertainty"].max()), 6),
        }
    liveness["_classification"] = dict(UNCERTAINTY_COMPONENT_CLASS)
    report["uncertainty_liveness"] = liveness
    report["contribution_analysis"] = compute_contribution_analysis(frame)

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
