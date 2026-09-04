"""Stage 4 calibration instrumentation - descriptive only.

Nothing in this module influences a Stage 4 output. It exists because Stage 4's
thresholds and weights are **judgements, not estimates**, and a judgement that
cannot be seen cannot be argued with. Everything here measures the system that
already ran.

Two rules hold throughout:

**Distributions are computed only over defined values.** A quantile taken over
a column where 20% of entries are NaN is not a quantile of anything. Every
statistic reports its own ``count_defined`` alongside, so a number computed over
nine records is never mistaken for one computed over nine thousand.

**No statistic here may become a threshold.** Reading a p95 back into the
pipeline as a cut point would make the corpus define its own normality, and a
corpus with systematic fraud would calibrate that fraud into the baseline. The
report is an input to a human decision, not to the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from src.core.constants import (
    ANOMALY_TYPES,
    CALIBRATION_QUANTILES,
    CLUSTER_NOISE_REASON,
    DECISION_CLASSES,
    DUPLICATE_DIAGNOSTIC_THRESHOLDS,
    DUPLICATE_MAX_BLOCK,
    DUPLICATE_REACHABLE_THRESHOLD,
    DUPLICATE_SIMILARITY_THRESHOLD,
    DUPLICATE_TAU_DAYS,
    FEATURE_MISSING_REASON,
    NOISE_CLUSTER_ID,
    PEER_NORM_ABSENT_REASONS,
    SEVERITY_WEIGHTS,
    STAGE3_SEED,
    STAGE4_VERSION,
    Z_INVESTIGATE_THRESHOLD,
    Z_SEVERITY_SCALE,
    Z_TYPE_THRESHOLD,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: The Stage 3 deviation columns whose distributions drive every z threshold.
DEVIATION_COLUMNS: Tuple[str, ...] = (
    "deviation_cell_cost",
    "deviation_cluster_cost",
    "deviation_spend_ratio",
    "deviation_duration",
)

_ONE_DAY = pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Distribution primitives
# ---------------------------------------------------------------------------


def describe_defined(
    values: pd.Series, quantiles: Sequence[float] = CALIBRATION_QUANTILES
) -> Dict[str, Any]:
    """Summarise a series over its **defined** entries only.

    Args:
        values: Any numeric series; NaN and non-finite entries are excluded.
        quantiles: Quantiles to report, as fractions.

    Returns:
        A mapping carrying ``count_defined``, ``count_total``, ``defined_pct``,
        ``mean``, ``std`` and one ``pNN`` key per quantile. Every statistic is
        ``None`` when nothing was defined - never 0, which would read as a
        measured value.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))]
    total = int(len(numeric))
    count = int(len(finite))

    summary: Dict[str, Any] = {
        "count_defined": count,
        "count_total": total,
        "defined_pct": round(100.0 * count / total, 4) if total else 0.0,
    }
    if count == 0:
        summary.update({"mean": None, "std": None})
        summary.update({_qkey(q): None for q in quantiles})
        return summary

    summary["mean"] = round(float(finite.mean()), 6)
    # Sample std is undefined for a single observation; report None rather than
    # the 0.0 numpy would hand back, which would read as "no dispersion".
    summary["std"] = round(float(finite.std(ddof=1)), 6) if count > 1 else None
    for q in quantiles:
        summary[_qkey(q)] = round(float(finite.quantile(q)), 6)
    return summary


def _qkey(quantile: float) -> str:
    """``0.95 -> 'p95'``, ``0.5 -> 'p50'``."""
    scaled = quantile * 100.0
    return f"p{int(round(scaled))}" if abs(scaled - round(scaled)) < 1e-9 else f"p{scaled:g}"


def _rate(mask: "pd.Series | np.ndarray", total: int) -> Dict[str, Any]:
    """Count and percentage for a boolean mask."""
    count = int(np.asarray(mask, dtype=bool).sum())
    return {"count": count, "pct": round(100.0 * count / total, 4) if total else 0.0}


# ---------------------------------------------------------------------------
# The calibration report
# ---------------------------------------------------------------------------


def compute_stage4_calibration_report(
    frame: pd.DataFrame, quantiles: Sequence[float] = CALIBRATION_QUANTILES
) -> Dict[str, Any]:
    """Measure the distributions behind every Stage 4 judgement.

    Args:
        frame: A corpus frame that has been through Stages 2, 3 and 4. Sections
            whose columns are absent are reported as unavailable rather than
            silently omitted, so a partial report is never mistaken for a
            complete one.
        quantiles: Quantiles to report.

    Returns:
        A JSON-serialisable report: deviation distributions, signal activation
        rates, the severity distribution, the decision distribution, and
        coverage. Purely descriptive - see the module docstring.
    """
    total = int(len(frame))
    report: Dict[str, Any] = {
        "stage4_version": STAGE4_VERSION,
        "n_records": total,
        "_note": (
            "Descriptive only. No value here is or becomes a threshold. The "
            "thresholds this report measures against remain uncalibrated "
            "judgements; reading a quantile back into the pipeline would let "
            "the corpus define its own normality."
        ),
        "thresholds_in_force": {
            "z_type_threshold": Z_TYPE_THRESHOLD,
            "z_investigate_threshold": Z_INVESTIGATE_THRESHOLD,
            "z_severity_scale": Z_SEVERITY_SCALE,
            "severity_weights": dict(SEVERITY_WEIGHTS),
            "duplicate_similarity_threshold": DUPLICATE_SIMILARITY_THRESHOLD,
            "_status": "judgements, not estimates - none is fitted to data",
        },
        "quantiles": list(quantiles),
    }

    report["deviations"] = _deviation_section(frame, quantiles)
    report["signal_activation"] = _activation_section(frame, total)
    report["severity"] = _severity_section(frame, quantiles, total)
    report["decisions"] = _decision_section(frame, total)
    report["coverage"] = _coverage_section(frame, total)

    LOGGER.info(
        "Stage 4 calibration report computed over %d record(s); descriptive only.",
        total,
    )
    return report


def _deviation_section(
    frame: pd.DataFrame, quantiles: Sequence[float]
) -> Dict[str, Any]:
    """Distribution of each Stage 3 deviation, over defined values only."""
    section: Dict[str, Any] = {}
    for column in DEVIATION_COLUMNS:
        if column not in frame.columns:
            section[column] = {"unavailable": "column not present"}
            continue
        summary = describe_defined(frame[column], quantiles)
        # The z thresholds act on |z|, so the absolute distribution is the one
        # that actually decides anything. Both are reported: the signed
        # distribution shows skew, the absolute one shows what the gate sees.
        summary["absolute"] = describe_defined(frame[column].abs(), quantiles)
        for name, threshold in (
            ("at_or_above_type_threshold", Z_TYPE_THRESHOLD),
            ("at_or_above_investigate_threshold", Z_INVESTIGATE_THRESHOLD),
        ):
            values = frame[column].abs().to_numpy(dtype="float64", na_value=np.nan)
            with np.errstate(invalid="ignore"):
                summary[name] = _rate(values >= threshold, len(frame))
        section[column] = summary
    return section


def _activation_section(frame: pd.DataFrame, total: int) -> Dict[str, Any]:
    """How often each anomaly type actually fires."""
    if "anomaly_types" not in frame.columns:
        return {"unavailable": "Stage 4 has not been attached"}

    section: Dict[str, Any] = {}
    present: Dict[str, np.ndarray] = {}
    for name in ANOMALY_TYPES:
        column = f"type_{name}"
        if column in frame.columns:
            mask = frame[column].to_numpy(dtype=bool)
        else:
            mask = np.asarray(
                [name in (types or []) for types in frame["anomaly_types"]], dtype=bool
            )
        present[name] = mask
        section[name] = _rate(mask, total)

    # The brief names `spend_anomaly` and `temporal_anomaly`; the implemented
    # taxonomy splits the first by direction and calls the second an outlier.
    # Both readings are served rather than one silently substituted.
    section["_aggregates"] = {
        "_note": (
            "Aliases for the brief's coarser names. overspend and underspend are "
            "separate types because the lifecycle gate applies to one and not "
            "the other; temporal_anomaly is this system's temporal_outlier."
        ),
        "spend_anomaly": _rate(
            present["overspend_anomaly"] | present["underspend_anomaly"], total
        ),
        "temporal_anomaly": _rate(present["temporal_outlier"], total),
    }
    section["anomaly_count"] = (
        {str(k): int(v) for k, v in frame["anomaly_count"].value_counts().sort_index().items()}
        if "anomaly_count" in frame.columns
        else {}
    )
    return section


def _severity_section(
    frame: pd.DataFrame, quantiles: Sequence[float], total: int
) -> Dict[str, Any]:
    """Severity distribution, plus why it is undefined where it is."""
    if "severity_score" not in frame.columns:
        return {"unavailable": "Stage 4 has not been attached"}

    section = describe_defined(frame["severity_score"], quantiles)
    section["_note"] = (
        "Undefined severity is NOT low severity. The records excluded here are "
        "unmeasured, not unremarkable."
    )
    if "severity_defined_reason" in frame.columns:
        section["undefined_by_reason"] = {
            str(reason): int(count)
            for reason, count in frame.loc[
                ~frame["severity_defined"].to_numpy(dtype=bool),
                "severity_defined_reason",
            ]
            .value_counts()
            .items()
        }
    return section


def _decision_section(frame: pd.DataFrame, total: int) -> Dict[str, Any]:
    """Triage distribution and the rule that produced each class."""
    if "decision_class" not in frame.columns:
        return {"unavailable": "Stage 4 has not been attached"}

    counts = frame["decision_class"].value_counts()
    section: Dict[str, Any] = {
        name: {
            "count": int(counts.get(name, 0)),
            "pct": round(100.0 * int(counts.get(name, 0)) / total, 4) if total else 0.0,
        }
        for name in DECISION_CLASSES
    }
    if "decision_reason" in frame.columns:
        section["_by_rule"] = {
            str(k): int(v) for k, v in frame["decision_reason"].value_counts().items()
        }
    if "confidence_flag" in frame.columns:
        low = frame["confidence_flag"].to_numpy() == "low"
        escalated = frame["decision_class"].to_numpy() == "INVESTIGATE"
        section["_gate"] = {
            "low_confidence": _rate(low, total),
            "escalated_on_low_confidence": _rate(low & escalated, total),
            "_invariant": "escalated_on_low_confidence must be 0",
        }
    return section


def _coverage_section(frame: pd.DataFrame, total: int) -> Dict[str, Any]:
    """Why records could not be compared: structure, noise, or missing input."""
    section: Dict[str, Any] = {}

    reason_columns = [f"{name}_reason" for name in DEVIATION_COLUMNS]
    available = [name for name in reason_columns if name in frame.columns]
    if available:
        reasons = frame[available]
        no_norm = np.zeros(total, dtype=bool)
        noise = np.zeros(total, dtype=bool)
        missing = np.zeros(total, dtype=bool)
        for column in available:
            values = reasons[column].astype("object").to_numpy()
            no_norm |= np.isin(values, PEER_NORM_ABSENT_REASONS)
            noise |= values == CLUSTER_NOISE_REASON
            missing |= values == FEATURE_MISSING_REASON
        section["no_peer_norm"] = _rate(no_norm, total)
        section["cluster_noise"] = _rate(noise, total)
        section["missing_features"] = _rate(missing, total)
        section["_note"] = (
            "Not mutually exclusive: one record may lack one deviation for a "
            "missing feature and another for an unstable cell."
        )
        section["_by_column"] = {
            column: {
                str(k): int(v)
                for k, v in frame[column].value_counts().items()
            }
            for column in available
        }

    for column, label in (
        ("peer_cell_stable", "peer_cell_unstable"),
        ("cluster_has_norm", "cluster_without_norm"),
    ):
        if column in frame.columns:
            section[label] = _rate(
                ~frame[column].fillna(False).to_numpy(dtype=bool), total
            )
    if "valid_signal_count" in frame.columns:
        section["valid_signal_count"] = {
            str(k): int(v)
            for k, v in frame["valid_signal_count"].value_counts().sort_index().items()
        }
    return section


# ---------------------------------------------------------------------------
# Duplicate diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateDiagnostics:
    """What the duplicate detector could see, separated from what it flagged.

    Attributes:
        best_cosine: Highest **raw** cosine each record reached against any
            candidate, before the same-district and temporal-decay terms.
        reachable: ``best_cosine >= DUPLICATE_REACHABLE_THRESHOLD``.
        pair_counts: Candidate pairs at or above each diagnostic cut point.
        summary: The JSON-serialisable report.
    """

    best_cosine: pd.Series
    reachable: pd.Series
    pair_counts: Mapping[float, int]
    n_pairs: int
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return dict(self.summary)


def compute_duplicate_diagnostics(
    frame: pd.DataFrame,
    record_vectors: Optional[sparse.csr_matrix] = None,
    thresholds: Sequence[float] = DUPLICATE_DIAGNOSTIC_THRESHOLDS,
    reachable_threshold: float = DUPLICATE_REACHABLE_THRESHOLD,
    detection_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    tau_days: float = DUPLICATE_TAU_DAYS,
    max_block: int = DUPLICATE_MAX_BLOCK,
    seed: int = STAGE3_SEED,
) -> DuplicateDiagnostics:
    """Measure what the duplicate detector can and cannot see.

    The detector scores ``cosine x 1[same district] x exp(-dt/tau)`` and flags at
    0.85. Only the product is retained downstream, so the two ways a real
    duplicate can be missed - too dissimilar in text, or too far apart in time -
    are indistinguishable in the output. This function separates them.

    Nothing is changed: the embedding, the blocking, the similarity and the
    threshold are all Stage 3's. The vectors are rebuilt when not supplied
    because Stage 3 does not retain them; the rebuild is deterministic and
    produces the identical matrix.

    Args:
        frame: Corpus frame carrying ``cluster_id``, ``district`` and
            ``date_proposal``.
        record_vectors: Stage 3's duplicate-detection TF-IDF rows. Rebuilt from
            ``frame`` when omitted.
        thresholds: Cosine cut points to report, ascending.
        reachable_threshold: Cosine at or above which a pair is *reachable*.
        detection_threshold: The detector's own threshold, reported for contrast.
        tau_days: Temporal decay constant, to quantify decay attenuation.
        max_block: Largest block compared, matching Stage 3.
        seed: Embedding seed, matching Stage 3.

    Returns:
        A :class:`DuplicateDiagnostics`.
    """
    index = frame.index
    n_records = len(index)
    ordered = tuple(sorted(float(t) for t in thresholds))

    if record_vectors is None:
        from src.stage3.embedding import embed_work_names

        record_vectors = embed_work_names(
            frame,
            n_components=0,
            seed=seed,
            truncate_locality=False,
            keep_digits=True,
        ).record_tfidf()
    if record_vectors.shape[0] != n_records:
        raise ValueError(
            f"record_vectors has {record_vectors.shape[0]} rows for "
            f"{n_records} records"
        )

    best = np.zeros(n_records, dtype="float64")
    best_blended = np.zeros(n_records, dtype="float64")
    pair_counts = {threshold: 0 for threshold in ordered}
    # Pairs whose text alone would clear detection but whose blended score does
    # not. This is the number that attributes the detector's low recall.
    attenuated = 0
    n_pairs = 0
    n_blocks = 0

    if n_records:
        cluster_id = frame["cluster_id"].to_numpy()
        districts = (
            frame["district"].astype("object").fillna("__unknown__").to_numpy()
            if "district" in frame.columns
            else np.full(n_records, "__unknown__", dtype=object)
        )
        if "date_proposal" in frame.columns:
            days = (
                (frame["date_proposal"] - pd.Timestamp("1970-01-01")) / _ONE_DAY
            ).to_numpy(dtype="float64")
        else:
            days = np.full(n_records, np.nan, dtype="float64")

        positions = {label: position for position, label in enumerate(index)}
        blocks = pd.DataFrame(
            {"cluster": cluster_id, "district": districts}, index=index
        )

        for (cluster, district), block in blocks.groupby(
            ["cluster", "district"], sort=True
        ):
            if cluster == NOISE_CLUSTER_ID or district == "__unknown__":
                continue
            members = [positions[label] for label in block.index]
            if len(members) < 2:
                continue
            if len(members) > max_block:
                members = members[:max_block]
            n_blocks += 1

            rows = np.asarray(members, dtype="int64")
            similarity = (record_vectors[rows] @ record_vectors[rows].T).toarray()
            np.fill_diagonal(similarity, 0.0)

            block_days = days[rows]
            gap = np.abs(block_days[:, None] - block_days[None, :])
            decay = np.where(np.isfinite(gap), np.exp(-gap / float(tau_days)), 0.0)
            blended = np.clip(similarity * decay, 0.0, 1.0)

            upper = np.triu_indices(rows.size, k=1)
            cosines = similarity[upper]
            blends = blended[upper]
            n_pairs += int(cosines.size)
            for threshold in ordered:
                pair_counts[threshold] += int((cosines >= threshold).sum())
            attenuated += int(
                ((cosines >= detection_threshold) & (blends < detection_threshold)).sum()
            )

            np.maximum.at(best, rows, similarity.max(axis=1))
            np.maximum.at(best_blended, rows, blended.max(axis=1))

    best_cosine = pd.Series(
        np.clip(best, 0.0, 1.0), index=index, dtype="float64", name="duplicate_cosine"
    )
    reachable = pd.Series(
        best_cosine.to_numpy() >= float(reachable_threshold),
        index=index,
        dtype=bool,
        name="duplicate_reachable",
    )

    flagged = (
        frame["duplicate_flag"].to_numpy(dtype=bool)
        if "duplicate_flag" in frame.columns
        else np.zeros(n_records, dtype=bool)
    )
    # Guaranteed by construction - the decay factor lies in [0,1], so the
    # blended score can never exceed the cosine - but asserted rather than
    # assumed, because it is the one property that makes `reachable` a
    # meaningful superset of `flagged`.
    if flagged.any():
        assert bool(reachable.to_numpy()[flagged].all()), (
            "a flagged duplicate was not reachable; the blended score exceeded "
            "its own cosine, which is impossible unless the inputs disagree"
        )

    n_reachable = int(reachable.sum())
    n_flagged = int(flagged.sum())
    at_detection = pair_counts.get(float(detection_threshold))

    summary: Dict[str, Any] = {
        "stage4_version": STAGE4_VERSION,
        "n_records": n_records,
        "n_blocks": n_blocks,
        "n_candidate_pairs": n_pairs,
        "detection_threshold": float(detection_threshold),
        "reachable_threshold": float(reachable_threshold),
        "tau_days": float(tau_days),
        "_note": (
            "Diagnostic only: the detector is unchanged. `cosine` is the raw "
            "text similarity; the detector scores cosine x same-district x "
            "exp(-dt/tau) and flags at the detection threshold."
        ),
        "pairs_at_or_above": {
            f"{threshold:.2f}": pair_counts[threshold] for threshold in ordered
        },
        "pairs_at_or_above_pct": {
            f"{threshold:.2f}": round(100.0 * pair_counts[threshold] / n_pairs, 6)
            if n_pairs
            else 0.0
            for threshold in ordered
        },
        "cosine_distribution": describe_defined(best_cosine[best_cosine > 0.0]),
        "records": {
            "reachable": {
                "count": n_reachable,
                "pct": round(100.0 * n_reachable / n_records, 4) if n_records else 0.0,
            },
            "flagged": {
                "count": n_flagged,
                "pct": round(100.0 * n_flagged / n_records, 4) if n_records else 0.0,
            },
        },
        "reachable_pairs_rate": round(
            pair_counts[ordered[0]] / n_pairs, 8
        )
        if n_pairs
        else 0.0,
        "decay_attenuation": {
            "pairs_above_cosine_threshold": at_detection,
            "pairs_lost_to_decay": attenuated,
            "pct_lost_to_decay": round(100.0 * attenuated / at_detection, 4)
            if at_detection
            else None,
            "_note": (
                "Pairs whose TEXT alone clears the detection threshold but whose "
                "blended score does not. A large value here means recall is "
                "limited by the temporal decay, not by the text representation - "
                "two different problems with two different fixes."
            ),
        },
    }

    LOGGER.info(
        "Duplicate diagnostics: %d pair(s) over %d block(s); %d record(s) "
        "reachable (cos >= %.2f), %d flagged.",
        n_pairs,
        n_blocks,
        n_reachable,
        reachable_threshold,
        n_flagged,
    )

    return DuplicateDiagnostics(
        best_cosine=best_cosine,
        reachable=reachable,
        pair_counts=pair_counts,
        n_pairs=n_pairs,
        summary=summary,
    )
