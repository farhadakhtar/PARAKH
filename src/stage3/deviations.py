"""Peer deviations - the raw material Stage 4 turns into signals.

    deviation = (x - median_group(x)) / (1.4826 * MAD_group(x))

This module computes **how far** a record sits from its peers. It deliberately
stops there. It does not combine deviations into an anomaly score, does not
classify anomaly types and does not decide what is anomalous - those are Stage
4's responsibility, and doing them here would mean two layers owning one
decision.

Undefined is not zero
---------------------
A deviation is emitted only when it means something. It is left ``NaN``, with a
recorded reason, when:

* the peer cell is unstable (fewer than 15 records, or built on noise),
* the record's own feature is missing,
* the group has too few high-confidence members to define a norm,
* ``MAD = 0`` - every reference value identical, so no scale exists.

Reporting zero in any of those cases would say "this record is exactly normal",
which is the opposite of what is known. This mirrors the definedness rule Stage
2 enforces for its components.

Two levels, on purpose
----------------------
``deviation_cell_cost`` compares within ``(cluster, stratum)`` and is
conservative: a large school is not flagged for being larger than a borewell.
But stratifying also hides gross inflation, because a tenfold-inflated work
simply lands in a higher stratum among other inflated works.
``deviation_cluster_cost`` compares within the cluster alone and recovers that
sensitivity. Both are reported; Stage 4 decides how to weigh them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    DEVIATION_BUCKETS,
    DEVIATION_REASON_CLUSTER_NOISE,
    NOISE_CLUSTER_ID,
    Z_EXTREME_THRESHOLD,
    Z_HIGH_THRESHOLD,
)
from src.core.logger import get_logger
from src.stage3.peer_cells import PeerStatistics

LOGGER = get_logger(__name__)

#: ``(output name, feature column, group level)``. Group level is "cell" or
#: "cluster".
DEVIATION_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("deviation_cell_cost", "log_cost", "cell"),
    ("deviation_cluster_cost", "log_cost", "cluster"),
    ("deviation_spend_ratio", "spend_ratio", "cell"),
    ("deviation_duration", "duration_days", "cell"),
)

#: Why a deviation could not be computed. Ordered by precedence: the cause
#: listed first wins, so an explanation names the first thing that would have
#: to be fixed.
UNDEFINED_REASONS: Tuple[str, ...] = (
    "defined",
    "feature_missing",
    DEVIATION_REASON_CLUSTER_NOISE,
    "cell_unstable",
    "no_peer_norm",
    "zero_dispersion",
)


@dataclass(frozen=True)
class DeviationResult:
    """Per-record deviations, with a reason wherever one is undefined."""

    frame: pd.DataFrame
    specs: Tuple[Tuple[str, str, str], ...] = DEVIATION_SPECS
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> Tuple[str, ...]:
        """Deviation column names."""
        return tuple(name for name, _, _ in self.specs)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        summary: Dict[str, Any] = {}
        for name in self.names:
            values = self.frame[name]
            defined = values.notna()
            summary[name] = {
                "defined_pct": round(100.0 * float(defined.mean()), 4)
                if len(values)
                else 0.0,
                "abs_median": round(float(values.abs().median()), 4)
                if defined.any()
                else None,
                "abs_p95": round(float(values.abs().quantile(0.95)), 4)
                if defined.any()
                else None,
                "abs_max": round(float(values.abs().max()), 4)
                if defined.any()
                else None,
                "reasons": self.frame[f"{name}_reason"].value_counts().to_dict(),
                "buckets": self.frame[f"{name}_bucket"].value_counts().to_dict(),
                "n_extreme": int(
                    (self.frame[f"{name}_bucket"] == "extreme").sum()
                ),
            }
        return {"deviations": summary, **self.diagnostics}



def magnitude_bucket(
    values: np.ndarray,
    high: float = Z_HIGH_THRESHOLD,
    extreme: float = Z_EXTREME_THRESHOLD,
) -> np.ndarray:
    """Band each deviation by magnitude, without altering it.

    AUDIT M4. Deviations are deliberately **not clipped**: a 1e300 sanction
    genuinely does sit thousands of MADs from its peers, and truncating that
    would be exactly the silent corruption Stage 1 exists to prevent. But the
    raw distribution has a maximum 255x its own p99, so a magnitude-weighted
    aggregation in Stage 4 would be decided by a couple of dozen records.

    The bucket lets Stage 4 see the tail coming while the raw value survives
    untouched. ``undefined`` is a band of its own rather than a gap, so the
    NaN-carries-a-reason discipline holds here too.

    Args:
        values: Signed deviations; NaN where undefined.
        high: |z| above which a deviation is "high".
        extreme: |z| above which it is "extreme".

    Returns:
        Object array of bucket labels, aligned to ``values``.
    """
    magnitude = np.abs(values)
    buckets = np.full(values.shape, "undefined", dtype=object)
    defined = np.isfinite(values)
    buckets[defined] = "normal"
    buckets[defined & (magnitude >= high)] = "high"
    buckets[defined & (magnitude >= extreme)] = "extreme"
    return buckets


def _lookup(stats: pd.DataFrame, keys: pd.Series, column: str) -> np.ndarray:
    """Broadcast a per-group statistic onto records."""
    if column not in stats.columns or not len(stats):
        return np.full(len(keys), np.nan, dtype="float64")
    return keys.map(stats[column]).to_numpy(dtype="float64", na_value=np.nan)


def compute_deviations(
    features: pd.DataFrame,
    statistics: PeerStatistics,
    peer_cell_id: pd.Series,
    cluster_id: pd.Series,
    peer_cell_stable: pd.Series,
    specs: Sequence[Tuple[str, str, str]] = DEVIATION_SPECS,
    high_threshold: float = Z_HIGH_THRESHOLD,
    extreme_threshold: float = Z_EXTREME_THRESHOLD,
) -> DeviationResult:
    """Compute robust deviations from peer norms.

    Args:
        features: Feature table carrying the testing columns.
        statistics: Confidence-gated peer statistics.
        peer_cell_id: Cell per record.
        cluster_id: Cluster per record.
        peer_cell_stable: Whether each record's cell may be trusted.
        specs: ``(output, feature, level)`` triples to compute.
        high_threshold: |z| above which a deviation is bucketed "high".
        extreme_threshold: |z| above which it is bucketed "extreme".

    Returns:
        A :class:`DeviationResult` whose frame carries one float column and one
        reason column per spec, aligned to ``features.index``.
    """
    index = features.index
    n_records = len(index)
    output = pd.DataFrame(index=index)

    stable = peer_cell_stable.to_numpy(dtype=bool)
    # AUDIT M1. The noise cluster no longer emits a norm, so a noise record
    # would otherwise fall out as the generic "no_peer_norm". It gets its own
    # reason because the cause is different and so is the remedy: the norm is
    # not thin, the record simply could not be classified.
    is_noise = (cluster_id == NOISE_CLUSTER_ID).to_numpy(dtype=bool)

    for name, feature, level in specs:
        reason = np.full(n_records, "defined", dtype=object)
        values = pd.Series(np.nan, index=index, dtype="float64")

        if feature not in features.columns or n_records == 0:
            output[name] = values
            output[f"{name}_reason"] = pd.Series(
                np.full(n_records, "feature_missing", dtype=object), index=index
            )
            output[f"{name}_bucket"] = pd.Series(
                np.full(n_records, "undefined", dtype=object), index=index
            )
            continue

        raw = features[feature].to_numpy(dtype="float64", na_value=np.nan)
        if level == "cell":
            stats, keys = statistics.cell_stats, peer_cell_id
        else:
            stats, keys = statistics.cluster_stats, cluster_id

        median = _lookup(stats, keys, f"{feature}_median")
        mad = _lookup(stats, keys, f"{feature}_mad")

        feature_missing = ~np.isfinite(raw)
        # A cluster is a valid comparison group even where one of its strata is
        # too thin to be one, so cell-level deviations require stability and
        # cluster-level ones do not.
        unstable = ~stable if level == "cell" else np.zeros(n_records, dtype=bool)
        no_norm = ~np.isfinite(median)
        zero_scale = np.isfinite(median) & ~np.isfinite(mad)

        computable = (
            ~feature_missing & ~unstable & ~no_norm & ~zero_scale & ~is_noise
        )
        if computable.any():
            values.loc[computable] = (raw[computable] - median[computable]) / mad[
                computable
            ]

        # Precedence: the most fundamental cause wins, so an explanation names
        # the first thing that would have to be fixed.
        reason[zero_scale] = "zero_dispersion"
        reason[no_norm] = "no_peer_norm"
        reason[unstable] = "cell_unstable"
        reason[is_noise] = DEVIATION_REASON_CLUSTER_NOISE
        reason[feature_missing] = "feature_missing"
        reason[computable] = "defined"

        output[name] = values.astype("float64")
        output[f"{name}_reason"] = pd.Series(reason, index=index, dtype="object")
        # Raw value preserved; the bucket is metadata beside it, never a
        # replacement for it.
        output[f"{name}_bucket"] = pd.Series(
            magnitude_bucket(
                output[name].to_numpy(dtype="float64"),
                high=high_threshold,
                extreme=extreme_threshold,
            ),
            index=index,
            dtype="object",
        )

    for name, _, _ in specs:
        finite = np.isfinite(output[name].to_numpy(dtype="float64"))
        assert not np.isinf(output[name].to_numpy(dtype="float64")).any(), (
            f"{name} produced an infinite deviation"
        )

    defined_counts = {
        name: int(output[name].notna().sum()) for name, _, _ in specs
    }
    LOGGER.info(
        "Computed %d deviation field(s) for %d record(s); defined counts %s",
        len(specs),
        n_records,
        defined_counts,
    )

    return DeviationResult(
        frame=output,
        specs=tuple(specs),
        diagnostics={"defined_counts": defined_counts},
    )
