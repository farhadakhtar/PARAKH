"""Interpretable per-record feature table (brief step 6).

Every column here is a named quantity a human can read off a file. There are no
identifiers, no raw strings and no learned representations.

Reuse, never recompute
----------------------
``spend_ratio`` is taken from the Stage 2 breakdown, not recalculated. Stage 2
already resolved every edge case around it - non-finite amounts, non-positive
sanctions, the lifecycle gate - and a second implementation would eventually
disagree with the first.

The three feature roles
-----------------------
Stage 3 keeps three sets strictly disjoint:

============  =====================================  ==========================
role          columns                                purpose
============  =====================================  ==========================
grouping      cluster_id, cost_stratum               who is comparable
testing       log_cost, spend_ratio, duration_days   how far from normal
gating        confidence, reconciliation_branch,     whose evidence counts
              temporal_hard_fail, lifecycle_state
============  =====================================  ==========================

The separation is load-bearing. If a testing feature entered the grouping set,
its own signal would be normalised away - every district compared only against
itself detects no district-level anomaly at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Numeric quantities deviation is measured on.
TESTING_FEATURES: Tuple[str, ...] = ("log_cost", "spend_ratio", "duration_days")

#: Structural keys deciding who is comparable.
GROUPING_FEATURES: Tuple[str, ...] = ("cluster_id", "cost_stratum", "peer_cell_id")

#: Stage 2 signals deciding whose evidence may shape a norm.
GATING_FEATURES: Tuple[str, ...] = (
    "confidence",
    "reconciliation_branch",
    "temporal_hard_fail",
    "lifecycle_state",
)

_ONE_DAY = np.timedelta64(1, "D")


@dataclass(frozen=True)
class FeatureTable:
    """Numeric feature frame plus the reason each cell is usable or not."""

    frame: pd.DataFrame
    testing: Tuple[str, ...] = TESTING_FEATURES
    grouping: Tuple[str, ...] = GROUPING_FEATURES
    gating: Tuple[str, ...] = GATING_FEATURES
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        coverage = {
            name: round(100.0 * float(self.frame[name].notna().mean()), 4)
            for name in self.testing
            if name in self.frame.columns
        }
        return {
            "n_records": int(len(self.frame)),
            "testing_features": list(self.testing),
            "grouping_features": list(self.grouping),
            "gating_features": list(self.gating),
            "coverage_pct": coverage,
            **self.diagnostics,
        }


def compute_duration_days(
    frame: pd.DataFrame,
    start_field: str = "date_proposal",
    end_field: str = "date_completion",
) -> pd.Series:
    """Project duration in days, defined only where it is meaningful.

    Undefined when either date is absent, when the pair is inverted, or when
    Stage 2 recorded a temporal hard fail - a duration derived from an
    impossible timeline is a number without a referent, and letting it into the
    peer statistics would corrupt them.

    Args:
        frame: Corpus records.
        start_field: Earlier milestone.
        end_field: Later milestone.

    Returns:
        Float Series of days, ``NaN`` where undefined.
    """
    index = frame.index
    duration = pd.Series(np.nan, index=index, dtype="float64", name="duration_days")
    if len(index) == 0 or start_field not in frame.columns or end_field not in frame.columns:
        return duration

    start = frame[start_field]
    end = frame[end_field]
    both = (start.notna() & end.notna()).to_numpy()
    if not both.any():
        return duration

    days = ((end - start) / _ONE_DAY).to_numpy(dtype="float64")
    usable = both & np.isfinite(days) & (days >= 0.0)

    if "temporal_hard_fail" in frame.columns:
        usable &= ~frame["temporal_hard_fail"].fillna(False).to_numpy(dtype=bool)

    duration.loc[usable] = days[usable]
    return duration


def build_feature_table(
    frame: pd.DataFrame,
    log_cost: pd.Series,
    cluster_id: pd.Series,
    cost_stratum: pd.Series,
    peer_cell_id: pd.Series,
    peer_cell_size: pd.Series,
    peer_cell_stable: pd.Series,
) -> FeatureTable:
    """Assemble the Stage 3 feature table.

    Args:
        frame: Corpus records with the Stage 2 breakdown attached.
        log_cost: ``log(sanction + 1)`` from stratification.
        cluster_id: Semantic cluster per record.
        cost_stratum: Cost stratum per record.
        peer_cell_id: Peer cell per record.
        peer_cell_size: Records in each peer cell.
        peer_cell_stable: Whether the cell may be trusted.

    Returns:
        A :class:`FeatureTable` aligned to ``frame.index``.

    Raises:
        ValueError: If the Stage 2 breakdown is absent.
    """
    required = ("confidence", "spend_ratio")
    absent = [name for name in required if name not in frame.columns]
    if absent:
        raise ValueError(
            f"Stage 3 features require the Stage 2 breakdown; missing {absent!r}. "
            "Run attach_confidence(corpus) first."
        )

    index = frame.index
    table = pd.DataFrame(index=index)

    # --- testing ---------------------------------------------------------
    table["log_cost"] = log_cost.astype("float64")
    table["spend_ratio"] = frame["spend_ratio"].astype("float64")  # reused, not recomputed
    table["duration_days"] = compute_duration_days(frame)

    # --- grouping --------------------------------------------------------
    table["cluster_id"] = cluster_id.astype("int64")
    table["cost_stratum"] = cost_stratum.astype("int64")
    table["peer_cell_id"] = peer_cell_id.astype("int64")
    table["peer_cell_size"] = peer_cell_size.astype("int64")
    table["peer_cell_stable"] = peer_cell_stable.astype(bool)

    # --- gating (carried, never used for grouping or testing) -------------
    table["confidence"] = frame["confidence"].astype("float64")
    for name in ("reconciliation_branch", "lifecycle_state"):
        table[name] = (
            frame[name].astype("object")
            if name in frame.columns
            else pd.Series("unknown", index=index, dtype="object")
        )
    table["temporal_hard_fail"] = (
        frame["temporal_hard_fail"].fillna(False).astype(bool)
        if "temporal_hard_fail" in frame.columns
        else pd.Series(False, index=index, dtype=bool)
    )

    numeric = [name for name in TESTING_FEATURES if name in table.columns]
    for name in numeric:
        values = table[name].to_numpy(dtype="float64")
        if np.isinf(values).any():
            # An infinite testing feature would poison a median silently.
            LOGGER.warning(
                "Feature %r contains %d non-finite value(s); marked undefined.",
                name,
                int(np.isinf(values).sum()),
            )
            table.loc[np.isinf(values), name] = np.nan

    LOGGER.info(
        "Built feature table: %d record(s), %d testing feature(s); coverage %s",
        len(index),
        len(numeric),
        {name: f"{100 * table[name].notna().mean():.1f}%" for name in numeric},
    )

    return FeatureTable(
        frame=table,
        diagnostics={
            "n_stable_records": int(table["peer_cell_stable"].sum()),
        },
    )
