"""Cost stratification (Stage3.md sec.7).

Peer comparison must hold scale constant. A 50-lakh school building is not
anomalous merely for costing more than a 2-lakh borewell, and without
stratification every large work in a mixed cluster would look like an outlier.

    x = log(sanction_amount + 1)      # sec.7.2
    s = quantile bin of x, 5 bins     # sec.7.3 option A

Quantile bins rather than fixed log buckets, because the cost distribution of a
real register is neither known in advance nor stable across schemes; quantiles
adapt while remaining fully deterministic.

The honest cost of stratifying
------------------------------
Holding scale constant is defensive against false positives, but it also blinds
the cell-level statistic to *gross* cost inflation: a work inflated tenfold may
simply land in a higher stratum and be compared against other inflated works.
:mod:`src.stage3.deviations` therefore reports the cluster-level deviation
(ignoring stratum) alongside the cell-level one, so the sensitive view survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import COST_STRATA_BINS, MISSING_STRATUM
from src.core.logger import get_logger

LOGGER = get_logger(__name__)


def _occupancy(values: np.ndarray, edges: np.ndarray, n_bins: int) -> list:
    """Share of reference values falling in each band.

    Recorded with the frozen boundaries so a later run can measure how far
    a new corpus's cost distribution has moved from the one that defined
    them.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    counts = np.bincount(
        np.searchsorted(edges, finite, side="right"), minlength=n_bins
    ).astype("float64")
    return [round(float(v), 6) for v in counts / counts.sum()]


@dataclass(frozen=True)
class StratificationResult:
    """Per-record cost stratum plus the transform behind it."""

    cost_stratum: pd.Series
    log_cost: pd.Series
    #: Upper edges of each stratum in log space, for audit and explanation.
    edges: Tuple[float, ...]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_strata(self) -> int:
        """Number of populated strata, excluding the missing bucket."""
        assigned = self.cost_stratum[self.cost_stratum != MISSING_STRATUM]
        return int(assigned.nunique())

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        counts = self.cost_stratum.value_counts().sort_index()
        return {
            "n_strata": self.n_strata,
            "edges_log": [round(float(edge), 6) for edge in self.edges],
            "edges_amount": [round(float(np.expm1(edge)), 2) for edge in self.edges],
            "counts": {str(k): int(v) for k, v in counts.items()},
            **self.diagnostics,
        }


def stratify_cost(
    frame: pd.DataFrame,
    amount_field: str = "sanction_amount",
    n_bins: int = COST_STRATA_BINS,
    usable_mask: Optional[pd.Series] = None,
    frozen_edges: Optional[Sequence[float]] = None,
) -> StratificationResult:
    """Assign each record a cost stratum.

    Args:
        frame: Corpus records.
        amount_field: Column holding the sanctioned amount.
        n_bins: Requested number of quantile bins.
        usable_mask: Optional gate on which records may define the quantile
            edges. Records excluded here are still assigned a stratum - they
            simply do not shape the boundaries.
        frozen_edges: Boundaries from an earlier corpus. Supplying them
            makes a record's cost band reproducible: without it the bands
            are quantiles of whatever corpus is present, so the same work
            can change stratum between runs having not changed at all.

    Returns:
        A :class:`StratificationResult` aligned to ``frame.index``.

    Raises:
        ValueError: If ``amount_field`` is absent or ``n_bins`` is below 1.

    Note:
        A record whose amount is absent, non-finite or non-positive gets
        stratum ``-1``. That is not a sixth band: ``log`` is undefined for it,
        so no comparison group exists, and its peer cell is marked unstable.
    """
    if amount_field not in frame.columns:
        raise ValueError(f"Column {amount_field!r} is absent from the frame")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    index = frame.index
    amounts = frame[amount_field].to_numpy(dtype="float64", na_value=np.nan)
    valid = np.isfinite(amounts) & (amounts > 0.0)

    log_cost = pd.Series(np.nan, index=index, dtype="float64", name="log_cost")
    if valid.any():
        log_cost.loc[valid] = np.log1p(amounts[valid])

    strata = pd.Series(MISSING_STRATUM, index=index, dtype="int64", name="cost_stratum")
    if len(index) == 0 or not valid.any():
        LOGGER.warning("No usable sanction amounts; every record gets stratum %d.", MISSING_STRATUM)
        return StratificationResult(
            cost_stratum=strata,
            log_cost=log_cost,
            edges=(),
            diagnostics={"degenerate": True, "n_valid": 0, "requested_bins": n_bins},
        )

    if frozen_edges is not None:
        edges = np.asarray(list(frozen_edges), dtype="float64")
        assigned = np.searchsorted(edges, log_cost.to_numpy()[valid], side="right")
        strata.loc[valid] = assigned.astype("int64")
        effective = int(pd.Series(assigned).nunique())
        LOGGER.info(
            "Stratified %d record(s) against %d frozen boundary/ies; "
            "%d populated stratum/a.",
            len(index),
            edges.size,
            effective,
        )
        return StratificationResult(
            cost_stratum=strata,
            log_cost=log_cost,
            edges=tuple(float(edge) for edge in edges),
            diagnostics={
                "degenerate": False,
                "frozen": True,
                "n_valid": int(valid.sum()),
                "n_missing": int((~valid).sum()),
                "requested_bins": n_bins,
                "effective_bins": effective,
            },
        )

    basis = valid.copy()
    if usable_mask is not None:
        gated = basis & usable_mask.reindex(index).fillna(False).to_numpy(dtype=bool)
        # Falling back protects a corpus where the gate is very aggressive:
        # better to derive edges from all valid amounts than from none.
        if gated.sum() >= n_bins:
            basis = gated
        else:
            LOGGER.warning(
                "Only %d gated record(s) available for quantile edges; using all "
                "%d valid amounts instead.",
                int(gated.sum()),
                int(valid.sum()),
            )

    reference = log_cost.to_numpy()[basis]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = (
        np.unique(np.quantile(reference, quantiles))
        if len(quantiles)
        else np.asarray([])
    )

    # A constant reference distribution yields identical quantiles, which
    # np.unique collapses to a single edge rather than none. Detect it on the
    # values themselves so the degenerate flag is truthful.
    if np.unique(reference).size <= 1:
        edges = np.asarray([])

    if edges.size == 0:
        # Every reference value is identical: one band is the truthful answer.
        strata.loc[valid] = 0
        LOGGER.warning(
            "Cost distribution is degenerate (all equal); collapsed to a single stratum."
        )
        return StratificationResult(
            cost_stratum=strata,
            log_cost=log_cost,
            edges=(),
            diagnostics={
                "degenerate": True,
                "n_valid": int(valid.sum()),
                "requested_bins": n_bins,
                "effective_bins": 1,
            },
        )

    # searchsorted is exact and order-preserving, and unlike qcut it never
    # raises on duplicate edges - it simply yields fewer populated strata.
    assigned = np.searchsorted(edges, log_cost.to_numpy()[valid], side="right")
    strata.loc[valid] = assigned.astype("int64")

    effective = int(pd.Series(assigned).nunique())
    if effective < n_bins:
        LOGGER.info(
            "Cost distribution supports only %d of %d requested strata.",
            effective,
            n_bins,
        )

    LOGGER.info(
        "Stratified %d record(s) into %d cost stratum/a; %d without a usable amount.",
        len(index),
        effective,
        int((~valid).sum()),
    )

    return StratificationResult(
        cost_stratum=strata,
        log_cost=log_cost,
        edges=tuple(float(edge) for edge in edges),
        diagnostics={
            "degenerate": False,
            "frozen": False,
            "n_valid": int(valid.sum()),
            "n_missing": int((~valid).sum()),
            "requested_bins": n_bins,
            "effective_bins": effective,
            "reference_occupancy": _occupancy(
                log_cost.to_numpy()[basis], edges, len(edges) + 1
            ),
        },
    )
