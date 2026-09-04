"""Peer cell formation and confidence-gated peer statistics.

    peer_cell = (cluster_id, cost_stratum)          # Stage3.md sec.8

A peer cell is the answer to "compare like with like": the same kind of work at
the same scale. Every downstream deviation is measured against its cell.

Confidence gating - the critical part
-------------------------------------
A cell's median and MAD are the yardstick every member is judged against. If a
record with an unreadable amount or a fabricated timeline contributes to that
yardstick, **the corruption propagates to every honest record in the cell**: the
norm bends toward the garbage and genuine outliers move closer to it.

So the statistics basis excludes records with ``confidence`` below
:data:`~src.core.constants.PEER_STAT_MIN_CONFIDENCE` and those whose
reconciliation branch is ``non_finite`` or ``implausible_magnitude``.

Those records are **not dropped**. They keep their cell assignment and are
measured against the clean norm - they are precisely the REMEDIATE population,
and discarding them would repeat Stage 1's silent-corruption mistake one layer
up. They simply get no vote on what normal looks like.

Median and MAD only
-------------------
Never mean and standard deviation. Median and MAD tolerate up to 50%
contamination (README sec.2); a cluster half-full of fraud must still yield a
usable norm. One consequence is deliberate: ``MAD = 0`` - every reference value
identical - leaves the deviation **undefined, not zero**. Unmeasurable is not
the same as normal, the same rule Stage 2 enforces for its components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    IMPLAUSIBLE_AMOUNT_THRESHOLD,
    MAD_SCALE,
    MISSING_STRATUM,
    NOISE_CLUSTER_ID,
    PEER_CELL_MIN_SIZE,
    PEER_STAT_EXCLUDED_BRANCHES,
    PEER_STAT_MIN_CONFIDENCE,
    PEER_STAT_MIN_REFERENCE,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Statistics computed per group. Each becomes ``<name>_median`` / ``<name>_mad``.
PEER_STAT_FIELDS: Tuple[str, ...] = ("log_cost", "spend_ratio", "duration_days")


@dataclass(frozen=True)
class PeerCellResult:
    """Per-record peer cell assignment."""

    peer_cell_id: pd.Series
    peer_cell_size: pd.Series
    peer_cell_stable: pd.Series
    #: peer_cell_id -> (cluster_id, cost_stratum).
    keys: Dict[int, Tuple[int, int]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def key_of(self, cell_id: int) -> Tuple[int, int]:
        """The ``(cluster, stratum)`` pair behind a cell id."""
        return self.keys.get(int(cell_id), (NOISE_CLUSTER_ID, MISSING_STRATUM))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        sizes = pd.Series({cid: 0 for cid in self.keys})
        observed = self.peer_cell_id.value_counts()
        for cell_id, count in observed.items():
            sizes[cell_id] = int(count)
        return {
            "n_cells": int(len(self.keys)),
            "n_stable_cells": int(self.diagnostics.get("n_stable_cells", 0)),
            "stable_record_pct": round(
                100.0 * float(self.peer_cell_stable.mean()), 4
            )
            if len(self.peer_cell_stable)
            else 0.0,
            "cell_size_min": int(sizes.min()) if len(sizes) else 0,
            "cell_size_median": int(sizes.median()) if len(sizes) else 0,
            "cell_size_max": int(sizes.max()) if len(sizes) else 0,
            **self.diagnostics,
        }


@dataclass(frozen=True)
class PeerStatistics:
    """Robust per-group statistics, and the basis they were computed from.

    Attributes:
        cell_stats: Indexed by ``peer_cell_id``.
        cluster_stats: Indexed by ``cluster_id`` - the unstratified view, kept
            because stratifying is conservative and hides gross cost inflation.
        reference_mask: Records that were allowed to shape the norms.
    """

    cell_stats: pd.DataFrame
    cluster_stats: pd.DataFrame
    reference_mask: pd.Series
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def cluster_has_norm(self, cluster_id: int) -> bool:
        """Whether a cluster carries a usable norm.

        False for the noise cluster by construction (AUDIT M1) and for any
        cluster with too few effective reference values.
        """
        if int(cluster_id) == NOISE_CLUSTER_ID:
            return False
        if int(cluster_id) not in self.cluster_stats.index:
            return False
        row = self.cluster_stats.loc[int(cluster_id)]
        medians = [c for c in self.cluster_stats.columns if c.endswith("_median")]
        return bool(any(pd.notna(row[column]) for column in medians))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "n_cells_with_stats": int(len(self.cell_stats)),
            "n_clusters_with_stats": int(len(self.cluster_stats)),
            "reference_record_pct": round(
                100.0 * float(self.reference_mask.mean()), 4
            )
            if len(self.reference_mask)
            else 0.0,
            **self.diagnostics,
        }


def form_peer_cells(
    cluster_id: pd.Series,
    cost_stratum: pd.Series,
    min_size: int = PEER_CELL_MIN_SIZE,
) -> PeerCellResult:
    """Assemble ``(cluster, stratum)`` peer cells and mark stability.

    Args:
        cluster_id: Semantic cluster per record.
        cost_stratum: Cost stratum per record.
        min_size: Records required before a cell's statistics may be trusted
            (Stage3.md sec.8.1).

    Returns:
        A :class:`PeerCellResult` aligned to the input index.

    Note:
        A cell built on ``cluster_id == -1`` is forced unstable however large it
        is. Noise points are not similar to one another, and four thousand of
        them sharing a stratum is a bucket, not a peer group. The same applies
        to ``cost_stratum == -1``, where no scale is known.
    """
    index = cluster_id.index
    if len(index) == 0:
        empty_int = pd.Series([], dtype="int64", index=index)
        return PeerCellResult(
            peer_cell_id=empty_int.rename("peer_cell_id"),
            peer_cell_size=empty_int.rename("peer_cell_size"),
            peer_cell_stable=pd.Series([], dtype=bool, index=index),
            keys={},
            diagnostics={"n_stable_cells": 0},
        )

    pairs = list(zip(cluster_id.to_numpy(), cost_stratum.to_numpy()))
    unique_keys = sorted({(int(k), int(s)) for k, s in pairs})
    key_to_id = {key: position for position, key in enumerate(unique_keys)}
    cell_ids = np.asarray([key_to_id[(int(k), int(s))] for k, s in pairs], dtype="int64")

    peer_cell_id = pd.Series(cell_ids, index=index, dtype="int64", name="peer_cell_id")
    sizes = peer_cell_id.map(peer_cell_id.value_counts()).astype("int64")

    well_formed = (cluster_id != NOISE_CLUSTER_ID) & (cost_stratum != MISSING_STRATUM)
    stable = ((sizes >= int(min_size)) & well_formed).rename("peer_cell_stable")

    n_stable_cells = int(peer_cell_id[stable].nunique())
    LOGGER.info(
        "Formed %d peer cell(s); %d stable, covering %.2f%% of records "
        "(min size %d).",
        len(unique_keys),
        n_stable_cells,
        100.0 * float(stable.mean()),
        min_size,
    )

    return PeerCellResult(
        peer_cell_id=peer_cell_id,
        peer_cell_size=sizes.rename("peer_cell_size"),
        peer_cell_stable=stable,
        keys={position: key for key, position in key_to_id.items()},
        diagnostics={
            "n_stable_cells": n_stable_cells,
            "min_size": int(min_size),
            "n_noise_cells": int(
                sum(1 for k, _ in unique_keys if k == NOISE_CLUSTER_ID)
            ),
        },
    )


def build_reference_mask(
    frame: pd.DataFrame,
    min_confidence: float = PEER_STAT_MIN_CONFIDENCE,
    excluded_branches: Sequence[str] = PEER_STAT_EXCLUDED_BRANCHES,
    amount_fields: Sequence[str] = ("sanction_amount", "amount_spent"),
    implausible_threshold: float = IMPLAUSIBLE_AMOUNT_THRESHOLD,
) -> pd.Series:
    """Select the records permitted to define peer norms.

    Args:
        frame: Corpus records carrying the Stage 2 breakdown columns.
        min_confidence: Confidence floor for contributing to a norm.
        excluded_branches: Reconciliation branches barred outright.
        amount_fields: Money columns checked for magnitude.
        implausible_threshold: Amounts beyond this are barred from the
            basis whatever their reconciliation branch.

    Returns:
        Boolean Series; ``True`` means "may shape the norm".

    Raises:
        ValueError: If ``confidence`` is absent. Stage 3 must not guess at
            reliability - run ``attach_confidence`` first.
    """
    if "confidence" not in frame.columns:
        raise ValueError(
            "Peer statistics require Stage 2 output: column 'confidence' is "
            "absent. Run attach_confidence(corpus) before Stage 3."
        )

    confidence = frame["confidence"].to_numpy(dtype="float64", na_value=0.0)
    mask = confidence >= float(min_confidence)

    if "reconciliation_branch" in frame.columns and len(excluded_branches):
        branch = frame["reconciliation_branch"].astype("object")
        mask &= ~branch.isin(list(excluded_branches)).to_numpy()

    # The branch gate alone is not enough. Stage 2 labels a record
    # 'implausible_magnitude' only when BOTH amounts are present; a record
    # with a 1e300 sanction and a missing spend is labelled 'one_null' and
    # would slip through. Measured on the reference corpus, 18 records with
    # infinite or 1e300 sanctions were shaping peer norms because of this.
    # Check the magnitude directly.
    n_implausible = 0
    for name in amount_fields:
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy(dtype="float64", na_value=0.0)
        unusable = ~np.isfinite(values) | (np.abs(values) > implausible_threshold)
        n_implausible += int((unusable & mask).sum())
        mask &= ~unusable
    if n_implausible:
        LOGGER.info(
            "Barred %d record(s) from the statistics basis for a non-finite "
            "or implausible amount.",
            n_implausible,
        )

    reference = pd.Series(mask, index=frame.index, dtype=bool, name="peer_reference")
    LOGGER.info(
        "Peer statistics basis: %d of %d record(s) (%.2f%%) pass confidence >= "
        "%.2f and a usable reconciliation branch.",
        int(reference.sum()),
        len(frame),
        100.0 * float(reference.mean()) if len(frame) else 0.0,
        min_confidence,
    )
    return reference


def _robust_stats(values: np.ndarray) -> Tuple[float, float, int]:
    """Median, scaled MAD and count for one group.

    Returns:
        ``(median, mad, n)``. ``mad`` is ``NaN`` when it evaluates to zero,
        because a zero scale makes every deviation undefined rather than
        infinite - and certainly not zero.
    """
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    median = float(np.median(finite))
    mad = MAD_SCALE * float(np.median(np.abs(finite - median)))
    return median, (mad if mad > 0.0 else float("nan")), n


def compute_peer_statistics(
    frame: pd.DataFrame,
    peer_cell_id: pd.Series,
    cluster_id: pd.Series,
    reference_mask: pd.Series,
    stable_mask: pd.Series,
    fields: Sequence[str] = PEER_STAT_FIELDS,
    min_reference: int = PEER_STAT_MIN_REFERENCE,
) -> PeerStatistics:
    """Compute median and MAD per peer cell and per cluster.

    Args:
        frame: Feature frame carrying every column named in ``fields``.
        peer_cell_id: Cell assignment per record.
        cluster_id: Cluster assignment per record.
        reference_mask: Records allowed to shape the norms.
        stable_mask: Records in stable cells; only these define cell norms.
        fields: Numeric columns to summarise.
        min_reference: Minimum contributing records before a norm is emitted.

    Returns:
        A :class:`PeerStatistics`.

    Note:
        Cluster-level norms use ``reference_mask`` alone, without
        ``stable_mask``: a cluster is a valid comparison group even when one of
        its strata is too thin to be one.

    Note:
        **The noise cluster never defines a norm** (AUDIT M1). No row is
        emitted for ``cluster_id == -1`` at all, so downstream code that
        looks one up gets a miss rather than a plausible-looking statistic.

    Note:
        **Gating uses the effective sample size** (AUDIT M3): the count of
        finite values of the field being summarised, not the count of group
        members. Both are reported - ``n_reference`` and
        ``<field>_n_effective`` - and they may legitimately differ.
    """
    available = [name for name in fields if name in frame.columns]
    missing = [name for name in fields if name not in frame.columns]
    if missing:
        LOGGER.warning("Statistic field(s) absent from the frame, skipped: %s", missing)

    def _empty_stats(label: str) -> pd.DataFrame:
        """An empty, correctly-shaped statistics frame."""
        columns = [label, "n_reference"] + [
            f"{name}_{suffix}"
            for name in available
            for suffix in ("median", "mad", "n_effective")
        ]
        return pd.DataFrame(columns=columns).set_index(label)

    def _group_stats(
        keys: pd.Series, basis: pd.Series, label: str
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        if not len(keys):
            columns = [label, "n_reference"] + [
                f"{name}_{suffix}"
                for name in available
                for suffix in ("median", "mad", "n_effective")
            ]
            return pd.DataFrame(columns=columns).set_index(label)

        basis_array = basis.to_numpy(dtype=bool)
        key_array = keys.to_numpy()
        for key in np.unique(key_array):
            # AUDIT M1: the noise cluster is not a comparison group. Its members
            # are precisely the records the clusterer judged similar to nothing,
            # so pooling them yields a norm that describes no population. It is
            # skipped outright rather than gated, so no row exists to be misread
            # as "a norm that happened to come out empty".
            if label == "cluster_id" and int(key) == NOISE_CLUSTER_ID:
                continue

            member = (key_array == key) & basis_array
            n_reference = int(member.sum())
            row: Dict[str, Any] = {label: int(key), "n_reference": n_reference}

            for name in available:
                # AUDIT M3: the guard and the estimator must count the same
                # thing. n_reference counts group membership; _robust_stats then
                # independently drops non-finite values, so a group of 15 could
                # emit a median from 2 points while reporting n_reference = 15.
                # Gate on the EFFECTIVE count instead - the values actually used.
                values = frame.loc[member, name].to_numpy(
                    dtype="float64", na_value=np.nan
                )
                n_effective = int(np.isfinite(values).sum())
                if n_effective >= int(min_reference):
                    median, mad, _ = _robust_stats(values)
                else:
                    median, mad = float("nan"), float("nan")
                row[f"{name}_median"] = median
                row[f"{name}_mad"] = mad
                # Always the true finite count, even when the norm was withheld,
                # so "withheld" and "zero values" stay distinguishable.
                row[f"{name}_n_effective"] = n_effective
            rows.append(row)
        # Every key may have been skipped - a corpus that is entirely noise
        # yields no cluster rows at all. Return the empty shape rather than
        # letting set_index fail on a frame with no columns.
        if not rows:
            return _empty_stats(label)
        return pd.DataFrame(rows).set_index(label)

    cell_basis = reference_mask & stable_mask
    cell_stats = _group_stats(peer_cell_id, cell_basis, "peer_cell_id")
    cluster_stats = _group_stats(cluster_id, reference_mask, "cluster_id")

    # Counted on the emitted median, not on n_reference, now that the two
    # can legitimately disagree (AUDIT M3).
    usable_cells = (
        int(cell_stats[f"{available[0]}_median"].notna().sum())
        if len(cell_stats) and available
        else 0
    )
    LOGGER.info(
        "Peer statistics: %d/%d cell(s) and %d/%d cluster(s) have enough "
        "high-confidence members (min %d).",
        usable_cells,
        len(cell_stats),
        int(cluster_stats[f"{available[0]}_median"].notna().sum())
        if len(cluster_stats) and available
        else 0,
        len(cluster_stats),
        min_reference,
    )

    return PeerStatistics(
        cell_stats=cell_stats,
        cluster_stats=cluster_stats,
        reference_mask=reference_mask,
        diagnostics={
            "fields": list(available),
            "min_reference": int(min_reference),
            "usable_cells": usable_cells,
            "noise_cluster_excluded": True,
            "clusters_with_norm": [int(k) for k in cluster_stats.index],
            "withheld_for_small_effective_n": {
                name: int(
                    (cell_stats[f"{name}_n_effective"] < min_reference).sum()
                )
                if len(cell_stats)
                else 0
                for name in available
            },
            "zero_mad_cells": {
                name: int(
                    (
                        cell_stats[f"{name}_median"].notna()
                        & cell_stats[f"{name}_mad"].isna()
                    ).sum()
                )
                if len(cell_stats)
                else 0
                for name in available
            },
        },
    )
