"""Near-duplicate work detection (Stage3.md sec.9).

    D(i,j) = cos(e_i, e_j) * 1[district_i == district_j] * exp(-|t_i - t_j| / tau)

Two records describing the same work, in the same district, close in time are
the classic signature of a double-claimed sanction. The three factors encode
exactly that: same words, same place, same period.

Blocking keeps it tractable
---------------------------
A full pairwise comparison is O(N^2) - 400 million pairs at 20k records. The
district indicator is zero across districts, so any pair spanning two districts
contributes nothing and need never be formed. Comparing only within
``(cluster, district)`` blocks reduces the work to O(N*b) where b is the mean
block size (Stage3.md sec.9.3).

Outputs
-------
* ``duplicate_score`` - ``D_max(i)``, the Stage3.md sec.9.4 statistic Stage 4
  sec.8 consumes.
* ``duplicate_flag`` / ``duplicate_group_id`` - the interpretable view: which
  records are near-duplicates of one another, and of whom.

Groups are formed by union-find over pairs above the threshold, so the grouping
is transitive and independent of row order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from src.core.constants import (
    DUPLICATE_MAX_BLOCK,
    DUPLICATE_SIMILARITY_THRESHOLD,
    DUPLICATE_TAU_DAYS,
    NOISE_CLUSTER_ID,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

_ONE_DAY = np.timedelta64(1, "D")
NO_DUPLICATE_GROUP = -1


@dataclass(frozen=True)
class DuplicateResult:
    """Per-record duplicate signals."""

    duplicate_score: pd.Series
    duplicate_flag: pd.Series
    duplicate_group_id: pd.Series
    #: Highest-scoring counterpart per flagged record, for explanation.
    duplicate_partner: pd.Series
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        flagged = int(self.duplicate_flag.sum())
        groups = self.duplicate_group_id[self.duplicate_group_id != NO_DUPLICATE_GROUP]
        return {
            "n_flagged": flagged,
            "flagged_pct": round(100.0 * float(self.duplicate_flag.mean()), 4)
            if len(self.duplicate_flag)
            else 0.0,
            "n_groups": int(groups.nunique()),
            "largest_group": int(groups.value_counts().max()) if len(groups) else 0,
            "score_max": round(float(self.duplicate_score.max()), 4)
            if len(self.duplicate_score)
            else 0.0,
            **self.diagnostics,
        }


class _UnionFind:
    """Disjoint-set over record positions, for transitive duplicate grouping."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        """Representative of ``item``'s set, with path compression."""
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        """Merge two sets."""
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def detect_duplicates(
    frame: pd.DataFrame,
    record_vectors: sparse.csr_matrix,
    cluster_id: pd.Series,
    threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    tau_days: float = DUPLICATE_TAU_DAYS,
    max_block: int = DUPLICATE_MAX_BLOCK,
    district_field: str = "district",
    date_field: str = "date_proposal",
) -> DuplicateResult:
    """Score and group near-duplicate works.

    Args:
        frame: Corpus records.
        record_vectors: One TF-IDF row per record.
        cluster_id: Semantic cluster per record, used for blocking.
        threshold: Cosine above which a pair is a near-duplicate.
        tau_days: Temporal decay constant in days.
        max_block: Largest block compared pairwise; larger blocks are split on
            a stable sort so the cost stays bounded.
        district_field: Column carrying the district.
        date_field: Column carrying the reference date.

    Returns:
        A :class:`DuplicateResult` aligned to ``frame.index``.

    Raises:
        ValueError: If ``record_vectors`` does not have one row per record.
    """
    index = frame.index
    n_records = len(index)
    if record_vectors.shape[0] != n_records:
        raise ValueError(
            f"record_vectors has {record_vectors.shape[0]} rows for "
            f"{n_records} records"
        )

    score = np.zeros(n_records, dtype="float64")
    partner = np.full(n_records, -1, dtype="int64")
    union = _UnionFind(n_records)

    if n_records < 2 or record_vectors.shape[1] == 0:
        LOGGER.info("Too few records or no vocabulary; duplicate detection skipped.")
        return DuplicateResult(
            duplicate_score=pd.Series(score, index=index, name="duplicate_score"),
            duplicate_flag=pd.Series(False, index=index, name="duplicate_flag"),
            duplicate_group_id=pd.Series(
                NO_DUPLICATE_GROUP, index=index, dtype="int64", name="duplicate_group_id"
            ),
            duplicate_partner=pd.Series(
                partner, index=index, dtype="int64", name="duplicate_partner"
            ),
            diagnostics={"n_blocks": 0, "n_pairs": 0, "skipped": True},
        )

    vectors = normalize(record_vectors.tocsr(), copy=True)

    districts = (
        frame[district_field].astype("object").fillna("__unknown__")
        if district_field in frame.columns
        else pd.Series("__unknown__", index=index, dtype="object")
    )
    if date_field in frame.columns:
        days = (
            (frame[date_field] - pd.Timestamp("1970-01-01")) / _ONE_DAY
        ).to_numpy(dtype="float64")
    else:
        days = np.full(n_records, np.nan, dtype="float64")

    block_keys = pd.DataFrame(
        {"cluster": cluster_id.to_numpy(), "district": districts.to_numpy()},
        index=index,
    )
    positions = {label: position for position, label in enumerate(index)}

    n_blocks = 0
    n_pairs = 0
    n_truncated = 0

    for (cluster, district), block in block_keys.groupby(
        ["cluster", "district"], sort=True
    ):
        # Noise carries no semantic claim, and an unknown district cannot
        # satisfy the same-district indicator.
        if cluster == NOISE_CLUSTER_ID or district == "__unknown__":
            continue
        members = [positions[label] for label in block.index]
        if len(members) < 2:
            continue
        if len(members) > max_block:
            n_truncated += 1
            members = members[:max_block]

        n_blocks += 1
        rows = np.asarray(members, dtype="int64")
        similarity = (vectors[rows] @ vectors[rows].T).toarray()
        np.fill_diagonal(similarity, 0.0)

        block_days = days[rows]
        gap = np.abs(block_days[:, None] - block_days[None, :])
        # An unknown date cannot evidence temporal proximity, so the decay term
        # is 0 rather than 1: absence of evidence must not manufacture a match.
        decay = np.where(np.isfinite(gap), np.exp(-gap / float(tau_days)), 0.0)

        # Cosine of two identical unit vectors can land a few ULPs above 1.0,
        # which would put the reported score outside [0,1]. Clip, do not
        # renormalise: the overshoot is float error, not signal.
        pair_score = np.clip(similarity * decay, 0.0, 1.0)
        n_pairs += rows.size * (rows.size - 1) // 2

        best = pair_score.argmax(axis=1)
        best_score = pair_score[np.arange(rows.size), best]
        improved = best_score > score[rows]
        score[rows[improved]] = best_score[improved]
        partner[rows[improved]] = rows[best[improved]]

        left, right = np.where(np.triu(pair_score >= float(threshold), k=1))
        for i, j in zip(left, right):
            union.union(int(rows[i]), int(rows[j]))

    score = np.clip(score, 0.0, 1.0)
    flag = score >= float(threshold)
    group = np.full(n_records, NO_DUPLICATE_GROUP, dtype="int64")
    roots: Dict[int, int] = {}
    for position in range(n_records):
        if not flag[position]:
            continue
        root = union.find(position)
        if root not in roots:
            roots[root] = len(roots)
        group[position] = roots[root]

    LOGGER.info(
        "Duplicate detection over %d block(s), ~%d pair(s): %d record(s) flagged "
        "(%.2f%%) in %d group(s).",
        n_blocks,
        n_pairs,
        int(flag.sum()),
        100.0 * float(flag.mean()),
        len(roots),
    )
    if n_truncated:
        LOGGER.warning(
            "%d block(s) exceeded max_block=%d and were truncated; some pairs "
            "were not compared.",
            n_truncated,
            max_block,
        )

    return DuplicateResult(
        duplicate_score=pd.Series(
            score, index=index, dtype="float64", name="duplicate_score"
        ),
        duplicate_flag=pd.Series(flag, index=index, dtype=bool, name="duplicate_flag"),
        duplicate_group_id=pd.Series(
            group, index=index, dtype="int64", name="duplicate_group_id"
        ),
        duplicate_partner=pd.Series(
            partner, index=index, dtype="int64", name="duplicate_partner"
        ),
        diagnostics={
            "n_blocks": n_blocks,
            "n_pairs": n_pairs,
            "n_truncated_blocks": n_truncated,
            "threshold": float(threshold),
            "tau_days": float(tau_days),
            "skipped": False,
        },
    )
