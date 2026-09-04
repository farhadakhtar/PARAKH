"""Semantic clustering into work-type groups (Stage3.md sec.6).

HDBSCAN is used because it needs no ``k``, tolerates uneven cluster sizes, and
labels what it cannot place rather than forcing every point into a group. It is
deterministic: no seed is required.

Clustering runs over **distinct normalised texts**, not records
-----------------------------------------------------------------
Measured on the reference corpus, 20,000 records reduce to 879 distinct
normalised strings and 50,000 to 1,030. Clustering the distinct strings and
broadcasting the labels back took 0.08s and 0.04s respectively, against 5.01s
and 27.0s for clustering records - the difference between sitting inside
Stage3.md sec.11's 10-second budget and missing it by 2.7x.

It is also the better statistics. Identical text must receive an identical
cluster regardless, and collapsing duplicates removes the artificial density
spikes that templated naming creates in a density-based algorithm.

The consequence to keep in mind: ``min_cluster_size`` counts distinct **texts**,
not records. Stage3.md sec.6.2's record-level floor is enforced separately by
:data:`~src.core.constants.CLUSTER_MIN_RECORDS`, which merges undersized
clusters into their nearest neighbour (sec.6.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from src.core.constants import (
    CLUSTER_LABEL_TERMS,
    CLUSTER_MIN_RECORDS,
    HDBSCAN_MIN_CLUSTER_SIZE,
    NOISE_CLUSTER_ID,
)
from src.core.logger import get_logger
from src.stage3.embedding import TextEmbedding

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ClusterResult:
    """Per-record cluster assignment plus the labels that explain it."""

    cluster_id: pd.Series
    cluster_size: pd.Series
    is_noise: pd.Series
    #: cluster_id -> human-readable top-term label, e.g. "cc road".
    labels: Dict[int, str]
    #: cluster_id -> the top TF-IDF terms behind that label.
    top_terms: Dict[int, Tuple[str, ...]]
    #: Clusters merged away by the record-count floor: source -> destination.
    merged: Dict[int, int]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        """Number of retained clusters, excluding noise."""
        return int((self.cluster_id != NOISE_CLUSTER_ID).pipe(lambda s: self.cluster_id[s].nunique()))

    def label_of(self, cluster_id: int) -> str:
        """Readable label for a cluster."""
        if cluster_id == NOISE_CLUSTER_ID:
            return "unclustered"
        return self.labels.get(int(cluster_id), f"cluster {cluster_id}")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "n_clusters": self.n_clusters,
            "n_noise": int(self.is_noise.sum()),
            "noise_pct": round(100.0 * float(self.is_noise.mean()), 4)
            if len(self.is_noise)
            else 0.0,
            "labels": {str(k): v for k, v in sorted(self.labels.items())},
            "merged_clusters": {str(k): int(v) for k, v in sorted(self.merged.items())},
            **self.diagnostics,
        }


def _centroids(
    projection: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> Dict[int, np.ndarray]:
    """Record-weighted centroid of every non-noise cluster."""
    centroids: Dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        if label == NOISE_CLUSTER_ID:
            continue
        mask = labels == label
        weight = weights[mask].astype("float64")
        total = weight.sum()
        if total <= 0:
            continue
        centroid = (projection[mask] * weight[:, None]).sum(axis=0) / total
        norm = np.linalg.norm(centroid)
        centroids[int(label)] = centroid / norm if norm > 0 else centroid
    return centroids


def cluster_records(
    embedding: TextEmbedding,
    index: pd.Index,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_records: int = CLUSTER_MIN_RECORDS,
    label_terms: int = CLUSTER_LABEL_TERMS,
) -> ClusterResult:
    """Assign every record a semantic cluster.

    Args:
        embedding: Output of :func:`~src.stage3.embedding.embed_work_names`.
        index: Corpus index, for row alignment.
        min_cluster_size: HDBSCAN floor, in **distinct texts**.
        min_records: Clusters with fewer records than this are merged into the
            nearest retained cluster by centroid cosine (Stage3.md sec.6.4).
        label_terms: Top TF-IDF terms kept as each cluster's label.

    Returns:
        A :class:`ClusterResult` aligned to ``index``.
    """
    n_records = len(index)
    if n_records == 0:
        empty_int = pd.Series([], dtype="int64", index=index)
        return ClusterResult(
            cluster_id=empty_int.rename("cluster_id"),
            cluster_size=empty_int.rename("cluster_size"),
            is_noise=pd.Series([], dtype=bool, index=index),
            labels={},
            top_terms={},
            merged={},
            diagnostics={"n_unique_texts": 0, "degenerate": True},
        )

    inverse = embedding.inverse
    n_unique = embedding.n_unique
    text_weight = np.bincount(inverse, minlength=n_unique).astype("float64")

    # Texts that were emptied by normalisation carry no semantic content and
    # must not anchor a cluster: they are noise by definition, not by density.
    empty_text = np.asarray(
        [text == "" for text in embedding.unique_texts], dtype=bool
    )

    if embedding.diagnostics.get("degenerate") or n_unique <= min_cluster_size:
        LOGGER.warning(
            "Too few distinct work names (%d) to cluster; all records marked noise.",
            n_unique,
        )
        text_labels = np.full(n_unique, NOISE_CLUSTER_ID, dtype="int64")
    else:
        model = HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
        )
        # The projection is L2-normalised, so euclidean distance is a strictly
        # monotone function of cosine distance and the two induce identical
        # neighbour orderings. Stage3.md sec.6.2 asks for cosine; this is it.
        text_labels = model.fit_predict(embedding.projection).astype("int64")
        text_labels[empty_text] = NOISE_CLUSTER_ID

    record_labels = text_labels[inverse]

    # --- merge undersized clusters (Stage3.md sec.6.4) ---------------------
    merged: Dict[int, int] = {}
    counts = pd.Series(record_labels).value_counts()
    undersized = sorted(
        int(label)
        for label, size in counts.items()
        if label != NOISE_CLUSTER_ID and size < min_records
    )
    if undersized:
        centroids = _centroids(embedding.projection, text_labels, text_weight)
        retained = {
            label: centroid
            for label, centroid in centroids.items()
            if label not in undersized
        }
        if retained:
            retained_ids = sorted(retained)
            retained_matrix = np.vstack([retained[label] for label in retained_ids])
            for label in undersized:
                source = centroids.get(label)
                if source is None:
                    continue
                nearest = retained_ids[int(np.argmax(retained_matrix @ source))]
                merged[label] = nearest
            if merged:
                remap = np.arange(text_labels.max() + 2, dtype="int64")
                text_labels = np.asarray(
                    [merged.get(int(label), int(label)) for label in text_labels],
                    dtype="int64",
                )
                record_labels = text_labels[inverse]
                LOGGER.info(
                    "Merged %d undersized cluster(s) into their nearest neighbour: %s",
                    len(merged),
                    merged,
                )

    cluster_id = pd.Series(record_labels, index=index, dtype="int64", name="cluster_id")
    sizes = cluster_id.map(cluster_id.value_counts()).astype("int64")
    is_noise = (cluster_id == NOISE_CLUSTER_ID).rename("cluster_is_noise")

    # --- labels: formed in SVD space, named in token space ------------------
    top_terms: Dict[int, Tuple[str, ...]] = {}
    labels: Dict[int, str] = {}
    for label in sorted(set(int(x) for x in text_labels)):
        if label == NOISE_CLUSTER_ID:
            continue
        rows = [i for i, value in enumerate(text_labels) if int(value) == label]
        terms = embedding.top_terms(rows, k=label_terms)
        top_terms[label] = terms
        labels[label] = ", ".join(terms) if terms else f"cluster {label}"

    n_clusters = len(labels)
    LOGGER.info(
        "Clustered %d record(s) into %d cluster(s); %d noise (%.2f%%).",
        n_records,
        n_clusters,
        int(is_noise.sum()),
        100.0 * float(is_noise.mean()),
    )

    return ClusterResult(
        cluster_id=cluster_id,
        cluster_size=sizes.rename("cluster_size"),
        is_noise=is_noise,
        labels=labels,
        top_terms=top_terms,
        merged=merged,
        diagnostics={
            "n_unique_texts": n_unique,
            "min_cluster_size_texts": int(min_cluster_size),
            "min_cluster_records": int(min_records),
            "degenerate": bool(n_clusters == 0),
            "largest_cluster": int(sizes.max()) if n_clusters else 0,
            "smallest_cluster": int(
                cluster_id[~is_noise].map(cluster_id.value_counts()).min()
            )
            if n_clusters
            else 0,
        },
    )
