"""Reproducibility contract: freeze the feature space, measure the drift.

The problem this solves
-----------------------
Stage 3 is **corpus-relative**. The TF-IDF vocabulary, the IDF weights and the
cost-strata boundaries are all estimated from whatever corpus is in front of
it, so the same record scored against two different corpora lands in a
different feature space and, potentially, a different peer cell. That makes two
runs incomparable - which for an audit system is a serious defect, because a
finding must survive being re-derived next quarter.

Freezing the vocabulary and the strata boundaries pins the feature space, so a
record's embedding and cost band are reproducible across runs.

What this does NOT make reproducible
------------------------------------
**Cluster ids are still not stable across corpora.** SVD and HDBSCAN are refit
on each run, so cluster 7 in one run need not be cluster 7 in the next, even
with a frozen vocabulary. What is stabilised is the *space* the clustering
happens in, not the labels it emits. Anything downstream that must survive
across runs should key on the cluster's top-term label and its peer statistics,
never on the integer id. This is measured and reported rather than asserted.

Default behaviour is compute-and-save. Reuse is opt-in, because silently
scoring a new corpus against a stale vocabulary is worse than recomputing one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ARTIFACT_DIR,
    RUNTIME_ARTIFACT_DIR,
    COST_STRATA_FILE,
    MAX_STRATA_DRIFT,
    MAX_UNSEEN_TOKEN_RATE,
    STAGE3_VERSION,
    TFIDF_VOCAB_FILE,
)
from src.core.logger import get_logger
from src.utils.helpers import ensure_dir, write_json

LOGGER = get_logger(__name__)


class ArtifactError(RuntimeError):
    """Raised when a frozen artefact cannot be used for the current corpus."""


@dataclass(frozen=True)
class VocabularyArtifact:
    """A frozen TF-IDF feature space.

    Both the vocabulary and the IDF weights are stored. Freezing the vocabulary
    alone is not enough: IDF is re-estimated from document frequencies, so the
    same token would carry a different weight on a different corpus and the
    embedding would still drift.
    """

    vocabulary: Mapping[str, int]
    idf: Tuple[float, ...]
    ngram_range: Tuple[int, int]
    sublinear_tf: bool
    n_source_documents: int
    stage3_version: str = STAGE3_VERSION

    @property
    def n_terms(self) -> int:
        """Number of frozen terms."""
        return len(self.vocabulary)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form, with vocabulary sorted for stable diffs."""
        return {
            "stage3_version": self.stage3_version,
            "n_terms": self.n_terms,
            "n_source_documents": self.n_source_documents,
            "ngram_range": list(self.ngram_range),
            "sublinear_tf": self.sublinear_tf,
            "vocabulary": {
                term: int(index) for term, index in sorted(self.vocabulary.items())
            },
            # NOT rounded. Rounding an artefact that exists to reproduce a
            # run is self-defeating: at 10 decimal places the perturbed IDF
            # shifted the SVD projection enough to move cluster assignments.
            "idf": [float(value) for value in self.idf],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VocabularyArtifact":
        """Rebuild from a saved payload.

        Raises:
            ArtifactError: If the payload is malformed or internally inconsistent.
        """
        try:
            vocabulary = {str(k): int(v) for k, v in payload["vocabulary"].items()}
            idf = tuple(float(value) for value in payload["idf"])
            ngram = tuple(int(value) for value in payload["ngram_range"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"Malformed vocabulary artefact: {exc}") from exc
        if len(idf) != len(vocabulary):
            raise ArtifactError(
                f"Vocabulary artefact is inconsistent: {len(vocabulary)} terms "
                f"but {len(idf)} idf weights"
            )
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ArtifactError(
                "Vocabulary artefact indices are not a contiguous 0..n-1 range"
            )
        return cls(
            vocabulary=vocabulary,
            idf=idf,
            ngram_range=(ngram[0], ngram[1]),
            sublinear_tf=bool(payload.get("sublinear_tf", True)),
            n_source_documents=int(payload.get("n_source_documents", 0)),
            stage3_version=str(payload.get("stage3_version", STAGE3_VERSION)),
        )


@dataclass(frozen=True)
class StrataArtifact:
    """Frozen cost-stratum boundaries, in both log and rupee scale."""

    edges_log: Tuple[float, ...]
    n_bins: int
    n_reference: int
    #: Share of reference records that fell in each stratum when frozen.
    occupancy: Tuple[float, ...] = ()
    stage3_version: str = STAGE3_VERSION

    @property
    def edges_amount(self) -> Tuple[float, ...]:
        """The same boundaries on the original rupee scale."""
        return tuple(float(np.expm1(edge)) for edge in self.edges_log)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form carrying both scales."""
        return {
            "stage3_version": self.stage3_version,
            "n_bins": self.n_bins,
            "n_reference": self.n_reference,
            # Full precision: a rounded boundary reassigns every record
            # sitting within the rounding error of it.
            "edges_log": [float(edge) for edge in self.edges_log],
            # Display scale only, never used for assignment.
            "edges_amount": [round(float(edge), 2) for edge in self.edges_amount],
            "occupancy": [round(float(value), 6) for value in self.occupancy],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrataArtifact":
        """Rebuild from a saved payload.

        Raises:
            ArtifactError: If the payload is malformed.
        """
        try:
            edges = tuple(float(value) for value in payload["edges_log"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"Malformed strata artefact: {exc}") from exc
        if list(edges) != sorted(edges):
            raise ArtifactError("Strata artefact edges are not ascending")
        return cls(
            edges_log=edges,
            n_bins=int(payload.get("n_bins", len(edges) + 1)),
            n_reference=int(payload.get("n_reference", 0)),
            occupancy=tuple(float(v) for v in payload.get("occupancy", ())),
            stage3_version=str(payload.get("stage3_version", STAGE3_VERSION)),
        )


@dataclass(frozen=True)
class ArtifactBundle:
    """Both frozen artefacts, plus where they came from."""

    vocabulary: Optional[VocabularyArtifact] = None
    strata: Optional[StrataArtifact] = None
    source_dir: Optional[str] = None

    @property
    def complete(self) -> bool:
        """Whether both artefacts are present."""
        return self.vocabulary is not None and self.strata is not None


class ArtifactWriteError(RuntimeError):
    """Raised when a run tries to write into the committed artefact bundle.

    ``artifacts/`` holds the reference vocabulary and cost strata that the
    reuse contract is defined against. Before this guard, every pipeline run -
    the test suite included - overwrote them, so the "reuse a frozen bundle"
    guarantee was being silently destroyed by the act of running the pipeline.
    Runs now write to ``runtime_artifacts/``; overwriting the committed bundle
    requires saying so.
    """


def _guard_destination(directory: Path, allow_committed_write: bool) -> None:
    """Refuse a write into the committed bundle unless explicitly permitted.

    Args:
        directory: Where the caller wants to write.
        allow_committed_write: Whether overwriting the reference bundle is
            intended. Deliberately not a default anywhere.

    Raises:
        ArtifactWriteError: On an unpermitted write to the committed bundle.
    """
    if allow_committed_write:
        return
    try:
        target = Path(directory).resolve()
        committed = Path(ARTIFACT_DIR).resolve()
    except OSError:  # pragma: no cover - unresolvable path
        return
    if target == committed:
        raise ArtifactWriteError(
            f"refusing to write into the committed artefact bundle at "
            f"{committed}. That directory is the frozen reference the reuse "
            f"contract is defined against; a run that overwrites it destroys "
            f"the comparison it exists for. Write to {RUNTIME_ARTIFACT_DIR} "
            f"instead, or pass allow_committed_write=True to refresh the "
            f"reference deliberately."
        )


def save_artifacts(
    vocabulary: VocabularyArtifact,
    strata: StrataArtifact,
    artifact_dir: Path = RUNTIME_ARTIFACT_DIR,
    allow_committed_write: bool = False,
) -> Dict[str, Path]:
    """Write both artefacts as JSON.

    Args:
        vocabulary: The frozen TF-IDF feature space.
        strata: The frozen cost boundaries.
        artifact_dir: Destination directory, created if absent.

    Returns:
        Mapping of artefact name to the path written.
    """
    _guard_destination(artifact_dir, allow_committed_write)
    directory = ensure_dir(Path(artifact_dir))
    written = {
        "tfidf_vocab": write_json(vocabulary.to_dict(), directory / TFIDF_VOCAB_FILE),
        "cost_strata": write_json(strata.to_dict(), directory / COST_STRATA_FILE),
    }
    LOGGER.info(
        "Froze %d TF-IDF term(s) and %d cost boundary/ies to %s",
        vocabulary.n_terms,
        len(strata.edges_log),
        directory,
    )
    return written


def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> ArtifactBundle:
    """Load frozen artefacts if they exist.

    Args:
        artifact_dir: Directory to read from.

    Returns:
        An :class:`ArtifactBundle`; members are ``None`` when absent.

    Raises:
        ArtifactError: If a file exists but cannot be parsed. A corrupt artefact
            is an error, not a reason to silently recompute.
    """
    directory = Path(artifact_dir)
    vocabulary: Optional[VocabularyArtifact] = None
    strata: Optional[StrataArtifact] = None

    vocab_path = directory / TFIDF_VOCAB_FILE
    if vocab_path.exists():
        try:
            vocabulary = VocabularyArtifact.from_dict(
                json.loads(vocab_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, OSError) as exc:
            raise ArtifactError(f"Cannot read {vocab_path}: {exc}") from exc

    strata_path = directory / COST_STRATA_FILE
    if strata_path.exists():
        try:
            strata = StrataArtifact.from_dict(
                json.loads(strata_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, OSError) as exc:
            raise ArtifactError(f"Cannot read {strata_path}: {exc}") from exc

    if vocabulary or strata:
        LOGGER.info(
            "Loaded frozen artefact(s) from %s: vocabulary=%s, strata=%s",
            directory,
            vocabulary.n_terms if vocabulary else None,
            len(strata.edges_log) if strata else None,
        )
    return ArtifactBundle(
        vocabulary=vocabulary, strata=strata, source_dir=str(directory)
    )


def measure_vocabulary_drift(
    texts: Sequence[str],
    artifact: VocabularyArtifact,
    max_unseen_rate: float = MAX_UNSEEN_TOKEN_RATE,
) -> Dict[str, Any]:
    """Measure how much of a new corpus the frozen vocabulary covers.

    Two rates, because they answer different questions:

    * **type coverage** - what share of the corpus's *distinct* tokens are
      known. Sensitive to a long tail of typos.
    * **token coverage** - what share of *occurrences* are known. This is the
      one that decides whether records still embed meaningfully, so it drives
      the rejection.

    Args:
        texts: Normalised texts about to be embedded.
        artifact: The frozen vocabulary.
        max_unseen_rate: Unseen *occurrence* rate above which the run is
            rejected.

    Returns:
        Drift metrics, including ``acceptable``.
    """
    known = set(artifact.vocabulary)
    seen_types: Dict[str, int] = {}
    for text in texts:
        for token in str(text).split():
            seen_types[token] = seen_types.get(token, 0) + 1

    n_types = len(seen_types)
    n_tokens = sum(seen_types.values())
    unseen_types = [token for token in seen_types if token not in known]
    unseen_tokens = sum(seen_types[token] for token in unseen_types)

    unseen_type_rate = (len(unseen_types) / n_types) if n_types else 0.0
    unseen_token_rate = (unseen_tokens / n_tokens) if n_tokens else 0.0

    return {
        "frozen_terms": artifact.n_terms,
        "observed_types": n_types,
        "observed_tokens": n_tokens,
        "unseen_types": len(unseen_types),
        "unseen_type_rate": round(unseen_type_rate, 6),
        "unseen_token_rate": round(unseen_token_rate, 6),
        "max_unseen_token_rate": max_unseen_rate,
        "acceptable": bool(unseen_token_rate <= max_unseen_rate),
        "unseen_examples": sorted(unseen_types)[:20],
    }


def measure_strata_drift(
    log_costs: Sequence[float],
    artifact: StrataArtifact,
    max_drift: float = MAX_STRATA_DRIFT,
) -> Dict[str, Any]:
    """Measure how differently a new corpus falls into the frozen bands.

    Drift is total variation distance between the recorded occupancy and the
    observed one: half the sum of absolute differences, so 0 means identical
    and 1 means disjoint. It is the natural measure here because occupancy is a
    probability vector over bins.

    Args:
        log_costs: ``log(sanction+1)`` for the new corpus; NaN entries ignored.
        artifact: The frozen boundaries.
        max_drift: Rejection threshold.

    Returns:
        Drift metrics, including ``acceptable``.
    """
    values = np.asarray(list(log_costs), dtype="float64")
    finite = values[np.isfinite(values)]
    edges = np.asarray(artifact.edges_log, dtype="float64")
    n_bins = len(edges) + 1

    if finite.size == 0:
        return {
            "n_observed": 0,
            "observed_occupancy": [],
            "frozen_occupancy": list(artifact.occupancy),
            "total_variation_distance": 0.0,
            "max_drift": max_drift,
            "acceptable": True,
        }

    assigned = np.searchsorted(edges, finite, side="right")
    counts = np.bincount(assigned, minlength=n_bins).astype("float64")
    observed = counts / counts.sum()

    frozen = np.asarray(artifact.occupancy, dtype="float64")
    if frozen.size != observed.size or not frozen.size:
        distance = float("nan")
        acceptable = True
    else:
        distance = 0.5 * float(np.abs(observed - frozen).sum())
        acceptable = distance <= max_drift

    return {
        "n_observed": int(finite.size),
        "observed_occupancy": [round(float(v), 6) for v in observed],
        "frozen_occupancy": [round(float(v), 6) for v in frozen],
        "total_variation_distance": None
        if not np.isfinite(distance)
        else round(distance, 6),
        "max_drift": max_drift,
        "acceptable": bool(acceptable),
    }


def validate_reuse(
    frame: pd.DataFrame,
    drift: Mapping[str, Any],
    required_features: Sequence[str],
) -> None:
    """Reject a reuse run that the frozen artefacts cannot legitimately serve.

    Args:
        frame: The corpus about to be scored.
        drift: Combined vocabulary and strata drift metrics.
        required_features: Columns that must exist.

    Raises:
        ArtifactError: If a required feature is missing or drift is excessive.
            Failing loudly is correct: a corpus the frozen space does not
            describe would embed as near-zero vectors, cluster as noise, and
            silently lose every peer cell.
    """
    missing = [name for name in required_features if name not in frame.columns]
    if missing:
        raise ArtifactError(
            f"Cannot reuse frozen artefacts: required feature(s) {missing!r} "
            "are absent from the corpus."
        )

    vocabulary = drift.get("vocabulary") or {}
    if vocabulary and not vocabulary.get("acceptable", True):
        raise ArtifactError(
            "Cannot reuse the frozen vocabulary: "
            f"{100 * vocabulary['unseen_token_rate']:.1f}% of tokens are unseen, "
            f"above the {100 * vocabulary['max_unseen_token_rate']:.1f}% limit. "
            "The frozen feature space no longer describes this corpus; refit it."
        )

    strata = drift.get("strata") or {}
    if strata and not strata.get("acceptable", True):
        raise ArtifactError(
            "Cannot reuse the frozen cost strata: occupancy drifted by "
            f"{strata['total_variation_distance']:.3f}, above the "
            f"{strata['max_drift']:.3f} limit. The cost distribution has moved; "
            "refit the boundaries."
        )
