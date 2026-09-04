"""``SemanticLayer`` - the Stage 3 orchestrator (Stage3.md sec.10.2).

    embed -> cluster -> stratify -> peer cells -> statistics -> features
          -> duplicates -> deviations

Stage 3 produces **structure**. It ends at deviations from peer norms and does
not cross into scoring or classification, which belong to Stage 4.

Every output column is row-aligned to ``corpus.records``, deterministic, and
serialisable. As in Stage 2, ``Corpus.records`` returns a live reference, so
attachment needs no change to a locked stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.constants import (
    ARTIFACT_DIR,
    CLUSTER_MIN_RECORDS,
    COST_STRATA_BINS,
    DUPLICATE_SIMILARITY_THRESHOLD,
    DUPLICATE_TAU_DAYS,
    HDBSCAN_MIN_CLUSTER_SIZE,
    PEER_CELL_MIN_SIZE,
    PEER_STAT_MIN_CONFIDENCE,
    PEER_STAT_MIN_REFERENCE,
    STAGE3_CALIBRATION_REPORT,
    STAGE3_REPRODUCIBILITY_REPORT,
    STAGE3_REUSE_ARTIFACTS_DEFAULT,
    STAGE3_SAVE_ARTIFACTS_DEFAULT,
    STAGE3_SEED,
    STAGE3_VERSION,
    SVD_COMPONENTS,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)
from src.core.logger import get_logger
from src.stage3.artifacts import (
    ArtifactBundle,
    StrataArtifact,
    VocabularyArtifact,
    load_artifacts,
    measure_strata_drift,
    measure_vocabulary_drift,
    save_artifacts,
    validate_reuse,
)
from src.stage3.calibration import ConfigSnapshot, build_calibration_report
from src.stage3.clustering import ClusterResult, cluster_records
from src.stage3.deviations import DEVIATION_SPECS, DeviationResult, compute_deviations
from src.stage3.duplicate_detection import DuplicateResult, detect_duplicates
from src.stage3.embedding import TextEmbedding, embed_work_names
from src.stage3.explanation import build_explanation_inputs
from src.stage3.features import FeatureTable, build_feature_table
from src.stage3.peer_cells import (
    PeerCellResult,
    PeerStatistics,
    build_reference_mask,
    compute_peer_statistics,
    form_peer_cells,
)
from src.stage3.stratification import StratificationResult, stratify_cost
from src.utils.helpers import write_json

if TYPE_CHECKING:  # pragma: no cover
    from src.stage1.corpus import Corpus

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

#: The complete Stage 3 -> Stage 4 column contract, in a fixed order.
#:
#: Every column is written by :meth:`SemanticLayer.attach`, row-aligned to the
#: corpus index, deterministic and serialisable. Columns are only ever added;
#: removing one breaks the downstream contract.
STAGE3_COLUMNS: Tuple[str, ...] = (
    # --- semantic structure ------------------------------------------------
    "cluster_id",
    # cluster_id is RUN-LOCAL. Measured: reusing frozen artefacts reproduces
    # the partition exactly (adjusted Rand index 1.0) but permutes the
    # integer labels, because HDBSCAN numbers clusters in an order that
    # turns on float ties at the 1e-16 level. Downstream code that must
    # survive across runs should key on cluster_label, never on cluster_id.
    "cluster_label",
    "cluster_size",
    "cluster_is_noise",
    # AUDIT M1: the noise cluster no longer defines a norm, so downstream
    # code can tell "measured against a real peer group" from "no group".
    "cluster_has_norm",
    # --- scale -------------------------------------------------------------
    "log_cost",
    "cost_stratum",
    # --- peer cell ---------------------------------------------------------
    "peer_cell_id",
    "peer_cell_size",
    "peer_cell_stable",
    "peer_reference",
    # --- testing features --------------------------------------------------
    "duration_days",
    # --- deviations (raw material for Stage 4; NOT scores) -----------------
    "deviation_cell_cost",
    "deviation_cell_cost_reason",
    "deviation_cell_cost_bucket",
    "deviation_cluster_cost",
    "deviation_cluster_cost_reason",
    "deviation_cluster_cost_bucket",
    "deviation_spend_ratio",
    "deviation_spend_ratio_reason",
    "deviation_spend_ratio_bucket",
    "deviation_duration",
    "deviation_duration_reason",
    "deviation_duration_bucket",
    # --- duplicates --------------------------------------------------------
    "duplicate_score",
    "duplicate_flag",
    "duplicate_group_id",
)


@dataclass(frozen=True)
class SemanticConfig:
    """Stage 3 parameters, all named and all deterministic."""

    seed: int = STAGE3_SEED
    n_components: int = SVD_COMPONENTS
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE
    cluster_min_records: int = CLUSTER_MIN_RECORDS
    cost_bins: int = COST_STRATA_BINS
    peer_cell_min_size: int = PEER_CELL_MIN_SIZE
    min_confidence: float = PEER_STAT_MIN_CONFIDENCE
    min_reference: int = PEER_STAT_MIN_REFERENCE
    duplicate_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD
    duplicate_tau_days: float = DUPLICATE_TAU_DAYS

    #: Reproducibility contract. Default is compute-and-save; reuse is
    #: opt-in, because silently scoring a new corpus against a stale
    #: vocabulary is worse than recomputing one.
    artifact_dir: Path = ARTIFACT_DIR
    reuse_artifacts: bool = STAGE3_REUSE_ARTIFACTS_DEFAULT
    save_artifacts: bool = STAGE3_SAVE_ARTIFACTS_DEFAULT

    def __post_init__(self) -> None:
        """Reject a malformed configuration at construction."""
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if self.cost_bins < 1:
            raise ValueError("cost_bins must be >= 1")
        if self.peer_cell_min_size < 1:
            raise ValueError("peer_cell_min_size must be >= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0,1]")
        if not 0.0 <= self.duplicate_threshold <= 1.0:
            raise ValueError("duplicate_threshold must lie in [0,1]")
        if self.duplicate_tau_days <= 0.0:
            raise ValueError("duplicate_tau_days must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable echo."""
        return {
            "stage3_version": STAGE3_VERSION,
            "seed": self.seed,
            "embedding": {"n_components": self.n_components},
            "clustering": {
                "min_cluster_size_texts": self.min_cluster_size,
                "cluster_min_records": self.cluster_min_records,
            },
            "stratification": {"cost_bins": self.cost_bins},
            "peer_cells": {"min_size": self.peer_cell_min_size},
            "peer_statistics": {
                "min_confidence": self.min_confidence,
                "min_reference": self.min_reference,
            },
            "duplicates": {
                "threshold": self.duplicate_threshold,
                "tau_days": self.duplicate_tau_days,
            },
            "reproducibility": {
                "artifact_dir": str(self.artifact_dir),
                "reuse_artifacts": self.reuse_artifacts,
                "save_artifacts": self.save_artifacts,
            },
        }


@dataclass(frozen=True)
class SemanticResult:
    """Everything Stage 3 produced (Stage3.md sec.10.3)."""

    frame: pd.DataFrame
    embedding: TextEmbedding
    clusters: ClusterResult
    stratification: StratificationResult
    peer_cells: PeerCellResult
    statistics: PeerStatistics
    features: FeatureTable
    duplicates: DuplicateResult
    deviations: DeviationResult
    config: SemanticConfig
    elapsed_seconds: float = 0.0
    reproducibility: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def cluster_ids(self) -> pd.Series:
        """Semantic cluster per record."""
        return self.frame["cluster_id"]

    @property
    def cost_strata(self) -> pd.Series:
        """Cost stratum per record."""
        return self.frame["cost_stratum"]

    @property
    def peer_cell_ids(self) -> pd.Series:
        """Peer cell per record."""
        return self.frame["peer_cell_id"]

    @property
    def duplicate_scores(self) -> pd.Series:
        """``D_max`` per record."""
        return self.frame["duplicate_score"]

    def explain(self, row: Any) -> Dict[str, Any]:
        """Structured explanation inputs for one record.

        Reads stored outputs; recomputes nothing.
        """
        return build_explanation_inputs(
            self.frame,
            row,
            cluster_labels=self.clusters.labels,
            peer_cell_keys=self.peer_cells.keys,
            cell_stats=self.statistics.cell_stats,
            cluster_stats=self.statistics.cluster_stats,
        )

    def calibration_report(self) -> Dict[str, Any]:
        """Parameters used and the distributions they produced.

        Instrumentation only: nothing here has been tuned.
        """
        return build_calibration_report(self)

    def reproducibility_report(self) -> Dict[str, Any]:
        """Vocabulary size, unseen-token rate and strata drift."""
        return {
            "stage3_version": STAGE3_VERSION,
            "n_records": len(self.frame),
            **self.reproducibility,
        }

    def report(self) -> Dict[str, Any]:
        """Corpus-level Stage 3 summary.

        Carries no wall-clock value - ``elapsed_seconds`` stays on the result
        object and out of the serialised report - so two runs over the same
        corpus write byte-identical artefacts, as Stages 1 and 2 do.
        """
        return {
            "stage3_version": STAGE3_VERSION,
            "n_records": len(self.frame),
            "config": self.config.to_dict(),
            "embedding": self.embedding.diagnostics,
            "clustering": self.clusters.to_dict(),
            "stratification": self.stratification.to_dict(),
            "peer_cells": self.peer_cells.to_dict(),
            "peer_statistics": self.statistics.to_dict(),
            "features": self.features.to_dict(),
            "duplicates": self.duplicates.to_dict(),
            "deviations": self.deviations.to_dict(),
            "reproducibility": self.reproducibility,
        }

    def save_reports(self, output_dir: PathLike) -> Dict[str, Path]:
        """Write the Stage 3 report and peer statistics as JSON/CSV."""
        directory = Path(output_dir)
        written = {
            "stage3_report": write_json(
                self.report(), directory / "stage3_report.json"
            ),
            "calibration_report": write_json(
                self.calibration_report(), directory / STAGE3_CALIBRATION_REPORT
            ),
            "reproducibility_report": write_json(
                self.reproducibility_report(),
                directory / STAGE3_REPRODUCIBILITY_REPORT,
            ),
        }
        cell_path = directory / "stage3_peer_statistics.csv"
        self.statistics.cell_stats.to_csv(cell_path, lineterminator="\n")
        written["peer_statistics"] = cell_path
        LOGGER.info("Wrote %d Stage 3 artefact(s) to %s", len(written), directory)
        return written


class SemanticLayer:
    """Stage 3 pipeline: structure, peer cells and peer statistics."""

    def __init__(self, config: Optional[SemanticConfig] = None) -> None:
        """Build the layer.

        Args:
            config: Stage 3 parameters; defaults to the calibrated constants.
        """
        self.config = config or SemanticConfig()

    def __repr__(self) -> str:
        return f"<SemanticLayer {STAGE3_VERSION} seed={self.config.seed}>"

    def _frame_of(self, source: Union["Corpus", pd.DataFrame]) -> pd.DataFrame:
        """Accept a Corpus or a bare frame."""
        if isinstance(source, pd.DataFrame):
            return source
        records = getattr(source, "records", None)
        if isinstance(records, pd.DataFrame):
            return records
        raise TypeError(f"Expected a Corpus or DataFrame, got {type(source).__name__}")

    def run(self, source: Union["Corpus", pd.DataFrame]) -> SemanticResult:
        """Execute the full Stage 3 pipeline.

        Args:
            source: A corpus that has already been through Stage 2's
                ``attach_confidence``, or its ``records`` frame.

        Returns:
            A :class:`SemanticResult`; row count, order and index preserved.

        Raises:
            ValueError: If the Stage 2 breakdown is absent. Stage 3 must not
                guess at reliability.
        """
        frame = self._frame_of(source)
        config = self.config
        started = time.perf_counter()

        bundle = (
            load_artifacts(config.artifact_dir)
            if config.reuse_artifacts
            else ArtifactBundle()
        )
        reuse_vocabulary = config.reuse_artifacts and bundle.vocabulary is not None
        reuse_strata = config.reuse_artifacts and bundle.strata is not None

        # 1. embedding. Probe first so drift can be measured and the run
        #    rejected BEFORE a stale feature space silently mangles it.
        reproducibility: Dict[str, Any] = {
            "mode": "reuse" if config.reuse_artifacts else "fit",
            "artifact_dir": str(config.artifact_dir),
            "reused_vocabulary": reuse_vocabulary,
            "reused_strata": reuse_strata,
        }
        if reuse_vocabulary or reuse_strata:
            probe = embed_work_names(frame, n_components=0, seed=config.seed)
            drift: Dict[str, Any] = {}
            if reuse_vocabulary:
                drift["vocabulary"] = measure_vocabulary_drift(
                    list(probe.unique_texts), bundle.vocabulary
                )
            if reuse_strata:
                log_costs = np.log1p(
                    frame["sanction_amount"].to_numpy(
                        dtype="float64", na_value=np.nan
                    )
                )
                drift["strata"] = measure_strata_drift(log_costs, bundle.strata)
            validate_reuse(frame, drift, required_features=("work_name",))
            reproducibility["drift"] = drift

        embedding = embed_work_names(
            frame,
            n_components=config.n_components,
            seed=config.seed,
            frozen_vocabulary=bundle.vocabulary.vocabulary
            if reuse_vocabulary
            else None,
            frozen_idf=bundle.vocabulary.idf if reuse_vocabulary else None,
        )
        # 2. clustering
        clusters = cluster_records(
            embedding,
            frame.index,
            min_cluster_size=config.min_cluster_size,
            min_records=config.cluster_min_records,
        )
        # 3. confidence gate, computed before stratification so the cost bands
        #    are shaped by trustworthy amounts rather than by garbage.
        reference = build_reference_mask(frame, min_confidence=config.min_confidence)
        stratification = stratify_cost(
            frame,
            n_bins=config.cost_bins,
            usable_mask=reference,
            frozen_edges=bundle.strata.edges_log if reuse_strata else None,
        )
        # 4. peer cells
        peer_cells = form_peer_cells(
            clusters.cluster_id,
            stratification.cost_stratum,
            min_size=config.peer_cell_min_size,
        )
        # 5. features
        features = build_feature_table(
            frame,
            log_cost=stratification.log_cost,
            cluster_id=clusters.cluster_id,
            cost_stratum=stratification.cost_stratum,
            peer_cell_id=peer_cells.peer_cell_id,
            peer_cell_size=peer_cells.peer_cell_size,
            peer_cell_stable=peer_cells.peer_cell_stable,
        )
        # 6. peer statistics
        statistics = compute_peer_statistics(
            features.frame,
            peer_cell_id=peer_cells.peer_cell_id,
            cluster_id=clusters.cluster_id,
            reference_mask=reference,
            stable_mask=peer_cells.peer_cell_stable,
            min_reference=config.min_reference,
        )
        # 7. duplicates - on the UNTRUNCATED text.
        #
        # Clustering strips the locality so that one work type does not
        # fragment across places. Duplicate detection needs the opposite:
        # the ward or village is precisely what distinguishes two genuinely
        # different CC roads in one district. Sharing the clustering vectors
        # here measured 0.047 precision and 0.119 recall against Stage 1's
        # injected duplicates, because every road in a district looked
        # identical once the locality was gone. Digits are kept for the same
        # reason: "ward no. 11" and "ward no. 35" are otherwise identical
        # after stopword removal, and the number is the entire distinction.
        duplicate_embedding = embed_work_names(
            frame,
            n_components=0,
            seed=config.seed,
            truncate_locality=False,
            keep_digits=True,
        )
        duplicates = detect_duplicates(
            frame,
            duplicate_embedding.record_tfidf(),
            clusters.cluster_id,
            threshold=config.duplicate_threshold,
            tau_days=config.duplicate_tau_days,
        )
        # 8. deviations - the last step. No scoring, no classification.
        deviations = compute_deviations(
            features.frame,
            statistics,
            peer_cell_id=peer_cells.peer_cell_id,
            cluster_id=clusters.cluster_id,
            peer_cell_stable=peer_cells.peer_cell_stable,
        )

        output = pd.DataFrame(index=frame.index)
        output["cluster_id"] = clusters.cluster_id
        output["cluster_label"] = clusters.cluster_id.map(
            lambda value: clusters.label_of(int(value))
        )
        output["cluster_size"] = clusters.cluster_size
        output["cluster_is_noise"] = clusters.is_noise
        output["cluster_has_norm"] = clusters.cluster_id.map(
            lambda value: statistics.cluster_has_norm(int(value))
        ).astype(bool)
        output["log_cost"] = stratification.log_cost
        output["cost_stratum"] = stratification.cost_stratum
        output["peer_cell_id"] = peer_cells.peer_cell_id
        output["peer_cell_size"] = peer_cells.peer_cell_size
        output["peer_cell_stable"] = peer_cells.peer_cell_stable
        output["peer_reference"] = reference
        output["duration_days"] = features.frame["duration_days"]
        for column in deviations.frame.columns:
            output[column] = deviations.frame[column]
        output["duplicate_score"] = duplicates.duplicate_score
        output["duplicate_flag"] = duplicates.duplicate_flag
        output["duplicate_group_id"] = duplicates.duplicate_group_id

        missing = [name for name in STAGE3_COLUMNS if name not in output.columns]
        if missing:
            raise RuntimeError(f"Stage 3 contract incomplete: missing {missing!r}")
        output = output.loc[:, list(STAGE3_COLUMNS)]

        # Freeze this run's feature space so a later run can reproduce it.
        if config.save_artifacts and not config.reuse_artifacts:
            vocabulary_artifact = VocabularyArtifact(
                vocabulary=embedding.vocabulary,
                idf=tuple(embedding.diagnostics.get("idf", ())),
                ngram_range=TFIDF_NGRAM_RANGE,
                sublinear_tf=TFIDF_SUBLINEAR_TF,
                n_source_documents=embedding.n_unique,
            )
            strata_artifact = StrataArtifact(
                edges_log=stratification.edges,
                n_bins=config.cost_bins,
                n_reference=int(
                    stratification.diagnostics.get("n_valid", 0)
                ),
                occupancy=tuple(
                    stratification.diagnostics.get("reference_occupancy", ())
                ),
            )
            if vocabulary_artifact.n_terms:
                paths = save_artifacts(
                    vocabulary_artifact, strata_artifact, config.artifact_dir
                )
                ConfigSnapshot.from_config(config).save(config.artifact_dir)
                reproducibility["saved"] = {
                    name: str(path) for name, path in paths.items()
                }
        reproducibility["cluster_id_is_run_local"] = True
        reproducibility["stable_cluster_key"] = "cluster_label"
        reproducibility["vocabulary_size"] = len(embedding.vocabulary)
        reproducibility["strata_edges_log"] = [
            round(float(edge), 6) for edge in stratification.edges
        ]
        reproducibility["strata_edges_amount"] = [
            round(float(np.expm1(edge)), 2) for edge in stratification.edges
        ]

        elapsed = time.perf_counter() - started
        LOGGER.info(
            "Stage 3 complete in %.2fs: %d cluster(s), %d stable peer cell(s), "
            "%d record(s) contributing to norms.",
            elapsed,
            clusters.n_clusters,
            peer_cells.diagnostics.get("n_stable_cells", 0),
            int(reference.sum()),
        )

        return SemanticResult(
            frame=output,
            embedding=embedding,
            clusters=clusters,
            stratification=stratification,
            peer_cells=peer_cells,
            statistics=statistics,
            features=features,
            duplicates=duplicates,
            deviations=deviations,
            config=config,
            elapsed_seconds=elapsed,
            reproducibility=reproducibility,
        )

    def fit_transform(self, source: Union["Corpus", pd.DataFrame]) -> SemanticResult:
        """Alias for :meth:`run` (Stage3.md sec.10.2 names this method)."""
        return self.run(source)


def attach_structure(
    corpus: "Corpus",
    result: Optional[SemanticResult] = None,
    config: Optional[SemanticConfig] = None,
) -> SemanticResult:
    """Attach Stage 3 columns onto a corpus in place.

    Args:
        corpus: Corpus already carrying the Stage 2 breakdown.
        result: A previously computed result; computed here when omitted.
        config: Configuration used when ``result`` is omitted.

    Returns:
        The :class:`SemanticResult` that was attached.

    Raises:
        ValueError: If the result does not align with the corpus index. Silent
            misalignment would attach one record's peer group to another.
    """
    frame = corpus.records
    computed = result if result is not None else SemanticLayer(config).run(corpus)

    if len(computed.frame) != len(frame):
        raise ValueError(
            f"Stage 3 produced {len(computed.frame)} rows for {len(frame)} records"
        )
    if not computed.frame.index.equals(frame.index):
        raise ValueError("Stage 3 index does not match the corpus index")

    for column in STAGE3_COLUMNS:
        frame[column] = computed.frame[column]

    LOGGER.info(
        "Attached %d Stage 3 column(s) to %d record(s); row order unchanged.",
        len(STAGE3_COLUMNS),
        len(frame),
    )
    return computed
