"""Stage 3 - Semantic Layer & Peer Cell Formation.

Builds the comparison groups every downstream signal is measured against:

    peer_cell = (semantic cluster k, cost stratum s)

Stage 3 produces **structure**. It ends at deviations from peer norms and does
not score or classify anomalies - Stage 4 owns that.

Public surface:

* :class:`~src.stage3.pipeline.SemanticLayer` - the pipeline
* :func:`~src.stage3.pipeline.attach_structure` - integration with Corpus
* :data:`~src.stage3.pipeline.STAGE3_COLUMNS` - the Stage 4 column contract
"""

from src.stage3.clustering import ClusterResult, cluster_records
from src.stage3.deviations import (
    DEVIATION_SPECS,
    UNDEFINED_REASONS,
    DeviationResult,
    compute_deviations,
)
from src.stage3.duplicate_detection import (
    NO_DUPLICATE_GROUP,
    DuplicateResult,
    detect_duplicates,
)
from src.stage3.embedding import (
    TextEmbedding,
    build_stopwords,
    embed_work_names,
    normalize_work_text,
)
from src.stage3.explanation import REASON_TEXT, build_explanation_inputs
from src.stage3.features import (
    GATING_FEATURES,
    GROUPING_FEATURES,
    TESTING_FEATURES,
    FeatureTable,
    build_feature_table,
    compute_duration_days,
)
from src.stage3.peer_cells import (
    PEER_STAT_FIELDS,
    PeerCellResult,
    PeerStatistics,
    build_reference_mask,
    compute_peer_statistics,
    form_peer_cells,
)
from src.stage3.pipeline import (
    STAGE3_COLUMNS,
    SemanticConfig,
    SemanticLayer,
    SemanticResult,
    attach_structure,
)
from src.stage3.stratification import StratificationResult, stratify_cost

__all__ = [
    "DEVIATION_SPECS",
    "GATING_FEATURES",
    "GROUPING_FEATURES",
    "NO_DUPLICATE_GROUP",
    "PEER_STAT_FIELDS",
    "REASON_TEXT",
    "STAGE3_COLUMNS",
    "TESTING_FEATURES",
    "UNDEFINED_REASONS",
    "ClusterResult",
    "DeviationResult",
    "DuplicateResult",
    "FeatureTable",
    "PeerCellResult",
    "PeerStatistics",
    "SemanticConfig",
    "SemanticLayer",
    "SemanticResult",
    "StratificationResult",
    "TextEmbedding",
    "attach_structure",
    "build_explanation_inputs",
    "build_feature_table",
    "build_reference_mask",
    "build_stopwords",
    "cluster_records",
    "compute_deviations",
    "compute_duration_days",
    "compute_peer_statistics",
    "detect_duplicates",
    "embed_work_names",
    "form_peer_cells",
    "normalize_work_text",
    "stratify_cost",
]
