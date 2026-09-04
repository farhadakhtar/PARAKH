"""Stage 2 - Evidentiary Confidence Engine.

Turns Stage 1's structured unreliability into a number:

    C(r) = exp( w1 log C_comp + w2 log C_temp + w3 log C_recon )

Public surface:

* :class:`~src.stage2.confidence.ConfidenceModel` - the engine
* :func:`~src.stage2.confidence.compute_confidence` - functional entry point
* :func:`~src.stage2.confidence.attach_confidence` - integration with Corpus
* the three component functions, each independently usable and explainable
"""

from src.stage2.completeness import (
    CompletenessResult,
    FieldWeights,
    bernoulli_entropy,
    compute_completeness,
    compute_completeness_result,
    compute_field_weights,
    credit_matrix,
    resolve_reasons,
    value_entropy,
)
from src.stage2.confidence import (
    COMPONENT_COLUMNS,
    CONFIDENCE_COLUMN,
    ConfidenceConfig,
    ConfidenceModel,
    ConfidenceReport,
    ConfidenceResult,
    attach_confidence,
    compute_confidence,
    confidence_summary_frame,
    log_space_geometric_mean,
)
from src.stage2.reconciliation import (
    ReconciliationResult,
    compute_reconciliation,
    compute_reconciliation_result,
)
from src.stage2.temporal import (
    TemporalResult,
    compute_temporal,
    compute_temporal_result,
)

__all__ = [
    "COMPONENT_COLUMNS",
    "CONFIDENCE_COLUMN",
    "CompletenessResult",
    "ConfidenceConfig",
    "ConfidenceModel",
    "ConfidenceReport",
    "ConfidenceResult",
    "FieldWeights",
    "ReconciliationResult",
    "TemporalResult",
    "attach_confidence",
    "bernoulli_entropy",
    "compute_completeness",
    "compute_completeness_result",
    "compute_confidence",
    "compute_field_weights",
    "compute_reconciliation",
    "compute_reconciliation_result",
    "compute_temporal",
    "compute_temporal_result",
    "confidence_summary_frame",
    "credit_matrix",
    "log_space_geometric_mean",
    "resolve_reasons",
    "value_entropy",
]
