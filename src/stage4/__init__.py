"""Stage 4 - Contextual Anomaly Interpretation.

Consumes Stage 2 confidence and Stage 3 peer deviations; recomputes nothing.
Decides what the deviations MEAN under the confidence available, and routes.

Public surface:

* :class:`~src.stage4.pipeline.AnomalyLayer` - the pipeline
* :func:`~src.stage4.pipeline.attach_anomalies` - integration with Corpus
* :data:`~src.stage4.pipeline.STAGE4_COLUMNS` - the output contract
"""

from src.stage4.calibration import (
    DEVIATION_COLUMNS,
    DuplicateDiagnostics,
    compute_duplicate_diagnostics,
    compute_stage4_calibration_report,
    describe_defined,
)
from src.stage4.anomaly import (
    DEFINED_REASON,
    REQUIRED_COLUMNS,
    AnomalySignals,
    SignalValidation,
    Stage4InputError,
    build_signals,
    classify_types,
    require_contract,
    validate_signals,
)
from src.stage4.decision import (
    DecisionResult,
    SeverityDefinedness,
    SeverityResult,
    compute_severity,
    route,
    severity_definedness,
)
from src.stage4.explanation import (
    REASON_PHRASES,
    TYPE_PHRASES,
    build_explanations,
    explain_record,
)
from src.stage4.pipeline import (
    OPTIONAL_STAGE4_COLUMNS,
    PASSTHROUGH_COLUMNS,
    STAGE4_COLUMNS,
    AnomalyConfig,
    AnomalyLayer,
    AnomalyResult,
    attach_anomalies,
)

__all__ = [
    "DEFINED_REASON",
    "DEVIATION_COLUMNS",
    "DuplicateDiagnostics",
    "SeverityDefinedness",
    "compute_duplicate_diagnostics",
    "compute_stage4_calibration_report",
    "describe_defined",
    "severity_definedness",
    "OPTIONAL_STAGE4_COLUMNS",
    "PASSTHROUGH_COLUMNS",
    "REASON_PHRASES",
    "REQUIRED_COLUMNS",
    "STAGE4_COLUMNS",
    "TYPE_PHRASES",
    "AnomalyConfig",
    "AnomalyLayer",
    "AnomalyResult",
    "AnomalySignals",
    "DecisionResult",
    "SeverityResult",
    "SignalValidation",
    "Stage4InputError",
    "attach_anomalies",
    "build_explanations",
    "build_signals",
    "classify_types",
    "compute_severity",
    "explain_record",
    "require_contract",
    "route",
    "validate_signals",
]
