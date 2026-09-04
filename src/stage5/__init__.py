"""Stage 5 - Risk Scoring Layer.

Converts Stage 4's deviation signals into actionable risk under uncertainty.
Recomputes nothing. Labels no record as fraud.

    risk_score = signal_strength x data_quality x (1 - uncertainty)

Public surface:

* :class:`~src.stage5.pipeline.RiskLayer` - the pipeline
* :func:`~src.stage5.pipeline.attach_risk` - integration with Corpus
* :data:`~src.stage5.pipeline.STAGE5_COLUMNS` - the output contract
"""

from src.stage5.calibration import (
    COMPONENT_COLUMNS,
    compute_stage5_calibration_report,
)
from src.stage5.components import (
    MAX_SIGNAL_COVERAGE,
    OPTIONAL_COLUMNS,
    REQUIRED_STAGE2,
    REQUIRED_STAGE3,
    REQUIRED_STAGE4,
    DataQuality,
    SignalStrength,
    Stage5InputError,
    Uncertainty,
    compute_data_quality,
    compute_signal_strength,
    compute_uncertainty,
    require_contract,
)
from src.stage5.explanation import (
    TYPE_PHRASES,
    UNDEFINED_PHRASES,
    build_risk_explanations,
    explain_risk,
)
from src.stage5.pipeline import (
    STAGE5_COLUMNS,
    STAGE5_DETAIL_COLUMNS,
    RiskConfig,
    RiskLayer,
    Stage5Result,
    attach_risk,
)
from src.stage5.risk import RiskResult, compute_risk

__all__ = [
    "COMPONENT_COLUMNS",
    "MAX_SIGNAL_COVERAGE",
    "OPTIONAL_COLUMNS",
    "REQUIRED_STAGE2",
    "REQUIRED_STAGE3",
    "REQUIRED_STAGE4",
    "STAGE5_COLUMNS",
    "STAGE5_DETAIL_COLUMNS",
    "TYPE_PHRASES",
    "UNDEFINED_PHRASES",
    "DataQuality",
    "RiskConfig",
    "RiskLayer",
    "RiskResult",
    "SignalStrength",
    "Stage5InputError",
    "Stage5Result",
    "Uncertainty",
    "attach_risk",
    "build_risk_explanations",
    "compute_data_quality",
    "compute_risk",
    "compute_signal_strength",
    "compute_stage5_calibration_report",
    "compute_uncertainty",
    "explain_risk",
    "require_contract",
]
