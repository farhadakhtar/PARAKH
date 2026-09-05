"""Stage 8 - Calibration Layer. SCAFFOLD ONLY.

Designed, not implemented. With no real labelled outcomes there is no
calibration, and the honest answer is None rather than a number fitted to
synthetic data the system generated for itself.
"""

from src.stage8.calibration import (
    CALIBRATION_COLUMNS,
    CALIBRATION_LABEL_SCHEMA,
    CalibrationDataset,
    CalibrationRefusedError,
    CalibrationResult,
    calibrate,
    calibration_design,
)

__all__ = [
    "CALIBRATION_COLUMNS",
    "CALIBRATION_LABEL_SCHEMA",
    "CalibrationDataset",
    "CalibrationRefusedError",
    "CalibrationResult",
    "calibrate",
    "calibration_design",
]
