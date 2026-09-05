"""Stage 8 - Calibration Layer. **SCAFFOLD ONLY: fits nothing.**

The job is to turn a rank into a probability: ``risk_score`` orders records,
and a calibrated value would say *how often a record like this turns out to be
a real finding*. Those are different claims, and only the second is a
probability.

Why this file computes nothing yet
----------------------------------
Calibration needs labelled outcomes - records a human investigated and
resolved. This system has never seen one. Every number it has produced came
from a synthetic corpus **it generated itself**, with defects it injected on
purpose. Fitting a calibration curve to that would measure the generator, and
would hand back a number carrying the authority of the word "probability" and
none of the evidence.

So :func:`calibrate` returns ``None`` and says why. That is the whole point of
the scaffold: the interface exists, the refusal is the implementation, and
nothing downstream can accidentally receive a fabricated probability while the
labels are missing.

What is designed
----------------
* Three fitting methods, chosen because each is monotone and adds no
  information the labels do not contain (:data:`CALIBRATION_METHODS`).
* A label schema (:class:`CalibrationDataset`) that is explicit about
  provenance, so synthetic data cannot be passed off as real.
* Three evaluation metrics, because a single one hides different failures:
  Brier score conflates calibration with discrimination, ECE hides where the
  error lives, and only the curve shows the shape.
* Gates on volume and class balance, below which fitting is noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    CALIBRATION_METHODS,
    CALIBRATION_MIN_LABELS,
    CALIBRATION_MIN_PER_CLASS,
    CALIBRATION_REFUSAL_NOTE,
    CALIBRATION_STATUSES,
    STAGE8_VERSION,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Columns Stage 8 would add. Named now so a consumer can code against the
#: shape before the implementation exists.
CALIBRATION_COLUMNS: Tuple[str, ...] = (
    "calibrated_risk",
    "calibration_method",
    "calibration_confidence",
    "calibration_status",
)

#: The label schema. Every field is required, and ``provenance`` is required
#: precisely so that synthetic data cannot be supplied by accident.
CALIBRATION_LABEL_SCHEMA: Mapping[str, str] = {
    "record_id": "Unique record identifier, matching the corpus index.",
    "risk_score": "The uncalibrated Stage 5 score this record received.",
    "outcome": (
        "The resolved ground truth: True where investigation confirmed a real "
        "finding, False where it did not. Never a prediction, never a proxy."
    ),
    "resolved_at": "ISO8601 date the outcome was established.",
    "resolver": "Who established it - a team or role, for accountability.",
    "provenance": (
        "'real' or 'synthetic'. Required, not inferred. Calibration refuses "
        "'synthetic' outright."
    ),
}


class CalibrationRefusedError(RuntimeError):
    """Raised when calibration is attempted on data that cannot support it."""


@dataclass(frozen=True)
class CalibrationDataset:
    """Labelled outcomes offered for calibration.

    Attributes:
        labels: One row per resolved record, following
            :data:`CALIBRATION_LABEL_SCHEMA`.
        provenance: ``"real"`` or ``"synthetic"``. Declared by the caller
            rather than sniffed: a system that guesses at provenance will
            eventually guess wrong in the direction that flatters it.
    """

    labels: pd.DataFrame
    provenance: str = "synthetic"

    def validate(self) -> List[str]:
        """Every reason this dataset cannot be calibrated on.

        Returns:
            A list of problems, empty when the dataset is usable. Returns all
            of them rather than the first: a caller fixing a dataset should
            see the whole gap, not discover it one run at a time.
        """
        problems: List[str] = []
        if self.provenance != "real":
            problems.append(
                f"provenance is {self.provenance!r}; calibration requires real "
                "labelled outcomes. Synthetic data measures the generator."
            )
        missing = [
            name for name in CALIBRATION_LABEL_SCHEMA if name not in self.labels.columns
        ]
        if missing:
            problems.append(f"label schema is missing {missing!r}")
        if "outcome" in self.labels.columns:
            outcomes = self.labels["outcome"].dropna()
            total = int(len(outcomes))
            if total < CALIBRATION_MIN_LABELS:
                problems.append(
                    f"{total} labelled outcome(s); at least "
                    f"{CALIBRATION_MIN_LABELS} are needed before a curve is "
                    "anything but noise"
                )
            positives = int(outcomes.astype(bool).sum())
            negatives = total - positives
            if min(positives, negatives) < CALIBRATION_MIN_PER_CLASS:
                problems.append(
                    f"class balance {positives} positive / {negatives} negative; "
                    f"at least {CALIBRATION_MIN_PER_CLASS} of each are needed"
                )
        return problems


@dataclass(frozen=True)
class CalibrationResult:
    """What Stage 8 produced. ``calibrated_risk`` is None until labels exist."""

    calibrated_risk: Optional[pd.Series]
    method: Optional[str]
    confidence: Optional[float]
    status: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "stage8_version": STAGE8_VERSION,
            "calibration_status": self.status,
            "calibration_method": self.method,
            "calibration_confidence": self.confidence,
            "n_calibrated": int(self.calibrated_risk.notna().sum())
            if self.calibrated_risk is not None
            else 0,
            **self.diagnostics,
        }


def calibrate(
    frame: pd.DataFrame,
    dataset: Optional[CalibrationDataset] = None,
    method: str = "isotonic_regression",
) -> CalibrationResult:
    """Calibrate, or refuse and say why. **Currently always refuses.**

    Args:
        frame: Corpus frame carrying ``risk_score``.
        dataset: Labelled outcomes. None means none exist.
        method: One of :data:`CALIBRATION_METHODS`.

    Returns:
        A :class:`CalibrationResult`. ``calibrated_risk`` is None unless real
        labels were supplied and passed every gate - which cannot happen yet,
        because no fitting is implemented.

    Raises:
        ValueError: On an unknown method. Refusing to calibrate is a result;
            being asked for a method that does not exist is a caller error.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"unknown calibration method {method!r}; expected one of "
            f"{CALIBRATION_METHODS}"
        )

    if dataset is None:
        LOGGER.warning("Stage 8: no labelled outcomes supplied; refusing.")
        return CalibrationResult(
            calibrated_risk=None,
            method=None,
            confidence=None,
            status="UNAVAILABLE",
            diagnostics={"_note": CALIBRATION_REFUSAL_NOTE, "problems": ["no dataset"]},
        )

    problems = dataset.validate()
    if problems:
        status = (
            "REFUSED_SYNTHETIC"
            if dataset.provenance != "real"
            else "INSUFFICIENT_LABELS"
        )
        LOGGER.warning("Stage 8: refusing to calibrate - %s", problems)
        return CalibrationResult(
            calibrated_risk=None,
            method=None,
            confidence=None,
            status=status,
            diagnostics={"_note": CALIBRATION_REFUSAL_NOTE, "problems": problems},
        )

    # Reachable only with real, sufficient, balanced labels - which do not
    # exist yet. Refusing here rather than returning an untested fit keeps the
    # scaffold honest: the contract is defined, the implementation is not.
    raise NotImplementedError(
        "Stage 8 fitting is not implemented. A valid real-label dataset was "
        "supplied, which is the first time this branch has been reachable. "
        f"Implement {method} and its evaluation metrics before use; do not "
        "return a probability from an unvalidated fit."
    )


def calibration_design() -> Dict[str, Any]:
    """The design, as data - so it can be reviewed without reading the code."""
    return {
        "stage8_version": STAGE8_VERSION,
        "status": "SCAFFOLD - designed, not implemented",
        "objective": "Convert an uncalibrated rank into a calibrated probability.",
        "inputs": ["risk_score", "action_class", "decision_class", "labelled outcomes"],
        "outputs": list(CALIBRATION_COLUMNS),
        "methods": {
            "platt_scaling": (
                "Logistic fit on the score. Two parameters, so it survives "
                "small samples, but it assumes a sigmoid shape the data may "
                "not have."
            ),
            "isotonic_regression": (
                "Monotone step fit. Assumes only that higher score means "
                "higher probability - the one thing the ranking already "
                "claims - but needs more labels and can overfit the tails."
            ),
            "histogram_binning": (
                "Empirical rate per bin. The most transparent and the easiest "
                "to audit; resolution is limited by bin width."
            ),
        },
        "label_schema": dict(CALIBRATION_LABEL_SCHEMA),
        "gates": {
            "min_labels": CALIBRATION_MIN_LABELS,
            "min_per_class": CALIBRATION_MIN_PER_CLASS,
            "provenance_must_be": "real",
        },
        "evaluation": {
            "brier_score": (
                "Mean squared error of the probability. Conflates calibration "
                "with discrimination, so never reported alone."
            ),
            "calibration_curve": (
                "Predicted versus observed rate per bin. The only one that "
                "shows the SHAPE of the error."
            ),
            "expected_calibration_error": (
                "Weighted mean gap between predicted and observed. A single "
                "number, so it hides where the error lives - report beside "
                "the curve, never instead of it."
            ),
        },
        "safety_rules": [
            "No real labels -> calibrated_risk is None, status UNAVAILABLE.",
            "Synthetic provenance -> refused outright, status REFUSED_SYNTHETIC.",
            "Below the volume or balance gate -> status INSUFFICIENT_LABELS.",
            "A fit is never returned without its evaluation metrics.",
            "Calibration never overwrites risk_score; it adds a column.",
        ],
        "_critical": CALIBRATION_REFUSAL_NOTE,
    }
