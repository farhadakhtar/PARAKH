"""``RiskLayer`` - the Stage 5 orchestrator.

    validate -> signal strength -> data quality -> uncertainty
             -> gate and compose -> band -> explain

Stage 5 recomputes nothing. It reads Stage 2's confidence, Stage 3's structure
and Stage 4's severity, and answers one question they deliberately do not: how
much is this worth acting on, given how much of it can be believed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.constants import (
    MIN_CONFIDENCE_FOR_RISK,
    R_HIGH,
    R_LOW,
    RISK_BREADTH_SATURATION,
    RISK_BREADTH_WEIGHT,
    RISK_CRITICAL_DEFICIT_DECAY,
    RISK_DUPLICATE_WEIGHT,
    RISK_EXTREME_WEIGHT,
    RISK_FLAGS,
    RISK_LOW_CONFIDENCE_PENALTY,
    RISK_TEMPORAL_HARD_FAIL_QUALITY,
    RISK_UNCERTAINTY_COVERAGE_WEIGHT,
    RISK_UNCERTAINTY_NO_NORM,
    RISK_UNCERTAINTY_NO_SEVERITY,
    RISK_UNCERTAINTY_UNREACHABLE_DUPLICATE,
    RISK_UNCERTAINTY_UNSTABLE_CELL,
    RISK_UNDEFINED_REASONS,
    STAGE5_CALIBRATION_REPORT,
    STAGE5_RISK_REPORT,
    STAGE5_VERSION,
)
from src.core.logger import get_logger
from src.stage5.calibration import compute_stage5_calibration_report
from src.stage5.components import (
    DataQuality,
    SignalStrength,
    Stage5InputError,
    Uncertainty,
    compute_data_quality,
    compute_signal_strength,
    compute_uncertainty,
    require_contract,
)
from src.stage5.explanation import build_risk_explanations
from src.stage5.risk import RiskResult, compute_risk
from src.utils.helpers import write_json

if TYPE_CHECKING:  # pragma: no cover
    from src.stage1.corpus import Corpus

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

#: The Stage 5 output contract, in a fixed order.
STAGE5_COLUMNS: Tuple[str, ...] = (
    # --- the score and its definedness --------------------------------------
    "risk_score",
    "risk_defined",
    "risk_defined_reason",
    # --- the three components, always reported alongside --------------------
    "risk_signal_strength",
    "risk_data_quality",
    "risk_uncertainty",
    # --- band (descriptive, not a decision) ---------------------------------
    "risk_flag",
    # --- explanation ---------------------------------------------------------
    "risk_explanation",
)

#: Component internals kept for audit and for the explanation's arithmetic.
STAGE5_DETAIL_COLUMNS: Tuple[str, ...] = (
    "risk_breadth",
    "risk_extreme",
    "risk_duplicate",
    "risk_component_floor",
    "risk_deficit_factor",
)


@dataclass(frozen=True)
class RiskConfig:
    """Stage 5 parameters. Every one is a judgement, none is fitted."""

    min_confidence: float = MIN_CONFIDENCE_FOR_RISK
    r_high: float = R_HIGH
    r_low: float = R_LOW
    breadth_weight: float = RISK_BREADTH_WEIGHT
    extreme_weight: float = RISK_EXTREME_WEIGHT
    duplicate_weight: float = RISK_DUPLICATE_WEIGHT
    breadth_saturation: int = RISK_BREADTH_SATURATION
    deficit_decay: float = RISK_CRITICAL_DEFICIT_DECAY
    hard_fail_quality: float = RISK_TEMPORAL_HARD_FAIL_QUALITY
    low_confidence_penalty: float = RISK_LOW_CONFIDENCE_PENALTY
    uncertainty_no_severity: float = RISK_UNCERTAINTY_NO_SEVERITY
    uncertainty_no_norm: float = RISK_UNCERTAINTY_NO_NORM
    uncertainty_unstable_cell: float = RISK_UNCERTAINTY_UNSTABLE_CELL
    uncertainty_coverage_weight: float = RISK_UNCERTAINTY_COVERAGE_WEIGHT
    uncertainty_unreachable_duplicate: float = RISK_UNCERTAINTY_UNREACHABLE_DUPLICATE
    compute_calibration: bool = True

    def __post_init__(self) -> None:
        """Reject a malformed configuration at construction."""
        if not 0.0 <= self.r_low <= self.r_high <= 1.0:
            raise ValueError("risk bands must satisfy 0 <= r_low <= r_high <= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0,1]")
        if self.duplicate_weight > RISK_DUPLICATE_WEIGHT:
            raise ValueError(
                "duplicate weight exceeds its cap; the duplicate signal may "
                "support a case, never make one"
            )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable echo."""
        return {
            "stage5_version": STAGE5_VERSION,
            "min_confidence": self.min_confidence,
            "r_high": self.r_high,
            "r_low": self.r_low,
            "signal_weights": {
                "breadth": self.breadth_weight,
                "extreme": self.extreme_weight,
                "duplicate": self.duplicate_weight,
            },
            "deficit_decay": self.deficit_decay,
            "hard_fail_quality": self.hard_fail_quality,
            "compute_calibration": self.compute_calibration,
        }


@dataclass(frozen=True)
class Stage5Result:
    """Everything Stage 5 produced."""

    frame: pd.DataFrame
    strength: SignalStrength
    quality: DataQuality
    uncertainty: Uncertainty
    risk: RiskResult
    config: RiskConfig
    calibration: Optional[Dict[str, Any]] = None
    elapsed_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.frame)

    def band(self, flag: str) -> pd.DataFrame:
        """Records in one risk band.

        Args:
            flag: One of :data:`RISK_FLAGS`.

        Returns:
            A view of the Stage 5 frame.

        Raises:
            ValueError: On an unknown band.
        """
        if flag not in RISK_FLAGS:
            raise ValueError(f"unknown risk flag {flag!r}; expected one of {RISK_FLAGS}")
        return self.frame.loc[self.frame["risk_flag"] == flag]

    def explain(self, row: Any) -> str:
        """The stored explanation for one record; recomputes nothing."""
        if row not in self.frame.index:
            raise KeyError(f"row {row!r} is not in the frame index")
        return str(self.frame.loc[row, "risk_explanation"])

    def report(self) -> Dict[str, Any]:
        """Corpus-level Stage 5 summary. Carries no wall-clock value."""
        return {
            "stage5_version": STAGE5_VERSION,
            "n_records": len(self.frame),
            "_note": (
                "Risk is an estimate under uncertainty. NO record is labelled "
                "fraud. Bands are descriptive; Stage 6 owns routing. An "
                "undefined risk means unmeasured, never safe."
            ),
            "config": self.config.to_dict(),
            "risk": self.risk.to_dict(),
            "components": {
                "signal_strength": self.strength.to_dict(),
                "data_quality": self.quality.to_dict(),
                "uncertainty": self.uncertainty.to_dict(),
            },
        }

    def save_reports(self, output_dir: PathLike) -> Dict[str, Path]:
        """Write the Stage 5 reports as JSON."""
        directory = Path(output_dir)
        written = {
            "risk_report": write_json(self.report(), directory / STAGE5_RISK_REPORT)
        }
        if self.calibration is not None:
            written["calibration"] = write_json(
                self.calibration, directory / STAGE5_CALIBRATION_REPORT
            )
        LOGGER.info("Wrote %d Stage 5 report(s) to %s", len(written), directory)
        return written


class RiskLayer:
    """Stage 5: convert deviation signals into risk conditional on evidence."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        """Build the layer.

        Args:
            config: Stage 5 parameters; defaults to the named constants.
        """
        self.config = config or RiskConfig()

    def __repr__(self) -> str:
        return f"<RiskLayer {STAGE5_VERSION}>"

    def _frame_of(self, source: Union["Corpus", pd.DataFrame]) -> pd.DataFrame:
        """Accept a Corpus or a bare frame."""
        if isinstance(source, pd.DataFrame):
            return source
        records = getattr(source, "records", None)
        if isinstance(records, pd.DataFrame):
            return records
        raise TypeError(f"Expected a Corpus or DataFrame, got {type(source).__name__}")

    def run(self, source: Union["Corpus", pd.DataFrame]) -> Stage5Result:
        """Execute the Stage 5 pipeline.

        Args:
            source: A corpus already carrying Stage 2, 3 and 4 output.

        Returns:
            A :class:`Stage5Result`; row count, order and index preserved.

        Raises:
            Stage5InputError: If the upstream contract is incomplete.
        """
        frame = self._frame_of(source)
        require_contract(frame)
        config = self.config
        started = time.perf_counter()

        strength = compute_signal_strength(
            frame,
            breadth_weight=config.breadth_weight,
            extreme_weight=config.extreme_weight,
            duplicate_weight=config.duplicate_weight,
            saturation=config.breadth_saturation,
        )
        quality = compute_data_quality(
            frame,
            min_confidence=config.min_confidence,
            deficit_decay=config.deficit_decay,
            hard_fail_quality=config.hard_fail_quality,
            low_confidence_penalty=config.low_confidence_penalty,
        )
        uncertainty = compute_uncertainty(
            frame,
            no_severity=config.uncertainty_no_severity,
            no_norm=config.uncertainty_no_norm,
            unstable_cell=config.uncertainty_unstable_cell,
            coverage_weight=config.uncertainty_coverage_weight,
            unreachable_duplicate=config.uncertainty_unreachable_duplicate,
        )
        risk = compute_risk(
            frame,
            signal_strength=strength.value,
            data_quality=quality.value,
            uncertainty=uncertainty.value,
            min_confidence=config.min_confidence,
            r_high=config.r_high,
            r_low=config.r_low,
        )

        output = pd.DataFrame(index=frame.index)
        output["risk_score"] = risk.score
        output["risk_defined"] = risk.defined
        output["risk_defined_reason"] = risk.reason
        output["risk_signal_strength"] = strength.value
        output["risk_data_quality"] = quality.value
        output["risk_uncertainty"] = uncertainty.value
        output["risk_flag"] = risk.flag
        output["risk_breadth"] = strength.breadth
        output["risk_extreme"] = strength.extreme
        output["risk_duplicate"] = strength.duplicate
        output["risk_component_floor"] = quality.component_floor
        output["risk_deficit_factor"] = quality.deficit_factor

        context = [
            name
            for name in (
                "confidence",
                "severity_score",
                "peer_cell_stable",
                "valid_signal_count",
                "duplicate_flag",
                "temporal_hard_fail",
                # The contract column. The per-type booleans live only on the
                # Stage 4 result frame, so this is the form that is always here.
                "anomaly_types",
            )
            if name in frame.columns
        ]
        context += [name for name in frame.columns if name.startswith("type_")]
        output["risk_explanation"] = build_risk_explanations(
            output.join(frame[context])
        )

        missing = [name for name in STAGE5_COLUMNS if name not in output.columns]
        if missing:
            raise RuntimeError(f"Stage 5 contract incomplete: missing {missing!r}")

        self._assert_guarantees(output, frame)
        elapsed = time.perf_counter() - started

        calibration: Optional[Dict[str, Any]] = None
        if config.compute_calibration:
            calibration = compute_stage5_calibration_report(
                output.join(
                    frame[[c for c in frame.columns if c not in output.columns]]
                )
            )

        LOGGER.info(
            "Stage 5 complete in %.2fs: %s",
            elapsed,
            {k: int(v) for k, v in output["risk_flag"].value_counts().items()},
        )

        return Stage5Result(
            frame=output,
            strength=strength,
            quality=quality,
            uncertainty=uncertainty,
            risk=risk,
            config=config,
            calibration=calibration,
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _assert_guarantees(output: pd.DataFrame, frame: pd.DataFrame) -> None:
        """Enforce the Stage 5 invariants before returning.

        These are the six non-negotiables. They are assertions rather than
        validation because a violation is a defect in this module, not bad
        input - the input was already validated.
        """
        defined = output["risk_defined"].to_numpy(dtype=bool)

        # 1. Undefined risk is NaN, never 0.
        assert output.loc[~defined, "risk_score"].isna().all(), (
            "an undefined risk carries a score"
        )
        assert output.loc[defined, "risk_score"].notna().all(), (
            "a defined risk carries no score"
        )

        # 2. Below the confidence gate cannot be high risk.
        confidence = pd.to_numeric(frame["confidence"], errors="coerce").to_numpy(
            dtype="float64", na_value=0.0
        )
        high = output["risk_flag"].to_numpy() == "high_risk"
        assert not bool((high & (confidence < MIN_CONFIDENCE_FOR_RISK)).any()), (
            "a record below the confidence gate was banded high_risk"
        )

        # 3. Undefined severity implies undefined risk.
        severity_defined = frame["severity_defined"].fillna(False).to_numpy(dtype=bool)
        assert not bool((defined & ~severity_defined).any()), (
            "a risk score was produced without a severity"
        )

        # 4. Everything is finite or NaN - never inf.
        for column in (
            "risk_score",
            "risk_signal_strength",
            "risk_data_quality",
            "risk_uncertainty",
        ):
            values = output[column].to_numpy(dtype="float64")
            assert not np.isinf(values).any(), f"{column} contains an infinity"
            present = values[np.isfinite(values)]
            assert bool(((present >= 0.0) & (present <= 1.0)).all()), (
                f"{column} escaped [0,1]"
            )

        # 5. No NaN on the descriptive surface.
        for column in ("risk_defined_reason", "risk_flag", "risk_explanation"):
            assert output[column].notna().all(), f"{column} contains NaN"

        # 6. Declared vocabularies only.
        assert set(output["risk_flag"].unique()) <= set(RISK_FLAGS)
        assert set(output["risk_defined_reason"].unique()) <= set(RISK_UNDEFINED_REASONS)


def attach_risk(
    corpus: "Corpus",
    result: Optional[Stage5Result] = None,
    config: Optional[RiskConfig] = None,
) -> Stage5Result:
    """Attach Stage 5 columns onto a corpus in place.

    Args:
        corpus: Corpus already carrying Stage 2, 3 and 4 output.
        result: A previously computed result; computed here when omitted.
        config: Configuration used when ``result`` is omitted.

    Returns:
        The :class:`Stage5Result` that was attached.

    Raises:
        ValueError: If the result does not align with the corpus index.
    """
    frame = corpus.records
    computed = result if result is not None else RiskLayer(config).run(corpus)

    if len(computed.frame) != len(frame):
        raise ValueError(
            f"Stage 5 produced {len(computed.frame)} rows for {len(frame)} records"
        )
    if not computed.frame.index.equals(frame.index):
        raise ValueError("Stage 5 index does not match the corpus index")

    for column in (*STAGE5_COLUMNS, *STAGE5_DETAIL_COLUMNS):
        frame[column] = computed.frame[column]

    LOGGER.info(
        "Attached %d Stage 5 column(s) to %d record(s); row order unchanged.",
        len(STAGE5_COLUMNS) + len(STAGE5_DETAIL_COLUMNS),
        len(frame),
    )
    return computed
