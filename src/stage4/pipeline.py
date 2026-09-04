"""``AnomalyLayer`` - the Stage 4 orchestrator.

    validate signals -> assemble -> classify types -> severity -> route
                     -> explain

Stage 4 recomputes nothing from Stages 1-3. It reads their outputs, decides
what the deviations mean under the confidence available, and produces a
structured, explainable result.

Naming, deliberately
--------------------
The composed score is ``severity_score``, never ``risk_score``: Stage5.md owns
``R(r)``. The routing emits ``INSUFFICIENT_CONTEXT`` rather than Stage6.md's
``CLEAR``, so the two vocabularies cannot be confused and Stage 6 remains free
to supersede this provisional triage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.constants import (
    ANOMALY_TYPES,
    CONFIDENCE_GATE_THRESHOLD,
    DECISION_CLASSES,
    SEVERITY_DEFINED_REASONS,
    SEVERITY_WEIGHTS,
    STAGE4_ANOMALY_REPORT,
    STAGE4_CALIBRATION_REPORT,
    STAGE4_DUPLICATE_DIAGNOSTICS,
    STAGE4_VERSION,
    Z_INVESTIGATE_THRESHOLD,
    Z_SEVERITY_SCALE,
    Z_TYPE_THRESHOLD,
)
from src.core.logger import get_logger
from src.stage4.anomaly import (
    AnomalySignals,
    Stage4InputError,
    build_signals,
    classify_types,
    require_contract,
)
from src.stage4.calibration import (
    DuplicateDiagnostics,
    compute_duplicate_diagnostics,
    compute_stage4_calibration_report,
)
from src.stage4.decision import (
    DecisionResult,
    SeverityDefinedness,
    SeverityResult,
    compute_severity,
    route,
    severity_definedness,
)
from src.stage4.explanation import build_explanations
from src.utils.helpers import write_json

if TYPE_CHECKING:  # pragma: no cover
    from src.stage1.corpus import Corpus

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

#: The Stage 4 output contract, in a fixed order. Columns are only ever added.
STAGE4_COLUMNS: Tuple[str, ...] = (
    # --- core ---------------------------------------------------------------
    "anomaly_types",
    "anomaly_count",
    "severity_score",
    # Added by the hardening pass. severity_score itself is untouched; these
    # only state what was already true of it.
    "severity_defined",
    "severity_defined_reason",
    # --- decision -----------------------------------------------------------
    "decision_class",
    "decision_reason",
    # --- signal breakdown ---------------------------------------------------
    "z_cost",
    "cost_scope",
    "z_spend",
    "z_duration",
    "valid_signal_count",
    # --- confidence integration --------------------------------------------
    "confidence_flag",
    # --- duplicate (supporting only) ---------------------------------------
    "duplicate_flag_stage4",
    # --- explanation --------------------------------------------------------
    "explanation_text",
)

#: Produced only when the duplicate diagnostics run, so never part of the
#: mandatory contract - a consumer must treat their absence as "not measured",
#: which is exactly what it means.
OPTIONAL_STAGE4_COLUMNS: Tuple[str, ...] = (
    "duplicate_cosine",
    "duplicate_reachable",
)

#: Columns Stage 4 reads but does not own; already present from Stages 2-3 and
#: listed in the brief's output structure. Not re-emitted, to avoid two columns
#: of the same name disagreeing.
PASSTHROUGH_COLUMNS: Tuple[str, ...] = (
    "confidence",
    "peer_cell_stable",
    "cluster_has_norm",
    "duplicate_score",
    "duplicate_flag",
)


@dataclass(frozen=True)
class AnomalyConfig:
    """Stage 4 parameters. Every one is a judgement, not an estimate."""

    confidence_threshold: float = CONFIDENCE_GATE_THRESHOLD
    z_type_threshold: float = Z_TYPE_THRESHOLD
    z_investigate_threshold: float = Z_INVESTIGATE_THRESHOLD
    z_severity_scale: float = Z_SEVERITY_SCALE
    severity_weights: Dict[str, float] = field(
        default_factory=lambda: dict(SEVERITY_WEIGHTS)
    )
    #: Measurement passes. Both are descriptive and neither can alter a
    #: decision, so they are opt-in purely on cost, not on safety.
    compute_calibration: bool = True
    compute_duplicate_diagnostics: bool = False

    def __post_init__(self) -> None:
        """Reject a malformed configuration at construction."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0,1]")
        if self.z_type_threshold <= 0 or self.z_investigate_threshold <= 0:
            raise ValueError("z thresholds must be positive")
        if self.z_investigate_threshold < self.z_type_threshold:
            raise ValueError(
                "z_investigate_threshold must be >= z_type_threshold: naming an "
                "anomaly must be a lower bar than escalating one"
            )
        if self.z_severity_scale <= 0:
            raise ValueError("z_severity_scale must be positive")
        if any(weight < 0 for weight in self.severity_weights.values()):
            raise ValueError("severity weights must be non-negative")
        if sum(self.severity_weights.values()) <= 0:
            raise ValueError("severity weights must not all be zero")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable echo."""
        return {
            "stage4_version": STAGE4_VERSION,
            "confidence_threshold": self.confidence_threshold,
            "z_type_threshold": self.z_type_threshold,
            "z_investigate_threshold": self.z_investigate_threshold,
            "z_severity_scale": self.z_severity_scale,
            "severity_weights": dict(self.severity_weights),
            "compute_calibration": self.compute_calibration,
            "compute_duplicate_diagnostics": self.compute_duplicate_diagnostics,
        }


@dataclass(frozen=True)
class AnomalyResult:
    """Everything Stage 4 produced."""

    frame: pd.DataFrame
    signals: AnomalySignals
    types: pd.DataFrame
    severity: SeverityResult
    decision: DecisionResult
    config: AnomalyConfig
    definedness: Optional[SeverityDefinedness] = None
    calibration: Optional[Dict[str, Any]] = None
    duplicates: Optional[DuplicateDiagnostics] = None
    elapsed_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.frame)

    def queue(self, decision_class: str) -> pd.DataFrame:
        """Records routed to one decision class.

        Args:
            decision_class: One of :data:`DECISION_CLASSES`.

        Returns:
            A view of the Stage 4 frame.

        Raises:
            ValueError: On an unknown class.
        """
        if decision_class not in DECISION_CLASSES:
            raise ValueError(
                f"unknown decision class {decision_class!r}; "
                f"expected one of {DECISION_CLASSES}"
            )
        return self.frame.loc[self.frame["decision_class"] == decision_class]

    def explain(self, row: Any) -> str:
        """The stored explanation for one record; recomputes nothing."""
        if row not in self.frame.index:
            raise KeyError(f"row {row!r} is not in the frame index")
        return str(self.frame.loc[row, "explanation_text"])

    def report(self) -> Dict[str, Any]:
        """Corpus-level Stage 4 summary. Carries no wall-clock value."""
        frame = self.frame
        type_counts = {
            name: int(self.types[name].sum()) for name in ANOMALY_TYPES
        }
        return {
            "stage4_version": STAGE4_VERSION,
            "n_records": len(frame),
            "_note": (
                "Provisional triage. severity_score is NOT Stage 5's R(r), and "
                "decision_class is superseded by Stage 6's routing. Severity is "
                "NaN - never 0 - where no signal was usable."
            ),
            "config": self.config.to_dict(),
            "decision": self.decision.to_dict(),
            "anomaly_types": type_counts,
            "anomaly_count": frame["anomaly_count"].value_counts().sort_index().to_dict(),
            "severity": {
                **self.severity.to_dict(),
                **(
                    {"definedness": self.definedness.to_dict()}
                    if self.definedness is not None
                    else {}
                ),
            },
            "signals": self.signals.to_dict(),
        }

    def save_reports(self, output_dir: PathLike) -> Dict[str, Path]:
        """Write the Stage 4 report as JSON."""
        directory = Path(output_dir)
        written = {
            "anomaly_report": write_json(
                self.report(), directory / STAGE4_ANOMALY_REPORT
            )
        }
        if self.calibration is not None:
            written["calibration"] = write_json(
                self.calibration, directory / STAGE4_CALIBRATION_REPORT
            )
        if self.duplicates is not None:
            written["duplicate_diagnostics"] = write_json(
                self.duplicates.to_dict(), directory / STAGE4_DUPLICATE_DIAGNOSTICS
            )
        LOGGER.info("Wrote %d Stage 4 report(s) to %s", len(written), directory)
        return written


class AnomalyLayer:
    """Stage 4: interpret Stage 3's deviations under Stage 2's confidence."""

    def __init__(self, config: Optional[AnomalyConfig] = None) -> None:
        """Build the layer.

        Args:
            config: Stage 4 parameters; defaults to the named constants.
        """
        self.config = config or AnomalyConfig()

    def __repr__(self) -> str:
        return f"<AnomalyLayer {STAGE4_VERSION}>"

    def _frame_of(self, source: Union["Corpus", pd.DataFrame]) -> pd.DataFrame:
        """Accept a Corpus or a bare frame."""
        if isinstance(source, pd.DataFrame):
            return source
        records = getattr(source, "records", None)
        if isinstance(records, pd.DataFrame):
            return records
        raise TypeError(f"Expected a Corpus or DataFrame, got {type(source).__name__}")

    def run(self, source: Union["Corpus", pd.DataFrame]) -> AnomalyResult:
        """Execute the Stage 4 pipeline.

        Args:
            source: A corpus already carrying Stage 2 and Stage 3 outputs.

        Returns:
            An :class:`AnomalyResult`; row count, order and index preserved.

        Raises:
            Stage4InputError: If the upstream contract is incomplete.
        """
        frame = self._frame_of(source)
        require_contract(frame)
        config = self.config
        started = time.perf_counter()

        signals = build_signals(frame)
        types = classify_types(
            frame,
            signals.frame,
            z_threshold=config.z_type_threshold,
            confidence_threshold=config.confidence_threshold,
        )
        severity = compute_severity(
            signals.frame,
            weights=config.severity_weights,
            scale=config.z_severity_scale,
        )
        decision = route(
            frame,
            signals.frame,
            types,
            confidence_threshold=config.confidence_threshold,
            investigate_threshold=config.z_investigate_threshold,
        )

        output = pd.DataFrame(index=frame.index)
        for column in ("z_cost", "cost_scope", "z_spend", "z_duration",
                       "valid_signal_count"):
            output[column] = signals.frame[column]
        output["severity_score"] = severity.score

        # Descriptive: reads the severity above, never rewrites it.
        definedness = severity_definedness(frame, signals.frame, severity.score)
        output["severity_defined"] = definedness.defined
        output["severity_defined_reason"] = definedness.reason

        output["decision_class"] = decision.decision_class
        output["decision_reason"] = decision.decision_reason
        output["confidence_flag"] = decision.confidence_flag
        output["duplicate_flag_stage4"] = signals.frame["duplicate_flag"]

        # Types as a list per record, plus one boolean column each so the frame
        # stays queryable without parsing strings.
        for name in ANOMALY_TYPES:
            output[f"type_{name}"] = types[name]
        output["anomaly_types"] = [
            [name for name in ANOMALY_TYPES if bool(row[name])]
            for _, row in types.iterrows()
        ]
        output["anomaly_count"] = types.sum(axis=1).astype("int64")

        # Explanations read the Stage 3 reason and context columns too, so the
        # text can say WHY a signal is absent instead of ignoring it.
        context_columns = [
            name
            for name in (
                "confidence",
                "cluster_label",
                "peer_cell_stable",
                "peer_cell_size",
                "cluster_has_norm",
                "duplicate_score",
                "duplicate_flag",
                "deviation_cell_cost_reason",
                "deviation_spend_ratio_reason",
                "deviation_duration_reason",
            )
            if name in frame.columns
        ]
        output["explanation_text"] = build_explanations(
            output.join(frame[context_columns])
        )

        # Measurement passes. Both run after every decision is final, and
        # neither writes to a column any decision was read from.
        duplicates: Optional[DuplicateDiagnostics] = None
        if config.compute_duplicate_diagnostics:
            duplicates = compute_duplicate_diagnostics(frame)
            output["duplicate_cosine"] = duplicates.best_cosine
            output["duplicate_reachable"] = duplicates.reachable

        missing = [name for name in STAGE4_COLUMNS if name not in output.columns]
        if missing:
            raise RuntimeError(f"Stage 4 contract incomplete: missing {missing!r}")

        self._assert_guarantees(output, frame)
        elapsed = time.perf_counter() - started

        calibration: Optional[Dict[str, Any]] = None
        if config.compute_calibration:
            calibration = compute_stage4_calibration_report(
                output.join(frame[[
                    name for name in frame.columns if name not in output.columns
                ]])
            )

        LOGGER.info(
            "Stage 4 complete in %.2fs: %s",
            elapsed,
            {k: int(v) for k, v in output["decision_class"].value_counts().items()},
        )

        return AnomalyResult(
            frame=output,
            signals=signals,
            types=types,
            severity=severity,
            decision=decision,
            config=config,
            definedness=definedness,
            calibration=calibration,
            duplicates=duplicates,
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _assert_guarantees(output: pd.DataFrame, frame: pd.DataFrame) -> None:
        """Enforce the Stage 4 final guarantees before returning.

        These are assertions, not validation: a violation is a defect in this
        module, and every one of them is a listed requirement.
        """
        # No NaN leakage into the decision surface.
        for column in ("decision_class", "confidence_flag", "cost_scope",
                       "anomaly_count", "valid_signal_count"):
            assert output[column].notna().all(), f"{column} contains NaN"

        # No unjustified escalation.
        escalated = output["decision_class"] == "INVESTIGATE"
        assert not bool((escalated & (output["confidence_flag"] == "low")).any()), (
            "a low-confidence record reached INVESTIGATE"
        )

        # No fabricated signals: a z is present only where Stage 3 defined it.
        for column, source_column in (
            ("z_spend", "deviation_spend_ratio"),
            ("z_duration", "deviation_duration"),
        ):
            fabricated = output[column].notna() & frame[source_column].isna()
            assert not bool(fabricated.any()), f"{column} fabricated a value"

        # Severity is never 0 by default; undefined stays undefined.
        no_signal = output["valid_signal_count"] == 0
        assert output.loc[no_signal, "severity_score"].isna().all(), (
            "severity collapsed an unmeasurable record to a number"
        )
        finite = output["severity_score"].dropna()
        assert bool(((finite >= 0.0) & (finite <= 1.0)).all()), "severity left [0,1]"

        # Severity definedness is exact, in both directions: an undefined
        # severity must be NaN, and a defined one must not be.
        undefined = ~output["severity_defined"].to_numpy(dtype=bool)
        assert output.loc[undefined, "severity_score"].isna().all(), (
            "a record marked severity_defined=False carries a severity"
        )
        assert output.loc[~undefined, "severity_score"].notna().all(), (
            "a record marked severity_defined=True carries no severity"
        )
        assert (
            output.loc[~undefined, "severity_defined_reason"] == "ok"
        ).all(), "a defined severity was given a reason for being undefined"
        assert (
            output.loc[undefined, "severity_defined_reason"] != "ok"
        ).all(), "an undefined severity was given no reason"
        declared = set(output["severity_defined_reason"].unique())
        assert declared <= set(SEVERITY_DEFINED_REASONS), (
            f"undeclared severity reason(s): {sorted(declared - set(SEVERITY_DEFINED_REASONS))}"
        )

        # Duplicate observability, when it was measured. A flagged duplicate
        # that is not reachable would mean the blended score exceeded its own
        # cosine - impossible, and worth failing loudly on.
        if "duplicate_reachable" in output.columns:
            flagged = output["duplicate_flag_stage4"].to_numpy(dtype=bool)
            reachable = output["duplicate_reachable"].to_numpy(dtype=bool)
            assert bool(reachable[flagged].all()), (
                "a flagged duplicate was not reachable"
            )


def attach_anomalies(
    corpus: "Corpus",
    result: Optional[AnomalyResult] = None,
    config: Optional[AnomalyConfig] = None,
) -> AnomalyResult:
    """Attach Stage 4 columns onto a corpus in place.

    Args:
        corpus: Corpus already carrying Stage 2 and Stage 3 outputs.
        result: A previously computed result; computed here when omitted.
        config: Configuration used when ``result`` is omitted.

    Returns:
        The :class:`AnomalyResult` that was attached.

    Raises:
        ValueError: If the result does not align with the corpus index.
    """
    frame = corpus.records
    computed = result if result is not None else AnomalyLayer(config).run(corpus)

    if len(computed.frame) != len(frame):
        raise ValueError(
            f"Stage 4 produced {len(computed.frame)} rows for {len(frame)} records"
        )
    if not computed.frame.index.equals(frame.index):
        raise ValueError("Stage 4 index does not match the corpus index")

    for column in STAGE4_COLUMNS:
        frame[column] = computed.frame[column]

    # Optional diagnostic columns, present only when the duplicate diagnostics
    # ran. They were computed and then stranded on the result before this;
    # a consumer that asks for the measurement should receive it.
    for column in OPTIONAL_STAGE4_COLUMNS:
        if column in computed.frame.columns:
            frame[column] = computed.frame[column]

    LOGGER.info(
        "Attached %d Stage 4 column(s) to %d record(s); row order unchanged.",
        len(STAGE4_COLUMNS),
        len(frame),
    )
    return computed
