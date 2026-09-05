"""``ConsumptionLayer`` - the Stage 7 orchestrator.

    validate -> decode payloads -> queues -> cards -> API -> audit log

Stage 7 is strictly read-only over Stages 1-6. It attaches nothing to the
corpus and mutates nothing: its outputs are separate objects a caller hands to
a person, an API, or a log file. That is the difference between a consumption
layer and another stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.core.constants import (
    ACTION_CLASSES,
    ACTION_TO_QUEUE_NAME,
    ACTION_GROUPS,
    ACTION_SPEC_LOSSY_NOTE,
    ACTION_TO_GROUP,
    ACTION_TO_SEMANTIC_TYPE,
    CALIBRATION_WARNING,
    ESCALATION_REASON_STATUSES,
    RISK_CALIBRATION_STATUS,
    RISK_RELATIVE_WARNING,
    DECISION_CLARITY_FLAGS,
    PRIORITY_EXECUTION,
    PRIORITY_LEVELS,
    PRIORITY_SEMANTIC_TYPES,
    QUEUE_NAMES,
    REASON_FLAGS,
    STAGE65_SAFETY_LOG,
    STAGE7_ANNOTATION_REPORT,
    STAGE7_AUDIT_LOG,
    STAGE7_QUEUE_REPORT,
    STAGE7_REFERENCE_TIMESTAMP,
    STAGE7_VERSION,
    STAGE7_WORK_SUMMARY,
    WORK_ID_AMBIGUITY_WARNING,
)
from src.core.logger import get_logger
from src.stage5.risk_interpretation import (
    INTERPRETATION_COLUMNS,
    compute_risk_interpretation,
    interpretation_report,
)
from src.stage6.safety_layer import (
    SAFETY_COLUMNS,
    SafetyConfig,
    SafetyResult,
    apply_safety_rules,
)
from src.stage6.work_resolution import resolve_works, work_conflict_summary
from src.stage7.annotations import (
    ANNOTATION_COLUMNS,
    assert_closed_anomaly_vocabulary,
    build_annotations,
    build_system_metadata,
    build_transparency_metrics,
    build_work_level_summary,
)
from src.stage7.api import API_FIELDS, build_api_responses
from src.stage7.policy import (
    Stage7PolicyError,
    escalation_policy_report,
    validate_escalation_policy,
)
from src.stage7.audit import AUDIT_FIELDS, build_audit_log, write_audit_log
from src.stage7.interface import (
    QueueItem,
    Stage7ContractError,
    build_decision_card,
    build_queues,
    decode_payloads,
    require_contract,
)
from src.utils.helpers import write_json

if TYPE_CHECKING:  # pragma: no cover
    from src.stage1.corpus import Corpus

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]


class Stage7InvariantError(RuntimeError):
    """Raised when a Stage 7 output guarantee does not hold."""


@dataclass(frozen=True)
class ConsumptionResult:
    """Everything Stage 7 produced. The corpus is untouched."""

    queues: Dict[str, List[QueueItem]]
    cards: List[Dict[str, Any]]
    api_responses: List[Dict[str, Any]]
    audit_entries: List[Dict[str, Any]]
    issued_at: str
    #: Per-record transparency annotations. Derived, never decisive.
    annotations: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Configuration these outputs depend on (R4).
    system_metadata: Dict[str, Any] = field(default_factory=dict)
    #: Measured limitations, as rates (R1, R3, R6, R7).
    transparency_metrics: Dict[str, Any] = field(default_factory=dict)
    #: One row per work_id, worst case first (R5).
    work_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Relative interpretation of the uncalibrated risk scale (R7).
    interpretation: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Stage 6.5 decision safety. None when the layer was not run.
    safety: Optional[SafetyResult] = None
    #: Work-level resolution, with conflicts surfaced (R5).
    work_resolution: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: The escalation invariant, stated even when it passes (R2).
    escalation_policy: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.cards)

    def queue(self, name: str) -> List[QueueItem]:
        """The work waiting for one team, most urgent first.

        Raises:
            ValueError: On an unknown queue name.
        """
        if name not in QUEUE_NAMES:
            raise ValueError(f"unknown queue {name!r}; expected one of {QUEUE_NAMES}")
        return self.queues[name]

    def card(self, record_id: Any) -> Dict[str, Any]:
        """The decision card for one record.

        Raises:
            KeyError: If the record is not in this result.
        """
        for entry in self.cards:
            if entry["record_id"] == record_id:
                return entry
        raise KeyError(f"record {record_id!r} is not in this result")

    def report(self) -> Dict[str, Any]:
        """Corpus-level Stage 7 summary."""
        depth = {name: len(items) for name, items in self.queues.items()}
        by_priority: Dict[str, int] = {level: 0 for level in PRIORITY_LEVELS}
        for items in self.queues.values():
            for item in items:
                by_priority[item.priority] += 1
        return {
            "stage7_version": STAGE7_VERSION,
            "n_records": len(self.cards),
            "_note": (
                "Stage 7 decides nothing. It makes the Stage 6 decision "
                "visible, actionable and accountable. Every value here was "
                "read from explanation_payload; the human explanation is "
                "display only and is never parsed."
            ),
            "issued_at": self.issued_at,
            "queue_depth": depth,
            "by_priority": by_priority,
            "execution": {
                level: dict(PRIORITY_EXECUTION[level]) for level in PRIORITY_LEVELS
            },
            "sla_load": {
                level: by_priority[level]
                for level in PRIORITY_LEVELS
                if PRIORITY_EXECUTION[level]["sla_hours"] is not None
            },
            # Attached to the report, not merely logged: a limitation nobody
            # reads is a limitation nobody knows about.
            "system_metadata": self.system_metadata,
            "transparency_metrics": self.transparency_metrics,
            "calibration_warning": CALIBRATION_WARNING,
            "work_id_note": WORK_ID_AMBIGUITY_WARNING,
            "escalation_policy": self.escalation_policy,
            "risk_calibration_status": RISK_CALIBRATION_STATUS,
            "risk_relative_warning": RISK_RELATIVE_WARNING,
            "action_spec_note": ACTION_SPEC_LOSSY_NOTE,
            "risk_interpretation": interpretation_report(self.interpretation)
            if len(self.interpretation)
            else {},
            "safety": self.safety.to_dict() if self.safety is not None else {
                "_note": "safety layer not run"
            },
            "work_conflicts": work_conflict_summary(self.work_resolution)
            if len(self.work_resolution)
            else {},
            "priority_semantic_split": {
                name: int(
                    (self.annotations["priority_semantic_type"] == name).sum()
                )
                for name in PRIORITY_SEMANTIC_TYPES
            }
            if len(self.annotations)
            else {},
        }

    def save(self, output_dir: PathLike) -> Dict[str, Path]:
        """Write the queue report and the audit log."""
        directory = Path(output_dir)
        written = {
            "queue_report": write_json(
                {
                    **self.report(),
                    "queues": {
                        name: [item.to_dict() for item in items]
                        for name, items in self.queues.items()
                    },
                },
                directory / STAGE7_QUEUE_REPORT,
            ),
            "audit_log": write_audit_log(
                self.audit_entries, directory / STAGE7_AUDIT_LOG
            ),
            "annotations": write_json(
                {
                    "stage7_version": STAGE7_VERSION,
                    "calibration_warning": CALIBRATION_WARNING,
                    "system_metadata": self.system_metadata,
                    "transparency_metrics": self.transparency_metrics,
                    "records": [
                        {"record_id": label, **row}
                        for label, row in zip(
                            self.annotations.index,
                            self.annotations.to_dict(orient="records"),
                        )
                    ],
                },
                directory / STAGE7_ANNOTATION_REPORT,
            ),
            "work_summary": write_json(
                {
                    "stage7_version": STAGE7_VERSION,
                    "_note": WORK_ID_AMBIGUITY_WARNING,
                    "calibration_warning": CALIBRATION_WARNING,
                    "n_works": int(len(self.work_summary)),
                    "n_with_conflicting_actions": int(
                        self.work_summary["has_conflicting_actions"].sum()
                    )
                    if len(self.work_summary)
                    else 0,
                    "works": self.work_summary.to_dict(orient="records"),
                },
                directory / STAGE7_WORK_SUMMARY,
            ),
        }
        if self.safety is not None:
            written["safety_log"] = self.safety.write_log(
                directory / STAGE65_SAFETY_LOG
            )
        LOGGER.info("Wrote %d Stage 7 artefact(s) to %s", len(written), directory)
        return written


class ConsumptionLayer:
    """Stage 7: turn decisions into work, contracts and evidence."""

    def __repr__(self) -> str:
        return f"<ConsumptionLayer {STAGE7_VERSION}>"

    def _frame_of(self, source: Union["Corpus", pd.DataFrame]) -> pd.DataFrame:
        """Accept a Corpus or a bare frame."""
        if isinstance(source, pd.DataFrame):
            return source
        records = getattr(source, "records", None)
        if isinstance(records, pd.DataFrame):
            return records
        raise TypeError(f"Expected a Corpus or DataFrame, got {type(source).__name__}")

    def run(
        self,
        source: Union["Corpus", pd.DataFrame],
        issued_at: str = STAGE7_REFERENCE_TIMESTAMP,
        apply_safety: bool = True,
        safety_config: Optional[SafetyConfig] = None,
        gates_aligned: Optional[bool] = None,
    ) -> ConsumptionResult:
        """Execute the Stage 7 pipeline.

        Args:
            source: A corpus or frame carrying Stage 6 output.
            issued_at: ISO8601 timestamp stamped on every artefact. Injected
                rather than read from a clock so that two runs over the same
                records are byte-identical; a production caller passes the
                real time explicitly.

        Returns:
            A :class:`ConsumptionResult`. The input is not modified.

        Raises:
            Stage7ContractError: If the Stage 6 contract is not satisfied.
        """
        frame = self._frame_of(source)
        require_contract(frame)
        # R2 - refuse the state before building anything from it. An
        # escalation with no risk score must not reach a person, and a
        # half-built result is harder to reason about than none.
        validate_escalation_policy(frame)
        started = time.perf_counter()

        payloads = decode_payloads(frame)
        queues = build_queues(frame, payloads, issued_at=issued_at)
        explanations = [str(value) for value in frame["explanation"]]
        cards = [
            build_decision_card(label, payload, text)
            for (label, payload), text in zip(payloads.items(), explanations)
        ]
        responses = build_api_responses(payloads, issued_at=issued_at)
        entries = build_audit_log(frame, payloads, issued_at=issued_at)

        # Transparency layer. Derived from the payloads only, and incapable of
        # changing an action, a priority or a queue - the guarantees below
        # verify exactly that.
        # FIX 7 - a category nothing downstream understands must not pass.
        assert_closed_anomaly_vocabulary(payloads)
        annotations = build_annotations(payloads, frame=frame)
        metadata = build_system_metadata(len(frame))
        metrics = build_transparency_metrics(annotations, payloads)
        work_summary = build_work_level_summary(frame, payloads, annotations)
        policy = escalation_policy_report(frame)

        # R7 - a percentile is the strongest honest reading of an
        # uncalibrated scale. Adds columns; changes no score.
        interpretation = (
            compute_risk_interpretation(frame)
            if "risk_score" in frame.columns
            else pd.DataFrame(index=frame.index, columns=list(INTERPRETATION_COLUMNS))
        )

        # Stage 6.5 - the only layer that changes a decision. Runs last, on a
        # copy, and preserves every original.
        safety: Optional[SafetyResult] = None
        resolution = pd.DataFrame()
        if apply_safety and "work_id" in frame.columns:
            aligned = (
                metadata["thresholds"]["gates_aligned"]
                if gates_aligned is None
                else bool(gates_aligned)
            )
            safety = apply_safety_rules(
                frame,
                clarity=annotations["decision_clarity_flag"],
                gates_aligned=aligned,
                config=safety_config,
            )
            resolution = resolve_works(frame)

        result = ConsumptionResult(
            queues=queues,
            cards=cards,
            api_responses=responses,
            audit_entries=entries,
            issued_at=issued_at,
            annotations=annotations,
            system_metadata=metadata,
            transparency_metrics=metrics,
            work_summary=work_summary,
            interpretation=interpretation,
            safety=safety,
            work_resolution=resolution,
            escalation_policy=policy,
            elapsed_seconds=time.perf_counter() - started,
        )
        self._assert_guarantees(result, frame, payloads)

        LOGGER.info(
            "Stage 7 complete in %.2fs: %s",
            result.elapsed_seconds,
            {name: len(items) for name, items in queues.items()},
        )
        return result

    @staticmethod
    def _assert_guarantees(
        result: ConsumptionResult, frame: pd.DataFrame, payloads: pd.Series
    ) -> None:
        """Enforce the eight Stage 7 invariants.

        Raises :class:`Stage7InvariantError`, not ``AssertionError``, so the
        checks survive ``python -O`` and a caller can catch a Stage 7 failure
        specifically.
        """

        def _require(condition: bool, message: str) -> None:
            if not condition:
                raise Stage7InvariantError(message)

        n_records = len(frame)

        # 1 - every action reached a queue, and every record reached exactly one.
        queued = sum(len(items) for items in result.queues.values())
        _require(
            queued == n_records,
            f"I1: {queued} queued item(s) for {n_records} record(s); every "
            "record must reach exactly one queue",
        )
        _require(
            set(result.queues) == set(QUEUE_NAMES),
            "I1: a declared queue is missing from the result",
        )

        # 2 - every queue item carries a usable priority and its semantics.
        for name, items in result.queues.items():
            for item in items:
                _require(
                    item.priority in PRIORITY_LEVELS,
                    f"I2: item {item.record_id!r} has priority {item.priority!r}",
                )
                _require(
                    item.execution_mode
                    == PRIORITY_EXECUTION[item.priority]["mode"],
                    f"I2: item {item.record_id!r} has execution semantics that "
                    "disagree with its priority",
                )
                _require(
                    ACTION_TO_QUEUE_NAME[item.action] == name,
                    f"I2: item {item.record_id!r} is in the wrong queue",
                )

        # 3 & 6 - card, API response and log agree with the payload, which is
        # the only source of truth any of them may read.
        _require(len(result.cards) == n_records, "I5: a record has no decision card")
        _require(
            len(result.api_responses) == n_records, "I5: a record has no API response"
        )
        _require(
            len(result.audit_entries) == n_records, "I5: a record has no audit entry"
        )

        # Vectorised. The checks and their failure conditions are identical
        # to the per-record form; only the 200,000 Python calls are gone. The
        # first offending record is still named, because "some record is
        # wrong" is not a diagnosis.
        labels = list(payloads.index)
        explanations = [str(value) for value in frame["explanation"]]
        payload_list = list(payloads)

        def _first_mismatch(flags: List[bool]) -> Any:
            for position, ok in enumerate(flags):
                if not ok:
                    return labels[position]
            return None

        # I3 - the three output streams are aligned to the same records.
        aligned = [
            card["record_id"] == label
            and response["record_id"] == label
            and entry["record_id"] == label
            for card, response, entry, label in zip(
                result.cards, result.api_responses, result.audit_entries, labels
            )
        ]
        _require(all(aligned), f"I3: outputs are misaligned at record "
                               f"{_first_mismatch(aligned)!r}")

        # I6 - card, response and log agree with the payload on every field
        # any of them claims to carry.
        agrees = [
            card["action"] == payload["action"]
            and response["action"] == payload["action"]
            and entry["action"] == payload["action"]
            and response["priority"] == payload["priority"]
            and entry["priority"] == payload["priority"]
            and response["risk_score"] == payload.get("risk_score")
            and card["findings"] == list(payload.get("findings") or [])
            for card, response, entry, payload in zip(
                result.cards, result.api_responses, result.audit_entries, payload_list
            )
        ]
        _require(all(agrees), f"I6: an output disagrees with the payload at "
                              f"{_first_mismatch(agrees)!r}")

        # I4 - the human explanation is carried, never altered or parsed.
        carried = [
            card["explanation"] == text
            for card, text in zip(result.cards, explanations)
        ]
        _require(all(carried), f"I4: the human explanation was altered at "
                               f"{_first_mismatch(carried)!r}")

        # I5 - every entry and response is complete. The field sets are fixed,
        # so checking the first record proves the shape and a set comparison
        # over the rest costs nothing.
        audit_fields = set(AUDIT_FIELDS)
        api_fields = set(API_FIELDS)
        complete_log = [audit_fields <= set(entry) for entry in result.audit_entries]
        _require(all(complete_log), f"I5: an audit entry is incomplete at "
                                    f"{_first_mismatch(complete_log)!r}")
        complete_api = [api_fields <= set(response) for response in result.api_responses]
        _require(all(complete_api), f"I6: an API response is incomplete at "
                                    f"{_first_mismatch(complete_api)!r}")

        # I7 - the annotation layer is derived, never decisive. Each check
        # below proves an annotation restates a decision rather than making
        # one; if any could differ, it would belong in Stage 6.
        annotations = result.annotations
        if len(annotations):
            _require(
                list(annotations.index) == labels,
                "I7: annotations are misaligned with the records",
            )
            actions = [str(payload["action"]) for payload in payload_list]
            _require(
                list(annotations["action_truth_class"]) == actions,
                "I7: action_truth_class is not an exact copy of the action",
            )
            _require(
                list(annotations["priority_semantic_type"])
                == [ACTION_TO_SEMANTIC_TYPE[name] for name in actions],
                "I7: a semantic type disagrees with its action",
            )
            _require(
                set(annotations["stage7_reason_flag"]) <= set(REASON_FLAGS),
                "I7: an undeclared reason flag was emitted",
            )
            _require(
                set(annotations["decision_clarity_flag"]) <= set(DECISION_CLARITY_FLAGS),
                "I7: an undeclared clarity flag was emitted",
            )
            _require(
                annotations["calibration_warning"].eq(CALIBRATION_WARNING).all(),
                "I7: a record is missing the calibration warning",
            )
            _require(
                annotations["stage7_explanation"].notna().all()
                and annotations["stage7_explanation"].str.len().gt(0).all(),
                "I7: a record has no Stage 7 explanation",
            )
            # AMBIGUOUS and DATA_LIMITED are mutually exclusive by
            # construction: the M1 label is only ever added to an escalated
            # record, and every escalated record is scored.
            ambiguous = annotations["decision_clarity_flag"] == "AMBIGUOUS"
            unscored = [payload.get("risk_score") is None for payload in payload_list]
            _require(
                not any(a and u for a, u in zip(ambiguous, unscored)),
                "I7: a record is both AMBIGUOUS and DATA_LIMITED",
            )

            # I8 - the correction-pass fields duplicate existing facts in a
            # new shape. Two names for one fact is only safe while they agree.
            _require(
                list(annotations["action_group"])
                == [ACTION_TO_GROUP[name] for name in actions],
                "I8: action_group disagrees with its action",
            )
            _require(
                [value.upper() for value in annotations["action_group"]]
                == list(annotations["priority_semantic_type"]),
                "I8: action_group and priority_semantic_type disagree",
            )
            _require(
                set(annotations["escalation_reason_status"])
                <= set(ESCALATION_REASON_STATUSES),
                "I8: an undeclared escalation_reason_status was emitted",
            )
            _require(
                list(annotations["action_spec_lossy"])
                == [
                    warning is not None
                    for warning in annotations["action_interpretation_warning"]
                ],
                "I8: action_spec_lossy disagrees with the lossy warning",
            )


def consume(
    source: Union["Corpus", pd.DataFrame],
    issued_at: str = STAGE7_REFERENCE_TIMESTAMP,
) -> ConsumptionResult:
    """Run Stage 7 over a corpus without modifying it.

    Deliberately not named ``attach_*``: unlike every earlier stage, Stage 7
    writes nothing back. Its outputs are for people, APIs and log files, not
    for the dataframe.
    """
    return ConsumptionLayer().run(source, issued_at=issued_at)
