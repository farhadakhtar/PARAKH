"""``ActionLayer`` - the Stage 6 orchestrator.

    validate -> M1 correction -> route -> explain

Stage 6 is the end of the line: it turns two upstream verdicts into a queue, a
priority and a sentence. It computes nothing. Every number it prints was
produced by an earlier stage, and every routing decision is a lookup in a table
that a non-developer can read.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.constants import (
    ACTION_CLASSES,
    ACTION_TO_PRIORITY,
    ACTION_TO_QUEUE,
    ESCALATING_ACTIONS,
    PRIORITY_LEVELS,
    SPEC_ACTION_ALIAS,
    SPEC_ACTION_CLASSES,
    SPEC_COLUMN_ALIAS,
    STAGE6_ACTION_REPORT,
    STAGE6_VERSION,
)
from src.core.logger import get_logger
from src.stage6.explanation import (
    build_action_explanations,
    build_action_payloads,
    parse_action_payload,
)
from src.stage6.routing import (
    POLICY,
    M1Correction,
    RoutingResult,
    Stage6InputError,
    Stage6InvariantError,
    apply_m1_correction,
    require_contract,
    route,
)
from src.utils.helpers import write_json

if TYPE_CHECKING:  # pragma: no cover
    from src.stage1.corpus import Corpus

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

#: The Stage 6 output contract, in a fixed order.
STAGE6_COLUMNS: Tuple[str, ...] = (
    "action_class",
    "priority_level",
    "reviewer_queue",
    "explanation",
)

#: Aliases required by the Stage 6 audit specification. Pure renames of the
#: columns above plus the spec action vocabulary and the machine-readable
#: payload. Additive: no existing column is renamed or removed, and every alias
#: is asserted equal to its source on each run.
STAGE6_SPEC_COLUMNS: Tuple[str, ...] = (
    "action",
    "priority",
    "action_reason",
    "action_spec",
    "explanation_payload",
)

#: Audit columns: which rule fired, the corrected finding list, and whether the
#: M1 correction was needed. Not part of the minimum contract, but a routed
#: record that cannot say why it was routed is not auditable.
STAGE6_DETAIL_COLUMNS: Tuple[str, ...] = (
    "action_rule",
    "action_anomaly_types",
    "anomaly_types_corrected",
)


@dataclass(frozen=True)
class ActionResult:
    """Everything Stage 6 produced."""

    frame: pd.DataFrame
    routing: RoutingResult
    elapsed_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.frame)

    def queue(self, name: str) -> pd.DataFrame:
        """The work waiting for one team.

        Args:
            name: A reviewer queue name.

        Returns:
            A view of the Stage 6 frame, most urgent first, ordered by
            priority then by the rule that routed the record. Deterministic:
            the sort is stable and index order breaks every remaining tie.

        Raises:
            ValueError: On an unknown queue.
        """
        known = set(ACTION_TO_QUEUE.values())
        if name not in known:
            raise ValueError(f"unknown queue {name!r}; expected one of {sorted(known)}")
        subset = self.frame.loc[self.frame["reviewer_queue"] == name]
        order = {level: position for position, level in enumerate(PRIORITY_LEVELS)}
        return subset.assign(
            _rank=subset["priority_level"].map(order)
        ).sort_values("_rank", kind="stable").drop(columns="_rank")

    def by_priority(self, level: str) -> pd.DataFrame:
        """Every record at one priority.

        Raises:
            ValueError: On an unknown priority level.
        """
        if level not in PRIORITY_LEVELS:
            raise ValueError(
                f"unknown priority {level!r}; expected one of {PRIORITY_LEVELS}"
            )
        return self.frame.loc[self.frame["priority_level"] == level]

    def explain(self, row: Any) -> str:
        """The stored explanation for one record; recomputes nothing."""
        if row not in self.frame.index:
            raise KeyError(f"row {row!r} is not in the frame index")
        return str(self.frame.loc[row, "explanation"])

    def report(self) -> Dict[str, Any]:
        """Corpus-level Stage 6 summary. Carries no wall-clock value."""
        return {
            "stage6_version": STAGE6_VERSION,
            "n_records": len(self.frame),
            "_note": (
                "Policy, not inference. Stage 6 maps the Stage 4 decision and "
                "the Stage 5 risk band onto an action, a priority and a queue. "
                "No record is labelled fraud; escalation means a human should "
                "look, not that anything has been concluded."
            ),
            "routing": self.routing.to_dict(),
            "policy": [
                {"rule": rule.name, "action": rule.action, "note": rule.note}
                for rule in POLICY
            ],
        }

    def save_reports(self, output_dir: PathLike) -> Dict[str, Path]:
        """Write the Stage 6 report as JSON."""
        directory = Path(output_dir)
        written = {
            "action_report": write_json(
                self.report(), directory / STAGE6_ACTION_REPORT
            )
        }
        LOGGER.info("Wrote %d Stage 6 report(s) to %s", len(written), directory)
        return written


class ActionLayer:
    """Stage 6: map upstream verdicts onto human-actionable work."""

    def __repr__(self) -> str:
        return f"<ActionLayer {STAGE6_VERSION}>"

    def _frame_of(self, source: Union["Corpus", pd.DataFrame]) -> pd.DataFrame:
        """Accept a Corpus or a bare frame."""
        if isinstance(source, pd.DataFrame):
            return source
        records = getattr(source, "records", None)
        if isinstance(records, pd.DataFrame):
            return records
        raise TypeError(f"Expected a Corpus or DataFrame, got {type(source).__name__}")

    def run(self, source: Union["Corpus", pd.DataFrame]) -> ActionResult:
        """Execute the Stage 6 pipeline.

        Args:
            source: A corpus already carrying Stage 4 and Stage 5 output.

        Returns:
            An :class:`ActionResult`; row count, order and index preserved.

        Raises:
            Stage6InputError: If the upstream contract is incomplete.
        """
        frame = self._frame_of(source)
        require_contract(frame)
        started = time.perf_counter()

        routing = route(frame)
        correction = routing.correction

        output = pd.DataFrame(index=frame.index)
        output["action_class"] = routing.action_class
        output["priority_level"] = routing.priority_level
        output["reviewer_queue"] = routing.reviewer_queue
        output["action_rule"] = routing.action_rule
        output["action_anomaly_types"] = correction.types
        output["anomaly_types_corrected"] = correction.corrected

        # Specification aliases. Renames only - the values are the same
        # objects the routing produced, never recomputed.
        for alias, source in SPEC_COLUMN_ALIAS.items():
            output[alias] = output[source]
        output["action_spec"] = output["action_class"].map(SPEC_ACTION_ALIAS)

        context = [
            name
            for name in (
                "severity_score",
                "severity_defined",
                "risk_score",
                "risk_defined",
                "decision_class",
                "risk_flag",
                "decision_reason",
            )
            if name in frame.columns
        ]
        joined = output.join(frame[context])
        output["explanation"] = build_action_explanations(joined)
        output["explanation_payload"] = build_action_payloads(joined)

        missing = [
            name
            for name in (*STAGE6_COLUMNS, *STAGE6_SPEC_COLUMNS)
            if name not in output.columns
        ]
        if missing:
            raise RuntimeError(f"Stage 6 contract incomplete: missing {missing!r}")

        self._assert_guarantees(output, frame)
        elapsed = time.perf_counter() - started

        LOGGER.info(
            "Stage 6 complete in %.2fs: %s",
            elapsed,
            {k: int(v) for k, v in output["reviewer_queue"].value_counts().items()},
        )
        return ActionResult(frame=output, routing=routing, elapsed_seconds=elapsed)

    @staticmethod
    def _assert_guarantees(output: pd.DataFrame, frame: pd.DataFrame) -> None:
        """Enforce the Stage 6 output guarantees before returning.

        Raises :class:`Stage6InvariantError`, not ``AssertionError``, so the
        checks survive ``python -O`` and a caller can catch a Stage 6
        guarantee failure specifically.
        """
        def _require(condition: bool, message: str) -> None:
            if not condition:
                raise Stage6InvariantError(message)

        # 1 & 2 - nothing is unrouted or unqueued.
        for column in STAGE6_COLUMNS:
            _require(bool(output[column].notna().all()),
                     f"{column} contains NaN")
            _require(bool((output[column].astype(str).str.len() > 0).all()),
                     f"{column} contains an empty value")

        # Action, priority and queue agree with the policy tables.
        _require(output["priority_level"].equals(
                     output["action_class"].map(ACTION_TO_PRIORITY)),
                 "priority_level disagrees with the policy table")
        _require(output["reviewer_queue"].equals(
                     output["action_class"].map(ACTION_TO_QUEUE)),
                 "reviewer_queue disagrees with the policy table")

        # 5 - the explanation prints no value the record does not carry.
        undefined_risk = ~frame["risk_defined"].fillna(False).to_numpy(dtype=bool)
        _require(bool(output.loc[undefined_risk, "explanation"].str.contains(
                     "Risk: not defined").all()),
                 "an explanation printed a risk score for an unscored record")
        no_severity = ~frame["severity_defined"].fillna(False).to_numpy(dtype=bool)
        _require(bool(output.loc[no_severity, "explanation"].str.contains(
                     "Severity: not defined").all()),
                 "an explanation printed a severity for a record without one")

        # C1 - every alias is exactly its source, and the spec vocabulary is
        # closed. An alias that drifted from its source would be worse than no
        # alias at all: two columns disagreeing about one decision.
        for alias, source in SPEC_COLUMN_ALIAS.items():
            _require(output[alias].equals(output[source]),
                     f"alias {alias!r} has drifted from {source!r}")
        _require(bool(output["action_spec"].notna().all()),
                 "an action has no spec alias")
        _require(set(output["action_spec"]) <= set(SPEC_ACTION_CLASSES),
                 "action_spec emitted a name outside the specification vocabulary")

        # 6 - an escalated record always names a finding.
        escalating = output["action_class"].isin(ESCALATING_ACTIONS).to_numpy()
        _require(not bool((escalating & output["explanation"].str.contains(
                     "none recorded").to_numpy()).any()),
                 "an escalated record's explanation names no finding")

        # I5 - the payload agrees with the columns it claims to describe.
        # Checked on every record: a payload that drifted would be a machine
        # contract quietly disagreeing with the decision it encodes.
        for label, payload in output["explanation_payload"].items():
            fields = parse_action_payload(payload)
            _require(fields["action"] == output.at[label, "action_class"],
                     f"I5: payload action disagrees at {label!r}")
            _require(fields["priority"] == output.at[label, "priority_level"],
                     f"I5: payload priority disagrees at {label!r}")
            _require(fields["rule"] == output.at[label, "action_rule"],
                     f"I5: payload rule disagrees at {label!r}")
            _require(fields["findings"] == list(
                         output.at[label, "action_anomaly_types"]),
                     f"I5: payload findings disagree at {label!r}")
            _require(fields["findings"] == fields["anomaly_types"],
                     f"I5: findings and its synonym disagree at {label!r}")


def attach_actions(
    corpus: "Corpus",
    result: Optional[ActionResult] = None,
) -> ActionResult:
    """Attach Stage 6 columns onto a corpus in place.

    Args:
        corpus: Corpus already carrying Stage 4 and Stage 5 output.
        result: A previously computed result; computed here when omitted.

    Returns:
        The :class:`ActionResult` that was attached.

    Raises:
        ValueError: If the result does not align with the corpus index.
    """
    frame = corpus.records
    computed = result if result is not None else ActionLayer().run(corpus)

    if len(computed.frame) != len(frame):
        raise ValueError(
            f"Stage 6 produced {len(computed.frame)} rows for {len(frame)} records"
        )
    if not computed.frame.index.equals(frame.index):
        raise ValueError("Stage 6 index does not match the corpus index")

    for column in (*STAGE6_COLUMNS, *STAGE6_SPEC_COLUMNS, *STAGE6_DETAIL_COLUMNS):
        frame[column] = computed.frame[column]

    LOGGER.info(
        "Attached %d Stage 6 column(s) to %d record(s); row order unchanged.",
        len(STAGE6_COLUMNS) + len(STAGE6_SPEC_COLUMNS) + len(STAGE6_DETAIL_COLUMNS),
        len(frame),
    )
    return computed
