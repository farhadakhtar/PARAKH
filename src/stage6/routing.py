"""Stage 6 routing policy: a table, applied in order.

This module is deliberately unintelligent. It computes nothing, infers nothing
and reads no upstream column except the four the policy dispatches on. If a
change here requires thinking about what a record *means*, the change belongs
in Stage 4 or Stage 5.

The rules are an ordered list rather than a chain of ``np.where`` calls so the
precedence is visible, auditable, and reportable: every record records the name
of the rule that routed it, and the corpus report counts how often each fired.
A policy nobody can see the shape of is not a policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ACTION_CLASSES,
    ACTION_TO_PRIORITY,
    ACTION_TO_QUEUE,
    CONFIDENCE_GATE_THRESHOLD,
    DECISION_CLASSES,
    ESCALATING_ACTIONS,
    M1_CORRECTION_LABEL,
    MIN_CONFIDENCE_FOR_RISK,
    PRIORITY_LEVELS,
    RISK_FLAGS,
    STAGE6_VERSION,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Columns Stage 6 dispatches on. Nothing else is read.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "decision_class",
    "severity_score",
    "severity_defined",
    "anomaly_types",
    "risk_score",
    "risk_flag",
    "risk_defined",
    "risk_defined_reason",
)

#: Named in the Stage 6 brief but not emitted by Stage 4, which calls it
#: ``decision_reason``. Read when present, never required.
OPTIONAL_COLUMNS: Tuple[str, ...] = ("decision_reason", "anomaly_reason")


class Stage6InputError(RuntimeError):
    """Raised when the Stage 4 / Stage 5 contract is incomplete."""


class Stage6ConfigError(Stage6InputError):
    """Raised when Stage 6's own configuration cannot support its invariants.

    Distinct from a data problem: no record is at fault and no rerun will help
    until a threshold is changed.
    """


class Stage6InvariantError(RuntimeError):
    """Raised when a post-routing guarantee does not hold.

    Not an ``AssertionError``: these checks must survive ``python -O``, and a
    caller must be able to catch a Stage 6 guarantee failure specifically
    rather than every assertion in the process.
    """


class Stage6ContractError(Stage6InputError):
    """Raised when the input is complete but internally contradictory.

    Distinct from :class:`Stage6InputError` because the remedy is different: a
    missing column means a stage was not run, whereas a contradictory one means
    a stage produced something impossible and the routing must not proceed.
    """


def assert_gate_alignment(
    stage4_gate: float = CONFIDENCE_GATE_THRESHOLD,
    stage5_gate: float = MIN_CONFIDENCE_FOR_RISK,
) -> None:
    """Refuse to route when the two confidence gates have drifted apart.

    Stage 6's invariant *insufficient_data is never escalated* is not a
    property of its own policy. It holds only because Stage 4 refuses to
    escalate below its confidence gate and Stage 5 refuses to score below
    the same number, so ``INVESTIGATE`` implies ``risk_defined``. The audit
    demonstrated the consequence: raising the Stage 5 gate to 0.80 produced
    **73 records** that were both INVESTIGATE and insufficient_data, every one
    of them escalated.

    Both constants derive from ``PEER_STAT_MIN_CONFIDENCE``, so this check
    passes by construction today. It exists so that a future edit to either
    one fails here, loudly, instead of silently breaking an invariant three
    stages away.

    Args:
        stage4_gate: Stage 4's escalation gate.
        stage5_gate: Stage 5's scoring gate.

    Raises:
        Stage6ConfigError: If the two gates differ.
    """
    if float(stage4_gate) != float(stage5_gate):
        raise Stage6ConfigError(
            f"confidence gates have drifted: Stage 4 escalates at "
            f"{stage4_gate} but Stage 5 scores at {stage5_gate}. Stage 6 "
            "cannot guarantee that an escalated record carries a risk score "
            "while they differ, so it refuses to route. Realign the two "
            "thresholds, or accept that insufficient_data records will be "
            "escalated and revise the invariant deliberately."
        )


def validate_stage5_contract(frame: pd.DataFrame) -> None:
    """Verify the cross-field assumptions Stage 6 dispatches on.

    Eight of the policy predicates read ``risk_flag`` and five read
    ``risk_defined``. Nothing previously checked that the two agree, so a
    disagreement surfaced as an internal ``AssertionError`` from deep inside
    the invariant block rather than as a contract violation at the door.

    One property is checked: ``risk_flag == "insufficient_data"`` if and only
    if ``not risk_defined``. Stage 5 guarantees it by construction - the band
    is assigned from the same gate that sets ``risk_defined`` - but a
    guarantee that is never verified is an assumption.

    **Deliberately NOT checked here:** ``decision_class == "INVESTIGATE"``
    implies ``risk_defined``. That property is what protects the "never
    escalate insufficient data" invariant, and a *configured* gate drift
    (``RiskConfig(min_confidence=0.80)``) breaks it while leaving the
    equivalence above perfectly intact. Rejecting it would make the
    ``investigate_unscored`` backstop rule unreachable and would refuse input
    that the existing policy tests construct on purpose. Closing that vector
    means choosing which of two invariants to break - never downgrade an
    escalation, or never escalate insufficient data - and that is a policy
    decision, not a validation one. :func:`assert_gate_alignment` catches the
    constant-level case; the configured case remains open and is documented.

    Args:
        frame: Corpus frame with Stage 4 and Stage 5 output.

    Raises:
        Stage6ContractError: If the two fields disagree.
    """
    flag_says_missing = (
        frame["risk_flag"].astype("object") == "insufficient_data"
    ).to_numpy(dtype=bool)
    not_defined = ~frame["risk_defined"].fillna(False).to_numpy(dtype=bool)

    mismatch = flag_says_missing != not_defined
    disagree = int(mismatch.sum())
    if disagree:
        sample = frame.index[mismatch][:5].tolist()
        pairs = [
            (
                str(frame.loc[label, "risk_flag"]),
                bool(frame.loc[label, "risk_defined"]),
            )
            for label in sample
        ]
        raise Stage6ContractError(
            f"Stage 5 emitted {disagree} record(s) where risk_flag and "
            "risk_defined disagree. They must be equivalent: a record is "
            "banded 'insufficient_data' exactly when it has no risk score. "
            f"Sample indices {sample}, as (risk_flag, risk_defined): {pairs}. "
            "Stage 6 dispatches on both fields and cannot route while they "
            "contradict each other."
        )



def require_unique_index(frame: pd.DataFrame) -> None:
    """Reject a duplicated index before pandas does it less clearly.

    Stage 6 joins its output back onto the input to build explanations, which
    raises an opaque reindex error on duplicate labels. The corpus always
    carries a unique RangeIndex, so this only fires for a hand-built frame -
    but a caller deserves to be told which requirement they broke.

    Raises:
        Stage6ContractError: If the index has duplicate labels.
    """
    if not frame.index.is_unique:
        duplicated = frame.index[frame.index.duplicated()].unique().tolist()
        raise Stage6ContractError(
            f"Stage 6 requires a unique index; {len(duplicated)} label(s) are "
            f"duplicated (e.g. {duplicated[:5]}). Every routed record must be "
            "addressable, and its explanation joined back to exactly one row."
        )


def require_contract(frame: pd.DataFrame) -> None:
    """Fail loudly when a dispatch column is absent.

    Args:
        frame: The corpus frame Stage 6 is asked to route.

    Raises:
        Stage6InputError: If any required column is missing.
        Stage6ContractError: If the gates have drifted, the index is not
            unique, or Stage 5's fields contradict each other.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise Stage6InputError(
            f"Stage 6 requires Stage 4 and Stage 5 output; missing {missing!r}. "
            "Run attach_anomalies(corpus) then attach_risk(corpus) first."
        )
    # Self-validation, in the order a failure is cheapest to diagnose:
    # configuration, then shape, then cross-field consistency.
    assert_gate_alignment()
    require_unique_index(frame)
    validate_stage5_contract(frame)


# ---------------------------------------------------------------------------
# M1 - the labelling correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class M1Correction:
    """The corrected finding list, and how often the correction was needed."""

    types: pd.Series
    corrected: pd.Series

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "label": M1_CORRECTION_LABEL,
            "n_corrected": int(self.corrected.sum()),
            "pct": round(100.0 * float(self.corrected.mean()), 4)
            if len(self.corrected)
            else 0.0,
        }


def apply_m1_correction(
    frame: pd.DataFrame, escalating: Optional[np.ndarray] = None
) -> M1Correction:
    """Give every escalated record at least one named finding.

    Stage 4 gates ``underspend_anomaly`` on lifecycle: a work that is not yet
    complete cannot be accused of underspending. But Stage 4's routing and
    Stage 5's severity both read ``|z|``, so such a record can be escalated and
    scored high while carrying an empty ``anomaly_types``. A reviewer then
    receives the top of the queue with no reason attached.

    This is a **labelling** correction, not a new signal. The deviation was
    already measured by Stage 3, already scored by Stage 4 and already
    escalated by Stage 4. All that was missing was a word for it.

    The corrected list is written to a **new** column. Stage 4's
    ``anomaly_types`` is left exactly as it was: it is a locked upstream
    contract with byte-identical guarantees, and a downstream layer quietly
    rewriting it would make a re-run of Stage 5 read different inputs than the
    first run did.

    The correction is keyed on the **action**, not on ``decision_class``.
    Invariant 6 requires a finding on every ``ESCALATE_*`` record, and while
    every INVESTIGATE does escalate, the reverse is not guaranteed: a
    cross-stage disagreement can escalate a record Stage 4 chose to monitor.
    Keying on the action covers the mandated INVESTIGATE case exactly and
    closes that one too.

    Args:
        frame: Corpus frame carrying ``decision_class`` and ``anomaly_types``.
        escalating: Boolean mask of records being escalated. Defaults to
            ``decision_class == "INVESTIGATE"``, which is the mandated rule and
            the right answer when routing has not run yet.

    Returns:
        An :class:`M1Correction` aligned to ``frame.index``.
    """
    if escalating is None:
        escalating = (
            frame["decision_class"].astype("object") == "INVESTIGATE"
        ).to_numpy(dtype=bool)
    raw = frame["anomaly_types"].to_numpy()

    corrected: List[bool] = []
    types: List[List[str]] = []
    for will_escalate, value in zip(escalating, raw):
        listed = list(value) if isinstance(value, (list, tuple)) else []
        needs = bool(will_escalate) and not listed
        if needs:
            listed = [M1_CORRECTION_LABEL]
        corrected.append(needs)
        types.append(listed)

    correction = M1Correction(
        types=pd.Series(types, index=frame.index, dtype="object",
                        name="action_anomaly_types"),
        corrected=pd.Series(corrected, index=frame.index, dtype=bool,
                            name="anomaly_types_corrected"),
    )
    n = int(correction.corrected.sum())
    if n:
        LOGGER.info(
            "M1: %d escalated record(s) carried no named finding and were "
            "labelled %r. Stage 4's anomaly_types is unchanged.",
            n,
            M1_CORRECTION_LABEL,
        )
    return correction


# ---------------------------------------------------------------------------
# The policy table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One routing rule. Order in :data:`POLICY` is its precedence."""

    name: str
    action: str
    predicate: Callable[[pd.DataFrame], np.ndarray]
    note: str = ""


def _mask(frame: pd.DataFrame, column: str, value: str) -> np.ndarray:
    return (frame[column].astype("object") == value).to_numpy(dtype=bool)


def _not_defined(frame: pd.DataFrame) -> np.ndarray:
    return ~frame["risk_defined"].fillna(False).to_numpy(dtype=bool)


#: The policy, highest precedence first.
#:
#: Two entries resolve defects in the specification, both measured on the
#: 20,000-record reference corpus before being written:
#:
#: * ``investigate_low`` closes a **gap**. CASES 1-5 do not cover
#:   ``INVESTIGATE + low_risk`` (6 records). The edge-case section requires
#:   ESCALATE_REVIEW for it, so the gap is filled from there.
#: * ``remediate`` resolves a **collision**. All 2,638 REMEDIATE records also
#:   satisfy CASE 5's ``risk_defined == False``, because Stage 4 routes to
#:   REMEDIATE precisely when confidence is too low for Stage 5 to score. CASE
#:   3 names the decision class explicitly and CASE 5 is a fallback, so CASE 3
#:   wins. It is also the more actionable of the two: REMEDIATE means *this
#:   record's own evidence is weak*, which the field officer who filed it can
#:   fix, whereas the data-quality queue is for records nothing could be said
#:   about. Resolving the other way would move 2,638 records into P1, making
#:   30% of the corpus high priority and emptying the term of meaning.
POLICY: Tuple[Rule, ...] = (
    Rule(
        name="investigate_high",
        action="ESCALATE_IMMEDIATE",
        predicate=lambda f: _mask(f, "decision_class", "INVESTIGATE")
        & _mask(f, "risk_flag", "high_risk"),
        note="CASE 1",
    ),
    Rule(
        name="investigate_moderate",
        action="ESCALATE_REVIEW",
        predicate=lambda f: _mask(f, "decision_class", "INVESTIGATE")
        & _mask(f, "risk_flag", "moderate_risk"),
        note="CASE 2",
    ),
    Rule(
        name="investigate_low",
        action="ESCALATE_REVIEW",
        predicate=lambda f: _mask(f, "decision_class", "INVESTIGATE")
        & _mask(f, "risk_flag", "low_risk"),
        note="EDGE CASE 3 - closes a gap in CASES 1-5; never downgraded",
    ),
    Rule(
        name="investigate_unscored",
        action="ESCALATE_REVIEW",
        predicate=lambda f: _mask(f, "decision_class", "INVESTIGATE")
        & _not_defined(f),
        note=(
            "Backstop. Does not occur upstream (0 records): Stage 4 cannot "
            "escalate below the confidence gate, and Stage 5 uses the same "
            "gate. Kept so an INVESTIGATE can never fall through to a "
            "non-escalating action, whatever Stage 5 does later."
        ),
    ),
    Rule(
        name="disagreement_high_risk",
        action="ESCALATE_REVIEW",
        predicate=lambda f: ~_mask(f, "decision_class", "INVESTIGATE")
        & ~_mask(f, "decision_class", "REMEDIATE")
        & _mask(f, "risk_flag", "high_risk"),
        note=(
            "Cross-stage disagreement: Stage 4 found nothing to escalate while "
            "Stage 5 scored the record high. Does not occur on the reference "
            "corpus (0 records) but is reachable - Stage 4 monitors below "
            "|z| 3.5 while Stage 5 can still band a record high on a clean, "
            "fully-covered comparison. Routing it to PASSIVE_MONITOR would "
            "breach invariant 4 (high_risk is never P3). ESCALATE_REVIEW is "
            "the mirror of EDGE CASE 3, which resolves the opposite "
            "disagreement the same way: when the two stages differ, a human "
            "looks. Stage 6 invents no verdict, it declines to pick the "
            "quieter of two upstream answers."
        ),
    ),
    Rule(
        name="remediate",
        action="REQUEST_CORRECTION",
        predicate=lambda f: _mask(f, "decision_class", "REMEDIATE"),
        note="CASE 3 - wins over CASE 5; see the module note",
    ),
    Rule(
        name="insufficient_context",
        action="DATA_QUALITY_REVIEW",
        predicate=lambda f: _mask(f, "decision_class", "INSUFFICIENT_CONTEXT"),
        note="CASE 5a",
    ),
    Rule(
        name="risk_undefined",
        action="DATA_QUALITY_REVIEW",
        predicate=_not_defined,
        note="CASE 5b - anything still unscored",
    ),
    Rule(
        name="monitor",
        action="PASSIVE_MONITOR",
        predicate=lambda f: _mask(f, "decision_class", "MONITOR"),
        note="CASE 4 - last, so it can never capture an escalating record",
    ),
)


@dataclass(frozen=True)
class RoutingResult:
    """Per-record action, priority, queue, and the rule that produced them."""

    action_class: pd.Series
    priority_level: pd.Series
    reviewer_queue: pd.Series
    action_rule: pd.Series
    correction: M1Correction
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "action_class": {
                name: int((self.action_class == name).sum())
                for name in ACTION_CLASSES
            },
            "priority_level": {
                name: int((self.priority_level == name).sum())
                for name in PRIORITY_LEVELS
            },
            "reviewer_queue": {
                str(k): int(v) for k, v in self.reviewer_queue.value_counts().items()
            },
            "rule_fired": {
                str(k): int(v) for k, v in self.action_rule.value_counts().items()
            },
            "m1_correction": self.correction.to_dict(),
            **self.diagnostics,
        }


def route(frame: pd.DataFrame) -> RoutingResult:
    """Apply the policy table.

    Routing reads no finding list, so it runs BEFORE the M1 correction; the
    correction then keys off the action this produced. Invariant 6 is checked
    by the pipeline once both halves exist.

    Args:
        frame: Corpus frame with Stage 4 and Stage 5 output.

    Returns:
        A :class:`RoutingResult` aligned to ``frame.index``.

    Raises:
        Stage6InputError: If the upstream contract is incomplete.
        RuntimeError: If any record falls through every rule, which would mean
            the policy table has a hole.
    """
    require_contract(frame)
    n_records = len(frame)

    action = np.full(n_records, "", dtype=object)
    rule_name = np.full(n_records, "", dtype=object)
    unassigned = np.ones(n_records, dtype=bool)

    for rule in POLICY:
        matched = rule.predicate(frame) & unassigned
        action[matched] = rule.action
        rule_name[matched] = rule.name
        unassigned &= ~matched

    if unassigned.any():
        sample = frame.loc[unassigned, ["decision_class", "risk_flag"]].head(5)
        raise RuntimeError(
            f"{int(unassigned.sum())} record(s) matched no routing rule; the "
            f"policy table has a hole. Examples:\n{sample.to_string()}"
        )

    priority = np.array([ACTION_TO_PRIORITY[value] for value in action], dtype=object)
    queue = np.array([ACTION_TO_QUEUE[value] for value in action], dtype=object)

    escalating = np.isin(action, ESCALATING_ACTIONS)
    corrected = apply_m1_correction(frame, escalating=escalating)
    _assert_invariants(frame, action, priority, corrected)

    LOGGER.info(
        "Stage 6 routed %d record(s): %s",
        n_records,
        {k: int(v) for k, v in pd.Series(action).value_counts().items()},
    )

    return RoutingResult(
        action_class=pd.Series(action, index=frame.index, dtype="object",
                               name="action_class"),
        priority_level=pd.Series(priority, index=frame.index, dtype="object",
                                 name="priority_level"),
        reviewer_queue=pd.Series(queue, index=frame.index, dtype="object",
                                 name="reviewer_queue"),
        action_rule=pd.Series(rule_name, index=frame.index, dtype="object",
                              name="action_rule"),
        correction=corrected,
        diagnostics={
            "stage6_version": STAGE6_VERSION,
            "n_records": n_records,
            "_note": (
                "Policy, not inference. Stage 6 maps upstream decisions onto "
                "actions and computes nothing."
            ),
        },
    )


def _assert_invariants(
    frame: pd.DataFrame,
    action: np.ndarray,
    priority: np.ndarray,
    correction: M1Correction,
) -> None:
    """Enforce the six Stage 6 invariants before returning.

    These raise :class:`Stage6InvariantError` rather than asserting, so they
    survive ``python -O`` and a caller can catch a guarantee failure without
    catching every assertion in the process. A violation is still a defect in
    the policy table above, not bad input.
    """
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise Stage6InvariantError(message)

    decision = frame["decision_class"].astype("object").to_numpy()
    risk_flag = frame["risk_flag"].astype("object").to_numpy()

    # 1 & 2 - everything is routed and queued.
    _require(bool((action != "").all()), "a record has no action_class")
    _require(set(np.unique(action)) <= set(ACTION_CLASSES),
             "an undeclared action_class was emitted")
    _require(set(np.unique(priority)) <= set(PRIORITY_LEVELS),
             "an undeclared priority_level was emitted")

    # 3 - INVESTIGATE is never downgraded.
    investigate = decision == "INVESTIGATE"
    _require(not bool((investigate & (action == "PASSIVE_MONITOR")).any()),
             "I3: an INVESTIGATE record was downgraded to PASSIVE_MONITOR")
    _require(bool(np.isin(action[investigate], ESCALATING_ACTIONS).all()),
             "I3: an INVESTIGATE record was routed to a non-escalating action")

    # 4 - a high-risk record is never deprioritised.
    high = risk_flag == "high_risk"
    _require(not bool((high & (priority == "P3")).any()),
             "I2: a high_risk record was given the lowest priority")

    # 6 - an escalated record always carries a finding.
    escalating = np.isin(action, ESCALATING_ACTIONS)
    empty = correction.types.apply(lambda values: len(values) == 0).to_numpy()
    _require(not bool((escalating & empty).any()),
             "I1: an escalated record carries no named finding")

    # Edge case 1 - an unscored record never escalates to the top priority.
    undefined = ~frame["risk_defined"].fillna(False).to_numpy(dtype=bool)
    _require(not bool((undefined & (priority == "P0")).any()),
             "a record with no risk score was routed P0")

    # Edge case 2 - severity alone cannot justify escalation.
    no_severity = ~frame["severity_defined"].fillna(False).to_numpy(dtype=bool)
    _require(not bool((no_severity & (action == "ESCALATE_IMMEDIATE")).any()),
             "a record with no severity was escalated immediately")
