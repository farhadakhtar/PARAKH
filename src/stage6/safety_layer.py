"""Stage 6.5 - the first layer that deliberately changes a decision.

Every earlier stage adds information. This one takes it away: it blocks
escalations the system cannot justify, and it does so knowing that a blocked
escalation is a lead nobody looks at. That asymmetry governs the whole module.

**Nothing is overridden silently.** Every record keeps ``original_decision``,
records which rules fired, and carries a sentence saying why. A safety layer
that quietly rewrote decisions would be worse than none: the system would look
confident about a choice no one could trace.

**Every rule was measured before it was written.** The counts are in
``constants.py`` beside each rule, and two of them needed a decision that the
specification did not settle:

* **S2** reads either as "every P0/P1 record" (3,402 records move) or as its
  own title, "data-limited escalation block" (0 records). The title wins by
  default; ``S2_APPLIES_TO`` flips it.
* **S3** as literally specified would **delete 3 escalations, 2 of them P0**,
  because an unrelated record shares an injected duplicate ``work_id``.
  ``S3_PRESERVE_ESCALATIONS`` keeps those escalations while still surfacing
  the conflict on every record in the group.

Both choices are one constant each, both are documented with their measured
blast radius, and both are reversible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.core.constants import (
    ACTION_CLASSES,
    ESCALATING_ACTIONS,
    PRIORITY_LEVELS,
    S1_REASON,
    S2_APPLIES_TO,
    S2_REASON,
    S3_ESCALATION_PRESERVED_REASON,
    S3_PRESERVE_ESCALATIONS,
    S3_REASON,
    S4_REASON,
    SAFETY_DECISIONS,
    SAFETY_RULES,
    STAGE65_SAFETY_LOG,
    STAGE65_VERSION,
)
from src.core.logger import get_logger
from src.stage6.work_resolution import conflicting_work_ids, resolve_works

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

#: Columns the safety layer adds. All additive.
SAFETY_COLUMNS: Tuple[str, ...] = (
    "original_decision",
    "final_decision",
    "safety_flags",
    "safety_reason",
    "safety_intervened",
)

#: Priorities S1 and S2 consider "escalated enough to need justification".
_GUARDED_PRIORITIES: Tuple[str, ...] = ("P0", "P1")


class SafetyConfigError(ValueError):
    """Raised when the safety layer is configured incompatibly."""


@dataclass(frozen=True)
class SafetyConfig:
    """Which readings of the ambiguous rules are in force."""

    s2_applies_to: str = S2_APPLIES_TO
    s3_preserve_escalations: bool = S3_PRESERVE_ESCALATIONS

    def __post_init__(self) -> None:
        if self.s2_applies_to not in ("escalations", "all_p0_p1"):
            raise SafetyConfigError(
                f"s2_applies_to must be 'escalations' or 'all_p0_p1', got "
                f"{self.s2_applies_to!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable echo, so a run states which reading it used."""
        return {
            "s2_applies_to": self.s2_applies_to,
            "s3_preserve_escalations": self.s3_preserve_escalations,
            "_s2_note": (
                "'escalations' follows the rule's title and fires on 0 records; "
                "'all_p0_p1' follows its literal condition and moves 3,402 "
                "DATA_QUALITY_REVIEW records to REMEDIATE."
            ),
            "_s3_note": (
                "True keeps an escalation that the work-conflict override "
                "would otherwise delete (measured: 3 escalations, 2 at P0). "
                "False applies the literal specification."
            ),
        }


@dataclass(frozen=True)
class SafetyResult:
    """What the safety layer did, and to which records."""

    frame: pd.DataFrame
    log: List[Dict[str, Any]]
    config: SafetyConfig
    work_conflicts: frozenset
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)

    def interventions(self) -> pd.DataFrame:
        """Only the records the layer changed."""
        return self.frame.loc[self.frame["safety_intervened"]]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        intervened = self.frame["safety_intervened"]
        fired: Dict[str, int] = {rule: 0 for rule in SAFETY_RULES}
        for flags in self.frame["safety_flags"]:
            for rule in flags:
                if rule in fired:
                    fired[rule] += 1
        return {
            "stage65_version": STAGE65_VERSION,
            "n_records": len(self.frame),
            "n_intervened": int(intervened.sum()),
            "pct_intervened": round(100.0 * float(intervened.mean()), 4)
            if len(self.frame)
            else 0.0,
            "rules_fired": fired,
            "config": self.config.to_dict(),
            "final_decision_counts": {
                str(k): int(v)
                for k, v in self.frame["final_decision"].value_counts().items()
            },
            "n_work_conflicts": len(self.work_conflicts),
            **self.diagnostics,
        }

    def write_log(self, path: PathLike) -> Path:
        """Write the intervention log as JSONL.

        Only interventions are logged. A log of every record would bury the
        few decisions a human needs to check under 20,000 that changed nothing.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in self.log:
                handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        LOGGER.info("Wrote %d safety intervention(s) to %s", len(self.log), destination)
        return destination


def apply_safety_rules(
    frame: pd.DataFrame,
    clarity: pd.Series,
    gates_aligned: bool = True,
    config: Optional[SafetyConfig] = None,
) -> SafetyResult:
    """Apply S1-S4 in order, preserving every original decision.

    Args:
        frame: Corpus frame with Stage 6 output. **Read only** - the result is
            a new frame; nothing is written back.
        clarity: Stage 7's ``decision_clarity_flag`` per record.
        gates_aligned: Whether the Stage 4 and Stage 5 confidence gates agree.
            Supplied rather than read so a caller can exercise S4.
        config: Which readings of S2 and S3 are in force.

    Returns:
        A :class:`SafetyResult` aligned to ``frame.index``.

    Raises:
        KeyError: If a required column is absent.
    """
    for required in ("action_class", "priority_level", "work_id"):
        if required not in frame.columns:
            raise KeyError(f"safety layer requires a {required!r} column")

    settings = config or SafetyConfig()
    n_records = len(frame)
    actions = frame["action_class"].astype("object").to_numpy()
    priorities = frame["priority_level"].astype("object").to_numpy()
    clarity_values = clarity.reindex(frame.index).astype("object").to_numpy()

    escalating = np.isin(actions, ESCALATING_ACTIONS)
    guarded = np.isin(priorities, _GUARDED_PRIORITIES)

    final = actions.astype(object).copy()
    flags: List[List[str]] = [[] for _ in range(n_records)]
    reasons: List[List[str]] = [[] for _ in range(n_records)]

    # --- S4 first: a misaligned gate invalidates the basis of everything
    # below it, so it is applied before any rule that trusts a risk band.
    if not gates_aligned and "risk_flag" in frame.columns:
        high = (frame["risk_flag"].astype("object") == "high_risk").to_numpy()
        for position in np.flatnonzero(high):
            final[position] = "MONITOR"
            flags[position].append("S4")
            reasons[position].append(S4_REASON)

    # --- S1: an escalation nobody can characterise is not actionable.
    s1 = escalating & guarded & (clarity_values == "AMBIGUOUS")
    for position in np.flatnonzero(s1):
        final[position] = "ESCALATE_REVIEW_REQUIRED"
        flags[position].append("S1")
        reasons[position].append(S1_REASON)

    # --- S2: an unmeasurable record cannot support a finding.
    if settings.s2_applies_to == "escalations":
        s2 = escalating & guarded & (clarity_values == "DATA_LIMITED")
    else:
        s2 = guarded & (clarity_values == "DATA_LIMITED")
    for position in np.flatnonzero(s2):
        final[position] = "REMEDIATE"
        flags[position].append("S2")
        reasons[position].append(S2_REASON)

    # --- S3: records sharing a work_id that disagree.
    resolution = resolve_works(frame)
    conflicts = conflicting_work_ids(resolution)
    in_conflict = frame["work_id"].astype("object").isin(conflicts).to_numpy()
    n_preserved = 0
    for position in np.flatnonzero(in_conflict):
        flags[position].append("S3")
        if settings.s3_preserve_escalations and escalating[position]:
            # The conflict is still surfaced; the lead is not deleted.
            reasons[position].append(S3_ESCALATION_PRESERVED_REASON)
            n_preserved += 1
        else:
            final[position] = "INCONSISTENT_WORK"
            reasons[position].append(S3_REASON)

    intervened = final != actions
    result_frame = pd.DataFrame(
        {
            "original_decision": actions,
            "final_decision": final,
            "safety_flags": pd.Series(
                [list(entry) for entry in flags], index=frame.index, dtype="object"
            ),
            "safety_reason": pd.Series(
                [" ".join(entry) if entry else None for entry in reasons],
                index=frame.index,
                dtype="object",
            ),
            "safety_intervened": intervened,
        },
        index=frame.index,
    )

    log = [
        {
            "record_id": label,
            "work_id": str(frame.at[label, "work_id"]),
            "original_decision": str(result_frame.at[label, "original_decision"]),
            "final_decision": str(result_frame.at[label, "final_decision"]),
            "triggered_rules": list(result_frame.at[label, "safety_flags"]),
            "explanation": result_frame.at[label, "safety_reason"],
            "stage65_version": STAGE65_VERSION,
        }
        for label in result_frame.index[intervened]
    ]

    _assert_guarantees(result_frame, actions, settings)

    LOGGER.info(
        "Safety layer: %d of %d record(s) changed; %d escalation(s) preserved "
        "through a work conflict.",
        int(intervened.sum()),
        n_records,
        n_preserved,
    )
    return SafetyResult(
        frame=result_frame,
        log=log,
        config=settings,
        work_conflicts=conflicts,
        diagnostics={
            "n_escalations_preserved_through_conflict": n_preserved,
            "n_records_in_conflicting_works": int(in_conflict.sum()),
            "gates_aligned": bool(gates_aligned),
        },
    )


def _assert_guarantees(
    frame: pd.DataFrame, original: np.ndarray, config: SafetyConfig
) -> None:
    """S5 and the vocabulary guarantees.

    Raises:
        AssertionError: These are defects in this module, not bad input.
    """
    # S5 - the original decision survives, always.
    assert frame["original_decision"].notna().all(), "an original decision was lost"
    assert list(frame["original_decision"]) == list(original), (
        "S5: an original decision was modified rather than preserved"
    )

    # Every final decision is either an untouched action or a declared
    # safety substitution. Nothing else may appear.
    permitted = set(ACTION_CLASSES) | set(SAFETY_DECISIONS)
    emitted = set(frame["final_decision"].unique())
    assert emitted <= permitted, (
        f"undeclared final decision(s): {sorted(emitted - permitted)}"
    )

    # Every intervention is explained. An unexplained override is exactly the
    # silent rewrite this layer exists to prevent.
    intervened = frame["safety_intervened"].to_numpy(dtype=bool)
    assert frame.loc[intervened, "safety_reason"].notna().all(), (
        "S5: a decision changed with no reason recorded"
    )
    assert frame.loc[intervened, "safety_flags"].apply(len).gt(0).all(), (
        "S5: a decision changed with no rule recorded"
    )

    # A record nothing fired on must be untouched.
    untouched = ~intervened
    assert (
        frame.loc[untouched, "final_decision"]
        == frame.loc[untouched, "original_decision"]
    ).all(), "a record changed without being marked as intervened"
