"""What the reviewer found - stored, never fed back.

This module captures human verdicts and does **nothing** with them. That is
deliberate and worth being explicit about: the moment feedback influences a
score, the system is learning from its own outputs, and every calibration
question becomes circular. Stage 8 may close that loop with proper holdout
discipline. Stage 7 only opens the notebook.

Feedback lives in its own file, never on the corpus. A verdict is an opinion
about a decision, not a property of the record, and merging the two would make
it impossible to re-run the pipeline without carrying opinions forward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from src.core.constants import (
    FEEDBACK_OUTCOMES,
    STAGE7_REFERENCE_TIMESTAMP,
    STAGE7_VERSION,
)
from src.core.logger import get_logger
from src.stage7.interface import Stage7ContractError

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

#: Fields every feedback entry carries.
FEEDBACK_FIELDS: Sequence[str] = (
    "record_id",
    "action_taken",
    "was_correct",
    "outcome",
    "reviewer_notes",
    "timestamp",
    "stage7_version",
)


def build_feedback_entry(
    record_id: Any,
    action_taken: str,
    was_correct: bool,
    reviewer_notes: str = "",
    outcome: str = "inconclusive",
    issued_at: str = STAGE7_REFERENCE_TIMESTAMP,
) -> Dict[str, Any]:
    """Record one reviewer verdict.

    ``was_correct`` and ``outcome`` are both kept because they answer
    different questions. The boolean is about the *system* - was routing this
    record the right call. The outcome is about the *work* - was anything
    actually wrong with it. A record can be correctly escalated and turn out
    clean, and a calibration pass that conflated the two would learn the wrong
    lesson from it.

    Args:
        record_id: The record the verdict is about.
        action_taken: What the reviewer actually did, in their words.
        was_correct: Whether the routing decision was appropriate.
        reviewer_notes: Free text.
        outcome: One of :data:`FEEDBACK_OUTCOMES`.
        issued_at: ISO8601 timestamp.

    Returns:
        A JSON-serialisable entry carrying :data:`FEEDBACK_FIELDS`.

    Raises:
        Stage7ContractError: On an unknown outcome, or a non-boolean verdict.
    """
    if outcome not in FEEDBACK_OUTCOMES:
        raise Stage7ContractError(
            f"unknown feedback outcome {outcome!r}; expected one of "
            f"{FEEDBACK_OUTCOMES}"
        )
    if not isinstance(was_correct, bool):
        raise Stage7ContractError(
            f"was_correct must be a bool, got {type(was_correct).__name__}. A "
            "reviewer either endorsed the decision or did not; there is no "
            "third state, and a truthy value would hide which they meant."
        )
    return {
        "record_id": record_id,
        "action_taken": str(action_taken),
        "was_correct": was_correct,
        "outcome": outcome,
        "reviewer_notes": str(reviewer_notes),
        "timestamp": issued_at,
        "stage7_version": STAGE7_VERSION,
    }


def append_feedback(entry: Mapping[str, Any], path: PathLike) -> Path:
    """Append one verdict to the feedback log.

    Append, never rewrite: a reviewer changing their mind is a second entry,
    not an edit of the first. The history of an opinion is part of the record.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
    return destination


def write_feedback_log(
    entries: Iterable[Mapping[str, Any]], path: PathLike
) -> Path:
    """Write a whole feedback log, replacing any existing file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    LOGGER.info("Wrote feedback log to %s", destination)
    return destination


def read_feedback_log(path: PathLike) -> List[Dict[str, Any]]:
    """Read a JSONL feedback log back."""
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarise_feedback(entries: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Count verdicts. Descriptive only - it calibrates nothing.

    Returns:
        Counts by outcome and by endorsement, plus the total. Deliberately
        does not compute a precision or an accuracy: those words imply a
        ground truth, and a reviewer sample is not one.
    """
    collected = list(entries)
    outcomes: Dict[str, int] = {name: 0 for name in FEEDBACK_OUTCOMES}
    endorsed = 0
    for entry in collected:
        name = str(entry.get("outcome", ""))
        if name in outcomes:
            outcomes[name] += 1
        if bool(entry.get("was_correct")):
            endorsed += 1
    return {
        "n_entries": len(collected),
        "by_outcome": outcomes,
        "n_routing_endorsed": endorsed,
        "_note": (
            "Descriptive only. Stage 7 never feeds these back into a score: "
            "learning from the system's own outputs makes every later "
            "calibration circular. Reserved for Stage 8 under holdout "
            "discipline. No accuracy is reported because a reviewer sample "
            "is not a ground truth."
        ),
    }
