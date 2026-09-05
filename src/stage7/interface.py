"""Queues and decision cards - what a person sees and works.

Two rules govern this module, and everything else follows from them.

**The payload is the source of truth.** Every field on a card or a queue item
is read from ``explanation_payload``, the canonical JSON Stage 6 emits. The
human ``explanation`` string is carried through verbatim and is **never
parsed** - it is display only. That is not a stylistic preference: the Stage 6
audit proved the sentence is ambiguous under delimiters its vocabulary does not
currently contain, and the payload is not.

**Stage 7 decides nothing.** A queue assignment is a table lookup on the action
Stage 6 already chose. An SLA states what a priority means operationally. No
score is recomputed, reinterpreted or rescaled anywhere in this stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ACTION_CLASSES,
    ACTION_TO_QUEUE_NAME,
    CONFIDENCE_CONTEXT,
    PRIORITY_EXECUTION,
    PRIORITY_LEVELS,
    QUEUE_NAMES,
    STAGE7_REFERENCE_TIMESTAMP,
    STAGE7_VERSION,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Columns Stage 7 reads. ``explanation_payload`` carries the decision;
#: everything else is either identity or display.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "action_class",
    "priority_level",
    "action_rule",
    "explanation",
    "explanation_payload",
)

#: Read when present. ``work_id`` is the business identifier and is **not**
#: unique - Stage 1 injects duplicate ids deliberately - so it is carried
#: alongside the record id, never as it.
OPTIONAL_COLUMNS: Tuple[str, ...] = ("work_id", "risk_defined_reason", "district")

#: Fields every decision card carries, in a fixed order.
DECISION_CARD_FIELDS: Tuple[str, ...] = (
    "action",
    "priority",
    "risk",
    "decision",
    "findings",
    "reason",
    "confidence_context",
    "explanation",
)


class Stage7ContractError(RuntimeError):
    """Raised when Stage 7's input violates the contract it depends on."""


def require_contract(frame: pd.DataFrame) -> None:
    """Refuse to consume output Stage 7 cannot trust.

    Args:
        frame: The corpus frame carrying Stage 6 output.

    Raises:
        Stage7ContractError: If a required column is absent, the index is not
            unique, or any payload is missing or malformed.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise Stage7ContractError(
            f"Stage 7 requires Stage 6 output; missing {missing!r}. "
            "Run attach_actions(corpus) first."
        )

    # The record id must address exactly one record. Stage 1's work_id does
    # not - 200 of 20,000 share one - so the index carries identity here.
    if not frame.index.is_unique:
        duplicated = frame.index[frame.index.duplicated()].unique().tolist()
        raise Stage7ContractError(
            f"Stage 7 requires a unique index to key its audit log; "
            f"{len(duplicated)} label(s) are duplicated (e.g. {duplicated[:5]})."
        )

    absent = frame["explanation_payload"].isna()
    if bool(absent.any()):
        raise Stage7ContractError(
            f"{int(absent.sum())} record(s) carry no explanation_payload. The "
            "payload is Stage 7's source of truth; it cannot fall back to the "
            "human explanation, which is display only."
        )


def _decode(payload: Any, label: Any) -> Dict[str, Any]:
    """Parse one payload, or say precisely which record failed.

    Raises:
        Stage7ContractError: On malformed JSON, a non-object, or an action or
            priority outside the declared vocabularies.
    """
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise Stage7ContractError(
            f"record {label!r}: explanation_payload is not valid JSON ({exc})"
        ) from exc
    if not isinstance(decoded, dict):
        raise Stage7ContractError(
            f"record {label!r}: explanation_payload is not a JSON object"
        )

    action = decoded.get("action")
    if action not in ACTION_CLASSES:
        raise Stage7ContractError(
            f"record {label!r}: unknown action {action!r}; expected one of "
            f"{ACTION_CLASSES}"
        )
    priority = decoded.get("priority")
    if priority not in PRIORITY_LEVELS:
        raise Stage7ContractError(
            f"record {label!r}: unknown priority {priority!r}; expected one of "
            f"{PRIORITY_LEVELS}"
        )
    return decoded


def decode_payloads(frame: pd.DataFrame) -> pd.Series:
    """Decode every payload once, so nothing downstream parses twice.

    Args:
        frame: A frame satisfying :func:`require_contract`.

    Returns:
        Object Series of decoded dicts, aligned to ``frame.index``.
    """
    decoded = [
        _decode(payload, label)
        for label, payload in frame["explanation_payload"].items()
    ]
    return pd.Series(decoded, index=frame.index, dtype="object", name="payload")


def build_decision_card(
    record_id: Any, payload: Mapping[str, Any], explanation: str
) -> Dict[str, Any]:
    """Assemble the structured view a reviewer reads.

    Every field except ``explanation`` is taken from the payload. The
    explanation is passed through byte-for-byte from Stage 6 and is never
    parsed - if the two ever disagreed, the payload is right.

    Args:
        record_id: The unique record identifier.
        payload: A decoded Stage 6 payload.
        explanation: Stage 6's human sentence, verbatim.

    Returns:
        A JSON-serialisable card carrying :data:`DECISION_CARD_FIELDS`.
    """
    reason = payload.get("reason")
    return {
        "record_id": record_id,
        "action": payload["action"],
        "priority": payload["priority"],
        # None, never 0.0: an unscored record has no risk, which is a
        # different claim from a risk of zero. Stage 5 draws that line and
        # Stage 7 must not blur it.
        "risk": payload.get("risk_score"),
        "decision": payload.get("decision_class"),
        "findings": list(payload.get("findings") or []),
        "reason": reason if isinstance(reason, str) and reason else None,
        "confidence_context": _confidence_context(payload),
        "explanation": explanation,
    }


def _confidence_context(payload: Mapping[str, Any]) -> str:
    """One sentence on whether the risk number can be relied on.

    Derived from the risk band, not recomputed: an ``insufficient_data`` band
    is exactly Stage 5 saying it declined to score, and the wording is keyed on
    that so it can never contradict the number beside it.
    """
    if payload.get("risk_score") is None:
        flag = str(payload.get("risk_flag", ""))
        if flag == "insufficient_data":
            return CONFIDENCE_CONTEXT["severity_undefined"]
        return CONFIDENCE_CONTEXT["severity_undefined"]
    return CONFIDENCE_CONTEXT["ok"]


@dataclass(frozen=True)
class QueueItem:
    """One unit of work waiting for a team."""

    record_id: Any
    queue: str
    priority: str
    reason: Optional[str]
    findings: List[str]
    timestamp: str
    action: str
    sla_hours: Optional[int]
    execution_mode: str
    business_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "record_id": self.record_id,
            "queue": self.queue,
            "priority": self.priority,
            "reason": self.reason,
            "findings": list(self.findings),
            "timestamp": self.timestamp,
            "action": self.action,
            "sla_hours": self.sla_hours,
            "execution_mode": self.execution_mode,
            "business_id": self.business_id,
        }


def build_queues(
    frame: pd.DataFrame,
    payloads: pd.Series,
    issued_at: str = STAGE7_REFERENCE_TIMESTAMP,
) -> Dict[str, List[QueueItem]]:
    """Sort every record into the queue that works it.

    Ordering within a queue is by priority then by record id - both total and
    both deterministic, so two runs produce the same worklist in the same
    order. No wall clock is consulted: ``issued_at`` is injected precisely so
    that a queue can be reproduced and diffed.

    Args:
        frame: A frame satisfying :func:`require_contract`.
        payloads: Output of :func:`decode_payloads`.
        issued_at: ISO8601 timestamp stamped on every item.

    Returns:
        Every queue name mapped to its items, including empty queues - an
        absent key would read as "not computed" rather than "nothing waiting".
    """
    rank = {level: position for position, level in enumerate(PRIORITY_LEVELS)}
    queues: Dict[str, List[QueueItem]] = {name: [] for name in QUEUE_NAMES}
    # Extracted once. Per-record `.at` lookups dominated the runtime at 20k
    # records; this is a mechanical change and produces identical items.
    business_ids = (
        [str(value) for value in frame["work_id"]]
        if "work_id" in frame.columns
        else [None] * len(frame)
    )

    for (label, payload), business_id in zip(payloads.items(), business_ids):
        action = payload["action"]
        priority = payload["priority"]
        execution = PRIORITY_EXECUTION[priority]
        reason = payload.get("reason")
        queues[ACTION_TO_QUEUE_NAME[action]].append(
            QueueItem(
                record_id=label,
                queue=ACTION_TO_QUEUE_NAME[action],
                priority=priority,
                reason=reason if isinstance(reason, str) and reason else None,
                findings=list(payload.get("findings") or []),
                timestamp=issued_at,
                action=action,
                sla_hours=execution["sla_hours"],  # type: ignore[index]
                execution_mode=str(execution["mode"]),
                business_id=business_id,
            )
        )

    for name, items in queues.items():
        items.sort(key=lambda item: (rank[item.priority], str(item.record_id)))

    LOGGER.info(
        "Stage 7 queues: %s",
        {name: len(items) for name, items in queues.items()},
    )
    return queues
