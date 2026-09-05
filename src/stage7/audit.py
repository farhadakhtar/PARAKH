"""The immutable record of what the system decided.

An audit entry answers one question after the fact: *what did the system say
about this record, and on what basis?* It therefore carries the payload
verbatim rather than a summary of it - a summary is a second opinion, and two
opinions in a log is how a dispute becomes unresolvable.

The hash covers the decision content, not the log entry, so two runs over
unchanged input produce identical hashes and a diff shows only what actually
moved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

import pandas as pd

from src.core.constants import STAGE7_REFERENCE_TIMESTAMP, STAGE7_VERSION
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

#: Fields every entry carries. No entry may omit one, even as null.
AUDIT_FIELDS: Sequence[str] = (
    "record_id",
    "input_hash",
    "action",
    "priority",
    "decision_class",
    "risk_score",
    "timestamp",
    "explanation_payload",
    "stage7_version",
)


def compute_input_hash(record_id: Any, payload_json: str) -> str:
    """Hash the decision this entry records.

    Deterministic by construction: it covers the record identity and the
    canonical payload bytes Stage 6 emitted, and nothing else. The timestamp
    is excluded deliberately - hashing it would make every replay look like a
    change, which defeats the purpose of having a hash at all.

    Args:
        record_id: The unique record identifier.
        payload_json: Stage 6's canonical payload string, unmodified.

    Returns:
        A 64-character SHA-256 hex digest.
    """
    material = json.dumps(
        {"record_id": str(record_id), "payload": payload_json},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_audit_entry(
    record_id: Any,
    payload: Mapping[str, Any],
    payload_json: str,
    issued_at: str = STAGE7_REFERENCE_TIMESTAMP,
) -> Dict[str, Any]:
    """Build one immutable log entry.

    Args:
        record_id: The unique record identifier.
        payload: The decoded payload, for the indexed fields.
        payload_json: The canonical payload string, stored verbatim.
        issued_at: ISO8601 timestamp.

    Returns:
        A JSON-serialisable entry carrying exactly :data:`AUDIT_FIELDS`.
    """
    return {
        "record_id": record_id,
        "input_hash": compute_input_hash(record_id, payload_json),
        "action": payload["action"],
        "priority": payload["priority"],
        "decision_class": str(payload.get("decision_class", "")),
        "risk_score": payload.get("risk_score"),
        "timestamp": issued_at,
        # Verbatim. A log that paraphrases its own evidence is not a log.
        "explanation_payload": payload_json,
        "stage7_version": STAGE7_VERSION,
    }


def build_audit_log(
    frame: pd.DataFrame,
    payloads: pd.Series,
    issued_at: str = STAGE7_REFERENCE_TIMESTAMP,
) -> List[Dict[str, Any]]:
    """Build one entry per record, in frame order.

    Raises:
        ValueError: If any entry would be incomplete, which would make the log
            unusable as evidence.
    """
    # Extracted once rather than looked up per record; identical output.
    payload_json = [str(value) for value in frame["explanation_payload"]]
    entries = [
        build_audit_entry(label, payload, raw, issued_at=issued_at)
        for (label, payload), raw in zip(payloads.items(), payload_json)
    ]
    for entry in entries:
        missing = [name for name in AUDIT_FIELDS if name not in entry]
        if missing:
            raise ValueError(
                f"audit entry for {entry.get('record_id')!r} is missing "
                f"{missing!r}; an incomplete log is not evidence"
            )
    LOGGER.info("Built %d audit entr(ies).", len(entries))
    return entries


def write_audit_log(entries: Iterable[Mapping[str, Any]], path: PathLike) -> Path:
    """Write the log as JSONL, one entry per line.

    JSONL rather than a single JSON array because an audit log is appended per
    record and read by streaming; an array would have to be rewritten whole,
    and a rewritten log is a mutable one.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    LOGGER.info("Wrote audit log to %s", destination)
    return destination


def read_audit_log(path: PathLike) -> List[Dict[str, Any]]:
    """Read a JSONL audit log back."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
