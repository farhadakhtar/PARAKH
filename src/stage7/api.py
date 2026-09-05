"""The machine-facing contract.

A consumer pins :data:`~src.core.constants.STAGE7_API_VERSION` and reads the
seven top-level fields below. The shape is fixed, every field is JSON-native,
and nothing in it is derived from the human explanation string.

Stability is the whole product here. Two runs over the same records emit
byte-identical responses, because the only time value is injected rather than
read from a clock, and the encoding is canonical.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from src.core.constants import (
    STAGE7_API_VERSION,
    STAGE7_REFERENCE_TIMESTAMP,
    STAGE7_VERSION,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Top-level keys of every response. A consumer may rely on all of them
#: existing on every record, including where the value is null.
API_FIELDS: Sequence[str] = (
    "record_id",
    "action",
    "priority",
    "risk_score",
    "risk_status",
    "decision_class",
    "findings",
    "reason",
    "metadata",
)


def build_api_response(
    record_id: Any,
    payload: Mapping[str, Any],
    issued_at: str = STAGE7_REFERENCE_TIMESTAMP,
) -> Dict[str, Any]:
    """Render one record as the API contract.

    Args:
        record_id: The unique record identifier.
        payload: A decoded Stage 6 payload - the source of truth.
        issued_at: ISO8601 timestamp for ``metadata``.

    Returns:
        A JSON-serialisable mapping carrying exactly :data:`API_FIELDS`.
    """
    reason = payload.get("reason")
    return {
        "record_id": record_id,
        "action": payload["action"],
        "priority": payload["priority"],
        # null, not 0.0. The distinction between "measured as zero" and "could
        # not be measured" is the one this system exists to preserve, and it
        # must survive the API boundary intact.
        "risk_score": payload.get("risk_score"),
        # The band, reported separately so a consumer never has to infer
        # status from a null.
        "risk_status": str(payload.get("risk_flag", "")),
        "decision_class": str(payload.get("decision_class", "")),
        "findings": list(payload.get("findings") or []),
        "reason": reason if isinstance(reason, str) and reason else None,
        "metadata": {
            "source_stage": "stage6",
            "timestamp": issued_at,
            "version": STAGE7_API_VERSION,
            "stage7_version": STAGE7_VERSION,
        },
    }


def serialise(response: Mapping[str, Any]) -> str:
    """Encode a response canonically, so bytes are stable across runs."""
    return json.dumps(response, sort_keys=True, separators=(",", ":"))


def build_api_responses(
    payloads: pd.Series, issued_at: str = STAGE7_REFERENCE_TIMESTAMP
) -> List[Dict[str, Any]]:
    """Render every record, in frame order.

    Args:
        payloads: Output of :func:`~src.stage7.interface.decode_payloads`.
        issued_at: ISO8601 timestamp stamped on every response.

    Returns:
        One response per record, ordered as the frame is.
    """
    responses = [
        build_api_response(label, payload, issued_at=issued_at)
        for label, payload in payloads.items()
    ]
    LOGGER.info(
        "Rendered %d API response(s), contract %s.", len(responses), STAGE7_API_VERSION
    )
    return responses
