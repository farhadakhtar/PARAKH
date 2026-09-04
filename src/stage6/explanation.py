"""The human-facing sentence, in the exact format the policy specifies.

Five lines, fixed order, one field each::

    Record routed to {action_class} because:
    - Findings: {anomaly_types}
    - Severity: {severity_score or 'not defined'}
    - Risk: {risk_score or 'not defined'}
    - Decision basis: {decision_class} with {risk_flag}

Every value is read from a column that already exists. Nothing is computed,
nothing is inferred, and no field appears that is not in the input contract -
which is what makes the round-trip test possible: the text can be parsed back
into a dict and compared field by field against the stored values.

The format is fixed rather than adaptive on purpose. A queue that a human reads
hundreds of times a day should look the same every time, so the eye can find
the one line it cares about without re-reading the sentence.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.constants import SPEC_ACTION_ALIAS
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: What an absent number reads as. One string, used everywhere, so a parser can
#: recognise it and a reader can never mistake it for a measured zero.
NOT_DEFINED = "not defined"

#: The field labels, in the order they appear. The parser is built from this,
#: so the writer and the reader cannot drift apart.
FIELD_ORDER: Sequence[str] = ("Findings", "Severity", "Risk", "Decision basis")


def _number(value: Any, digits: int = 3) -> str:
    """Format a number, or say plainly that it is absent."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NOT_DEFINED
    if not np.isfinite(number):
        return NOT_DEFINED
    return f"{number:.{digits}f}"


def _findings(value: Any) -> str:
    """Render the finding list. Never empty on an escalated record."""
    if isinstance(value, (list, tuple)) and len(value):
        return ", ".join(str(item) for item in value)
    return "none recorded"


def explain_action(row: Mapping[str, Any]) -> str:
    """Compose the routing explanation for one record.

    Args:
        row: A mapping carrying ``action_class``, ``action_anomaly_types``,
            ``severity_score``, ``severity_defined``, ``risk_score``,
            ``risk_defined``, ``decision_class`` and ``risk_flag``.

    Returns:
        The five-line explanation, exactly as specified.
    """
    action = str(row.get("action_class", ""))
    findings = _findings(row.get("action_anomaly_types"))

    # `severity_defined` and `risk_defined` are authoritative. A value is only
    # printed when the upstream stage says it exists, so a stray number can
    # never be presented as a measurement.
    severity = (
        _number(row.get("severity_score"))
        if bool(row.get("severity_defined", False))
        else NOT_DEFINED
    )
    risk = (
        _number(row.get("risk_score"))
        if bool(row.get("risk_defined", False))
        else NOT_DEFINED
    )

    return (
        f"Record routed to {action} because:\n"
        f"- Findings: {findings}\n"
        f"- Severity: {severity}\n"
        f"- Risk: {risk}\n"
        f"- Decision basis: {row.get('decision_class', '')} with "
        f"{row.get('risk_flag', '')}"
    )


def parse_action_explanation(text: str) -> Dict[str, str]:
    """Read an explanation back into its fields.

    The inverse of :func:`explain_action`, provided in the module it mirrors so
    the two cannot drift. Its purpose is verification: a test parses every
    generated explanation and compares each field against the stored column, so
    a narrative that stops matching its own record fails the build.

    **Not the machine contract.** This parser is exact for the current fixed
    vocabularies and ambiguous outside them - see
    :func:`build_action_payload` for why. A consumer that needs to read a
    routing decision programmatically should read ``explanation_payload``,
    which round-trips arbitrary content and carries the priority this format
    does not.

    Args:
        text: An explanation produced by :func:`explain_action`.

    Returns:
        A mapping with ``action_class`` and one entry per :data:`FIELD_ORDER`.

    Raises:
        ValueError: If the text does not have the expected shape.
    """
    lines = text.split("\n")
    if len(lines) != 1 + len(FIELD_ORDER):
        raise ValueError(f"expected {1 + len(FIELD_ORDER)} lines, got {len(lines)}")

    head = lines[0]
    prefix, suffix = "Record routed to ", " because:"
    if not head.startswith(prefix) or not head.endswith(suffix):
        raise ValueError(f"malformed first line: {head!r}")

    parsed: Dict[str, str] = {"action_class": head[len(prefix) : -len(suffix)]}
    for label, line in zip(FIELD_ORDER, lines[1:]):
        marker = f"- {label}: "
        if not line.startswith(marker):
            raise ValueError(f"expected {marker!r}, got {line!r}")
        parsed[label] = line[len(marker) :]
    return parsed


#: Fields carried by the machine-readable payload, in a fixed order.
PAYLOAD_FIELDS: Sequence[str] = (
    # --- the specified seven ---------------------------------------------
    "action",
    "priority",
    "rule",
    "decision_class",
    "risk_flag",
    "findings",
    "reason",
    # --- retained from the previous contract, additive --------------------
    # `anomaly_types` is a synonym of `findings` and is kept so that a
    # consumer written against the earlier payload keeps working. Both are
    # built from the same list object, and an assertion checks they agree.
    "action_spec",
    "anomaly_types",
    "severity_score",
    "risk_score",
)


def build_action_payload(row: Mapping[str, Any]) -> str:
    """Render one record's routing decision as canonical JSON.

    Why a second form at all
    -----------------------
    The five-line sentence above is written for a person, and it pays for that
    with two defects the audit proved by construction:

    * it does not carry ``priority``, so a reader cannot see urgency and a
      parser cannot recover it (0 of 20,000 records);
    * its delimiters are ambiguous. A finding containing ``", "`` parses back
      as two findings, a ``decision_class`` containing ``" with "`` splits in
      the wrong place, and a finding literally named ``"none recorded"`` is
      indistinguishable from having none.

    Neither is reachable under the current fixed vocabularies - the eight
    finding labels, four decision classes and four risk flags contain none of
    those delimiters - but nothing asserted that they never would.

    JSON removes the whole class: it escapes every delimiter, distinguishes an
    empty list from any string, and round-trips arbitrary content. This is the
    **authoritative machine contract**; the sentence remains the human one, and
    is left byte-identical so that every existing consumer and test is
    unaffected.

    Args:
        row: A mapping carrying the Stage 6 output columns for one record.

    Returns:
        A compact JSON object string with sorted keys - deterministic, so two
        runs on the same record produce identical bytes.
    """
    action = str(row.get("action_class", ""))
    types = row.get("action_anomaly_types")
    findings = list(types) if isinstance(types, (list, tuple)) else []
    # `reason` is the upstream decision_reason where Stage 4 supplied one.
    # None, never "", so an absent reason cannot be mistaken for an empty one.
    reason = row.get("decision_reason")
    payload: Dict[str, Any] = {
        "action": action,
        "action_spec": SPEC_ACTION_ALIAS.get(action),
        "priority": str(row.get("priority_level", "")),
        "rule": str(row.get("action_rule", "")),
        "decision_class": str(row.get("decision_class", "")),
        "risk_flag": str(row.get("risk_flag", "")),
        "findings": findings,
        "reason": str(reason) if isinstance(reason, str) and reason else None,
        # Synonym of `findings`, retained for the earlier payload contract.
        "anomaly_types": findings,
        # The definedness flag stays authoritative here too: a value is
        # carried only when the upstream stage says it exists.
        "severity_score": _payload_number(
            row.get("severity_score"), bool(row.get("severity_defined", False))
        ),
        "risk_score": _payload_number(
            row.get("risk_score"), bool(row.get("risk_defined", False))
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _payload_number(value: Any, defined: bool) -> Optional[float]:
    """A float when the stage says it exists and it is finite, else None.

    ``None`` rather than ``NaN``: JSON has no NaN, and a null is unambiguous
    where a bare number would invite a reader to treat an absent measurement
    as a measured zero.
    """
    if not defined:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def parse_action_payload(payload: str) -> Dict[str, Any]:
    """Read a payload back into its fields. Exact inverse of the builder.

    Args:
        payload: A string produced by :func:`build_action_payload`.

    Returns:
        The decoded mapping, with every :data:`PAYLOAD_FIELDS` key present.

    Raises:
        ValueError: If the payload is not a JSON object or omits a field.
    """
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"payload is not a JSON object: {type(decoded).__name__}")
    missing = [name for name in PAYLOAD_FIELDS if name not in decoded]
    if missing:
        raise ValueError(f"payload is missing field(s): {missing}")
    return decoded


def build_action_payloads(frame: pd.DataFrame) -> pd.Series:
    """Generate the machine-readable payload for every record."""
    if len(frame) == 0:
        return pd.Series([], dtype="object", index=frame.index,
                         name="explanation_payload")
    columns = list(frame.columns)
    payloads = [
        build_action_payload(dict(zip(columns, values)))
        for values in frame.itertuples(index=False, name=None)
    ]
    return pd.Series(payloads, index=frame.index, dtype="object",
                     name="explanation_payload")


def build_action_explanations(frame: pd.DataFrame) -> pd.Series:
    """Generate the explanation for every record.

    Args:
        frame: The assembled Stage 6 frame joined with its Stage 4-5 context.

    Returns:
        Object Series aligned to ``frame.index``.
    """
    if len(frame) == 0:
        return pd.Series([], dtype="object", index=frame.index, name="explanation")
    columns = list(frame.columns)
    texts = [
        explain_action(dict(zip(columns, values)))
        for values in frame.itertuples(index=False, name=None)
    ]
    LOGGER.info("Generated %d routing explanation(s).", len(texts))
    return pd.Series(texts, index=frame.index, dtype="object", name="explanation")
