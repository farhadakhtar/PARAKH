"""Stage 8 - matching audit findings to records.

This module decides, for each (record, finding) pair, whether the finding says
anything about that record and how firmly. It is the component that turns a
CAG paragraph into a label, so its failure mode is the expensive one: a wrong
match does not raise, it produces a confident label that teaches a model to
predict audit coverage rather than irregularity.

Two implementations are kept deliberately
-----------------------------------------
:func:`match_naive` compares every record against every finding. It is
obviously correct and unusably slow.

:func:`match_blocked` uses an inverted index to compare only records and
findings that already agree on some key. It is what runs.

Both are kept, and ``tests/test_stage8_matching.py`` asserts they return
identical matches on every input it can generate. That parity test is the
entire correctness argument for blocking: the point of an index is to change
the cost and not the answer, and a blocking key that quietly drops true pairs
is invisible downstream, because a metric computed on the labels that exist
cannot notice the labels that were never produced.

The cascade
-----------
Keys are tried strongest first, and a pair is recorded at the strongest key it
shares - so ``match_method`` is always the best available evidence for that
pair, never merely the first one tried.

The cascade stops at ``DISTRICT_YEAR``. Weaker keys (``STATE_YEAR``,
``STATE_ONLY``) map to ``LEVEL_0_GEOGRAPHIC_COINCIDENCE``, which is defined as
not being evidence about a record, so generating those pairs costs work to
produce rows that every downstream gate is required to discard. On the real
corpus the weakest tier alone would have produced roughly 100,000 comparisons
and exactly zero usable labels.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.core.constants import (
    MATCH_METHOD_CONFIDENCE,
    MATCH_METHOD_EVIDENCE,
    MATCH_METHODS,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Columns every match frame carries.
MATCH_COLUMNS: Tuple[str, ...] = (
    "record_id",
    "audit_id",
    "match_method",
    "match_confidence",
    "evidence_level",
    "match_key",
)

#: Values that look like data but assert nothing. They must never become a
#: join key: if ``UNKNOWN`` were a bucket, every record with a missing
#: district would match every finding with a missing district, manufacturing a
#: large and entirely artificial match set.
NULL_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "-",
        "--",
        "NA",
        "N/A",
        "NAN",
        "NONE",
        "NULL",
        "NIL",
        "UNKNOWN",
        "NOT AVAILABLE",
        "NOT APPLICABLE",
        "TOTAL",
        "GRAND TOTAL",
        "ALL INDIA",
        "OTHERS",
        "OTHER",
    }
)

#: Shortest string that can be a real place or scheme name. Guards against
#: single-character artefacts becoming buckets; a real corpus supplied a state
#: literally called "-".
#:
#: Applies to NAMES ONLY. Identifiers are exempt: "W1" is a perfectly good
#: work_id and rejecting it would silently delete the strongest evidence level
#: the matcher has. Found by the ordering test, which expected WORK_ID first
#: and got SCHEME_DISTRICT_YEAR because the work_id had been normalised away.
MIN_KEY_LENGTH: int = 3

#: Fields holding identifiers rather than names.
IDENTIFIER_FIELDS: frozenset[str] = frozenset({"work_id", "transaction_id"})

_PUNCTUATION = re.compile(r"[^A-Z0-9 ]+")
_WHITESPACE = re.compile(r"\s+")

#: Scheme names appear as an acronym in structured data and spelled out in
#: audit prose. Without this, the two never share a bucket and every
#: scheme-level match is lost - a pure recall failure, and the kind that is
#: invisible because it produces no output to inspect.
#:
#: Deliberately a fixed table rather than fuzzy string similarity: a
#: similarity threshold that merges "PMGSY" with "PMAY" would be a precision
#: failure in the same component, and these names are few enough to enumerate.
SCHEME_ALIASES: Mapping[str, str] = {
    "PRADHAN MANTRI GRAM SADAK YOJANA": "PMGSY",
    "PRADHAN MANTRI GRAMIN SADAK YOJANA": "PMGSY",
    "GRAM SADAK YOJANA": "PMGSY",
    "MAHATMA GANDHI NATIONAL RURAL EMPLOYMENT GUARANTEE ACT": "MGNREGA",
    "MAHATMA GANDHI NATIONAL RURAL EMPLOYMENT GUARANTEE SCHEME": "MGNREGA",
    "NATIONAL RURAL EMPLOYMENT GUARANTEE ACT": "MGNREGA",
    "NREGA": "MGNREGA",
    "MGNREGS": "MGNREGA",
    "PRADHAN MANTRI AWAAS YOJANA": "PMAY",
    "PRADHAN MANTRI AWAS YOJANA": "PMAY",
    "NATIONAL RURAL LIVELIHOOD MISSION": "NRLM",
    "DEENDAYAL ANTYODAYA YOJANA": "NRLM",
    "SWACHH BHARAT MISSION": "SBM",
    "JAL JEEVAN MISSION": "JJM",
    "MEMBER OF PARLIAMENT LOCAL AREA DEVELOPMENT SCHEME": "MPLADS",
}

#: State-name variants that are the same place. "&" versus "and" is the common
#: one and appears in almost every Indian government table.
STATE_ALIASES: Mapping[str, str] = {
    "JAMMU AND KASHMIR": "JAMMU AND KASHMIR",
    "J AND K": "JAMMU AND KASHMIR",
    "JK": "JAMMU AND KASHMIR",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "UTTARANCHAL": "UTTARAKHAND",
    "KERALAM": "KERALA",
    "NCT OF DELHI": "DELHI",
    "DELHI NCT": "DELHI",
}

#: Keys the cascade generates, strongest first. Stops above LEVEL_0 - see the
#: module docstring.
CASCADE_METHODS: Tuple[str, ...] = (
    "WORK_ID",
    "TRANSACTION_ID",
    "SCHEME_WORK_DISTRICT_YEAR",
    "VENDOR_DISTRICT_YEAR",
    "SCHEME_DISTRICT_YEAR",
    "DISTRICT_YEAR",
)

#: Which record fields each key is built from. A key is only generated when
#: every one of its fields is present, because a key with a hole in it is a
#: weaker key wearing a stronger key's name.
METHOD_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "WORK_ID": ("work_id",),
    "TRANSACTION_ID": ("transaction_id",),
    "SCHEME_WORK_DISTRICT_YEAR": ("scheme", "work_id", "district", "financial_year"),
    "VENDOR_DISTRICT_YEAR": ("vendor", "district", "financial_year"),
    "SCHEME_DISTRICT_YEAR": ("scheme", "district", "financial_year"),
    "DISTRICT_YEAR": ("district", "financial_year"),
}


def normalise_key(value: Any, *, is_identifier: bool = False) -> Optional[str]:
    """Canonical form of a join key, or None when the value asserts nothing.

    Returning None rather than a sentinel string is deliberate: None cannot
    accidentally become a bucket, whereas the string ``"UNKNOWN"`` can and
    once did.

    Args:
        value: Raw cell contents from a record or a parsed finding.
        is_identifier: True for work/transaction IDs. Skips the minimum-length
            and alias rules, which describe place names and would corrupt an
            identifier that happens to be short or to collide with an alias.

    Returns:
        An uppercase, punctuation-free, alias-resolved key, or None when the
        value is missing, a null token, or too short to be a real name.
    """
    if value is None:
        return None
    text = str(value)
    if text.strip().lower() in {"nan", "nat", "none", "<na>"}:
        return None
    text = text.upper().replace("&", " AND ")
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if not text or text in NULL_TOKENS:
        return None
    if is_identifier:
        return text
    if len(text) < MIN_KEY_LENGTH:
        return None
    text = STATE_ALIASES.get(text, text)
    return SCHEME_ALIASES.get(text, text)


def blocking_keys(row: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Every blocking key a row supports, strongest first.

    Args:
        row: A record or finding as a mapping of field name to value.

    Returns:
        ``(method, key)`` pairs ordered by descending match confidence. A
        method is omitted entirely when any field it needs is missing.
    """
    keys: List[Tuple[str, str]] = []
    for method in CASCADE_METHODS:
        parts: List[str] = []
        for field_name in METHOD_FIELDS[method]:
            normalised = normalise_key(
                row.get(field_name),
                is_identifier=field_name in IDENTIFIER_FIELDS,
            )
            if normalised is None:
                parts = []
                break
            parts.append(normalised)
        if parts:
            keys.append((method, "|".join(parts)))
    return tuple(keys)


@dataclass(frozen=True)
class MatchResult:
    """Matched pairs plus the cost it took to find them.

    ``comparisons`` is reported so the blocking speedup is a measured number
    rather than a claim, and so a regression that quietly disables blocking
    shows up as a cost change instead of passing silently.
    """

    matches: pd.DataFrame
    comparisons: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=object) for name in MATCH_COLUMNS})


def _rows(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Frame as plain dicts, so both matchers read identical inputs."""
    return frame.to_dict(orient="records")


def _finalise(
    pairs: Mapping[Tuple[str, str], Tuple[str, str]], comparisons: int
) -> MatchResult:
    """Build the output frame from strongest-key-per-pair decisions.

    Sorted on the way out so the result is a function of content alone; an
    unsorted frame would still be *correct* but would make the determinism and
    permutation tests fail for a reason that has nothing to do with matching.
    """
    if not pairs:
        return MatchResult(_empty_matches(), comparisons)

    records = [
        {
            "record_id": record_id,
            "audit_id": audit_id,
            "match_method": method,
            "match_confidence": MATCH_METHOD_CONFIDENCE[method],
            "evidence_level": MATCH_METHOD_EVIDENCE[method],
            "match_key": key,
        }
        for (record_id, audit_id), (method, key) in pairs.items()
    ]
    frame = pd.DataFrame(records, columns=list(MATCH_COLUMNS))
    frame = frame.sort_values(["record_id", "audit_id"]).reset_index(drop=True)
    return MatchResult(frame, comparisons)


def _strength(method: str) -> float:
    return MATCH_METHOD_CONFIDENCE[method]


def _keep_strongest(
    pairs: Dict[Tuple[str, str], Tuple[str, str]],
    record_id: str,
    audit_id: str,
    method: str,
    key: str,
) -> None:
    """Record a pair at ``method``, unless a stronger key already matched it.

    Ties are broken by key text so the outcome does not depend on which order
    two equally strong methods were visited in.
    """
    identity = (record_id, audit_id)
    existing = pairs.get(identity)
    if existing is None:
        pairs[identity] = (method, key)
        return
    current_method, current_key = existing
    candidate = (_strength(method), method, key)
    incumbent = (_strength(current_method), current_method, current_key)
    if candidate > incumbent:
        pairs[identity] = (method, key)


def match_naive(records: pd.DataFrame, findings: pd.DataFrame) -> MatchResult:
    """Reference matcher: every record against every finding.

    Exists to be compared against, not to be run on real data. Its cost is
    ``len(records) * len(findings)`` by construction.

    Args:
        records: Structured records, one row each, carrying ``record_id``.
        findings: Parsed audit findings carrying ``audit_id``.

    Returns:
        A :class:`MatchResult` whose ``matches`` is one row per matched pair,
        at the strongest key that pair shares.
    """
    if records.empty or findings.empty:
        return MatchResult(_empty_matches(), 0)

    record_rows = _rows(records)
    finding_rows = _rows(findings)
    pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}

    for record in record_rows:
        record_keys = dict(blocking_keys(record))
        for finding in finding_rows:
            finding_keys = dict(blocking_keys(finding))
            for method in CASCADE_METHODS:
                key = record_keys.get(method)
                if key is not None and finding_keys.get(method) == key:
                    _keep_strongest(
                        pairs,
                        str(record["record_id"]),
                        str(finding["audit_id"]),
                        method,
                        key,
                    )
                    break

    return _finalise(pairs, len(record_rows) * len(finding_rows))


def match_blocked(records: pd.DataFrame, findings: pd.DataFrame) -> MatchResult:
    """Blocked matcher: compare only rows that already share a key.

    Builds one inverted index per cascade level over the findings, then probes
    it with each record's keys. A record is compared against a finding only
    when the two already agree on something, so the comparison count falls
    from ``n * m`` to the number of genuine candidates.

    Every level is probed rather than stopping at the first hit, because a
    record may match one finding on ``WORK_ID`` and a different finding only
    on ``DISTRICT_YEAR``; stopping early would drop the second pair and break
    parity with :func:`match_naive`.

    Args:
        records: Structured records carrying ``record_id``.
        findings: Parsed audit findings carrying ``audit_id``.

    Returns:
        A :class:`MatchResult` identical to :func:`match_naive`'s, with a
        much smaller ``comparisons``.
    """
    if records.empty or findings.empty:
        return MatchResult(_empty_matches(), 0)

    finding_rows = _rows(findings)

    # One bucket table per level. Keyed by (method, key) so two levels can
    # never collide on a shared key string.
    index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for finding in finding_rows:
        audit_id = str(finding["audit_id"])
        for method, key in blocking_keys(finding):
            index[(method, key)].append(audit_id)

    pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    comparisons = 0

    for record in _rows(records):
        record_id = str(record["record_id"])
        for method, key in blocking_keys(record):
            bucket = index.get((method, key))
            if not bucket:
                continue
            comparisons += len(bucket)
            for audit_id in bucket:
                _keep_strongest(pairs, record_id, audit_id, method, key)

    result = _finalise(pairs, comparisons)
    LOGGER.info(
        "Stage 8 matching: %d record(s) x %d finding(s) -> %d pair(s) "
        "in %d comparison(s) (%.4f%% of the full product)",
        len(records),
        len(findings),
        len(result.matches),
        comparisons,
        100.0 * comparisons / max(1, len(records) * len(findings)),
    )
    return result


def match_summary(result: MatchResult) -> Dict[str, Any]:
    """Counts by method and evidence level, for the inventory and the report."""
    matches = result.matches
    if matches.empty:
        return {
            "n_matches": 0,
            "comparisons": result.comparisons,
            "by_method": {},
            "by_evidence_level": {},
        }
    return {
        "n_matches": int(len(matches)),
        "comparisons": int(result.comparisons),
        "by_method": {
            str(k): int(v) for k, v in matches["match_method"].value_counts().items()
        },
        "by_evidence_level": {
            str(k): int(v)
            for k, v in matches["evidence_level"].value_counts().items()
        },
    }
