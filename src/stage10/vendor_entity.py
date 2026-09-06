"""Stage 10 - vendor entity resolution.

Deterministic normalisation first, then exact matching on the normalised
form. Fuzzy matching is available but deliberately narrow, and never earns
HIGH confidence.

Why the fuzzy threshold is set so high
--------------------------------------
"Kumar" and "Kumari" are one character apart and are different companies.
"ABC Constructions" and "XYZ Constructions" share two of three tokens. An
edit-distance rule loose enough to merge the first pair merges most of a
district, and the resulting entity is not a vendor - it is an artefact that
every concentration statistic downstream will then treat as one firm.

So the ordering is: strip what is provably not identity (legal suffixes,
honorifics, punctuation), match exactly on what remains, and treat anything
short of that as a separate entity with the ambiguity recorded. A prior
method survey on this project concluded, on ARACHNE guidance, that a
name-only resolver degrades badly without a registration number, an address
or a director - none of which the corpus has. That conclusion is respected
here: this module resolves obvious spelling variants and stops.

What it must never do
---------------------
Merge on weak similarity, or let a missing vendor become a bucket. If empty
names normalised to a shared key, every record without a vendor would resolve
to one entity - the largest and most spurious firm in the corpus, and one that
would top any concentration ranking.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.constants import (
    VENDOR_FUZZY_SIMILARITY,
    VENDOR_LEGAL_SUFFIXES,
    VENDOR_MIN_NAME_LENGTH,
    VENDOR_NAME_PREFIXES,
    VENDOR_NULL_TOKENS,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

VENDOR_COLUMNS: Tuple[str, ...] = (
    "vendor_entity_id",
    "vendor_confidence",
    "vendor_consistency",
    "vendor_normalised",
    "vendor_group_size",
)

_PUNCT = re.compile(r"[^a-z0-9ऀ-ॿ ]+")
_SPACE = re.compile(r"\s+")

#: A run of standalone single letters - an initialism whose dots were
#: stripped by punctuation removal. Anchored on BOTH sides: unanchored,
#: the pattern also matches the "e l" spanning "privatE Limited" and
#: welds the two words together, so the legal suffix is then never
#: recognised and never stripped.
_INITIALISM = re.compile("\\b(?:[a-z] )+[a-z]\\b")

#: Longest suffixes first, so "private limited" is stripped as a unit rather
#: than leaving a stray "private" behind after "limited" is removed.
_SUFFIXES = tuple(
    sorted(VENDOR_LEGAL_SUFFIXES, key=len, reverse=True)
)

#: Prefixes as they appear BEFORE and AFTER punctuation stripping. "m/s"
#: becomes "m s" once "/" is removed, so both spellings must be listed or the
#: honorific survives into the key and "M/s Sharma" never matches "Sharma".
_PREFIXES = tuple(
    sorted(
        set(VENDOR_NAME_PREFIXES) | {"m s", "ms", "messrs"},
        key=len,
        reverse=True,
    )
)

#: Written-out forms that mean the same as a punctuation-joined one. Applied
#: before punctuation is stripped, because "&" and "and" are the same word and
#: stripping "&" first would leave "xyz co" against "xyz and co".
_WORD_EQUIVALENTS = (
    (r"\s*&\s*", " and "),
    (r"\bpvt\b", " private "),
    (r"\bltd\b", " limited "),
)


def normalise_vendor_name(value: Any) -> Optional[str]:
    """Canonical vendor key, or None when the value names nobody.

    None rather than a sentinel string, deliberately: None cannot become a
    bucket, whereas ``"UNKNOWN"`` can and would collect every record with a
    missing vendor into one entity.

    Args:
        value: Raw vendor cell.

    Returns:
        A lowercase, suffix-stripped, punctuation-free key, or None when the
        value is missing, a null token, or too short to be a firm name.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "<na>"}:
        return None

    for pattern, replacement in _WORD_EQUIVALENTS:
        text = re.sub(pattern, replacement, text)
    text = _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()

    # Collapse initialisms: stripping punctuation turns "A.B.C." into "a b c"
    # while plain "ABC" stays "abc", so the two spellings of one firm would
    # never match. A run of single letters is an initialism in every case that
    # matters here, so it is rejoined.
    text = _INITIALISM.sub(lambda m: m.group(0).replace(" ", ""), text)

    # Prefixes first: "m/s sharma builders" must lose the honorific before
    # the length check, or a short real name behind one would survive on the
    # honorific's characters.
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if text.startswith(prefix + " "):
                text = text[len(prefix) + 1:].strip()
                changed = True
            elif text == prefix:
                # A bare honorific names nobody. "M/s" alone must not survive
                # as a vendor key.
                text = ""
                changed = True
        for suffix in _SUFFIXES:
            if text.endswith(" " + suffix):
                text = text[: -len(suffix) - 1].strip()
                changed = True
            elif text == suffix:
                text = ""
                changed = True

    text = _SPACE.sub(" ", text).strip()
    if not text or text in VENDOR_NULL_TOKENS:
        return None
    if len(text) < VENDOR_MIN_NAME_LENGTH:
        return None
    return text


def vendor_similarity(left: str, right: str) -> float:
    """Deterministic similarity of two normalised vendor keys, in [0, 1]."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return float(SequenceMatcher(None, left, right).ratio())


def resolve_vendor_entities(
    frame: pd.DataFrame, *, allow_fuzzy: bool = False
) -> pd.DataFrame:
    """Group records into candidate vendor entities.

    Args:
        frame: Corpus records. Read only; the index is preserved and no
            original column is modified.
        allow_fuzzy: Enable near-match merging above
            :data:`VENDOR_FUZZY_SIMILARITY`. **Off by default.** A fuzzy merge
            asserts two differently-spelled firms are one firm, and with only
            a free-text name to go on that assertion is not supportable. When
            enabled, such groups are capped at MEDIUM confidence and marked
            AMBIGUOUS.

    Returns:
        A copy of ``frame`` with :data:`VENDOR_COLUMNS` attached.

    Raises:
        KeyError: If no vendor column is present.
    """
    column = next(
        (c for c in ("vendor_name", "vendor", "contractor_name") if c in frame.columns),
        None,
    )
    if column is None:
        raise KeyError("vendor resolution requires a vendor_name column")

    out = frame.copy()
    if out.empty:
        for name in VENDOR_COLUMNS:
            out[name] = pd.Series(dtype=object)
        return out

    normalised = out[column].map(normalise_vendor_name)
    out["vendor_normalised"] = normalised

    # Exact match on the normalised key. Everything unresolvable gets its own
    # entity keyed on its row position: two unknown vendors are not the same
    # vendor, and pretending otherwise creates a phantom firm.
    keys: Dict[str, str] = {}
    assigned: List[str] = []
    confidence: List[str] = []
    consistency: List[str] = []

    for position, (label, key) in enumerate(normalised.items()):
        if key is None:
            assigned.append(f"VE-UNRESOLVED-{position:07d}")
            confidence.append("LOW")
            consistency.append("AMBIGUOUS")
            continue
        if key not in keys:
            keys[key] = f"VE-{len(keys):07d}"
        assigned.append(keys[key])
        confidence.append("HIGH")
        consistency.append("CONSISTENT")

    out["vendor_entity_id"] = assigned
    out["vendor_confidence"] = confidence
    out["vendor_consistency"] = consistency

    if allow_fuzzy and len(keys) > 1:
        merged = _fuzzy_merge(keys)
        if merged:
            out["vendor_entity_id"] = out["vendor_entity_id"].map(
                lambda v: merged.get(v, v)
            )
            touched = set(merged) | set(merged.values())
            in_fuzzy = out["vendor_entity_id"].isin(touched)
            # A fuzzy group is a hypothesis, so it is demoted and marked -
            # never presented at the same confidence as an exact match.
            out.loc[in_fuzzy, "vendor_confidence"] = "MEDIUM"
            out.loc[in_fuzzy, "vendor_consistency"] = "AMBIGUOUS"
            LOGGER.info(
                "Stage 10 vendors: %d fuzzy merge(s), all capped at MEDIUM",
                len(merged),
            )

    out["vendor_group_size"] = out.groupby("vendor_entity_id")[
        "vendor_entity_id"
    ].transform("size")

    resolved = int(normalised.notna().sum())
    LOGGER.info(
        "Stage 10 vendors: %d record(s) -> %d entity(ies); %d unresolvable "
        "(%.1f%%), each kept separate",
        len(out),
        out["vendor_entity_id"].nunique(),
        len(out) - resolved,
        100 * (1 - resolved / max(1, len(out))),
    )
    return out


def _fuzzy_merge(keys: Dict[str, str]) -> Dict[str, str]:
    """Map entity ids onto a canonical id for near-identical keys.

    Compared within a first-character block: two firm names that differ in
    their first character are not spelling variants of each other, and the
    block turns an all-pairs sweep over every distinct vendor into a set of
    small ones.
    """
    canonical: Dict[str, str] = {}
    blocks: Dict[str, List[str]] = {}
    for key in sorted(keys):
        blocks.setdefault(key[0], []).append(key)

    for block in blocks.values():
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                if vendor_similarity(block[i], block[j]) >= VENDOR_FUZZY_SIMILARITY:
                    target, source = keys[block[i]], keys[block[j]]
                    canonical[source] = canonical.get(target, target)
    return canonical


def vendor_summary(out: pd.DataFrame) -> Dict[str, Any]:
    """Counts for the report, including how much could not be resolved."""
    if out.empty:
        return {"n_records": 0, "n_vendors": 0}
    unresolved = out["vendor_normalised"].isna().sum()
    sizes = out.groupby("vendor_entity_id").size()
    return {
        "n_records": int(len(out)),
        "n_vendors": int(out["vendor_entity_id"].nunique()),
        "unresolvable_records": int(unresolved),
        "unresolvable_share": float(unresolved / len(out)),
        "largest_vendor_group": int(sizes.max()),
        "by_confidence": {
            str(k): int(v) for k, v in out["vendor_confidence"].value_counts().items()
        },
        "_note": (
            "Unresolvable vendors each hold their own entity. They are not "
            "one firm, and a concentration statistic that treated them as one "
            "would report the largest supplier in the corpus as a fiction."
        ),
    }
