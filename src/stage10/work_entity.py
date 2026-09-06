"""Stage 10 - work entity resolution.

What this fixes
---------------
PARAKH has never been able to guarantee that a "group" corresponds to a real
work. Two measured defects:

* **R5** - 100 ``work_id`` values are reused across 200 rows of the corpus.
  Keying on the id merges works in different districts.
* **EXP-009** - a reconstructed join key produced groups of median size ONE.
  Every within-group statistic was then computed on a single record, a model
  scored 0.52 on it, and nothing in the output indicated a problem. A
  degenerate group returns a number, not an error.

This module groups records into candidate entities deterministically and
reports how much it actually knows. It contains no model, no learned
parameter, and no probability.

The governing rule
------------------
**Better unknown than wrong merge.** A weak match leaves records apart. The
layer may improve grouping; it may not invent truth. Original columns are
never modified - the entity structure is attached alongside them.

Blocking, not similarity, decides what may merge
------------------------------------------------
``district``, ``scheme`` and ``fiscal_year`` are hard partitions. Two
identically named works in different districts are different works and no
amount of name similarity may override that. Only inside a block are names
and costs compared at all, which also makes the pass cheap: one groupby
instead of an all-pairs sweep.

``scheme`` is absent from the current corpus. It is treated as optional, and
when it is missing a core field cannot be verified - so HIGH confidence
becomes unreachable and groups cap at MEDIUM. That is deliberate. The
alternative is to drop the requirement quietly and keep claiming HIGH.

Transitivity is the trap
------------------------
Union-find over pairwise similarity is transitive: A~B and B~C merges A with
C even when A and C share nothing. Unchecked, a chain of 10% steps walks a
group across an arbitrary cost range while every pair that built it was
within tolerance. So a group is **re-validated after assembly**, against the
whole group rather than the pairs - and if the assembled group violates the
criteria for its tier, it is demoted or marked CONFLICTING. Building a group
correctly and describing it correctly are two different jobs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ENTITY_FIELD_ALIASES,
    FISCAL_YEAR_START_MONTH,
    WORK_COST_CONFLICT_TOLERANCE,
    WORK_COST_TOLERANCE,
    WORK_NAME_STRONG_SIMILARITY,
    WORK_NAME_WEAK_SIMILARITY,
    WORK_TIMELINE_CONFLICT_DAYS,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)


class EntityResolutionError(ValueError):
    """An entity invariant was violated.

    Raised rather than warned: a violated invariant means the grouping does
    not mean what the columns say it means, and every downstream statistic
    computed on it would be wrong in a way nothing else detects.
    """


#: Columns attached to every record. Additive - nothing is overwritten.
ENTITY_COLUMNS: Tuple[str, ...] = (
    "work_entity_id",
    "entity_confidence",
    "entity_consistency",
    "group_size",
    "group_evidence",
    "degenerate_group",
    "work_id_reused",
    "fiscal_year_derived",
)

#: Fields whose agreement earns confidence, and whose disagreement is a
#: conflict. Named once so the two halves cannot drift apart.
CORE_FIELDS: Tuple[str, ...] = ("district", "scheme", "estimated_cost", "work_name")

_PUNCT = re.compile(r"[^a-z0-9ऀ-ॿ ]+")
_SPACE = re.compile(r"\s+")

#: Boilerplate carrying no distinguishing information. "Construction of X" and
#: "X" are the same work; leaving these in inflates every similarity score
#: toward 1.0 and would merge unrelated works that happen to share a verb.
_WORK_STOPWORDS = frozenset({
    "construction", "constrn", "const", "of", "at", "in", "the", "a", "for",
    "work", "works", "kary", "nirman", "repair", "repairs", "maintenance",
    "supply", "providing", "provision", "and", "to", "phase", "no", "nos",
})


def _resolve_column(frame: pd.DataFrame, logical: str) -> Optional[str]:
    """The actual column backing a logical field name, or None if absent."""
    for candidate in ENTITY_FIELD_ALIASES.get(logical, (logical,)):
        if candidate in frame.columns:
            return candidate
    return None


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    if text in {"nan", "none", "null", "<na>"}:
        return ""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


def derive_fiscal_year(frame: pd.DataFrame) -> pd.Series:
    """Indian financial year from the approval date.

    April-March, so an approval in March 2021 is FY2020-21 and one in April
    2021 is FY2021-22. Returned as a new Series - the source date column is
    not touched - and recorded as derived so nobody mistakes it for supplied
    data.
    """
    for column in ("date_approval", "date_proposal", "date_completion"):
        if column in frame.columns:
            stamps = pd.to_datetime(frame[column], errors="coerce")
            if stamps.notna().any():
                start = stamps.dt.year.where(
                    stamps.dt.month >= FISCAL_YEAR_START_MONTH,
                    stamps.dt.year - 1,
                )
                return start.map(
                    lambda y: "UNKNOWN" if pd.isna(y) else f"{int(y)}-{int(y) + 1 % 100:02d}"
                )
    return pd.Series(["UNKNOWN"] * len(frame), index=frame.index)


def name_similarity(left: str, right: str) -> float:
    """Deterministic similarity of two work names, in [0, 1].

    Two views combined by taking the max, because they fail on different
    inputs: token overlap handles reordering and extra words but is blind to
    spelling drift, while a sequence ratio handles "constrn"/"construction"
    but is thrown by word order. Neither is learned; both are stdlib.
    """
    left_clean, right_clean = _clean(left), _clean(right)
    if not left_clean or not right_clean:
        return 0.0
    if left_clean == right_clean:
        return 1.0

    left_tokens = {t for t in left_clean.split() if t not in _WORK_STOPWORDS}
    right_tokens = {t for t in right_clean.split() if t not in _WORK_STOPWORDS}
    if left_tokens and right_tokens:
        jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    else:
        jaccard = 0.0

    ratio = SequenceMatcher(None, left_clean, right_clean).ratio()
    return float(max(jaccard, ratio))


def _cost_within(a: float, b: float, tolerance: float) -> bool:
    """True when two costs are within ``tolerance`` of the smaller one."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    low, high = (a, b) if a <= b else (b, a)
    if low <= 0:
        return high <= 0
    return (high - low) / low <= tolerance


class _Union:
    """Union-find with deterministic representatives.

    The representative is the smallest member index, not whichever root the
    union happened to produce, so entity ids do not depend on row order.
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Always attach the larger root to the smaller so the eventual
            # representative is the minimum index in the component.
            if ra < rb:
                self._parent[rb] = ra
            else:
                self._parent[ra] = rb


def _block_keys(frame: pd.DataFrame, fiscal: pd.Series) -> pd.Series:
    """The hard partition. Records in different blocks are never compared."""
    district = _resolve_column(frame, "district")
    scheme = _resolve_column(frame, "scheme")

    parts = [
        frame[district].map(_clean) if district else pd.Series([""] * len(frame), index=frame.index),
        frame[scheme].map(_clean) if scheme else pd.Series(["*"] * len(frame), index=frame.index),
        fiscal.astype(str),
    ]
    return parts[0] + "|" + parts[1] + "|" + parts[2]


def _group_evidence(
    group: pd.DataFrame, cost_column: Optional[str], has_scheme: bool
) -> Tuple[List[str], List[str]]:
    """Which core fields agree across an assembled group, and which do not.

    Computed on the whole group rather than on the pairs that built it - see
    the module docstring on transitivity.
    """
    matched: List[str] = []
    conflicting: List[str] = []

    for logical, column in (("district", _resolve_column(group, "district")),
                            ("scheme", _resolve_column(group, "scheme"))):
        if column is None:
            continue
        if group[column].map(_clean).nunique() <= 1:
            matched.append(logical)
        else:
            conflicting.append(logical)

    agency = _resolve_column(group, "agency")
    if agency is not None:
        if group[agency].map(_clean).nunique() <= 1:
            matched.append("implementing_agency")
        else:
            conflicting.append("implementing_agency")

    if cost_column is not None:
        costs = pd.to_numeric(group[cost_column], errors="coerce").dropna()
        if len(costs) >= 2 and costs.min() > 0:
            spread = (costs.max() - costs.min()) / costs.min()
            if spread <= WORK_COST_TOLERANCE:
                matched.append("estimated_cost")
            elif spread > WORK_COST_CONFLICT_TOLERANCE:
                conflicting.append("estimated_cost")
        elif len(costs) >= 1:
            matched.append("estimated_cost")

    if "work_name" in group.columns and len(group) >= 2:
        names = group["work_name"].tolist()
        worst = min(
            name_similarity(names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        )
        if worst >= WORK_NAME_STRONG_SIMILARITY:
            matched.append("work_name")
        elif worst < WORK_NAME_WEAK_SIMILARITY:
            conflicting.append("work_name")
    elif "work_name" in group.columns:
        matched.append("work_name")

    if not has_scheme:
        # Recorded as a gap rather than silently omitted: the group cannot
        # claim every core field matched when one could not be checked.
        conflicting.append("scheme:UNVERIFIABLE")

    return matched, conflicting


def _timeline_conflict(group: pd.DataFrame) -> bool:
    """True when members' approval dates are implausibly far apart."""
    if "date_approval" not in group.columns or len(group) < 2:
        return False
    stamps = pd.to_datetime(group["date_approval"], errors="coerce").dropna()
    if len(stamps) < 2:
        return False
    return (stamps.max() - stamps.min()).days > WORK_TIMELINE_CONFLICT_DAYS


def resolve_work_entities(frame: pd.DataFrame) -> pd.DataFrame:
    """Group records into candidate work entities, preserving ambiguity.

    Args:
        frame: Corpus records. Read only - every original column is returned
            unchanged and the index is preserved.

    Returns:
        A copy of ``frame`` with :data:`ENTITY_COLUMNS` attached, plus
        ``attrs`` carrying ``n_entities``, ``degenerate_share`` and
        ``conflicting_share`` so a caller can see the shape of the result
        without recomputing it. A high degenerate share is the EXP-009
        condition and is surfaced rather than buried.

    Raises:
        EntityResolutionError: If a required column is missing, or if an
            invariant fails after assembly.
    """
    if "district" not in frame.columns:
        raise EntityResolutionError(
            "work entity resolution requires a 'district' column; it is a "
            "blocking key, not an optional similarity input"
        )
    if "work_name" not in frame.columns:
        raise EntityResolutionError(
            "work entity resolution requires a 'work_name' column"
        )

    out = frame.copy()
    if out.empty:
        for column in ENTITY_COLUMNS:
            out[column] = pd.Series(dtype=object)
        out.attrs.update(n_entities=0, degenerate_share=0.0, conflicting_share=0.0)
        return out

    cost_column = _resolve_column(out, "estimated_cost")
    has_scheme = _resolve_column(out, "scheme") is not None

    fiscal = derive_fiscal_year(out)
    blocks = _block_keys(out, fiscal)
    positions = {label: i for i, label in enumerate(out.index)}
    union = _Union(len(out))

    names = out["work_name"].tolist()
    costs = (
        pd.to_numeric(out[cost_column], errors="coerce").to_numpy()
        if cost_column
        else np.full(len(out), np.nan)
    )

    # Pairwise comparison INSIDE blocks only. Blocks are small, so this is
    # cheap; across the corpus it would be quadratic and pointless, since a
    # cross-block pair can never merge however similar its names are.
    for _, index in out.groupby(blocks, sort=True).groups.items():
        members = [positions[label] for label in index]
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                similarity = name_similarity(names[a], names[b])
                if similarity < WORK_NAME_WEAK_SIMILARITY:
                    continue
                if cost_column is not None and not _cost_within(
                    costs[a], costs[b], WORK_COST_TOLERANCE
                ):
                    continue
                if similarity >= WORK_NAME_WEAK_SIMILARITY:
                    union.union(a, b)

    roots = [union.find(i) for i in range(len(out))]
    entity_ids = [f"WE-{root:07d}" for root in roots]
    out["work_entity_id"] = entity_ids
    out["fiscal_year_derived"] = fiscal.to_numpy()

    # --- describe each assembled group ------------------------------------
    confidence: Dict[str, str] = {}
    consistency: Dict[str, str] = {}
    evidence: Dict[str, Dict[str, List[str]]] = {}
    sizes: Dict[str, int] = {}

    for entity_id, group in out.groupby("work_entity_id", sort=True):
        size = len(group)
        sizes[entity_id] = size
        matched, conflicting = _group_evidence(group, cost_column, has_scheme)
        if _timeline_conflict(group):
            conflicting.append("timeline")

        hard_conflicts = [c for c in conflicting if not c.endswith(":UNVERIFIABLE")]

        if size == 1:
            # The EXP-009 fix. A group of one is a statement about ignorance,
            # not about agreement, and it must be impossible to overlook.
            confidence[entity_id] = "LOW"
            consistency[entity_id] = "AMBIGUOUS"
        elif hard_conflicts:
            confidence[entity_id] = "MEDIUM" if len(matched) >= 2 else "LOW"
            consistency[entity_id] = "CONFLICTING"
        elif set(CORE_FIELDS).issubset(set(matched)) and has_scheme:
            confidence[entity_id] = "HIGH"
            consistency[entity_id] = "CONSISTENT"
        elif len(matched) >= 3:
            confidence[entity_id] = "MEDIUM"
            consistency[entity_id] = "CONSISTENT"
        else:
            confidence[entity_id] = "LOW"
            consistency[entity_id] = "AMBIGUOUS"

        evidence[entity_id] = {
            "matched_fields": sorted(matched),
            "conflicting_fields": sorted(conflicting),
        }

    out["entity_confidence"] = out["work_entity_id"].map(confidence)
    out["entity_consistency"] = out["work_entity_id"].map(consistency)
    out["group_size"] = out["work_entity_id"].map(sizes).astype(int)
    out["group_evidence"] = out["work_entity_id"].map(evidence)
    out["degenerate_group"] = out["group_size"] == 1

    if "work_id" in out.columns:
        counts = out["work_id"].value_counts()
        out["work_id_reused"] = out["work_id"].map(counts).gt(1).fillna(False)
    else:
        out["work_id_reused"] = False

    _assert_invariants(out, frame, cost_column)

    n_entities = out["work_entity_id"].nunique()
    out.attrs.update(
        n_entities=int(n_entities),
        degenerate_share=float(out["degenerate_group"].mean()),
        conflicting_share=float((out["entity_consistency"] == "CONFLICTING").mean()),
        scheme_available=has_scheme,
    )
    LOGGER.info(
        "Stage 10 works: %d record(s) -> %d entity(ies); %.1f%% degenerate, "
        "%.1f%% conflicting%s",
        len(out),
        n_entities,
        100 * out.attrs["degenerate_share"],
        100 * out.attrs["conflicting_share"],
        "" if has_scheme else " (scheme absent - HIGH confidence unreachable)",
    )
    return out


def _assert_invariants(
    out: pd.DataFrame, original: pd.DataFrame, cost_column: Optional[str]
) -> None:
    """Every guarantee this module makes, checked rather than assumed."""
    if len(out) != len(original):
        raise EntityResolutionError(
            f"grouping changed the record count: {len(original)} -> {len(out)}"
        )
    if not out.index.equals(original.index):
        raise EntityResolutionError("grouping altered the caller's index")
    if out["work_entity_id"].isna().any():
        raise EntityResolutionError("a record received no work_entity_id")

    for column in original.columns:
        if not out[column].equals(original[column]):
            raise EntityResolutionError(
                f"original column {column!r} was modified; Stage 10 is additive"
            )

    degenerate = out["group_size"] == 1
    if not out.loc[degenerate, "degenerate_group"].all():
        raise EntityResolutionError("a singleton group was not marked degenerate")
    if out.loc[degenerate, "entity_confidence"].ne("LOW").any():
        raise EntityResolutionError("a degenerate group claims above LOW confidence")

    # The transitivity guard: a HIGH group must survive inspection of the
    # whole group, not merely of the pairs that assembled it.
    high = out[out["entity_confidence"] == "HIGH"]
    for entity_id, group in high.groupby("work_entity_id"):
        if group["group_evidence"].iloc[0]["conflicting_fields"]:
            raise EntityResolutionError(
                f"HIGH-confidence entity {entity_id} carries conflicting fields"
            )
        if cost_column is not None:
            values = pd.to_numeric(group[cost_column], errors="coerce").dropna()
            if len(values) >= 2 and values.min() > 0:
                spread = (values.max() - values.min()) / values.min()
                if spread > WORK_COST_TOLERANCE:
                    raise EntityResolutionError(
                        f"HIGH-confidence entity {entity_id} spans {spread:.1%} "
                        f"in cost, above the {WORK_COST_TOLERANCE:.0%} tolerance"
                    )


def entity_summary(out: pd.DataFrame) -> Dict[str, Any]:
    """Counts a reviewer needs, including the ones that look bad."""
    if out.empty:
        return {"n_records": 0, "n_entities": 0}
    sizes = out.groupby("work_entity_id").size()
    return {
        "n_records": int(len(out)),
        "n_entities": int(out["work_entity_id"].nunique()),
        "degenerate_entities": int((sizes == 1).sum()),
        "degenerate_share": float(out["degenerate_group"].mean()),
        "largest_group": int(sizes.max()),
        "by_confidence": {
            str(k): int(v) for k, v in out["entity_confidence"].value_counts().items()
        },
        "by_consistency": {
            str(k): int(v) for k, v in out["entity_consistency"].value_counts().items()
        },
        "work_id_reused_rows": int(out["work_id_reused"].sum()),
        "_note": (
            "A high degenerate share is the EXP-009 condition: statistics "
            "computed over groups of one look identical to statistics over "
            "real groups. Read this before trusting any group-level metric."
        ),
    }
