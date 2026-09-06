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


class EntityUsageError(RuntimeError):
    """A group-level computation was attempted over an unusable group.

    Distinct from :class:`EntityResolutionError`, which means the resolver
    itself produced something invalid. This means the resolver worked and a
    *caller* tried to aggregate over groups the resolver has declared it does
    not stand behind.

    Raised rather than warned because that is the whole point. EXP-009 was
    survivable precisely because a degenerate group returned a number instead
    of failing: the number looked exactly like every other number, a model
    scored 0.52 on it, and nothing surfaced. Marking such groups was not
    enough - a mark can be ignored by any code that does not check for it. An
    exception cannot.
    """


#: Columns attached to every record. Additive - nothing is overwritten.
ENTITY_COLUMNS: Tuple[str, ...] = (
    "work_entity_id",
    "entity_confidence",
    "entity_consistency",
    "group_size",
    "group_evidence",
    "degenerate_group",
    "group_usable",
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

    # A group may be aggregated over only when all three disqualifiers are
    # absent. The conditions overlap - every singleton is already LOW - and
    # they are kept separate anyway, so that a later change to how confidence
    # is assigned cannot silently re-admit singletons through the back door.
    out["group_usable"] = (
        (~out["degenerate_group"])
        & out["entity_confidence"].ne("LOW")
        & out["entity_consistency"].ne("CONFLICTING")
    )

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

    # The usability invariant, checked rather than trusted: nothing marked
    # usable may carry any of the three disqualifiers.
    usable = out["group_usable"].fillna(False).astype(bool)
    if out.loc[usable, "degenerate_group"].any():
        raise EntityResolutionError("a degenerate group was marked usable")
    if out.loc[usable, "entity_confidence"].eq("LOW").any():
        raise EntityResolutionError("a LOW-confidence group was marked usable")
    if out.loc[usable, "entity_consistency"].eq("CONFLICTING").any():
        raise EntityResolutionError("a CONFLICTING group was marked usable")

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


def require_usable_groups(
    frame: pd.DataFrame, *, on_unusable: str = "raise"
) -> pd.DataFrame:
    """Gate any group-level computation on entity usability.

    Call this immediately before aggregating over ``work_entity_id``. It is
    the mechanism that turns EXP-009 from a silent wrong answer into a loud
    failure: a degenerate group can no longer contribute a number to a mean,
    a correlation or a screen without someone having decided to let it.

    Args:
        frame: Stage 10 output carrying ``group_usable``.
        on_unusable: ``"raise"`` (default) to refuse the whole computation, or
            ``"filter"`` to return only the defensible rows. Those are the two
            honest options; aggregating over everything and hoping is not one
            of them, which is why there is no third mode.

    Returns:
        The usable rows. With ``on_unusable="raise"`` that is the whole frame,
        because anything else has already raised.

    Raises:
        EntityUsageError: If unusable groups are present and ``on_unusable``
            is ``"raise"``, or if filtering leaves nothing behind.
        KeyError: If the frame did not come from Stage 10.
    """
    if "group_usable" not in frame.columns:
        raise KeyError(
            "require_usable_groups needs a 'group_usable' column; pass the "
            "output of resolve_work_entities, not the raw corpus"
        )
    if on_unusable not in {"raise", "filter"}:
        raise ValueError(
            f"on_unusable must be 'raise' or 'filter', got {on_unusable!r}"
        )

    usable = frame["group_usable"].fillna(False).astype(bool)
    if usable.all():
        return frame

    bad = frame.loc[~usable]
    reasons = []
    if "degenerate_group" in bad.columns and bad["degenerate_group"].any():
        reasons.append(f"{int(bad['degenerate_group'].sum())} degenerate")
    if "entity_confidence" in bad.columns:
        n_low = int(bad["entity_confidence"].eq("LOW").sum())
        if n_low:
            reasons.append(f"{n_low} LOW confidence")
    if "entity_consistency" in bad.columns:
        n_conf = int(bad["entity_consistency"].eq("CONFLICTING").sum())
        if n_conf:
            reasons.append(f"{n_conf} CONFLICTING")

    sample = sorted(bad["work_entity_id"].astype(str).unique())[:5]
    detail = (
        f"{int((~usable).sum())} of {len(frame)} record(s) sit in unusable "
        f"groups ({'; '.join(reasons)}). Examples: {', '.join(sample)}"
    )

    if on_unusable == "raise":
        raise EntityUsageError(
            detail
            + ". Aggregating over these would repeat EXP-009, where a group "
            "of one produced a number indistinguishable from a real one. "
            "Pass on_unusable='filter' to aggregate over the defensible "
            "subset instead, and report how much was dropped."
        )

    kept = frame.loc[usable]
    if kept.empty:
        raise EntityUsageError(
            "no usable groups remain after filtering; there is nothing to "
            f"aggregate over. {detail}"
        )
    LOGGER.warning(
        "Stage 10 guard: dropped %d of %d record(s) as unusable (%s)",
        int((~usable).sum()),
        len(frame),
        "; ".join(reasons),
    )
    return kept


def entity_status(frame: pd.DataFrame) -> Dict[str, Any]:
    """Corpus-level entity health, including the ceilings.

    ``HIGH_CONFIDENCE_UNREACHABLE`` is emitted when no group anywhere reached
    HIGH. That is a systemic fact - usually a missing core field - rather than
    a property of any record, and stating it as a status means nobody has to
    notice the absence of a value in a distribution.
    """
    if frame.empty:
        return {"status": "EMPTY", "n_records": 0, "n_high": 0,
                "usable_share": 0.0}

    n_high = int(frame["entity_confidence"].eq("HIGH").sum())
    usable_share = float(frame["group_usable"].fillna(False).mean())
    scheme_available = bool(frame.attrs.get("scheme_available", True))

    if n_high == 0:
        status = "HIGH_CONFIDENCE_UNREACHABLE"
        reason = (
            "No group reached HIGH confidence. The 'scheme' column is absent, "
            "so a core field cannot be verified and HIGH is unreachable by "
            "construction - not a property of these records."
            if not scheme_available
            else "No group reached HIGH confidence, though every core field "
            "was available to check."
        )
    elif usable_share == 0.0:
        status = "NO_USABLE_GROUPS"
        reason = "Groups exist but none is defensible for aggregation."
    else:
        status = "OK"
        reason = f"{usable_share:.1%} of records sit in usable groups."

    return {
        "status": status,
        "reason": reason,
        "n_records": int(len(frame)),
        "n_entities": int(frame["work_entity_id"].nunique()),
        "n_high": n_high,
        "n_usable_records": int(frame["group_usable"].fillna(False).sum()),
        "usable_share": usable_share,
        "degenerate_share": float(frame["degenerate_group"].mean()),
        "conflicting_share": float(
            frame["entity_consistency"].eq("CONFLICTING").mean()
        ),
        "scheme_available": scheme_available,
    }
