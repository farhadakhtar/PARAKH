"""Duplicate-detection evaluation harness.

The problem
-----------
Stage 3's duplicate detector could not be measured. Stage 1's duplicate channel
clones a work name from *any* row, so only 70 of 1,000 injected clones land in
the same district - and ``Stage3.md`` sec.9.1's ``1[d_i = d_j]`` deliberately
excludes the rest. Precision and recall computed against that channel are
meaningless, and the earlier figures (0.047 / 0.119) measured the mismatch
between the two definitions rather than anything about the detector.

The fix
-------
Inject duplicates that match the detector's own definition: **same district,
close in time, near-identical text**, carrying a known ``duplicate_id``.

Why this lives here and not in Stage 1
--------------------------------------
Stage 1 is locked, and injecting after generation is both safer and stricter.
``duplicate_id`` is returned as a **separate object**, never as a frame column,
so it is structurally impossible for it to reach the pipeline - a guarantee a
hidden column could only offer by convention. A test asserts the frame handed
to Stage 3 carries no such column.

The correction (AUDIT M2)
-------------------------
The first version of this harness perturbed only the **action verb**. Action
verbs are stopwords in ``normalize_work_text``, so the perturbation was
erased before the detector ever saw it. Verified afterwards: **60 of 60**
injected pairs were byte-identical in the detector's own text view.

    The previously reported F1 of 0.929 was INVALID. It measured
    exact-match retrieval, not near-duplicate detection, and is withdrawn.

Perturbations now act on tokens that **survive** preprocessing - a typo
inside a content word, a synonym swap, or a dropped token - so the injected
duplicate is genuinely a near match. ``assert_perturbations_are_real``
checks this property directly rather than trusting it, and the test suite
asserts cosine similarity is strictly below 1.0.

The detector itself is untouched.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    EVAL_AMOUNT_JITTER,
    EVAL_DUPLICATE_ACTIONS,
    EVAL_DUPLICATE_MAX_DAY_GAP,
    EVAL_PERTURBATIONS,
    EVAL_TOKEN_SWAPS,
    FIELD_ORDER,
    STAGE3_VERSION,
)
from src.core.logger import get_logger
from src.stage3.embedding import build_stopwords

LOGGER = get_logger(__name__)

DUPLICATE_ID_COLUMN = "duplicate_id"


@dataclass(frozen=True)
class DuplicateTruth:
    """Ground truth for injected duplicates, held outside the corpus.

    Attributes:
        duplicate_id: Positional row index -> injected group id. Rows absent
            from this mapping belong to no injected group.
        injected_rows: Positions of the rows that were added.
        source_rows: Positions of the originals they were cloned from.
    """

    duplicate_id: Mapping[int, int]
    injected_rows: Tuple[int, ...]
    source_rows: Tuple[int, ...]
    n_pairs: int
    max_day_gap: int = EVAL_DUPLICATE_MAX_DAY_GAP
    #: Perturbation applied to each pair, aligned to ``source_rows``.
    #: Recorded so recall can be broken down by difficulty rather than
    #: reported as a single number that explains nothing.
    perturbations: Tuple[str, ...] = ()

    @property
    def true_pairs(self) -> Set[frozenset]:
        """Every unordered pair of rows sharing an injected group."""
        groups: Dict[int, List[int]] = {}
        for row, group in self.duplicate_id.items():
            groups.setdefault(int(group), []).append(int(row))
        pairs: Set[frozenset] = set()
        for members in groups.values():
            for left, right in itertools.combinations(sorted(members), 2):
                pairs.add(frozenset((left, right)))
        return pairs

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "n_pairs": self.n_pairs,
            "n_labelled_rows": len(self.duplicate_id),
            "n_true_pairs": len(self.true_pairs),
            "max_day_gap": self.max_day_gap,
            "perturbation_counts": {
                kind: int(self.perturbations.count(kind))
                for kind in sorted(set(self.perturbations))
            },
        }



def _content_tokens(name: str, stopwords: frozenset) -> List[Tuple[int, str]]:
    """Positions and values of tokens that survive preprocessing.

    A perturbation is only real if it lands on a token the detector can still
    see. Stopwords, digits and short fragments are excluded: changing a
    stopword is invisible, and changing the ward number would make the record a
    different work rather than a duplicate of the same one.

    Args:
        name: The raw work name.
        stopwords: The stopword set the detector applies.

    Returns:
        ``(index, token)`` pairs into ``name.split()``.
    """
    tokens = name.split()
    return [
        (position, token)
        for position, token in enumerate(tokens)
        if token.isalpha() and len(token) >= 4 and token.lower() not in stopwords
    ]


def perturb_work_name(
    name: str, kind: str, stopwords: frozenset, position_seed: int
) -> str:
    """Apply one preprocessing-surviving perturbation to a work name.

    Args:
        name: The source work name.
        kind: ``"typo"``, ``"swap"`` or ``"truncate"``.
        stopwords: The stopword set the detector applies.
        position_seed: Deterministically selects which content token is hit.

    Returns:
        The perturbed name, or the original when no content token exists to
        perturb - in which case the pair is dropped rather than counted as a
        detected duplicate it never was.
    """
    tokens = name.split()
    candidates = _content_tokens(name, stopwords)
    if not candidates:
        return name

    index, token = candidates[position_seed % len(candidates)]

    if kind == "swap":
        lowered = token.lower()
        for left, right in EVAL_TOKEN_SWAPS:
            if lowered == left:
                tokens[index] = right
                return " ".join(tokens)
            if lowered == right:
                tokens[index] = left
                return " ".join(tokens)
        kind = "typo"  # no synonym for this token; fall through

    if kind == "truncate":
        # Drop the token entirely - a shortened description of the same work.
        del tokens[index]
        return " ".join(tokens)

    # "typo": transpose two interior characters. Survives preprocessing because
    # the result is still an alphabetic non-stopword, and it is exactly the
    # data-entry error a real register carries.
    middle = max(1, len(token) // 2)
    if middle + 1 >= len(token):
        return " ".join(tokens)
    mangled = (
        token[:middle] + token[middle + 1] + token[middle] + token[middle + 2 :]
    )
    tokens[index] = mangled
    return " ".join(tokens)


def assert_perturbations_are_real(
    frame: pd.DataFrame,
    truth: "DuplicateTruth",
    stopwords: frozenset,
) -> Dict[str, Any]:
    """Verify each injected pair actually differs after preprocessing.

    The failure this guards against is the one that invalidated the first
    harness: a perturbation the detector cannot see makes the evaluation a test
    of exact matching wearing a near-duplicate label.

    Args:
        frame: The augmented frame.
        truth: The injected ground truth.
        stopwords: The stopword set the detector applies.

    Returns:
        Counts of identical and differing pairs, post-preprocessing.
    """
    from src.stage3.embedding import normalize_work_text

    view = normalize_work_text(
        frame["work_name"],
        stopwords,
        truncate_at_locality_clause=False,
        keep_digits=True,
    )
    identical = 0
    for source, injected in zip(truth.source_rows, truth.injected_rows):
        if view.iloc[source] == view.iloc[injected]:
            identical += 1
    return {
        "pairs": len(truth.source_rows),
        "identical_after_preprocessing": identical,
        "distinct_after_preprocessing": len(truth.source_rows) - identical,
        "trivial": identical > 0,
    }


def inject_duplicate_pairs(
    frame: pd.DataFrame,
    n_pairs: int = 200,
    seed: int = 7,
    max_day_gap: int = EVAL_DUPLICATE_MAX_DAY_GAP,
    actions: Sequence[str] = EVAL_DUPLICATE_ACTIONS,
) -> Tuple[pd.DataFrame, DuplicateTruth]:
    """Append labelled near-duplicates that satisfy the detector's definition.

    Each injected record copies a source row, keeps its **district**, shifts the
    proposal date by at most ``max_day_gap`` days, and swaps the leading action
    verb so the pair is a near duplicate rather than a copy.

    Args:
        frame: A raw generated frame (pre-``Corpus``), object dtype.
        n_pairs: How many duplicates to inject.
        seed: Seed for the single RNG; nothing else here is random.
        max_day_gap: Largest date shift, in days.
        actions: Action verbs to swap in.

    Returns:
        ``(augmented_frame, truth)``. The frame carries **no** ``duplicate_id``
        column - the labels live only in ``truth``.

    Raises:
        ValueError: If ``n_pairs`` is negative or the frame lacks the columns
            needed to build a valid duplicate.
    """
    if n_pairs < 0:
        raise ValueError(f"n_pairs must be non-negative, got {n_pairs}")
    required = ("work_name", "district", "date_proposal")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Cannot inject duplicates; frame lacks {missing!r}")

    working = frame.reset_index(drop=True).copy()
    n_source = len(working)
    if n_source == 0 or n_pairs == 0:
        return working, DuplicateTruth({}, (), (), 0, max_day_gap)

    rng = np.random.default_rng(seed)
    stopwords = build_stopwords(working)

    # Only rows with a usable name, district and date can seed a valid pair:
    # the detector requires all three, so a source lacking any of them would
    # produce a duplicate it is not designed to find.
    # A raw generated frame holds dates as STRINGS, including the deliberate
    # garbage Stage 1 injects, so notna() is not enough - 'pending' passes it.
    # Parse first and require a real date, otherwise the source cannot anchor
    # a temporally-close duplicate at all.
    parsed_dates = pd.to_datetime(working["date_proposal"], errors="coerce")
    usable = working.index[
        working["work_name"].notna()
        & working["district"].notna()
        & parsed_dates.notna()
        & working["work_name"].astype(str).str.contains(" at ", regex=False)
    ].to_numpy()
    if usable.size == 0:
        LOGGER.warning("No record carries name, district and date; nothing injected.")
        return working, DuplicateTruth({}, (), (), 0, max_day_gap)

    take = min(int(n_pairs), int(usable.size))
    sources = np.sort(rng.choice(usable, size=take, replace=False))
    shifts = rng.integers(1, max_day_gap + 1, size=take)
    action_choice = rng.integers(0, len(actions), size=take)
    kind_choice = rng.integers(0, len(EVAL_PERTURBATIONS), size=take)
    token_choice = rng.integers(0, 997, size=take)
    jitter = rng.uniform(EVAL_AMOUNT_JITTER[0], EVAL_AMOUNT_JITTER[1], size=take)
    jitter_sign = rng.choice(np.asarray([-1.0, 1.0]), size=take)

    duplicate_id: Dict[int, int] = {}
    injected_rows: List[int] = []
    used_kinds: List[str] = []
    # Tracked alongside the rows actually created. Slicing the source array
    # afterwards would misalign the two lists the moment one source is
    # skipped, silently pairing each duplicate with the wrong original.
    used_sources: List[int] = []
    new_rows: List[Dict[str, Any]] = []

    for position, source in enumerate(sources):
        original = working.loc[source]
        name = str(original["work_name"])

        # 1. Swap the leading action verb. Realistic, but INVISIBLE to the
        #    detector because action verbs are stopwords - which is precisely
        #    what invalidated the first harness. Kept for realism only; it is
        #    never relied on to make the pair a near match.
        tail = name.split(" of ", 1)[1] if " of " in name else name
        new_name = f"{actions[int(action_choice[position])]} {tail}"

        # 2. Perturb a token that SURVIVES preprocessing. This is the change
        #    that makes the evaluation non-trivial.
        kind = EVAL_PERTURBATIONS[int(kind_choice[position])]
        perturbed = perturb_work_name(
            new_name, kind, stopwords, int(token_choice[position])
        )
        if perturbed == new_name:
            # No surviving token to perturb, so this pair would be an exact
            # match and would inflate the score. Drop it rather than count it.
            continue
        new_name = perturbed

        base_date = parsed_dates.loc[source]
        if pd.isna(base_date):
            continue
        new_date = base_date + pd.Timedelta(days=int(shifts[position]))

        row = {name_: original.get(name_) for name_ in FIELD_ORDER}
        row["work_id"] = f"EVAL-DUP-{position:06d}"
        row["work_name"] = new_name
        row["date_proposal"] = new_date.date().isoformat()
        row["date_approval"] = None
        row["date_completion"] = None

        # 3. Amount jitter: a re-estimate of the same work, not a copy of the
        #    figure. Does not affect detection (which is text + district + time)
        #    but keeps the injected record realistic for anything downstream.
        for money_field in ("sanction_amount", "amount_spent"):
            value = original.get(money_field)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                row[money_field] = round(
                    numeric
                    * (1.0 + float(jitter_sign[position]) * float(jitter[position])),
                    2,
                )
        new_rows.append(row)

        new_position = n_source + len(new_rows) - 1
        injected_rows.append(new_position)
        used_sources.append(int(source))
        used_kinds.append(kind)
        duplicate_id[int(source)] = position
        duplicate_id[int(new_position)] = position

    if not new_rows:
        return working, DuplicateTruth({}, (), (), 0, max_day_gap)

    augmented = pd.concat(
        [working, pd.DataFrame(new_rows, columns=list(FIELD_ORDER)).astype("object")],
        ignore_index=True,
    )
    assert DUPLICATE_ID_COLUMN not in augmented.columns, (
        "duplicate_id must never become a frame column"
    )

    LOGGER.info(
        "Injected %d labelled duplicate pair(s): same district, a "
        "preprocessing-surviving token perturbation, date shifted by 1-%d days.",
        len(new_rows),
        max_day_gap,
    )
    return augmented, DuplicateTruth(
        duplicate_id=duplicate_id,
        injected_rows=tuple(injected_rows),
        source_rows=tuple(used_sources),
        perturbations=tuple(used_kinds),
        n_pairs=len(new_rows),
        max_day_gap=max_day_gap,
    )


def predicted_pairs(duplicate_group_id: pd.Series) -> Set[frozenset]:
    """Every unordered pair the detector placed in the same group.

    Args:
        duplicate_group_id: Per-record group, ``-1`` where ungrouped.

    Returns:
        Set of frozensets of positional row indices.
    """
    pairs: Set[frozenset] = set()
    positions = {label: index for index, label in enumerate(duplicate_group_id.index)}
    grouped = duplicate_group_id[duplicate_group_id >= 0]
    for _, members in grouped.groupby(grouped):
        rows = sorted(positions[label] for label in members.index)
        for left, right in itertools.combinations(rows, 2):
            pairs.add(frozenset((left, right)))
    return pairs


def evaluate_duplicates(
    duplicate_group_id: pd.Series,
    truth: DuplicateTruth,
    duplicate_score: Optional[pd.Series] = None,
    pair_similarity: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Score the detector against injected ground truth, pairwise.

    Pairwise precision/recall is the right metric for a grouping task: it asks
    whether each pair of records was correctly placed together, which is what a
    reviewer opening two files actually cares about, and it does not reward a
    detector for splitting one true group into several.

    Args:
        duplicate_group_id: Stage 3's per-record group assignment.
        truth: Injected ground truth.
        duplicate_score: Optional per-record score, summarised for context.
        pair_similarity: Optional cosine per injected pair, aligned to
            ``truth.source_rows``. Supplying it turns a bare recall number
            into a diagnosis: whether the detector missed pairs it could
            see, or was never shown a pair above its threshold.
        threshold: The detector's similarity threshold, for that diagnosis.

    Returns:
        A JSON-serialisable report with precision, recall and F1.
    """
    predicted = predicted_pairs(duplicate_group_id)
    actual = truth.true_pairs

    true_positive = len(predicted & actual)
    false_positive = len(predicted - actual)
    false_negative = len(actual - predicted)

    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / (true_positive + false_negative) if actual else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    report: Dict[str, Any] = {
        "stage3_version": STAGE3_VERSION,
        "_note": (
            "Measured against duplicates injected by "
            "src.stage3.evaluation.inject_duplicate_pairs: same district, "
            "within a few days, and perturbed on a token that SURVIVES "
            "preprocessing. Stage 1's own duplicate channel clones names "
            "across districts and cannot validate this detector."
        ),
        "_withdrawn": (
            "A previously reported F1 of 0.929 is WITHDRAWN. That harness "
            "perturbed only the action verb, which is a stopword, so 60/60 "
            "injected pairs were byte-identical in the detector's own text "
            "view. It measured exact-match retrieval, not near-duplicate "
            "detection."
        ),
        "ground_truth": truth.to_dict(),
        "pairs": {
            "predicted": len(predicted),
            "actual": len(actual),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }

    # Per-perturbation recall. A single recall number cannot distinguish a
    # detector that is broken from one that is being handed pairs outside
    # its operating range; this can.
    if truth.perturbations:
        by_kind: Dict[str, Dict[str, Any]] = {}
        for kind in sorted(set(truth.perturbations)):
            rows = [
                index
                for index, value in enumerate(truth.perturbations)
                if value == kind
            ]
            pairs_of_kind = {
                frozenset((truth.source_rows[i], truth.injected_rows[i]))
                for i in rows
            }
            found = len(pairs_of_kind & predicted)
            by_kind[kind] = {
                "pairs": len(pairs_of_kind),
                "found": found,
                "recall": round(found / len(pairs_of_kind), 6)
                if pairs_of_kind
                else 0.0,
            }
        report["recall_by_perturbation"] = by_kind

    # Cosine diagnosis: how many injected pairs the detector could even
    # have found, given its threshold.
    if pair_similarity is not None and len(pair_similarity):
        similarity = np.asarray(list(pair_similarity), dtype="float64")
        entry: Dict[str, Any] = {
            "min": round(float(similarity.min()), 6),
            "median": round(float(np.median(similarity)), 6),
            "max": round(float(similarity.max()), 6),
            "n_exactly_one": int((similarity >= 1.0 - 1e-9).sum()),
        }
        if threshold is not None:
            reachable = int((similarity >= threshold).sum())
            entry["threshold"] = float(threshold)
            entry["pairs_at_or_above_threshold"] = reachable
            entry["pct_reachable"] = round(
                100.0 * reachable / similarity.size, 4
            )
            entry["_reading"] = (
                "Pairs below the threshold are unreachable BEFORE the "
                "temporal decay is applied, so they bound recall from above. "
                "Low recall against a low reachable share is a "
                "representation limit, not a defect in the grouping logic."
            )
        report["pair_similarity"] = entry
    return report

    if duplicate_score is not None and len(duplicate_score):
        labelled = sorted(truth.duplicate_id)
        scores = duplicate_score.to_numpy(dtype="float64")
        injected = scores[[row for row in labelled if row < len(scores)]]
        report["scores"] = {
            "injected_median": round(float(np.median(injected)), 6)
            if injected.size
            else None,
            "injected_min": round(float(injected.min()), 6) if injected.size else None,
            "corpus_median": round(float(np.median(scores)), 6),
            "corpus_p99": round(float(np.percentile(scores, 99)), 6),
        }

    LOGGER.info(
        "Duplicate evaluation: precision %.3f, recall %.3f, F1 %.3f over %d "
        "true pair(s).",
        precision,
        recall,
        f1,
        len(actual),
    )
