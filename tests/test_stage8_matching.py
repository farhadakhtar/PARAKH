"""Property tests for the Stage 8 record/finding matcher.

Written before the implementation. The matcher is the component that decides
whether a label is evidence or coincidence, and its failure mode is silent: a
wrong matcher does not crash, it returns a plausible-looking label set that
trains a model to predict audit coverage instead of irregularity. The previous
calibration build produced 5,237 "positives" on a postal-office directory and
every count in its report looked healthy.

So the tests here assert the properties a correct matcher has, not the output
it happened to produce on one dataset.

The load-bearing test is :class:`TestBlockingParity`. Blocking exists to avoid
comparing every record against every finding; the entire correctness claim of
blocking is that it changes the cost and *not* the answer. Anything else is a
different algorithm with a recall hole in it, and a recall hole in a labeller
is invisible downstream - the labels that were never generated cannot be
missed by a metric computed on the labels that were.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    CALIBRATION_ELIGIBLE_EVIDENCE,
    MATCH_METHOD_CONFIDENCE,
    MATCH_METHOD_EVIDENCE,
    MATCH_METHODS,
)
from src.stage8.matching import (
    MatchResult,
    blocking_keys,
    match_blocked,
    match_naive,
    normalise_key,
)

# ---------------------------------------------------------------------------
# Fixtures: small synthetic record/finding sets with known correct answers.
# These are test scaffolding only. Nothing here is a label and nothing here
# reaches a model.
# ---------------------------------------------------------------------------

STATES = ["NAGALAND", "JHARKHAND", "RAJASTHAN", "KERALA"]
DISTRICTS = ["KOHIMA", "DIMAPUR", "RANCHI", "JAIPUR", "KOCHI"]
SCHEMES = ["PMGSY", "MGNREGA", "NRLM"]
YEARS = ["2020-21", "2021-22", "2022-23"]


def _districts(n_districts: Optional[int]) -> Sequence[str]:
    """District vocabulary: the small fixed one, or a wide synthetic one."""
    if n_districts is None:
        return DISTRICTS
    return [f"DISTRICT{i:04d}" for i in range(n_districts)]


def make_records(
    n: int, seed: int, *, with_work_id: bool = True, n_districts: Optional[int] = None
) -> pd.DataFrame:
    """A record frame spanning the vocabularies above."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "record_id": [f"R{i:05d}" for i in range(n)],
            "state": rng.choice(STATES, n),
            "district": rng.choice(_districts(n_districts), n),
            "scheme": rng.choice(SCHEMES, n),
            "financial_year": rng.choice(YEARS, n),
        }
    )
    frame["work_id"] = (
        [f"W{i % max(1, n // 3):04d}" for i in range(n)] if with_work_id else None
    )
    return frame


def make_findings(
    n: int, seed: int, *, with_work_id: bool = True, n_districts: Optional[int] = None
) -> pd.DataFrame:
    """An audit-finding frame drawn from the same vocabularies."""
    rng = np.random.default_rng(seed + 7919)
    frame = pd.DataFrame(
        {
            "audit_id": [f"A{i:04d}" for i in range(n)],
            "state": rng.choice(STATES, n),
            "district": rng.choice(_districts(n_districts), n),
            "scheme": rng.choice(SCHEMES, n),
            "financial_year": rng.choice(YEARS, n),
        }
    )
    frame["work_id"] = (
        [f"W{i % max(1, n // 2):04d}" for i in range(n)] if with_work_id else None
    )
    return frame


# ===========================================================================
# 1. BLOCKING PARITY - the load-bearing invariant
# ===========================================================================


class TestBlockingParity:
    """Blocking must change the cost and not the answer.

    This is the analogue of checking a sparse attention kernel against dense
    attention on a full mask: if the fast path and the obvious path disagree
    on any input, the fast path is not an optimisation, it is a bug with a
    benchmark attached.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_blocked_equals_naive(self, seed: int) -> None:
        """Same matches, same methods, same confidences - for every seed."""
        records = make_records(200, seed)
        findings = make_findings(40, seed)

        naive = match_naive(records, findings)
        blocked = match_blocked(records, findings)

        assert_match_frames_equal(naive.matches, blocked.matches)

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_parity_without_work_ids(self, seed: int) -> None:
        """Parity must hold on the weak-key path too.

        Real public data rarely carries a work identifier, so the branch that
        actually runs in production is the one with no strong key. Testing
        parity only on the easy path would test the path that never executes.
        """
        records = make_records(200, seed, with_work_id=False)
        findings = make_findings(40, seed, with_work_id=False)

        naive = match_naive(records, findings)
        blocked = match_blocked(records, findings)

        assert_match_frames_equal(naive.matches, blocked.matches)

    def test_parity_on_empty_inputs(self) -> None:
        """Degenerate inputs agree rather than raising."""
        records = make_records(0, 0)
        findings = make_findings(0, 0)

        naive = match_naive(records, findings)
        blocked = match_blocked(records, findings)

        assert len(naive.matches) == 0
        assert_match_frames_equal(naive.matches, blocked.matches)

    def test_blocking_actually_reduces_comparisons(self) -> None:
        """The optimisation must do something, or it is dead code.

        Asserted as a ratio rather than a runtime: a wall-clock assertion is
        flaky on a shared machine, while the comparison count is exact and is
        the quantity blocking is supposed to reduce.

        Uses realistic district cardinality. Blocking's benefit is a function
        of key cardinality, and the four-state toy vocabulary the parity tests
        use has almost nothing to block on - measuring the speedup there would
        understate it for a reason that is a property of the fixture rather
        than of the algorithm. India has ~750 districts.
        """
        records = make_records(2000, 42, n_districts=750)
        findings = make_findings(200, 42, n_districts=750)

        naive = match_naive(records, findings)
        blocked = match_blocked(records, findings)

        assert naive.comparisons == 2000 * 200
        assert blocked.comparisons < naive.comparisons / 10
        assert_match_frames_equal(naive.matches, blocked.matches)


# ===========================================================================
# 2. NEGATIVE CONTROL - the test that would have caught the 5,237
# ===========================================================================


class TestNegativeControl:
    """A matcher that finds structure in unrelated data finds it everywhere.

    This is the regression test for the defect that made the previous
    calibration dataset worthless: findings from Jharkhand were joined onto a
    postal-office directory because the two shared district names and nothing
    else. The matcher must return nothing when there is nothing to return.
    """

    def test_disjoint_geography_yields_no_matches(self) -> None:
        """No shared state means no match, at any evidence level.

        Built without work IDs. The first version of this test kept them and
        failed: the two frames still shared synthetic work_ids, so the matcher
        correctly matched on WORK_ID. That was the fixture contradicting its
        own premise, not a defect - a shared work identifier IS stronger
        evidence than a differing district, and the matcher should say so.
        The scenario being modelled here (a postal directory against a CAG
        report) has no work identifier on either side.
        """
        records = make_records(500, 3, with_work_id=False)
        records["state"] = "KERALA"
        records["district"] = "KOCHI"
        findings = make_findings(50, 3, with_work_id=False)
        findings["state"] = "NAGALAND"
        findings["district"] = "KOHIMA"

        result = match_blocked(records, findings)

        assert len(result.matches) == 0

    def test_disjoint_years_yield_no_calibration_eligible_matches(self) -> None:
        """The exact failure of the real corpus, as an executable assertion.

        The audit evidence is FY2020-21 and the structured financial data is
        FY2023-24 onward. Sharing a state must not be enough to produce a
        label that a fit would consume.
        """
        records = make_records(500, 4, with_work_id=False)
        records["state"] = "NAGALAND"
        records["financial_year"] = "2023-24"
        findings = make_findings(50, 4, with_work_id=False)
        findings["state"] = "NAGALAND"
        findings["financial_year"] = "2020-21"

        result = match_blocked(records, findings)
        eligible = result.matches[
            result.matches["evidence_level"].isin(CALIBRATION_ELIGIBLE_EVIDENCE)
        ]

        assert len(eligible) == 0

    def test_district_only_overlap_never_reaches_eligible_evidence(self) -> None:
        """A district-level coincidence is LEVEL_1 at best, never eligible.

        A CAG report finding an irregularity somewhere in a district does not
        make every work in that district a positive. The evidence level is the
        mechanism that enforces this, so it is asserted directly.
        """
        records = make_records(300, 5, with_work_id=False)
        records["scheme"] = "SCHEME_NOT_IN_FINDINGS"
        findings = make_findings(30, 5, with_work_id=False)

        result = match_blocked(records, findings)
        eligible = result.matches[
            result.matches["evidence_level"].isin(CALIBRATION_ELIGIBLE_EVIDENCE)
        ]

        assert len(eligible) == 0


# ===========================================================================
# 3. ORDER AND DETERMINISM
# ===========================================================================


class TestInvariance:
    """Matching is a function of content, not of row order or run count."""

    @pytest.mark.parametrize("seed", [21, 22, 23])
    def test_record_permutation_invariance(self, seed: int) -> None:
        """Shuffling the records must not change which pairs match."""
        records = make_records(300, seed)
        findings = make_findings(30, seed)
        rng = np.random.default_rng(seed)
        shuffled = records.iloc[rng.permutation(len(records))].reset_index(drop=True)

        base = match_blocked(records, findings).matches
        perm = match_blocked(shuffled, findings).matches

        assert_match_frames_equal(base, perm)

    @pytest.mark.parametrize("seed", [31, 32])
    def test_finding_permutation_invariance(self, seed: int) -> None:
        """Shuffling the findings must not change the result either."""
        records = make_records(300, seed)
        findings = make_findings(30, seed)
        rng = np.random.default_rng(seed)
        shuffled = findings.iloc[rng.permutation(len(findings))].reset_index(drop=True)

        base = match_blocked(records, findings).matches
        perm = match_blocked(records, shuffled).matches

        assert_match_frames_equal(base, perm)

    def test_determinism_across_repeated_runs(self) -> None:
        """Same input, same output. Twice is enough to catch set iteration."""
        records = make_records(400, 99)
        findings = make_findings(40, 99)

        first = match_blocked(records, findings).matches
        second = match_blocked(records, findings).matches

        assert_match_frames_equal(first, second)


# ===========================================================================
# 4. EVIDENCE SEMANTICS
# ===========================================================================


class TestEvidenceSemantics:
    """The match method and its evidence level must stay consistent."""

    def test_strongest_available_key_wins(self) -> None:
        """A pair sharing a work_id is matched on it, not on a weaker key."""
        records = pd.DataFrame(
            [
                {
                    "record_id": "R1",
                    "state": "NAGALAND",
                    "district": "KOHIMA",
                    "scheme": "PMGSY",
                    "financial_year": "2020-21",
                    "work_id": "W1",
                }
            ]
        )
        findings = pd.DataFrame(
            [
                {
                    "audit_id": "A1",
                    "state": "NAGALAND",
                    "district": "KOHIMA",
                    "scheme": "PMGSY",
                    "financial_year": "2020-21",
                    "work_id": "W1",
                }
            ]
        )

        result = match_blocked(records, findings)

        assert len(result.matches) == 1
        assert result.matches.iloc[0]["match_method"] == "WORK_ID"
        assert result.matches.iloc[0]["evidence_level"] == "LEVEL_3_WORK_IDENTIFIED"

    def test_method_and_evidence_always_agree_with_the_constants(self) -> None:
        """No row may carry a method/evidence pairing the constants deny.

        The mapping lives in constants precisely so a reviewer can audit it
        without reading the matcher; this asserts the matcher actually obeys
        it rather than carrying a second private copy.
        """
        records = make_records(500, 77)
        findings = make_findings(50, 77)

        matches = match_blocked(records, findings).matches

        for _, row in matches.iterrows():
            method = row["match_method"]
            assert method in MATCH_METHODS
            assert row["evidence_level"] == MATCH_METHOD_EVIDENCE[method]
            assert row["match_confidence"] == pytest.approx(
                MATCH_METHOD_CONFIDENCE[method]
            )

    def test_no_match_is_absent_not_zero(self) -> None:
        """Unmatched records are omitted, never emitted with a 0 outcome.

        Emitting an unmatched record with outcome 0 is how "never audited"
        silently becomes "audited and clean". The matcher must not be the
        place that conversion can happen.
        """
        records = make_records(200, 8)
        records["state"] = "KERALA"
        findings = make_findings(20, 8)
        findings["state"] = "NAGALAND"

        matches = match_blocked(records, findings).matches

        assert "outcome" not in matches.columns or matches["outcome"].isna().all()
        assert (matches["match_method"] != "NO_MATCH").all()


# ===========================================================================
# 5. KEY NORMALISATION
# ===========================================================================


class TestNormalisation:
    """Blocking keys collide only when they should."""

    @pytest.mark.parametrize(
        "left,right",
        [
            ("Jammu & Kashmir", "JAMMU AND KASHMIR"),
            ("  Kohima  ", "kohima"),
            ("Jammu and Kashmir", "jammu & kashmir"),
        ],
    )
    def test_equivalent_spellings_share_a_key(self, left: str, right: str) -> None:
        """The same place written two ways must land in the same bucket.

        This is the recall side of blocking: a normalisation that misses an
        equivalence silently drops every match in that bucket.
        """
        assert normalise_key(left) == normalise_key(right)

    @pytest.mark.parametrize(
        "left,right",
        [
            ("KOHIMA", "DIMAPUR"),
            ("NAGALAND", "MEGHALAYA"),
            ("PMGSY", "MGNREGA"),
        ],
    )
    def test_distinct_places_do_not_share_a_key(self, left: str, right: str) -> None:
        """The precision side: over-normalising merges unrelated places."""
        assert normalise_key(left) != normalise_key(right)

    def test_empty_and_unknown_never_form_a_bucket(self) -> None:
        """UNKNOWN must not become a join key.

        If it did, every record with a missing district would match every
        finding with a missing district - producing a large, entirely
        artificial match set. This is the second way the postal-directory bug
        could reappear.
        """
        for value in ["", "  ", "UNKNOWN", "unknown", None, np.nan, "-", "NA"]:
            assert normalise_key(value) is None

    def test_blocking_keys_are_ordered_strongest_first(self) -> None:
        """Candidate generation must try the specific key before the vague one."""
        row = {
            "state": "NAGALAND",
            "district": "KOHIMA",
            "scheme": "PMGSY",
            "financial_year": "2020-21",
            "work_id": "W1",
        }

        keys = blocking_keys(row)
        methods = [method for method, _ in keys]

        assert methods == sorted(
            methods, key=lambda m: -MATCH_METHOD_CONFIDENCE[m]
        )
        assert methods[0] == "WORK_ID"


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def assert_match_frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Compare two match frames as sets of pairs, ignoring row order.

    Row order is not part of the answer - two matchers that find the same
    pairs in a different sequence are the same matcher - so comparing frames
    positionally would fail on a correct implementation.
    """
    columns = ["record_id", "audit_id", "match_method", "evidence_level"]
    if len(left) == 0 and len(right) == 0:
        return
    left_sorted = (
        left[columns].sort_values(columns).reset_index(drop=True).astype(str)
    )
    right_sorted = (
        right[columns].sort_values(columns).reset_index(drop=True).astype(str)
    )
    pd.testing.assert_frame_equal(left_sorted, right_sorted)
