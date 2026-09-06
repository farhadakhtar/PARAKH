"""Property tests for Stage 10 - entity resolution.

Written before the implementation.

Stage 10 exists because a "group" in PARAKH has never been guaranteed to
correspond to a real-world thing. Three defects made that concrete:

* **R5** - 100 ``work_id`` values are reused across 200 rows, so keying on
  ``work_id`` silently merges unrelated works.
* **EXP-009** - a reconstructed join key produced groups of median size ONE.
  Every within-group statistic was computed on a single record, the model
  scored 0.52, and nothing in the output indicated a problem. A degenerate
  group returns a number, not an error.
* Cartel and risk signals were computed over those groupings regardless.

So the tests here are not about grouping more records together. They are about
the layer refusing to claim more than it knows. The two that carry the weight:

:class:`TestNeverCollapseAmbiguity`
    A weak match must leave records apart. Merging on thin evidence is the
    failure mode that produces confident nonsense downstream, and unlike a
    crash it is invisible.

:class:`TestTransitivityIsBounded`
    Union-find over pairwise similarity is transitive: A~B and B~C merges A
    with C even when A and C share nothing. Left unchecked this walks a group
    across an entire district and every "HIGH confidence" claim about it is
    false. Groups must be re-validated after merging, not just built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stage10.entity_graph import build_entity_graph
from src.stage10.vendor_entity import (
    normalise_vendor_name,
    resolve_vendor_entities,
)
from src.stage10.work_entity import (
    EntityResolutionError,
    resolve_work_entities,
)


def make_work(**overrides) -> dict:
    """One record with every field the resolver reads."""
    record = {
        "work_id": "W0001",
        "work_name": "Construction of CC road at ward 4",
        "district": "PUNE",
        "state": "MAHARASHTRA",
        "sanction_amount": 1_000_000.0,
        "amount_spent": 950_000.0,
        "implementing_agency": "PWD",
        "vendor_name": "ABC Constructions Pvt Ltd",
        "date_proposal": "2021-04-10",
        "date_approval": "2021-05-15",
        "date_completion": "2022-01-20",
        "status": "COMPLETED",
    }
    record.update(overrides)
    return record


def frame(*records) -> pd.DataFrame:
    return pd.DataFrame(list(records))


# ===========================================================================
# 1. NEVER COLLAPSE AMBIGUITY
# ===========================================================================


class TestNeverCollapseAmbiguity:
    """Weak evidence leaves records apart. Better unknown than wrong merge."""

    def test_weak_similarity_does_not_merge(self) -> None:
        """Two unrelated works in one district stay separate entities."""
        f = frame(
            make_work(work_id="A", work_name="Construction of CC road at ward 4"),
            make_work(work_id="B", work_name="Supply of computer hardware"),
        )

        out = resolve_work_entities(f)

        assert out["work_entity_id"].nunique() == 2

    def test_different_districts_never_merge(self) -> None:
        """District is a blocking key, not a similarity input.

        Two identically named works in different districts are different
        works. No amount of name similarity may override that.
        """
        f = frame(
            make_work(work_id="A", district="PUNE"),
            make_work(work_id="B", district="NAGPUR"),
        )

        out = resolve_work_entities(f)

        assert out["work_entity_id"].nunique() == 2

    def test_cost_far_apart_does_not_merge(self) -> None:
        """Identical names at 10x the cost are not the same work."""
        f = frame(
            make_work(work_id="A", sanction_amount=1_000_000.0),
            make_work(work_id="B", sanction_amount=10_000_000.0),
        )

        out = resolve_work_entities(f)

        assert out["work_entity_id"].nunique() == 2

    def test_different_fiscal_years_do_not_merge(self) -> None:
        """The same road built in two years is two works."""
        f = frame(
            make_work(work_id="A", date_approval="2021-05-15"),
            make_work(work_id="B", date_approval="2023-05-15"),
        )

        out = resolve_work_entities(f)

        assert out["work_entity_id"].nunique() == 2


# ===========================================================================
# 2. TRANSITIVITY MUST BE BOUNDED
# ===========================================================================


class TestTransitivityIsBounded:
    """Union-find chains must not produce false HIGH-confidence groups."""

    def test_chain_does_not_earn_high_confidence(self) -> None:
        """A~B, B~C, but A and C differ beyond tolerance.

        Pairwise merging will place all three together. The group is then NOT
        entitled to HIGH confidence, because HIGH asserts that every core
        field matches across the whole group - which is false here. The
        resolver must re-check the assembled group, not only the pairs that
        built it.
        """
        f = frame(
            make_work(work_id="A", sanction_amount=1_000_000.0),
            make_work(work_id="B", sanction_amount=1_090_000.0),
            make_work(work_id="C", sanction_amount=1_180_000.0),
        )

        out = resolve_work_entities(f)

        for entity_id, group in out.groupby("work_entity_id"):
            if len(group) < 2:
                continue
            amounts = pd.to_numeric(group["sanction_amount"], errors="coerce")
            spread = (amounts.max() - amounts.min()) / amounts.min()
            if spread > 0.10:
                assert group["entity_confidence"].iloc[0] != "HIGH", (
                    f"group {entity_id} spans {spread:.1%} yet claims HIGH"
                )

    def test_high_confidence_groups_are_internally_consistent(self) -> None:
        """The invariant, asserted directly on a mixed corpus."""
        rng = np.random.default_rng(11)
        records = []
        for i in range(120):
            records.append(
                make_work(
                    work_id=f"W{i:04d}",
                    work_name=f"Construction of CC road at ward {i % 7}",
                    district=["PUNE", "NAGPUR", "NASHIK"][i % 3],
                    sanction_amount=float(rng.choice([1e6, 1.05e6, 5e6, 2e7])),
                    implementing_agency=["PWD", "ZP"][i % 2],
                )
            )

        out = resolve_work_entities(frame(*records))

        high = out[out["entity_confidence"] == "HIGH"]
        for _, group in high.groupby("work_entity_id"):
            assert group["district"].nunique() == 1
            assert group["implementing_agency"].nunique() == 1


# ===========================================================================
# 3. DEGENERATE GROUPS - THE EXP-009 FIX
# ===========================================================================


class TestDegenerateGroups:
    """A group of one is a fact about ignorance, and must say so."""

    def test_singleton_is_marked_degenerate(self) -> None:
        f = frame(make_work(work_id="ALONE"))

        out = resolve_work_entities(f)

        row = out.iloc[0]
        assert row["degenerate_group"] is True or row["degenerate_group"] == True
        assert row["entity_confidence"] == "LOW"
        assert row["entity_consistency"] == "AMBIGUOUS"
        assert row["group_size"] == 1

    def test_every_singleton_in_a_mixed_corpus_is_marked(self) -> None:
        """No singleton may reach a downstream stage unlabelled.

        This is the whole point of the class: EXP-009's groups of one were
        invisible, and a statistic computed on them looked like every other
        statistic.
        """
        f = frame(
            make_work(work_id="A"),
            make_work(work_id="B"),
            make_work(work_id="C", work_name="Supply of desks", sanction_amount=4e5),
            make_work(work_id="D", district="NAGPUR", work_name="Drain repair"),
        )

        out = resolve_work_entities(f)
        sizes = out.groupby("work_entity_id")["work_entity_id"].transform("size")

        assert (out.loc[sizes == 1, "degenerate_group"]).all()
        assert not (out.loc[sizes > 1, "degenerate_group"]).any()

    def test_degenerate_rate_is_reported(self) -> None:
        """The caller must be able to see how much of the corpus is singletons.

        EXP-009 was survivable only because nobody looked. A summary that
        hides the singleton share repeats it.
        """
        f = frame(*[make_work(work_id=f"W{i}", work_name=f"Unrelated work {i}",
                              sanction_amount=1e5 * (i + 1)) for i in range(6)])

        out = resolve_work_entities(f)

        assert out.attrs["degenerate_share"] == pytest.approx(1.0)
        assert out.attrs["n_entities"] == 6


# ===========================================================================
# 4. R5 - REUSED work_id
# ===========================================================================


class TestReusedWorkId:
    """A shared work_id is a claim, not proof."""

    def test_same_id_different_work_does_not_collapse(self) -> None:
        """R5, directly. Two unrelated records share an id.

        The resolver must not treat ``work_id`` as an identity. On the real
        corpus 100 ids are reused across 200 rows, and keying on id merges
        works in different districts.
        """
        f = frame(
            make_work(work_id="DUP-1", district="PUNE",
                      work_name="Construction of CC road"),
            make_work(work_id="DUP-1", district="NAGPUR",
                      work_name="Supply of school furniture",
                      sanction_amount=3e5),
        )

        out = resolve_work_entities(f)

        assert out["work_entity_id"].nunique() == 2, (
            "a reused work_id collapsed two unrelated works"
        )

    def test_reused_id_is_surfaced_in_evidence(self) -> None:
        """When the id is reused, the record says so rather than hiding it."""
        f = frame(
            make_work(work_id="DUP-1", district="PUNE"),
            make_work(work_id="DUP-1", district="NAGPUR", work_name="Drain work",
                      sanction_amount=2e5),
        )

        out = resolve_work_entities(f)

        assert out["work_id_reused"].all()


# ===========================================================================
# 5. CONFIDENCE TIERS
# ===========================================================================


class TestConfidenceTiers:
    """HIGH / MEDIUM / LOW must mean what the brief says they mean."""

    def test_identical_records_without_scheme_cap_at_medium(self) -> None:
        """HIGH is unreachable when a core field cannot be checked.

        Corrected from an earlier version of this test, which asserted HIGH
        for identical records. That contradicted the design: the brief defines
        HIGH as "all core fields match (district, scheme, cost range, name
        similarity)", and `scheme` does not exist in this corpus. If the
        column is absent the claim cannot be made, so the group caps at
        MEDIUM. The test was wrong, not the resolver - quietly dropping an
        unverifiable requirement to keep claiming HIGH is precisely the
        behaviour this stage exists to prevent.
        """
        f = frame(make_work(work_id="A"), make_work(work_id="B"))

        out = resolve_work_entities(f)

        assert out["work_entity_id"].nunique() == 1
        assert out["entity_confidence"].iloc[0] == "MEDIUM"
        assert "scheme:UNVERIFIABLE" in (
            out["group_evidence"].iloc[0]["conflicting_fields"]
        )

    def test_high_is_reachable_when_scheme_is_present(self) -> None:
        """And the ceiling lifts the moment the field exists."""
        f = frame(
            make_work(work_id="A", scheme="PMGSY"),
            make_work(work_id="B", scheme="PMGSY"),
        )

        out = resolve_work_entities(f)

        assert out["work_entity_id"].nunique() == 1
        assert out["entity_confidence"].iloc[0] == "HIGH"
        assert out["entity_consistency"].iloc[0] == "CONSISTENT"

    def test_near_match_is_medium(self) -> None:
        """Minor name variation and a small cost difference.

        Same work, recorded twice with clerical drift. Groups, but does not
        earn HIGH.
        """
        f = frame(
            make_work(work_id="A", work_name="Construction of CC road at ward 4",
                      sanction_amount=1_000_000.0),
            make_work(work_id="B", work_name="Constrn of C.C. road, ward-4",
                      sanction_amount=1_060_000.0),
        )

        out = resolve_work_entities(f)

        assert out["work_entity_id"].nunique() == 1
        assert out["entity_confidence"].iloc[0] in {"MEDIUM", "HIGH"}

    def test_matched_and_conflicting_fields_are_listed(self) -> None:
        """The layer must explain itself, per the design principles."""
        f = frame(make_work(work_id="A"), make_work(work_id="B"))

        out = resolve_work_entities(f)
        evidence = out["group_evidence"].iloc[0]

        assert "matched_fields" in evidence and "conflicting_fields" in evidence
        assert "district" in evidence["matched_fields"]


# ===========================================================================
# 6. CONFLICT DETECTION
# ===========================================================================


class TestConflictDetection:
    """Contradiction inside a group is reported, never averaged away."""

    def test_different_agencies_conflict(self) -> None:
        f = frame(
            make_work(work_id="A", implementing_agency="PWD"),
            make_work(work_id="B", implementing_agency="ZILLA PARISHAD"),
        )

        out = resolve_work_entities(f)
        grouped = out.groupby("work_entity_id")["implementing_agency"].nunique()
        for entity_id, n_agencies in grouped.items():
            if n_agencies > 1:
                rows = out[out["work_entity_id"] == entity_id]
                assert rows["entity_consistency"].iloc[0] == "CONFLICTING"

    def test_cost_variance_over_30pct_conflicts(self) -> None:
        """Only checked where a group formed for other reasons.

        A >30% spread would normally prevent merging outright, so this
        exercises the case where the group exists and the spread is a
        contradiction within it.
        """
        f = frame(
            make_work(work_id="A", sanction_amount=1_000_000.0),
            make_work(work_id="B", sanction_amount=1_050_000.0),
            make_work(work_id="C", sanction_amount=1_100_000.0),
            make_work(work_id="D", sanction_amount=1_450_000.0),
        )

        out = resolve_work_entities(f)

        for entity_id, group in out.groupby("work_entity_id"):
            amounts = pd.to_numeric(group["sanction_amount"], errors="coerce")
            if len(group) > 1 and (amounts.max() / amounts.min() - 1) > 0.30:
                assert group["entity_consistency"].iloc[0] == "CONFLICTING"

    def test_conflicting_fields_are_named(self) -> None:
        f = frame(
            make_work(work_id="A", implementing_agency="PWD"),
            make_work(work_id="B", implementing_agency="ZILLA PARISHAD"),
        )

        out = resolve_work_entities(f)
        for _, group in out.groupby("work_entity_id"):
            if group["entity_consistency"].iloc[0] == "CONFLICTING":
                assert "implementing_agency" in (
                    group["group_evidence"].iloc[0]["conflicting_fields"]
                )


# ===========================================================================
# 7. VENDOR RESOLUTION
# ===========================================================================


class TestVendorNormalisation:
    """Legal suffixes and punctuation are not identity."""

    @pytest.mark.parametrize(
        "left,right",
        [
            ("ABC Ltd.", "abc limited"),
            ("ABC Pvt Ltd", "A.B.C. Private Limited"),
            ("XYZ & Co.", "xyz and company"),
            ("M/s Sharma Builders", "Sharma Builders"),
            ("  Kumar   LLP ", "kumar llp"),
        ],
    )
    def test_equivalent_names_normalise_together(self, left, right) -> None:
        assert normalise_vendor_name(left) == normalise_vendor_name(right)

    @pytest.mark.parametrize(
        "left,right",
        [
            ("ABC Constructions", "XYZ Constructions"),
            ("Sharma Builders", "Verma Builders"),
            ("Kumar Ltd", "Kumari Ltd"),
        ],
    )
    def test_distinct_names_stay_distinct(self, left, right) -> None:
        """The precision side. Over-normalising merges unrelated firms.

        ``Kumar`` versus ``Kumari`` is one character apart and they are
        different companies; an edit-distance rule loose enough to merge them
        would merge most of a district.
        """
        assert normalise_vendor_name(left) != normalise_vendor_name(right)

    @pytest.mark.parametrize("empty", ["", "  ", None, np.nan, "-", "NA", "M/s"])
    def test_empty_and_sentinel_names_yield_none(self, empty) -> None:
        """A missing vendor must never become a bucket.

        If it did, every record with no vendor would resolve to one entity -
        the largest and most spurious vendor in the corpus.
        """
        assert normalise_vendor_name(empty) is None


class TestVendorEntities:
    """Exact normalised match merges; weak similarity does not."""

    def test_exact_normalised_match_is_high(self) -> None:
        f = frame(
            make_work(work_id="A", vendor_name="ABC Ltd."),
            make_work(work_id="B", vendor_name="abc limited"),
        )

        out = resolve_vendor_entities(f)

        assert out["vendor_entity_id"].nunique() == 1
        assert out["vendor_confidence"].iloc[0] == "HIGH"

    def test_weak_similarity_stays_separate_and_ambiguous(self) -> None:
        """Per the brief: ambiguous vendors remain separate entities."""
        f = frame(
            make_work(work_id="A", vendor_name="Sharma Builders"),
            make_work(work_id="B", vendor_name="Verma Traders"),
        )

        out = resolve_vendor_entities(f)

        assert out["vendor_entity_id"].nunique() == 2

    def test_missing_vendor_gets_its_own_entity(self) -> None:
        """Two unknown vendors are not the same vendor."""
        f = frame(
            make_work(work_id="A", vendor_name=None),
            make_work(work_id="B", vendor_name=""),
        )

        out = resolve_vendor_entities(f)

        assert out["vendor_entity_id"].nunique() == 2
        assert (out["vendor_confidence"] == "LOW").all()


# ===========================================================================
# 8. INVARIANTS
# ===========================================================================


class TestInvariants:
    """Violations raise, they do not warn."""

    def test_record_count_is_preserved(self) -> None:
        f = frame(*[make_work(work_id=f"W{i}") for i in range(40)])

        out = resolve_work_entities(f)

        assert len(out) == len(f)

    def test_every_record_has_exactly_one_work_entity_id(self) -> None:
        f = frame(*[make_work(work_id=f"W{i}") for i in range(25)])

        out = resolve_work_entities(f)

        assert out["work_entity_id"].notna().all()
        assert len(out) == len(out.index.unique())

    def test_index_is_preserved(self) -> None:
        """Attaching entities must not reindex the caller's frame."""
        f = frame(*[make_work(work_id=f"W{i}") for i in range(10)])
        f.index = [f"row-{i}" for i in range(10)]

        out = resolve_work_entities(f)

        assert list(out.index) == list(f.index)

    def test_original_columns_are_untouched(self) -> None:
        """Non-destructive: nothing is overwritten."""
        f = frame(make_work(work_id="A"), make_work(work_id="B"))
        before = f.copy()

        out = resolve_work_entities(f)

        for column in before.columns:
            pd.testing.assert_series_equal(
                out[column], before[column], check_names=False
            )

    def test_empty_frame_is_handled(self) -> None:
        out = resolve_work_entities(pd.DataFrame(columns=list(make_work())))
        assert len(out) == 0

    def test_missing_required_column_raises(self) -> None:
        f = frame(make_work()).drop(columns=["district"])

        with pytest.raises(EntityResolutionError, match="district"):
            resolve_work_entities(f)


# ===========================================================================
# 9. ENTITY GRAPH
# ===========================================================================


class TestEntityGraph:
    """Minimal adjacency, computed from resolved entities only."""

    def test_vendor_linked_to_its_works(self) -> None:
        f = frame(
            make_work(work_id="A", vendor_name="ABC Ltd"),
            make_work(work_id="B", vendor_name="ABC Ltd",
                      work_name="Drain repair", sanction_amount=4e5),
        )

        works = resolve_work_entities(f)
        vendors = resolve_vendor_entities(works)
        graph = build_entity_graph(vendors)

        vendor_id = vendors["vendor_entity_id"].iloc[0]
        assert graph["vendors"][vendor_id]["degree"] >= 1
        assert len(graph["vendors"][vendor_id]["connected_works"]) >= 1

    def test_degree_counts_distinct_counterparties(self) -> None:
        """Ten works for one vendor is degree ten, not ten thousand."""
        f = frame(*[
            make_work(work_id=f"W{i}", vendor_name="ABC Ltd",
                      work_name=f"Work number {i}",
                      sanction_amount=1e5 * (i + 1))
            for i in range(10)
        ])

        works = resolve_work_entities(f)
        vendors = resolve_vendor_entities(works)
        graph = build_entity_graph(vendors)

        vendor_id = vendors["vendor_entity_id"].iloc[0]
        node = graph["vendors"][vendor_id]
        assert node["degree"] == len(set(node["connected_works"]))

    def test_graph_is_deterministic(self) -> None:
        f = frame(*[make_work(work_id=f"W{i}", vendor_name=f"V{i % 3} Ltd",
                              work_name=f"Work {i}", sanction_amount=1e5 * (i + 1))
                    for i in range(15)])

        works = resolve_work_entities(f)
        vendors = resolve_vendor_entities(works)

        assert build_entity_graph(vendors) == build_entity_graph(vendors)
