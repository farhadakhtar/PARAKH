"""Stage 2 test suite - the Evidentiary Confidence Engine.

Organised against Stage2.md's own structure:

* ``TestEntropy``               - sec.5.2 entropy primitives
* ``TestFieldWeights``          - sec.5.2 v_f estimation
* ``TestCompleteness``          - sec.5.2 C_comp
* ``TestTemporal``              - sec.5.3 C_temp
* ``TestReconciliation``        - sec.5.4 C_recon
* ``TestLogSpaceAggregation``   - sec.5.1 and sec.7 numerical stability
* ``TestFunctional``            - sec.8.1 the four mandated cases
* ``TestEdgeCases``             - sec.9 mandatory edge cases
* ``TestDistribution``          - sec.8.2 sanity checks
* ``TestDeterminism``           - sec.7
* ``TestIntegration``           - attachment to Corpus
* ``TestPerformance``           - sec.7, 50k in under 3s
* ``TestAcceptanceCriteria``    - sec.11
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    CONFIDENCE_SECONDS_BUDGET,
    FIELD_ORDER,
    PERFORMANCE_ROW_BUDGET,
    RECON_LAMBDA,
    RECON_ONE_SIDED_CREDIT,
    SCHEME_START_DATE,
)
from src.stage1.corpus import Corpus
from src.stage1.data_generator import generate_dataset
from src.stage1.schema import NullReason, null_reason_column
from src.stage2.completeness import (
    bernoulli_entropy,
    compute_completeness,
    compute_completeness_result,
    compute_field_weights,
    credit_matrix,
    resolve_reasons,
    value_entropy,
)
from src.stage2.confidence import (
    CONFIDENCE_COLUMN,
    ConfidenceConfig,
    ConfidenceModel,
    attach_confidence,
    compute_confidence,
    confidence_summary_frame,
    log_space_geometric_mean,
)
from src.stage2.reconciliation import (
    compute_reconciliation,
    compute_reconciliation_result,
)
from src.stage2.temporal import compute_temporal, compute_temporal_result

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

#: A clean, fully-populated, internally consistent record with amounts that
#: reconcile closely (ratio ~ 0.023 -> C_recon ~ 0.956).
BASE_ROW: Dict[str, Any] = {
    "work_id": "MPL-KA-2019-000001",
    "work_name": "Construction of CC Road at Ward No. 7, Mysuru",
    "district": "Mysuru",
    "state": "Karnataka",
    "sanction_amount": 850000.0,
    "amount_spent": 812345.50,
    "date_proposal": "2019-03-01",
    "date_approval": "2019-05-20",
    "date_completion": "2020-01-15",
    "implementing_agency": "Mysuru Zilla Parishad",
    "vendor_name": "Iyer Constructions",
    "status": "completed",
}

#: Field values that make each record distinct, so the corpus has entropy and
#: v_f does not collapse to the degenerate fallback.
_D0 = pd.Timestamp("2019-03-01")

_VARIETY = {
    "district": ["Mysuru", "Pune", "Patna", "Jaipur", "Surat", "Nadia"],
    "state": ["Karnataka", "Maharashtra", "Bihar", "Rajasthan", "Gujarat", "West Bengal"],
    "vendor_name": ["Iyer Constructions", "Rao Builders", "Das Infra", "Bose and Sons"],
    "status": ["completed", "approved", "proposed"],
}


def make_frame(*rows: Optional[Dict[str, Any]], vary: bool = True) -> pd.DataFrame:
    """Build an object-dtype frame from overrides on :data:`BASE_ROW`.

    Args:
        *rows: One dict of field overrides per row; ``None`` yields a pristine
            base row.
        vary: Rotate a few field values so the corpus carries entropy. Turn off
            to exercise the zero-variance path.

    Returns:
        A schema-ordered DataFrame.
    """
    records: List[Dict[str, Any]] = []
    for index, overrides in enumerate(rows or [None]):
        row = dict(BASE_ROW)
        row["work_id"] = f"MPL-KA-2019-{index + 1:06d}"
        if vary:
            for name, options in _VARIETY.items():
                row[name] = options[index % len(options)]
            row["work_name"] = f"{BASE_ROW['work_name']} unit {index}"
            # Dates and amounts must vary too. A field that is constant across
            # the corpus has H_value = 0, hence v_f = 0, and its defects cannot
            # move C_comp - correct behaviour, but it makes the field invisible
            # to any test that holds it fixed.
            row["date_proposal"] = (_D0 + pd.Timedelta(days=index)).date().isoformat()
            row["date_approval"] = (
                _D0 + pd.Timedelta(days=80 + index)
            ).date().isoformat()
            row["date_completion"] = (
                _D0 + pd.Timedelta(days=320 + index)
            ).date().isoformat()
            row["sanction_amount"] = 850000.0 + 1000.0 * index
            row["amount_spent"] = round((850000.0 + 1000.0 * index) * 0.9557, 2)
        if overrides:
            row.update(overrides)
        records.append(row)
    return pd.DataFrame(records, columns=list(FIELD_ORDER)).astype("object")


def corpus_of(*rows: Optional[Dict[str, Any]], vary: bool = True) -> Corpus:
    """Build a Stage 1 corpus directly from override dicts."""
    return Corpus.from_dataframe(make_frame(*rows, vary=vary))


def score_one(**overrides: Any) -> Dict[str, float]:
    """Score a single record padded with clean neighbours.

    The padding matters: ``v_f`` is estimated from the corpus, so scoring one
    record in isolation would collapse every weight to the degenerate uniform
    fallback and stop testing the real weighting path.

    Returns:
        ``{"confidence", "completeness", "temporal", "reconciliation"}`` for the
        record under test, which is always row 0.
    """
    padding = [None] * 12
    corpus = corpus_of(overrides, *padding)
    result = ConfidenceModel().score(corpus)
    return {
        "confidence": float(result.scores.iloc[0]),
        "completeness": float(result.completeness.scores.iloc[0]),
        "temporal": float(result.temporal.scores.iloc[0]),
        "reconciliation": float(result.reconciliation.scores.iloc[0]),
    }


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    """A full-size, realistically dirty Stage 1 corpus."""
    return Corpus.from_dataframe(generate_dataset(n=10_000, seed=42))


@pytest.fixture(scope="module")
def result(corpus: Corpus) -> Any:
    """Confidence scored over the full-size corpus."""
    return ConfidenceModel().score(corpus)


# ---------------------------------------------------------------------------
# sec.5.2 - entropy primitives
# ---------------------------------------------------------------------------


class TestEntropy:
    """Stage2.md sec.5.2: entropy computed across the corpus, normalised to [0,1]."""

    @pytest.mark.parametrize(
        ("p", "expected"),
        [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.25, 0.811278), (0.75, 0.811278)],
    )
    def test_bernoulli_entropy_values(self, p: float, expected: float) -> None:
        assert bernoulli_entropy(p) == pytest.approx(expected, abs=1e-6)

    def test_bernoulli_entropy_is_symmetric(self) -> None:
        for p in (0.1, 0.3, 0.42):
            assert bernoulli_entropy(p) == pytest.approx(bernoulli_entropy(1 - p))

    def test_bernoulli_entropy_is_bounded(self) -> None:
        for p in np.linspace(0.0, 1.0, 101):
            assert 0.0 <= bernoulli_entropy(float(p)) <= 1.0

    def test_bernoulli_entropy_handles_nonsense_input(self) -> None:
        assert bernoulli_entropy(float("nan")) == 0.0
        assert bernoulli_entropy(-1.0) == 0.0
        assert bernoulli_entropy(2.0) == 0.0

    def test_constant_field_has_zero_value_entropy(self) -> None:
        """Stage2.md: a constant field is non-informative and must not dominate."""
        normalized, k, _ = value_entropy(pd.Series(["a"] * 100))
        assert normalized == 0.0
        assert k == 1

    def test_empty_field_has_zero_value_entropy(self) -> None:
        assert value_entropy(pd.Series([], dtype="object")) == (0.0, 0, 0.0)

    def test_uniform_field_saturates_cardinality_normalisation(self) -> None:
        normalized, k, raw = value_entropy(pd.Series(list("abcd") * 25))
        assert normalized == pytest.approx(1.0)
        assert k == 4
        assert raw == pytest.approx(2.0)

    def test_skewed_field_scores_below_uniform(self) -> None:
        skewed, _, _ = value_entropy(pd.Series(["a"] * 99 + ["b"]))
        uniform, _, _ = value_entropy(pd.Series(["a"] * 50 + ["b"] * 50))
        assert skewed < uniform

    def test_sample_normalisation_penalises_low_cardinality(self) -> None:
        values = pd.Series(list("ab") * 500)
        by_cardinality, _, _ = value_entropy(values, normalization="cardinality")
        by_sample, _, _ = value_entropy(values, normalization="sample")
        assert by_cardinality == pytest.approx(1.0)
        assert by_sample < 0.2

    def test_unknown_normalisation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="normalization"):
            value_entropy(pd.Series(["a", "b"]), normalization="bogus")


# ---------------------------------------------------------------------------
# sec.5.2 - field weights
# ---------------------------------------------------------------------------


class TestFieldWeights:
    """Stage2.md sec.5.2: v_f = (1 - H_null) * H_value, cached for reuse."""

    def test_weights_cover_every_schema_field(self, corpus: Corpus) -> None:
        weights = compute_field_weights(corpus.records)
        assert set(weights.weights) == set(FIELD_ORDER)
        assert weights.total > 0

    def test_weights_are_non_negative_and_bounded(self, corpus: Corpus) -> None:
        weights = compute_field_weights(corpus.records)
        assert all(0.0 <= v <= 1.0 for v in weights.weights.values())

    def test_shares_sum_to_one(self, corpus: Corpus) -> None:
        weights = compute_field_weights(corpus.records)
        assert sum(weights.shares.values()) == pytest.approx(1.0)

    def test_rarely_null_field_outweighs_often_null_field(self, corpus: Corpus) -> None:
        """The surprisal design: an absence is informative only if absences are rare."""
        weights = compute_field_weights(corpus.records)
        assert weights.coverage["work_name"] > weights.coverage["vendor_name"]
        assert weights.weights["work_name"] > weights.weights["vendor_name"]

    def test_constant_field_gets_zero_weight(self) -> None:
        frame = make_frame(*[None] * 20)
        frame["state"] = "Karnataka"
        weights = compute_field_weights(Corpus.from_dataframe(frame).records)
        assert weights.weights["state"] == 0.0

    def test_zero_variance_corpus_falls_back_to_uniform(self) -> None:
        """Mandated edge case: identical values must not produce 0/0."""
        frame = make_frame(*[None] * 8, vary=False)
        frame["work_id"] = "MPL-KA-2019-000001"
        corpus = Corpus.from_dataframe(frame)
        weights = compute_field_weights(corpus.records)
        assert weights.degenerate is True
        assert set(weights.weights.values()) == {1.0}

    def test_zero_variance_perfect_records_still_score_high(self) -> None:
        """Without the fallback these would all score 0, which is plainly wrong."""
        frame = make_frame(*[None] * 8, vary=False)
        frame["work_id"] = "MPL-KA-2019-000001"
        corpus = Corpus.from_dataframe(frame)
        scores = ConfidenceModel().score(corpus).scores
        assert (scores > 0.9).all()

    def test_low_coverage_field_is_excluded(self) -> None:
        """Guards the non-monotonicity of (1 - H_null), high at BOTH p=0 and p=1."""
        rows = [{"vendor_name": None} for _ in range(199)] + [{"vendor_name": "Sole Co"}]
        corpus = Corpus.from_dataframe(make_frame(*rows))
        weights = compute_field_weights(corpus.records, min_coverage=0.02)
        assert "vendor_name" in weights.excluded_fields
        assert weights.weights["vendor_name"] == 0.0

    def test_structural_floor_is_reported(self, corpus: Corpus) -> None:
        """work_id is never null, so it lifts every record's C_comp identically."""
        weights = compute_field_weights(corpus.records)
        assert 0.0 < weights.structural_floor < 1.0

    def test_weights_are_frozen_and_reusable_across_batches(self, corpus: Corpus) -> None:
        """v_f is corpus state; frozen weights keep two batches comparable."""
        weights = compute_field_weights(corpus.records)
        first = compute_completeness(corpus.records.head(500), weights=weights)
        second = compute_completeness(corpus.records.head(500))
        assert not np.allclose(first.to_numpy(), second.to_numpy())
        again = compute_completeness(corpus.records.head(500), weights=weights)
        pd.testing.assert_series_equal(first, again)

    def test_weights_serialise(self, corpus: Corpus) -> None:
        payload = compute_field_weights(corpus.records).to_dict()
        json.dumps(payload)
        assert set(payload["per_field"]) == set(FIELD_ORDER)


# ---------------------------------------------------------------------------
# sec.5.2 - completeness
# ---------------------------------------------------------------------------


class TestCompleteness:
    """Stage2.md sec.5.2 plus the brief's defect-awareness rule."""

    def test_perfect_record_scores_one(self) -> None:
        assert score_one()["completeness"] == pytest.approx(1.0)

    def test_output_is_bounded(self, result: Any) -> None:
        scores = result.completeness.scores
        assert scores.between(0.0, 1.0).all()
        assert np.isfinite(scores).all()

    def test_more_defects_lower_the_score(self) -> None:
        clean = score_one()["completeness"]
        one_gap = score_one(vendor_name=None)["completeness"]
        two_gaps = score_one(vendor_name=None, district=None)["completeness"]
        assert clean > one_gap > two_gaps

    def test_defect_awareness_ordering(self) -> None:
        """Brief: missing < placeholder < unparseable in severity.

        The three null reasons look identical downstream - all are just NaN -
        so this is the test that proves Stage 1's taxonomy is actually being
        consumed rather than collapsed.
        """
        missing = score_one(date_completion=None)["completeness"]
        placeholder = score_one(date_completion="N/A")["completeness"]
        unparseable = score_one(date_completion="not a date")["completeness"]
        clean = score_one()["completeness"]
        assert clean > missing > placeholder > unparseable

    def test_null_reasons_are_actually_distinct_in_stage1(self) -> None:
        """Guards the premise of the test above."""
        corpus = corpus_of(
            {"date_completion": None},
            {"date_completion": "N/A"},
            {"date_completion": "not a date"},
        )
        column = corpus.records[null_reason_column("date_completion")].astype(object)
        assert list(column) == [
            NullReason.MISSING.value,
            NullReason.PLACEHOLDER.value,
            NullReason.UNPARSEABLE.value,
        ]

    def test_a_constant_field_cannot_move_the_score(self) -> None:
        """The flip side of entropy weighting, asserted rather than assumed.

        A field with the same value in every record has H_value = 0, hence
        v_f = 0, so losing it costs nothing. That is correct - its presence
        proved nothing to begin with - but it is surprising enough to pin down.
        """
        rows = [{"state": "Karnataka"} for _ in range(12)]
        rows[0] = {"state": None}
        frame = make_frame(*rows)
        frame["state"] = ["Karnataka"] * 12
        frame.loc[0, "state"] = None
        corpus = Corpus.from_dataframe(frame)
        outcome = compute_completeness_result(corpus.records)
        assert outcome.weights.weights["state"] == 0.0
        assert outcome.scores.iloc[0] == pytest.approx(outcome.scores.iloc[1])

    def test_credit_matrix_maps_reasons_to_credits(self) -> None:
        corpus = corpus_of({"vendor_name": "N/A"}, None)
        matrix = credit_matrix(corpus.records, list(FIELD_ORDER))
        position = list(FIELD_ORDER).index("vendor_name")
        assert matrix[0, position] == pytest.approx(0.08)
        assert matrix[1, position] == pytest.approx(1.0)

    def test_no_present_field_forces_zero(self) -> None:
        """Residual credit on no evidence at all is still nothing."""
        blank = {name: None for name in FIELD_ORDER}
        corpus = Corpus.from_dataframe(
            pd.DataFrame([blank, dict(BASE_ROW)], columns=list(FIELD_ORDER)).astype(object)
        )
        outcome = compute_completeness_result(corpus.records)
        assert outcome.scores.iloc[0] == 0.0
        assert bool(outcome.no_evidence.iloc[0])
        assert not bool(outcome.no_evidence.iloc[1])

    def test_evidence_rule_can_be_disabled(self) -> None:
        blank = {name: None for name in FIELD_ORDER}
        corpus = Corpus.from_dataframe(
            pd.DataFrame([blank], columns=list(FIELD_ORDER)).astype(object)
        )
        relaxed = compute_completeness_result(corpus.records, require_evidence=False)
        assert relaxed.scores.iloc[0] > 0.0

    def test_no_imputation_occurs(self) -> None:
        """The scorer must never write a value back into the corpus."""
        corpus = corpus_of({"sanction_amount": None, "vendor_name": None}, None)
        before = corpus.records.copy(deep=True)
        compute_completeness(corpus.records)
        pd.testing.assert_frame_equal(before, corpus.records)

    def test_falls_back_when_null_reason_columns_are_absent(self) -> None:
        frame = make_frame(None, {"vendor_name": None}, None)
        corpus = Corpus.from_dataframe(frame)
        bare = corpus.records[list(FIELD_ORDER)].copy()
        reasons, derived = resolve_reasons(bare, list(FIELD_ORDER))
        assert derived is True
        assert reasons["vendor_name"].iloc[1] == NullReason.MISSING.value
        scores = compute_completeness(bare)
        assert scores.between(0.0, 1.0).all()

    def test_index_and_order_are_preserved(self, corpus: Corpus) -> None:
        scores = compute_completeness(corpus.records)
        assert scores.index.equals(corpus.records.index)
        assert len(scores) == len(corpus)


# ---------------------------------------------------------------------------
# sec.5.3 - temporal coherence
# ---------------------------------------------------------------------------


class TestTemporal:
    """Stage2.md sec.5.3, all five cases from the brief."""

    def test_case1_ordered_dates_score_one(self) -> None:
        assert score_one()["temporal"] == pytest.approx(1.0)

    def test_equal_dates_are_not_a_violation(self) -> None:
        scores = score_one(
            date_proposal="2019-03-01",
            date_approval="2019-03-01",
            date_completion="2019-03-01",
        )
        assert scores["temporal"] == pytest.approx(1.0)

    def test_case2_inversion_decays_exponentially(self) -> None:
        """kappa is per DAY; the unit is load-bearing."""
        scores = score_one(date_proposal="2019-03-01", date_approval="2019-02-19")
        assert scores["temporal"] == pytest.approx(np.exp(-0.01 * 10), rel=1e-6)

    def test_larger_inversions_are_penalised_harder(self) -> None:
        small = score_one(date_proposal="2019-03-01", date_approval="2019-02-27")
        large = score_one(date_proposal="2019-03-01", date_approval="2018-03-01")
        assert small["temporal"] > large["temporal"] > 0.0

    def test_violations_multiply_across_pairs(self) -> None:
        """C_temp is a PRODUCT over pairs, not a mean."""
        both = score_one(
            date_proposal="2019-03-01",
            date_approval="2019-02-19",
            date_completion="2019-02-09",
        )
        assert both["temporal"] == pytest.approx(np.exp(-0.01 * 20), rel=1e-6)

    def test_case3_missing_date_is_neutral_for_its_pair(self) -> None:
        """Absence is a completeness defect; charging it here double-bills it."""
        outcome = compute_temporal_result(
            corpus_of({"date_completion": None}, None).records
        )
        assert outcome.scores.iloc[0] == pytest.approx(1.0)
        assert int(outcome.pairs_evaluated.iloc[0]) == 1

    def test_case4_unparseable_date_is_a_hard_fail(self) -> None:
        outcome = compute_temporal_result(
            corpus_of({"date_approval": "not a date"}, None).records
        )
        assert outcome.scores.iloc[0] == 0.0
        assert bool(outcome.hard_fail.iloc[0])

    def test_placeholder_date_is_not_a_hard_fail(self) -> None:
        """"0000-00-00" is a declared absence, not a broken parse."""
        outcome = compute_temporal_result(
            corpus_of({"date_approval": "0000-00-00"}, None).records
        )
        assert outcome.scores.iloc[0] == pytest.approx(1.0)
        assert not bool(outcome.hard_fail.iloc[0])

    def test_case5_pre_scheme_date_is_a_hard_fail(self) -> None:
        outcome = compute_temporal_result(
            corpus_of(
                {
                    "date_proposal": "1985-01-01",
                    "date_approval": "1985-06-01",
                    "date_completion": "1986-01-01",
                },
                None,
            ).records
        )
        assert outcome.scores.iloc[0] == 0.0

    def test_date_on_the_scheme_boundary_is_accepted(self) -> None:
        outcome = compute_temporal_result(
            corpus_of(
                {
                    "date_proposal": SCHEME_START_DATE.isoformat(),
                    "date_approval": "1993-06-01",
                    "date_completion": "1994-01-01",
                },
                None,
            ).records
        )
        assert outcome.scores.iloc[0] == pytest.approx(1.0)

    def test_hard_fail_overrides_a_soft_violation(self) -> None:
        scores = score_one(date_approval="not a date", date_completion="2018-01-01")
        assert scores["temporal"] == 0.0

    def test_future_date_hard_fail_is_opt_in(self) -> None:
        frame = corpus_of({"date_completion": "2030-01-01"}, None).records
        assert compute_temporal(frame).iloc[0] == pytest.approx(1.0)
        strict = compute_temporal(frame, hard_fail_on_future=True)
        assert strict.iloc[0] == 0.0

    def test_no_dates_marks_the_component_undefined(self) -> None:
        """C_temp = 1 on zero evidence must be distinguishable from coherence."""
        outcome = compute_temporal_result(
            corpus_of(
                {"date_proposal": None, "date_approval": None, "date_completion": None},
                None,
            ).records
        )
        assert int(outcome.pairs_evaluated.iloc[0]) == 0
        assert not bool(outcome.defined.iloc[0])
        assert bool(outcome.defined.iloc[1])
        assert outcome.coverage.iloc[0] == 0.0

    def test_hard_fail_counts_as_defined(self) -> None:
        """An impossible date is evidence of incoherence, not absence of evidence."""
        outcome = compute_temporal_result(
            corpus_of({"date_approval": "not a date"}, None).records
        )
        assert bool(outcome.defined.iloc[0])

    def test_output_is_bounded(self, result: Any) -> None:
        scores = result.temporal.scores
        assert scores.between(0.0, 1.0).all()
        assert np.isfinite(scores).all()

    def test_negative_kappa_is_rejected(self, corpus: Corpus) -> None:
        with pytest.raises(ValueError, match="kappa"):
            compute_temporal(corpus.records.head(5), kappa=-1.0)

    def test_out_of_range_missing_credit_is_rejected(self, corpus: Corpus) -> None:
        with pytest.raises(ValueError, match="missing_pair_credit"):
            compute_temporal(corpus.records.head(5), missing_pair_credit=1.5)


# ---------------------------------------------------------------------------
# sec.5.4 - reconciliation
# ---------------------------------------------------------------------------


class TestReconciliation:
    """Stage2.md sec.5.4."""

    def test_identical_amounts_score_one(self) -> None:
        assert score_one(sanction_amount=1000.0, amount_spent=1000.0)[
            "reconciliation"
        ] == pytest.approx(1.0, abs=1e-5)

    def test_matches_the_closed_form(self) -> None:
        score = score_one(sanction_amount=1000.0, amount_spent=800.0)["reconciliation"]
        expected = float(np.exp(-RECON_LAMBDA * (200.0 / 1800.0)))
        assert score == pytest.approx(expected, rel=1e-6)

    def test_larger_mismatch_scores_lower(self) -> None:
        close = score_one(sanction_amount=1000.0, amount_spent=950.0)["reconciliation"]
        far = score_one(sanction_amount=1000.0, amount_spent=100.0)["reconciliation"]
        assert close > far

    def test_score_is_scale_free(self) -> None:
        """Scale-free up to epsilon, which perturbs the ratio at tiny magnitudes."""
        small = score_one(sanction_amount=100.0, amount_spent=80.0)["reconciliation"]
        large = score_one(sanction_amount=100_000_000.0, amount_spent=80_000_000.0)
        assert small == pytest.approx(large["reconciliation"], rel=1e-7)

    def test_theoretical_floor_is_respected(self, result: Any) -> None:
        """Symmetric normalisation bounds the ratio, so C_recon >= exp(-lambda)."""
        scores = result.reconciliation.scores
        non_zero = scores[scores > 0]
        assert non_zero.min() >= float(np.exp(-RECON_LAMBDA)) - 1e-9

    def test_both_zero_reconciles_perfectly(self) -> None:
        """0 vs 0 must not divide by zero; epsilon makes it well defined."""
        assert score_one(sanction_amount=0.0, amount_spent=0.0)[
            "reconciliation"
        ] == pytest.approx(1.0)

    def test_one_null_takes_the_fixed_penalty(self) -> None:
        outcome = compute_reconciliation_result(
            corpus_of({"amount_spent": None}, None).records
        )
        assert outcome.scores.iloc[0] == pytest.approx(RECON_ONE_SIDED_CREDIT)
        assert outcome.branch.iloc[0] == "one_null"

    def test_both_null_is_undefined_not_perfect(self) -> None:
        """Stage2.md's "ignore component" means drop and renormalise, not score 1."""
        outcome = compute_reconciliation_result(
            corpus_of({"sanction_amount": None, "amount_spent": None}, None).records
        )
        assert outcome.branch.iloc[0] == "both_null"
        assert not bool(outcome.defined.iloc[0])
        assert bool(outcome.defined.iloc[1])

    def test_infinite_amount_hard_fails(self) -> None:
        """inf/inf is NaN; left unhandled it would poison the log-sum silently."""
        outcome = compute_reconciliation_result(
            corpus_of({"sanction_amount": "1.2e400"}, None).records
        )
        assert outcome.scores.iloc[0] == 0.0
        assert outcome.branch.iloc[0] == "non_finite"

    def test_extreme_but_finite_amount_is_scored_normally(self) -> None:
        outcome = compute_reconciliation_result(
            corpus_of({"sanction_amount": 1e300, "amount_spent": 1e300}, None).records
        )
        assert outcome.scores.iloc[0] == pytest.approx(1.0)
        assert np.isfinite(outcome.scores).all()

    def test_negative_amounts_stay_bounded(self) -> None:
        outcome = compute_reconciliation_result(
            corpus_of({"amount_spent": -812345.5}, None).records
        )
        assert 0.0 <= float(outcome.scores.iloc[0]) <= 1.0

    def test_max_normalisation_is_available(self) -> None:
        frame = corpus_of({"sanction_amount": 1000.0, "amount_spent": 800.0}, None).records
        by_max = compute_reconciliation(frame, normalization="max")
        assert by_max.iloc[0] == pytest.approx(np.exp(-RECON_LAMBDA * 200.0 / 1000.0))

    def test_max_normalisation_is_not_sign_safe(self) -> None:
        """Documents exactly why symmetric is the default."""
        frame = corpus_of({"sanction_amount": -1000.0, "amount_spent": -500.0}, None).records
        symmetric = compute_reconciliation(frame, normalization="symmetric")
        by_max = compute_reconciliation(frame, normalization="max")
        assert symmetric.iloc[0] > 0.5
        assert by_max.iloc[0] == pytest.approx(0.0)

    def test_output_is_bounded(self, result: Any) -> None:
        scores = result.reconciliation.scores
        assert scores.between(0.0, 1.0).all()
        assert np.isfinite(scores).all()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"lam": -1.0}, "lambda"),
            ({"epsilon": 0.0}, "epsilon"),
            ({"one_sided_credit": 2.0}, "one_sided_credit"),
            ({"normalization": "bogus"}, "normalization"),
        ],
    )
    def test_invalid_parameters_are_rejected(
        self, corpus: Corpus, kwargs: Dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            compute_reconciliation(corpus.records.head(5), **kwargs)

    def test_missing_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="absent"):
            compute_reconciliation(pd.DataFrame({"a": [1.0]}))


# ---------------------------------------------------------------------------
# sec.5.1 / sec.7 - aggregation and numerical stability
# ---------------------------------------------------------------------------


class TestLogSpaceAggregation:
    """Stage2.md sec.5.1 and sec.7."""

    def test_equals_the_closed_form_geometric_mean(self) -> None:
        components = [np.array([0.5]), np.array([0.8]), np.array([0.2])]
        got = log_space_geometric_mean(components, (1 / 3, 1 / 3, 1 / 3))
        assert got[0] == pytest.approx((0.5 * 0.8 * 0.2) ** (1 / 3))

    def test_is_not_the_arithmetic_mean(self) -> None:
        """The listed fail condition, asserted directly."""
        components = [np.array([1.0]), np.array([1.0]), np.array([0.1])]
        got = log_space_geometric_mean(components, (1 / 3, 1 / 3, 1 / 3))
        assert got[0] == pytest.approx(0.1 ** (1 / 3))
        assert got[0] != pytest.approx((1.0 + 1.0 + 0.1) / 3)

    @pytest.mark.parametrize("position", [0, 1, 2])
    def test_zero_dominance(self, position: int) -> None:
        components = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
        components[position] = np.array([0.0])
        assert log_space_geometric_mean(components, (1 / 3, 1 / 3, 1 / 3))[0] == 0.0

    def test_zero_weight_paired_with_zero_component_is_not_nan(self) -> None:
        """0 * log(0) is NaN; the mask is what stops it reaching the output."""
        components = [np.array([0.0]), np.array([0.9]), np.array([0.9])]
        got = log_space_geometric_mean(components, (0.0, 0.5, 0.5))
        assert np.isfinite(got).all()
        assert got[0] == 0.0

    def test_ranking_survives_extreme_underflow(self) -> None:
        """Direct multiplication would flatten these to 0 and destroy the order."""
        components = [
            np.array([1e-200, 1e-250]),
            np.array([1e-200, 1e-250]),
            np.array([1e-200, 1e-250]),
        ]
        got = log_space_geometric_mean(components, (1 / 3, 1 / 3, 1 / 3))
        assert got[0] > got[1] > 0.0

    def test_non_finite_component_annihilates(self) -> None:
        components = [np.array([np.nan]), np.array([1.0]), np.array([1.0])]
        got = log_space_geometric_mean(components, (1 / 3, 1 / 3, 1 / 3))
        assert got[0] == 0.0

    def test_undefined_components_are_dropped_and_weights_renormalised(self) -> None:
        components = [np.array([0.25]), np.array([1.0]), np.array([1.0])]
        defined = [np.array([True]), np.array([False]), np.array([False])]
        got = log_space_geometric_mean(
            components, (1 / 3, 1 / 3, 1 / 3), defined=defined
        )
        assert got[0] == pytest.approx(0.25)

    def test_no_defined_component_scores_zero(self) -> None:
        components = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
        defined = [np.array([False])] * 3
        got = log_space_geometric_mean(
            components, (1 / 3, 1 / 3, 1 / 3), defined=defined
        )
        assert got[0] == 0.0

    def test_result_is_always_bounded(self) -> None:
        rng = np.random.default_rng(0)
        components = [rng.random(500) for _ in range(3)]
        got = log_space_geometric_mean(components, (0.2, 0.3, 0.5))
        assert np.all((got >= 0.0) & (got <= 1.0))
        assert np.isfinite(got).all()

    def test_empty_input(self) -> None:
        got = log_space_geometric_mean([np.array([])] * 3, (1 / 3, 1 / 3, 1 / 3))
        assert len(got) == 0

    @pytest.mark.parametrize(
        "weights", [(0.5, 0.5, 0.5), (-0.5, 0.75, 0.75), (0.5, 0.5)]
    )
    def test_malformed_weights_are_rejected(self, weights: Any) -> None:
        with pytest.raises(ValueError):
            log_space_geometric_mean([np.array([1.0])] * 3, weights)

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            log_space_geometric_mean(
                [np.array([1.0]), np.array([1.0, 1.0]), np.array([1.0])],
                (1 / 3, 1 / 3, 1 / 3),
            )


class TestConfigValidation:
    """A malformed calibration must fail at construction, not at scoring time."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"weights": (0.5, 0.5, 0.5)},
            {"weights": (-1.0, 1.0, 1.0)},
            {"weights": (0.5, 0.5)},
            {"low_threshold": 0.9, "high_threshold": 0.1},
            {"histogram_bins": 0},
            {"completeness_credit": {"present": 2.0}},
        ],
    )
    def test_invalid_config_is_rejected(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            ConfidenceConfig(**kwargs)

    def test_custom_weights_are_honoured(self) -> None:
        model = ConfidenceModel(weights=(0.5, 0.25, 0.25))
        assert model.config.weights == (0.5, 0.25, 0.25)


# ---------------------------------------------------------------------------
# sec.8.1 - mandated functional tests
# ---------------------------------------------------------------------------


class TestFunctional:
    """Stage2.md sec.8.1, one test per listed case."""

    def test_perfect_record_scores_above_point_nine(self) -> None:
        scores = score_one()
        assert scores["completeness"] == pytest.approx(1.0)
        assert scores["temporal"] == pytest.approx(1.0)
        assert scores["confidence"] > 0.9

    def test_fully_missing_record_scores_zero(self) -> None:
        blank = {name: None for name in FIELD_ORDER}
        frame = pd.DataFrame(
            [blank] + [dict(BASE_ROW, work_id=f"W{i}") for i in range(5)],
            columns=list(FIELD_ORDER),
        ).astype(object)
        scores = ConfidenceModel().score(Corpus.from_dataframe(frame)).scores
        assert scores.iloc[0] == 0.0

    def test_missing_fields_lower_completeness(self) -> None:
        assert score_one(vendor_name=None, district=None, status=None)[
            "completeness"
        ] < score_one()["completeness"]

    def test_invalid_dates_zero_the_whole_score(self) -> None:
        scores = score_one(date_approval="not a date")
        assert scores["temporal"] == 0.0
        assert scores["confidence"] == 0.0

    def test_mismatched_amounts_lower_reconciliation(self) -> None:
        mismatched = score_one(sanction_amount=1_000_000.0, amount_spent=50_000.0)
        assert mismatched["reconciliation"] < 0.3
        assert mismatched["confidence"] < score_one()["confidence"]

    def test_mixed_defects_land_in_between(self) -> None:
        clean = score_one()["confidence"]
        mixed = score_one(vendor_name="N/A", amount_spent=400_000.0)["confidence"]
        broken = score_one(date_approval="not a date")["confidence"]
        assert broken < mixed < clean

    def test_per_record_output_matches_the_prd_shape(self) -> None:
        records = ConfidenceModel().score(corpus_of(None, None)).to_records()
        assert set(records[0]) == {"confidence", "components"}
        assert set(records[0]["components"]) == {
            "completeness",
            "temporal",
            "reconciliation",
        }


# ---------------------------------------------------------------------------
# sec.9 - mandatory edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Stage2.md sec.9."""

    def test_empty_corpus(self) -> None:
        empty = pd.DataFrame({name: pd.Series([], dtype="object") for name in FIELD_ORDER})
        outcome = ConfidenceModel().score(Corpus.from_dataframe(empty))
        assert len(outcome) == 0
        assert outcome.report.n_records == 0
        assert outcome.report.mean_confidence == 0.0
        json.dumps(outcome.report.to_dict(), default=str)

    def test_single_record_corpus(self) -> None:
        outcome = ConfidenceModel().score(corpus_of(None))
        assert len(outcome) == 1
        assert 0.0 <= float(outcome.scores.iloc[0]) <= 1.0

    def test_all_fields_missing(self) -> None:
        blank = {name: None for name in FIELD_ORDER}
        corpus = Corpus.from_dataframe(
            pd.DataFrame([blank], columns=list(FIELD_ORDER)).astype(object)
        )
        assert ConfidenceModel().score(corpus).scores.iloc[0] == 0.0

    def test_all_dates_invalid(self) -> None:
        outcome = ConfidenceModel().score(
            corpus_of(
                {
                    "date_proposal": "not a date",
                    "date_approval": "31/02/2020",
                    "date_completion": "20200-01-01",
                },
                None,
            )
        )
        assert outcome.temporal.scores.iloc[0] == 0.0
        assert outcome.scores.iloc[0] == 0.0

    def test_all_null_column(self) -> None:
        frame = make_frame(*[None] * 10)
        frame["vendor_name"] = None
        outcome = ConfidenceModel().score(Corpus.from_dataframe(frame))
        assert np.isfinite(outcome.scores).all()
        assert outcome.scores.between(0.0, 1.0).all()

    def test_extremely_large_values(self) -> None:
        outcome = ConfidenceModel().score(
            corpus_of({"sanction_amount": 1e300, "amount_spent": 1e-300}, None)
        )
        assert np.isfinite(outcome.scores).all()
        assert outcome.scores.between(0.0, 1.0).all()

    def test_infinity_never_reaches_the_output(self) -> None:
        outcome = ConfidenceModel().score(
            corpus_of({"sanction_amount": "1.2e400"}, None)
        )
        assert np.isfinite(outcome.scores).all()
        assert outcome.scores.iloc[0] == 0.0

    def test_identical_values_no_variance(self) -> None:
        frame = make_frame(*[None] * 6, vary=False)
        frame["work_id"] = "MPL-KA-2019-000001"
        outcome = ConfidenceModel().score(Corpus.from_dataframe(frame))
        assert np.isfinite(outcome.scores).all()
        assert outcome.scores.nunique() == 1

    def test_division_by_zero_is_impossible(self) -> None:
        outcome = ConfidenceModel().score(
            corpus_of({"sanction_amount": 0.0, "amount_spent": 0.0}, None)
        )
        assert np.isfinite(outcome.scores).all()

    def test_no_warnings_are_emitted_on_dirty_data(self, corpus: Corpus) -> None:
        """A RuntimeWarning here means an inf or NaN was computed somewhere."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ConfidenceModel().score(corpus)


# ---------------------------------------------------------------------------
# sec.8.2 - sanity checks
# ---------------------------------------------------------------------------


class TestDistribution:
    """Stage2.md sec.8.2: range within [0,1], distribution not collapsed."""

    def test_range_is_within_zero_one(self, result: Any) -> None:
        assert float(result.scores.min()) >= 0.0
        assert float(result.scores.max()) <= 1.0

    def test_distribution_is_not_collapsed(self, result: Any) -> None:
        assert result.scores.nunique() > 100
        assert float(result.scores.std()) > 0.05

    def test_both_tails_are_populated(self, result: Any) -> None:
        assert result.report.low_confidence_pct > 0.0
        assert result.report.high_confidence_pct > 0.0

    def test_synthetic_validation_clean_beats_dirty(self, corpus: Corpus, result: Any) -> None:
        """Stage2.md sec.8.3, checked against Stage 1's own defect signals."""
        breakdown = result.breakdown
        hard_failed = breakdown.loc[breakdown["temporal_hard_fail"], CONFIDENCE_COLUMN]
        clean = breakdown.loc[
            (breakdown["completeness"] >= 0.999) & (breakdown["temporal"] >= 1.0),
            CONFIDENCE_COLUMN,
        ]
        assert float(hard_failed.mean()) == 0.0
        assert float(clean.mean()) > 0.8

    def test_more_defects_means_lower_confidence(self, corpus: Corpus, result: Any) -> None:
        n_null = corpus.records[list(FIELD_ORDER)].isna().sum(axis=1)
        correlation = np.corrcoef(n_null.to_numpy(), result.scores.to_numpy())[0, 1]
        assert correlation < -0.3, correlation

    def test_histogram_covers_every_record(self, result: Any) -> None:
        assert sum(result.report.histogram.values()) == len(result)

    def test_report_exposes_the_prd_keys(self, result: Any) -> None:
        view = result.report.prd_view()
        assert set(view) == {
            "mean_confidence",
            "low_confidence_pct",
            "high_confidence_pct",
        }

    def test_summary_frame_renders(self, result: Any) -> None:
        table = confidence_summary_frame(result)
        assert list(table.index) == [
            "confidence",
            "completeness",
            "temporal",
            "reconciliation",
        ]


# ---------------------------------------------------------------------------
# sec.7 - determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Stage2.md sec.7: same input, same output."""

    def test_repeated_scoring_is_identical(self, corpus: Corpus) -> None:
        first = ConfidenceModel().score(corpus)
        second = ConfidenceModel().score(corpus)
        pd.testing.assert_series_equal(first.scores, second.scores)
        assert first.report.to_dict() == second.report.to_dict()

    def test_functional_api_matches_the_model(self, corpus: Corpus) -> None:
        pd.testing.assert_series_equal(
            compute_confidence(corpus.records), ConfidenceModel().score(corpus).scores
        )

    def test_no_dependence_on_global_random_state(self, corpus: Corpus) -> None:
        np.random.seed(0)
        first = compute_confidence(corpus.records)
        np.random.seed(12345)
        _ = np.random.random(100)
        second = compute_confidence(corpus.records)
        pd.testing.assert_series_equal(first, second)

    def test_row_order_does_not_change_a_record_score(self, corpus: Corpus) -> None:
        """v_f is corpus-level, so a permutation must not move any score."""
        head = corpus.records.head(2000)
        weights = compute_field_weights(head)
        straight = compute_completeness(head, weights=weights)
        shuffled_frame = head.iloc[::-1]
        shuffled = compute_completeness(shuffled_frame, weights=weights)
        pd.testing.assert_series_equal(
            straight.sort_index(), shuffled.sort_index()
        )

    def test_reports_serialise_stably(self, result: Any, tmp_path: Path) -> None:
        written = result.save_reports(tmp_path)
        assert set(written) == {"confidence_report", "field_weights"}
        for path in written.values():
            json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Stage2.md sec.6 / the brief's Step 6: attach to the Corpus."""

    def test_attaches_the_confidence_column(self) -> None:
        corpus = corpus_of(*[None] * 6)
        attach_confidence(corpus)
        assert CONFIDENCE_COLUMN in corpus.records.columns
        assert corpus.records[CONFIDENCE_COLUMN].between(0.0, 1.0).all()

    def test_attaches_the_component_columns(self) -> None:
        corpus = corpus_of(*[None] * 6)
        attach_confidence(corpus)
        for name in ("completeness", "temporal", "reconciliation"):
            assert name in corpus.records.columns

    def test_row_order_and_index_are_preserved(self, corpus: Corpus) -> None:
        before_index = corpus.records.index.copy()
        before_ids = corpus.records["work_id"].copy()
        attach_confidence(corpus)
        assert corpus.records.index.equals(before_index)
        pd.testing.assert_series_equal(corpus.records["work_id"], before_ids)

    def test_no_row_is_added_or_lost(self, corpus: Corpus) -> None:
        before = len(corpus)
        attach_confidence(corpus)
        assert len(corpus) == before

    def test_attachment_is_aligned_record_by_record(self) -> None:
        """Guards against the silent off-by-one that would misattribute scores."""
        corpus = corpus_of(
            {"date_approval": "not a date"}, None, {"date_approval": "not a date"}
        )
        attach_confidence(corpus)
        column = corpus.records[CONFIDENCE_COLUMN]
        assert column.iloc[0] == 0.0
        assert column.iloc[1] > 0.0
        assert column.iloc[2] == 0.0

    def test_misaligned_result_is_rejected(self) -> None:
        corpus = corpus_of(*[None] * 4)
        outcome = ConfidenceModel().score(corpus_of(*[None] * 2))
        with pytest.raises(ValueError, match="length"):
            attach_confidence(corpus, outcome)

    def test_model_accepts_a_bare_frame(self, corpus: Corpus) -> None:
        pd.testing.assert_series_equal(
            ConfidenceModel().score(corpus.records).scores,
            ConfidenceModel().score(corpus).scores,
        )

    def test_model_rejects_nonsense_input(self) -> None:
        with pytest.raises(TypeError):
            ConfidenceModel().score([1, 2, 3])  # type: ignore[arg-type]

    def test_breakdown_exposes_the_evidence_base(self, result: Any) -> None:
        breakdown = result.breakdown
        for column in (
            CONFIDENCE_COLUMN,
            "completeness",
            "temporal",
            "reconciliation",
            "temporal_pairs_evaluated",
            "reconciliation_branch",
        ):
            assert column in breakdown.columns
        assert len(breakdown) == len(result)


# ---------------------------------------------------------------------------
# sec.7 - performance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Stage2.md sec.7: 50k records scored in under 3 seconds."""

    def test_fifty_thousand_records_within_budget(self) -> None:
        scoring_corpus = Corpus.from_dataframe(
            generate_dataset(n=PERFORMANCE_ROW_BUDGET, seed=42)
        )
        model = ConfidenceModel()
        started = time.perf_counter()
        outcome = model.score(scoring_corpus)
        elapsed = time.perf_counter() - started
        assert len(outcome) == PERFORMANCE_ROW_BUDGET
        assert elapsed < CONFIDENCE_SECONDS_BUDGET, (
            f"scoring took {elapsed:.2f}s, budget is {CONFIDENCE_SECONDS_BUDGET}s"
        )


# ---------------------------------------------------------------------------
# sec.11 - acceptance criteria
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """One test per checkbox in Stage2.md sec.11."""

    def test_confidence_computed_for_all_records(self, corpus: Corpus, result: Any) -> None:
        assert len(result.scores) == len(corpus)
        assert result.scores.notna().all()

    def test_component_scores_are_available(self, result: Any) -> None:
        for component in (result.completeness, result.temporal, result.reconciliation):
            assert len(component.scores) == len(result)
            assert component.scores.between(0.0, 1.0).all()

    def test_edge_cases_handled_correctly(self, result: Any) -> None:
        assert np.isfinite(result.scores).all()
        assert result.scores.between(0.0, 1.0).all()

    def test_outputs_are_stable_and_interpretable(self, result: Any) -> None:
        """Every component must be independently explainable for one record."""
        breakdown = result.breakdown
        row = breakdown.iloc[0]
        assert set(("completeness", "temporal", "reconciliation")).issubset(breakdown.columns)
        assert 0.0 <= float(row["completeness"]) <= 1.0
        assert result.field_weights.total > 0

    def test_works_on_the_synthetic_dataset(self, result: Any) -> None:
        assert result.report.n_records == 10_000
        assert 0.0 < result.report.mean_confidence < 1.0

    def test_ready_for_stage_three(self, corpus: Corpus) -> None:
        """Stage 3 needs (R, C) pairs on an intact, ordered corpus."""
        attach_confidence(corpus)
        assert CONFIDENCE_COLUMN in corpus.records.columns
        assert len(corpus.records) == corpus.validation_report.total_records
