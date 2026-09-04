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
    ALLOWED_STATUS,
    CLUSTER_PENALTY_DELTA,
    CONFIDENCE_SECONDS_BUDGET,
    CONFIDENCE_WEIGHTS,
    CONFIDENCE_WEIGHTS_V1,
    CRITICAL_FIELDS,
    FIELD_CRITICALITY,
    FIELD_ORDER,
    PERFORMANCE_ROW_BUDGET,
    RECON_LAMBDA,
    RECON_NON_POSITIVE_SANCTION_CREDIT,
    RECON_ONE_SIDED_CREDIT,
    RECON_OVERSPEND_TOLERANCE,
    RECON_PRE_COMPLETION_STATUSES,
    RECON_TERMINAL_STATUSES,
    RECON_UNDERSPEND_FLOOR,
    RECON_UNKNOWN_STATUS_GAMMA_SCALE,
    RECON_UNDERSPEND_GAMMA,
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
    BREAKDOWN_COLUMNS,
    CONFIDENCE_COLUMN,
    ConfidenceConfig,
    ConfidenceModel,
    attach_confidence,
    compute_confidence,
    confidence_summary_frame,
    explain_confidence,
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


def legacy_v1_config() -> ConfidenceConfig:
    """The exact v1 calibration, so the refinement stays reversible and testable."""
    return ConfidenceConfig(
        weights=CONFIDENCE_WEIGHTS_V1,
        recon_mode="agreement",
        completeness_weight_mode="entropy",
        cluster_delta=0.0,
        one_sided_credit=0.2,
        non_finite_credit=0.0,
        non_positive_sanction_credit=1.0,
    )


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
        """v2: r = 0.8 sits inside the normal band, so no penalty applies."""
        score = score_one(sanction_amount=1000.0, amount_spent=800.0)["reconciliation"]
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_overspend_matches_the_closed_form(self) -> None:
        """The penalty is measured from 1 + tau, not from 1."""
        score = score_one(sanction_amount=1000.0, amount_spent=1500.0)["reconciliation"]
        expected = float(np.exp(-RECON_LAMBDA * (0.5 - RECON_OVERSPEND_TOLERANCE)))
        assert score == pytest.approx(expected, rel=1e-5)

    def test_underspend_matches_the_closed_form(self) -> None:
        score = score_one(sanction_amount=1000.0, amount_spent=100.0)["reconciliation"]
        expected = float(np.exp(-RECON_UNDERSPEND_GAMMA * (RECON_UNDERSPEND_FLOOR - 0.1)))
        assert score == pytest.approx(expected, rel=1e-5)

    def test_larger_mismatch_scores_lower(self) -> None:
        close = score_one(sanction_amount=1000.0, amount_spent=950.0)["reconciliation"]
        far = score_one(sanction_amount=1000.0, amount_spent=100.0)["reconciliation"]
        assert close > far

    def test_routine_underspend_is_no_longer_penalised(self) -> None:
        """The v1 semantic error, pinned down.

        74.57% of comparable records sat in this band and were charged a mean
        penalty of 0.8875 for executing their budget correctly.
        """
        for spent in (950.0, 900.0, 800.0, 700.0, 500.0, 300.0):
            score = score_one(sanction_amount=1000.0, amount_spent=spent)
            assert score["reconciliation"] == pytest.approx(1.0, abs=1e-6), spent

    def test_overspend_and_underspend_are_treated_asymmetrically(self) -> None:
        """v1 was symmetric and could not tell these apart. They are not alike."""
        over = score_one(sanction_amount=1000.0, amount_spent=1300.0)["reconciliation"]
        under = score_one(sanction_amount=1000.0, amount_spent=700.0)["reconciliation"]
        assert over < under
        assert under == pytest.approx(1.0, abs=1e-6)

    def test_overspend_decays_monotonically(self) -> None:
        scores = [
            score_one(sanction_amount=1000.0, amount_spent=spent)["reconciliation"]
            for spent in (1000.0, 1200.0, 1500.0, 2000.0, 3000.0)
        ]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0, abs=1e-6)
        assert scores[-1] < 0.05

    def test_underspend_decays_monotonically_below_the_floor(self) -> None:
        scores = [
            score_one(sanction_amount=1000.0, amount_spent=spent)["reconciliation"]
            for spent in (200.0, 150.0, 100.0, 50.0, 0.0)
        ]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0, abs=1e-6)

    def test_total_underspend_lands_in_the_strong_penalty_tier(self) -> None:
        score = score_one(sanction_amount=1000.0, amount_spent=0.0)["reconciliation"]
        assert score == pytest.approx(
            float(np.exp(-RECON_UNDERSPEND_GAMMA * RECON_UNDERSPEND_FLOOR)), rel=1e-6
        )
        assert score < 0.35

    def test_score_is_scale_free(self) -> None:
        """Scale-free up to epsilon, which perturbs the ratio at tiny magnitudes."""
        small = score_one(sanction_amount=100.0, amount_spent=80.0)["reconciliation"]
        large = score_one(sanction_amount=100_000_000.0, amount_spent=80_000_000.0)
        assert small == pytest.approx(large["reconciliation"], rel=1e-7)

    def test_scores_stay_bounded_on_real_data(self, result: Any) -> None:
        scores = result.reconciliation.scores
        assert scores.between(0.0, 1.0).all()
        assert np.isfinite(scores).all()

    def test_reported_floor_matches_the_underspend_limit(self) -> None:
        outcome = compute_reconciliation_result(corpus_of(None, None).records)
        assert outcome.diagnostics["theoretical_floor"] == pytest.approx(
            float(np.exp(-RECON_UNDERSPEND_GAMMA * RECON_UNDERSPEND_FLOOR)), rel=1e-5
        )

    def test_zero_sanction_is_implausible_not_perfect(self) -> None:
        """BEHAVIOUR CHANGE from v1, and a deliberate one.

        Under v1's equality reading, 0 == 0 was perfect agreement and scored
        1.0. Under a plausibility reading there is no budget to have executed
        against, so the ratio is meaningless and the record is penalised.
        Still no division by zero: the branch is taken before any arithmetic.
        """
        outcome = compute_reconciliation_result(
            corpus_of({"sanction_amount": 0.0, "amount_spent": 0.0}, None).records
        )
        assert outcome.scores.iloc[0] == pytest.approx(
            RECON_NON_POSITIVE_SANCTION_CREDIT
        )
        assert outcome.branch.iloc[0] == "non_positive_sanction"
        assert np.isfinite(outcome.scores).all()

    def test_negative_sanction_takes_the_same_branch(self) -> None:
        outcome = compute_reconciliation_result(
            corpus_of({"sanction_amount": -500.0}, None).records
        )
        assert outcome.branch.iloc[0] == "non_positive_sanction"

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

    def test_infinite_amount_is_refused(self) -> None:
        """inf/inf is NaN; left unhandled it would poison the log-sum silently.

        CORRECTION (audit finding 3): back to an outright refusal. The v2
        interim value of 0.25 was moderate enough to survive aggregation, so a
        record whose amount was literally infinite could still produce
        respectable confidence. Garbage is refused, not discounted.
        """
        outcome = compute_reconciliation_result(
            corpus_of({"sanction_amount": "1.2e400"}, None).records
        )
        assert outcome.branch.iloc[0] == "non_finite"
        assert outcome.scores.iloc[0] == 0.0
        assert np.isfinite(outcome.scores).all()

    def test_extreme_finite_amount_is_refused(self) -> None:
        """CORRECTION: 1e300 is finite, so it slipped past the non-finite check.

        Stage 1 already flags it VALUE_IMPLAUSIBLE_MAGNITUDE. A ratio of
        1e300/1e300 = 1.0 previously scored a perfect reconciliation on two
        numbers that are data-entry accidents.
        """
        outcome = compute_reconciliation_result(
            corpus_of({"sanction_amount": 1e300, "amount_spent": 1e300}, None).records
        )
        assert outcome.branch.iloc[0] == "implausible_magnitude"
        assert outcome.scores.iloc[0] == 0.0
        assert np.isfinite(outcome.scores).all()

    def test_negative_amounts_stay_bounded(self) -> None:
        outcome = compute_reconciliation_result(
            corpus_of({"amount_spent": -812345.5}, None).records
        )
        assert 0.0 <= float(outcome.scores.iloc[0]) <= 1.0

    def test_max_normalisation_is_available_in_agreement_mode(self) -> None:
        frame = corpus_of({"sanction_amount": 1000.0, "amount_spent": 800.0}, None).records
        by_max = compute_reconciliation(frame, mode="agreement", normalization="max")
        assert by_max.iloc[0] == pytest.approx(np.exp(-RECON_LAMBDA * 200.0 / 1000.0))

    def test_max_normalisation_is_not_sign_safe(self) -> None:
        """Documents exactly why symmetric is the default within agreement mode."""
        frame = corpus_of({"sanction_amount": -1000.0, "amount_spent": -500.0}, None).records
        symmetric = compute_reconciliation(
            frame, mode="agreement", normalization="symmetric"
        )
        by_max = compute_reconciliation(frame, mode="agreement", normalization="max")
        assert symmetric.iloc[0] > 0.5
        assert by_max.iloc[0] == pytest.approx(0.0)

    def test_agreement_mode_reproduces_v1_exactly(self) -> None:
        """The refinement must be reversible, not merely documented."""
        frame = corpus_of({"sanction_amount": 1000.0, "amount_spent": 800.0}, None).records
        legacy = compute_reconciliation(frame, mode="agreement")
        assert legacy.iloc[0] == pytest.approx(
            float(np.exp(-RECON_LAMBDA * (200.0 / 1800.0))), rel=1e-6
        )

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            compute_reconciliation(corpus_of(None, None).records, mode="bogus")

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
        """v2 reads "mismatch" as implausibility, so the mismatch must be real.

        A 5% execution rate is implausible; a 20% underspend is not, and no
        longer counts as a mismatch at all.
        """
        mismatched = score_one(sanction_amount=1_000_000.0, amount_spent=50_000.0)
        assert mismatched["reconciliation"] < 0.5
        assert mismatched["confidence"] < score_one()["confidence"]

        overspent = score_one(sanction_amount=1_000_000.0, amount_spent=2_500_000.0)
        assert overspent["reconciliation"] < 0.1

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
        """Infinity must not propagate, and must still be penalised hard."""
        outcome = ConfidenceModel().score(
            corpus_of({"sanction_amount": "1.2e400"}, None)
        )
        assert np.isfinite(outcome.scores).all()
        assert outcome.scores.between(0.0, 1.0).all()
        assert outcome.reconciliation.scores.iloc[0] < 0.3
        assert outcome.scores.iloc[0] < outcome.scores.iloc[1]

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


# ---------------------------------------------------------------------------
# v2 refinement - the three corrections and the evidence for each
# ---------------------------------------------------------------------------


class TestCriticalityWeighting:
    """Refinement 3: fix the C_comp structural floor."""

    def test_critical_fields_now_dominate_the_basis(self, corpus: Corpus) -> None:
        """v1 gave the evidentiary spine 30.56% of weight. It should have most."""
        weights = compute_field_weights(corpus.records)
        critical_share = sum(weights.shares[name] for name in CRITICAL_FIELDS)
        assert critical_share > 0.65, critical_share

    def test_the_identifier_no_longer_dominates(self, corpus: Corpus) -> None:
        """work_id is never null, so its weight is a constant added to everyone.

        At 18.11% under v1 it was the single largest term in the whole score
        while carrying no discriminating power whatsoever.
        """
        weights = compute_field_weights(corpus.records)
        assert weights.shares["work_id"] < 0.06
        for name in CRITICAL_FIELDS:
            assert weights.weights[name] > weights.weights["work_id"], name

    def test_structural_floor_is_much_lower_than_v1(self, corpus: Corpus) -> None:
        criticality = compute_field_weights(corpus.records)
        entropy = compute_field_weights(corpus.records, weight_mode="entropy")
        assert criticality.structural_floor < 0.06
        assert criticality.structural_floor < entropy.structural_floor / 3

    def test_completeness_spread_is_wider_than_v1(self, corpus: Corpus) -> None:
        """v1: min 0.5150, sd 0.0670 - very nearly a constant."""
        refined = compute_completeness(corpus.records)
        legacy = compute_completeness(
            corpus.records, weight_mode="entropy", cluster_delta=0.0
        )
        assert refined.std() > 2 * legacy.std()
        assert refined.min() < legacy.min() / 2

    def test_entropy_mode_reproduces_v1_weights(self, corpus: Corpus) -> None:
        """The refinement must be reversible, not merely documented."""
        legacy = compute_field_weights(corpus.records, weight_mode="entropy")
        assert legacy.weight_mode == "entropy"
        assert legacy.shares["work_id"] > 0.15

    def test_hybrid_mode_is_available(self, corpus: Corpus) -> None:
        hybrid = compute_field_weights(corpus.records, weight_mode="hybrid")
        assert hybrid.weight_mode == "hybrid"
        assert hybrid.total > 0

    def test_unknown_weight_mode_is_rejected(self, corpus: Corpus) -> None:
        with pytest.raises(ValueError, match="weight_mode"):
            compute_field_weights(corpus.records, weight_mode="bogus")

    def test_criticality_is_reported_for_audit(self, corpus: Corpus) -> None:
        payload = compute_field_weights(corpus.records).to_dict()
        assert payload["weight_mode"] == "criticality"
        assert payload["per_field"]["sanction_amount"]["criticality"] == pytest.approx(
            FIELD_CRITICALITY["sanction_amount"]
        )


class TestClusterPenalty:
    """Refinement 3b: evidence loss is super-additive."""

    def test_losing_more_critical_fields_decays_faster_than_linearly(self) -> None:
        losses = [
            "date_proposal",
            "date_approval",
            "date_completion",
            "sanction_amount",
            "amount_spent",
        ]
        scores = []
        for count in range(len(losses) + 1):
            overrides = {name: None for name in losses[:count]}
            corpus = corpus_of(overrides, *([None] * 12))
            outcome = compute_completeness_result(corpus.records)
            scores.append(float(outcome.scores.iloc[0]))
        assert scores == sorted(scores, reverse=True)
        first_drop = scores[0] - scores[1]
        last_drop = scores[3] - scores[4]
        assert last_drop > first_drop, (first_drop, last_drop)

    def test_all_critical_fields_missing_gives_low_completeness(self) -> None:
        overrides = {name: None for name in CRITICAL_FIELDS}
        corpus = corpus_of(overrides, *([None] * 12))
        outcome = compute_completeness_result(corpus.records)
        assert float(outcome.scores.iloc[0]) < 0.3
        assert float(outcome.cluster_factor.iloc[0]) < 0.5

    def test_a_single_missing_critical_field_is_not_cluster_penalised(self) -> None:
        corpus = corpus_of({"date_completion": None}, *([None] * 12))
        outcome = compute_completeness_result(corpus.records)
        assert float(outcome.cluster_factor.iloc[0]) == pytest.approx(1.0)

    def test_non_critical_losses_do_not_trigger_the_cluster_penalty(self) -> None:
        corpus = corpus_of(
            {
                "vendor_name": None,
                "district": None,
                "state": None,
                "implementing_agency": None,
            },
            *([None] * 12),
        )
        outcome = compute_completeness_result(corpus.records)
        assert float(outcome.cluster_factor.iloc[0]) == pytest.approx(1.0)

    def test_cluster_deficit_preserves_null_reason_ordering(self) -> None:
        """The deficit is fractional, built from the same credit vector."""
        results = {}
        cases = {
            "missing": {
                "date_proposal": None,
                "date_approval": None,
                "sanction_amount": None,
            },
            "placeholder": {
                "date_proposal": "N/A",
                "date_approval": "N/A",
                "sanction_amount": "N/A",
            },
            "unparseable": {
                "date_proposal": "not a date",
                "date_approval": "not a date",
                "sanction_amount": "abcd",
            },
        }
        for label, overrides in cases.items():
            corpus = corpus_of(overrides, *([None] * 12))
            outcome = compute_completeness_result(corpus.records)
            results[label] = (
                float(outcome.critical_deficit.iloc[0]),
                float(outcome.cluster_factor.iloc[0]),
            )
        assert (
            results["missing"][0] < results["placeholder"][0] < results["unparseable"][0]
        )
        assert (
            results["missing"][1] > results["placeholder"][1] > results["unparseable"][1]
        )

    def test_cluster_penalty_can_be_disabled(self) -> None:
        overrides = {name: None for name in CRITICAL_FIELDS}
        corpus = corpus_of(overrides, *([None] * 12))
        without = compute_completeness_result(corpus.records, cluster_delta=0.0)
        with_penalty = compute_completeness_result(corpus.records)
        assert without.scores.iloc[0] > with_penalty.scores.iloc[0]
        assert float(without.cluster_factor.iloc[0]) == pytest.approx(1.0)

    def test_cluster_factor_never_leaves_zero_one(self, result: Any) -> None:
        factor = result.completeness.cluster_factor
        assert factor.between(0.0, 1.0).all()
        assert np.isfinite(factor).all()

    @pytest.mark.parametrize(
        "kwargs", [{"cluster_delta": -1.0}, {"cluster_allowance": -1.0}]
    )
    def test_invalid_cluster_config_is_rejected(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            ConfidenceConfig(**kwargs)


class TestComponentBalance:
    """Refinement 2: no single component may dominate the output."""

    @staticmethod
    def _penalty_shares(result: Any, weights: Any) -> Dict[str, float]:
        """Share of total mean penalty (in nats) carried by each component.

        This, rather than variance share, is the meaningful dominance metric:
        it answers "on a typical record, which component is taking the
        confidence away". Variance in log space is dominated by a small tail of
        severe temporal violations and says little about typical behaviour.
        """
        breakdown = result.breakdown
        alive = (
            (breakdown["completeness"] > 0)
            & (breakdown["temporal"] > 0)
            & (breakdown["reconciliation"] > 0)
        )
        penalties = {
            name: float((-np.log(breakdown[name][alive]) * weight).mean())
            for name, weight in zip(
                ("completeness", "temporal", "reconciliation"), weights
            )
        }
        total = sum(penalties.values())
        return {name: value / total for name, value in penalties.items()}

    def test_no_component_takes_most_of_the_penalty(self, result: Any) -> None:
        shares = self._penalty_shares(result, CONFIDENCE_WEIGHTS)
        assert max(shares.values()) < 0.55, shares

    def test_every_component_carries_real_weight(self, result: Any) -> None:
        shares = self._penalty_shares(result, CONFIDENCE_WEIGHTS)
        assert min(shares.values()) > 0.10, shares

    def test_v2_is_better_balanced_than_v1(self, corpus: Corpus) -> None:
        """v1's reconciliation carried 69.4% of all penalty mass."""
        legacy = ConfidenceModel(config=legacy_v1_config()).score(corpus)
        refined = ConfidenceModel().score(corpus)
        v1_max = max(self._penalty_shares(legacy, CONFIDENCE_WEIGHTS_V1).values())
        v2_max = max(self._penalty_shares(refined, CONFIDENCE_WEIGHTS).values())
        assert v2_max < v1_max, (v1_max, v2_max)

    def test_the_one_sided_artefact_spike_is_gone(self, result: Any) -> None:
        """v1 pinned 4,255 records into [0.5,0.6); 96.1% were the one_null branch."""
        scores = result.scores
        spike = int(((scores >= 0.5) & (scores < 0.6)).sum())
        assert spike / len(scores) < 0.05, spike

    def test_weights_are_the_rebalanced_values(self) -> None:
        assert ConfidenceModel().config.weights == (0.4, 0.4, 0.2)
        assert sum(CONFIDENCE_WEIGHTS) == pytest.approx(1.0)

    def test_missing_financials_take_only_a_partial_penalty(self) -> None:
        """v1 capped these records at 0.585 however sound the rest of them was."""
        scores = score_one(amount_spent=None)
        assert scores["reconciliation"] == pytest.approx(RECON_ONE_SIDED_CREDIT)
        assert scores["confidence"] > 0.8


class TestRefinementInvariants:
    """Everything the refinement was forbidden to break."""

    def test_public_api_is_unchanged(self, corpus: Corpus) -> None:
        frame = corpus.records
        for function in (
            compute_completeness,
            compute_temporal,
            compute_reconciliation,
            compute_confidence,
        ):
            series = function(frame)
            assert isinstance(series, pd.Series)
            assert len(series) == len(frame)
            assert series.index.equals(frame.index)

    def test_zero_dominance_still_holds(self) -> None:
        scores = score_one(date_approval="not a date")
        assert scores["temporal"] == 0.0
        assert scores["confidence"] == 0.0

    def test_all_missing_still_scores_zero(self) -> None:
        blank = {name: None for name in FIELD_ORDER}
        frame = pd.DataFrame(
            [blank] + [dict(BASE_ROW, work_id=f"W{i}") for i in range(5)],
            columns=list(FIELD_ORDER),
        ).astype(object)
        scores = ConfidenceModel().score(Corpus.from_dataframe(frame)).scores
        assert scores.iloc[0] == 0.0

    def test_perfect_record_still_scores_high(self) -> None:
        assert score_one()["confidence"] > 0.9

    def test_null_reason_ordering_still_holds(self) -> None:
        clean = score_one()["completeness"]
        missing = score_one(date_completion=None)["completeness"]
        placeholder = score_one(date_completion="N/A")["completeness"]
        unparseable = score_one(date_completion="not a date")["completeness"]
        assert clean > missing > placeholder > unparseable

    def test_outputs_remain_bounded_and_finite(self, result: Any) -> None:
        for series in (
            result.scores,
            result.completeness.scores,
            result.temporal.scores,
            result.reconciliation.scores,
        ):
            assert series.between(0.0, 1.0).all()
            assert np.isfinite(series).all()

    def test_still_deterministic(self, corpus: Corpus) -> None:
        first = ConfidenceModel().score(corpus)
        second = ConfidenceModel().score(corpus)
        pd.testing.assert_series_equal(first.scores, second.scores)
        assert first.report.to_dict() == second.report.to_dict()

    def test_no_imputation_occurred(self, corpus: Corpus) -> None:
        snapshot = corpus.records[list(FIELD_ORDER)].copy(deep=True)
        ConfidenceModel().score(corpus)
        pd.testing.assert_frame_equal(snapshot, corpus.records[list(FIELD_ORDER)])

    def test_log_space_aggregation_is_intact(self) -> None:
        components = [np.array([0.5]), np.array([0.8]), np.array([0.2])]
        got = log_space_geometric_mean(components, CONFIDENCE_WEIGHTS)
        expected = 0.5**0.4 * 0.8**0.4 * 0.2**0.2
        assert got[0] == pytest.approx(expected)

    def test_distribution_did_not_collapse(self, result: Any) -> None:
        assert result.scores.nunique() > 1000
        assert float(result.scores.std()) > 0.15

    def test_v1_behaviour_is_fully_recoverable(self, corpus: Corpus) -> None:
        """A refinement you cannot roll back is not a refinement."""
        legacy = ConfidenceModel(config=legacy_v1_config()).score(corpus)
        assert legacy.reconciliation.scores.max() <= 1.0
        assert legacy.completeness.scores.min() > 0.4
        assert legacy.field_weights.shares["work_id"] > 0.15

    def test_ground_truth_separation_is_preserved(
        self, corpus: Corpus, result: Any
    ) -> None:
        """Injected defects must still push confidence down, in severity order."""
        n_null = corpus.records[list(FIELD_ORDER)].isna().sum(axis=1)
        grouped = result.scores.groupby(n_null.clip(upper=4)).mean()
        assert list(grouped) == sorted(grouped, reverse=True), grouped.to_dict()


# ---------------------------------------------------------------------------
# Final corrections - the three audit findings
# ---------------------------------------------------------------------------


def _recon_for(status: Optional[str], spent: float, sanction: float = 1000.0) -> float:
    """Reconciliation score for one record at a given lifecycle stage."""
    corpus = corpus_of(
        {"status": status, "sanction_amount": sanction, "amount_spent": spent},
        *([None] * 12),
    )
    return float(compute_reconciliation(corpus.records).iloc[0])


class TestLifecycleAwareness:
    """Audit finding 1: a proposed work with no spend was penalised for being normal."""

    @pytest.mark.parametrize("status", ["proposed", "approved", "pending"])
    def test_pre_completion_work_with_no_spend_is_not_penalised(
        self, status: str
    ) -> None:
        assert _recon_for(status, 0.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("status", ["proposed", "approved", "pending"])
    def test_pre_completion_work_with_low_spend_is_not_penalised(
        self, status: str
    ) -> None:
        for spent in (0.0, 50.0, 100.0, 150.0):
            assert _recon_for(status, spent) == pytest.approx(1.0), (status, spent)

    @pytest.mark.parametrize("status", ["completed", "closed"])
    def test_terminal_work_with_low_spend_is_penalised(self, status: str) -> None:
        assert _recon_for(status, 0.0) == pytest.approx(
            float(np.exp(-RECON_UNDERSPEND_GAMMA * RECON_UNDERSPEND_FLOOR)), rel=1e-5
        )
        assert _recon_for(status, 0.0) < 0.35

    def test_the_same_spend_is_scored_differently_by_stage(self) -> None:
        """The whole point of the fix, in one assertion."""
        assert _recon_for("proposed", 0.0) > _recon_for("completed", 0.0)

    @pytest.mark.parametrize("status", [None, "N/A", "half done", "not a date"])
    def test_unknown_status_takes_a_mild_penalty(self, status: Optional[str]) -> None:
        """Neither excused nor condemned: we do not know if the gap is legitimate."""
        unknown = _recon_for(status, 0.0)
        assert _recon_for("completed", 0.0) < unknown < _recon_for("proposed", 0.0)
        expected = float(
            np.exp(
                -RECON_UNDERSPEND_GAMMA
                * RECON_UNKNOWN_STATUS_GAMMA_SCALE
                * RECON_UNDERSPEND_FLOOR
            )
        )
        assert unknown == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize("status", ["proposed", "completed", None])
    def test_overspend_is_never_excused_by_lifecycle_stage(
        self, status: Optional[str]
    ) -> None:
        """Spending past the sanction is a control failure at any stage."""
        assert _recon_for(status, 2000.0) < 0.2

    def test_lifecycle_class_is_reported(self) -> None:
        corpus = corpus_of(
            {"status": "proposed"}, {"status": "completed"}, {"status": None}
        )
        outcome = compute_reconciliation_result(corpus.records)
        assert list(outcome.lifecycle) == ["pre_completion", "terminal", "unknown"]
        assert outcome.diagnostics["lifecycle_counts"]["pre_completion"] == 1

    def test_absent_status_column_does_not_crash(self) -> None:
        frame = pd.DataFrame(
            {"sanction_amount": [1000.0], "amount_spent": [0.0]}
        )
        outcome = compute_reconciliation_result(frame)
        assert list(outcome.lifecycle) == ["unknown"]
        assert 0.0 <= float(outcome.scores.iloc[0]) <= 1.0

    def test_status_gate_covers_stage_one_vocabulary(self) -> None:
        """Every status Stage 1 can emit must land in a known class."""
        for status in ALLOWED_STATUS:
            assert (
                status in RECON_PRE_COMPLETION_STATUSES
                or status in RECON_TERMINAL_STATUSES
            ), status

    def test_invalid_unknown_scale_is_rejected(self, corpus: Corpus) -> None:
        with pytest.raises(ValueError, match="unknown_status_gamma_scale"):
            compute_reconciliation(
                corpus.records.head(5), unknown_status_gamma_scale=2.0
            )


class TestOverspendTolerance:
    """Audit finding 2: rounding and price variation were charged as anomaly."""

    @pytest.mark.parametrize("ratio", [1.0, 1.01, 1.02, 1.04, 1.05])
    def test_overspend_inside_tolerance_is_free(self, ratio: float) -> None:
        assert _recon_for("completed", 1000.0 * ratio) == pytest.approx(1.0)

    @pytest.mark.parametrize("ratio", [1.06, 1.10, 1.20, 1.50])
    def test_overspend_beyond_tolerance_is_penalised(self, ratio: float) -> None:
        assert _recon_for("completed", 1000.0 * ratio) < 1.0

    def test_the_tolerance_boundary_is_exactly_where_it_should_be(self) -> None:
        assert _recon_for("completed", 1000.0 * (1.0 + RECON_OVERSPEND_TOLERANCE)) == (
            pytest.approx(1.0)
        )
        just_over = _recon_for("completed", 1000.0 * (1.0 + RECON_OVERSPEND_TOLERANCE) + 1.0)
        assert just_over < 1.0

    def test_penalty_is_measured_from_the_tolerance_edge(self) -> None:
        score = _recon_for("completed", 1200.0)
        expected = float(
            np.exp(-RECON_LAMBDA * (0.2 - RECON_OVERSPEND_TOLERANCE))
        )
        assert score == pytest.approx(expected, rel=1e-6)

    def test_large_overspend_is_still_punished_hard(self) -> None:
        assert _recon_for("completed", 3000.0) < 0.05

    def test_overspend_remains_monotonic(self) -> None:
        scores = [_recon_for("completed", s) for s in (1050.0, 1200.0, 1500.0, 3000.0)]
        assert scores == sorted(scores, reverse=True)

    def test_tolerance_can_be_disabled(self, corpus: Corpus) -> None:
        frame = corpus_of(
            {"status": "completed", "sanction_amount": 1000.0, "amount_spent": 1020.0},
            *([None] * 12),
        ).records
        assert compute_reconciliation(frame).iloc[0] == pytest.approx(1.0)
        strict = compute_reconciliation(frame, overspend_tolerance=0.0)
        assert strict.iloc[0] < 1.0

    def test_negative_tolerance_is_rejected(self, corpus: Corpus) -> None:
        with pytest.raises(ValueError, match="overspend_tolerance"):
            compute_reconciliation(corpus.records.head(5), overspend_tolerance=-0.1)


class TestGarbageRefusal:
    """Audit finding 3: garbage must not be able to produce moderate confidence."""

    def test_non_finite_amount_drives_confidence_to_zero(self) -> None:
        scores = score_one(sanction_amount="1.2e400")
        assert scores["reconciliation"] == 0.0
        assert scores["confidence"] == 0.0

    def test_implausible_magnitude_drives_confidence_to_zero(self) -> None:
        scores = score_one(sanction_amount=1e300)
        assert scores["reconciliation"] == 0.0
        assert scores["confidence"] == 0.0

    def test_zero_dominance_is_restored_for_garbage(self) -> None:
        """The interim 0.25 let an infinite amount survive aggregation."""
        corpus = corpus_of({"sanction_amount": "1.2e400"}, None)
        outcome = ConfidenceModel().score(corpus)
        assert outcome.scores.iloc[0] == 0.0
        assert outcome.scores.iloc[1] > 0.9

    def test_garbage_never_produces_nan_or_inf(self) -> None:
        corpus = corpus_of(
            {"sanction_amount": "1.2e400"},
            {"amount_spent": "1.2e400"},
            {"sanction_amount": 1e300, "amount_spent": 1e-300},
            {"sanction_amount": "abcd"},
            None,
        )
        outcome = ConfidenceModel().score(corpus)
        assert np.isfinite(outcome.scores).all()
        assert outcome.scores.between(0.0, 1.0).all()

    def test_amount_below_the_threshold_is_still_scored(self) -> None:
        """The refusal must not swallow legitimately large public works."""
        score = _recon_for("completed", 9.0e14, sanction=1.0e15)
        assert score > 0.9

    def test_refusal_branches_are_labelled_distinctly(self) -> None:
        corpus = corpus_of(
            {"sanction_amount": "1.2e400"}, {"sanction_amount": 1e300}, None
        )
        branches = list(compute_reconciliation_result(corpus.records).branch)
        assert branches[0] == "non_finite"
        assert branches[1] == "implausible_magnitude"
        assert branches[2] == "both_present"


class TestCorrectionInvariants:
    """What the corrections were forbidden to disturb."""

    def test_weights_are_untouched(self) -> None:
        assert ConfidenceModel().config.weights == (0.4, 0.4, 0.2)

    def test_completeness_logic_is_untouched(self, corpus: Corpus) -> None:
        weights = compute_field_weights(corpus.records)
        assert weights.weight_mode == "criticality"
        assert sum(weights.shares[name] for name in CRITICAL_FIELDS) > 0.65

    def test_temporal_logic_is_untouched(self) -> None:
        assert score_one(date_approval="not a date")["temporal"] == 0.0
        assert score_one()["temporal"] == pytest.approx(1.0)

    def test_log_space_aggregation_is_untouched(self) -> None:
        components = [np.array([0.5]), np.array([0.8]), np.array([0.2])]
        got = log_space_geometric_mean(components, CONFIDENCE_WEIGHTS)
        assert got[0] == pytest.approx(0.5**0.4 * 0.8**0.4 * 0.2**0.2)

    def test_still_deterministic(self, corpus: Corpus) -> None:
        first = ConfidenceModel().score(corpus)
        second = ConfidenceModel().score(corpus)
        pd.testing.assert_series_equal(first.scores, second.scores)
        assert first.report.to_dict() == second.report.to_dict()

    def test_public_api_is_unchanged(self, corpus: Corpus) -> None:
        for function in (
            compute_completeness,
            compute_temporal,
            compute_reconciliation,
            compute_confidence,
        ):
            series = function(corpus.records)
            assert isinstance(series, pd.Series)
            assert series.index.equals(corpus.records.index)

    def test_no_imputation_occurred(self, corpus: Corpus) -> None:
        snapshot = corpus.records[list(FIELD_ORDER)].copy(deep=True)
        ConfidenceModel().score(corpus)
        pd.testing.assert_frame_equal(snapshot, corpus.records[list(FIELD_ORDER)])

    def test_distribution_stays_reasonable(self, result: Any) -> None:
        report = result.report
        assert 0.70 <= report.mean_confidence <= 0.85
        assert result.scores.nunique() > 1000
        assert float(result.scores.std()) > 0.15

    def test_no_artificial_inflation(self, result: Any) -> None:
        """The tolerance must lift normal records without whitewashing bad ones."""
        assert result.report.zero_confidence_pct > 5.0
        assert result.report.low_confidence_pct > 5.0


# ---------------------------------------------------------------------------
# Finalization - downstream contract and explainability
# ---------------------------------------------------------------------------


class TestStage3Contract:
    """What Stage 3 is promised, asserted rather than documented."""

    def test_every_contract_column_is_attached(self, corpus: Corpus) -> None:
        attach_confidence(corpus)
        for column in BREAKDOWN_COLUMNS:
            assert column in corpus.records.columns, column

    def test_breakdown_and_attached_columns_agree(self, corpus: Corpus) -> None:
        """A consumer must see the same signals whichever surface it reads."""
        scored = attach_confidence(corpus)
        for column in BREAKDOWN_COLUMNS:
            pd.testing.assert_series_equal(
                corpus.records[column],
                scored.breakdown[column],
                check_names=False,
            )

    def test_contract_columns_are_row_aligned(self, corpus: Corpus) -> None:
        attach_confidence(corpus)
        for column in BREAKDOWN_COLUMNS:
            assert corpus.records[column].index.equals(corpus.records.index), column

    def test_contract_columns_are_serialisable(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        attach_confidence(corpus)
        path = tmp_path / "contract.csv"
        corpus.records.loc[:, list(BREAKDOWN_COLUMNS)].to_csv(path, index=False)
        assert len(pd.read_csv(path)) == len(corpus)

    def test_confidence_is_bounded_and_finite(self, corpus: Corpus) -> None:
        attach_confidence(corpus)
        column = corpus.records["confidence"]
        assert column.between(0.0, 1.0).all()
        assert np.isfinite(column).all()

    def test_zero_means_reject_not_merely_low(self, corpus: Corpus) -> None:
        """Every zero must be traceable to a refusal, never to arithmetic."""
        scored = attach_confidence(corpus)
        records = corpus.records
        zeros = records.loc[records["confidence"] == 0.0]
        assert len(zeros) > 0
        refused = (
            zeros["temporal_hard_fail"]
            | (zeros["completeness"] <= 0.0)
            | ((zeros["reconciliation"] <= 0.0) & zeros["reconciliation_defined"])
        )
        assert bool(refused.all())

    def test_monotonicity_against_ground_truth(self, corpus: Corpus) -> None:
        scored = ConfidenceModel().score(corpus)
        n_null = corpus.records[list(FIELD_ORDER)].isna().sum(axis=1)
        grouped = scored.scores.groupby(n_null.clip(upper=4)).mean()
        assert list(grouped) == sorted(grouped, reverse=True)

    def test_defined_mask_explains_component_count(self, corpus: Corpus) -> None:
        attach_confidence(corpus)
        records = corpus.records
        expected = (
            records["completeness_defined"].astype(int)
            + records["temporal_defined"].astype(int)
            + records["reconciliation_defined"].astype(int)
        )
        pd.testing.assert_series_equal(
            records["n_components_used"].astype(int), expected, check_names=False
        )
        assert (records["n_components_used"] >= 1).all()

    def test_dropped_components_are_visible_not_silent(self, corpus: Corpus) -> None:
        """"temporal = 1.0" must be separable from "nothing to check"."""
        attach_confidence(corpus)
        records = corpus.records
        unmeasured = records.loc[~records["temporal_defined"]]
        assert len(unmeasured) > 0
        assert (unmeasured["temporal_pairs_evaluated"] == 0).all()

    def test_critical_missing_count_is_an_integer_count(self, corpus: Corpus) -> None:
        """It must not be confused with the fractional deficit the formula uses."""
        attach_confidence(corpus)
        records = corpus.records
        counts = records["critical_missing_count"]
        assert (counts == counts.round()).all()
        assert counts.between(0, len(CRITICAL_FIELDS)).all()
        placeholders = records.loc[
            (records["critical_missing_count"] > 0)
            & (records["critical_deficit"] < records["critical_missing_count"])
        ]
        assert len(placeholders) > 0, "fractional deficit should differ from the count"

    def test_attach_is_idempotent(self, corpus: Corpus) -> None:
        first = attach_confidence(corpus)
        snapshot = corpus.records.loc[:, list(BREAKDOWN_COLUMNS)].copy(deep=True)
        attach_confidence(corpus)
        pd.testing.assert_frame_equal(
            snapshot, corpus.records.loc[:, list(BREAKDOWN_COLUMNS)]
        )

    def test_attach_adds_no_rows_and_reorders_nothing(self, corpus: Corpus) -> None:
        before_index = corpus.records.index.copy()
        before_ids = corpus.records["work_id"].copy()
        attach_confidence(corpus)
        assert corpus.records.index.equals(before_index)
        pd.testing.assert_series_equal(corpus.records["work_id"], before_ids)


class TestExplainConfidence:
    """The explanation contract: read-only, and it must agree with the score."""

    def test_explanation_matches_the_computed_values(self, corpus: Corpus) -> None:
        """The mandated test: explanation output matches computed values."""
        scored = attach_confidence(corpus)
        records = corpus.records
        for row in (0, 7, 500, int(scored.scores.idxmin()), int(scored.scores.idxmax())):
            explanation = explain_confidence(records, row)
            assert explanation["confidence"] == pytest.approx(
                float(scored.scores.loc[row]), abs=1e-6
            )
            for name in ("completeness", "temporal", "reconciliation"):
                assert explanation["components"][name]["score"] == pytest.approx(
                    float(records.loc[row, name]), abs=1e-6
                )
            assert explanation["components"]["completeness"]["evidence"][
                "critical_missing_count"
            ] == int(records.loc[row, "critical_missing_count"])
            assert explanation["components"]["reconciliation"]["evidence"][
                "lifecycle_state"
            ] == str(records.loc[row, "lifecycle_state"])

    def test_explanation_reconstructs_the_score(self, corpus: Corpus) -> None:
        """Effective weights and component scores must reproduce C exactly.

        This is the strongest possible check that the explanation describes the
        aggregation actually performed rather than a plausible story about it.
        """
        scored = attach_confidence(corpus)
        for row in (1, 42, 900, 9_000):
            explanation = explain_confidence(corpus.records, row)
            if explanation["confidence"] == 0.0:
                continue
            log_sum = sum(
                part["effective_weight"] * np.log(part["score"])
                for part in explanation["components"].values()
                if part["defined"] and part["score"] > 0
            )
            assert float(np.exp(log_sum)) == pytest.approx(
                explanation["confidence"], abs=1e-6
            ), row

    def test_result_explain_uses_its_own_weights(self, corpus: Corpus) -> None:
        scored = ConfidenceModel().score(corpus)
        explanation = scored.explain(0)
        assert explanation["aggregation"]["weights"] == {
            "completeness": pytest.approx(0.4),
            "temporal": pytest.approx(0.4),
            "reconciliation": pytest.approx(0.2),
        }

    def test_explanation_refuses_to_recompute(self) -> None:
        """Missing breakdown must raise, not silently score."""
        corpus = corpus_of(None, None)
        with pytest.raises(ValueError, match="never recomputes"):
            explain_confidence(corpus.records, 0)

    def test_unknown_row_raises(self, corpus: Corpus) -> None:
        attach_confidence(corpus)
        with pytest.raises(KeyError):
            explain_confidence(corpus.records, 10**9)

    def test_explanation_is_json_serialisable(self, corpus: Corpus) -> None:
        attach_confidence(corpus)
        payload = json.dumps(explain_confidence(corpus.records, 0), default=str)
        assert json.loads(payload)["verdict"] in {
            "refused",
            "low",
            "moderate",
            "high",
        }

    def test_refused_record_names_its_refusal(self) -> None:
        corpus = corpus_of({"date_approval": "not a date"}, *([None] * 6))
        attach_confidence(corpus)
        explanation = explain_confidence(corpus.records, 0)
        assert explanation["verdict"] == "refused"
        assert explanation["primary_driver"] == "temporal"
        assert explanation["components"]["temporal"]["refused"] is True
        assert any("REFUSED" in reason for reason in explanation["reasons"])

    def test_clean_record_reports_no_defect(self) -> None:
        corpus = corpus_of(None, *([None] * 12))
        attach_confidence(corpus)
        explanation = explain_confidence(corpus.records, 0)
        assert explanation["verdict"] == "high"
        assert explanation["reasons"] == [
            "No defect detected: every component was measurable and scored at "
            "or near its maximum."
        ]

    def test_lifecycle_exemption_is_explained(self) -> None:
        corpus = corpus_of(
            {"status": "proposed", "sanction_amount": 1000.0, "amount_spent": 0.0},
            *([None] * 12),
        )
        attach_confidence(corpus)
        explanation = explain_confidence(corpus.records, 0)
        assert explanation["components"]["reconciliation"]["evidence"][
            "lifecycle_state"
        ] == "pre_completion"
        assert any("pre-completion" in reason for reason in explanation["reasons"])

    def test_penalty_shares_sum_to_one(self, corpus: Corpus) -> None:
        attach_confidence(corpus)
        for row in (3, 77, 4_321):
            explanation = explain_confidence(corpus.records, row)
            shares = [
                part["share_of_penalty"]
                for part in explanation["components"].values()
                if "share_of_penalty" in part
            ]
            if shares:
                assert sum(shares) == pytest.approx(1.0, abs=1e-3)

    def test_primary_driver_is_the_largest_penalty(self, corpus: Corpus) -> None:
        attach_confidence(corpus)
        for row in (5, 250, 8_000):
            explanation = explain_confidence(corpus.records, row)
            if explanation["verdict"] == "refused":
                continue
            penalties = {
                name: part["penalty_nats"]
                for name, part in explanation["components"].items()
                if part["penalty_nats"] is not None
            }
            if penalties:
                assert explanation["primary_driver"] == max(
                    penalties, key=lambda key: penalties[key]
                )


class TestFinalizationInvariants:
    """The hardening pass must not have moved a single number."""

    def test_scores_are_unchanged_by_the_dedup(self, corpus: Corpus) -> None:
        """One shared null-reason pass must equal three independent ones."""
        reasons, _ = resolve_reasons(corpus.records, list(FIELD_ORDER))
        shared = compute_completeness_result(corpus.records).scores
        independent = compute_completeness(
            corpus.records,
            weights=compute_field_weights(corpus.records, reasons=reasons),
        )
        pd.testing.assert_series_equal(shared, independent)

    def test_still_deterministic(self, corpus: Corpus) -> None:
        first = ConfidenceModel().score(corpus)
        second = ConfidenceModel().score(corpus)
        pd.testing.assert_series_equal(first.scores, second.scores)
        assert first.report.to_dict() == second.report.to_dict()

    def test_weights_and_constants_untouched(self) -> None:
        assert ConfidenceModel().config.weights == (0.4, 0.4, 0.2)
        assert RECON_OVERSPEND_TOLERANCE == 0.05
        assert RECON_UNDERSPEND_FLOOR == 0.2
        assert CLUSTER_PENALTY_DELTA == 0.35

    def test_log_space_aggregation_untouched(self) -> None:
        components = [np.array([0.5]), np.array([0.8]), np.array([0.2])]
        got = log_space_geometric_mean(components, CONFIDENCE_WEIGHTS)
        assert got[0] == pytest.approx(0.5**0.4 * 0.8**0.4 * 0.2**0.2)

    def test_no_contract_column_was_removed(self) -> None:
        """Guards the downstream promise: columns are added, never dropped."""
        for column in (
            "confidence",
            "completeness",
            "temporal",
            "reconciliation",
            "temporal_pairs_evaluated",
        ):
            assert column in BREAKDOWN_COLUMNS
