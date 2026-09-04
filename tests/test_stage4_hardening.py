"""Stage 4 hardening - measurement, exposure and contract completion.

This pass added no behaviour. These tests therefore fall into two kinds:

* **Instrumentation** tests, which check that the new measurements are correct
  and are computed only over defined values.
* **Non-breaking** tests, which check that adding them changed nothing - the
  pre-existing columns must be byte-identical with every measurement pass
  turned on and off.

The second kind matters more. A hardening pass that quietly moved a decision
would be worse than no hardening pass at all.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    CALIBRATION_QUANTILES,
    CLUSTER_NOISE_REASON,
    DECISION_CLASSES,
    DUPLICATE_DIAGNOSTIC_THRESHOLDS,
    DUPLICATE_REACHABLE_THRESHOLD,
    DUPLICATE_SIMILARITY_THRESHOLD,
    SEVERITY_DEFINED_REASONS,
    STAGE4_CALIBRATION_REPORT,
    STAGE4_DUPLICATE_DIAGNOSTICS,
)
from src.stage4.calibration import (
    DEVIATION_COLUMNS,
    compute_duplicate_diagnostics,
    compute_stage4_calibration_report,
    describe_defined,
)
from src.stage4.decision import severity_definedness
from src.stage4.pipeline import (
    STAGE4_COLUMNS,
    AnomalyConfig,
    AnomalyLayer,
)

from tests.test_stage4 import ALL_UNDEFINED, BASELINE, make_frame, undefined

#: The Stage 4 contract as it stood BEFORE this pass. Nothing here may move.
PRE_HARDENING_COLUMNS: List[str] = [
    "anomaly_types",
    "anomaly_count",
    "severity_score",
    "decision_class",
    "decision_reason",
    "z_cost",
    "cost_scope",
    "z_spend",
    "z_duration",
    "valid_signal_count",
    "confidence_flag",
    "duplicate_flag_stage4",
    "explanation_text",
]

#: Columns this pass added.
ADDED_COLUMNS: List[str] = ["severity_defined", "severity_defined_reason"]


def run_one(**overrides: Any) -> pd.Series:
    """Run the layer over a single record and return its output row."""
    return AnomalyLayer().run(make_frame(dict(overrides))).frame.iloc[0]


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """A frame covering every severity-definedness reason."""
    return make_frame(
        {},                                                        # ok
        {"deviation_cell_cost": 6.0},                              # ok, extreme
        dict(ALL_UNDEFINED),                                       # all missing
        {                                                          # noise cluster
            "cluster_id": -1,
            "cluster_label": "unclustered",
            "cluster_has_norm": False,
            **{k: v for k, v in ALL_UNDEFINED.items()},
        },
        {                                                          # no peer norm
            "peer_cell_stable": False,
            **undefined("cell_cost", "cell_unstable"),
            **undefined("cluster_cost", "cell_unstable"),
            **undefined("spend_ratio", "cell_unstable"),
            **undefined("duration", "cell_unstable"),
        },
        {"confidence": 0.1},                                       # gated
        {"duplicate_score": 0.95, "duplicate_flag": True},         # duplicate
    )


@pytest.fixture(scope="module")
def scored(spread: pd.DataFrame) -> pd.DataFrame:
    """The Stage 4 output for ``spread``, joined with its inputs."""
    output = AnomalyLayer().run(spread).frame
    return output.join(spread[[c for c in spread.columns if c not in output.columns]])


# ---------------------------------------------------------------------------
# FIX 1 - calibration instrumentation
# ---------------------------------------------------------------------------


class TestDescribeDefined:
    """The primitive every distribution rests on."""

    def test_statistics_ignore_undefined_entries(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan])
        summary = describe_defined(values)
        assert summary["count_defined"] == 3
        assert summary["count_total"] == 5
        assert summary["mean"] == pytest.approx(2.0)

    def test_nan_never_pollutes_a_quantile(self) -> None:
        """A NaN-laden column must give the same answer as its clean subset."""
        clean = pd.Series([1.0, 2.0, 3.0, 4.0])
        dirty = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0, np.nan, 4.0])
        for key, value in describe_defined(clean).items():
            if key.startswith("p") or key in {"mean", "std"}:
                assert describe_defined(dirty)[key] == value

    def test_infinities_are_excluded(self) -> None:
        """An overflow is not a measurement."""
        summary = describe_defined(pd.Series([1.0, 2.0, np.inf, -np.inf]))
        assert summary["count_defined"] == 2
        assert np.isfinite(summary["mean"])

    def test_an_empty_series_reports_none_not_zero(self) -> None:
        """0.0 would read as a measured value; None reads as unmeasured."""
        summary = describe_defined(pd.Series([], dtype="float64"))
        assert summary["count_defined"] == 0
        assert summary["mean"] is None
        assert summary["p95"] is None

    def test_an_all_nan_series_reports_none(self) -> None:
        summary = describe_defined(pd.Series([np.nan, np.nan]))
        assert summary["count_defined"] == 0
        assert summary["mean"] is None

    def test_a_single_observation_has_no_dispersion_estimate(self) -> None:
        """Sample std of one value is undefined, not zero."""
        assert describe_defined(pd.Series([5.0]))["std"] is None

    def test_every_requested_quantile_is_reported(self) -> None:
        summary = describe_defined(pd.Series(np.arange(100.0)))
        for quantile in CALIBRATION_QUANTILES:
            assert f"p{int(round(quantile * 100))}" in summary

    def test_quantiles_are_monotone(self) -> None:
        summary = describe_defined(pd.Series(np.random.default_rng(0).normal(size=500)))
        values = [summary[f"p{int(round(q * 100))}"] for q in CALIBRATION_QUANTILES]
        assert values == sorted(values)


class TestCalibrationReport:
    """Descriptive, complete, and inert."""

    def test_it_reports_every_deviation_column(self, scored: pd.DataFrame) -> None:
        report = compute_stage4_calibration_report(scored)
        assert set(report["deviations"]) == set(DEVIATION_COLUMNS)

    def test_deviation_counts_match_the_defined_entries(
        self, scored: pd.DataFrame
    ) -> None:
        report = compute_stage4_calibration_report(scored)
        for column in DEVIATION_COLUMNS:
            expected = int(scored[column].notna().sum())
            assert report["deviations"][column]["count_defined"] == expected

    def test_it_reports_every_decision_class(self, scored: pd.DataFrame) -> None:
        report = compute_stage4_calibration_report(scored)
        for name in DECISION_CLASSES:
            assert name in report["decisions"]

    def test_decision_percentages_sum_to_one_hundred(
        self, scored: pd.DataFrame
    ) -> None:
        report = compute_stage4_calibration_report(scored)
        total = sum(report["decisions"][name]["pct"] for name in DECISION_CLASSES)
        # Percentages are rounded to 4 dp for byte-determinism, so the sum can
        # sit up to half a unit in the last place away from 100 per class.
        assert total == pytest.approx(100.0, abs=1e-3)

    def test_the_brief_aliases_are_provided(self, scored: pd.DataFrame) -> None:
        """The brief names spend_anomaly / temporal_anomaly; both are served."""
        activation = compute_stage4_calibration_report(scored)["signal_activation"]
        assert "spend_anomaly" in activation["_aggregates"]
        assert "temporal_anomaly" in activation["_aggregates"]
        assert "overspend_anomaly" in activation
        assert "underspend_anomaly" in activation

    def test_the_spend_alias_is_the_union_of_both_directions(
        self, scored: pd.DataFrame
    ) -> None:
        activation = compute_stage4_calibration_report(scored)["signal_activation"]
        over = activation["overspend_anomaly"]["count"]
        under = activation["underspend_anomaly"]["count"]
        alias = activation["_aggregates"]["spend_anomaly"]["count"]
        assert max(over, under) <= alias <= over + under

    def test_coverage_metrics_are_reported(self, scored: pd.DataFrame) -> None:
        coverage = compute_stage4_calibration_report(scored)["coverage"]
        for key in ("no_peer_norm", "cluster_noise", "missing_features"):
            assert key in coverage
            assert 0 <= coverage[key]["pct"] <= 100

    def test_severity_section_excludes_undefined_records(
        self, scored: pd.DataFrame
    ) -> None:
        report = compute_stage4_calibration_report(scored)
        assert report["severity"]["count_defined"] == int(
            scored["severity_score"].notna().sum()
        )

    def test_it_states_that_it_defines_no_threshold(
        self, scored: pd.DataFrame
    ) -> None:
        report = compute_stage4_calibration_report(scored)
        assert "threshold" in report["_note"].lower()
        assert report["thresholds_in_force"]["_status"].startswith("judgements")

    def test_it_cannot_change_a_decision(self, spread: pd.DataFrame) -> None:
        """The report is computed from the output; running it changes nothing."""
        first = AnomalyLayer(AnomalyConfig(compute_calibration=False)).run(spread)
        second = AnomalyLayer(AnomalyConfig(compute_calibration=True)).run(spread)
        pd.testing.assert_frame_equal(first.frame, second.frame)

    def test_it_does_not_mutate_the_frame(self, scored: pd.DataFrame) -> None:
        before = scored.copy(deep=True)
        compute_stage4_calibration_report(scored)
        pd.testing.assert_frame_equal(scored, before)

    def test_it_survives_a_frame_without_stage_four(
        self, spread: pd.DataFrame
    ) -> None:
        """A partial report must announce itself, not look complete."""
        report = compute_stage4_calibration_report(spread)
        assert "unavailable" in report["severity"]
        assert "unavailable" in report["decisions"]

    def test_it_survives_an_empty_frame(self, spread: pd.DataFrame) -> None:
        report = compute_stage4_calibration_report(spread.iloc[0:0])
        assert report["n_records"] == 0

    def test_it_is_json_serialisable(self, scored: pd.DataFrame) -> None:
        blob = json.dumps(compute_stage4_calibration_report(scored))
        assert "NaN" not in blob

    def test_it_is_deterministic(self, scored: pd.DataFrame) -> None:
        first = json.dumps(compute_stage4_calibration_report(scored), sort_keys=True)
        second = json.dumps(compute_stage4_calibration_report(scored), sort_keys=True)
        assert first == second


# ---------------------------------------------------------------------------
# FIX 2 - explicit severity definedness
# ---------------------------------------------------------------------------


class TestSeverityDefinedness:
    """No severity is now an explicit statement, not an implicit gap."""

    def test_the_columns_are_in_the_contract(self) -> None:
        for column in ADDED_COLUMNS:
            assert column in STAGE4_COLUMNS

    def test_undefined_severity_is_always_nan(self, scored: pd.DataFrame) -> None:
        undefined_rows = ~scored["severity_defined"].to_numpy(dtype=bool)
        assert scored.loc[undefined_rows, "severity_score"].isna().all()

    def test_defined_severity_is_never_nan(self, scored: pd.DataFrame) -> None:
        defined_rows = scored["severity_defined"].to_numpy(dtype=bool)
        assert scored.loc[defined_rows, "severity_score"].notna().all()

    def test_the_flag_agrees_exactly_with_the_score(
        self, scored: pd.DataFrame
    ) -> None:
        assert scored["severity_defined"].equals(scored["severity_score"].notna())

    def test_only_declared_reasons_are_emitted(self, scored: pd.DataFrame) -> None:
        assert set(scored["severity_defined_reason"]) <= set(SEVERITY_DEFINED_REASONS)

    def test_a_defined_severity_reads_ok(self) -> None:
        row = run_one()
        assert row["severity_defined"]
        assert row["severity_defined_reason"] == "ok"

    def test_missing_inputs_read_as_insufficient_features(self) -> None:
        row = run_one(**ALL_UNDEFINED)
        assert not row["severity_defined"]
        assert row["severity_defined_reason"] == "insufficient_features"

    def test_a_noise_cluster_reads_as_cluster_noise(self) -> None:
        row = run_one(
            cluster_id=-1,
            cluster_label="unclustered",
            cluster_has_norm=False,
            **{
                **undefined("cell_cost", CLUSTER_NOISE_REASON),
                **undefined("cluster_cost", CLUSTER_NOISE_REASON),
                **undefined("spend_ratio", CLUSTER_NOISE_REASON),
                **undefined("duration", CLUSTER_NOISE_REASON),
            },
        )
        assert row["severity_defined_reason"] == "cluster_noise"

    def test_an_unstable_cell_reads_as_no_peer_norm(self) -> None:
        row = run_one(
            peer_cell_stable=False,
            **undefined("cell_cost", "cell_unstable"),
            **undefined("cluster_cost", "cell_unstable"),
            **undefined("spend_ratio", "cell_unstable"),
            **undefined("duration", "cell_unstable"),
        )
        assert row["severity_defined_reason"] == "no_peer_norm"

    def test_zero_dispersion_also_reads_as_no_peer_norm(self) -> None:
        """A norm with no scale is not a norm you can deviate from."""
        row = run_one(
            **undefined("cell_cost", "zero_dispersion"),
            **undefined("cluster_cost", "zero_dispersion"),
            **undefined("spend_ratio", "zero_dispersion"),
            **undefined("duration", "zero_dispersion"),
        )
        assert row["severity_defined_reason"] == "no_peer_norm"

    def test_noise_outranks_the_other_reasons(self) -> None:
        """A record with no work type has no peer structure to be missing from."""
        row = run_one(
            cluster_has_norm=False,
            **undefined("cell_cost", "cell_unstable"),
            **undefined("cluster_cost", "cell_unstable"),
            **undefined("spend_ratio", FEATURE_MISSING := "feature_missing"),
            **undefined("duration", "feature_missing"),
        )
        assert row["severity_defined_reason"] == "cluster_noise"

    @pytest.mark.parametrize("reason", ["ok", "cluster_noise", "insufficient_features",
                                        "no_peer_norm"])
    def test_each_reason_is_reachable(self, reason: str) -> None:
        cases = {
            "ok": {},
            "insufficient_features": dict(ALL_UNDEFINED),
            "cluster_noise": {
                "cluster_has_norm": False,
                **{
                    **undefined("cell_cost", CLUSTER_NOISE_REASON),
                    **undefined("cluster_cost", CLUSTER_NOISE_REASON),
                    **undefined("spend_ratio", CLUSTER_NOISE_REASON),
                    **undefined("duration", CLUSTER_NOISE_REASON),
                },
            },
            "no_peer_norm": {
                **undefined("cell_cost", "cell_unstable"),
                **undefined("cluster_cost", "cell_unstable"),
                **undefined("spend_ratio", "cell_unstable"),
                **undefined("duration", "cell_unstable"),
            },
        }
        assert run_one(**cases[reason])["severity_defined_reason"] == reason

    def test_a_mixed_cause_falls_through_to_no_valid_deviation(self) -> None:
        """The catch-all exists so no record is ever left without a reason."""
        frame = make_frame({
            **undefined("cell_cost", "some_future_reason"),
            **undefined("cluster_cost", "some_future_reason"),
            **undefined("spend_ratio", "some_future_reason"),
            **undefined("duration", "some_future_reason"),
        })
        row = AnomalyLayer().run(frame).frame.iloc[0]
        assert row["severity_defined_reason"] == "no_valid_deviation"

    def test_the_rule_does_not_diverge_from_the_score(
        self, spread: pd.DataFrame
    ) -> None:
        result = AnomalyLayer().run(spread)
        assert result.definedness.rule_divergence == 0

    def test_divergence_is_reported_rather_than_resolved(self) -> None:
        """A contradictory input must not silently blank an existing score.

        A record with a defined deviation but no cluster norm cannot come out of
        Stage 3. If one ever does, the severity stays exactly as computed and
        the divergence is counted - the alternative would change an output.
        """
        frame = make_frame({"cluster_has_norm": False})
        result = AnomalyLayer().run(frame)
        assert result.definedness.rule_divergence == 1
        assert result.frame["severity_score"].notna().all()
        assert result.frame["severity_defined"].all()

    def test_definedness_appears_in_the_report(self, spread: pd.DataFrame) -> None:
        report = AnomalyLayer().run(spread).report()
        assert "definedness" in report["severity"]
        assert "by_reason" in report["severity"]["definedness"]


# ---------------------------------------------------------------------------
# FIX 3 - duplicate signal instrumentation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDuplicateDiagnostics:
    """The detector is unchanged; what it can see is now measurable."""

    @pytest.fixture(scope="class")
    def corpus(self) -> Any:
        from src.stage1.corpus import Corpus
        from src.stage1.data_generator import generate_dataset
        from src.stage2.confidence import attach_confidence
        from src.stage3.pipeline import attach_structure

        built = Corpus.from_dataframe(generate_dataset(n=4000, seed=42))
        attach_confidence(built)
        attach_structure(built)
        return built

    @pytest.fixture(scope="class")
    def diagnostics(self, corpus: Any) -> Any:
        return compute_duplicate_diagnostics(corpus.records)

    def test_the_threshold_constant_is_used_not_hardcoded(self) -> None:
        import src.stage3.duplicate_detection as detection

        assert detection.DUPLICATE_SIMILARITY_THRESHOLD == 0.85
        source = Path("src/stage3/duplicate_detection.py").read_text(encoding="utf-8")
        assert "0.85" not in source, "a literal threshold bypasses the constant"

    def test_reachable_threshold_is_below_detection(self) -> None:
        assert DUPLICATE_REACHABLE_THRESHOLD < DUPLICATE_SIMILARITY_THRESHOLD

    def test_pair_counts_are_monotone_in_the_threshold(
        self, diagnostics: Any
    ) -> None:
        """A higher bar can never admit more pairs."""
        counts = [
            diagnostics.summary["pairs_at_or_above"][f"{t:.2f}"]
            for t in sorted(DUPLICATE_DIAGNOSTIC_THRESHOLDS)
        ]
        assert counts == sorted(counts, reverse=True)

    def test_reachable_pairs_rate_is_a_proportion(self, diagnostics: Any) -> None:
        rate = diagnostics.summary["reachable_pairs_rate"]
        assert 0.0 <= rate <= 1.0

    def test_reachable_pairs_rate_matches_the_lowest_cut(
        self, diagnostics: Any
    ) -> None:
        lowest = min(DUPLICATE_DIAGNOSTIC_THRESHOLDS)
        expected = (
            diagnostics.summary["pairs_at_or_above"][f"{lowest:.2f}"]
            / diagnostics.summary["n_candidate_pairs"]
        )
        assert diagnostics.summary["reachable_pairs_rate"] == pytest.approx(
            expected, abs=1e-6
        )

    def test_cosine_is_bounded(self, diagnostics: Any) -> None:
        values = diagnostics.best_cosine
        assert bool(((values >= 0.0) & (values <= 1.0)).all())

    def test_reachable_is_the_cosine_above_the_cut(self, diagnostics: Any) -> None:
        expected = diagnostics.best_cosine >= DUPLICATE_REACHABLE_THRESHOLD
        assert diagnostics.reachable.equals(expected.rename("duplicate_reachable"))

    def test_every_flagged_duplicate_is_reachable(
        self, corpus: Any, diagnostics: Any
    ) -> None:
        """The invariant the brief requires. It holds because the decay term
        lies in [0,1], so the blended score can never exceed its own cosine."""
        flagged = corpus.records["duplicate_flag"].to_numpy(dtype=bool)
        assert bool(diagnostics.reachable.to_numpy()[flagged].all())

    def test_reachable_is_a_superset_of_flagged(
        self, corpus: Any, diagnostics: Any
    ) -> None:
        assert int(diagnostics.reachable.sum()) >= int(
            corpus.records["duplicate_flag"].sum()
        )

    def test_detection_is_not_altered(self, corpus: Any, diagnostics: Any) -> None:
        """Running diagnostics must not change a single flag."""
        before = corpus.records["duplicate_flag"].copy()
        compute_duplicate_diagnostics(corpus.records)
        pd.testing.assert_series_equal(corpus.records["duplicate_flag"], before)

    def test_decay_attenuation_is_quantified(self, diagnostics: Any) -> None:
        """Text failure and time failure must be separable."""
        decay = diagnostics.summary["decay_attenuation"]
        assert decay["pairs_lost_to_decay"] <= decay["pairs_above_cosine_threshold"]

    def test_it_is_deterministic(self, corpus: Any) -> None:
        first = compute_duplicate_diagnostics(corpus.records)
        second = compute_duplicate_diagnostics(corpus.records)
        pd.testing.assert_series_equal(first.best_cosine, second.best_cosine)
        assert first.summary == second.summary

    def test_it_rejects_misaligned_vectors(self, corpus: Any) -> None:
        from scipy import sparse

        bad = sparse.csr_matrix((len(corpus.records) - 1, 5))
        with pytest.raises(ValueError, match="rows"):
            compute_duplicate_diagnostics(corpus.records, record_vectors=bad)

    def test_it_is_json_serialisable(self, diagnostics: Any) -> None:
        json.dumps(diagnostics.to_dict())


# ---------------------------------------------------------------------------
# Non-breaking guarantees
# ---------------------------------------------------------------------------


class TestNothingChanged:
    """The whole point of the pass."""

    def test_pre_hardening_columns_are_byte_identical(
        self, spread: pd.DataFrame
    ) -> None:
        plain = AnomalyLayer(AnomalyConfig(compute_calibration=False)).run(spread)
        full = AnomalyLayer(
            AnomalyConfig(compute_calibration=True, compute_duplicate_diagnostics=False)
        ).run(spread)
        for column in PRE_HARDENING_COLUMNS:
            assert plain.frame[column].equals(full.frame[column]), column

    def test_the_contract_was_extended_not_reordered(self) -> None:
        """Existing consumers index by position in some places; order holds."""
        positions = [STAGE4_COLUMNS.index(name) for name in PRE_HARDENING_COLUMNS]
        assert positions == sorted(positions)

    def test_severity_scores_are_unchanged_by_instrumentation(
        self, spread: pd.DataFrame
    ) -> None:
        plain = AnomalyLayer(AnomalyConfig(compute_calibration=False)).run(spread)
        full = AnomalyLayer(AnomalyConfig(compute_calibration=True)).run(spread)
        pd.testing.assert_series_equal(
            plain.frame["severity_score"], full.frame["severity_score"]
        )

    def test_decisions_are_unchanged_by_instrumentation(
        self, spread: pd.DataFrame
    ) -> None:
        plain = AnomalyLayer(AnomalyConfig(compute_calibration=False)).run(spread)
        full = AnomalyLayer(AnomalyConfig(compute_calibration=True)).run(spread)
        pd.testing.assert_series_equal(
            plain.frame["decision_class"], full.frame["decision_class"]
        )

    def test_row_order_is_preserved(self, spread: pd.DataFrame) -> None:
        frame = spread.copy()
        frame.index = pd.Index(range(900, 900 + len(frame)), name="record")
        output = AnomalyLayer().run(frame).frame
        assert list(output.index) == list(frame.index)

    def test_determinism_is_preserved(self, spread: pd.DataFrame) -> None:
        first = AnomalyLayer().run(spread)
        second = AnomalyLayer().run(spread)
        pd.testing.assert_frame_equal(first.frame, second.frame)
        assert json.dumps(first.report(), sort_keys=True, default=str) == json.dumps(
            second.report(), sort_keys=True, default=str
        )

    def test_reports_still_carry_no_wall_clock(self, spread: pd.DataFrame) -> None:
        result = AnomalyLayer(AnomalyConfig(compute_calibration=True)).run(spread)
        blob = json.dumps(result.report(), default=str) + json.dumps(
            result.calibration, default=str
        )
        assert "elapsed" not in blob and "timestamp" not in blob

    def test_calibration_is_off_the_critical_path(self, spread: pd.DataFrame) -> None:
        """It runs after every decision is final."""
        result = AnomalyLayer(AnomalyConfig(compute_calibration=False)).run(spread)
        assert result.calibration is None
        assert result.duplicates is None
        assert result.definedness is not None

    def test_reports_are_written_when_enabled(
        self, spread: pd.DataFrame, tmp_path: Path
    ) -> None:
        result = AnomalyLayer(AnomalyConfig(compute_calibration=True)).run(spread)
        written = result.save_reports(tmp_path)
        assert written["calibration"].name == STAGE4_CALIBRATION_REPORT
        loaded = json.loads(written["calibration"].read_text(encoding="utf-8"))
        assert loaded["n_records"] == len(spread)

    def test_no_report_is_written_when_disabled(
        self, spread: pd.DataFrame, tmp_path: Path
    ) -> None:
        result = AnomalyLayer(AnomalyConfig(compute_calibration=False)).run(spread)
        written = result.save_reports(tmp_path)
        assert "calibration" not in written
        assert "duplicate_diagnostics" not in written

    def test_the_config_echoes_the_new_flags(self) -> None:
        config = AnomalyConfig(compute_duplicate_diagnostics=True)
        assert config.to_dict()["compute_duplicate_diagnostics"] is True
