"""Stage 5 hardening - stability, explanation exactness, diagnostics.

This pass changed no score. Its tests therefore prove two kinds of thing:

* that the composition survives inputs chosen to break it, and
* that what the system *says* about itself is exactly what it *does*.

The centrepiece is `TestExplanationRoundTrip`, which never reads a component
column. It parses the numbers back out of the English sentence and re-multiplies
them. If the narrative and the arithmetic ever diverge, that test fails - and it
is the only test in the suite that would catch a narrative drifting from a score
it no longer describes.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    CONTRIBUTION_FLAG_THRESHOLD_PCT,
    MIN_CONFIDENCE_FOR_RISK,
    R_HIGH,
    R_LOW,
    RISK_FLAGS,
    RISK_NOT_A_THRESHOLD_NOTE,
    UNCERTAINTY_COMPONENT_CLASS,
)
from src.stage5.calibration import compute_contribution_analysis
from src.stage5.pipeline import RiskConfig, RiskLayer

from tests.test_stage5 import NO_SEVERITY, make_frame, run_one

#: The sentence every scored explanation opens with.
COMPOSITION = re.compile(
    r"Risk (?P<risk>[\d.]+) \((?P<band>[a-z ]+)\), composed as "
    r"signal (?P<signal>[\d.]+) x data quality (?P<quality>[\d.]+) "
    r"x stability (?P<stability>[\d.]+)\."
)

#: Values are printed to 3 decimal places, so each carries at most 5e-4 of
#: rounding. Propagated through a product of three factors each <= 1, the error
#: in the reconstructed product is bounded by 3 * 5e-4. Derived, not chosen.
ROUNDING_BOUND = 3 * 5e-4


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """Records spanning every branch the explanation can take."""
    return make_frame(
        {},
        {"severity_score": 1.0, "z_cost": 100.0, "anomaly_count": 3,
         "anomaly_types": ["cost_outlier", "overspend_anomaly", "temporal_outlier"]},
        {"severity_score": 0.0, "z_cost": 0.0, "z_spend": 0.0, "z_duration": 0.0,
         "anomaly_types": [], "anomaly_count": 0},
        {"confidence": 0.15},
        dict(NO_SEVERITY),
        {"peer_cell_stable": False, "valid_signal_count": 1},
        {"completeness": 0.55},
        {"critical_deficit": 4.0},
        {"temporal_hard_fail": True, "temporal": 0.0},
        {"duplicate_flag": True, "duplicate_score": 0.95,
         "anomaly_types": ["duplicate_suspect"]},
    )


# ---------------------------------------------------------------------------
# FIX 1 / FIX 3 - uncertainty is load-bearing, and that is measured
# ---------------------------------------------------------------------------


class TestUncertaintyIsLoadBearing:
    """The variance share said 0.92%. The decision test says otherwise."""

    def test_dropping_stability_changes_bands(self) -> None:
        """The decisive test for redundancy: does removing it change a decision?

        A factor can carry almost none of the spread and still move records
        across a band boundary. Variance share cannot see that; this can.
        """
        rng = np.random.default_rng(3)
        rows = [
            {
                "severity_score": float(rng.uniform(0.3, 1.0)),
                "valid_signal_count": int(rng.integers(1, 4)),
                "peer_cell_stable": bool(rng.integers(0, 2)),
                "z_cost": float(rng.uniform(0, 10)),
            }
            for _ in range(600)
        ]
        output = RiskLayer().run(make_frame(*rows)).frame
        actual = output["risk_score"].to_numpy(dtype="float64")
        without = (
            output["risk_signal_strength"].to_numpy(dtype="float64")
            * output["risk_data_quality"].to_numpy(dtype="float64")
        )
        band = lambda v: np.where(
            v >= R_HIGH, "high", np.where(v >= R_LOW, "moderate", "low")
        )
        assert int((band(actual) != band(without)).sum()) > 0

    def test_the_report_runs_the_removal_test(self, spread: pd.DataFrame) -> None:
        analysis = RiskLayer().run(spread).calibration["contribution_analysis"]
        assert "stability_removal_test" in analysis
        assert analysis["stability_removal_test"]["verdict"] in {
            "load-bearing",
            "redundant",
        }

    def test_stability_only_ever_reduces_risk(self) -> None:
        """Whatever it does, it cannot inflate. Bounds the whole question."""
        rng = np.random.default_rng(5)
        rows = [
            {
                "severity_score": float(rng.uniform(0, 1)),
                "valid_signal_count": int(rng.integers(1, 4)),
                "peer_cell_stable": bool(rng.integers(0, 2)),
            }
            for _ in range(300)
        ]
        output = RiskLayer().run(make_frame(*rows)).frame
        without = (
            output["risk_signal_strength"] * output["risk_data_quality"]
        )
        defined = output["risk_defined"].to_numpy(dtype=bool)
        assert bool(
            (output.loc[defined, "risk_score"] <= without[defined] + 1e-12).all()
        )

    def test_removing_it_never_inverts_an_ordering(self) -> None:
        """It scales each record by a factor in [0.45, 1]; order is not random."""
        rng = np.random.default_rng(9)
        rows = [
            {"severity_score": float(rng.uniform(0, 1)), "valid_signal_count": 3}
            for _ in range(200)
        ]
        output = RiskLayer().run(make_frame(*rows)).frame
        # With uncertainty held constant, ordering must track severity exactly.
        joined = output.join(make_frame(*rows)[["severity_score"]])
        assert joined["risk_score"].corr(
            joined["severity_score"], method="spearman"
        ) == pytest.approx(1.0)

    @pytest.mark.parametrize("name,expected", list(UNCERTAINTY_COMPONENT_CLASS.items()))
    def test_every_component_is_classified(self, name: str, expected: str) -> None:
        assert expected in {"active", "gate_redundant", "structurally_impossible"}

    def test_the_classification_is_published(self, spread: pd.DataFrame) -> None:
        liveness = RiskLayer().run(spread).calibration["uncertainty_liveness"]
        assert liveness["_classification"] == dict(UNCERTAINTY_COMPONENT_CLASS)

    def test_the_gate_redundant_claim_is_verified_not_asserted(
        self, spread: pd.DataFrame
    ) -> None:
        """The published class must match the measured behaviour."""
        liveness = RiskLayer().run(spread).calibration["uncertainty_liveness"]
        for name in ("no_severity", "no_norm"):
            if name in liveness:
                assert liveness[name]["dead_in_score"] is True
                assert liveness[name]["class"] == "gate_redundant"

    def test_liveness_reports_records_affected(self, spread: pd.DataFrame) -> None:
        liveness = RiskLayer().run(spread).calibration["uncertainty_liveness"]
        for name in ("no_severity", "no_norm", "unstable_cell"):
            if name in liveness:
                assert "records_affected" in liveness[name]
                assert "records_affected_scored" in liveness[name]


# ---------------------------------------------------------------------------
# FIX 2 - the explanation reconstructs the score exactly
# ---------------------------------------------------------------------------


class TestExplanationRoundTrip:
    """Parse the sentence, re-multiply, compare. No column is consulted."""

    def test_every_scored_explanation_reconstructs(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        checked = 0
        for _, row in output.iterrows():
            if not row["risk_defined"]:
                continue
            match = COMPOSITION.search(row["risk_explanation"])
            assert match is not None, f"no composition sentence: {row['risk_explanation']}"
            values = {k: float(v) for k, v in match.groupdict().items() if k != "band"}
            product = values["signal"] * values["quality"] * values["stability"]
            assert abs(values["risk"] - product) <= ROUNDING_BOUND
            checked += 1
        assert checked > 0

    def test_the_parsed_score_matches_the_stored_score(
        self, spread: pd.DataFrame
    ) -> None:
        output = RiskLayer().run(spread).frame
        for _, row in output.iterrows():
            match = COMPOSITION.search(row["risk_explanation"])
            if match is None:
                continue
            assert float(match.group("risk")) == pytest.approx(
                row["risk_score"], abs=5e-4
            )

    def test_the_parsed_band_matches_the_stored_band(
        self, spread: pd.DataFrame
    ) -> None:
        output = RiskLayer().run(spread).frame
        for _, row in output.iterrows():
            match = COMPOSITION.search(row["risk_explanation"])
            if match is None:
                continue
            assert match.group("band").replace(" ", "_") == row["risk_flag"]

    def test_it_never_names_an_inactive_factor(self, spread: pd.DataFrame) -> None:
        """The score does not use these; the narrative must not imply it does."""
        output = RiskLayer().run(spread).frame
        for text in output["risk_explanation"]:
            lowered = text.lower()
            assert "deficit" not in lowered
            assert "cluster penalty" not in lowered

    def test_it_never_names_a_gate_redundant_uncertainty_term(
        self, spread: pd.DataFrame
    ) -> None:
        """A scored record cannot have them, so mentioning them would be false."""
        output = RiskLayer().run(spread).frame
        scored = output["risk_defined"].to_numpy(dtype=bool)
        for text in output.loc[scored, "risk_explanation"]:
            assert "no peer norm" not in text.lower()

    def test_a_reconstruction_survives_extreme_records(self) -> None:
        for override in (
            {"severity_score": 1.0, "z_cost": 1e6},
            {"severity_score": 0.0, "z_cost": 0.0},
            {"completeness": 1e-9},
            {"valid_signal_count": 1, "peer_cell_stable": False},
        ):
            row = run_one(**override)
            if not row["risk_defined"]:
                continue
            match = COMPOSITION.search(row["risk_explanation"])
            assert match is not None
            product = (
                float(match.group("signal"))
                * float(match.group("quality"))
                * float(match.group("stability"))
            )
            assert abs(float(match.group("risk")) - product) <= ROUNDING_BOUND


# ---------------------------------------------------------------------------
# FIX 4 - the zero-signal case, proven
# ---------------------------------------------------------------------------


class TestZeroSignalIsMeasuredNormal:
    """Proven from the Stage 3 reason, not assumed from the value."""

    def test_a_defined_deviation_of_zero_is_a_measurement(self) -> None:
        """Stage 3 computed a comparison and it landed on the peer median.

        On the reference corpus all five such records carry
        `deviation_cell_cost_reason == "defined"` with a deviation of exactly
        0.0, and `valid_signal_count >= 1`. That is a measurement, not a gap,
        so risk 0 is the informative answer and NaN would be a false claim of
        ignorance.
        """
        row = run_one(
            severity_score=0.0, z_cost=0.0, z_spend=0.0, z_duration=0.0,
            valid_signal_count=1, anomaly_types=[], anomaly_count=0,
        )
        assert row["risk_signal_strength"] == pytest.approx(0.0)
        assert row["risk_score"] == pytest.approx(0.0)
        assert row["risk_defined"]
        assert row["risk_defined_reason"] == "ok"
        assert row["risk_flag"] == "low_risk"

    def test_it_is_distinguishable_from_an_unmeasured_record(self) -> None:
        """The distinction the whole system rests on."""
        measured = run_one(severity_score=0.0, z_cost=0.0, z_spend=0.0,
                           z_duration=0.0, anomaly_types=[], anomaly_count=0)
        unmeasured = run_one(**NO_SEVERITY)
        assert measured["risk_score"] == 0.0
        assert pd.isna(unmeasured["risk_score"])
        assert measured["risk_flag"] != unmeasured["risk_flag"]

    def test_its_explanation_does_not_claim_ignorance(self) -> None:
        text = run_one(severity_score=0.0, z_cost=0.0, z_spend=0.0,
                       z_duration=0.0, anomaly_types=[], anomaly_count=0)[
            "risk_explanation"
        ]
        assert "No risk score" not in text
        assert "Risk 0.000" in text


# ---------------------------------------------------------------------------
# FIX 5 - stability under extreme inputs
# ---------------------------------------------------------------------------


class TestExtremeInputs:
    """The four mandated adversarial shapes, plus numeric extremes."""

    def test_1_max_severity_poor_quality(self) -> None:
        row = run_one(severity_score=1.0, z_cost=50.0, confidence=0.55,
                      completeness=0.2)
        assert 0.0 <= row["risk_score"] <= 1.0
        assert row["risk_score"] <= 0.2 + 1e-12  # bounded by the quality term

    def test_2_low_severity_perfect_quality(self) -> None:
        row = run_one(severity_score=0.2, z_cost=1.0)
        assert row["risk_score"] <= row["risk_signal_strength"] + 1e-12
        assert row["risk_flag"] in {"low_risk", "moderate_risk"}

    def test_3_saturated_uncertainty_gives_no_score(self) -> None:
        row = run_one(**NO_SEVERITY)
        assert row["risk_uncertainty"] == pytest.approx(1.0)
        assert pd.isna(row["risk_score"])

    def test_4_all_inputs_minimal_but_defined(self) -> None:
        row = run_one(
            severity_score=0.0, confidence=MIN_CONFIDENCE_FOR_RISK,
            completeness=MIN_CONFIDENCE_FOR_RISK, temporal=MIN_CONFIDENCE_FOR_RISK,
            reconciliation=MIN_CONFIDENCE_FOR_RISK, valid_signal_count=1,
            peer_cell_stable=False, z_cost=0.0, z_spend=0.0, z_duration=0.0,
            anomaly_types=[], anomaly_count=0,
        )
        assert row["risk_defined"]
        assert row["risk_score"] == pytest.approx(0.0)
        assert row["risk_score"] >= 0.0

    @pytest.mark.parametrize(
        "override",
        [
            {"severity_score": 1e-300},
            {"completeness": 1e-300},
            {"confidence": 1.0, "completeness": 1e-12},
            {"z_cost": 1e308},
            {"z_cost": -1e308},
            {"critical_deficit": 1e6},
        ],
    )
    def test_numeric_extremes_do_not_break_the_score(
        self, override: Dict[str, Any]
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            row = run_one(**override)
        value = row["risk_score"]
        assert pd.isna(value) or (0.0 <= value <= 1.0)
        assert not np.isinf(row["risk_signal_strength"])
        assert not np.isinf(row["risk_data_quality"])

    def test_risk_is_never_negative(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        assert bool((output["risk_score"].dropna() >= 0.0).all())

    def test_monotonicity_survives_at_the_extremes(self) -> None:
        scores = [
            run_one(severity_score=v, completeness=0.5)["risk_score"]
            for v in (0.0, 1e-9, 0.001, 0.5, 0.999, 1.0)
        ]
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# FIX 6 / FIX 7 - honesty and visibility
# ---------------------------------------------------------------------------


class TestDiagnosticVisibility:
    """The report must state what it is and where the score comes from."""

    def test_the_not_a_threshold_note_is_present(self, spread: pd.DataFrame) -> None:
        report = RiskLayer().run(spread).calibration
        assert report["risk_score"]["_not_a_threshold"] == RISK_NOT_A_THRESHOLD_NOTE
        assert "NOT calibrated thresholds" in RISK_NOT_A_THRESHOLD_NOTE

    def test_contribution_shares_are_reported(self, spread: pd.DataFrame) -> None:
        analysis = RiskLayer().run(spread).calibration["contribution_analysis"]
        for name in ("signal_strength", "data_quality", "stability"):
            assert "variance_share_pct" in analysis["factors"][name]
            assert "covariance_share_pct" in analysis["factors"][name]

    def test_covariance_shares_sum_to_one_hundred(self) -> None:
        """The property that makes covariance the honest attribution."""
        rng = np.random.default_rng(13)
        rows = [
            {
                "severity_score": float(rng.uniform(0, 1)),
                "confidence": float(rng.uniform(0.5, 1)),
                "completeness": float(rng.uniform(0.5, 1)),
                "valid_signal_count": int(rng.integers(1, 4)),
            }
            for _ in range(500)
        ]
        output = RiskLayer().run(make_frame(*rows)).frame
        analysis = compute_contribution_analysis(output)
        total = sum(
            entry["covariance_share_pct"] for entry in analysis["factors"].values()
        )
        assert total == pytest.approx(100.0, abs=1e-6)

    def test_a_zero_factor_record_is_excluded_not_approximated(self) -> None:
        """log(0) has no value; clipping it would break the 100% identity."""
        rng = np.random.default_rng(23)
        rows = [
            {"severity_score": float(rng.uniform(0.1, 1)), "valid_signal_count": 3}
            for _ in range(200)
        ]
        rows.append({"severity_score": 0.0, "z_cost": 0.0, "z_spend": 0.0,
                     "z_duration": 0.0, "anomaly_types": [], "anomaly_count": 0})
        analysis = compute_contribution_analysis(
            RiskLayer().run(make_frame(*rows)).frame
        )
        assert analysis["n_excluded_zero_factor"] == 1
        assert analysis["n_attributed"] == analysis["n_scored"] - 1
        total = sum(
            entry["covariance_share_pct"] for entry in analysis["factors"].values()
        )
        assert total == pytest.approx(100.0, abs=1e-6)

    def test_variance_and_covariance_shares_can_disagree(self) -> None:
        """Which is why both are published rather than one."""
        rng = np.random.default_rng(17)
        rows = [
            {
                "severity_score": float(rng.uniform(0, 1)),
                "valid_signal_count": int(rng.integers(1, 4)),
            }
            for _ in range(400)
        ]
        output = RiskLayer().run(make_frame(*rows)).frame
        analysis = compute_contribution_analysis(output)
        stability = analysis["factors"]["stability"]
        assert stability["variance_share_pct"] != stability["covariance_share_pct"]

    def test_a_low_contributor_is_flagged(self, spread: pd.DataFrame) -> None:
        analysis = RiskLayer().run(spread).calibration["contribution_analysis"]
        assert "flagged" in analysis
        assert analysis["_flag_threshold_pct"] == CONTRIBUTION_FLAG_THRESHOLD_PCT

    def test_a_flag_is_not_an_instruction_to_remove(
        self, spread: pd.DataFrame
    ) -> None:
        """The distinction that kept a load-bearing factor in the system."""
        analysis = RiskLayer().run(spread).calibration["contribution_analysis"]
        if analysis.get("flagged"):
            assert "NOT FOR REMOVAL" in analysis["_flag_note"]

    def test_the_analysis_handles_too_few_records(self) -> None:
        output = RiskLayer().run(make_frame({})).frame
        assert "unavailable" in compute_contribution_analysis(output)

    def test_the_analysis_handles_a_frame_without_components(self) -> None:
        assert "unavailable" in compute_contribution_analysis(
            pd.DataFrame({"risk_score": [0.1, 0.2, 0.3]})
        )

    def test_the_report_is_json_serialisable(self, spread: pd.DataFrame) -> None:
        json.dumps(RiskLayer().run(spread).calibration)

    def test_diagnostics_change_no_score(self, spread: pd.DataFrame) -> None:
        off = RiskLayer(RiskConfig(compute_calibration=False)).run(spread)
        on = RiskLayer(RiskConfig(compute_calibration=True)).run(spread)
        pd.testing.assert_frame_equal(off.frame, on.frame)


# ---------------------------------------------------------------------------
# Ranking preservation - this pass changed no score
# ---------------------------------------------------------------------------


class TestNoRankingChange:
    """The hardening pass must be numerically inert."""

    def test_the_composition_is_still_the_plain_product(
        self, spread: pd.DataFrame
    ) -> None:
        output = RiskLayer().run(spread).frame
        defined = output["risk_defined"].to_numpy(dtype=bool)
        expected = (
            output["risk_signal_strength"]
            * output["risk_data_quality"]
            * (1.0 - output["risk_uncertainty"])
        )
        np.testing.assert_allclose(
            output.loc[defined, "risk_score"].to_numpy(),
            expected[defined].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_quality_is_still_the_minimum(self) -> None:
        row = run_one(confidence=0.9, completeness=0.6, temporal=0.8)
        assert row["risk_data_quality"] == pytest.approx(0.6)

    def test_determinism(self, spread: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            RiskLayer().run(spread).frame, RiskLayer().run(spread).frame
        )

    def test_bands_remain_the_untouched_constants(self) -> None:
        assert (R_LOW, R_HIGH, MIN_CONFIDENCE_FOR_RISK) == (0.20, 0.50, 0.5)

    def test_all_invariants_hold(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        undefined = ~output["risk_defined"].to_numpy(dtype=bool)
        assert output.loc[undefined, "risk_score"].isna().all()
        assert (output.loc[undefined, "risk_defined_reason"] != "ok").all()
        present = output["risk_score"].dropna()
        assert bool(((present >= 0.0) & (present <= 1.0)).all())
        assert set(output["risk_flag"]) <= set(RISK_FLAGS)
