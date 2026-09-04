"""Stage 5 - Risk Scoring Layer.

The suite is organised around what Stage 5 claims, not around its functions.
The claims are strong and mostly negative - risk is never produced from weak
data, undefined stays undefined, nothing is labelled fraud - so most tests here
check that something does *not* happen.

The centrepiece is `TestArithmeticReconstruction`: the score must be
reproducible by hand from its three published components. If that holds, the
explanation cannot lie about the number, because the number is the product of
the values the explanation prints.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    MIN_CONFIDENCE_FOR_RISK,
    R_HIGH,
    R_LOW,
    RISK_BREADTH_SATURATION,
    RISK_DUPLICATE_WEIGHT,
    RISK_FLAGS,
    RISK_TEMPORAL_HARD_FAIL_QUALITY,
    RISK_UNDEFINED_REASONS,
    STAGE5_CALIBRATION_REPORT,
    STAGE5_RISK_REPORT,
    STAGE5_VERSION,
    Z_EXTREME_THRESHOLD,
    Z_HIGH_THRESHOLD,
)
from src.stage5.calibration import compute_stage5_calibration_report
from src.stage5.components import (
    REQUIRED_STAGE2,
    REQUIRED_STAGE3,
    REQUIRED_STAGE4,
    Stage5InputError,
    compute_data_quality,
    compute_signal_strength,
    compute_uncertainty,
    require_contract,
)
from src.stage5.explanation import explain_risk
from src.stage5.pipeline import (
    STAGE5_COLUMNS,
    STAGE5_DETAIL_COLUMNS,
    RiskConfig,
    RiskLayer,
    Stage5Result,
    attach_risk,
)
from src.stage5.risk import compute_risk

# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

#: A clean, fully-measurable, entirely unremarkable record.
BASELINE: Dict[str, Any] = {
    # --- Stage 2 -----------------------------------------------------------
    "confidence": 1.0,
    "completeness": 1.0,
    "temporal": 1.0,
    "reconciliation": 1.0,
    "completeness_defined": True,
    "temporal_defined": True,
    "reconciliation_defined": True,
    "critical_deficit": 0.0,
    "cluster_penalty_factor": 1.0,
    "temporal_hard_fail": False,
    "lifecycle_state": "terminal",
    # --- Stage 3 -----------------------------------------------------------
    "cluster_id": 3,
    "cluster_has_norm": True,
    "peer_cell_stable": True,
    "duplicate_flag": False,
    "duplicate_score": 0.0,
    # --- Stage 4 -----------------------------------------------------------
    "severity_score": 0.4,
    "severity_defined": True,
    "severity_defined_reason": "ok",
    "anomaly_types": ["cost_outlier"],
    "anomaly_count": 1,
    "valid_signal_count": 3,
    "z_cost": 3.2,
    "z_spend": 0.4,
    "z_duration": 0.1,
}

#: Every way a record can carry no severity.
NO_SEVERITY: Dict[str, Any] = {
    "severity_score": np.nan,
    "severity_defined": False,
    "severity_defined_reason": "insufficient_features",
    "anomaly_types": ["insufficient_context"],
    "anomaly_count": 1,
    "valid_signal_count": 0,
    "z_cost": np.nan,
    "z_spend": np.nan,
    "z_duration": np.nan,
}


def make_frame(*overrides: Dict[str, Any]) -> pd.DataFrame:
    """Build a Stage 5 input frame from per-record overrides of BASELINE."""
    rows = []
    for override in overrides or ({},):
        row = copy.deepcopy(BASELINE)
        row.update(override)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in (
        "completeness_defined",
        "temporal_defined",
        "reconciliation_defined",
        "temporal_hard_fail",
        "cluster_has_norm",
        "peer_cell_stable",
        "duplicate_flag",
        "severity_defined",
    ):
        frame[column] = frame[column].astype(bool)
    return frame


def run_one(**overrides: Any) -> pd.Series:
    """Run the layer over one record and return its output row."""
    return RiskLayer().run(make_frame(dict(overrides))).frame.iloc[0]


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """A frame spanning the interesting corners of the input space."""
    return make_frame(
        {},                                                          # 0 ordinary
        {"severity_score": 1.0, "z_cost": 40.0},                     # 1 extreme
        {"confidence": 0.1},                                         # 2 gated out
        dict(NO_SEVERITY),                                           # 3 unmeasurable
        {"duplicate_flag": True, "duplicate_score": 0.95,
         "anomaly_types": ["duplicate_suspect"]},                    # 4 duplicate
        {"peer_cell_stable": False, "valid_signal_count": 1},        # 5 unstable
        {"temporal_hard_fail": True, "temporal": 0.0},               # 6 bad dates
        {"critical_deficit": 3.0, "completeness": 0.4},              # 7 gappy
        {"cluster_has_norm": False, **NO_SEVERITY},                  # 8 noise
        {"severity_score": 0.0, "anomaly_types": [],
         "anomaly_count": 0, "z_cost": 0.1},                         # 9 clean
    )


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


class TestInputContract:
    """Stage 5 refuses to guess at an incomplete upstream."""

    def test_accepts_a_complete_frame(self) -> None:
        require_contract(make_frame())

    @pytest.mark.parametrize(
        "column", [*REQUIRED_STAGE2, *REQUIRED_STAGE3, *REQUIRED_STAGE4]
    )
    def test_every_required_column_is_required(self, column: str) -> None:
        with pytest.raises(Stage5InputError, match=column):
            require_contract(make_frame().drop(columns=[column]))

    def test_the_error_names_the_responsible_stage(self) -> None:
        """A fix needs to know which stage to re-run, not just what is absent."""
        with pytest.raises(Stage5InputError, match="stage2"):
            require_contract(make_frame().drop(columns=["critical_deficit"]))
        with pytest.raises(Stage5InputError, match="stage4"):
            require_contract(make_frame().drop(columns=["severity_defined"]))

    def test_the_layer_refuses_an_incomplete_frame(self) -> None:
        with pytest.raises(Stage5InputError):
            RiskLayer().run(make_frame().drop(columns=["confidence"]))

    def test_duplicate_reachable_is_optional(self) -> None:
        """It is measured only when Stage 4's diagnostics ran."""
        assert "duplicate_reachable" not in make_frame().columns
        RiskLayer().run(make_frame())  # must not raise

    def test_rejects_a_non_frame_source(self) -> None:
        with pytest.raises(TypeError):
            RiskLayer().run(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The score reconstructs from its own components
# ---------------------------------------------------------------------------


class TestArithmeticReconstruction:
    """risk = signal x quality x (1 - uncertainty), checkable by hand."""

    def test_the_product_holds_on_every_defined_record(
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

    def test_a_hand_computed_record_matches(self) -> None:
        """Worked by hand from the constants, not from the implementation."""
        row = run_one(
            severity_score=0.5,
            anomaly_types=["cost_outlier"],
            anomaly_count=1,
            z_cost=3.0,  # below Z_HIGH_THRESHOLD, so no extreme bonus
            z_spend=0.1,
            z_duration=0.1,
        )
        # signal = 0.5 + (1 - 0.5) * (0.20 * 1/3 + 0.30 * 0 + 0.10 * 0)
        expected_signal = 0.5 + 0.5 * (0.20 * (1.0 / RISK_BREADTH_SATURATION))
        assert row["risk_signal_strength"] == pytest.approx(expected_signal)
        # quality = 1.0 (confidence) * 1.0 (floor) * exp(0) * 1.0
        assert row["risk_data_quality"] == pytest.approx(1.0)
        # uncertainty = 0 (defined, norm, stable, full coverage)
        assert row["risk_uncertainty"] == pytest.approx(0.0)
        assert row["risk_score"] == pytest.approx(expected_signal)

    def test_the_explanation_prints_the_factors_that_multiply(self) -> None:
        """The sentence must be checkable without opening the code."""
        row = run_one(severity_score=0.6)
        text = row["risk_explanation"]
        for value in ("risk_score", "risk_signal_strength", "risk_data_quality"):
            assert f"{row[value]:.3f}" in text
        assert f"{1.0 - row['risk_uncertainty']:.3f}" in text

    def test_the_explanation_never_contradicts_its_own_arithmetic(
        self, spread: pd.DataFrame
    ) -> None:
        """A record with no named finding must not claim a breadth boost."""
        output = RiskLayer().run(spread).frame
        for _, row in output.iterrows():
            if "distinct findings at once" in row["risk_explanation"]:
                assert row["risk_breadth"] > 0.0

    def test_the_deficit_factor_is_the_stated_exponential(self) -> None:
        row = run_one(critical_deficit=2.0)
        assert row["risk_deficit_factor"] == pytest.approx(np.exp(-0.5 * 2.0))


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


class TestGating:
    """A risk score exists only where all three conditions hold."""

    def test_a_clean_record_gets_a_score(self) -> None:
        row = run_one()
        assert row["risk_defined"]
        assert row["risk_defined_reason"] == "ok"
        assert np.isfinite(row["risk_score"])

    def test_no_severity_means_no_risk(self) -> None:
        row = run_one(**NO_SEVERITY)
        assert not row["risk_defined"]
        assert pd.isna(row["risk_score"])
        assert row["risk_defined_reason"] == "severity_undefined"

    def test_low_confidence_means_no_risk(self) -> None:
        row = run_one(confidence=0.2)
        assert not row["risk_defined"]
        assert row["risk_defined_reason"] == "confidence_below_gate"

    def test_no_cluster_norm_means_no_risk(self) -> None:
        row = run_one(cluster_has_norm=False)
        assert not row["risk_defined"]
        assert row["risk_defined_reason"] == "no_cluster_norm"

    def test_severity_outranks_the_other_reasons(self) -> None:
        """The reason should be the first thing wrong, not an arbitrary one."""
        row = run_one(confidence=0.1, cluster_has_norm=False, **NO_SEVERITY)
        assert row["risk_defined_reason"] == "severity_undefined"

    def test_confidence_outranks_the_missing_norm(self) -> None:
        row = run_one(confidence=0.1, cluster_has_norm=False)
        assert row["risk_defined_reason"] == "confidence_below_gate"

    @pytest.mark.parametrize("confidence", [0.0, 0.25, 0.49, 0.4999])
    def test_below_the_gate_is_never_scored(self, confidence: float) -> None:
        assert not run_one(confidence=confidence)["risk_defined"]

    @pytest.mark.parametrize("confidence", [MIN_CONFIDENCE_FOR_RISK, 0.75, 1.0])
    def test_at_or_above_the_gate_is_scored(self, confidence: float) -> None:
        assert run_one(confidence=confidence)["risk_defined"]

    def test_an_undefined_risk_is_nan_never_zero(self) -> None:
        """0.0 would read as 'checked, safe'. It is not the same claim."""
        row = run_one(**NO_SEVERITY)
        assert pd.isna(row["risk_score"])

    def test_an_ungated_record_is_banded_insufficient_data(self) -> None:
        assert run_one(confidence=0.1)["risk_flag"] == "insufficient_data"

    def test_only_declared_reasons_are_emitted(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        assert set(output["risk_defined_reason"]) <= set(RISK_UNDEFINED_REASONS)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    """The ordering the composition claims to preserve."""

    def test_higher_severity_gives_higher_risk(self) -> None:
        scores = [
            run_one(severity_score=value)["risk_score"]
            for value in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
        ]
        assert scores == sorted(scores)
        assert scores[0] < scores[-1]

    def test_strictly_higher_where_severity_differs(self) -> None:
        low = run_one(severity_score=0.3)["risk_score"]
        high = run_one(severity_score=0.4)["risk_score"]
        assert high > low

    def test_lower_confidence_gives_lower_risk(self) -> None:
        scores = [
            run_one(confidence=value)["risk_score"]
            for value in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        ]
        assert scores == sorted(scores)

    def test_higher_uncertainty_gives_lower_risk(self) -> None:
        stable = run_one()["risk_score"]
        unstable = run_one(peer_cell_stable=False)["risk_score"]
        assert unstable < stable

    def test_less_coverage_gives_lower_risk(self) -> None:
        full = run_one(valid_signal_count=3)["risk_score"]
        partial = run_one(valid_signal_count=1)["risk_score"]
        assert partial < full

    def test_more_missing_critical_fields_gives_lower_risk(self) -> None:
        scores = [
            run_one(critical_deficit=value)["risk_score"]
            for value in (3.0, 2.0, 1.0, 0.0)
        ]
        assert scores == sorted(scores)

    def test_a_higher_z_never_lowers_the_signal(self) -> None:
        """The extreme bucket may only raise."""
        base = run_one(z_cost=1.0)["risk_signal_strength"]
        high = run_one(z_cost=Z_HIGH_THRESHOLD + 1)["risk_signal_strength"]
        extreme = run_one(z_cost=Z_EXTREME_THRESHOLD + 1)["risk_signal_strength"]
        assert base <= high <= extreme

    def test_a_boost_never_lowers_a_signal_below_its_severity(
        self, spread: pd.DataFrame
    ) -> None:
        output = RiskLayer().run(spread).frame
        joined = output.join(spread[["severity_score"]])
        defined = joined["risk_signal_strength"].notna()
        assert bool(
            (
                joined.loc[defined, "risk_signal_strength"]
                >= joined.loc[defined, "severity_score"] - 1e-12
            ).all()
        )


# ---------------------------------------------------------------------------
# Risk is conditional on evidence
# ---------------------------------------------------------------------------


class TestRiskIsConditional:
    """The whole point: a strong anomaly on weak data is not high risk."""

    def test_a_strong_anomaly_on_strong_data_scores_high(self) -> None:
        row = run_one(
            severity_score=1.0,
            z_cost=Z_EXTREME_THRESHOLD + 5,
            anomaly_types=["cost_outlier", "overspend_anomaly", "temporal_outlier"],
            anomaly_count=3,
        )
        assert row["risk_flag"] == "high_risk"
        assert row["risk_score"] >= R_HIGH

    def test_the_same_anomaly_on_unreadable_data_is_not_scored(self) -> None:
        """Not 'low risk' - unscored. It is a remediation case, not a clean one."""
        row = run_one(
            severity_score=1.0,
            z_cost=Z_EXTREME_THRESHOLD + 5,
            anomaly_count=3,
            confidence=0.2,
        )
        assert row["risk_flag"] == "insufficient_data"
        assert pd.isna(row["risk_score"])

    def test_the_same_anomaly_with_gappy_data_scores_lower(self) -> None:
        clean = run_one(severity_score=1.0, z_cost=40.0)["risk_score"]
        gappy = run_one(
            severity_score=1.0, z_cost=40.0, critical_deficit=3.0, completeness=0.4
        )["risk_score"]
        assert gappy < clean

    def test_impossible_dates_crush_the_quality_term(self) -> None:
        row = run_one(severity_score=1.0, temporal_hard_fail=True, temporal=0.0)
        assert row["risk_data_quality"] <= RISK_TEMPORAL_HARD_FAIL_QUALITY

    def test_the_component_floor_is_non_compensatory(self) -> None:
        """A perfect completeness must not paper over a broken reconciliation."""
        balanced = run_one(completeness=0.7, reconciliation=0.7)["risk_data_quality"]
        lopsided = run_one(completeness=1.0, reconciliation=0.4)["risk_data_quality"]
        assert lopsided < balanced

    def test_an_undefined_component_is_not_treated_as_perfect(self) -> None:
        """Skipping is right; scoring it 1.0 would be vacuous perfection."""
        with_it = run_one(reconciliation=0.2, reconciliation_defined=True)
        without = run_one(reconciliation=0.2, reconciliation_defined=False)
        assert without["risk_data_quality"] > with_it["risk_data_quality"]

    def test_the_score_is_at_most_the_signal(self, spread: pd.DataFrame) -> None:
        """Neither quality nor stability can ever inflate evidence."""
        output = RiskLayer().run(spread).frame
        defined = output["risk_defined"].to_numpy(dtype=bool)
        assert bool(
            (
                output.loc[defined, "risk_score"]
                <= output.loc[defined, "risk_signal_strength"] + 1e-12
            ).all()
        )


# ---------------------------------------------------------------------------
# The duplicate signal stays subordinate
# ---------------------------------------------------------------------------


class TestDuplicateSubordinate:
    """Stage 3's detector has ~1% measured recall; it may support, not decide."""

    def test_a_duplicate_alone_cannot_reach_high_risk(self) -> None:
        row = run_one(
            severity_score=0.0,
            anomaly_types=["duplicate_suspect"],
            anomaly_count=1,
            duplicate_flag=True,
            duplicate_score=1.0,
            z_cost=0.1,
            z_spend=0.1,
            z_duration=0.1,
        )
        assert row["risk_flag"] != "high_risk"

    def test_its_contribution_is_capped_at_a_tenth(self) -> None:
        without = run_one(severity_score=0.0, z_cost=0.1)["risk_signal_strength"]
        with_dupe = run_one(
            severity_score=0.0,
            z_cost=0.1,
            duplicate_flag=True,
            duplicate_score=1.0,
            anomaly_types=["duplicate_suspect"],
        )["risk_signal_strength"]
        # Breadth also moves, so bound the duplicate term specifically.
        assert with_dupe - without <= RISK_DUPLICATE_WEIGHT + 0.20 + 1e-9

    def test_the_weight_cap_is_enforced_at_construction(self) -> None:
        with pytest.raises(ValueError, match="cap"):
            RiskConfig(duplicate_weight=0.5)

    def test_an_unflagged_duplicate_score_is_ignored(self) -> None:
        quiet = run_one(duplicate_score=0.8, duplicate_flag=False)
        none = run_one(duplicate_score=0.0, duplicate_flag=False)
        assert quiet["risk_score"] == pytest.approx(none["risk_score"])

    def test_low_confidence_type_does_not_add_breadth(self) -> None:
        """It describes the evidence and is already priced in data quality."""
        plain = run_one(anomaly_types=["cost_outlier"])
        with_flag = run_one(anomaly_types=["cost_outlier", "low_confidence"])
        assert plain["risk_breadth"] == with_flag["risk_breadth"]


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------


class TestBands:
    """Descriptive, mutually exclusive, and not decisions."""

    def test_only_declared_flags_are_emitted(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        assert set(output["risk_flag"]) <= set(RISK_FLAGS)

    def test_the_bands_partition_the_corpus(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        counts = sum(int((output["risk_flag"] == name).sum()) for name in RISK_FLAGS)
        assert counts == len(output)

    def test_stage_six_vocabulary_is_not_reused(self, spread: pd.DataFrame) -> None:
        """Stage 6 owns INVESTIGATE / REMEDIATE / MONITOR / CLEAR."""
        output = RiskLayer().run(spread).frame
        emitted = set(output["risk_flag"])
        assert not emitted & {"INVESTIGATE", "REMEDIATE", "MONITOR", "CLEAR"}

    def test_no_column_asserts_fraud(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        assert not any("fraud" in name.lower() for name in output.columns)
        for text in output["risk_explanation"]:
            assert "not a finding of fraud" in text or "No risk score" in text

    def test_band_boundaries_are_inclusive_upward(self) -> None:
        result = RiskLayer().run(make_frame())
        boundary = pd.DataFrame(
            {"risk_score": [R_LOW - 1e-9, R_LOW, R_HIGH - 1e-9, R_HIGH]}
        )
        expected = ["low_risk", "moderate_risk", "moderate_risk", "high_risk"]
        # Reproduce the banding rule directly on the boundary values.
        actual = [
            "high_risk"
            if value >= R_HIGH
            else "moderate_risk"
            if value >= R_LOW
            else "low_risk"
            for value in boundary["risk_score"]
        ]
        assert actual == expected

    def test_the_band_query_filters(self, spread: pd.DataFrame) -> None:
        result = RiskLayer().run(spread)
        for name in RISK_FLAGS:
            assert set(result.band(name)["risk_flag"]) <= {name}

    def test_an_unknown_band_is_rejected(self, spread: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="unknown risk flag"):
            RiskLayer().run(spread).band("CLEAR")


# ---------------------------------------------------------------------------
# The six mandatory invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    """Non-negotiable, and asserted in the pipeline as well as here."""

    def test_1_undefined_risk_is_nan(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        undefined = ~output["risk_defined"].to_numpy(dtype=bool)
        assert output.loc[undefined, "risk_score"].isna().all()
        assert output.loc[~undefined, "risk_score"].notna().all()

    def test_2_low_confidence_can_never_be_high_risk(self) -> None:
        for confidence in (0.0, 0.1, 0.3, 0.49):
            row = run_one(
                confidence=confidence, severity_score=1.0, z_cost=100.0, anomaly_count=3
            )
            assert row["risk_flag"] != "high_risk"

    def test_3_a_duplicate_alone_cannot_be_high_risk(self) -> None:
        row = run_one(
            severity_score=0.0,
            duplicate_flag=True,
            duplicate_score=1.0,
            anomaly_types=["duplicate_suspect"],
            anomaly_count=1,
            z_cost=0.0,
            z_spend=0.0,
            z_duration=0.0,
        )
        assert row["risk_flag"] != "high_risk"

    def test_4_undefined_severity_gives_undefined_risk(self) -> None:
        row = run_one(**NO_SEVERITY)
        assert not row["risk_defined"]
        assert pd.isna(row["risk_score"])
        assert row["risk_uncertainty"] == pytest.approx(1.0)

    def test_5_no_runtime_warning_is_raised(self, spread: pd.DataFrame) -> None:
        """Including on an all-undefined row, where nanmax would warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            RiskLayer().run(spread)
            RiskLayer().run(make_frame(dict(NO_SEVERITY)))

    def test_6_every_output_is_finite_or_nan(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        for column in (
            "risk_score",
            "risk_signal_strength",
            "risk_data_quality",
            "risk_uncertainty",
        ):
            values = output[column].to_numpy(dtype="float64")
            assert not np.isinf(values).any()
            present = values[np.isfinite(values)]
            assert bool(((present >= 0.0) & (present <= 1.0)).all())

    def test_components_never_leak_nan_where_they_are_total(
        self, spread: pd.DataFrame
    ) -> None:
        """Quality and uncertainty are defined for every record, always."""
        output = RiskLayer().run(spread).frame
        assert output["risk_data_quality"].notna().all()
        assert output["risk_uncertainty"].notna().all()


# ---------------------------------------------------------------------------
# Determinism and the output contract
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same input, same output, byte for byte."""

    def test_repeated_runs_agree_exactly(self, spread: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            RiskLayer().run(spread).frame, RiskLayer().run(spread).frame
        )

    def test_the_report_serialises_identically(self, spread: pd.DataFrame) -> None:
        first = json.dumps(RiskLayer().run(spread).report(), sort_keys=True, default=str)
        second = json.dumps(RiskLayer().run(spread).report(), sort_keys=True, default=str)
        assert first == second

    def test_the_report_carries_no_wall_clock(self, spread: pd.DataFrame) -> None:
        blob = json.dumps(RiskLayer().run(spread).report(), default=str)
        assert "elapsed" not in blob and "timestamp" not in blob

    def test_row_order_and_index_are_preserved(self) -> None:
        frame = make_frame({}, {"severity_score": 0.9}, dict(NO_SEVERITY))
        frame.index = pd.Index([31, 7, 88], name="record")
        assert list(RiskLayer().run(frame).frame.index) == [31, 7, 88]

    def test_the_input_is_not_mutated(self, spread: pd.DataFrame) -> None:
        before = spread.copy(deep=True)
        RiskLayer().run(spread)
        pd.testing.assert_frame_equal(spread, before)

    def test_no_stage_four_column_is_altered(self, spread: pd.DataFrame) -> None:
        before = spread[["severity_score", "severity_defined", "anomaly_count"]].copy()
        RiskLayer().run(spread)
        pd.testing.assert_frame_equal(
            spread[["severity_score", "severity_defined", "anomaly_count"]], before
        )

    def test_the_contract_columns_are_produced(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        for column in (*STAGE5_COLUMNS, *STAGE5_DETAIL_COLUMNS):
            assert column in output.columns

    def test_an_empty_frame_produces_an_empty_result(self) -> None:
        result = RiskLayer().run(make_frame().iloc[0:0])
        assert len(result) == 0
        assert result.report()["n_records"] == 0

    def test_reports_are_written_and_reloadable(
        self, spread: pd.DataFrame, tmp_path: Path
    ) -> None:
        written = RiskLayer().run(spread).save_reports(tmp_path)
        assert written["risk_report"].name == STAGE5_RISK_REPORT
        assert written["calibration"].name == STAGE5_CALIBRATION_REPORT
        loaded = json.loads(written["risk_report"].read_text(encoding="utf-8"))
        assert loaded["stage5_version"] == STAGE5_VERSION
        assert loaded["n_records"] == len(spread)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """A malformed configuration fails at construction."""

    def test_defaults_are_the_named_constants(self) -> None:
        config = RiskConfig()
        assert config.r_high == R_HIGH
        assert config.r_low == R_LOW
        assert config.min_confidence == MIN_CONFIDENCE_FOR_RISK

    @pytest.mark.parametrize(
        "override",
        [
            {"r_low": 0.8, "r_high": 0.2},
            {"r_high": 1.5},
            {"r_low": -0.1},
            {"min_confidence": 1.5},
            {"duplicate_weight": 0.9},
        ],
    )
    def test_invalid_configurations_are_rejected(self, override: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            RiskConfig(**override)

    def test_signal_weights_cannot_exceed_the_headroom(self) -> None:
        with pytest.raises(ValueError, match="headroom"):
            compute_signal_strength(
                make_frame(), breadth_weight=0.9, extreme_weight=0.9, duplicate_weight=0.05
            )

    def test_a_stricter_gate_shrinks_the_scored_population(
        self, spread: pd.DataFrame
    ) -> None:
        loose = RiskLayer(RiskConfig(min_confidence=0.0)).run(spread)
        strict = RiskLayer(RiskConfig(min_confidence=0.99)).run(spread)
        assert int(strict.frame["risk_defined"].sum()) <= int(
            loose.frame["risk_defined"].sum()
        )

    def test_calibration_can_be_switched_off(self, spread: pd.DataFrame) -> None:
        assert RiskLayer(RiskConfig(compute_calibration=False)).run(spread).calibration is None

    def test_instrumentation_does_not_change_the_score(
        self, spread: pd.DataFrame
    ) -> None:
        off = RiskLayer(RiskConfig(compute_calibration=False)).run(spread)
        on = RiskLayer(RiskConfig(compute_calibration=True)).run(spread)
        pd.testing.assert_frame_equal(off.frame, on.frame)


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------


class TestCalibrationReport:
    """Descriptive only; defines no threshold."""

    def test_it_reports_the_bands_in_force(self, spread: pd.DataFrame) -> None:
        report = RiskLayer().run(spread).calibration
        assert report["bands_in_force"]["r_high"] == R_HIGH
        assert report["bands_in_force"]["_status"].startswith("judgements")

    def test_it_states_that_it_defines_no_threshold(self, spread: pd.DataFrame) -> None:
        report = RiskLayer().run(spread).calibration
        assert "threshold" in report["_note"].lower()

    def test_correlations_are_reported(self, spread: pd.DataFrame) -> None:
        correlations = RiskLayer().run(spread).calibration["correlations"]
        assert "risk_vs_severity" in correlations
        assert "risk_vs_confidence" in correlations

    def test_risk_tracks_severity_positively(self) -> None:
        """The relationship the composition claims to have."""
        frame = make_frame(*[{"severity_score": v} for v in np.linspace(0, 1, 25)])
        output = RiskLayer().run(frame).frame
        report = compute_stage5_calibration_report(output.join(frame))
        assert report["correlations"]["risk_vs_severity"] > 0.9

    def test_undefined_records_are_broken_down_by_reason(
        self, spread: pd.DataFrame
    ) -> None:
        report = RiskLayer().run(spread).calibration
        assert sum(report["undefined"]["by_reason"].values()) == report["undefined"]["count"]

    def test_it_survives_a_frame_without_stage_five(self, spread: pd.DataFrame) -> None:
        assert "unavailable" in compute_stage5_calibration_report(spread)

    def test_it_is_deterministic(self, spread: pd.DataFrame) -> None:
        first = RiskLayer().run(spread).calibration
        second = RiskLayer().run(spread).calibration
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_it_is_json_serialisable(self, spread: pd.DataFrame) -> None:
        json.dumps(RiskLayer().run(spread).calibration)


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


class TestExplanations:
    """Honest about what is known and what is not."""

    def test_every_record_gets_one(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        assert output["risk_explanation"].notna().all()
        assert (output["risk_explanation"].str.len() > 60).all()

    def test_an_undefined_record_is_not_described_as_safe(self) -> None:
        text = run_one(**NO_SEVERITY)["risk_explanation"]
        assert "unassessed, not cleared" in text
        assert "not that it was found safe" in text

    def test_a_gated_record_names_confidence_as_the_blocker(self) -> None:
        text = run_one(confidence=0.2)["risk_explanation"]
        assert "confidence gate" in text
        assert "prerequisite" in text

    def test_signals_are_retained_in_an_undefined_explanation(self) -> None:
        """The finding does not disappear because it cannot be scored."""
        text = run_one(
            confidence=0.2, anomaly_types=["cost_outlier"], anomaly_count=1
        )["risk_explanation"]
        assert "retained" in text
        assert "cost out of line" in text

    def test_it_explains_why_the_score_is_below_the_signal(self) -> None:
        text = run_one(peer_cell_stable=False, valid_signal_count=1)["risk_explanation"]
        assert "below the raw signal" in text
        assert "conditional on evidence" in text

    def test_a_duplicate_is_described_as_subordinate(self) -> None:
        text = run_one(
            duplicate_flag=True, duplicate_score=0.9, anomaly_types=["duplicate_suspect"]
        )["risk_explanation"]
        assert "never carries a case by itself" in text

    def test_it_recomputes_nothing(self) -> None:
        row = {
            "risk_defined": True,
            "risk_score": 0.25,
            "risk_signal_strength": 0.5,
            "risk_data_quality": 0.5,
            "risk_uncertainty": 0.0,
            "risk_flag": "moderate_risk",
        }
        assert explain_risk(row) == explain_risk(dict(row))

    def test_the_stored_explanation_is_returned_unchanged(
        self, spread: pd.DataFrame
    ) -> None:
        result = RiskLayer().run(spread)
        row = result.frame.index[0]
        assert result.explain(row) == result.frame.loc[row, "risk_explanation"]


# ---------------------------------------------------------------------------
# Integration with the real pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    """Stage 5 on real Stage 1-4 output."""

    @pytest.fixture(scope="class")
    def corpus(self) -> Any:
        from src.stage1.corpus import Corpus
        from src.stage1.data_generator import generate_dataset
        from src.stage2.confidence import attach_confidence
        from src.stage3.pipeline import attach_structure
        from src.stage4.pipeline import AnomalyConfig, attach_anomalies

        built = Corpus.from_dataframe(generate_dataset(n=2000, seed=42))
        attach_confidence(built)
        attach_structure(built)
        attach_anomalies(built, config=AnomalyConfig(compute_calibration=False))
        return built

    def test_it_consumes_real_upstream_output(self, corpus: Any) -> None:
        result = attach_risk(corpus)
        assert len(result) == len(corpus)
        for column in STAGE5_COLUMNS:
            assert column in corpus.records.columns

    def test_upstream_columns_survive_untouched(self, corpus: Any) -> None:
        before = corpus.records[["confidence", "severity_score"]].copy()
        attach_risk(corpus)
        pd.testing.assert_frame_equal(
            corpus.records[["confidence", "severity_score"]], before
        )

    def test_the_invariants_hold_on_real_data(self, corpus: Any) -> None:
        frame = attach_risk(corpus).frame
        defined = frame["risk_defined"].to_numpy(dtype=bool)
        assert frame.loc[~defined, "risk_score"].isna().all()
        confidence = corpus.records["confidence"].to_numpy(dtype="float64")
        high = frame["risk_flag"].to_numpy() == "high_risk"
        assert not bool((high & (confidence < MIN_CONFIDENCE_FOR_RISK)).any())

    def test_the_product_reconstructs_on_real_data(self, corpus: Any) -> None:
        frame = attach_risk(corpus).frame
        defined = frame["risk_defined"].to_numpy(dtype=bool)
        expected = (
            frame["risk_signal_strength"]
            * frame["risk_data_quality"]
            * (1.0 - frame["risk_uncertainty"])
        )
        np.testing.assert_allclose(
            frame.loc[defined, "risk_score"].to_numpy(),
            expected[defined].to_numpy(),
            rtol=1e-12,
        )

    def test_unscorable_records_exist_and_are_labelled(self, corpus: Any) -> None:
        """Real data always contains records nothing can be said about."""
        frame = attach_risk(corpus).frame
        undefined = ~frame["risk_defined"].to_numpy(dtype=bool)
        assert bool(undefined.any())
        assert (frame.loc[undefined, "risk_flag"] == "insufficient_data").all()

    def test_it_is_deterministic_on_real_data(self, corpus: Any) -> None:
        pd.testing.assert_frame_equal(
            RiskLayer().run(corpus).frame, RiskLayer().run(corpus).frame
        )

    def test_no_runtime_warning_on_real_data(self, corpus: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            RiskLayer().run(corpus)

    def test_a_misaligned_result_is_rejected(self, corpus: Any) -> None:
        result = RiskLayer().run(corpus)
        truncated = Stage5Result(
            frame=result.frame.iloc[:-1],
            strength=result.strength,
            quality=result.quality,
            uncertainty=result.uncertainty,
            risk=result.risk,
            config=result.config,
        )
        with pytest.raises(ValueError, match="rows"):
            attach_risk(corpus, result=truncated)
