"""Stage 5 audit - regression tests for the corrections, and adversarial cases.

Two kinds of test:

* **Regression** tests that pin the audit fixes so the double count cannot come
  back and the explanation cannot resume attributing quality to a factor the
  composition no longer uses.
* **Adversarial** tests that construct the specific records most likely to
  produce a dishonest score, and assert that they do not.

The double count is pinned structurally rather than by asserting a number: the
test proves ``min`` is idempotent by feeding the same defect twice and checking
the quality does not move. A weight that is applied once cannot be applied twice
by accident if the operator is idempotent.
"""

from __future__ import annotations

import copy
import json
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    CALIBRATION_STATUS_BANNER,
    MIN_CONFIDENCE_FOR_RISK,
    R_HIGH,
    R_LOW,
    RISK_FLAGS,
)
from src.stage5.components import compute_data_quality, compute_uncertainty
from src.stage5.pipeline import RiskConfig, RiskLayer

from tests.test_stage5 import BASELINE, NO_SEVERITY, make_frame, run_one


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """The adversarial set from AUDIT TASK 5."""
    return make_frame(
        # 1. high severity + low confidence: must never look actionable
        {"severity_score": 1.0, "z_cost": 60.0, "confidence": 0.15,
         "completeness": 0.2, "anomaly_count": 3,
         "anomaly_types": ["cost_outlier", "overspend_anomaly", "temporal_outlier"]},
        # 2. low severity + high confidence: measured, clean, must score low
        {"severity_score": 0.02, "z_cost": 0.2, "anomaly_types": [],
         "anomaly_count": 0},
        # 3. duplicate-only
        {"severity_score": 0.0, "z_cost": 0.0, "z_spend": 0.0, "z_duration": 0.0,
         "duplicate_flag": True, "duplicate_score": 1.0,
         "anomaly_types": ["duplicate_suspect"], "anomaly_count": 1},
        # 4. no peer norm
        {"cluster_has_norm": False, **NO_SEVERITY},
        # 5. all deviations NaN
        dict(NO_SEVERITY),
        # 6. perfect storm: max severity, perfect data, full coverage
        {"severity_score": 1.0, "z_cost": 100.0, "anomaly_count": 3,
         "anomaly_types": ["cost_outlier", "overspend_anomaly", "temporal_outlier"]},
    )


# ---------------------------------------------------------------------------
# TASK 1 - the double count is gone and cannot return
# ---------------------------------------------------------------------------


class TestDoubleCountRemoved:
    """Quality uses each Stage 2 quantity exactly once."""

    def test_quality_is_the_minimum_of_confidence_and_components(self) -> None:
        row = run_one(confidence=0.9, completeness=0.6, temporal=0.8,
                      reconciliation=0.95)
        assert row["risk_data_quality"] == pytest.approx(0.6)

    def test_confidence_can_be_the_binding_term(self) -> None:
        row = run_one(confidence=0.55, completeness=0.9, temporal=0.9,
                      reconciliation=0.9)
        assert row["risk_data_quality"] == pytest.approx(0.55)

    def test_the_operator_is_idempotent(self) -> None:
        """The structural guarantee: one defect charged twice charges once.

        This is why the fix is a minimum and not a re-weighted product. A
        product would need every input audited for overlap forever; ``min``
        makes the overlap harmless by construction.
        """
        once = run_one(confidence=0.4 + 1e-9, completeness=0.7)
        twice = run_one(confidence=0.4 + 1e-9, completeness=0.7, temporal=0.7)
        assert once["risk_data_quality"] == pytest.approx(twice["risk_data_quality"])

    def test_critical_deficit_no_longer_suppresses_quality(self) -> None:
        """It sits inside Stage 2 completeness; charging it again was the bug."""
        clean = run_one(critical_deficit=0.0)["risk_data_quality"]
        gappy = run_one(critical_deficit=4.0)["risk_data_quality"]
        assert clean == pytest.approx(gappy)

    def test_cluster_penalty_no_longer_suppresses_quality(self) -> None:
        full = run_one(cluster_penalty_factor=1.0)["risk_data_quality"]
        penalised = run_one(cluster_penalty_factor=0.3)["risk_data_quality"]
        assert full == pytest.approx(penalised)

    def test_the_deficit_is_still_reported(self) -> None:
        """Removing it from the score must not hide it from the auditor."""
        row = run_one(critical_deficit=2.0)
        assert row["risk_deficit_factor"] == pytest.approx(np.exp(-0.5 * 2.0))

    def test_quality_never_exceeds_confidence(self) -> None:
        for confidence in (0.5, 0.7, 0.9, 1.0):
            row = run_one(confidence=confidence)
            assert row["risk_data_quality"] <= confidence + 1e-12

    def test_the_non_compensatory_property_survives(self) -> None:
        """A perfect completeness still cannot rescue a broken reconciliation."""
        balanced = run_one(completeness=0.7, reconciliation=0.7)["risk_data_quality"]
        lopsided = run_one(completeness=1.0, reconciliation=0.4)["risk_data_quality"]
        assert lopsided < balanced

    def test_an_undefined_component_is_still_skipped(self) -> None:
        with_it = run_one(reconciliation=0.2, reconciliation_defined=True)
        without = run_one(reconciliation=0.2, reconciliation_defined=False)
        assert without["risk_data_quality"] > with_it["risk_data_quality"]

    def test_impossible_dates_still_cap_quality(self) -> None:
        row = run_one(temporal_hard_fail=True)
        assert row["risk_data_quality"] <= 0.05


# ---------------------------------------------------------------------------
# TASK 2 - severity, not confidence, must drive the score
# ---------------------------------------------------------------------------


class TestDominanceCorrected:
    """The ordering this layer exists to get right."""

    def test_severity_outranks_confidence_in_driving_risk(self) -> None:
        """Both vary over their full range; severity must win the correlation."""
        rng = np.random.default_rng(7)
        rows = []
        for _ in range(400):
            rows.append(
                {
                    "severity_score": float(rng.uniform(0, 1)),
                    "confidence": float(rng.uniform(0.5, 1.0)),
                    "completeness": 1.0,
                    "z_cost": 1.0,
                    "anomaly_types": [],
                    "anomaly_count": 0,
                }
            )
        frame = make_frame(*rows)
        output = RiskLayer().run(frame).frame.join(
            frame[["severity_score", "confidence"]]
        )
        by_severity = output["risk_score"].corr(
            output["severity_score"], method="spearman"
        )
        by_confidence = output["risk_score"].corr(
            output["confidence"], method="spearman"
        )
        assert by_severity > by_confidence

    def test_signal_carries_most_of_the_variance(self) -> None:
        """In log space, where the product is a sum."""
        rng = np.random.default_rng(11)
        rows = [
            {
                "severity_score": float(rng.uniform(0, 1)),
                "confidence": float(rng.uniform(0.5, 1.0)),
                "z_cost": float(rng.uniform(0, 8)),
            }
            for _ in range(400)
        ]
        output = RiskLayer().run(make_frame(*rows)).frame
        log = lambda values: np.log(np.clip(values.to_numpy(dtype="float64"), 1e-12, 1))
        signal = np.var(log(output["risk_signal_strength"]))
        quality = np.var(log(output["risk_data_quality"]))
        assert signal > quality


# ---------------------------------------------------------------------------
# TASK 3 - the transform that was rejected, and why
# ---------------------------------------------------------------------------


class TestNoInflatingTransform:
    """A decompressing transform would break a Stage 5 guarantee."""

    def test_the_score_never_exceeds_its_own_signal(
        self, spread: pd.DataFrame
    ) -> None:
        """The property that rules out a geometric mean.

        (S x Q x (1-U))^(1/3) exceeds S whenever Q and (1-U) are larger than S,
        which on the reference corpus is 98.93% of scored records - a worst case
        of 0.192 inflated to 0.577. Any Option A/B/C transform that raises the
        maximum does so by inflating weak evidence.
        """
        output = RiskLayer().run(spread).frame
        defined = output["risk_defined"].to_numpy(dtype=bool)
        assert bool(
            (
                output.loc[defined, "risk_score"]
                <= output.loc[defined, "risk_signal_strength"] + 1e-12
            ).all()
        )

    def test_a_geometric_mean_would_inflate(self, spread: pd.DataFrame) -> None:
        """Documented counter-example, so the rejection is not folklore."""
        output = RiskLayer().run(spread).frame
        defined = output["risk_defined"].to_numpy(dtype=bool)
        product = (
            output["risk_signal_strength"]
            * output["risk_data_quality"]
            * (1.0 - output["risk_uncertainty"])
        )
        geometric = np.cbrt(product)
        inflated = geometric[defined] > output.loc[defined, "risk_signal_strength"]
        assert bool(inflated.any()), (
            "the counter-example no longer holds; re-examine the rejection"
        )

    def test_a_perfect_storm_record_can_reach_the_top_of_the_scale(self) -> None:
        """The maximum is bounded by the data, not by the formula.

        If a record ever has maximal signal AND perfect evidence AND full
        coverage, it scores 1.0. That no real record does is a fact about the
        corpus, and is not a reason to rescale.
        """
        row = run_one(
            severity_score=1.0,
            z_cost=100.0,
            anomaly_count=3,
            anomaly_types=["cost_outlier", "overspend_anomaly", "temporal_outlier"],
        )
        assert row["risk_score"] == pytest.approx(1.0)
        assert row["risk_flag"] == "high_risk"


# ---------------------------------------------------------------------------
# TASK 4 - dead uncertainty logic is retained as a guard and published
# ---------------------------------------------------------------------------


class TestUncertaintyLiveness:
    """Terms that cannot fire inside the score are documented, not hidden."""

    def test_the_gate_redundant_terms_never_fire_on_a_scored_record(self) -> None:
        frame = make_frame(
            {},
            dict(NO_SEVERITY),
            {"cluster_has_norm": False, **NO_SEVERITY},
        )
        result = RiskLayer().run(frame)
        scored = result.frame["risk_defined"].to_numpy(dtype=bool)
        contributions = result.uncertainty.contributions
        for name in ("no_severity", "no_norm"):
            assert not bool((contributions[name].to_numpy()[scored] > 0).any())

    def test_they_still_fire_in_the_reported_column(self) -> None:
        """Alive for every record; dead only inside the score."""
        result = RiskLayer().run(make_frame(dict(NO_SEVERITY)))
        assert result.uncertainty.contributions["no_severity"].iloc[0] > 0
        assert result.frame["risk_uncertainty"].iloc[0] == pytest.approx(1.0)

    def test_the_unreachable_duplicate_term_is_provably_empty(self) -> None:
        """flagged implies reachable, because decay lies in [0,1]."""
        frame = make_frame({"duplicate_flag": True, "duplicate_score": 1.0})
        frame["duplicate_reachable"] = True
        result = RiskLayer().run(frame)
        assert result.uncertainty.contributions["unreachable_duplicate"].iloc[0] == 0.0

    def test_it_would_fire_if_the_upstream_ever_diverged(self) -> None:
        """The guard is retained for exactly this case."""
        frame = make_frame({"duplicate_flag": True, "duplicate_score": 1.0})
        frame["duplicate_reachable"] = False
        result = RiskLayer().run(frame)
        assert result.uncertainty.contributions["unreachable_duplicate"].iloc[0] > 0

    def test_firing_rates_are_published(self, spread: pd.DataFrame) -> None:
        rates = RiskLayer().run(spread).uncertainty.to_dict()["firing_rate_pct"]
        assert set(rates) >= {"no_severity", "no_norm", "unstable_cell", "coverage"}

    def test_the_calibration_report_names_the_dead_terms(
        self, spread: pd.DataFrame
    ) -> None:
        liveness = RiskLayer().run(spread).calibration["uncertainty_liveness"]
        assert liveness["no_severity"]["dead_in_score"] is True
        assert "redundant with the gate" in liveness["_note"]


# ---------------------------------------------------------------------------
# TASK 5 - adversarial records
# ---------------------------------------------------------------------------


class TestAdversarial:
    """The records most likely to produce a dishonest score."""

    def test_high_severity_low_confidence_is_never_scored(self) -> None:
        row = run_one(
            severity_score=1.0, z_cost=60.0, confidence=0.15, completeness=0.2,
            anomaly_count=3,
        )
        assert pd.isna(row["risk_score"])
        assert row["risk_flag"] == "insufficient_data"
        assert row["risk_defined_reason"] == "confidence_below_gate"

    def test_low_severity_high_confidence_scores_low_not_absent(self) -> None:
        """Measured and clean is a real answer, distinct from unmeasured."""
        row = run_one(severity_score=0.02, z_cost=0.2, anomaly_types=[],
                      anomaly_count=0)
        assert row["risk_defined"]
        assert row["risk_flag"] == "low_risk"
        assert np.isfinite(row["risk_score"])

    def test_a_zero_signal_record_scores_zero_not_nan(self) -> None:
        """AUDIT: an invariant was proposed that would make this NaN.

        It is deliberately NOT applied. These records have severity_defined,
        a usable signal count and high confidence - they were measured and
        found clean. Reporting NaN would assert "unknown", which is false, and
        would collapse the Stage 4/5 distinction between "measured and normal"
        (0) and "could not be measured" (NaN).
        """
        row = run_one(severity_score=0.0, z_cost=0.0, z_spend=0.0, z_duration=0.0,
                      anomaly_types=[], anomaly_count=0)
        assert row["risk_signal_strength"] == pytest.approx(0.0)
        assert row["risk_score"] == pytest.approx(0.0)
        assert row["risk_defined"]
        assert row["risk_defined_reason"] == "ok"

    def test_duplicate_only_cannot_reach_high_risk(self) -> None:
        row = run_one(
            severity_score=0.0, z_cost=0.0, z_spend=0.0, z_duration=0.0,
            duplicate_flag=True, duplicate_score=1.0,
            anomaly_types=["duplicate_suspect"], anomaly_count=1,
        )
        assert row["risk_flag"] != "high_risk"

    def test_no_peer_norm_is_unscored(self) -> None:
        row = run_one(cluster_has_norm=False, **NO_SEVERITY)
        assert pd.isna(row["risk_score"])
        assert row["risk_flag"] == "insufficient_data"

    def test_all_nan_deviations_is_unscored(self) -> None:
        row = run_one(**NO_SEVERITY)
        assert pd.isna(row["risk_score"])
        assert row["risk_uncertainty"] == pytest.approx(1.0)

    def test_no_adversarial_record_produces_an_illegal_score(
        self, spread: pd.DataFrame
    ) -> None:
        output = RiskLayer().run(spread).frame
        present = output["risk_score"].dropna()
        assert bool(((present >= 0.0) & (present <= 1.0)).all())
        assert set(output["risk_flag"]) <= set(RISK_FLAGS)

    def test_no_runtime_warning_on_the_adversarial_set(
        self, spread: pd.DataFrame
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            RiskLayer().run(spread)

    @pytest.mark.parametrize("value", [np.inf, -np.inf, 1e300])
    def test_a_non_finite_deviation_cannot_break_the_score(
        self, value: float
    ) -> None:
        row = run_one(z_cost=value)
        assert not np.isinf(row["risk_signal_strength"])
        present = row["risk_score"]
        assert pd.isna(present) or 0.0 <= present <= 1.0


# ---------------------------------------------------------------------------
# TASK 6 - explanation integrity after the fix
# ---------------------------------------------------------------------------


class TestExplanationIntegrityAfterFix:
    """The narrative must not attribute quality to a factor it no longer uses."""

    def test_it_never_mentions_the_inactive_deficit_factor(self) -> None:
        """Superseded the earlier wording: an inactive factor is not named at all.

        The hardening pass tightened this from "name it, but say it is not
        charged" to "do not name it". A narrative that claims to reconstruct
        the arithmetic cannot also discuss a term outside the arithmetic,
        however carefully the caveat is worded.
        """
        text = run_one(critical_deficit=3.0, confidence=0.9)["risk_explanation"]
        assert "deficit" not in text.lower()
        assert "critical fields" not in text.lower()

    def test_it_describes_quality_as_a_minimum(self) -> None:
        text = run_one(completeness=0.6)["risk_explanation"]
        assert "lowest of" in text

    def test_the_arithmetic_still_closes(self, spread: pd.DataFrame) -> None:
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

    def test_the_printed_factors_multiply_to_the_printed_score(
        self, spread: pd.DataFrame
    ) -> None:
        """Parsed back out of the sentence, not read from the columns."""
        import re

        output = RiskLayer().run(spread).frame
        pattern = re.compile(
            r"Risk ([\d.]+) \(.*?\), composed as signal ([\d.]+) x data quality "
            r"([\d.]+) x stability ([\d.]+)\."
        )
        checked = 0
        for text in output["risk_explanation"]:
            match = pattern.search(text)
            if not match:
                continue
            score, signal, quality, stability = (float(g) for g in match.groups())
            assert score == pytest.approx(signal * quality * stability, abs=1e-3)
            checked += 1
        assert checked > 0

    def test_gating_is_never_hidden(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        undefined = ~output["risk_defined"].to_numpy(dtype=bool)
        for text in output.loc[undefined, "risk_explanation"]:
            assert "No risk score" in text
            assert "not that it was found safe" in text


# ---------------------------------------------------------------------------
# TASK 7 - calibration honesty
# ---------------------------------------------------------------------------


class TestCalibrationHonesty:
    """The reports must say what they are."""

    def test_the_risk_report_carries_the_banner(self, spread: pd.DataFrame) -> None:
        report = RiskLayer().run(spread).report()
        assert report["_status"] == CALIBRATION_STATUS_BANNER
        assert "UNFIT FOR PRODUCTION" in report["_status"]

    def test_the_calibration_report_carries_the_banner(
        self, spread: pd.DataFrame
    ) -> None:
        report = RiskLayer().run(spread).calibration
        assert "UNFIT FOR PRODUCTION" in report["_status"]

    def test_the_banner_survives_serialisation(self, spread: pd.DataFrame) -> None:
        blob = json.dumps(RiskLayer().run(spread).report())
        assert "UNFIT FOR PRODUCTION" in blob

    def test_the_bands_are_labelled_as_judgements(self, spread: pd.DataFrame) -> None:
        bands = RiskLayer().run(spread).calibration["bands_in_force"]
        assert bands["_status"].startswith("judgements")
        assert bands["r_high"] == R_HIGH
        assert bands["r_low"] == R_LOW

    def test_the_thresholds_are_still_the_untouched_constants(self) -> None:
        """A tuned threshold would show up here as a non-round number."""
        assert R_HIGH == 0.50
        assert R_LOW == 0.20
        assert MIN_CONFIDENCE_FOR_RISK == 0.5


# ---------------------------------------------------------------------------
# Invariants, re-verified after the fix
# ---------------------------------------------------------------------------


class TestInvariantsAfterFix:
    """All six, on the adversarial set."""

    def test_no_false_confidence(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        undefined = ~output["risk_defined"].to_numpy(dtype=bool)
        assert output.loc[undefined, "risk_score"].isna().all()

    def test_no_leakage_from_raw_deviations(self) -> None:
        """Risk moves only through severity, not through the raw z, except for
        the declared extreme bucket."""
        below = run_one(z_cost=1.0)["risk_score"]
        still_below = run_one(z_cost=4.9)["risk_score"]
        assert below == pytest.approx(still_below)

    def test_monotone_in_severity(self) -> None:
        scores = [run_one(severity_score=v)["risk_score"] for v in np.linspace(0, 1, 20)]
        assert scores == sorted(scores)

    def test_monotone_in_confidence(self) -> None:
        scores = [run_one(confidence=v)["risk_score"] for v in np.linspace(0.5, 1.0, 20)]
        assert scores == sorted(scores)

    def test_antitone_in_uncertainty(self) -> None:
        scores = [
            run_one(valid_signal_count=n)["risk_score"] for n in (1, 2, 3)
        ]
        assert scores == sorted(scores)

    def test_bounded(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        present = output["risk_score"].dropna()
        assert bool(((present >= 0.0) & (present <= 1.0)).all())
        assert not np.isinf(output["risk_score"].to_numpy(dtype="float64")).any()

    def test_every_nan_has_a_reason(self, spread: pd.DataFrame) -> None:
        output = RiskLayer().run(spread).frame
        undefined = ~output["risk_defined"].to_numpy(dtype=bool)
        assert (output.loc[undefined, "risk_defined_reason"] != "ok").all()
        assert output.loc[undefined, "risk_defined_reason"].notna().all()

    def test_determinism_survives_the_fix(self, spread: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            RiskLayer().run(spread).frame, RiskLayer().run(spread).frame
        )
