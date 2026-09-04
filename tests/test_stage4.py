"""Stage 4 - Contextual Anomaly Interpretation.

The suite is organised around the guarantees Stage 4 claims, not around its
functions. Every mandatory validation from the brief has a named test:

* undefined signals are never treated as zero
* low confidence never escalates to INVESTIGATE
* peer instability leads to insufficient_context
* lifecycle correctly gates underspend
* the duplicate signal never dominates a decision
* outputs are deterministic

Fixtures build minimal frames by hand rather than running Stages 1-3, so a
Stage 4 failure is unambiguously a Stage 4 failure.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    ANOMALY_TYPES,
    CONFIDENCE_GATE_THRESHOLD,
    COST_SCOPES,
    DECISION_CLASSES,
    SEVERITY_WEIGHTS,
    STAGE4_ANOMALY_REPORT,
    STAGE4_VERSION,
    Z_INVESTIGATE_THRESHOLD,
    Z_TYPE_THRESHOLD,
)
from src.stage4.anomaly import (
    DEFINED_REASON,
    REQUIRED_COLUMNS,
    Stage4InputError,
    build_signals,
    classify_types,
    require_contract,
    validate_signals,
)
from src.stage4.decision import compute_severity, route
from src.stage4.explanation import build_explanations, explain_record
from src.stage4.pipeline import (
    STAGE4_COLUMNS,
    AnomalyConfig,
    AnomalyLayer,
    AnomalyResult,
    attach_anomalies,
)

# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

#: A wholly unremarkable record. Every test overrides only what it is about.
BASELINE: Dict[str, Any] = {
    "confidence": 0.9,
    "lifecycle_state": "terminal",
    "cluster_id": 3,
    "cluster_label": "road construction",
    "cluster_has_norm": True,
    "peer_cell_stable": True,
    "peer_cell_size": 40,
    "deviation_cell_cost": 0.4,
    "deviation_cell_cost_reason": DEFINED_REASON,
    "deviation_cluster_cost": 0.4,
    "deviation_cluster_cost_reason": DEFINED_REASON,
    "deviation_spend_ratio": 0.2,
    "deviation_spend_ratio_reason": DEFINED_REASON,
    "deviation_duration": 0.1,
    "deviation_duration_reason": DEFINED_REASON,
    "duplicate_score": 0.0,
    "duplicate_flag": False,
}


def make_frame(*overrides: Dict[str, Any]) -> pd.DataFrame:
    """Build a Stage 4 input frame from per-record overrides of BASELINE.

    Args:
        *overrides: One mapping per record; keys override BASELINE.

    Returns:
        A frame satisfying the Stage 4 input contract.
    """
    rows = []
    for override in overrides or ({},):
        row = copy.deepcopy(BASELINE)
        row.update(override)
        rows.append(row)
    frame = pd.DataFrame(rows)
    # Match the upstream dtypes so tests exercise the real conversion paths.
    frame["peer_cell_stable"] = frame["peer_cell_stable"].astype(bool)
    frame["cluster_has_norm"] = frame["cluster_has_norm"].astype(bool)
    frame["duplicate_flag"] = frame["duplicate_flag"].astype(bool)
    return frame


def undefined(name: str, reason: str = "feature_missing") -> Dict[str, Any]:
    """Overrides that make one deviation undefined, the way Stage 3 does."""
    return {f"deviation_{name}": np.nan, f"deviation_{name}_reason": reason}


ALL_UNDEFINED: Dict[str, Any] = {
    **undefined("cell_cost"),
    **undefined("cluster_cost"),
    **undefined("spend_ratio"),
    **undefined("duration"),
}


def run_one(**overrides: Any) -> pd.Series:
    """Run the full layer over a single record and return its output row."""
    result = AnomalyLayer().run(make_frame(dict(overrides)))
    return result.frame.iloc[0]


@pytest.fixture(scope="module")
def layer() -> AnomalyLayer:
    """A layer at default configuration."""
    return AnomalyLayer()


@pytest.fixture(scope="module")
def realistic() -> pd.DataFrame:
    """A frame spanning the interesting corners of the input space."""
    return make_frame(
        {},                                                       # 0 ordinary
        {"deviation_cell_cost": 6.0},                             # 1 escalating
        {"deviation_cell_cost": 6.0, "confidence": 0.2},          # 2 gated
        dict(ALL_UNDEFINED),                                      # 3 unmeasurable
        {"duplicate_score": 0.95, "duplicate_flag": True},        # 4 duplicate
        {"peer_cell_stable": False, "peer_cell_size": 2},         # 5 unstable
        {"deviation_spend_ratio": -5.0},                          # 6 underspend
        {"deviation_spend_ratio": -5.0,
         "lifecycle_state": "pre_completion"},                    # 7 not yet due
        {"cluster_id": -1, "cluster_label": "unclustered",
         "cluster_has_norm": False, **ALL_UNDEFINED},             # 8 noise
        {"deviation_duration": 9.0},                              # 9 temporal
    )


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


class TestInputContract:
    """Stage 4 refuses to guess at a broken upstream."""

    def test_accepts_a_complete_frame(self) -> None:
        require_contract(make_frame())

    @pytest.mark.parametrize("column", REQUIRED_COLUMNS)
    def test_every_required_column_is_actually_required(self, column: str) -> None:
        frame = make_frame().drop(columns=[column])
        with pytest.raises(Stage4InputError, match=column):
            require_contract(frame)

    def test_missing_columns_are_all_named_at_once(self) -> None:
        """A caller fixing an integration should see the whole list."""
        frame = make_frame().drop(columns=["confidence", "duplicate_score"])
        with pytest.raises(Stage4InputError) as excinfo:
            require_contract(frame)
        assert "confidence" in str(excinfo.value)
        assert "duplicate_score" in str(excinfo.value)

    def test_layer_refuses_an_incomplete_frame(self, layer: AnomalyLayer) -> None:
        with pytest.raises(Stage4InputError):
            layer.run(make_frame().drop(columns=["peer_cell_stable"]))

    def test_rejects_a_non_frame_source(self, layer: AnomalyLayer) -> None:
        with pytest.raises(TypeError):
            layer.run(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MANDATORY: undefined signals are never treated as zero
# ---------------------------------------------------------------------------


class TestUndefinedIsNeverZero:
    """The central claim of Stage 4: missing is not safe."""

    def test_undefined_deviation_yields_nan_not_zero(self) -> None:
        row = run_one(**ALL_UNDEFINED)
        for column in ("z_cost", "z_spend", "z_duration"):
            assert pd.isna(row[column]), f"{column} was filled in"

    def test_undefined_severity_is_nan_not_zero(self) -> None:
        row = run_one(**ALL_UNDEFINED)
        assert pd.isna(row["severity_score"])
        assert row["severity_score"] != 0.0 or pd.isna(row["severity_score"])

    def test_unmeasurable_record_is_not_routed_as_clean(self) -> None:
        """It must not land in MONITOR, which reads as 'checked, fine'."""
        row = run_one(**ALL_UNDEFINED)
        assert row["decision_class"] == "INSUFFICIENT_CONTEXT"
        assert row["decision_reason"] == "no_valid_signal"

    def test_unmeasurable_record_carries_no_deviation_types(self) -> None:
        row = run_one(**ALL_UNDEFINED)
        assert "cost_outlier" not in row["anomaly_types"]
        assert "overspend_anomaly" not in row["anomaly_types"]
        assert "temporal_outlier" not in row["anomaly_types"]
        assert "insufficient_context" in row["anomaly_types"]

    def test_a_partially_measurable_record_keeps_what_it_has(self) -> None:
        """One undefined signal must not void the others."""
        row = run_one(**undefined("duration"))
        assert pd.isna(row["z_duration"])
        assert not pd.isna(row["z_cost"])
        assert not pd.isna(row["z_spend"])
        assert row["valid_signal_count"] == 2

    def test_severity_renormalises_rather_than_diluting(self) -> None:
        """A record measured on fewer signals is not thereby made to look safer."""
        full = run_one(deviation_cell_cost=5.0, deviation_cluster_cost=5.0,
                       deviation_spend_ratio=5.0, deviation_duration=5.0)
        partial = run_one(deviation_cell_cost=5.0, deviation_cluster_cost=5.0,
                          **undefined("spend_ratio"), **undefined("duration"))
        assert full["severity_score"] == pytest.approx(1.0)
        assert partial["severity_score"] == pytest.approx(1.0)

    def test_a_zero_deviation_and_an_undefined_one_differ(self) -> None:
        """The whole point: 0.0 means measured-and-normal, NaN means unknown."""
        measured = run_one(deviation_cell_cost=0.0, deviation_cluster_cost=0.0,
                           deviation_spend_ratio=0.0, deviation_duration=0.0)
        unmeasured = run_one(**ALL_UNDEFINED)
        assert measured["severity_score"] == pytest.approx(0.0)
        assert pd.isna(unmeasured["severity_score"])
        assert measured["decision_class"] == "MONITOR"
        assert unmeasured["decision_class"] == "INSUFFICIENT_CONTEXT"

    def test_a_non_defined_reason_disqualifies_a_present_value(self) -> None:
        """A finite value with a non-defined reason is a contract violation.

        Trusting the number would let an upstream bug become a silent anomaly.
        """
        validation = validate_signals(
            make_frame({"deviation_cell_cost_reason": "cell_unstable"})
        )
        assert not bool(validation.usable["deviation_cell_cost"].iloc[0])

    def test_a_non_finite_value_is_never_usable(self) -> None:
        for bad in (np.inf, -np.inf, np.nan):
            validation = validate_signals(make_frame({"deviation_duration": bad}))
            assert not bool(validation.usable["deviation_duration"].iloc[0])


# ---------------------------------------------------------------------------
# MANDATORY: low confidence never escalates
# ---------------------------------------------------------------------------


class TestConfidenceGate:
    """Confidence controls interpretation, never value."""

    def test_extreme_deviation_at_low_confidence_does_not_escalate(self) -> None:
        row = run_one(confidence=0.1, deviation_cell_cost=50.0,
                      deviation_cluster_cost=50.0, deviation_spend_ratio=50.0,
                      deviation_duration=50.0)
        assert row["decision_class"] == "REMEDIATE"
        assert row["decision_reason"] == "confidence_below_gate"

    def test_the_deviation_value_itself_is_not_damped(self) -> None:
        """Low confidence must not quietly shrink the measurement."""
        high = run_one(confidence=0.95, deviation_cell_cost=6.0)
        low = run_one(confidence=0.05, deviation_cell_cost=6.0)
        assert high["z_cost"] == low["z_cost"] == 6.0
        assert high["severity_score"] == pytest.approx(low["severity_score"])

    def test_low_confidence_record_keeps_its_anomaly_types(self) -> None:
        """Measured, recorded, explained - simply not escalated."""
        row = run_one(confidence=0.1, deviation_cell_cost=6.0)
        assert "cost_outlier" in row["anomaly_types"]
        assert "low_confidence" in row["anomaly_types"]

    def test_no_record_in_the_corpus_escapes_the_gate(
        self, realistic: pd.DataFrame, layer: AnomalyLayer
    ) -> None:
        output = layer.run(realistic).frame
        escalated = output["decision_class"] == "INVESTIGATE"
        assert not bool((escalated & (output["confidence_flag"] == "low")).any())

    @pytest.mark.parametrize("confidence", [0.0, 0.1, 0.25, 0.49, 0.4999])
    def test_below_the_gate_always_remediates(self, confidence: float) -> None:
        row = run_one(confidence=confidence, deviation_cell_cost=99.0)
        assert row["decision_class"] == "REMEDIATE"

    @pytest.mark.parametrize("confidence", [0.5, 0.51, 0.9, 1.0])
    def test_at_or_above_the_gate_escalation_is_permitted(
        self, confidence: float
    ) -> None:
        row = run_one(confidence=confidence, deviation_cell_cost=99.0)
        assert row["decision_class"] == "INVESTIGATE"

    def test_the_gate_boundary_is_inclusive_upward(self) -> None:
        """Exactly at the threshold counts as sufficient, not insufficient."""
        row = run_one(confidence=CONFIDENCE_GATE_THRESHOLD, deviation_cell_cost=99.0)
        assert row["confidence_flag"] == "high"

    def test_missing_confidence_is_treated_as_untrusted(self) -> None:
        """A record with no confidence cannot be accused."""
        row = run_one(confidence=np.nan, deviation_cell_cost=99.0)
        assert row["confidence_flag"] == "low"
        assert row["decision_class"] == "REMEDIATE"

    def test_confidence_outranks_no_signal(self) -> None:
        """Bad evidence is a remediable problem; say so rather than shrugging."""
        row = run_one(confidence=0.1, **ALL_UNDEFINED)
        assert row["decision_class"] == "REMEDIATE"


# ---------------------------------------------------------------------------
# MANDATORY: peer instability leads to insufficient_context
# ---------------------------------------------------------------------------


class TestPeerInstability:
    """A norm nobody can trust is not a norm."""

    def test_unstable_cell_marks_insufficient_context(self) -> None:
        row = run_one(peer_cell_stable=False, peer_cell_size=2)
        assert "insufficient_context" in row["anomaly_types"]

    def test_missing_cluster_norm_marks_insufficient_context(self) -> None:
        row = run_one(cluster_has_norm=False)
        assert "insufficient_context" in row["anomaly_types"]

    def test_noise_cluster_with_no_deviations_routes_to_insufficient(self) -> None:
        row = run_one(cluster_id=-1, cluster_label="unclustered",
                      cluster_has_norm=False, peer_cell_stable=False,
                      **ALL_UNDEFINED)
        assert row["decision_class"] == "INSUFFICIENT_CONTEXT"

    def test_cost_falls_back_from_cell_to_cluster(self) -> None:
        """An unstable cell does not destroy the coarser comparison."""
        row = run_one(peer_cell_stable=False,
                      **undefined("cell_cost", "cell_unstable"),
                      deviation_cluster_cost=4.0)
        assert row["cost_scope"] == "cluster"
        assert row["z_cost"] == 4.0

    def test_cell_is_preferred_when_both_are_available(self) -> None:
        row = run_one(deviation_cell_cost=1.0, deviation_cluster_cost=9.0)
        assert row["cost_scope"] == "cell"
        assert row["z_cost"] == 1.0

    def test_cost_scope_is_none_when_neither_exists(self) -> None:
        row = run_one(**undefined("cell_cost"), **undefined("cluster_cost"))
        assert row["cost_scope"] == "none"
        assert pd.isna(row["z_cost"])

    @pytest.mark.parametrize("scope", COST_SCOPES)
    def test_every_declared_scope_is_reachable(self, scope: str) -> None:
        cases = {
            "cell": {},
            "cluster": {**undefined("cell_cost", "cell_unstable")},
            "none": {**undefined("cell_cost"), **undefined("cluster_cost")},
        }
        assert run_one(**cases[scope])["cost_scope"] == scope


# ---------------------------------------------------------------------------
# MANDATORY: lifecycle gates underspend
# ---------------------------------------------------------------------------


class TestLifecycleGating:
    """An unfinished work has not underspent - it has not finished spending."""

    def test_terminal_underspend_is_an_anomaly(self) -> None:
        row = run_one(lifecycle_state="terminal", deviation_spend_ratio=-5.0)
        assert "underspend_anomaly" in row["anomaly_types"]

    @pytest.mark.parametrize(
        "state", ["pre_completion", "proposed", "approved", "pending", "ongoing"]
    )
    def test_pre_completion_underspend_is_not_an_anomaly(self, state: str) -> None:
        row = run_one(lifecycle_state=state, deviation_spend_ratio=-5.0)
        assert "underspend_anomaly" not in row["anomaly_types"]

    def test_unknown_lifecycle_does_not_produce_an_underspend_claim(self) -> None:
        """Unknown status is not evidence of completion."""
        row = run_one(lifecycle_state="unknown", deviation_spend_ratio=-5.0)
        assert "underspend_anomaly" not in row["anomaly_types"]

    def test_overspend_is_not_lifecycle_gated(self) -> None:
        """Spending beyond the norm is anomalous whenever it happens."""
        row = run_one(lifecycle_state="pre_completion", deviation_spend_ratio=5.0)
        assert "overspend_anomaly" in row["anomaly_types"]

    def test_gating_suppresses_the_type_but_not_the_measurement(self) -> None:
        """The z stays; only the interpretation is withheld."""
        row = run_one(lifecycle_state="pre_completion", deviation_spend_ratio=-5.0)
        assert row["z_spend"] == -5.0
        assert row["valid_signal_count"] == 3

    def test_a_gated_underspend_can_still_escalate_on_magnitude(self) -> None:
        """Routing reads |z|; suppressing a label is not suppressing evidence.

        This is deliberate and worth stating: the record is not accused of
        underspending, but a deviation that large is still worth a look.
        """
        row = run_one(lifecycle_state="pre_completion", deviation_spend_ratio=-9.0)
        assert "underspend_anomaly" not in row["anomaly_types"]
        assert row["decision_class"] == "INVESTIGATE"


# ---------------------------------------------------------------------------
# MANDATORY: the duplicate signal never dominates
# ---------------------------------------------------------------------------


class TestDuplicateIsSupportingOnly:
    """Stage 3's duplicate detector has poor recall; it may support, not decide."""

    def test_a_duplicate_alone_does_not_escalate(self) -> None:
        row = run_one(duplicate_score=1.0, duplicate_flag=True)
        assert row["decision_class"] != "INVESTIGATE"

    def test_a_duplicate_alone_gives_no_severity(self) -> None:
        """Severity built from the duplicate alone would be it deciding."""
        row = run_one(duplicate_score=1.0, duplicate_flag=True, **ALL_UNDEFINED)
        assert pd.isna(row["severity_score"])
        assert "duplicate_suspect" in row["anomaly_types"]

    def test_a_duplicate_never_counts_as_context(self) -> None:
        """It is not a peer comparison, so it cannot rescue an unmeasured record."""
        row = run_one(duplicate_score=1.0, duplicate_flag=True, **ALL_UNDEFINED)
        assert row["valid_signal_count"] == 0
        assert row["decision_class"] == "INSUFFICIENT_CONTEXT"

    def test_its_severity_weight_is_the_smallest(self) -> None:
        assert SEVERITY_WEIGHTS["duplicate"] == min(SEVERITY_WEIGHTS.values())

    def test_it_cannot_move_severity_by_more_than_its_weight(self) -> None:
        without = run_one(deviation_cell_cost=1.0)
        with_dupe = run_one(deviation_cell_cost=1.0, duplicate_score=1.0,
                            duplicate_flag=True)
        assert with_dupe["severity_score"] > without["severity_score"]
        assert with_dupe["severity_score"] - without["severity_score"] < 0.5

    def test_an_unflagged_similarity_score_is_ignored(self) -> None:
        """Background similarity below the threshold is not evidence."""
        quiet = run_one(duplicate_score=0.7, duplicate_flag=False)
        none = run_one(duplicate_score=0.0, duplicate_flag=False)
        assert quiet["severity_score"] == pytest.approx(none["severity_score"])
        assert "duplicate_suspect" not in quiet["anomaly_types"]


# ---------------------------------------------------------------------------
# Type classification
# ---------------------------------------------------------------------------


class TestTypeClassification:
    """Types are named findings, not a score."""

    def test_every_declared_type_has_a_column(self) -> None:
        frame = make_frame()
        types = classify_types(frame, build_signals(frame).frame)
        assert list(types.columns) == list(ANOMALY_TYPES)
        assert all(types[name].dtype == bool for name in ANOMALY_TYPES)

    def test_a_clean_record_carries_no_type(self) -> None:
        row = run_one()
        assert row["anomaly_types"] == []
        assert row["anomaly_count"] == 0

    def test_types_are_not_mutually_exclusive(self) -> None:
        """No single-score collapse: a record can be several things at once."""
        row = run_one(deviation_cell_cost=6.0, deviation_cluster_cost=6.0,
                      deviation_spend_ratio=6.0, deviation_duration=6.0,
                      duplicate_score=0.9, duplicate_flag=True)
        assert row["anomaly_count"] >= 4
        assert {"cost_outlier", "overspend_anomaly", "temporal_outlier",
                "duplicate_suspect"} <= set(row["anomaly_types"])

    def test_anomaly_count_matches_the_list(self, realistic: pd.DataFrame) -> None:
        output = AnomalyLayer().run(realistic).frame
        for _, row in output.iterrows():
            assert row["anomaly_count"] == len(row["anomaly_types"])

    def test_cost_outlier_is_two_sided(self) -> None:
        """Suspiciously cheap is as informative as suspiciously expensive."""
        for value in (6.0, -6.0):
            row = run_one(deviation_cell_cost=value, deviation_cluster_cost=value)
            assert "cost_outlier" in row["anomaly_types"]

    def test_temporal_outlier_is_two_sided(self) -> None:
        for value in (6.0, -6.0):
            assert "temporal_outlier" in run_one(deviation_duration=value)["anomaly_types"]

    @pytest.mark.parametrize("z", [Z_TYPE_THRESHOLD, Z_TYPE_THRESHOLD + 1e-9])
    def test_the_type_threshold_is_inclusive(self, z: float) -> None:
        row = run_one(deviation_cell_cost=z, deviation_cluster_cost=z)
        assert "cost_outlier" in row["anomaly_types"]

    def test_just_below_the_type_threshold_is_not_a_type(self) -> None:
        z = Z_TYPE_THRESHOLD - 1e-6
        row = run_one(deviation_cell_cost=z, deviation_cluster_cost=z)
        assert "cost_outlier" not in row["anomaly_types"]

    def test_naming_is_a_lower_bar_than_escalating(self) -> None:
        """Between the two thresholds a record is named but only monitored."""
        z = (Z_TYPE_THRESHOLD + Z_INVESTIGATE_THRESHOLD) / 2.0
        row = run_one(deviation_cell_cost=z, deviation_cluster_cost=z)
        assert "cost_outlier" in row["anomaly_types"]
        assert row["decision_class"] == "MONITOR"


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


class TestSeverity:
    """A bounded summary that never becomes the verdict."""

    def test_severity_stays_in_the_unit_interval(
        self, realistic: pd.DataFrame
    ) -> None:
        score = AnomalyLayer().run(realistic).frame["severity_score"].dropna()
        assert bool(((score >= 0.0) & (score <= 1.0)).all())

    def test_extreme_deviations_saturate_rather_than_overflow(self) -> None:
        row = run_one(deviation_cell_cost=1e12, deviation_cluster_cost=1e12,
                      deviation_spend_ratio=1e12, deviation_duration=1e12)
        assert row["severity_score"] == pytest.approx(1.0)

    def test_severity_is_monotone_in_the_deviation(self) -> None:
        scores = [
            run_one(deviation_cell_cost=z, deviation_cluster_cost=z)["severity_score"]
            for z in (0.0, 1.0, 2.0, 4.0, 8.0)
        ]
        assert scores == sorted(scores)

    def test_severity_is_symmetric_in_sign(self) -> None:
        """Direction is carried by the type; magnitude by the severity."""
        up = run_one(deviation_duration=4.0)["severity_score"]
        down = run_one(deviation_duration=-4.0)["severity_score"]
        assert up == pytest.approx(down)

    def test_components_are_exposed_for_audit(self) -> None:
        """No single score collapse: the parts must remain inspectable."""
        frame = make_frame()
        severity = compute_severity(build_signals(frame).frame)
        assert list(severity.components.columns) == ["cost", "spend", "duration",
                                                     "duplicate"]

    def test_severity_never_overrides_the_decision(self) -> None:
        """A high severity at low confidence still does not escalate."""
        row = run_one(confidence=0.1, deviation_cell_cost=20.0,
                      deviation_cluster_cost=20.0, deviation_spend_ratio=20.0,
                      deviation_duration=20.0)
        assert row["severity_score"] == pytest.approx(1.0)
        assert row["decision_class"] == "REMEDIATE"

    def test_an_empty_frame_produces_an_empty_result(self) -> None:
        result = AnomalyLayer().run(make_frame().iloc[0:0])
        assert len(result) == 0
        assert result.report()["n_records"] == 0


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouting:
    """Precedence, not arithmetic."""

    def test_only_declared_classes_are_emitted(self, realistic: pd.DataFrame) -> None:
        output = AnomalyLayer().run(realistic).frame
        assert set(output["decision_class"]) <= set(DECISION_CLASSES)

    def test_the_stage_six_vocabulary_is_not_reused(
        self, realistic: pd.DataFrame
    ) -> None:
        """Stage 6 owns CLEAR; Stage 4 must not pre-empt it."""
        output = AnomalyLayer().run(realistic).frame
        assert "CLEAR" not in set(output["decision_class"])

    def test_no_column_is_named_risk_score(self, realistic: pd.DataFrame) -> None:
        """Stage 5 owns R(r). Stage 4's number is a severity, and says so."""
        output = AnomalyLayer().run(realistic).frame
        assert "risk_score" not in output.columns
        assert "severity_score" in output.columns

    def test_every_decision_carries_the_rule_that_produced_it(
        self, realistic: pd.DataFrame
    ) -> None:
        output = AnomalyLayer().run(realistic).frame
        assert output["decision_reason"].notna().all()
        assert (output["decision_reason"].str.len() > 0).all()

    def test_a_result_queue_filters_by_class(self, realistic: pd.DataFrame) -> None:
        result = AnomalyLayer().run(realistic)
        for decision_class in DECISION_CLASSES:
            queue = result.queue(decision_class)
            assert set(queue["decision_class"]) <= {decision_class}

    def test_an_unknown_queue_is_rejected(self, realistic: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="unknown decision class"):
            AnomalyLayer().run(realistic).queue("CLEAR")

    @pytest.mark.parametrize("signal", ["cell_cost", "spend_ratio", "duration"])
    def test_any_single_signal_can_escalate(self, signal: str) -> None:
        row = run_one(**{f"deviation_{signal}": Z_INVESTIGATE_THRESHOLD + 1.0,
                         **({"deviation_cluster_cost": Z_INVESTIGATE_THRESHOLD + 1.0}
                            if signal == "cell_cost" else {})})
        assert row["decision_class"] == "INVESTIGATE"


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


class TestExplanations:
    """Text that reports the absence of evidence as an absence."""

    def test_every_record_gets_one(self, realistic: pd.DataFrame) -> None:
        output = AnomalyLayer().run(realistic).frame
        assert output["explanation_text"].notna().all()
        assert (output["explanation_text"].str.len() > 40).all()

    def test_the_decision_appears_in_its_own_explanation(
        self, realistic: pd.DataFrame
    ) -> None:
        output = AnomalyLayer().run(realistic).frame
        for _, row in output.iterrows():
            assert row["decision_class"] in row["explanation_text"]

    def test_unmeasured_signals_are_named_as_unmeasured(self) -> None:
        text = run_one(**undefined("duration"))["explanation_text"]
        assert "Not assessed" in text
        assert "duration" in text
        assert "not that it was normal" in text

    def test_a_fully_unmeasured_record_is_not_described_as_clean(self) -> None:
        text = run_one(**ALL_UNDEFINED)["explanation_text"]
        assert "unassessed" in text
        assert "Severity is undefined" in text

    def test_a_low_confidence_record_reads_as_a_data_problem(self) -> None:
        text = run_one(confidence=0.1, deviation_cell_cost=20.0)["explanation_text"]
        assert "REMEDIATE" in text
        assert "evidence is repaired" in text
        assert "fraud" in text  # explicitly disclaimed, not asserted

    def test_a_duplicate_is_described_as_secondary(self) -> None:
        text = run_one(duplicate_score=0.95, duplicate_flag=True)["explanation_text"]
        assert "does not drive the decision" in text

    def test_an_unstable_cell_is_disclosed(self) -> None:
        text = run_one(peer_cell_stable=False, peer_cell_size=2)["explanation_text"]
        assert "not stable enough" in text

    def test_an_explanation_recomputes_nothing(self) -> None:
        """Given a row, the text is a pure function of it."""
        row = {"decision_class": "MONITOR", "confidence": 0.9,
               "confidence_flag": "high", "severity_score": 0.1,
               "valid_signal_count": 1, "z_cost": 1.0, "cost_scope": "cell"}
        assert explain_record(row) == explain_record(dict(row))

    def test_explanations_of_an_empty_frame(self) -> None:
        empty = pd.DataFrame(columns=["decision_class"])
        assert len(build_explanations(empty)) == 0


# ---------------------------------------------------------------------------
# Determinism and the output contract
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Byte-identical outputs, or the audit trail means nothing."""

    def test_repeated_runs_agree_exactly(self, realistic: pd.DataFrame) -> None:
        first = AnomalyLayer().run(realistic).frame
        second = AnomalyLayer().run(realistic).frame
        pd.testing.assert_frame_equal(first, second)

    def test_the_report_serialises_identically(self, realistic: pd.DataFrame) -> None:
        first = json.dumps(AnomalyLayer().run(realistic).report(), sort_keys=True,
                           default=str)
        second = json.dumps(AnomalyLayer().run(realistic).report(), sort_keys=True,
                            default=str)
        assert first == second

    def test_the_report_carries_no_wall_clock_value(
        self, realistic: pd.DataFrame
    ) -> None:
        blob = json.dumps(AnomalyLayer().run(realistic).report(), default=str)
        assert "elapsed" not in blob
        assert "timestamp" not in blob

    def test_row_order_and_index_are_preserved(self) -> None:
        frame = make_frame({}, {"deviation_cell_cost": 6.0}, dict(ALL_UNDEFINED))
        frame.index = pd.Index([77, 5, 41], name="record")
        output = AnomalyLayer().run(frame).frame
        assert list(output.index) == [77, 5, 41]

    def test_no_record_is_added_or_lost(self, realistic: pd.DataFrame) -> None:
        assert len(AnomalyLayer().run(realistic)) == len(realistic)

    def test_stage_four_does_not_mutate_its_input(
        self, realistic: pd.DataFrame
    ) -> None:
        before = realistic.copy(deep=True)
        AnomalyLayer().run(realistic)
        pd.testing.assert_frame_equal(realistic, before)

    def test_the_contract_columns_are_all_produced(
        self, realistic: pd.DataFrame
    ) -> None:
        output = AnomalyLayer().run(realistic).frame
        assert set(STAGE4_COLUMNS) <= set(output.columns)

    def test_no_nan_leaks_into_the_decision_surface(
        self, realistic: pd.DataFrame
    ) -> None:
        output = AnomalyLayer().run(realistic).frame
        for column in ("decision_class", "decision_reason", "confidence_flag",
                       "cost_scope", "anomaly_count", "valid_signal_count",
                       "explanation_text"):
            assert output[column].notna().all(), column

    def test_the_report_is_written_and_reloadable(
        self, realistic: pd.DataFrame, tmp_path: Path
    ) -> None:
        written = AnomalyLayer().run(realistic).save_reports(tmp_path)
        path = written["anomaly_report"]
        assert path.name == STAGE4_ANOMALY_REPORT
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["stage4_version"] == STAGE4_VERSION
        assert loaded["n_records"] == len(realistic)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """A malformed configuration fails at construction, not mid-corpus."""

    def test_defaults_are_the_named_constants(self) -> None:
        config = AnomalyConfig()
        assert config.confidence_threshold == CONFIDENCE_GATE_THRESHOLD
        assert config.z_type_threshold == Z_TYPE_THRESHOLD
        assert config.z_investigate_threshold == Z_INVESTIGATE_THRESHOLD

    @pytest.mark.parametrize(
        "override",
        [
            {"confidence_threshold": -0.1},
            {"confidence_threshold": 1.1},
            {"z_type_threshold": 0.0},
            {"z_investigate_threshold": -1.0},
            {"z_severity_scale": 0.0},
            {"z_investigate_threshold": 1.0, "z_type_threshold": 3.0},
            {"severity_weights": {"cost": -1.0}},
            {"severity_weights": {"cost": 0.0}},
        ],
    )
    def test_invalid_configurations_are_rejected(
        self, override: Dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError):
            AnomalyConfig(**override)

    def test_a_stricter_gate_shrinks_the_investigate_queue(
        self, realistic: pd.DataFrame
    ) -> None:
        loose = AnomalyLayer(AnomalyConfig(confidence_threshold=0.0)).run(realistic)
        strict = AnomalyLayer(AnomalyConfig(confidence_threshold=0.99)).run(realistic)
        assert len(strict.queue("INVESTIGATE")) <= len(loose.queue("INVESTIGATE"))

    def test_the_configuration_is_echoed_into_the_report(
        self, realistic: pd.DataFrame
    ) -> None:
        report = AnomalyLayer(AnomalyConfig(z_severity_scale=7.0)).run(
            realistic
        ).report()
        assert report["config"]["z_severity_scale"] == 7.0

    def test_the_report_states_its_own_provisional_status(
        self, realistic: pd.DataFrame
    ) -> None:
        note = AnomalyLayer().run(realistic).report()["_note"]
        assert "Provisional" in note
        assert "R(r)" in note


# ---------------------------------------------------------------------------
# Integration with the real pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    """Stage 4 on real Stage 1-3 output."""

    @pytest.fixture(scope="class")
    def corpus(self) -> Any:
        from src.stage1.corpus import Corpus
        from src.stage1.data_generator import generate_dataset
        from src.stage2.confidence import attach_confidence
        from src.stage3.pipeline import attach_structure

        built = Corpus.from_dataframe(generate_dataset(n=1500, seed=42))
        attach_confidence(built)
        attach_structure(built)
        return built

    def test_it_consumes_real_upstream_output(self, corpus: Any) -> None:
        result = attach_anomalies(corpus)
        assert len(result) == len(corpus)
        for column in STAGE4_COLUMNS:
            assert column in corpus.records.columns

    def test_upstream_columns_survive_untouched(self, corpus: Any) -> None:
        before = corpus.records["confidence"].copy()
        attach_anomalies(corpus)
        pd.testing.assert_series_equal(corpus.records["confidence"], before)

    def test_the_confidence_gate_holds_on_real_data(self, corpus: Any) -> None:
        result = attach_anomalies(corpus)
        escalated = result.frame["decision_class"] == "INVESTIGATE"
        low = result.frame["confidence_flag"] == "low"
        assert not bool((escalated & low).any())

    def test_unmeasurable_records_exist_and_are_labelled(self, corpus: Any) -> None:
        """Real data always contains records nothing can be said about."""
        result = attach_anomalies(corpus)
        unmeasurable = result.frame["valid_signal_count"] == 0
        assert bool(unmeasurable.any())
        assert result.frame.loc[unmeasurable, "severity_score"].isna().all()
        assert (
            result.frame.loc[unmeasurable, "decision_class"]
            .isin(["INSUFFICIENT_CONTEXT", "REMEDIATE"])
            .all()
        )

    def test_the_result_is_deterministic_on_real_data(self, corpus: Any) -> None:
        first = AnomalyLayer().run(corpus).frame
        second = AnomalyLayer().run(corpus).frame
        pd.testing.assert_frame_equal(first, second)

    def test_a_misaligned_result_is_rejected(self, corpus: Any) -> None:
        result = AnomalyLayer().run(corpus)
        truncated = AnomalyResult(
            frame=result.frame.iloc[:-1],
            signals=result.signals,
            types=result.types,
            severity=result.severity,
            decision=result.decision,
            config=result.config,
        )
        with pytest.raises(ValueError, match="rows"):
            attach_anomalies(corpus, result=truncated)
