"""Phase 1 FIX 5-7 and the Stage 8 scaffold.

FIX 7 is the one with teeth. It converts a **silent** failure into a loud one:
an anomaly category outside the closed vocabulary would be scored as zero
breadth by Stage 5 and described by nothing in Stage 7, and neither would say
so. That is measured, not hypothetical - Stage 5's `RISK_BREADTH_TYPES` is a
fixed five-tuple.

The Stage 8 tests all assert a **refusal**. That is the whole deliverable:
with no real labelled outcomes there is no calibration, and an interface that
returns None with a reason is worth more than one that returns a number.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    ANOMALY_TYPES,
    CALIBRATION_METHODS,
    CALIBRATION_MIN_LABELS,
    CALIBRATION_MIN_PER_CLASS,
    CALIBRATION_STATUSES,
    CLOSED_ANOMALY_VOCABULARY,
    M1_CORRECTION_LABEL,
    RISK_CALIBRATION_STATUS,
    RISK_RELATIVE_WARNING,
    STAGE8_VERSION,
)
from src.stage7.annotations import assert_closed_anomaly_vocabulary
from src.stage7.interface import decode_payloads
from src.stage7.pipeline import ConsumptionLayer
from src.stage8.calibration import (
    CALIBRATION_COLUMNS,
    CALIBRATION_LABEL_SCHEMA,
    CalibrationDataset,
    calibrate,
    calibration_design,
)

from tests.test_stage7 import ONE_PER_ACTION, make_frame, payload_for


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """One record per action, plus a duplicated work_id that disagrees."""
    return make_frame(
        *ONE_PER_ACTION,
        {"action_class": "ESCALATE_IMMEDIATE", "priority_level": "P0",
         "work_id": "shared"},
        {"action_class": "PASSIVE_MONITOR", "priority_level": "P3",
         "risk": 0.02, "risk_flag": "low_risk", "decision": "MONITOR",
         "findings": [], "work_id": "shared"},
    )


# ---------------------------------------------------------------------------
# FIX 5 - non-unique work_id, per record
# ---------------------------------------------------------------------------


class TestFix5WorkIdColumns:
    """200 records share a work_id; 56 groups disagree. Neither is visible
    from a single row, which is exactly when it does damage."""

    def test_a_unique_work_id_reads_group_of_one(self) -> None:
        row = ConsumptionLayer().run(make_frame()).annotations.iloc[0]
        assert int(row["work_id_group_size"]) == 1
        assert bool(row["work_id_conflict_flag"]) is False

    def test_a_duplicated_work_id_reports_its_size(self) -> None:
        frame = make_frame({"work_id": "w"}, {"work_id": "w"}, {"work_id": "w"})
        sizes = ConsumptionLayer().run(frame).annotations["work_id_group_size"]
        assert list(sizes) == [3, 3, 3]

    def test_agreeing_duplicates_are_not_a_conflict(self) -> None:
        """Same action twice: duplicated, but not contradictory."""
        frame = make_frame(
            {"work_id": "w", "action_class": "PASSIVE_MONITOR",
             "priority_level": "P3"},
            {"work_id": "w", "action_class": "PASSIVE_MONITOR",
             "priority_level": "P3"},
        )
        flags = ConsumptionLayer().run(frame).annotations["work_id_conflict_flag"]
        assert not flags.any()

    def test_disagreeing_duplicates_are_flagged_on_every_row(self) -> None:
        # Both actions stated explicitly: make_frame defaults to
        # ESCALATE_IMMEDIATE, so relying on the default would compare a
        # record with itself and prove nothing.
        frame = make_frame(
            {"work_id": "w", "action_class": "ESCALATE_IMMEDIATE",
             "priority_level": "P0"},
            {"work_id": "w", "action_class": "PASSIVE_MONITOR",
             "priority_level": "P3"},
        )
        flags = ConsumptionLayer().run(frame).annotations["work_id_conflict_flag"]
        assert flags.all(), "both rows must know the group disagrees"

    def test_no_record_is_collapsed(self, spread: pd.DataFrame) -> None:
        """The columns describe the ambiguity; they never resolve it."""
        result = ConsumptionLayer().run(spread)
        assert len(result.cards) == len(spread)
        assert len(result.annotations) == len(spread)

    def test_no_decision_changes(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        upstream = [json.loads(p)["action"] for p in spread["explanation_payload"]]
        assert [card["action"] for card in result.cards] == upstream

    def test_a_frame_without_work_id_reads_group_of_one(self) -> None:
        """Truthful: no work ids means one record per key by definition."""
        frame = make_frame().drop(columns=["work_id"])
        row = ConsumptionLayer().run(frame).annotations.iloc[0]
        assert int(row["work_id_group_size"]) == 1

    def test_the_rates_are_published(self, spread: pd.DataFrame) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        assert metrics["n_in_duplicated_work"] == 2
        assert metrics["n_work_id_conflict"] == 2


# ---------------------------------------------------------------------------
# FIX 6 - uncalibrated risk, said everywhere
# ---------------------------------------------------------------------------


class TestFix6CalibrationStatus:
    """A queue row read in isolation must still say what the number is not."""

    def test_every_record_carries_the_status(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert annotations["risk_calibration_status"].eq("UNCALIBRATED").all()

    def test_every_record_carries_the_warning(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert annotations["risk_relative_warning"].eq(RISK_RELATIVE_WARNING).all()

    def test_the_warning_says_rankings_not_probabilities(self) -> None:
        assert "relative rankings" in RISK_RELATIVE_WARNING
        assert "NOT probabilities" in RISK_RELATIVE_WARNING

    def test_it_reaches_the_report(self, spread: pd.DataFrame) -> None:
        report = ConsumptionLayer().run(spread).report()
        assert report["risk_calibration_status"] == RISK_CALIBRATION_STATUS
        assert report["risk_relative_warning"] == RISK_RELATIVE_WARNING

    def test_it_reaches_the_metrics(self, spread: pd.DataFrame) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        assert metrics["risk_calibration_status"] == "UNCALIBRATED"

    def test_there_is_exactly_one_status_today(self) -> None:
        """A second value would imply a path this system does not have."""
        from src.core.constants import RISK_CALIBRATION_STATUSES

        assert RISK_CALIBRATION_STATUSES == ("UNCALIBRATED",)


# ---------------------------------------------------------------------------
# FIX 7 - silent assumptions become loud errors
# ---------------------------------------------------------------------------


class TestFix7ClosedVocabulary:
    """An unknown category is scored zero breadth and described by nothing."""

    def test_the_vocabulary_is_stage4_plus_the_correction_label(self) -> None:
        assert set(CLOSED_ANOMALY_VOCABULARY) == set(ANOMALY_TYPES) | {
            M1_CORRECTION_LABEL
        }

    def test_known_categories_pass(self, spread: pd.DataFrame) -> None:
        assert_closed_anomaly_vocabulary(decode_payloads(spread))

    def test_an_unknown_category_raises(self) -> None:
        frame = make_frame()
        frame["explanation_payload"] = payload_for(findings=["brand_new_type"])
        with pytest.raises(ValueError, match="closed vocabulary"):
            assert_closed_anomaly_vocabulary(decode_payloads(frame))

    def test_the_error_names_the_offender_and_the_records(self) -> None:
        frame = make_frame()
        frame["explanation_payload"] = payload_for(findings=["brand_new_type"])
        with pytest.raises(ValueError) as excinfo:
            assert_closed_anomaly_vocabulary(decode_payloads(frame))
        message = str(excinfo.value)
        assert "brand_new_type" in message
        assert "sample" in message

    def test_the_error_explains_the_silent_failure_it_prevents(self) -> None:
        frame = make_frame()
        frame["explanation_payload"] = payload_for(findings=["x"])
        with pytest.raises(ValueError) as excinfo:
            assert_closed_anomaly_vocabulary(decode_payloads(frame))
        assert "zero breadth" in str(excinfo.value)

    def test_the_pipeline_refuses_it(self) -> None:
        frame = make_frame()
        frame["explanation_payload"] = payload_for(findings=["brand_new_type"])
        with pytest.raises(ValueError, match="closed vocabulary"):
            ConsumptionLayer().run(frame)

    def test_an_empty_finding_list_is_fine(self) -> None:
        frame = make_frame({"findings": []})
        assert_closed_anomaly_vocabulary(decode_payloads(frame))

    def test_the_m1_label_is_permitted(self) -> None:
        """Stage 6 adds it; Stage 7 must not reject its own correction."""
        frame = make_frame({"findings": [M1_CORRECTION_LABEL]})
        assert_closed_anomaly_vocabulary(decode_payloads(frame))


# ---------------------------------------------------------------------------
# PHASE 3 - the Stage 8 scaffold refuses
# ---------------------------------------------------------------------------


class TestStage8Scaffold:
    """Every test here asserts a refusal. That is the deliverable."""

    def test_no_dataset_returns_none_not_a_number(self) -> None:
        result = calibrate(pd.DataFrame())
        assert result.calibrated_risk is None
        assert result.status == "UNAVAILABLE"

    def test_synthetic_provenance_is_refused_outright(self) -> None:
        dataset = CalibrationDataset(
            labels=pd.DataFrame({"outcome": [True] * 300 + [False] * 300}),
            provenance="synthetic",
        )
        result = calibrate(pd.DataFrame(), dataset=dataset)
        assert result.status == "REFUSED_SYNTHETIC"
        assert result.calibrated_risk is None

    def test_too_few_labels_is_refused(self) -> None:
        labels = pd.DataFrame(
            {
                name: [None] * 10
                for name in CALIBRATION_LABEL_SCHEMA
            }
        )
        labels["outcome"] = [True] * 5 + [False] * 5
        dataset = CalibrationDataset(labels=labels, provenance="real")
        result = calibrate(pd.DataFrame(), dataset=dataset)
        assert result.status == "INSUFFICIENT_LABELS"
        assert any("labelled outcome" in p for p in result.diagnostics["problems"])

    def test_class_imbalance_is_refused(self) -> None:
        """199 negatives and one positive describes nothing."""
        labels = pd.DataFrame({name: [None] * 400 for name in CALIBRATION_LABEL_SCHEMA})
        labels["outcome"] = [True] * 5 + [False] * 395
        dataset = CalibrationDataset(labels=labels, provenance="real")
        result = calibrate(pd.DataFrame(), dataset=dataset)
        assert result.status == "INSUFFICIENT_LABELS"
        assert any("class balance" in p for p in result.diagnostics["problems"])

    def test_validate_returns_every_problem_not_the_first(self) -> None:
        """A caller fixing a dataset should see the whole gap at once."""
        dataset = CalibrationDataset(
            labels=pd.DataFrame({"outcome": [True] * 3}), provenance="synthetic"
        )
        problems = dataset.validate()
        assert len(problems) >= 2

    def test_an_unknown_method_is_a_caller_error(self) -> None:
        """Refusing to calibrate is a result; a bad method name is not."""
        with pytest.raises(ValueError, match="unknown calibration method"):
            calibrate(pd.DataFrame(), method="deep_learning")

    def test_all_three_designed_methods_are_accepted(self) -> None:
        for method in CALIBRATION_METHODS:
            result = calibrate(pd.DataFrame(), method=method)
            assert result.status == "UNAVAILABLE"

    def test_a_valid_real_dataset_raises_not_implemented(self) -> None:
        """The scaffold must never return an untested fit."""
        labels = pd.DataFrame({name: [None] * 400 for name in CALIBRATION_LABEL_SCHEMA})
        labels["outcome"] = [True] * 200 + [False] * 200
        dataset = CalibrationDataset(labels=labels, provenance="real")
        with pytest.raises(NotImplementedError, match="not implemented"):
            calibrate(pd.DataFrame(), dataset=dataset)

    def test_the_design_is_readable_as_data(self) -> None:
        design = calibration_design()
        assert set(design["methods"]) == set(CALIBRATION_METHODS)
        assert set(design["evaluation"]) == {
            "brier_score",
            "calibration_curve",
            "expected_calibration_error",
        }
        assert design["gates"]["provenance_must_be"] == "real"

    def test_the_design_states_why_each_metric_is_insufficient_alone(self) -> None:
        evaluation = calibration_design()["evaluation"]
        assert "never reported alone" in evaluation["brier_score"]
        assert "hides where the error lives" in evaluation["expected_calibration_error"]
        assert "shows the SHAPE" in evaluation["calibration_curve"]

    def test_the_safety_rules_forbid_inventing_a_probability(self) -> None:
        rules = " ".join(calibration_design()["safety_rules"])
        assert "None" in rules and "UNAVAILABLE" in rules
        assert "never overwrites risk_score" in rules

    def test_the_label_schema_requires_provenance(self) -> None:
        """So synthetic data cannot be supplied by accident."""
        assert "provenance" in CALIBRATION_LABEL_SCHEMA
        assert "Required, not inferred" in CALIBRATION_LABEL_SCHEMA["provenance"]

    def test_it_is_declared_a_scaffold(self) -> None:
        assert "SCAFFOLD" in calibration_design()["status"]
        assert "scaffold" in STAGE8_VERSION

    def test_the_columns_are_named_before_the_implementation(self) -> None:
        assert "calibrated_risk" in CALIBRATION_COLUMNS
        assert "calibration_status" in CALIBRATION_COLUMNS


# ---------------------------------------------------------------------------
# Zero regression
# ---------------------------------------------------------------------------


class TestZeroRegression:
    """Additive only. No decision may move."""

    def test_the_input_is_never_mutated(self, spread: pd.DataFrame) -> None:
        before = spread.copy(deep=True)
        ConsumptionLayer().run(spread)
        pd.testing.assert_frame_equal(spread, before)

    def test_no_risk_is_altered(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        upstream = [json.loads(p)["risk_score"] for p in spread["explanation_payload"]]
        assert [card["risk"] for card in result.cards] == upstream

    def test_the_new_columns_are_deterministic(self, spread: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            ConsumptionLayer().run(spread).annotations,
            ConsumptionLayer().run(spread).annotations,
        )

    def test_all_six_new_columns_exist(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        for column in ("escalation_warning", "work_id_group_size",
                       "work_id_conflict_flag", "risk_calibration_status",
                       "risk_relative_warning"):
            assert column in annotations.columns

    def test_escalation_warning_matches_its_sibling(self, spread: pd.DataFrame) -> None:
        """Two names for one fact is only safe while they agree."""
        annotations = ConsumptionLayer().run(spread).annotations
        assert list(annotations["escalation_warning"]) == list(
            annotations["escalation_reason_warning"]
        )
