"""Stage 7 hardening - seven measured risks, made visible.

This pass changed no decision. Its tests therefore prove two things:

* that every known limitation now appears **in the output** rather than only
  in a roadmap, and
* that making it appear changed nothing - `TestZeroMutation` is the one that
  would catch a transparency layer quietly becoming a decision layer.

The annotations are derived from the Stage 6 payload alone. If any test here
could be satisfied by reading the human explanation string, the layer would
have inherited the ambiguity that string was proven to carry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    ACTION_CLASSES,
    ACTION_TO_SEMANTIC_TYPE,
    ACTION_SPEC_LOSSY_WARNING,
    CALIBRATION_WARNING,
    CONFIDENCE_GATE_THRESHOLD,
    CONFIG_DEPENDENCY_WARNING,
    DECISION_CLARITY_FLAGS,
    M1_CORRECTION_LABEL,
    MIN_CONFIDENCE_FOR_RISK,
    PRIORITY_SEMANTIC_TYPES,
    R_HIGH,
    R_LOW,
    REASON_FLAGS,
    SPEC_ACTION_ALIAS,
    UNEXPLAINED_REASON_DETAIL,
    WORK_ID_AMBIGUITY_WARNING,
)
from src.stage7.annotations import (
    ANNOTATION_COLUMNS,
    build_annotations,
    build_system_metadata,
    build_transparency_metrics,
    build_work_level_summary,
)
from src.stage7.interface import decode_payloads
from src.stage7.pipeline import ConsumptionLayer, Stage7InvariantError

from tests.test_stage7 import ONE_PER_ACTION, make_frame, payload_for


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """One record per action, plus an unexplained escalation."""
    return make_frame(
        *ONE_PER_ACTION,
        {"action_class": "ESCALATE_IMMEDIATE", "priority_level": "P0",
         "findings": [M1_CORRECTION_LABEL], "work_id": "shared-work"},
        {"action_class": "PASSIVE_MONITOR", "priority_level": "P3",
         "risk": 0.02, "risk_flag": "low_risk", "decision": "MONITOR",
         "findings": [], "work_id": "shared-work"},
    )


# ---------------------------------------------------------------------------
# R1 - unexplained escalations
# ---------------------------------------------------------------------------


class TestUnexplainedEscalations:
    """18 of 419 escalations carry no named finding. 4 are P0."""

    def test_the_m1_label_is_flagged(self) -> None:
        frame = make_frame({"findings": [M1_CORRECTION_LABEL]})
        row = ConsumptionLayer().run(frame).annotations.iloc[0]
        assert row["stage7_reason_flag"] == "UNEXPLAINED_DEVIATION"

    def test_a_named_finding_is_not_flagged(self) -> None:
        frame = make_frame({"findings": ["cost_outlier"]})
        row = ConsumptionLayer().run(frame).annotations.iloc[0]
        assert row["stage7_reason_flag"] == "EXPLAINED"

    def test_the_detail_states_what_is_missing(self) -> None:
        frame = make_frame({"findings": [M1_CORRECTION_LABEL]})
        detail = ConsumptionLayer().run(frame).annotations.iloc[0][
            "stage7_reason_detail"
        ]
        assert detail == UNEXPLAINED_REASON_DETAIL
        assert "no specific anomaly category was assigned upstream" in detail.lower()

    def test_the_detail_does_not_claim_the_deviation_is_false(self) -> None:
        """The deviation was measured and is real; only the label is absent."""
        assert "was measured and is real" in UNEXPLAINED_REASON_DETAIL
        assert "requiring manual characterisation" in UNEXPLAINED_REASON_DETAIL

    def test_a_mixed_finding_list_still_flags(self) -> None:
        frame = make_frame({"findings": ["cost_outlier", M1_CORRECTION_LABEL]})
        row = ConsumptionLayer().run(frame).annotations.iloc[0]
        assert row["stage7_reason_flag"] == "UNEXPLAINED_DEVIATION"

    def test_only_declared_flags_are_emitted(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert set(annotations["stage7_reason_flag"]) <= set(REASON_FLAGS)

    def test_the_rate_is_published(self, spread: pd.DataFrame) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        assert "pct_unexplained_deviation" in metrics
        assert "pct_escalations_without_named_anomaly" in metrics
        assert metrics["n_escalations_without_named_anomaly"] >= 1


# ---------------------------------------------------------------------------
# R2 - lossy action_spec
# ---------------------------------------------------------------------------


class TestLossyAliasWarning:
    """action_spec merges 291 P0 referrals with 128 P1 reviews."""

    def test_truth_class_equals_the_action_exactly(
        self, spread: pd.DataFrame
    ) -> None:
        result = ConsumptionLayer().run(spread)
        actions = [json.loads(p)["action"] for p in spread["explanation_payload"]]
        assert list(result.annotations["action_truth_class"]) == actions

    @pytest.mark.parametrize("action", ["ESCALATE_IMMEDIATE", "ESCALATE_REVIEW"])
    def test_a_collapsed_action_is_warned(self, action: str) -> None:
        frame = make_frame({"action_class": action, "priority_level": "P1"})
        warning = ConsumptionLayer().run(frame).annotations.iloc[0][
            "action_interpretation_warning"
        ]
        assert warning == ACTION_SPEC_LOSSY_WARNING
        assert "action_truth_class" in warning

    @pytest.mark.parametrize(
        "action", ["PASSIVE_MONITOR", "REQUEST_CORRECTION", "DATA_QUALITY_REVIEW"]
    )
    def test_a_one_to_one_action_is_not_warned(self, action: str) -> None:
        """A false warning is as bad as a missing one."""
        frame = make_frame({"action_class": action, "priority_level": "P2"})
        assert ConsumptionLayer().run(frame).annotations.iloc[0][
            "action_interpretation_warning"
        ] is None

    def test_the_warning_set_is_derived_not_hardcoded(self) -> None:
        """It must stop warning by itself if the alias becomes one-to-one."""
        collapsed = {
            action
            for action, alias in SPEC_ACTION_ALIAS.items()
            if list(SPEC_ACTION_ALIAS.values()).count(alias) > 1
        }
        frames = [
            make_frame({"action_class": action, "priority_level": "P1"})
            for action in ACTION_CLASSES
        ]
        warned = {
            action
            for action, frame in zip(ACTION_CLASSES, frames)
            if ConsumptionLayer().run(frame).annotations.iloc[0][
                "action_interpretation_warning"
            ]
            is not None
        }
        assert warned == collapsed


# ---------------------------------------------------------------------------
# R3 - overloaded P1
# ---------------------------------------------------------------------------


class TestPrioritySemanticSplit:
    """P1 mixes 3,402 data-quality records with 128 audit escalations."""

    @pytest.mark.parametrize("action,expected", list(ACTION_TO_SEMANTIC_TYPE.items()))
    def test_each_action_maps_to_its_type(self, action: str, expected: str) -> None:
        frame = make_frame({"action_class": action, "priority_level": "P1"})
        assert ConsumptionLayer().run(frame).annotations.iloc[0][
            "priority_semantic_type"
        ] == expected

    def test_p1_is_split_into_distinct_meanings(self) -> None:
        """The decisive test: two P1 records, two different kinds of work."""
        frame = make_frame(
            {"action_class": "ESCALATE_REVIEW", "priority_level": "P1"},
            {"action_class": "DATA_QUALITY_REVIEW", "priority_level": "P1"},
        )
        result = ConsumptionLayer().run(frame)
        types = list(result.annotations["priority_semantic_type"])
        assert types == ["ESCALATION", "DATA_QUALITY"]
        assert len(set(types)) == 2

    def test_only_declared_types_are_emitted(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert set(annotations["priority_semantic_type"]) <= set(
            PRIORITY_SEMANTIC_TYPES
        )

    def test_the_split_is_published_in_the_report(self, spread: pd.DataFrame) -> None:
        report = ConsumptionLayer().run(spread).report()
        assert "priority_semantic_split" in report
        assert sum(report["priority_semantic_split"].values()) == len(spread)

    def test_percentages_are_published(self, spread: pd.DataFrame) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        assert set(metrics["by_priority_semantic_type"]) == set(
            PRIORITY_SEMANTIC_TYPES
        )
        assert sum(metrics["by_priority_semantic_type"].values()) == len(spread)


# ---------------------------------------------------------------------------
# R4 - configuration dependency
# ---------------------------------------------------------------------------


class TestConfigVisibility:
    """The gate coupling is invisible in the records themselves."""

    def test_metadata_is_returned_with_outputs(self, spread: pd.DataFrame) -> None:
        metadata = ConsumptionLayer().run(spread).system_metadata
        assert metadata["thresholds"]["r_high"] == R_HIGH
        assert metadata["thresholds"]["r_low"] == R_LOW
        assert (
            metadata["thresholds"]["min_confidence_stage5_gate"]
            == MIN_CONFIDENCE_FOR_RISK
        )

    def test_the_warning_is_present(self, spread: pd.DataFrame) -> None:
        metadata = ConsumptionLayer().run(spread).system_metadata
        assert metadata["_warning"] == CONFIG_DEPENDENCY_WARNING
        assert "identical configuration" in metadata["_warning"]

    def test_gate_alignment_is_reported_as_a_fact(self, spread: pd.DataFrame) -> None:
        """Not asserted - reported, so a divergence is visible not fatal here."""
        metadata = ConsumptionLayer().run(spread).system_metadata
        assert metadata["thresholds"]["gates_aligned"] is (
            CONFIDENCE_GATE_THRESHOLD == MIN_CONFIDENCE_FOR_RISK
        )
        assert "must be true" in metadata["_gate_note"]

    def test_metadata_reaches_the_report(self, spread: pd.DataFrame) -> None:
        report = ConsumptionLayer().run(spread).report()
        assert report["system_metadata"]["thresholds"]["r_high"] == R_HIGH


# ---------------------------------------------------------------------------
# R5 - duplicate work_id
# ---------------------------------------------------------------------------


class TestWorkLevelSummary:
    """200 records share a work_id; 56 groups get different actions."""

    def test_duplicates_are_aggregated_into_one_row(self, spread: pd.DataFrame) -> None:
        summary = ConsumptionLayer().run(spread).work_summary
        shared = summary[summary["work_id"] == "shared-work"]
        assert len(shared) == 1
        assert int(shared.iloc[0]["total_records"]) == 2

    def test_risk_is_the_maximum_never_the_mean(self) -> None:
        """Averaging would let a clean twin dilute a high-risk record."""
        frame = make_frame(
            {"work_id": "w", "risk": 0.9},
            {"work_id": "w", "risk": 0.1},
        )
        row = ConsumptionLayer().run(frame).work_summary.iloc[0]
        assert row["max_risk_score"] == pytest.approx(0.9)
        assert row["max_risk_score"] != pytest.approx(0.5)

    def test_conflicting_actions_are_flagged(self, spread: pd.DataFrame) -> None:
        """The fact a consumer most needs and is least likely to check."""
        summary = ConsumptionLayer().run(spread).work_summary
        shared = summary[summary["work_id"] == "shared-work"].iloc[0]
        assert bool(shared["has_conflicting_actions"]) is True

    def test_a_single_action_group_is_not_flagged(self) -> None:
        frame = make_frame({"work_id": "w"}, {"work_id": "w"})
        row = ConsumptionLayer().run(frame).work_summary.iloc[0]
        assert bool(row["has_conflicting_actions"]) is False

    def test_the_distribution_is_carried_beside_the_maximum(
        self, spread: pd.DataFrame
    ) -> None:
        summary = ConsumptionLayer().run(spread).work_summary
        shared = summary[summary["work_id"] == "shared-work"].iloc[0]
        assert sum(shared["action_distribution"].values()) == 2
        assert isinstance(shared["anomaly_distribution"], dict)

    def test_an_all_unscored_group_has_no_max(self) -> None:
        """None, never 0.0 - the distinction the system exists to keep."""
        frame = make_frame(
            {"work_id": "w", "risk": None, "risk_flag": "insufficient_data"},
            {"work_id": "w", "risk": None, "risk_flag": "insufficient_data"},
        )
        row = ConsumptionLayer().run(frame).work_summary.iloc[0]
        assert row["max_risk_score"] is None
        assert int(row["n_records_scored"]) == 0

    def test_the_dominant_action_is_deterministic_under_ties(self) -> None:
        frame = make_frame(
            {"work_id": "w", "action_class": "PASSIVE_MONITOR", "priority_level": "P3"},
            {"work_id": "w", "action_class": "ESCALATE_REVIEW", "priority_level": "P1"},
        )
        first = ConsumptionLayer().run(frame).work_summary.iloc[0]
        second = ConsumptionLayer().run(frame).work_summary.iloc[0]
        assert first["dominant_action_class"] == second["dominant_action_class"]

    def test_escalations_are_counted_per_work(self, spread: pd.DataFrame) -> None:
        summary = ConsumptionLayer().run(spread).work_summary
        shared = summary[summary["work_id"] == "shared-work"].iloc[0]
        assert int(shared["escalation_count"]) == 1

    def test_record_level_identity_is_untouched(self, spread: pd.DataFrame) -> None:
        """Aggregation is a view; every record still routes individually."""
        result = ConsumptionLayer().run(spread)
        assert len(result.cards) == len(spread)
        ids = [item.record_id for items in result.queues.values() for item in items]
        assert sorted(ids) == sorted(spread.index)


# ---------------------------------------------------------------------------
# R6 - decision clarity
# ---------------------------------------------------------------------------


class TestDecisionClarity:
    """A reviewer must be able to tell 'no issue' from 'cannot assess'."""

    def test_unexplained_is_ambiguous(self) -> None:
        frame = make_frame({"findings": [M1_CORRECTION_LABEL]})
        assert ConsumptionLayer().run(frame).annotations.iloc[0][
            "decision_clarity_flag"
        ] == "AMBIGUOUS"

    def test_unscored_is_data_limited(self) -> None:
        frame = make_frame({"risk": None, "risk_flag": "insufficient_data"})
        assert ConsumptionLayer().run(frame).annotations.iloc[0][
            "decision_clarity_flag"
        ] == "DATA_LIMITED"

    def test_an_ordinary_record_is_clear(self) -> None:
        assert ConsumptionLayer().run(make_frame()).annotations.iloc[0][
            "decision_clarity_flag"
        ] == "CLEAR"

    def test_a_zero_risk_is_clear_not_data_limited(self) -> None:
        """Measured as zero is a real answer; unmeasured is not."""
        frame = make_frame({"risk": 0.0, "risk_flag": "low_risk"})
        assert ConsumptionLayer().run(frame).annotations.iloc[0][
            "decision_clarity_flag"
        ] == "CLEAR"

    def test_only_declared_flags_are_emitted(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert set(annotations["decision_clarity_flag"]) <= set(DECISION_CLARITY_FLAGS)

    def test_the_two_failure_modes_are_mutually_exclusive(
        self, spread: pd.DataFrame
    ) -> None:
        """Guaranteed by construction; asserted so a change would show."""
        result = ConsumptionLayer().run(spread)
        for annotation, card in zip(
            result.annotations.to_dict(orient="records"), result.cards
        ):
            if annotation["decision_clarity_flag"] == "AMBIGUOUS":
                assert card["risk"] is not None


# ---------------------------------------------------------------------------
# R7 + Task 8 - calibration warning and the enhanced explanation
# ---------------------------------------------------------------------------


class TestCalibrationWarning:
    """0.5 reads as a probability. It is not one."""

    def test_every_record_carries_it(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert annotations["calibration_warning"].eq(CALIBRATION_WARNING).all()

    def test_it_reaches_the_report_and_metadata(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        assert result.report()["calibration_warning"] == CALIBRATION_WARNING
        assert result.system_metadata["_calibration"] == CALIBRATION_WARNING

    def test_it_says_the_thresholds_are_unvalidated(self) -> None:
        assert "NOT calibrated probabilities" in CALIBRATION_WARNING
        assert "not validated on real-world data" in CALIBRATION_WARNING


class TestEnhancedExplanation:
    """Clearer than Stage 6's, and derived from the payload alone."""

    def test_escalation_and_review_read_differently(self) -> None:
        immediate = ConsumptionLayer().run(
            make_frame({"action_class": "ESCALATE_IMMEDIATE", "priority_level": "P0"})
        ).annotations.iloc[0]["stage7_explanation"]
        review = ConsumptionLayer().run(
            make_frame({"action_class": "ESCALATE_REVIEW", "priority_level": "P1"})
        ).annotations.iloc[0]["stage7_explanation"]
        assert "ACT NOW" in immediate
        assert "REVIEW" in review
        assert "not an urgent fraud referral" in review
        assert immediate != review

    def test_a_data_problem_is_not_described_as_an_anomaly(self) -> None:
        text = ConsumptionLayer().run(
            make_frame({"action_class": "REQUEST_CORRECTION", "priority_level": "P2"})
        ).annotations.iloc[0]["stage7_explanation"]
        assert "the problem here is the data" in text
        assert "Nothing about the work itself has been concluded" in text

    def test_unassessed_is_never_rendered_as_clean(self) -> None:
        text = ConsumptionLayer().run(
            make_frame({"action_class": "DATA_QUALITY_REVIEW", "priority_level": "P1",
                        "risk": None, "risk_flag": "insufficient_data"})
        ).annotations.iloc[0]["stage7_explanation"]
        assert "CANNOT ASSESS" in text
        assert "NOT a clean record" in text
        assert "could not be measured, NOT that it was measured and found safe" in text

    def test_a_present_score_is_qualified_as_uncalibrated(self) -> None:
        text = ConsumptionLayer().run(make_frame()).annotations.iloc[0][
            "stage7_explanation"
        ]
        assert "uncalibrated scale" in text
        assert "not a probability" in text

    def test_an_unexplained_escalation_says_so(self) -> None:
        text = ConsumptionLayer().run(
            make_frame({"findings": [M1_CORRECTION_LABEL]})
        ).annotations.iloc[0]["stage7_explanation"]
        assert "No named anomaly category was assigned" in text
        assert "characterise it manually" in text

    def test_it_never_reads_the_human_explanation(self) -> None:
        """Corrupt Stage 6's sentence; the Stage 7 one must not move."""
        clean = make_frame()
        corrupt = clean.copy()
        corrupt.loc[0, "explanation"] = "GARBAGE with, none recorded nonsense"
        a = ConsumptionLayer().run(clean).annotations.iloc[0]["stage7_explanation"]
        b = ConsumptionLayer().run(corrupt).annotations.iloc[0]["stage7_explanation"]
        assert a == b

    def test_it_names_the_upstream_decision(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        for annotation, raw in zip(
            result.annotations.to_dict(orient="records"),
            spread["explanation_payload"],
        ):
            decision = json.loads(raw)["decision_class"]
            assert decision in annotation["stage7_explanation"]


# ---------------------------------------------------------------------------
# Task 9 - transparency metrics
# ---------------------------------------------------------------------------


class TestTransparencyMetrics:
    """The limitations, published as rates."""

    def test_all_four_required_metrics_are_present(
        self, spread: pd.DataFrame
    ) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        for name in (
            "pct_insufficient_data",
            "pct_unexplained_deviation",
            "pct_escalations_without_named_anomaly",
            "pct_by_priority_semantic_type",
        ):
            assert name in metrics

    def test_percentages_are_in_range(self, spread: pd.DataFrame) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        for name in ("pct_insufficient_data", "pct_unexplained_deviation"):
            assert 0.0 <= metrics[name] <= 100.0

    def test_the_semantic_split_sums_to_one_hundred(
        self, spread: pd.DataFrame
    ) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        assert sum(metrics["pct_by_priority_semantic_type"].values()) == pytest.approx(
            100.0, abs=1e-3
        )

    def test_clarity_counts_sum_to_the_corpus(self, spread: pd.DataFrame) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        assert sum(metrics["by_decision_clarity"].values()) == len(spread)

    def test_an_empty_corpus_is_handled(self) -> None:
        metrics = build_transparency_metrics(
            pd.DataFrame(columns=list(ANNOTATION_COLUMNS)), pd.Series(dtype="object")
        )
        assert metrics["n_records"] == 0

    def test_metrics_reach_the_report(self, spread: pd.DataFrame) -> None:
        report = ConsumptionLayer().run(spread).report()
        assert "transparency_metrics" in report
        assert report["transparency_metrics"]["n_records"] == len(spread)


# ---------------------------------------------------------------------------
# Task 10 - zero mutation
# ---------------------------------------------------------------------------


class TestZeroMutation:
    """The test that would catch a transparency layer becoming a decision one."""

    def test_the_input_frame_is_untouched(self, spread: pd.DataFrame) -> None:
        before = spread.copy(deep=True)
        ConsumptionLayer().run(spread)
        pd.testing.assert_frame_equal(spread, before)

    def test_no_action_is_altered(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        upstream = [json.loads(p)["action"] for p in spread["explanation_payload"]]
        assert [card["action"] for card in result.cards] == upstream
        assert list(result.annotations["action_truth_class"]) == upstream

    def test_no_priority_is_altered(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        upstream = [json.loads(p)["priority"] for p in spread["explanation_payload"]]
        assert [r["priority"] for r in result.api_responses] == upstream

    def test_no_risk_is_altered(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        upstream = [json.loads(p)["risk_score"] for p in spread["explanation_payload"]]
        assert [card["risk"] for card in result.cards] == upstream

    def test_queue_depths_are_unchanged_by_annotation(
        self, spread: pd.DataFrame
    ) -> None:
        """Annotations are built after routing and cannot feed back into it."""
        result = ConsumptionLayer().run(spread)
        for name, items in result.queues.items():
            for item in items:
                assert item.action in ACTION_CLASSES

    def test_annotations_are_deterministic(self, spread: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            ConsumptionLayer().run(spread).annotations,
            ConsumptionLayer().run(spread).annotations,
        )

    def test_the_work_summary_is_deterministic(self, spread: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            ConsumptionLayer().run(spread).work_summary,
            ConsumptionLayer().run(spread).work_summary,
        )

    def test_metrics_serialise_identically(self, spread: pd.DataFrame) -> None:
        first = json.dumps(
            ConsumptionLayer().run(spread).transparency_metrics, sort_keys=True
        )
        second = json.dumps(
            ConsumptionLayer().run(spread).transparency_metrics, sort_keys=True
        )
        assert first == second

    def test_every_annotation_column_is_produced(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert list(annotations.columns) == list(ANNOTATION_COLUMNS)
        assert len(annotations) == len(spread)

    def test_the_invariant_catches_a_corrupted_annotation(
        self, spread: pd.DataFrame
    ) -> None:
        layer = ConsumptionLayer()
        result = layer.run(spread)
        result.annotations.loc[result.annotations.index[0], "action_truth_class"] = (
            "PASSIVE_MONITOR"
        )
        with pytest.raises(Stage7InvariantError, match="I7"):
            layer._assert_guarantees(result, spread, decode_payloads(spread))

    def test_artefacts_are_written(self, spread: pd.DataFrame, tmp_path: Path) -> None:
        written = ConsumptionLayer().run(spread).save(tmp_path)
        assert "annotations" in written and "work_summary" in written
        loaded = json.loads(written["annotations"].read_text(encoding="utf-8"))
        assert loaded["calibration_warning"] == CALIBRATION_WARNING
        assert len(loaded["records"]) == len(spread)
        works = json.loads(written["work_summary"].read_text(encoding="utf-8"))
        assert works["_note"] == WORK_ID_AMBIGUITY_WARNING


# ---------------------------------------------------------------------------
# Integration - the measured rates on real data
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    """The seven risks, on real Stage 1-6 output."""

    @pytest.fixture(scope="class")
    def corpus(self) -> Any:
        from src.stage1.corpus import Corpus
        from src.stage1.data_generator import generate_dataset
        from src.stage2.confidence import attach_confidence
        from src.stage3.pipeline import attach_structure
        from src.stage4.pipeline import AnomalyConfig, attach_anomalies
        from src.stage5.pipeline import RiskConfig, attach_risk
        from src.stage6.pipeline import attach_actions

        built = Corpus.from_dataframe(generate_dataset(n=3000, seed=42))
        attach_confidence(built)
        attach_structure(built)
        attach_anomalies(built, config=AnomalyConfig(compute_calibration=False))
        attach_risk(built, config=RiskConfig(compute_calibration=False))
        attach_actions(built)
        return built

    def test_the_corpus_is_untouched(self, corpus: Any) -> None:
        before = corpus.records.copy(deep=True)
        ConsumptionLayer().run(corpus)
        pd.testing.assert_frame_equal(corpus.records, before)

    def test_unexplained_escalations_are_found_and_flagged(self, corpus: Any) -> None:
        result = ConsumptionLayer().run(corpus)
        flagged = result.annotations["stage7_reason_flag"] == "UNEXPLAINED_DEVIATION"
        assert int(flagged.sum()) > 0, "the R1 gap is real on this corpus"
        assert result.transparency_metrics["n_escalations_without_named_anomaly"] > 0

    def test_duplicate_work_ids_are_aggregated(self, corpus: Any) -> None:
        result = ConsumptionLayer().run(corpus)
        assert len(result.work_summary) < len(corpus)
        assert int(result.work_summary["total_records"].sum()) == len(corpus)

    def test_conflicting_actions_per_work_are_surfaced(self, corpus: Any) -> None:
        summary = ConsumptionLayer().run(corpus).work_summary
        assert "has_conflicting_actions" in summary.columns

    def test_all_semantic_types_appear(self, corpus: Any) -> None:
        annotations = ConsumptionLayer().run(corpus).annotations
        assert set(annotations["priority_semantic_type"]) <= set(
            PRIORITY_SEMANTIC_TYPES
        )

    def test_it_is_deterministic_on_real_data(self, corpus: Any) -> None:
        pd.testing.assert_frame_equal(
            ConsumptionLayer().run(corpus).annotations,
            ConsumptionLayer().run(corpus).annotations,
        )
