"""Surgical correction pass - R1 to R4.

Each test here corresponds to a **measured** failure, not a hypothetical one.
The counts asserted (18 unexplained escalations, 419 escalations, 3,402
data-quality records, 419 lossy aliases) come from the reference corpus and
are pinned so a regression is visible as a number rather than a feeling.

`TestR2EscalationPolicy` is the one that matters. It is the fix for the only
break in this system that was ever demonstrated live: a legal config change
producing 73 escalated records with no risk score. Stage 7 now refuses to hand
any of them to a person.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    ACTION_CLASSES,
    ACTION_GROUPS,
    ACTION_SPEC_LOSSY_NOTE,
    ACTION_TO_GROUP,
    ACTION_TO_SEMANTIC_TYPE,
    CONFIDENCE_GATE_THRESHOLD,
    ESCALATION_REASON_STATUSES,
    ESCALATION_UNEXPLAINED_WARNING,
    M1_CORRECTION_LABEL,
    MIN_CONFIDENCE_FOR_RISK,
)
from src.stage7.pipeline import ConsumptionLayer
from src.stage7.policy import (
    Stage7PolicyError,
    escalation_policy_report,
    validate_escalation_policy,
)

from tests.test_stage7 import ONE_PER_ACTION, make_frame, payload_for


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """One record per action, plus an unexplained escalation."""
    return make_frame(
        *ONE_PER_ACTION,
        {"action_class": "ESCALATE_IMMEDIATE", "priority_level": "P0",
         "findings": [M1_CORRECTION_LABEL]},
    )


# ---------------------------------------------------------------------------
# R1 - unexplained escalations
# ---------------------------------------------------------------------------


class TestR1UnexplainedEscalations:
    """18 escalations (4 P0) carry no named anomaly. No cause is invented."""

    def test_an_unexplained_escalation_is_flagged(self) -> None:
        frame = make_frame({"action_class": "ESCALATE_IMMEDIATE",
                            "priority_level": "P0",
                            "findings": [M1_CORRECTION_LABEL]})
        row = ConsumptionLayer().run(frame).annotations.iloc[0]
        assert row["escalation_reason_status"] == "unexplained_upstream"

    def test_a_named_escalation_reads_explained(self) -> None:
        frame = make_frame({"action_class": "ESCALATE_IMMEDIATE",
                            "priority_level": "P0",
                            "findings": ["cost_outlier"]})
        row = ConsumptionLayer().run(frame).annotations.iloc[0]
        assert row["escalation_reason_status"] == "explained"
        assert row["escalation_reason_warning"] is None

    def test_the_warning_names_stage_4_lifecycle_gating(self) -> None:
        frame = make_frame({"action_class": "ESCALATE_IMMEDIATE",
                            "priority_level": "P0",
                            "findings": [M1_CORRECTION_LABEL]})
        warning = ConsumptionLayer().run(frame).annotations.iloc[0][
            "escalation_reason_warning"
        ]
        assert warning == ESCALATION_UNEXPLAINED_WARNING
        assert "no named anomaly" in warning
        assert "Stage 4 lifecycle gating" in warning

    def test_no_cause_is_fabricated(self) -> None:
        """The warning states what is missing, never what it might have been."""
        assert "declined to assign it a category" in ESCALATION_UNEXPLAINED_WARNING
        assert "must characterise it manually" in ESCALATION_UNEXPLAINED_WARNING
        for invented in ("fraud", "corruption", "likely", "probably", "suspected"):
            assert invented not in ESCALATION_UNEXPLAINED_WARNING.lower()

    def test_a_non_escalation_is_never_called_unexplained(self) -> None:
        """A monitored record with no findings is correct, not unexplained."""
        frame = make_frame({"action_class": "PASSIVE_MONITOR",
                            "priority_level": "P3", "findings": []})
        row = ConsumptionLayer().run(frame).annotations.iloc[0]
        assert row["escalation_reason_status"] == "explained"
        assert row["escalation_reason_warning"] is None

    def test_only_declared_statuses_are_emitted(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert set(annotations["escalation_reason_status"]) <= set(
            ESCALATION_REASON_STATUSES
        )

    def test_the_count_is_published(self, spread: pd.DataFrame) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        assert metrics["n_unexplained_upstream"] == 1

    def test_stage_4_output_is_not_modified(self) -> None:
        frame = make_frame({"action_class": "ESCALATE_IMMEDIATE",
                            "priority_level": "P0",
                            "findings": [M1_CORRECTION_LABEL]})
        before = frame.copy(deep=True)
        ConsumptionLayer().run(frame)
        pd.testing.assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# R2 - config fragility (the only live break ever demonstrated)
# ---------------------------------------------------------------------------


class TestR2EscalationPolicy:
    """An escalation with no risk score must never reach a reviewer."""

    def _violating(self, n: int = 3) -> pd.DataFrame:
        rows = [
            {"action_class": "ESCALATE_IMMEDIATE", "priority_level": "P0",
             "risk": None, "risk_flag": "insufficient_data"}
            for _ in range(n)
        ]
        frame = make_frame(*rows)
        frame["decision_class"] = "INVESTIGATE"
        frame["risk_defined"] = False
        return frame

    def _clean(self) -> pd.DataFrame:
        frame = make_frame({"action_class": "ESCALATE_IMMEDIATE",
                            "priority_level": "P0"})
        frame["decision_class"] = "INVESTIGATE"
        frame["risk_defined"] = True
        return frame

    def test_a_clean_frame_passes(self) -> None:
        validate_escalation_policy(self._clean())

    def test_a_violation_raises(self) -> None:
        with pytest.raises(Stage7PolicyError):
            validate_escalation_policy(self._violating())

    def test_the_error_carries_the_count(self) -> None:
        with pytest.raises(Stage7PolicyError, match="3 record"):
            validate_escalation_policy(self._violating(3))

    def test_the_error_carries_sample_indices(self) -> None:
        with pytest.raises(Stage7PolicyError) as excinfo:
            validate_escalation_policy(self._violating())
        assert "sample record ids" in str(excinfo.value)
        assert "[0, 1, 2]" in str(excinfo.value)

    def test_the_error_carries_both_gate_values(self) -> None:
        with pytest.raises(Stage7PolicyError) as excinfo:
            validate_escalation_policy(
                self._violating(), stage4_gate=0.5, stage5_gate=0.8
            )
        message = str(excinfo.value)
        assert "0.5" in message and "0.8" in message
        assert "constants aligned                  : False" in message
        assert "the constants themselves disagree" in message

    def test_the_error_does_not_claim_alignment_it_cannot_verify(self) -> None:
        """A per-run RiskConfig override is invisible to the constants.

        Reporting "aligned: True" beside a violation would send a reader
        looking in the wrong place, so the message labels the values as
        constants and names the override as the thing to check.
        """
        with pytest.raises(Stage7PolicyError) as excinfo:
            validate_escalation_policy(self._violating())
        message = str(excinfo.value)
        assert "(constant)" in message
        assert "per-run override" in message
        assert "RiskConfig(min_confidence=" in message

    def test_the_pipeline_refuses_before_building_anything(self) -> None:
        """A half-built result is harder to reason about than none."""
        with pytest.raises(Stage7PolicyError):
            ConsumptionLayer().run(self._violating())

    def test_it_fails_loudly_not_silently(self) -> None:
        """This is the fix. The alternative was 73 unfounded urgent leads."""
        try:
            ConsumptionLayer().run(self._violating())
        except Stage7PolicyError as exc:
            assert "must never reach a reviewer" in str(exc)
        else:  # pragma: no cover
            pytest.fail("the policy violation was not raised")

    def test_a_passing_run_still_states_the_invariant(self) -> None:
        """A guarantee only visible when it fails is one nobody knows they have.

        Built with the Stage 4/5 columns present; the module-level `spread`
        omits them on purpose, and a frame without them is left to the
        contract layer rather than reported on here.
        """
        frame = self._clean()
        report = ConsumptionLayer().run(frame).report()["escalation_policy"]
        assert report["checked"] is True
        assert report["n_investigate_without_risk"] == 0
        assert report["n_investigate"] == 1
        assert report["gates_aligned"] is True
        assert "must carry a risk score" in report["_invariant"]

    def test_a_frame_without_the_columns_reports_unchecked(
        self, spread: pd.DataFrame
    ) -> None:
        """Not silently 'passing': it says it did not check."""
        report = ConsumptionLayer().run(spread).report()["escalation_policy"]
        assert report["checked"] is False

    def test_the_report_form_agrees_with_the_raising_form(self) -> None:
        frame = self._violating()
        report = escalation_policy_report(frame)
        assert report["n_investigate_without_risk"] == 3
        with pytest.raises(Stage7PolicyError):
            validate_escalation_policy(frame)

    def test_missing_columns_are_left_to_the_contract_layer(self) -> None:
        """This function must not duplicate - or contradict - that diagnosis."""
        frame = make_frame().drop(columns=[], errors="ignore")
        validate_escalation_policy(frame)  # no decision_class: must not raise

    def test_the_gates_are_currently_aligned(self) -> None:
        assert CONFIDENCE_GATE_THRESHOLD == MIN_CONFIDENCE_FOR_RISK


# ---------------------------------------------------------------------------
# R3 - P1 overloading
# ---------------------------------------------------------------------------


class TestR3ActionGroup:
    """P1 mixes 128 escalation reviews with 3,402 data-quality records."""

    @pytest.mark.parametrize("action,expected", list(ACTION_TO_GROUP.items()))
    def test_each_action_maps_to_its_group(self, action: str, expected: str) -> None:
        frame = make_frame({"action_class": action, "priority_level": "P1"})
        assert ConsumptionLayer().run(frame).annotations.iloc[0][
            "action_group"
        ] == expected

    def test_both_escalations_share_one_group(self) -> None:
        assert ACTION_TO_GROUP["ESCALATE_IMMEDIATE"] == "escalation"
        assert ACTION_TO_GROUP["ESCALATE_REVIEW"] == "escalation"

    def test_p1_is_split(self) -> None:
        """Two P1 records, two different kinds of work."""
        frame = make_frame(
            {"action_class": "ESCALATE_REVIEW", "priority_level": "P1"},
            {"action_class": "DATA_QUALITY_REVIEW", "priority_level": "P1"},
        )
        groups = list(ConsumptionLayer().run(frame).annotations["action_group"])
        assert groups == ["escalation", "data_quality"]

    def test_only_declared_groups_are_emitted(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert set(annotations["action_group"]) <= set(ACTION_GROUPS)

    def test_it_agrees_with_the_existing_partition(self, spread: pd.DataFrame) -> None:
        """Two names for one fact is only safe while they agree."""
        annotations = ConsumptionLayer().run(spread).annotations
        assert [value.upper() for value in annotations["action_group"]] == list(
            annotations["priority_semantic_type"]
        )

    def test_counts_are_published(self, spread: pd.DataFrame) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        assert set(metrics["by_action_group"]) == set(ACTION_GROUPS)
        assert sum(metrics["by_action_group"].values()) == len(spread)


# ---------------------------------------------------------------------------
# R4 - lossy action_spec
# ---------------------------------------------------------------------------


class TestR4LossyActionSpec:
    """ESCALATE_IMMEDIATE and ESCALATE_REVIEW both become INVESTIGATE."""

    @pytest.mark.parametrize("action", ["ESCALATE_IMMEDIATE", "ESCALATE_REVIEW"])
    def test_the_two_escalations_are_flagged(self, action: str) -> None:
        frame = make_frame({"action_class": action, "priority_level": "P1"})
        assert bool(
            ConsumptionLayer().run(frame).annotations.iloc[0]["action_spec_lossy"]
        ) is True

    @pytest.mark.parametrize(
        "action", ["DATA_QUALITY_REVIEW", "REQUEST_CORRECTION", "PASSIVE_MONITOR"]
    )
    def test_no_other_action_is_flagged(self, action: str) -> None:
        """A false warning is as bad as a missing one."""
        frame = make_frame({"action_class": action, "priority_level": "P2"})
        assert bool(
            ConsumptionLayer().run(frame).annotations.iloc[0]["action_spec_lossy"]
        ) is False

    def test_exactly_the_escalating_actions_are_flagged(self) -> None:
        flagged = set()
        for action in ACTION_CLASSES:
            frame = make_frame({"action_class": action, "priority_level": "P1"})
            if ConsumptionLayer().run(frame).annotations.iloc[0]["action_spec_lossy"]:
                flagged.add(action)
        assert flagged == {"ESCALATE_IMMEDIATE", "ESCALATE_REVIEW"}

    def test_it_agrees_with_the_existing_warning(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        assert list(annotations["action_spec_lossy"]) == [
            warning is not None
            for warning in annotations["action_interpretation_warning"]
        ]

    def test_the_note_is_in_the_report(self, spread: pd.DataFrame) -> None:
        report = ConsumptionLayer().run(spread).report()
        assert report["action_spec_note"] == ACTION_SPEC_LOSSY_NOTE
        assert "Use action_class for exact intent" in report["action_spec_note"]

    def test_the_count_is_published(self, spread: pd.DataFrame) -> None:
        metrics = ConsumptionLayer().run(spread).transparency_metrics
        # The spread holds three escalating records: one of each kind, plus
        # the unexplained one appended for R1.
        assert metrics["n_action_spec_lossy"] == 3


# ---------------------------------------------------------------------------
# Zero regression
# ---------------------------------------------------------------------------


class TestZeroRegression:
    """Every new field is derived; none may change a decision."""

    def test_the_input_is_never_mutated(self, spread: pd.DataFrame) -> None:
        before = spread.copy(deep=True)
        ConsumptionLayer().run(spread)
        pd.testing.assert_frame_equal(spread, before)

    def test_no_action_is_altered(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        upstream = [json.loads(p)["action"] for p in spread["explanation_payload"]]
        assert [card["action"] for card in result.cards] == upstream

    def test_no_risk_is_altered(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        upstream = [json.loads(p)["risk_score"] for p in spread["explanation_payload"]]
        assert [card["risk"] for card in result.cards] == upstream

    def test_the_new_fields_are_deterministic(self, spread: pd.DataFrame) -> None:
        first = ConsumptionLayer().run(spread).annotations
        second = ConsumptionLayer().run(spread).annotations
        pd.testing.assert_frame_equal(first, second)

    def test_all_four_new_columns_are_produced(self, spread: pd.DataFrame) -> None:
        annotations = ConsumptionLayer().run(spread).annotations
        for column in ("escalation_reason_status", "escalation_reason_warning",
                       "action_group", "action_spec_lossy"):
            assert column in annotations.columns

    def test_the_invariant_catches_a_corrupted_group(
        self, spread: pd.DataFrame
    ) -> None:
        from src.stage7.interface import decode_payloads
        from src.stage7.pipeline import Stage7InvariantError

        layer = ConsumptionLayer()
        result = layer.run(spread)
        result.annotations.loc[result.annotations.index[0], "action_group"] = "monitoring"
        with pytest.raises(Stage7InvariantError, match="I8"):
            layer._assert_guarantees(result, spread, decode_payloads(spread))


# ---------------------------------------------------------------------------
# Integration - the pinned counts from the reference corpus
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegrationCounts:
    """The measured numbers this pass exists to fix."""

    @pytest.fixture(scope="class")
    def corpus(self) -> Any:
        from src.stage1.corpus import Corpus
        from src.stage2.confidence import attach_confidence
        from src.stage3.pipeline import attach_structure
        from src.stage4.pipeline import AnomalyConfig, attach_anomalies
        from src.stage5.pipeline import RiskConfig, attach_risk
        from src.stage6.pipeline import attach_actions

        built = Corpus.from_csv("data/synthetic_dataset.csv")
        attach_confidence(built)
        attach_structure(built)
        attach_anomalies(built, config=AnomalyConfig(compute_calibration=False))
        attach_risk(built, config=RiskConfig(compute_calibration=False))
        attach_actions(built)
        return built

    def test_r1_exactly_eighteen_unexplained_escalations(self, corpus: Any) -> None:
        annotations = ConsumptionLayer().run(corpus).annotations
        unexplained = annotations["escalation_reason_status"] == "unexplained_upstream"
        assert int(unexplained.sum()) == 18

    def test_r2_the_policy_holds_on_the_reference_corpus(self, corpus: Any) -> None:
        report = ConsumptionLayer().run(corpus).report()["escalation_policy"]
        assert report["n_investigate_without_risk"] == 0
        assert report["n_investigate"] == 419

    def test_r3_group_counts(self, corpus: Any) -> None:
        groups = ConsumptionLayer().run(corpus).annotations["action_group"]
        assert int((groups == "escalation").sum()) == 419
        assert int((groups == "data_quality").sum()) == 3402

    def test_r4_exactly_419_lossy_rows(self, corpus: Any) -> None:
        annotations = ConsumptionLayer().run(corpus).annotations
        assert int(annotations["action_spec_lossy"].sum()) == 419

    def test_the_corpus_is_untouched(self, corpus: Any) -> None:
        before = corpus.records.copy(deep=True)
        ConsumptionLayer().run(corpus)
        pd.testing.assert_frame_equal(corpus.records, before)
