"""Stage 6 - Action & Routing Layer.

Stage 6 is policy, so its tests are mostly about what it must *never* do:
never downgrade an escalation, never deprioritise a high-risk record, never
route a record it cannot explain, never print a number the record does not
carry.

Two tests carry most of the weight:

* `TestExplanationRoundTrip` parses every generated explanation back into
  fields and compares each one against the stored column. A narrative that
  stops matching its record fails the build.
* `TestPolicyTotality` proves the policy table has no hole and no unreachable
  rule, by construction rather than by trying inputs and hoping.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    ACTION_CLASSES,
    ACTION_TO_PRIORITY,
    ACTION_TO_QUEUE,
    DECISION_CLASSES,
    ESCALATING_ACTIONS,
    M1_CORRECTION_LABEL,
    PRIORITY_LEVELS,
    RISK_FLAGS,
    STAGE6_ACTION_REPORT,
    STAGE6_VERSION,
)
from src.stage6.explanation import (
    FIELD_ORDER,
    NOT_DEFINED,
    explain_action,
    parse_action_explanation,
)
from src.stage6.pipeline import (
    STAGE6_COLUMNS,
    STAGE6_DETAIL_COLUMNS,
    ActionLayer,
    ActionResult,
    attach_actions,
)
from src.stage6.routing import (
    POLICY,
    REQUIRED_COLUMNS,
    Stage6InputError,
    apply_m1_correction,
    require_contract,
    route,
)

# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

#: An ordinary monitored record.
BASELINE: Dict[str, Any] = {
    "decision_class": "MONITOR",
    "severity_score": 0.12,
    "severity_defined": True,
    "anomaly_types": [],
    "decision_reason": "no_escalating_deviation",
    "risk_score": 0.08,
    "risk_flag": "low_risk",
    "risk_defined": True,
    "risk_defined_reason": "ok",
}

#: The four upstream shapes, as they actually occur.
INVESTIGATE_HIGH = {
    "decision_class": "INVESTIGATE", "severity_score": 0.72, "severity_defined": True,
    "anomaly_types": ["cost_outlier"], "risk_score": 0.61, "risk_flag": "high_risk",
    "risk_defined": True, "risk_defined_reason": "ok",
}
INVESTIGATE_MODERATE = {**INVESTIGATE_HIGH, "risk_score": 0.31,
                        "risk_flag": "moderate_risk"}
INVESTIGATE_LOW = {**INVESTIGATE_HIGH, "risk_score": 0.11, "risk_flag": "low_risk"}
REMEDIATE = {
    "decision_class": "REMEDIATE", "severity_score": 0.4, "severity_defined": True,
    "anomaly_types": ["low_confidence"], "risk_score": np.nan,
    "risk_flag": "insufficient_data", "risk_defined": False,
    "risk_defined_reason": "confidence_below_gate",
}
INSUFFICIENT = {
    "decision_class": "INSUFFICIENT_CONTEXT", "severity_score": np.nan,
    "severity_defined": False, "anomaly_types": ["insufficient_context"],
    "risk_score": np.nan, "risk_flag": "insufficient_data", "risk_defined": False,
    "risk_defined_reason": "severity_undefined",
}


def make_frame(*overrides: Dict[str, Any]) -> pd.DataFrame:
    """Build a Stage 6 input frame from per-record overrides of BASELINE."""
    rows = []
    for override in overrides or ({},):
        row = copy.deepcopy(BASELINE)
        row.update(override)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in ("severity_defined", "risk_defined"):
        frame[column] = frame[column].astype(bool)
    return frame


def run_one(**overrides: Any) -> pd.Series:
    """Route one record and return its output row."""
    return ActionLayer().run(make_frame(dict(overrides))).frame.iloc[0]


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """One record of every routable shape, plus the M1 cases."""
    return make_frame(
        {},                                                        # 0 monitor
        dict(INVESTIGATE_HIGH),                                    # 1 P0
        dict(INVESTIGATE_MODERATE),                                # 2 P1
        dict(INVESTIGATE_LOW),                                     # 3 gap case
        dict(REMEDIATE),                                           # 4 collision
        dict(INSUFFICIENT),                                        # 5 unscored
        {**INVESTIGATE_HIGH, "anomaly_types": []},                 # 6 M1 + P0
        {**INVESTIGATE_MODERATE, "anomaly_types": []},             # 7 M1 + P1
        {**INVESTIGATE_LOW, "anomaly_types": []},                  # 8 M1 + gap
        {"decision_class": "MONITOR", "risk_flag": "moderate_risk",
         "risk_score": 0.25},                                      # 9 monitor/mod
    )


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


class TestInputContract:
    """Stage 6 refuses to route what it cannot read."""

    def test_accepts_a_complete_frame(self) -> None:
        require_contract(make_frame())

    @pytest.mark.parametrize("column", REQUIRED_COLUMNS)
    def test_every_required_column_is_required(self, column: str) -> None:
        with pytest.raises(Stage6InputError, match=column):
            require_contract(make_frame().drop(columns=[column]))

    def test_anomaly_reason_is_not_required(self) -> None:
        """The brief names it; Stage 4 emits `decision_reason` instead."""
        assert "anomaly_reason" not in REQUIRED_COLUMNS
        ActionLayer().run(make_frame())  # must not raise

    def test_the_layer_refuses_an_incomplete_frame(self) -> None:
        with pytest.raises(Stage6InputError):
            ActionLayer().run(make_frame().drop(columns=["risk_flag"]))

    def test_rejects_a_non_frame_source(self) -> None:
        with pytest.raises(TypeError):
            ActionLayer().run(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RULE 3 / M1
# ---------------------------------------------------------------------------


class TestM1Correction:
    """No record is escalated without a named finding."""

    def test_investigate_with_no_findings_is_labelled(self) -> None:
        row = run_one(**{**INVESTIGATE_HIGH, "anomaly_types": []})
        assert row["action_anomaly_types"] == [M1_CORRECTION_LABEL]
        assert bool(row["anomaly_types_corrected"])

    @pytest.mark.parametrize(
        "shape", [INVESTIGATE_HIGH, INVESTIGATE_MODERATE, INVESTIGATE_LOW]
    )
    def test_it_applies_at_every_risk_band(self, shape: Dict[str, Any]) -> None:
        row = run_one(**{**shape, "anomaly_types": []})
        assert M1_CORRECTION_LABEL in row["action_anomaly_types"]

    def test_an_existing_finding_is_never_overwritten(self) -> None:
        row = run_one(**INVESTIGATE_HIGH)
        assert row["action_anomaly_types"] == ["cost_outlier"]
        assert not bool(row["anomaly_types_corrected"])

    def test_the_label_is_never_added_to_a_non_investigate_record(self) -> None:
        """MONITOR with no findings is correct and must stay empty."""
        row = run_one()
        assert row["action_anomaly_types"] == []
        assert not bool(row["anomaly_types_corrected"])

    def test_it_does_not_touch_the_stage_four_column(self) -> None:
        """Stage 4 is locked and has byte-identical guarantees.

        The correction is written to `action_anomaly_types`. Rewriting
        `anomaly_types` in place would mean a re-run of Stage 5 reading
        different inputs than the first run did.
        """
        frame = make_frame({**INVESTIGATE_HIGH, "anomaly_types": []})
        before = frame["anomaly_types"].copy()
        ActionLayer().run(frame)
        pd.testing.assert_series_equal(frame["anomaly_types"], before)

    def test_the_correction_is_counted_in_the_report(self, spread: pd.DataFrame) -> None:
        report = ActionLayer().run(spread).report()
        assert report["routing"]["m1_correction"]["n_corrected"] == 3

    def test_no_escalated_record_ever_has_an_empty_finding_list(
        self, spread: pd.DataFrame
    ) -> None:
        output = ActionLayer().run(spread).frame
        escalated = output["action_class"].isin(ESCALATING_ACTIONS)
        assert bool(
            output.loc[escalated, "action_anomaly_types"].apply(len).gt(0).all()
        )


# ---------------------------------------------------------------------------
# The mapping table
# ---------------------------------------------------------------------------


class TestActionMapping:
    """Every case in the policy, verified against the specification."""

    def test_case_1_investigate_high(self) -> None:
        row = run_one(**INVESTIGATE_HIGH)
        assert row["action_class"] == "ESCALATE_IMMEDIATE"
        assert row["priority_level"] == "P0"
        assert row["reviewer_queue"] == "fraud_investigation_team"

    def test_case_2_investigate_moderate(self) -> None:
        row = run_one(**INVESTIGATE_MODERATE)
        assert row["action_class"] == "ESCALATE_REVIEW"
        assert row["priority_level"] == "P1"
        assert row["reviewer_queue"] == "audit_team"

    def test_case_3_remediate(self) -> None:
        row = run_one(**REMEDIATE)
        assert row["action_class"] == "REQUEST_CORRECTION"
        assert row["priority_level"] == "P2"
        assert row["reviewer_queue"] == "field_officer"

    def test_case_4_monitor(self) -> None:
        row = run_one()
        assert row["action_class"] == "PASSIVE_MONITOR"
        assert row["priority_level"] == "P3"
        assert row["reviewer_queue"] == "automated_monitoring"

    def test_case_5_insufficient_context(self) -> None:
        row = run_one(**INSUFFICIENT)
        assert row["action_class"] == "DATA_QUALITY_REVIEW"
        assert row["priority_level"] == "P1"
        assert row["reviewer_queue"] == "data_quality_team"

    def test_edge_case_3_investigate_low_escalates(self) -> None:
        """A gap in CASES 1-5, filled from the edge-case section."""
        row = run_one(**INVESTIGATE_LOW)
        assert row["action_class"] == "ESCALATE_REVIEW"
        assert row["action_rule"] == "investigate_low"

    def test_the_remediate_collision_resolves_to_case_3(self) -> None:
        """REMEDIATE also satisfies CASE 5's risk_defined == False.

        CASE 3 names the decision class; CASE 5 is a fallback. Resolving the
        other way would move every REMEDIATE record into P1.
        """
        row = run_one(**REMEDIATE)
        assert not bool(row["anomaly_types_corrected"])
        assert row["action_class"] == "REQUEST_CORRECTION"
        assert row["action_rule"] == "remediate"

    @pytest.mark.parametrize("action", ACTION_CLASSES)
    def test_priority_and_queue_follow_the_tables(self, action: str) -> None:
        assert ACTION_TO_PRIORITY[action] in PRIORITY_LEVELS
        assert ACTION_TO_QUEUE[action]

    def test_every_action_class_is_reachable(self, spread: pd.DataFrame) -> None:
        emitted = set(ActionLayer().run(spread).frame["action_class"])
        assert emitted == set(ACTION_CLASSES)


# ---------------------------------------------------------------------------
# Policy totality
# ---------------------------------------------------------------------------


class TestPolicyTotality:
    """The table has no hole and no dead rule - proved, not sampled."""

    def test_every_upstream_combination_is_routed(self) -> None:
        """All decision x risk_flag x defined combinations, including
        impossible ones. A router must not fall through on input it will
        never see - that is precisely when a hole goes unnoticed."""
        rows = []
        for decision, flag in itertools.product(DECISION_CLASSES, RISK_FLAGS):
            defined = flag != "insufficient_data"
            rows.append(
                {
                    "decision_class": decision,
                    "risk_flag": flag,
                    "risk_defined": defined,
                    "risk_score": 0.3 if defined else np.nan,
                    "severity_defined": True,
                    "severity_score": 0.4,
                    "anomaly_types": ["cost_outlier"],
                }
            )
        output = ActionLayer().run(make_frame(*rows)).frame
        assert len(output) == len(DECISION_CLASSES) * len(RISK_FLAGS)
        assert output["action_class"].isin(ACTION_CLASSES).all()
        assert (output["action_rule"].str.len() > 0).all()

    def test_a_frame_with_an_unknown_decision_class_raises(self) -> None:
        """Better a loud failure than a silent default."""
        frame = make_frame({"decision_class": "SOMETHING_NEW", "risk_defined": True})
        with pytest.raises(RuntimeError, match="policy table has a hole"):
            ActionLayer().run(frame)

    def test_every_rule_is_reachable(self) -> None:
        """A rule that can never fire is dead policy and should be deleted."""
        rows = [
            dict(INVESTIGATE_HIGH),
            dict(INVESTIGATE_MODERATE),
            dict(INVESTIGATE_LOW),
            {**INVESTIGATE_HIGH, "risk_defined": False, "risk_score": np.nan,
             "risk_flag": "insufficient_data"},
            dict(REMEDIATE),
            dict(INSUFFICIENT),
            {"decision_class": "MONITOR", "risk_defined": False,
             "risk_score": np.nan, "risk_flag": "insufficient_data"},
            # Cross-stage disagreement: Stage 4 monitors, Stage 5 bands high.
            {"decision_class": "MONITOR", "risk_flag": "high_risk",
             "risk_score": 0.55, "risk_defined": True, "anomaly_types": []},
            {},
        ]
        fired = set(ActionLayer().run(make_frame(*rows)).frame["action_rule"])
        assert fired == {rule.name for rule in POLICY}

    def test_monitor_is_the_last_rule(self) -> None:
        """So it can never capture a record an earlier rule should escalate."""
        assert POLICY[-1].name == "monitor"

    def test_the_escalating_rules_come_first(self) -> None:
        names = [rule.name for rule in POLICY]
        escalating = [
            index for index, rule in enumerate(POLICY)
            if rule.action in ESCALATING_ACTIONS
        ]
        assert escalating == list(range(len(escalating))), (
            f"escalating rules must have highest precedence; order is {names}"
        )


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    """The six, plus the edge cases."""

    def test_1_every_record_has_an_action(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        assert output["action_class"].notna().all()
        assert output["action_class"].isin(ACTION_CLASSES).all()

    def test_2_every_record_has_a_queue(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        assert output["reviewer_queue"].notna().all()
        assert (output["reviewer_queue"].str.len() > 0).all()

    def test_3_investigate_never_becomes_passive_monitor(self) -> None:
        for flag, defined in (
            ("high_risk", True), ("moderate_risk", True), ("low_risk", True),
            ("insufficient_data", False),
        ):
            row = run_one(
                decision_class="INVESTIGATE", risk_flag=flag, risk_defined=defined,
                risk_score=0.3 if defined else np.nan,
                anomaly_types=["cost_outlier"],
            )
            assert row["action_class"] != "PASSIVE_MONITOR"
            assert row["action_class"] in ESCALATING_ACTIONS

    def test_4_high_risk_never_gets_p3(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame.join(spread[["risk_flag"]])
        high = output["risk_flag"] == "high_risk"
        assert not bool((high & (output["priority_level"] == "P3")).any())

    def test_5_explanation_prints_no_absent_value(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame.join(
            spread[["risk_defined", "severity_defined"]]
        )
        for _, row in output.iterrows():
            fields = parse_action_explanation(row["explanation"])
            if not row["risk_defined"]:
                assert fields["Risk"] == NOT_DEFINED
            if not row["severity_defined"]:
                assert fields["Severity"] == NOT_DEFINED

    def test_6_escalations_always_name_a_finding(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        escalated = output["action_class"].isin(ESCALATING_ACTIONS)
        for text in output.loc[escalated, "explanation"]:
            assert parse_action_explanation(text)["Findings"] != "none recorded"

    def test_edge_1_nan_risk_never_escalates_to_p0(self) -> None:
        row = run_one(**REMEDIATE)
        assert row["priority_level"] != "P0"

    def test_edge_2_no_severity_cannot_escalate_immediately(self) -> None:
        row = run_one(**INSUFFICIENT)
        assert row["action_class"] == "DATA_QUALITY_REVIEW"

    def test_priority_agrees_with_the_action(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        assert output["priority_level"].equals(
            output["action_class"].map(ACTION_TO_PRIORITY)
        )

    def test_queue_agrees_with_the_action(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        assert output["reviewer_queue"].equals(
            output["action_class"].map(ACTION_TO_QUEUE)
        )


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


class TestExplanationRoundTrip:
    """Parse it back and compare against the stored columns, field by field."""

    def test_the_format_is_exactly_five_lines(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        for text in output["explanation"]:
            assert len(text.split("\n")) == 1 + len(FIELD_ORDER)

    def test_every_field_matches_its_column(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame.join(
            spread[["severity_score", "severity_defined", "risk_score",
                    "risk_defined", "decision_class", "risk_flag"]]
        )
        for _, row in output.iterrows():
            fields = parse_action_explanation(row["explanation"])
            assert fields["action_class"] == row["action_class"]
            expected = (
                ", ".join(row["action_anomaly_types"])
                if len(row["action_anomaly_types"])
                else "none recorded"
            )
            assert fields["Findings"] == expected
            if row["severity_defined"]:
                assert fields["Severity"] == f"{row['severity_score']:.3f}"
            else:
                assert fields["Severity"] == NOT_DEFINED
            if row["risk_defined"]:
                assert fields["Risk"] == f"{row['risk_score']:.3f}"
            else:
                assert fields["Risk"] == NOT_DEFINED
            assert fields["Decision basis"] == (
                f"{row['decision_class']} with {row['risk_flag']}"
            )

    def test_it_names_no_signal_outside_the_input_contract(
        self, spread: pd.DataFrame
    ) -> None:
        """Invariant 5. Stage 3 and raw fields must not appear."""
        output = ActionLayer().run(spread).frame
        # `low_confidence` is a legitimate Stage 4 anomaly type, so the bare
        # word "confidence" is expected. What must not appear is a Stage 2
        # confidence VALUE or any Stage 3 / raw field.
        # Substring matching must target the upstream COLUMN names, not words
        # that legitimately occur inside a label: `low_confidence` is a Stage 4
        # anomaly type and `unexplained_deviation` is the M1 label.
        forbidden = ("cluster", "peer_cell", "duplicate_score", "confidence:",
                     "district", "sanctioned", "z_cost", "deviation_")
        for text in output["explanation"]:
            lowered = text.lower()
            for word in forbidden:
                assert word not in lowered, f"{word!r} leaked into: {text}"

    def test_it_never_contradicts_the_upstream_decision(
        self, spread: pd.DataFrame
    ) -> None:
        output = ActionLayer().run(spread).frame.join(spread[["decision_class"]])
        for _, row in output.iterrows():
            fields = parse_action_explanation(row["explanation"])
            assert fields["Decision basis"].startswith(row["decision_class"])

    def test_a_malformed_explanation_is_rejected_by_the_parser(self) -> None:
        for bad in ("", "Record routed to X because:", "nonsense\n- a\n- b\n- c\n- d"):
            with pytest.raises(ValueError):
                parse_action_explanation(bad)

    def test_the_parser_is_the_exact_inverse(self) -> None:
        row = {
            "action_class": "ESCALATE_IMMEDIATE",
            "action_anomaly_types": ["cost_outlier", "temporal_outlier"],
            "severity_score": 0.5, "severity_defined": True,
            "risk_score": 0.25, "risk_defined": True,
            "decision_class": "INVESTIGATE", "risk_flag": "high_risk",
        }
        fields = parse_action_explanation(explain_action(row))
        assert fields["action_class"] == "ESCALATE_IMMEDIATE"
        assert fields["Findings"] == "cost_outlier, temporal_outlier"
        assert fields["Severity"] == "0.500"
        assert fields["Risk"] == "0.250"

    def test_a_nan_is_never_printed_as_a_number(self) -> None:
        text = explain_action({
            "action_class": "DATA_QUALITY_REVIEW", "action_anomaly_types": [],
            "severity_score": np.nan, "severity_defined": False,
            "risk_score": np.nan, "risk_defined": False,
            "decision_class": "INSUFFICIENT_CONTEXT",
            "risk_flag": "insufficient_data",
        })
        assert "nan" not in text.lower()
        assert text.count(NOT_DEFINED) == 2

    def test_a_stray_value_is_suppressed_when_the_flag_says_undefined(self) -> None:
        """The definedness flag is authoritative, not the number."""
        text = explain_action({
            "action_class": "DATA_QUALITY_REVIEW", "action_anomaly_types": [],
            "severity_score": 0.9, "severity_defined": False,
            "risk_score": 0.9, "risk_defined": False,
            "decision_class": "INSUFFICIENT_CONTEXT",
            "risk_flag": "insufficient_data",
        })
        assert "0.900" not in text


# ---------------------------------------------------------------------------
# Determinism and the output contract
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same input, same output, byte for byte."""

    def test_repeated_runs_agree_exactly(self, spread: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            ActionLayer().run(spread).frame, ActionLayer().run(spread).frame
        )

    def test_the_report_serialises_identically(self, spread: pd.DataFrame) -> None:
        first = json.dumps(ActionLayer().run(spread).report(), sort_keys=True)
        second = json.dumps(ActionLayer().run(spread).report(), sort_keys=True)
        assert first == second

    def test_the_report_carries_no_wall_clock(self, spread: pd.DataFrame) -> None:
        blob = json.dumps(ActionLayer().run(spread).report())
        assert "elapsed" not in blob and "timestamp" not in blob

    def test_row_order_and_index_are_preserved(self) -> None:
        frame = make_frame({}, dict(INVESTIGATE_HIGH), dict(REMEDIATE))
        frame.index = pd.Index([44, 2, 91], name="record")
        assert list(ActionLayer().run(frame).frame.index) == [44, 2, 91]

    def test_routing_does_not_depend_on_row_order(self) -> None:
        """Policy is per-record; a shuffle must not change a single verdict."""
        frame = make_frame(*[dict(x) for x in
                             (INVESTIGATE_HIGH, REMEDIATE, INSUFFICIENT,
                              INVESTIGATE_LOW, BASELINE)])
        straight = ActionLayer().run(frame).frame["action_class"]
        shuffled = frame.iloc[::-1]
        reversed_result = ActionLayer().run(shuffled).frame["action_class"]
        pd.testing.assert_series_equal(
            straight, reversed_result.reindex(straight.index)
        )

    def test_the_input_is_not_mutated(self, spread: pd.DataFrame) -> None:
        before = spread.copy(deep=True)
        ActionLayer().run(spread)
        pd.testing.assert_frame_equal(spread, before)

    def test_the_contract_columns_are_produced(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        for column in (*STAGE6_COLUMNS, *STAGE6_DETAIL_COLUMNS):
            assert column in output.columns

    def test_an_empty_frame_produces_an_empty_result(self) -> None:
        result = ActionLayer().run(make_frame().iloc[0:0])
        assert len(result) == 0
        assert result.report()["n_records"] == 0

    def test_the_report_is_written_and_reloadable(
        self, spread: pd.DataFrame, tmp_path: Path
    ) -> None:
        written = ActionLayer().run(spread).save_reports(tmp_path)
        path = written["action_report"]
        assert path.name == STAGE6_ACTION_REPORT
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["stage6_version"] == STAGE6_VERSION
        assert loaded["n_records"] == len(spread)

    def test_the_policy_is_published_in_the_report(self, spread: pd.DataFrame) -> None:
        """An operator must be able to read the routing rules without the code."""
        policy = ActionLayer().run(spread).report()["policy"]
        assert len(policy) == len(POLICY)
        assert {entry["rule"] for entry in policy} == {rule.name for rule in POLICY}


# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------


class TestQueues:
    """What each team actually receives."""

    def test_a_queue_contains_only_its_own_work(self, spread: pd.DataFrame) -> None:
        result = ActionLayer().run(spread)
        for name in set(ACTION_TO_QUEUE.values()):
            assert set(result.queue(name)["reviewer_queue"]) <= {name}

    def test_a_queue_is_ordered_most_urgent_first(self) -> None:
        frame = make_frame(dict(INVESTIGATE_MODERATE), dict(INSUFFICIENT),
                           dict(INVESTIGATE_LOW))
        result = ActionLayer().run(frame)
        levels = list(result.queue("audit_team")["priority_level"])
        assert levels == sorted(levels)

    def test_an_unknown_queue_is_rejected(self, spread: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="unknown queue"):
            ActionLayer().run(spread).queue("legal_team")

    def test_by_priority_filters(self, spread: pd.DataFrame) -> None:
        result = ActionLayer().run(spread)
        for level in PRIORITY_LEVELS:
            assert set(result.by_priority(level)["priority_level"]) <= {level}

    def test_an_unknown_priority_is_rejected(self, spread: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="unknown priority"):
            ActionLayer().run(spread).by_priority("P9")

    def test_the_fraud_queue_receives_only_p0(self, spread: pd.DataFrame) -> None:
        queue = ActionLayer().run(spread).queue("fraud_investigation_team")
        assert set(queue["priority_level"]) <= {"P0"}

    def test_the_queues_partition_the_corpus(self, spread: pd.DataFrame) -> None:
        result = ActionLayer().run(spread)
        total = sum(
            len(result.queue(name)) for name in set(ACTION_TO_QUEUE.values())
        )
        assert total == len(spread)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    """Stage 6 on real Stage 1-5 output."""

    @pytest.fixture(scope="class")
    def corpus(self) -> Any:
        from src.stage1.corpus import Corpus
        from src.stage1.data_generator import generate_dataset
        from src.stage2.confidence import attach_confidence
        from src.stage3.pipeline import attach_structure
        from src.stage4.pipeline import AnomalyConfig, attach_anomalies
        from src.stage5.pipeline import RiskConfig, attach_risk

        built = Corpus.from_dataframe(generate_dataset(n=2000, seed=42))
        attach_confidence(built)
        attach_structure(built)
        attach_anomalies(built, config=AnomalyConfig(compute_calibration=False))
        attach_risk(built, config=RiskConfig(compute_calibration=False))
        return built

    def test_it_consumes_real_upstream_output(self, corpus: Any) -> None:
        result = attach_actions(corpus)
        assert len(result) == len(corpus)
        for column in STAGE6_COLUMNS:
            assert column in corpus.records.columns

    def test_upstream_columns_survive_untouched(self, corpus: Any) -> None:
        before = corpus.records[["decision_class", "risk_score", "anomaly_types"]].copy()
        attach_actions(corpus)
        pd.testing.assert_frame_equal(
            corpus.records[["decision_class", "risk_score", "anomaly_types"]], before
        )

    def test_no_investigate_is_downgraded_on_real_data(self, corpus: Any) -> None:
        frame = attach_actions(corpus).frame.join(corpus.records[["decision_class"]])
        investigate = frame["decision_class"] == "INVESTIGATE"
        assert frame.loc[investigate, "action_class"].isin(ESCALATING_ACTIONS).all()

    def test_every_explanation_round_trips_on_real_data(self, corpus: Any) -> None:
        frame = attach_actions(corpus).frame
        for text in frame["explanation"]:
            parse_action_explanation(text)

    def test_m1_fires_on_real_data(self, corpus: Any) -> None:
        """The gap this stage exists to close is real, not hypothetical."""
        result = attach_actions(corpus)
        assert result.routing.correction.to_dict()["n_corrected"] > 0

    def test_it_is_deterministic_on_real_data(self, corpus: Any) -> None:
        pd.testing.assert_frame_equal(
            ActionLayer().run(corpus).frame, ActionLayer().run(corpus).frame
        )

    def test_a_misaligned_result_is_rejected(self, corpus: Any) -> None:
        result = ActionLayer().run(corpus)
        truncated = ActionResult(frame=result.frame.iloc[:-1], routing=result.routing)
        with pytest.raises(ValueError, match="rows"):
            attach_actions(corpus, result=truncated)
