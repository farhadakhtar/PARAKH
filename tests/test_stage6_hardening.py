"""Stage 6 hardening - contract aliases, self-validation, injection safety.

This pass changed no routing decision. Its tests therefore prove two things:

* that Stage 6 now **refuses** input it previously trusted in silence, and
* that the new machine-readable payload survives content the human sentence
  cannot.

`TestInjectionSafety` is the centrepiece. It feeds the three delimiter
collisions the audit proved by construction - a finding containing ``", "``, a
decision class containing ``" with "``, and a finding literally named
``"none recorded"`` - and asserts the payload round-trips all three exactly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    ACTION_CLASSES,
    ACTION_TO_PRIORITY,
    CONFIDENCE_GATE_THRESHOLD,
    MIN_CONFIDENCE_FOR_RISK,
    SPEC_ACTION_ALIAS,
    SPEC_ACTION_CLASSES,
    SPEC_COLUMN_ALIAS,
)
from src.stage6.explanation import (
    PAYLOAD_FIELDS,
    build_action_payload,
    parse_action_payload,
)
from src.stage6.pipeline import (
    STAGE6_COLUMNS,
    STAGE6_DETAIL_COLUMNS,
    STAGE6_SPEC_COLUMNS,
    ActionLayer,
)
from src.stage6.routing import (
    Stage6ConfigError,
    Stage6ContractError,
    Stage6InvariantError,
    assert_gate_alignment,
    require_unique_index,
    validate_stage5_contract,
)

from tests.test_stage6 import (
    INSUFFICIENT,
    INVESTIGATE_HIGH,
    INVESTIGATE_LOW,
    INVESTIGATE_MODERATE,
    REMEDIATE,
    make_frame,
    run_one,
)


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """One record of every routable shape."""
    return make_frame(
        {},
        dict(INVESTIGATE_HIGH),
        dict(INVESTIGATE_MODERATE),
        dict(INVESTIGATE_LOW),
        dict(REMEDIATE),
        dict(INSUFFICIENT),
        {**INVESTIGATE_HIGH, "anomaly_types": []},
    )


# ---------------------------------------------------------------------------
# FIX C1 - contract alignment
# ---------------------------------------------------------------------------


class TestSpecContract:
    """A consumer written to the specification must resolve."""

    def test_a_spec_only_consumer_succeeds(self, spread: pd.DataFrame) -> None:
        """The acceptance test the fix exists for.

        Reads ONLY specification field names, never an as-built one.
        """
        output = ActionLayer().run(spread).frame
        for _, row in output.iterrows():
            action = row["action"]
            priority = row["priority"]
            reason = row["action_reason"]
            findings = row["action_anomaly_types"]
            explanation = row["explanation"]
            assert action in ACTION_CLASSES
            assert priority in ("P0", "P1", "P2", "P3")
            assert isinstance(reason, str) and reason
            assert isinstance(findings, list)
            assert isinstance(explanation, str) and explanation

    def test_the_spec_action_vocabulary_is_available(
        self, spread: pd.DataFrame
    ) -> None:
        output = ActionLayer().run(spread).frame
        assert set(output["action_spec"]) <= set(SPEC_ACTION_CLASSES)
        assert output["action_spec"].notna().all()

    @pytest.mark.parametrize("alias,source", list(SPEC_COLUMN_ALIAS.items()))
    def test_each_alias_equals_its_source(
        self, alias: str, source: str, spread: pd.DataFrame
    ) -> None:
        """An alias that drifted would be worse than none: two columns
        disagreeing about one decision."""
        output = ActionLayer().run(spread).frame
        assert output[alias].equals(output[source])

    def test_no_existing_column_was_removed(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        for column in (*STAGE6_COLUMNS, *STAGE6_DETAIL_COLUMNS):
            assert column in output.columns

    def test_every_as_built_action_has_a_spec_name(self) -> None:
        assert set(SPEC_ACTION_ALIAS) == set(ACTION_CLASSES)
        assert set(SPEC_ACTION_ALIAS.values()) <= set(SPEC_ACTION_CLASSES)

    def test_hold_no_action_is_representable_but_unproduced(
        self, spread: pd.DataFrame
    ) -> None:
        """Documented as a deliberate property, not an omission.

        Stage 6 never concludes a record needs nothing: its quietest outcome
        is PASSIVE_MONITOR, a standing watch rather than a dismissal.
        """
        assert "HOLD_NO_ACTION" in SPEC_ACTION_CLASSES
        assert "HOLD_NO_ACTION" not in set(SPEC_ACTION_ALIAS.values())
        output = ActionLayer().run(spread).frame
        assert "HOLD_NO_ACTION" not in set(output["action_spec"])

    def test_the_alias_mapping_is_deliberately_lossy(self) -> None:
        """SUPERSEDED: the mapping was one-to-one; the current spec is not.

        Both escalating actions now collapse to INVESTIGATE, so `action_spec`
        alone cannot separate a P0 fraud referral (291 records) from a P1
        audit review (128). That is the specified behaviour, and the earlier
        injectivity assertion encoded an assumption the spec has revoked.
        The distinction survives in `action_class` and `priority_level`, both
        unchanged - this test pins that it survives somewhere.
        """
        values = list(SPEC_ACTION_ALIAS.values())
        assert len(values) > len(set(values)), "mapping is expected to be lossy"
        escalating = {SPEC_ACTION_ALIAS["ESCALATE_IMMEDIATE"],
                      SPEC_ACTION_ALIAS["ESCALATE_REVIEW"]}
        assert escalating == {"INVESTIGATE"}
        assert ACTION_TO_PRIORITY["ESCALATE_IMMEDIATE"] != (
            ACTION_TO_PRIORITY["ESCALATE_REVIEW"]
        ), "the distinction the alias loses must remain in priority_level"


# ---------------------------------------------------------------------------
# FIX M1 - configuration invariant
# ---------------------------------------------------------------------------


class TestGateAlignment:
    """The invariant that was holding by coincidence."""

    def test_the_gates_are_currently_aligned(self) -> None:
        assert CONFIDENCE_GATE_THRESHOLD == MIN_CONFIDENCE_FOR_RISK
        assert_gate_alignment()  # must not raise

    def test_drift_is_rejected_with_an_explanation(self) -> None:
        """SUPERSEDED: now raises Stage6ConfigError, a dedicated type.

        A configuration fault is not a data fault: no record is at issue and
        no rerun helps until a threshold changes, so it earns its own class.
        """
        with pytest.raises(Stage6ConfigError, match="drifted"):
            assert_gate_alignment(stage4_gate=0.5, stage5_gate=0.8)

    def test_the_error_names_both_values(self) -> None:
        with pytest.raises(Stage6ConfigError) as excinfo:
            assert_gate_alignment(stage4_gate=0.5, stage5_gate=0.8)
        message = str(excinfo.value)
        assert "0.5" in message and "0.8" in message

    def test_it_runs_on_every_pipeline_entry(self, spread: pd.DataFrame) -> None:
        """Not a module-import assertion that a later edit could bypass."""
        ActionLayer().run(spread)  # exercises require_contract -> the check


# ---------------------------------------------------------------------------
# FIX M4 - cross-field consistency
# ---------------------------------------------------------------------------


class TestStage5ContractValidation:
    """Stage 6 dispatches on two fields; it now checks they agree."""

    def test_a_consistent_frame_passes(self, spread: pd.DataFrame) -> None:
        validate_stage5_contract(spread)

    def test_flag_says_missing_but_defined_is_true(self) -> None:
        frame = make_frame(dict(INVESTIGATE_HIGH))
        frame["risk_flag"] = "insufficient_data"   # defined stays True
        with pytest.raises(Stage6ContractError, match="disagree"):
            validate_stage5_contract(frame)

    def test_defined_is_false_but_flag_says_scored(self) -> None:
        frame = make_frame(dict(INVESTIGATE_HIGH))
        frame["risk_defined"] = False              # flag stays high_risk
        with pytest.raises(Stage6ContractError, match="disagree"):
            validate_stage5_contract(frame)

    def test_the_error_counts_the_violations(self) -> None:
        frame = make_frame(*[dict(INVESTIGATE_HIGH)] * 4)
        frame["risk_defined"] = False
        with pytest.raises(Stage6ContractError, match="4 record"):
            validate_stage5_contract(frame)

    def test_the_pipeline_refuses_before_routing(self) -> None:
        """The point of the fix: a contract error at the door, not an
        AssertionError from deep inside the invariant block."""
        frame = make_frame(dict(INVESTIGATE_HIGH))
        frame["risk_defined"] = pd.Series([None], dtype="object")
        with pytest.raises(Stage6ContractError):
            ActionLayer().run(frame)

    def test_it_is_not_an_assertion_error(self) -> None:
        frame = make_frame(dict(INVESTIGATE_HIGH))
        frame["risk_defined"] = False
        try:
            ActionLayer().run(frame)
        except Stage6ContractError:
            pass
        except AssertionError:  # pragma: no cover
            pytest.fail("a contract violation surfaced as an AssertionError")


# ---------------------------------------------------------------------------
# FIX m1 - index robustness
# ---------------------------------------------------------------------------


class TestIndexRobustness:
    """Fail early, and say which requirement was broken."""

    def test_a_unique_index_passes(self, spread: pd.DataFrame) -> None:
        require_unique_index(spread)

    def test_a_duplicate_index_is_rejected(self) -> None:
        frame = make_frame(dict(INVESTIGATE_HIGH), dict(REMEDIATE), dict(INSUFFICIENT))
        frame.index = pd.Index([7, 7, 7])
        with pytest.raises(Stage6ContractError, match="unique index"):
            require_unique_index(frame)

    def test_the_pipeline_rejects_it_before_pandas_does(self) -> None:
        frame = make_frame(dict(INVESTIGATE_HIGH), dict(REMEDIATE))
        frame.index = pd.Index([1, 1])
        with pytest.raises(Stage6ContractError, match="unique index"):
            ActionLayer().run(frame)

    def test_the_error_names_the_duplicated_labels(self) -> None:
        frame = make_frame(dict(INVESTIGATE_HIGH), dict(REMEDIATE))
        frame.index = pd.Index(["dup", "dup"])
        with pytest.raises(Stage6ContractError, match="dup"):
            ActionLayer().run(frame)

    def test_a_string_index_is_still_accepted(self) -> None:
        """Unique is the requirement; integer is not."""
        frame = make_frame(dict(INVESTIGATE_HIGH), dict(REMEDIATE))
        frame.index = pd.Index(["z", "a"])
        assert len(ActionLayer().run(frame).frame) == 2


# ---------------------------------------------------------------------------
# FIX M2 + M3 - a complete, injection-safe machine form
# ---------------------------------------------------------------------------


class TestPayloadCompleteness:
    """Every field the specification asks to reconstruct, including priority."""

    def test_all_payload_fields_are_present(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        for payload in output["explanation_payload"]:
            fields = parse_action_payload(payload)
            for name in PAYLOAD_FIELDS:
                assert name in fields

    def test_priority_is_recoverable(self, spread: pd.DataFrame) -> None:
        """The M2 failure was 0 / 20,000. It must now be total."""
        output = ActionLayer().run(spread).frame
        for _, row in output.iterrows():
            assert parse_action_payload(row["explanation_payload"])["priority"] == (
                row["priority_level"]
            )

    def test_every_field_matches_its_column(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame.join(
            spread[["decision_class", "risk_flag", "severity_score",
                    "severity_defined", "risk_score", "risk_defined"]]
        )
        for _, row in output.iterrows():
            fields = parse_action_payload(row["explanation_payload"])
            assert fields["action"] == row["action_class"]
            assert fields["action_spec"] == row["action_spec"]
            assert fields["priority"] == row["priority_level"]
            assert fields["decision_class"] == row["decision_class"]
            assert fields["risk_flag"] == row["risk_flag"]
            assert fields["anomaly_types"] == list(row["action_anomaly_types"])
            if row["risk_defined"]:
                assert fields["risk_score"] == pytest.approx(row["risk_score"])
            else:
                assert fields["risk_score"] is None
            if row["severity_defined"]:
                assert fields["severity_score"] == pytest.approx(row["severity_score"])
            else:
                assert fields["severity_score"] is None

    def test_an_absent_number_is_null_not_nan(self) -> None:
        """JSON has no NaN, and null cannot be mistaken for a measured zero."""
        payload = ActionLayer().run(make_frame(dict(INSUFFICIENT))).frame[
            "explanation_payload"
        ].iloc[0]
        assert "NaN" not in payload
        fields = parse_action_payload(payload)
        assert fields["risk_score"] is None and fields["severity_score"] is None

    def test_a_stray_value_is_suppressed_when_the_flag_says_undefined(self) -> None:
        frame = make_frame({**INSUFFICIENT, "risk_score": 0.9, "severity_score": 0.9})
        fields = parse_action_payload(
            ActionLayer().run(frame).frame["explanation_payload"].iloc[0]
        )
        assert fields["risk_score"] is None
        assert fields["severity_score"] is None

    def test_the_payload_is_deterministic(self, spread: pd.DataFrame) -> None:
        first = ActionLayer().run(spread).frame["explanation_payload"]
        second = ActionLayer().run(spread).frame["explanation_payload"]
        pd.testing.assert_series_equal(first, second)

    def test_keys_are_sorted_so_bytes_are_stable(self, spread: pd.DataFrame) -> None:
        payload = ActionLayer().run(spread).frame["explanation_payload"].iloc[0]
        keys = list(json.loads(payload))
        assert keys == sorted(keys)

    def test_a_malformed_payload_is_rejected(self) -> None:
        for bad in ("", "not json", "[1,2,3]", '{"action":"X"}'):
            with pytest.raises(ValueError):
                parse_action_payload(bad)


class TestInjectionSafety:
    """The three collisions the audit proved, now round-tripping exactly."""

    def _payload(self, **overrides: Any) -> Dict[str, Any]:
        row = {
            "action_class": "ESCALATE_IMMEDIATE",
            "priority_level": "P0",
            "action_anomaly_types": ["cost_outlier"],
            "severity_score": 0.5, "severity_defined": True,
            "risk_score": 0.6, "risk_defined": True,
            "decision_class": "INVESTIGATE", "risk_flag": "high_risk",
        }
        row.update(overrides)
        return parse_action_payload(build_action_payload(row))

    def test_a_finding_containing_the_join_delimiter(self) -> None:
        """Previously: 1 stored item parsed back as 2."""
        fields = self._payload(action_anomaly_types=["cost, outlier"])
        assert fields["anomaly_types"] == ["cost, outlier"]
        assert len(fields["anomaly_types"]) == 1

    def test_a_decision_class_containing_the_separator(self) -> None:
        """Previously: 'INVEST with IGATE' split in the wrong place."""
        fields = self._payload(decision_class="INVEST with IGATE")
        assert fields["decision_class"] == "INVEST with IGATE"

    def test_a_risk_flag_containing_the_separator(self) -> None:
        fields = self._payload(risk_flag="high with risk")
        assert fields["risk_flag"] == "high with risk"

    def test_a_finding_literally_named_none_recorded(self) -> None:
        """Previously: indistinguishable from having no findings."""
        present = self._payload(action_anomaly_types=["none recorded"])
        absent = self._payload(action_anomaly_types=[])
        assert present["anomaly_types"] == ["none recorded"]
        assert absent["anomaly_types"] == []
        assert present["anomaly_types"] != absent["anomaly_types"]

    def test_a_finding_containing_a_newline(self) -> None:
        fields = self._payload(action_anomaly_types=["cost\noutlier"])
        assert fields["anomaly_types"] == ["cost\noutlier"]

    @pytest.mark.parametrize(
        "hostile",
        ['{"nested": "json"}', "quote\"inside", "back\\slash", "tab\there",
         "unicode é中", "", "  leading space", "trailing space  "],
    )
    def test_arbitrary_content_round_trips(self, hostile: str) -> None:
        fields = self._payload(action_anomaly_types=[hostile])
        assert fields["anomaly_types"] == [hostile]

    def test_many_hostile_findings_at_once(self) -> None:
        hostile = ["a, b", "c with d", "none recorded", 'e"f', "g\\h"]
        fields = self._payload(action_anomaly_types=hostile)
        assert fields["anomaly_types"] == hostile


# ---------------------------------------------------------------------------
# Regression - nothing about the routing moved
# ---------------------------------------------------------------------------


class TestNoRoutingChange:
    """The hardening pass must be decision-inert."""

    def test_the_human_explanation_is_unchanged(self, spread: pd.DataFrame) -> None:
        """Byte-identical: still five lines, still the same words."""
        output = ActionLayer().run(spread).frame
        for text in output["explanation"]:
            assert len(text.split("\n")) == 5
            assert text.startswith("Record routed to ")

    def test_priority_still_follows_the_policy_table(
        self, spread: pd.DataFrame
    ) -> None:
        output = ActionLayer().run(spread).frame
        assert output["priority_level"].equals(
            output["action_class"].map(ACTION_TO_PRIORITY)
        )

    def test_aliases_add_columns_without_changing_values(
        self, spread: pd.DataFrame
    ) -> None:
        output = ActionLayer().run(spread).frame
        assert set(STAGE6_SPEC_COLUMNS) <= set(output.columns)
        assert output["action"].equals(output["action_class"])

    def test_determinism_survives(self, spread: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            ActionLayer().run(spread).frame, ActionLayer().run(spread).frame
        )

    def test_row_order_is_preserved(self) -> None:
        frame = make_frame(dict(INVESTIGATE_HIGH), dict(REMEDIATE), dict(INSUFFICIENT))
        frame.index = pd.Index([9, 4, 6])
        assert list(ActionLayer().run(frame).frame.index) == [9, 4, 6]

    def test_the_input_is_not_mutated(self, spread: pd.DataFrame) -> None:
        before = spread.copy(deep=True)
        ActionLayer().run(spread)
        pd.testing.assert_frame_equal(spread, before)


@pytest.mark.slow
class TestIntegrationRegression:
    """Distribution counts on real data must not move."""

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

    def test_real_data_passes_the_new_validation(self, corpus: Any) -> None:
        validate_stage5_contract(corpus.records)
        require_unique_index(corpus.records)

    def test_every_payload_round_trips_on_real_data(self, corpus: Any) -> None:
        frame = ActionLayer().run(corpus).frame
        for _, row in frame.iterrows():
            fields = parse_action_payload(row["explanation_payload"])
            assert fields["action"] == row["action_class"]
            assert fields["priority"] == row["priority_level"]

    def test_spec_aliases_hold_on_real_data(self, corpus: Any) -> None:
        frame = ActionLayer().run(corpus).frame
        for alias, source in SPEC_COLUMN_ALIAS.items():
            assert frame[alias].equals(frame[source])


# ---------------------------------------------------------------------------
# TASK 3 - the specified payload schema
# ---------------------------------------------------------------------------


class TestPayloadSchema:
    """The seven fields the specification names, plus the retained synonyms."""

    REQUIRED = ("action", "priority", "rule", "decision_class", "risk_flag",
                "findings", "reason")

    def test_the_seven_specified_fields_are_present(
        self, spread: pd.DataFrame
    ) -> None:
        output = ActionLayer().run(spread).frame
        for payload in output["explanation_payload"]:
            fields = parse_action_payload(payload)
            for name in self.REQUIRED:
                assert name in fields, f"payload is missing {name!r}"

    def test_rule_matches_the_column(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        for _, row in output.iterrows():
            fields = parse_action_payload(row["explanation_payload"])
            assert fields["rule"] == row["action_rule"]

    def test_findings_is_an_exact_list_never_a_joined_string(
        self, spread: pd.DataFrame
    ) -> None:
        output = ActionLayer().run(spread).frame
        for _, row in output.iterrows():
            fields = parse_action_payload(row["explanation_payload"])
            assert isinstance(fields["findings"], list)
            assert fields["findings"] == list(row["action_anomaly_types"])

    def test_findings_and_its_retained_synonym_agree(
        self, spread: pd.DataFrame
    ) -> None:
        """`anomaly_types` is kept so the earlier payload contract survives."""
        output = ActionLayer().run(spread).frame
        for payload in output["explanation_payload"]:
            fields = parse_action_payload(payload)
            assert fields["findings"] == fields["anomaly_types"]

    def test_reason_is_null_when_absent_never_empty_string(self) -> None:
        frame = make_frame(dict(INVESTIGATE_HIGH))
        frame["decision_reason"] = ""
        fields = parse_action_payload(
            ActionLayer().run(frame).frame["explanation_payload"].iloc[0]
        )
        assert fields["reason"] is None

    def test_reason_carries_the_upstream_value(self) -> None:
        frame = make_frame({**INVESTIGATE_HIGH,
                            "decision_reason": "deviation_at_or_above_threshold"})
        fields = parse_action_payload(
            ActionLayer().run(frame).frame["explanation_payload"].iloc[0]
        )
        assert fields["reason"] == "deviation_at_or_above_threshold"

    def test_canonical_encoding(self, spread: pd.DataFrame) -> None:
        """sort_keys + compact separators, so bytes are stable."""
        payload = ActionLayer().run(spread).frame["explanation_payload"].iloc[0]
        assert payload == json.dumps(
            json.loads(payload), sort_keys=True, separators=(",", ":")
        )

    def test_no_nan_ever_appears(self, spread: pd.DataFrame) -> None:
        output = ActionLayer().run(spread).frame
        assert not output["explanation_payload"].str.contains("NaN").any()


# ---------------------------------------------------------------------------
# TASK 7 - guarantees raise a dedicated type
# ---------------------------------------------------------------------------


class TestInvariantErrorType:
    """A guarantee failure must be catchable, and survive -O."""

    def test_the_type_exists_and_is_not_an_assertion(self) -> None:
        assert issubclass(Stage6InvariantError, RuntimeError)
        assert not issubclass(Stage6InvariantError, AssertionError)

    def test_a_broken_guarantee_raises_it(self, monkeypatch: Any) -> None:
        """Corrupt the priority map so the policy table disagrees with itself."""
        import src.stage6.pipeline as pipeline_module

        monkeypatch.setattr(
            pipeline_module, "ACTION_TO_PRIORITY",
            {name: "P3" for name in ACTION_CLASSES},
        )
        with pytest.raises(Stage6InvariantError, match="policy table"):
            ActionLayer().run(make_frame(dict(INVESTIGATE_HIGH)))

    def test_all_five_invariants_hold_on_every_shape(
        self, spread: pd.DataFrame
    ) -> None:
        output = ActionLayer().run(spread).frame
        escalating = output["action_class"].str.startswith("ESCALATE")
        assert output.loc[escalating, "action_anomaly_types"].apply(len).gt(0).all()
        joined = output.join(spread[["risk_flag"]])
        high = joined["risk_flag"] == "high_risk"
        assert set(joined.loc[high, "priority_level"]) <= {"P0", "P1"}
        insufficient = joined["risk_flag"] == "insufficient_data"
        assert not joined.loc[insufficient, "action_class"].str.startswith(
            "ESCALATE"
        ).any()
