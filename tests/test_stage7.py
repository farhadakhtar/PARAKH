"""Stage 7 - Decision Consumption Layer.

Stage 7 decides nothing, so its tests are about fidelity rather than judgement:
does what a human sees, what a machine consumes, and what the log records all
say the same thing the payload said?

Three tests carry most of the weight:

* `TestReadOnly` proves the corpus is untouched - the constraint that separates
  a consumption layer from another stage.
* `TestPayloadIsSourceOfTruth` proves every output field traces to the payload,
  and that a corrupted human explanation changes nothing but the display.
* `TestDeterminism` proves two runs are byte-identical, which is only possible
  because the clock is injected rather than read.
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
    ACTION_CLASSES,
    ACTION_TO_QUEUE_NAME,
    FEEDBACK_OUTCOMES,
    PRIORITY_EXECUTION,
    PRIORITY_LEVELS,
    QUEUE_NAMES,
    STAGE7_API_VERSION,
    STAGE7_REFERENCE_TIMESTAMP,
    STAGE7_VERSION,
)
from src.stage7.api import API_FIELDS, build_api_response, serialise
from src.stage7.audit import (
    AUDIT_FIELDS,
    compute_input_hash,
    read_audit_log,
    write_audit_log,
)
from src.stage7.feedback import (
    FEEDBACK_FIELDS,
    append_feedback,
    build_feedback_entry,
    read_feedback_log,
    summarise_feedback,
)
from src.stage7.interface import (
    DECISION_CARD_FIELDS,
    REQUIRED_COLUMNS,
    Stage7ContractError,
    build_decision_card,
    decode_payloads,
    require_contract,
)
from src.stage7.pipeline import ConsumptionLayer, Stage7InvariantError, consume

# ---------------------------------------------------------------------------
# Frame construction - a minimal, valid Stage 6 output
# ---------------------------------------------------------------------------


def payload_for(
    action: str = "ESCALATE_IMMEDIATE",
    priority: str = "P0",
    risk: Any = 0.61,
    findings: Any = None,
    decision: str = "INVESTIGATE",
    risk_flag: str = "high_risk",
    reason: Any = "deviation_at_or_above_investigate_threshold",
    rule: str = "investigate_high",
) -> str:
    """Build a canonical Stage 6 payload string."""
    body = {
        "action": action,
        "action_spec": "INVESTIGATE",
        "priority": priority,
        "rule": rule,
        "decision_class": decision,
        "risk_flag": risk_flag,
        "findings": list(findings if findings is not None else ["cost_outlier"]),
        "anomaly_types": list(findings if findings is not None else ["cost_outlier"]),
        "reason": reason,
        "risk_score": risk,
        "severity_score": 0.58,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def make_frame(*rows: Dict[str, Any]) -> pd.DataFrame:
    """Build a Stage 7 input frame."""
    built: List[Dict[str, Any]] = []
    for index, override in enumerate(rows or ({},)):
        action = override.get("action_class", "ESCALATE_IMMEDIATE")
        priority = override.get("priority_level", "P0")
        record = {
            "work_id": override.get("work_id", f"mpl-xx-2020-{index:06d}"),
            "action_class": action,
            "priority_level": priority,
            "action_rule": override.get("action_rule", "investigate_high"),
            "action_spec": "INVESTIGATE",
            "explanation": override.get(
                "explanation", "Record routed to X because:\n- Findings: y"
            ),
            "explanation_payload": override.get(
                "explanation_payload",
                payload_for(
                    action=action,
                    priority=priority,
                    risk=override.get("risk", 0.61),
                    findings=override.get("findings"),
                    decision=override.get("decision", "INVESTIGATE"),
                    risk_flag=override.get("risk_flag", "high_risk"),
                    reason=override.get("reason", "some_rule_fired"),
                ),
            ),
        }
        built.append(record)
    return pd.DataFrame(built)


ONE_PER_ACTION = [
    {"action_class": "ESCALATE_IMMEDIATE", "priority_level": "P0"},
    {"action_class": "ESCALATE_REVIEW", "priority_level": "P1"},
    {"action_class": "DATA_QUALITY_REVIEW", "priority_level": "P1",
     "risk": None, "risk_flag": "insufficient_data",
     "decision": "INSUFFICIENT_CONTEXT", "findings": ["insufficient_context"]},
    {"action_class": "REQUEST_CORRECTION", "priority_level": "P2",
     "risk": None, "risk_flag": "insufficient_data", "decision": "REMEDIATE"},
    {"action_class": "PASSIVE_MONITOR", "priority_level": "P3",
     "risk": 0.05, "risk_flag": "low_risk", "decision": "MONITOR",
     "findings": []},
]


@pytest.fixture(scope="module")
def spread() -> pd.DataFrame:
    """One record per action, covering scored and unscored."""
    return make_frame(*ONE_PER_ACTION)


# ---------------------------------------------------------------------------
# Contract and guardrails
# ---------------------------------------------------------------------------


class TestContractGuardrails:
    """Stage 7 refuses what it cannot consume safely."""

    def test_a_complete_frame_is_accepted(self, spread: pd.DataFrame) -> None:
        require_contract(spread)

    @pytest.mark.parametrize("column", REQUIRED_COLUMNS)
    def test_every_required_column_is_required(self, column: str) -> None:
        with pytest.raises(Stage7ContractError, match=column):
            require_contract(make_frame().drop(columns=[column]))

    def test_a_missing_payload_is_rejected(self) -> None:
        frame = make_frame()
        frame["explanation_payload"] = None
        with pytest.raises(Stage7ContractError, match="no explanation_payload"):
            require_contract(frame)

    def test_malformed_json_is_rejected_by_record(self) -> None:
        frame = make_frame({}, {})
        frame.loc[1, "explanation_payload"] = "{not json"
        with pytest.raises(Stage7ContractError, match="not valid JSON"):
            ConsumptionLayer().run(frame)

    def test_a_json_array_is_rejected(self) -> None:
        frame = make_frame()
        frame["explanation_payload"] = "[1,2,3]"
        with pytest.raises(Stage7ContractError, match="not a JSON object"):
            ConsumptionLayer().run(frame)

    def test_an_unknown_action_is_rejected(self) -> None:
        frame = make_frame()
        frame["explanation_payload"] = payload_for(action="TELEPORT_RECORD")
        with pytest.raises(Stage7ContractError, match="unknown action"):
            ConsumptionLayer().run(frame)

    def test_an_unknown_priority_is_rejected(self) -> None:
        frame = make_frame()
        frame["explanation_payload"] = payload_for(priority="P9")
        with pytest.raises(Stage7ContractError, match="unknown priority"):
            ConsumptionLayer().run(frame)

    def test_a_duplicate_index_is_rejected(self) -> None:
        """The audit log is keyed on the record id; it must address one row."""
        frame = make_frame({}, {})
        frame.index = pd.Index([3, 3])
        with pytest.raises(Stage7ContractError, match="unique index"):
            require_contract(frame)

    def test_a_non_unique_work_id_is_still_accepted(self) -> None:
        """Stage 1 injects duplicate work_ids on purpose; identity is the index."""
        frame = make_frame({"work_id": "same"}, {"work_id": "same"})
        result = ConsumptionLayer().run(frame)
        assert len(result) == 2
        ids = [item.business_id for items in result.queues.values() for item in items]
        assert ids.count("same") == 2

    def test_rejects_a_non_frame_source(self) -> None:
        with pytest.raises(TypeError):
            ConsumptionLayer().run(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Read-only over upstream
# ---------------------------------------------------------------------------


class TestReadOnly:
    """The constraint that makes this a consumption layer."""

    def test_the_input_frame_is_not_mutated(self, spread: pd.DataFrame) -> None:
        before = spread.copy(deep=True)
        ConsumptionLayer().run(spread)
        pd.testing.assert_frame_equal(spread, before)

    def test_no_column_is_added(self, spread: pd.DataFrame) -> None:
        before = list(spread.columns)
        ConsumptionLayer().run(spread)
        assert list(spread.columns) == before

    def test_no_upstream_value_is_recomputed(self, spread: pd.DataFrame) -> None:
        """Every card value must equal the payload it came from, exactly."""
        result = ConsumptionLayer().run(spread)
        for card, raw in zip(result.cards, spread["explanation_payload"]):
            payload = json.loads(raw)
            assert card["risk"] == payload["risk_score"]
            assert card["action"] == payload["action"]
            assert card["findings"] == payload["findings"]


# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------


class TestQueues:
    """Every record reaches exactly one queue, with its execution semantics."""

    def test_every_action_maps_to_a_queue(self) -> None:
        assert set(ACTION_TO_QUEUE_NAME) == set(ACTION_CLASSES)
        assert set(ACTION_TO_QUEUE_NAME.values()) == set(QUEUE_NAMES)

    def test_each_record_lands_in_its_action_queue(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        for name, items in result.queues.items():
            for item in items:
                assert ACTION_TO_QUEUE_NAME[item.action] == name

    def test_all_queues_are_present_even_when_empty(self) -> None:
        """An absent key would read as 'not computed', not 'nothing waiting'."""
        result = ConsumptionLayer().run(make_frame())
        assert set(result.queues) == set(QUEUE_NAMES)
        assert sum(len(items) for items in result.queues.values()) == 1

    def test_every_record_is_queued_exactly_once(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        ids = [item.record_id for items in result.queues.values() for item in items]
        assert sorted(ids) == sorted(spread.index)
        assert len(ids) == len(set(ids))

    def test_queue_items_carry_the_required_fields(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        for items in result.queues.values():
            for item in items:
                entry = item.to_dict()
                for field in ("record_id", "priority", "reason", "findings",
                              "timestamp"):
                    assert field in entry

    def test_items_are_ordered_most_urgent_first(self) -> None:
        frame = make_frame(
            {"action_class": "ESCALATE_REVIEW", "priority_level": "P1"},
            {"action_class": "DATA_QUALITY_REVIEW", "priority_level": "P1"},
        )
        result = ConsumptionLayer().run(frame)
        for items in result.queues.values():
            levels = [PRIORITY_LEVELS.index(item.priority) for item in items]
            assert levels == sorted(levels)

    def test_execution_semantics_match_the_priority(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        for items in result.queues.values():
            for item in items:
                expected = PRIORITY_EXECUTION[item.priority]
                assert item.execution_mode == expected["mode"]
                assert item.sla_hours == expected["sla_hours"]

    def test_p3_carries_no_sla(self) -> None:
        """Passive monitoring has no deadline; inventing one is false urgency."""
        assert PRIORITY_EXECUTION["P3"]["sla_hours"] is None

    def test_p0_is_the_tightest_sla(self) -> None:
        timed = {
            level: PRIORITY_EXECUTION[level]["sla_hours"]
            for level in PRIORITY_LEVELS
            if PRIORITY_EXECUTION[level]["sla_hours"] is not None
        }
        assert timed["P0"] == min(timed.values())

    def test_an_unknown_queue_is_rejected(self, spread: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="unknown queue"):
            ConsumptionLayer().run(spread).queue("legal_queue")


# ---------------------------------------------------------------------------
# The payload is the only source of truth
# ---------------------------------------------------------------------------


class TestPayloadIsSourceOfTruth:
    """The human explanation is display only, and never parsed."""

    def test_corrupting_the_explanation_changes_only_the_display(self) -> None:
        """The decisive test. If any decision field moved, something parsed it."""
        clean = make_frame()
        corrupt = clean.copy()
        corrupt.loc[0, "explanation"] = "TOTAL GARBAGE, none recorded with nonsense"

        card_a = ConsumptionLayer().run(clean).cards[0]
        card_b = ConsumptionLayer().run(corrupt).cards[0]
        assert card_a["explanation"] != card_b["explanation"]
        for field in ("action", "priority", "risk", "decision", "findings",
                      "reason", "confidence_context"):
            assert card_a[field] == card_b[field], f"{field} was derived from text"

    def test_the_explanation_is_carried_byte_for_byte(self) -> None:
        frame = make_frame({"explanation": "line one\nline two\nwith, commas"})
        card = ConsumptionLayer().run(frame).cards[0]
        assert card["explanation"] == "line one\nline two\nwith, commas"

    def test_findings_survive_delimiter_injection(self) -> None:
        """What the Stage 6 human sentence could not do, the payload does.

        SUPERSEDED IN PART by FIX 7: the consumption pipeline now rejects any
        anomaly category outside the closed vocabulary, so hostile strings can
        no longer be fed through it AS findings. The payload encoding itself is
        still injection-safe, which is what this test exists to prove, so it is
        exercised at the encoder rather than through the pipeline.
        """
        from src.stage6.explanation import (
            build_action_payload,
            parse_action_payload,
        )

        hostile = ["cost, outlier", "none recorded", "a with b", 'quote"here']
        payload = build_action_payload(
            {
                "action_class": "ESCALATE_IMMEDIATE",
                "priority_level": "P0",
                "action_anomaly_types": hostile,
                "action_rule": "investigate_high",
                "severity_score": 0.5,
                "severity_defined": True,
                "risk_score": 0.6,
                "risk_defined": True,
                "decision_class": "INVESTIGATE",
                "risk_flag": "high_risk",
            }
        )
        assert parse_action_payload(payload)["findings"] == hostile

    def test_a_category_outside_the_vocabulary_is_now_refused(self) -> None:
        """FIX 7. Stage 5 would score an unknown category as zero breadth and
        Stage 7 would have no phrase for it - both silently."""
        frame = make_frame()
        frame["explanation_payload"] = payload_for(findings=["brand_new_category"])
        with pytest.raises(ValueError, match="closed vocabulary"):
            ConsumptionLayer().run(frame)

    def test_card_carries_every_declared_field(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        for card in result.cards:
            for field in DECISION_CARD_FIELDS:
                assert field in card

    def test_a_null_risk_stays_null_never_zero(self) -> None:
        frame = make_frame({"risk": None, "risk_flag": "insufficient_data"})
        card = ConsumptionLayer().run(frame).cards[0]
        assert card["risk"] is None
        assert card["risk"] != 0.0

    def test_confidence_context_matches_the_risk_state(self) -> None:
        scored = ConsumptionLayer().run(make_frame()).cards[0]
        unscored = ConsumptionLayer().run(
            make_frame({"risk": None, "risk_flag": "insufficient_data"})
        ).cards[0]
        assert "measurable" in scored["confidence_context"]
        assert "unassessed, not cleared" in unscored["confidence_context"]


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


class TestApiContract:
    """Stable, serialisable, backward compatible."""

    def test_every_field_is_present(self, spread: pd.DataFrame) -> None:
        for response in ConsumptionLayer().run(spread).api_responses:
            for field in API_FIELDS:
                assert field in response

    def test_it_is_json_serialisable(self, spread: pd.DataFrame) -> None:
        for response in ConsumptionLayer().run(spread).api_responses:
            json.loads(serialise(response))

    def test_no_nan_ever_reaches_a_consumer(self, spread: pd.DataFrame) -> None:
        for response in ConsumptionLayer().run(spread).api_responses:
            assert "NaN" not in serialise(response)

    def test_metadata_names_its_source_and_version(self, spread: pd.DataFrame) -> None:
        response = ConsumptionLayer().run(spread).api_responses[0]
        assert response["metadata"]["source_stage"] == "stage6"
        assert response["metadata"]["version"] == STAGE7_API_VERSION

    def test_serialisation_is_byte_stable(self, spread: pd.DataFrame) -> None:
        first = [serialise(r) for r in ConsumptionLayer().run(spread).api_responses]
        second = [serialise(r) for r in ConsumptionLayer().run(spread).api_responses]
        assert first == second

    def test_risk_status_is_reported_beside_a_null_score(self) -> None:
        frame = make_frame({"risk": None, "risk_flag": "insufficient_data"})
        response = ConsumptionLayer().run(frame).api_responses[0]
        assert response["risk_score"] is None
        assert response["risk_status"] == "insufficient_data"

    def test_the_response_matches_the_payload(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        for response, raw in zip(result.api_responses, spread["explanation_payload"]):
            payload = json.loads(raw)
            assert response["action"] == payload["action"]
            assert response["priority"] == payload["priority"]
            assert response["risk_score"] == payload["risk_score"]
            assert response["findings"] == payload["findings"]


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    """Complete, deterministic, immutable."""

    def test_one_entry_per_record(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        assert len(result.audit_entries) == len(spread)

    def test_every_entry_is_complete(self, spread: pd.DataFrame) -> None:
        for entry in ConsumptionLayer().run(spread).audit_entries:
            for field in AUDIT_FIELDS:
                assert field in entry

    def test_the_hash_is_deterministic(self) -> None:
        payload = payload_for()
        assert compute_input_hash(7, payload) == compute_input_hash(7, payload)

    def test_the_hash_covers_the_payload(self) -> None:
        assert compute_input_hash(7, payload_for(risk=0.5)) != compute_input_hash(
            7, payload_for(risk=0.9)
        )

    def test_the_hash_covers_the_record_id(self) -> None:
        payload = payload_for()
        assert compute_input_hash(1, payload) != compute_input_hash(2, payload)

    def test_the_hash_excludes_the_timestamp(self) -> None:
        """Otherwise every replay would look like a change."""
        a = ConsumptionLayer().run(make_frame(), issued_at="2024-01-01T00:00:00+00:00")
        b = ConsumptionLayer().run(make_frame(), issued_at="2030-06-15T12:00:00+00:00")
        assert a.audit_entries[0]["input_hash"] == b.audit_entries[0]["input_hash"]
        assert a.audit_entries[0]["timestamp"] != b.audit_entries[0]["timestamp"]

    def test_the_payload_is_stored_verbatim(self, spread: pd.DataFrame) -> None:
        result = ConsumptionLayer().run(spread)
        for entry, raw in zip(result.audit_entries, spread["explanation_payload"]):
            assert entry["explanation_payload"] == raw

    def test_it_round_trips_through_jsonl(
        self, spread: pd.DataFrame, tmp_path: Path
    ) -> None:
        result = ConsumptionLayer().run(spread)
        path = write_audit_log(result.audit_entries, tmp_path / "audit.jsonl")
        assert read_audit_log(path) == result.audit_entries

    def test_the_written_file_is_byte_stable(
        self, spread: pd.DataFrame, tmp_path: Path
    ) -> None:
        first = write_audit_log(
            ConsumptionLayer().run(spread).audit_entries, tmp_path / "a.jsonl"
        ).read_bytes()
        second = write_audit_log(
            ConsumptionLayer().run(spread).audit_entries, tmp_path / "b.jsonl"
        ).read_bytes()
        assert first == second


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class TestFeedback:
    """Captured, stored separately, and fed back into nothing."""

    def test_an_entry_carries_every_field(self) -> None:
        entry = build_feedback_entry(1, "opened a case", True, "looked real",
                                     outcome="confirmed")
        for field in FEEDBACK_FIELDS:
            assert field in entry

    def test_an_unknown_outcome_is_rejected(self) -> None:
        with pytest.raises(Stage7ContractError, match="unknown feedback outcome"):
            build_feedback_entry(1, "x", True, outcome="maybe")

    def test_a_non_boolean_verdict_is_rejected(self) -> None:
        """A truthy value would hide which verdict the reviewer meant."""
        with pytest.raises(Stage7ContractError, match="must be a bool"):
            build_feedback_entry(1, "x", "yes")  # type: ignore[arg-type]

    def test_it_round_trips_through_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "feedback.jsonl"
        entries = [
            build_feedback_entry(1, "opened", True, outcome="confirmed"),
            build_feedback_entry(2, "closed", False, outcome="rejected"),
        ]
        for entry in entries:
            append_feedback(entry, path)
        assert read_feedback_log(path) == entries

    def test_appending_never_rewrites(self, tmp_path: Path) -> None:
        """A reviewer changing their mind is a second entry, not an edit."""
        path = tmp_path / "feedback.jsonl"
        append_feedback(build_feedback_entry(1, "opened", True), path)
        append_feedback(build_feedback_entry(1, "reopened", False), path)
        assert len(read_feedback_log(path)) == 2

    def test_a_missing_log_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_feedback_log(tmp_path / "absent.jsonl") == []

    def test_the_summary_counts_but_never_scores(self) -> None:
        summary = summarise_feedback([
            build_feedback_entry(1, "a", True, outcome="confirmed"),
            build_feedback_entry(2, "b", False, outcome="rejected"),
        ])
        assert summary["n_entries"] == 2
        assert summary["by_outcome"]["confirmed"] == 1
        assert "accuracy" not in summary
        assert "circular" in summary["_note"]

    def test_feedback_never_touches_the_corpus(
        self, spread: pd.DataFrame, tmp_path: Path
    ) -> None:
        before = spread.copy(deep=True)
        append_feedback(build_feedback_entry(0, "opened", True),
                        tmp_path / "f.jsonl")
        pd.testing.assert_frame_equal(spread, before)


# ---------------------------------------------------------------------------
# Determinism and invariants
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same input, same output - only possible because the clock is injected."""

    def test_two_runs_produce_identical_cards(self, spread: pd.DataFrame) -> None:
        assert ConsumptionLayer().run(spread).cards == ConsumptionLayer().run(
            spread
        ).cards

    def test_two_runs_produce_identical_api_responses(
        self, spread: pd.DataFrame
    ) -> None:
        assert (
            ConsumptionLayer().run(spread).api_responses
            == ConsumptionLayer().run(spread).api_responses
        )

    def test_two_runs_produce_identical_audit_entries(
        self, spread: pd.DataFrame
    ) -> None:
        assert (
            ConsumptionLayer().run(spread).audit_entries
            == ConsumptionLayer().run(spread).audit_entries
        )

    def test_no_wall_clock_is_read(self, spread: pd.DataFrame) -> None:
        """The timestamp is the injected default, not 'now'."""
        result = ConsumptionLayer().run(spread)
        assert result.issued_at == STAGE7_REFERENCE_TIMESTAMP
        for entry in result.audit_entries:
            assert entry["timestamp"] == STAGE7_REFERENCE_TIMESTAMP

    def test_an_injected_timestamp_is_used_everywhere(
        self, spread: pd.DataFrame
    ) -> None:
        stamp = "2031-12-25T09:30:00+00:00"
        result = ConsumptionLayer().run(spread, issued_at=stamp)
        assert all(e["timestamp"] == stamp for e in result.audit_entries)
        assert all(
            r["metadata"]["timestamp"] == stamp for r in result.api_responses
        )
        assert all(
            item.timestamp == stamp
            for items in result.queues.values()
            for item in items
        )

    def test_the_report_serialises_identically(self, spread: pd.DataFrame) -> None:
        first = json.dumps(ConsumptionLayer().run(spread).report(), sort_keys=True)
        second = json.dumps(ConsumptionLayer().run(spread).report(), sort_keys=True)
        assert first == second

    def test_an_empty_frame_produces_an_empty_result(self) -> None:
        result = ConsumptionLayer().run(make_frame().iloc[0:0])
        assert len(result) == 0
        assert set(result.queues) == set(QUEUE_NAMES)

    def test_artefacts_are_written_and_reloadable(
        self, spread: pd.DataFrame, tmp_path: Path
    ) -> None:
        written = ConsumptionLayer().run(spread).save(tmp_path)
        loaded = json.loads(written["queue_report"].read_text(encoding="utf-8"))
        assert loaded["n_records"] == len(spread)
        assert len(read_audit_log(written["audit_log"])) == len(spread)


class TestInvariants:
    """The eight guarantees, enforced on every run."""

    def test_a_misaligned_output_is_caught(self, spread: pd.DataFrame) -> None:
        """Corrupt a card and prove the guarantee fires."""
        layer = ConsumptionLayer()
        result = layer.run(spread)
        result.cards[0]["action"] = "PASSIVE_MONITOR"
        payloads = decode_payloads(spread)
        with pytest.raises(Stage7InvariantError, match="I6"):
            layer._assert_guarantees(result, spread, payloads)

    def test_an_altered_explanation_is_caught(self, spread: pd.DataFrame) -> None:
        layer = ConsumptionLayer()
        result = layer.run(spread)
        result.cards[0]["explanation"] = "edited"
        payloads = decode_payloads(spread)
        with pytest.raises(Stage7InvariantError, match="I4"):
            layer._assert_guarantees(result, spread, payloads)

    def test_the_error_type_is_not_an_assertion(self) -> None:
        assert issubclass(Stage7InvariantError, RuntimeError)
        assert not issubclass(Stage7InvariantError, AssertionError)

    def test_the_report_states_that_stage7_decides_nothing(
        self, spread: pd.DataFrame
    ) -> None:
        note = ConsumptionLayer().run(spread).report()["_note"]
        assert "decides nothing" in note
        assert "display only" in note


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    """Stage 7 on real Stage 1-6 output."""

    @pytest.fixture(scope="class")
    def corpus(self) -> Any:
        from src.stage1.corpus import Corpus
        from src.stage1.data_generator import generate_dataset
        from src.stage2.confidence import attach_confidence
        from src.stage3.pipeline import attach_structure
        from src.stage4.pipeline import AnomalyConfig, attach_anomalies
        from src.stage5.pipeline import RiskConfig, attach_risk
        from src.stage6.pipeline import attach_actions

        built = Corpus.from_dataframe(generate_dataset(n=2000, seed=42))
        attach_confidence(built)
        attach_structure(built)
        attach_anomalies(built, config=AnomalyConfig(compute_calibration=False))
        attach_risk(built, config=RiskConfig(compute_calibration=False))
        attach_actions(built)
        return built

    def test_it_consumes_real_upstream_output(self, corpus: Any) -> None:
        result = consume(corpus)
        assert len(result) == len(corpus)

    def test_the_corpus_is_untouched(self, corpus: Any) -> None:
        before = corpus.records.copy(deep=True)
        consume(corpus)
        pd.testing.assert_frame_equal(corpus.records, before)

    def test_every_record_is_queued_once(self, corpus: Any) -> None:
        result = consume(corpus)
        total = sum(len(items) for items in result.queues.values())
        assert total == len(corpus)

    def test_unscored_records_carry_a_null_risk(self, corpus: Any) -> None:
        result = consume(corpus)
        unscored = [c for c in result.cards if c["risk"] is None]
        assert unscored, "real data always contains unscorable records"
        for card in unscored:
            assert "unassessed, not cleared" in card["confidence_context"]

    def test_it_is_deterministic_on_real_data(self, corpus: Any) -> None:
        assert consume(corpus).audit_entries == consume(corpus).audit_entries

    def test_every_payload_traces_to_stage6(self, corpus: Any) -> None:
        result = consume(corpus)
        for entry, raw in zip(
            result.audit_entries, corpus.records["explanation_payload"]
        ):
            assert entry["explanation_payload"] == raw
