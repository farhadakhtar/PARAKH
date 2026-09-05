"""Stage 6.5 - the decision safety layer.

This is the first layer in the system that deliberately changes a decision, so
its tests are weighted differently from every other suite: most of them prove
that a change did **not** happen.

`TestZeroChangeWhenQuiet` is the important one. A safety layer that fires when
nothing is wrong is worse than no safety layer, because it teaches reviewers to
ignore its flags.

Two rules needed a judgement the specification did not settle, and both are
tested in *both* readings so the choice stays visible:

* **S2** - "every P0/P1" (3,402 records) versus its own title, "escalation
  block" (0 records).
* **S3** - the literal override deletes 3 escalations, 2 of them P0. The
  default preserves them while still surfacing the conflict.
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
    ESCALATING_ACTIONS,
    RISK_BANDS,
    SAFETY_DECISIONS,
    SAFETY_RULES,
    STAGE65_VERSION,
)
from src.stage3.artifacts import ArtifactWriteError, save_artifacts
from src.stage5.risk_interpretation import (
    INTERPRETATION_COLUMNS,
    band_for_percentile,
    compute_risk_interpretation,
    describe_risk,
)
from src.stage6.safety_layer import (
    SAFETY_COLUMNS,
    SafetyConfig,
    SafetyConfigError,
    apply_safety_rules,
)
from src.stage6.work_resolution import (
    conflicting_work_ids,
    resolve_works,
    work_conflict_summary,
)

# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------


def make_frame(*rows: Dict[str, Any]) -> pd.DataFrame:
    """A minimal Stage 6 output frame for the safety layer."""
    built: List[Dict[str, Any]] = []
    for index, override in enumerate(rows or ({},)):
        record = {
            "work_id": override.get("work_id", f"w-{index:04d}"),
            "action_class": override.get("action_class", "PASSIVE_MONITOR"),
            "priority_level": override.get("priority_level", "P3"),
            "risk_flag": override.get("risk_flag", "low_risk"),
            "risk_score": override.get("risk_score", 0.1),
            "risk_defined": override.get("risk_defined", True),
            "confidence": override.get("confidence", 0.9),
        }
        built.append(record)
    return pd.DataFrame(built)


def clarity_for(*values: str) -> pd.Series:
    """A clarity series aligned to a frame built by :func:`make_frame`."""
    return pd.Series(list(values), dtype="object")


ESCALATION = {
    "action_class": "ESCALATE_IMMEDIATE",
    "priority_level": "P0",
    "risk_flag": "high_risk",
    "risk_score": 0.62,
}
REVIEW = {
    "action_class": "ESCALATE_REVIEW",
    "priority_level": "P1",
    "risk_flag": "moderate_risk",
    "risk_score": 0.31,
}
DATA_QUALITY = {
    "action_class": "DATA_QUALITY_REVIEW",
    "priority_level": "P1",
    "risk_flag": "insufficient_data",
    "risk_score": None,
    "risk_defined": False,
}


# ---------------------------------------------------------------------------
# S1 - ambiguous escalation block
# ---------------------------------------------------------------------------


class TestS1AmbiguousEscalation:
    """An escalation nobody can characterise is not actionable."""

    def test_a_p0_ambiguous_escalation_is_downgraded(self) -> None:
        frame = make_frame(dict(ESCALATION))
        result = apply_safety_rules(frame, clarity_for("AMBIGUOUS"))
        row = result.frame.iloc[0]
        assert row["final_decision"] == "ESCALATE_REVIEW_REQUIRED"
        assert "S1" in row["safety_flags"]
        assert "no named anomaly category" in row["safety_reason"]

    def test_a_p1_ambiguous_escalation_is_downgraded(self) -> None:
        frame = make_frame(dict(REVIEW))
        result = apply_safety_rules(frame, clarity_for("AMBIGUOUS"))
        assert result.frame.iloc[0]["final_decision"] == "ESCALATE_REVIEW_REQUIRED"

    def test_a_clear_escalation_is_untouched(self) -> None:
        frame = make_frame(dict(ESCALATION))
        result = apply_safety_rules(frame, clarity_for("CLEAR"))
        row = result.frame.iloc[0]
        assert row["final_decision"] == "ESCALATE_IMMEDIATE"
        assert not bool(row["safety_intervened"])

    def test_an_ambiguous_non_escalation_is_untouched(self) -> None:
        """S1 guards escalations; a monitored record is not one."""
        frame = make_frame({})
        result = apply_safety_rules(frame, clarity_for("AMBIGUOUS"))
        assert result.frame.iloc[0]["final_decision"] == "PASSIVE_MONITOR"

    def test_the_original_survives(self) -> None:
        frame = make_frame(dict(ESCALATION))
        result = apply_safety_rules(frame, clarity_for("AMBIGUOUS"))
        assert result.frame.iloc[0]["original_decision"] == "ESCALATE_IMMEDIATE"


# ---------------------------------------------------------------------------
# S2 - data-limited block, in both readings
# ---------------------------------------------------------------------------


class TestS2DataLimited:
    """The specification's condition and its title disagree by 3,402 records."""

    def test_the_default_reading_is_the_title(self) -> None:
        assert SafetyConfig().s2_applies_to == "escalations"

    def test_under_the_default_a_data_quality_record_is_untouched(self) -> None:
        """It is not an escalation, so the 'escalation block' does not apply."""
        frame = make_frame(dict(DATA_QUALITY))
        result = apply_safety_rules(frame, clarity_for("DATA_LIMITED"))
        row = result.frame.iloc[0]
        assert row["final_decision"] == "DATA_QUALITY_REVIEW"
        assert not bool(row["safety_intervened"])

    def test_under_the_literal_reading_it_becomes_remediate(self) -> None:
        frame = make_frame(dict(DATA_QUALITY))
        result = apply_safety_rules(
            frame,
            clarity_for("DATA_LIMITED"),
            config=SafetyConfig(s2_applies_to="all_p0_p1"),
        )
        row = result.frame.iloc[0]
        assert row["final_decision"] == "REMEDIATE"
        assert "S2" in row["safety_flags"]

    def test_a_data_limited_escalation_is_blocked_in_both_readings(self) -> None:
        """Structurally impossible upstream, which is why S2 is a guard."""
        frame = make_frame({**ESCALATION, "risk_defined": False})
        for reading in ("escalations", "all_p0_p1"):
            result = apply_safety_rules(
                frame,
                clarity_for("DATA_LIMITED"),
                config=SafetyConfig(s2_applies_to=reading),
            )
            assert result.frame.iloc[0]["final_decision"] == "REMEDIATE"

    def test_an_invalid_reading_is_rejected(self) -> None:
        with pytest.raises(SafetyConfigError, match="s2_applies_to"):
            SafetyConfig(s2_applies_to="sometimes")

    def test_the_config_states_both_blast_radii(self) -> None:
        note = SafetyConfig().to_dict()["_s2_note"]
        assert "0 records" in note and "3,402" in note


# ---------------------------------------------------------------------------
# S3 - work conflict, and the escalation it would delete
# ---------------------------------------------------------------------------


class TestS3WorkConflict:
    """The rule that, as literally written, deletes a P0 lead."""

    def _conflicting(self) -> pd.DataFrame:
        return make_frame(
            {**ESCALATION, "work_id": "shared"},
            {"work_id": "shared"},  # PASSIVE_MONITOR
        )

    def test_a_conflict_is_flagged_on_every_record_in_the_group(self) -> None:
        result = apply_safety_rules(self._conflicting(), clarity_for("CLEAR", "CLEAR"))
        assert all("S3" in flags for flags in result.frame["safety_flags"])

    def test_the_non_escalating_record_is_overridden(self) -> None:
        result = apply_safety_rules(self._conflicting(), clarity_for("CLEAR", "CLEAR"))
        assert result.frame.iloc[1]["final_decision"] == "INCONSISTENT_WORK"

    def test_by_default_the_escalation_is_preserved(self) -> None:
        """Measured: the literal rule deletes 3 escalations, 2 at P0.

        The records share an id because Stage 1 injects duplicates, so they may
        be different works. Suppressing the lead would lose a clean, high-risk,
        unambiguous escalation because of an unrelated record.
        """
        result = apply_safety_rules(self._conflicting(), clarity_for("CLEAR", "CLEAR"))
        row = result.frame.iloc[0]
        assert row["final_decision"] == "ESCALATE_IMMEDIATE"
        assert "S3" in row["safety_flags"]
        assert "escalation is retained" in row["safety_reason"]

    def test_the_literal_rule_can_be_restored_with_one_flag(self) -> None:
        result = apply_safety_rules(
            self._conflicting(),
            clarity_for("CLEAR", "CLEAR"),
            config=SafetyConfig(s3_preserve_escalations=False),
        )
        assert result.frame.iloc[0]["final_decision"] == "INCONSISTENT_WORK"

    def test_agreeing_records_are_not_flagged(self) -> None:
        frame = make_frame({"work_id": "same"}, {"work_id": "same"})
        result = apply_safety_rules(frame, clarity_for("CLEAR", "CLEAR"))
        assert not result.frame["safety_intervened"].any()
        assert all(not flags for flags in result.frame["safety_flags"])

    def test_the_preserved_count_is_reported(self) -> None:
        result = apply_safety_rules(self._conflicting(), clarity_for("CLEAR", "CLEAR"))
        assert result.to_dict()["n_escalations_preserved_through_conflict"] == 1

    def test_a_unique_work_id_never_conflicts(self) -> None:
        frame = make_frame(dict(ESCALATION), {})
        result = apply_safety_rules(frame, clarity_for("CLEAR", "CLEAR"))
        assert len(result.work_conflicts) == 0


# ---------------------------------------------------------------------------
# S4 - gate misalignment
# ---------------------------------------------------------------------------


class TestS4GateMisalignment:
    """Inert while the gates agree; decisive the moment they do not."""

    def test_it_does_nothing_when_gates_are_aligned(self) -> None:
        frame = make_frame(dict(ESCALATION))
        result = apply_safety_rules(frame, clarity_for("CLEAR"), gates_aligned=True)
        assert result.frame.iloc[0]["final_decision"] == "ESCALATE_IMMEDIATE"

    def test_misalignment_downgrades_high_risk_to_monitor(self) -> None:
        frame = make_frame(dict(ESCALATION))
        result = apply_safety_rules(frame, clarity_for("CLEAR"), gates_aligned=False)
        row = result.frame.iloc[0]
        assert row["final_decision"] == "MONITOR"
        assert "S4" in row["safety_flags"]
        assert "gates are misaligned" in row["safety_reason"]

    def test_it_leaves_non_high_risk_records_alone(self) -> None:
        frame = make_frame(dict(REVIEW))  # moderate_risk
        result = apply_safety_rules(frame, clarity_for("CLEAR"), gates_aligned=False)
        assert result.frame.iloc[0]["final_decision"] == "ESCALATE_REVIEW"

    def test_it_is_recorded_in_the_diagnostics(self) -> None:
        frame = make_frame(dict(ESCALATION))
        result = apply_safety_rules(frame, clarity_for("CLEAR"), gates_aligned=False)
        assert result.to_dict()["gates_aligned"] is False


# ---------------------------------------------------------------------------
# S5 - no silent overrides
# ---------------------------------------------------------------------------


class TestS5NoSilentOverride:
    """Every change is preserved, attributed and explained."""

    def test_every_intervention_names_its_rules(self) -> None:
        frame = make_frame(dict(ESCALATION), dict(DATA_QUALITY))
        result = apply_safety_rules(frame, clarity_for("AMBIGUOUS", "DATA_LIMITED"))
        for _, row in result.interventions().iterrows():
            assert len(row["safety_flags"]) > 0
            assert row["safety_reason"]

    def test_every_intervention_is_logged(self) -> None:
        frame = make_frame(dict(ESCALATION))
        result = apply_safety_rules(frame, clarity_for("AMBIGUOUS"))
        assert len(result.log) == 1
        entry = result.log[0]
        for field in ("record_id", "original_decision", "final_decision",
                      "triggered_rules", "explanation"):
            assert field in entry

    def test_untouched_records_are_not_logged(self) -> None:
        """A log of 20,000 no-ops buries the few that matter."""
        frame = make_frame({}, {}, {})
        result = apply_safety_rules(frame, clarity_for("CLEAR", "CLEAR", "CLEAR"))
        assert result.log == []

    def test_the_log_round_trips_as_jsonl(self, tmp_path: Path) -> None:
        frame = make_frame(dict(ESCALATION))
        result = apply_safety_rules(frame, clarity_for("AMBIGUOUS"))
        path = result.write_log(tmp_path / "safety.jsonl")
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert lines == result.log

    def test_only_declared_decisions_are_emitted(self) -> None:
        frame = make_frame(dict(ESCALATION), dict(DATA_QUALITY), {})
        result = apply_safety_rules(
            frame,
            clarity_for("AMBIGUOUS", "DATA_LIMITED", "CLEAR"),
            config=SafetyConfig(s2_applies_to="all_p0_p1"),
        )
        permitted = set(ACTION_CLASSES) | set(SAFETY_DECISIONS)
        assert set(result.frame["final_decision"]) <= permitted

    def test_all_safety_columns_are_produced(self) -> None:
        result = apply_safety_rules(make_frame(), clarity_for("CLEAR"))
        for column in SAFETY_COLUMNS:
            assert column in result.frame.columns


# ---------------------------------------------------------------------------
# Zero change when nothing is wrong
# ---------------------------------------------------------------------------


class TestZeroChangeWhenQuiet:
    """A layer that fires on healthy records teaches people to ignore it."""

    def test_a_clean_corpus_is_completely_untouched(self) -> None:
        frame = make_frame(
            dict(ESCALATION), dict(REVIEW), dict(DATA_QUALITY), {},
        )
        result = apply_safety_rules(
            frame, clarity_for("CLEAR", "CLEAR", "DATA_LIMITED", "CLEAR")
        )
        assert not result.frame["safety_intervened"].any()
        assert list(result.frame["final_decision"]) == list(
            result.frame["original_decision"]
        )

    def test_the_input_frame_is_never_mutated(self) -> None:
        frame = make_frame(dict(ESCALATION))
        before = frame.copy(deep=True)
        apply_safety_rules(frame, clarity_for("AMBIGUOUS"))
        pd.testing.assert_frame_equal(frame, before)

    def test_it_is_deterministic(self) -> None:
        frame = make_frame(dict(ESCALATION), {"work_id": "w-0000"})
        first = apply_safety_rules(frame, clarity_for("AMBIGUOUS", "CLEAR"))
        second = apply_safety_rules(frame, clarity_for("AMBIGUOUS", "CLEAR"))
        pd.testing.assert_frame_equal(first.frame, second.frame)
        assert first.log == second.log

    def test_a_missing_column_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="work_id"):
            apply_safety_rules(
                make_frame().drop(columns=["work_id"]), clarity_for("CLEAR")
            )


# ---------------------------------------------------------------------------
# Work resolution
# ---------------------------------------------------------------------------


class TestWorkResolution:
    """Aggregate without deciding."""

    def test_conflicts_are_detected(self) -> None:
        frame = make_frame({**ESCALATION, "work_id": "w"}, {"work_id": "w"})
        table = resolve_works(frame)
        assert bool(table.iloc[0]["conflict"]) is True

    def test_agreement_is_not_a_conflict(self) -> None:
        frame = make_frame({"work_id": "w"}, {"work_id": "w"})
        assert bool(resolve_works(frame).iloc[0]["conflict"]) is False

    def test_risk_is_the_maximum_not_the_mean(self) -> None:
        frame = make_frame(
            {"work_id": "w", "risk_score": 0.9}, {"work_id": "w", "risk_score": 0.1}
        )
        row = resolve_works(frame).iloc[0]
        assert row["max_risk_score"] == pytest.approx(0.9)

    def test_final_action_is_the_most_severe_not_the_most_common(self) -> None:
        """One escalation among many monitored records must not be outvoted."""
        frame = make_frame(
            {**ESCALATION, "work_id": "w"},
            {"work_id": "w"},
            {"work_id": "w"},
        )
        assert resolve_works(frame).iloc[0]["final_action"] == "ESCALATE_IMMEDIATE"

    def test_supporting_records_are_carried(self) -> None:
        frame = make_frame({"work_id": "w"}, {"work_id": "w"})
        supporting = resolve_works(frame).iloc[0]["supporting_records"]
        assert len(supporting) == 2
        assert all("record_id" in entry for entry in supporting)

    def test_an_all_unscored_work_has_no_max(self) -> None:
        frame = make_frame(
            {"work_id": "w", "risk_score": None}, {"work_id": "w", "risk_score": None}
        )
        assert resolve_works(frame).iloc[0]["max_risk_score"] is None

    def test_conflicting_ids_are_reported(self) -> None:
        frame = make_frame({**ESCALATION, "work_id": "w"}, {"work_id": "w"})
        assert conflicting_work_ids(resolve_works(frame)) == frozenset({"w"})

    def test_the_summary_states_its_aggregation_rule(self) -> None:
        frame = make_frame({**ESCALATION, "work_id": "w"}, {"work_id": "w"})
        summary = work_conflict_summary(resolve_works(frame))
        assert "MAXIMUM" in summary["_aggregation"]
        assert summary["n_conflicting"] == 1

    def test_an_empty_frame_is_handled(self) -> None:
        assert conflicting_work_ids(pd.DataFrame()) == frozenset()


# ---------------------------------------------------------------------------
# Risk interpretation
# ---------------------------------------------------------------------------


class TestRiskInterpretation:
    """Relative, never probabilistic."""

    def _scored(self, n: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(4)
        return pd.DataFrame(
            {
                "risk_score": rng.uniform(0, 1, n),
                "risk_defined": [True] * n,
            }
        )

    def test_percentiles_span_the_range(self) -> None:
        result = compute_risk_interpretation(self._scored())
        assert result["risk_percentile"].min() >= 0.0
        assert result["risk_percentile"].max() <= 100.0

    def test_the_highest_risk_gets_the_highest_percentile(self) -> None:
        frame = self._scored()
        result = compute_risk_interpretation(frame)
        top = frame["risk_score"].idxmax()
        assert result.loc[top, "risk_percentile"] == result["risk_percentile"].max()

    def test_bands_are_ordered_by_percentile(self) -> None:
        assert band_for_percentile(99.5) == "TOP_1_PERCENT"
        assert band_for_percentile(96.0) == "TOP_5_PERCENT"
        assert band_for_percentile(80.0) == "HIGH"
        assert band_for_percentile(50.0) == "MEDIUM"
        assert band_for_percentile(5.0) == "LOW"

    def test_an_unscored_record_gets_no_band(self) -> None:
        """None, not LOW: unmeasured is not low-risk."""
        frame = pd.DataFrame(
            {"risk_score": [0.5, None], "risk_defined": [True, False]}
        )
        result = compute_risk_interpretation(frame)
        assert result.iloc[1]["risk_band"] is None
        assert pd.isna(result.iloc[1]["risk_percentile"])

    def test_unscored_records_do_not_shift_the_ranking(self) -> None:
        """Ranking a record with no risk against measured ones invents a
        comparison that was never made."""
        scored_only = pd.DataFrame(
            {"risk_score": [0.1, 0.5, 0.9], "risk_defined": [True] * 3}
        )
        with_unscored = pd.DataFrame(
            {
                "risk_score": [0.1, 0.5, 0.9, None, None],
                "risk_defined": [True, True, True, False, False],
            }
        )
        a = compute_risk_interpretation(scored_only)["risk_percentile"].tolist()
        b = compute_risk_interpretation(with_unscored)["risk_percentile"].tolist()[:3]
        assert a == b

    def test_only_declared_bands_are_emitted(self) -> None:
        result = compute_risk_interpretation(self._scored())
        assert set(result["risk_band"].dropna()) <= set(RISK_BANDS)

    def test_the_description_gives_a_rank_not_a_probability(self) -> None:
        text = describe_risk(0.53, 97.9, "TOP_5_PERCENT")
        assert "Top 2.1%" in text
        assert "uncalibrated" in text

    def test_an_absent_score_is_described_as_unmeasured(self) -> None:
        text = describe_risk(None, None, None)
        assert "could not be measured" in text
        assert "not the same as being measured and found safe" in text

    def test_it_changes_no_risk_score(self) -> None:
        frame = self._scored()
        before = frame.copy(deep=True)
        compute_risk_interpretation(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_it_is_deterministic(self) -> None:
        frame = self._scored()
        pd.testing.assert_frame_equal(
            compute_risk_interpretation(frame), compute_risk_interpretation(frame)
        )


# ---------------------------------------------------------------------------
# Stage 3 artefact reproducibility
# ---------------------------------------------------------------------------


def _minimal_artifacts() -> Any:
    """The smallest valid artefact pair, built from the real dataclasses."""
    from src.stage3.artifacts import StrataArtifact, VocabularyArtifact

    return (
        VocabularyArtifact(
            vocabulary={"a": 0},
            idf=(1.0,),
            ngram_range=(1, 1),
            sublinear_tf=True,
            n_source_documents=1,
        ),
        StrataArtifact(edges_log=(0.0, 1.0), n_bins=1, n_reference=1),
    )


class TestArtifactReproducibility:
    """The committed bundle is the reference; a run must not overwrite it."""

    def test_writing_to_the_committed_bundle_is_refused(self) -> None:
        from src.core.constants import ARTIFACT_DIR
        vocabulary, strata = _minimal_artifacts()
        with pytest.raises(ArtifactWriteError, match="committed artefact bundle"):
            save_artifacts(vocabulary, strata, artifact_dir=ARTIFACT_DIR)

    def test_an_explicit_refresh_is_permitted(self, tmp_path: Path) -> None:
        """The guard blocks accidents, not intent."""
        vocabulary, strata = _minimal_artifacts()
        written = save_artifacts(
            vocabulary, strata, artifact_dir=tmp_path, allow_committed_write=False
        )
        assert len(written) == 2

    def test_the_default_destination_is_not_the_committed_bundle(self) -> None:
        from src.core.constants import ARTIFACT_DIR, RUNTIME_ARTIFACT_DIR
        from src.stage3.pipeline import SemanticConfig

        assert SemanticConfig().artifact_dir != ARTIFACT_DIR
        assert SemanticConfig().artifact_dir == RUNTIME_ARTIFACT_DIR

    def test_the_error_says_where_to_write_instead(self) -> None:
        from src.core.constants import ARTIFACT_DIR
        vocabulary, strata = _minimal_artifacts()
        with pytest.raises(ArtifactWriteError) as excinfo:
            save_artifacts(vocabulary, strata, artifact_dir=ARTIFACT_DIR)
        assert "runtime_artifacts" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    """The measured blast radius, on real Stage 1-6 output."""

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

    def test_the_corpus_is_never_mutated(self, corpus: Any) -> None:
        from src.stage7.pipeline import ConsumptionLayer

        before = corpus.records.copy(deep=True)
        ConsumptionLayer().run(corpus)
        pd.testing.assert_frame_equal(corpus.records, before)

    def test_most_records_are_untouched(self, corpus: Any) -> None:
        """The layer must be rare. If it fires broadly, it is not a safety net."""
        from src.stage7.pipeline import ConsumptionLayer

        result = ConsumptionLayer().run(corpus)
        assert result.safety is not None
        rate = result.safety.to_dict()["pct_intervened"]
        assert rate < 5.0, f"safety layer fired on {rate}% of records"

    def test_no_escalation_is_silently_deleted(self, corpus: Any) -> None:
        """The reason S3_PRESERVE_ESCALATIONS defaults True."""
        from src.stage7.pipeline import ConsumptionLayer

        result = ConsumptionLayer().run(corpus)
        safety = result.safety.frame
        was_escalation = safety["original_decision"].isin(ESCALATING_ACTIONS)
        lost = was_escalation & (safety["final_decision"] == "INCONSISTENT_WORK")
        assert not bool(lost.any()), "a work conflict deleted an escalation"

    def test_every_intervention_is_explained(self, corpus: Any) -> None:
        from src.stage7.pipeline import ConsumptionLayer

        result = ConsumptionLayer().run(corpus)
        for _, row in result.safety.interventions().iterrows():
            assert row["safety_reason"]
            assert len(row["safety_flags"]) > 0

    def test_percentiles_are_computed_on_real_data(self, corpus: Any) -> None:
        from src.stage7.pipeline import ConsumptionLayer

        result = ConsumptionLayer().run(corpus)
        scored = result.interpretation["risk_percentile"].notna()
        assert int(scored.sum()) > 0
        assert set(result.interpretation.loc[scored, "risk_band"]) <= set(RISK_BANDS)

    def test_it_is_deterministic_on_real_data(self, corpus: Any) -> None:
        from src.stage7.pipeline import ConsumptionLayer

        first = ConsumptionLayer().run(corpus)
        second = ConsumptionLayer().run(corpus)
        pd.testing.assert_frame_equal(first.safety.frame, second.safety.frame)
