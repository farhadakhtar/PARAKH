"""Stage 3 hardening: calibration, duplicate evaluation, reproducibility.

Covers the three audit findings:

* ``TestCalibration``      - every parameter observable, distributions reported
* ``TestDuplicateEval``    - the detector is finally measurable
* ``TestReproducibility``  - frozen feature space, drift measured, reuse gated
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from src.core.constants import (
    ARTIFACT_DIR,
    COST_STRATA_FILE,
    DEVIATION_PERCENTILES,
    EVAL_DUPLICATE_MAX_DAY_GAP,
    FIELD_ORDER,
    MAX_UNSEEN_TOKEN_RATE,
    STAGE3_CONFIG_SNAPSHOT_FILE,
    TFIDF_VOCAB_FILE,
)
from src.stage1.corpus import Corpus
from src.stage1.data_generator import generate_dataset
from src.stage2.confidence import attach_confidence
from src.stage3.artifacts import (
    ArtifactError,
    StrataArtifact,
    VocabularyArtifact,
    load_artifacts,
    measure_strata_drift,
    measure_vocabulary_drift,
    save_artifacts,
    validate_reuse,
)
from src.stage3.calibration import (
    CALIBRATION_PARAMETERS,
    ConfigSnapshot,
    build_calibration_report,
)
from src.stage3.evaluation import (
    DUPLICATE_ID_COLUMN,
    DuplicateTruth,
    evaluate_duplicates,
    inject_duplicate_pairs,
    predicted_pairs,
)
from src.stage3.pipeline import (
    STAGE3_COLUMNS,
    SemanticConfig,
    SemanticLayer,
)

warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    """A corpus with Stage 2 attached."""
    built = Corpus.from_dataframe(generate_dataset(n=10_000, seed=42))
    attach_confidence(built)
    return built


@pytest.fixture(scope="module")
def result(corpus: Corpus, tmp_path_factory: Any) -> Any:
    """Stage 3 run writing artefacts to an isolated directory."""
    directory = tmp_path_factory.mktemp("artifacts_module")
    return SemanticLayer(SemanticConfig(artifact_dir=directory)).run(corpus)


# ---------------------------------------------------------------------------
# Task 1 - calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    """Every parameter observable; nothing tuned."""

    def test_report_lists_every_required_parameter(self, result: Any) -> None:
        report = result.calibration_report()
        for name in (
            "PEER_STAT_MIN_CONFIDENCE",
            "PEER_CELL_MIN_SIZE",
            "PEER_STAT_MIN_REFERENCE",
            "DUPLICATE_SIMILARITY_THRESHOLD",
            "HDBSCAN_MIN_CLUSTER_SIZE",
            "CLUSTER_MIN_RECORDS",
            "SVD_COMPONENTS",
        ):
            assert name in report["parameters"], name

    def test_each_parameter_declares_its_provenance(self, result: Any) -> None:
        """"source: default" is the admission that nobody estimated it."""
        for name, entry in result.calibration_report()["parameters"].items():
            assert entry["source"], name
            assert entry["governs"], name
            assert "risk_if_wrong" in entry, name

    def test_defaults_were_not_tuned(self, result: Any) -> None:
        """The whole point: make values observable WITHOUT changing them."""
        for name, entry in result.calibration_report()["parameters"].items():
            assert entry["is_default"] is True, name
            assert entry["value_used"] == CALIBRATION_PARAMETERS[name]["default"]

    def test_distributions_are_reported(self, result: Any) -> None:
        distributions = result.calibration_report()["distributions"]
        for key in (
            "cluster_size",
            "peer_cell_size",
            "n_stable_cells",
            "stable_cell_pct",
            "stable_record_pct",
            "reference_record_pct",
        ):
            assert key in distributions, key
        assert distributions["cluster_size"]["median"] > 0

    def test_deviation_percentiles_are_reported(self, result: Any) -> None:
        coverage = result.calibration_report()["deviation_coverage"]
        assert coverage
        for name, entry in coverage.items():
            for point in DEVIATION_PERCENTILES:
                assert f"p{point}" in entry["abs_percentiles"], (name, point)

    def test_percentiles_are_labelled_as_descriptive(self, result: Any) -> None:
        """They must not be mistaken for thresholds by Stage 4."""
        note = result.calibration_report()["_note"].lower()
        assert "not thresholds" in note or "descriptive" in note

    def test_norm_coverage_per_feature(self, result: Any) -> None:
        coverage = result.calibration_report()["norm_coverage"]
        for feature in ("log_cost", "spend_ratio", "duration_days"):
            assert feature in coverage
            assert 0.0 <= coverage[feature]["pct"] <= 100.0

    def test_report_is_serialisable_and_deterministic(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        config = SemanticConfig(artifact_dir=tmp_path)
        first = SemanticLayer(config).run(corpus).calibration_report()
        second = SemanticLayer(config).run(corpus).calibration_report()
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_report_carries_no_wall_clock(self, result: Any) -> None:
        blob = json.dumps(result.calibration_report(), default=str).lower()
        for forbidden in ("elapsed", "timestamp", "generated_at"):
            assert forbidden not in blob

    def test_config_snapshot_round_trips(self, tmp_path: Path) -> None:
        config = SemanticConfig(artifact_dir=tmp_path, peer_cell_min_size=21)
        snapshot = ConfigSnapshot.from_config(config)
        snapshot.save(tmp_path)
        reloaded = ConfigSnapshot.load(tmp_path)
        assert reloaded is not None
        assert reloaded.parameters["peer_cell_min_size"] == 21
        assert reloaded.to_dict() == snapshot.to_dict()

    def test_snapshot_absent_returns_none(self, tmp_path: Path) -> None:
        assert ConfigSnapshot.load(tmp_path) is None

    def test_overridden_parameter_is_flagged_as_non_default(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        outcome = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path, peer_cell_min_size=25)
        ).run(corpus)
        entry = outcome.calibration_report()["parameters"]["PEER_CELL_MIN_SIZE"]
        assert entry["value_used"] == 25
        assert entry["is_default"] is False

    def test_report_is_written_to_outputs(self, result: Any, tmp_path: Path) -> None:
        written = result.save_reports(tmp_path)
        assert "calibration_report" in written
        payload = json.loads(written["calibration_report"].read_text(encoding="utf-8"))
        assert payload["parameters"]


# ---------------------------------------------------------------------------
# Task 2 - duplicate evaluation
# ---------------------------------------------------------------------------


class TestDuplicateEval:
    """The detector is finally measurable against matching ground truth."""

    @pytest.fixture(scope="class")
    def evaluated(self, tmp_path_factory: Any) -> Dict[str, Any]:
        """Inject labelled duplicates, run Stage 3, score it."""
        frame = generate_dataset(n=6_000, seed=11)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=150, seed=3)
        built = Corpus.from_dataframe(augmented)
        attach_confidence(built)
        outcome = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path_factory.mktemp("dup"))
        ).run(built)
        report = evaluate_duplicates(
            outcome.frame["duplicate_group_id"],
            truth,
            outcome.frame["duplicate_score"],
        )
        return {"truth": truth, "result": outcome, "report": report}

    def test_duplicate_id_never_enters_the_pipeline(self) -> None:
        """Structural guarantee: the label is a separate object, not a column."""
        frame = generate_dataset(n=500, seed=5)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=20, seed=1)
        assert DUPLICATE_ID_COLUMN not in augmented.columns
        assert tuple(augmented.columns) == FIELD_ORDER
        assert len(truth.duplicate_id) > 0

    def test_injected_duplicates_share_a_district(self) -> None:
        frame = generate_dataset(n=2_000, seed=5)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=50, seed=1)
        for source, injected in zip(truth.source_rows, truth.injected_rows):
            assert (
                augmented.loc[source, "district"]
                == augmented.loc[injected, "district"]
            )

    def test_injected_duplicates_are_temporally_close(self) -> None:
        frame = generate_dataset(n=2_000, seed=5)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=50, seed=1)
        for source, injected in zip(truth.source_rows, truth.injected_rows):
            gap = abs(
                (
                    pd.to_datetime(augmented.loc[injected, "date_proposal"])
                    - pd.to_datetime(augmented.loc[source, "date_proposal"])
                ).days
            )
            assert gap <= EVAL_DUPLICATE_MAX_DAY_GAP

    def test_injected_duplicates_are_near_not_exact(self) -> None:
        """A detector that only catches byte identity is worthless on real data."""
        frame = generate_dataset(n=2_000, seed=5)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=50, seed=1)
        differing = sum(
            augmented.loc[source, "work_name"] != augmented.loc[injected, "work_name"]
            for source, injected in zip(truth.source_rows, truth.injected_rows)
        )
        assert differing > 0

    def test_perfect_duplicates_are_detected(self, evaluated: Dict[str, Any]) -> None:
        assert evaluated["report"]["recall"] > 0.5, evaluated["report"]

    def test_precision_is_high(self, evaluated: Dict[str, Any]) -> None:
        assert evaluated["report"]["precision"] > 0.5, evaluated["report"]

    def test_f1_is_reported(self, evaluated: Dict[str, Any]) -> None:
        report = evaluated["report"]
        expected = (
            2 * report["precision"] * report["recall"]
            / (report["precision"] + report["recall"])
        )
        assert report["f1"] == pytest.approx(expected, abs=1e-6)

    def test_cross_district_duplicates_are_rejected(self, tmp_path: Path) -> None:
        """Stage3.md sec.9.1's 1[d_i=d_j] is a hard gate."""
        frame = generate_dataset(n=2_000, seed=5)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=40, seed=1)
        # Move every injected duplicate to a different district.
        for injected in truth.injected_rows:
            current = augmented.loc[injected, "district"]
            other = next(
                d for d in augmented["district"].dropna().unique() if d != current
            )
            augmented.loc[injected, "district"] = other
        built = Corpus.from_dataframe(augmented)
        attach_confidence(built)
        outcome = SemanticLayer(SemanticConfig(artifact_dir=tmp_path)).run(built)
        report = evaluate_duplicates(outcome.frame["duplicate_group_id"], truth)
        assert report["recall"] < 0.05, report

    def test_temporal_violations_are_rejected(self, tmp_path: Path) -> None:
        """Far apart in time, the decay term kills the pair."""
        frame = generate_dataset(n=2_000, seed=5)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=40, seed=1)
        for injected in truth.injected_rows:
            augmented.loc[injected, "date_proposal"] = "2015-01-01"
        for source in truth.source_rows:
            augmented.loc[source, "date_proposal"] = "2022-12-01"
        built = Corpus.from_dataframe(augmented)
        attach_confidence(built)
        outcome = SemanticLayer(SemanticConfig(artifact_dir=tmp_path)).run(built)
        report = evaluate_duplicates(outcome.frame["duplicate_group_id"], truth)
        assert report["recall"] < 0.05, report

    def test_report_names_its_own_limitation(self, evaluated: Dict[str, Any]) -> None:
        note = evaluated["report"]["_note"]
        assert "Stage 1" in note and "CANNOT" in note

    def test_predicted_pairs_are_symmetric_and_unordered(self) -> None:
        groups = pd.Series([0, 0, 1, 1, -1], index=range(5))
        pairs = predicted_pairs(groups)
        assert pairs == {frozenset((0, 1)), frozenset((2, 3))}

    def test_no_injection_is_a_no_op(self) -> None:
        frame = generate_dataset(n=200, seed=5)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=0, seed=1)
        assert len(augmented) == len(frame)
        assert truth.n_pairs == 0
        assert truth.true_pairs == set()

    def test_negative_pair_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            inject_duplicate_pairs(generate_dataset(n=50, seed=1), n_pairs=-1)

    def test_injection_is_deterministic(self) -> None:
        frame = generate_dataset(n=1_000, seed=5)
        first, truth_a = inject_duplicate_pairs(frame, n_pairs=30, seed=4)
        second, truth_b = inject_duplicate_pairs(frame, n_pairs=30, seed=4)
        pd.testing.assert_frame_equal(first, second)
        assert truth_a.duplicate_id == truth_b.duplicate_id

    def test_eval_report_is_serialisable(
        self, evaluated: Dict[str, Any], tmp_path: Path
    ) -> None:
        path = tmp_path / "stage3_duplicate_eval.json"
        path.write_text(json.dumps(evaluated["report"], indent=2), encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8"))["f1"] >= 0.0


# ---------------------------------------------------------------------------
# Task 3 - reproducibility contract
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Freeze the feature space; measure what freezing does and does not fix."""

    def test_artifacts_are_written_by_default(self, result: Any) -> None:
        directory = Path(result.config.artifact_dir)
        assert (directory / TFIDF_VOCAB_FILE).exists()
        assert (directory / COST_STRATA_FILE).exists()
        assert (directory / STAGE3_CONFIG_SNAPSHOT_FILE).exists()

    def test_reuse_is_opt_in(self, result: Any) -> None:
        """Silently reusing a stale vocabulary is worse than recomputing one."""
        assert result.config.reuse_artifacts is False
        assert result.reproducibility["mode"] == "fit"

    def test_vocabulary_artifact_carries_idf(self, result: Any) -> None:
        """Freezing the vocabulary alone would still let the weights drift."""
        bundle = load_artifacts(result.config.artifact_dir)
        assert bundle.vocabulary is not None
        assert len(bundle.vocabulary.idf) == bundle.vocabulary.n_terms

    def test_strata_artifact_carries_both_scales(self, result: Any) -> None:
        bundle = load_artifacts(result.config.artifact_dir)
        payload = json.loads(
            (Path(result.config.artifact_dir) / COST_STRATA_FILE).read_text("utf-8")
        )
        assert len(payload["edges_log"]) == len(payload["edges_amount"])
        assert payload["edges_amount"] == pytest.approx(
            list(bundle.strata.edges_amount), rel=1e-6
        )

    def test_reuse_reproduces_the_partition_exactly(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        """The contract's actual promise, measured rather than asserted."""
        fit = SemanticLayer(SemanticConfig(artifact_dir=tmp_path)).run(corpus)
        reuse = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path, reuse_artifacts=True)
        ).run(corpus)
        ari = adjusted_rand_score(
            fit.frame["cluster_id"], reuse.frame["cluster_id"]
        )
        assert ari == pytest.approx(1.0, abs=1e-9), ari

    def test_reuse_reproduces_cost_strata_exactly(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        fit = SemanticLayer(SemanticConfig(artifact_dir=tmp_path)).run(corpus)
        reuse = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path, reuse_artifacts=True)
        ).run(corpus)
        pd.testing.assert_series_equal(
            fit.frame["cost_stratum"], reuse.frame["cost_stratum"]
        )

    def test_cluster_label_is_the_stable_key(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        """cluster_id permutes across runs; the label does not.

        HDBSCAN numbers clusters in an order that turns on float ties at the
        1e-16 level, so the integer id is run-local even when the partition is
        bit-identical. Downstream must key on the label.
        """
        fit = SemanticLayer(SemanticConfig(artifact_dir=tmp_path)).run(corpus)
        reuse = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path, reuse_artifacts=True)
        ).run(corpus)
        pd.testing.assert_series_equal(
            fit.frame["cluster_label"], reuse.frame["cluster_label"]
        )
        assert "cluster_label" in STAGE3_COLUMNS
        assert fit.reproducibility["cluster_id_is_run_local"] is True

    def test_drift_is_zero_on_the_same_corpus(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        SemanticLayer(SemanticConfig(artifact_dir=tmp_path)).run(corpus)
        reuse = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path, reuse_artifacts=True)
        ).run(corpus)
        drift = reuse.reproducibility["drift"]
        assert drift["vocabulary"]["unseen_token_rate"] == 0.0
        assert drift["strata"]["total_variation_distance"] < 0.05

    def test_drift_is_measured_on_a_different_corpus(self, tmp_path: Path) -> None:
        first = Corpus.from_dataframe(generate_dataset(n=4_000, seed=1))
        attach_confidence(first)
        SemanticLayer(SemanticConfig(artifact_dir=tmp_path)).run(first)

        second = Corpus.from_dataframe(generate_dataset(n=4_000, seed=99))
        attach_confidence(second)
        reuse = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path, reuse_artifacts=True)
        ).run(second)
        drift = reuse.reproducibility["drift"]["vocabulary"]
        assert 0.0 <= drift["unseen_token_rate"] <= 1.0
        assert "unseen_examples" in drift

    def test_excessive_vocabulary_drift_is_rejected(self, tmp_path: Path) -> None:
        """A corpus the frozen space cannot describe must fail loudly."""
        artifact = VocabularyArtifact(
            vocabulary={"aaa": 0, "bbb": 1},
            idf=(1.0, 1.0),
            ngram_range=(1, 2),
            sublinear_tf=True,
            n_source_documents=2,
        )
        drift = measure_vocabulary_drift(["zzz yyy xxx", "www vvv"], artifact)
        assert drift["unseen_token_rate"] > MAX_UNSEEN_TOKEN_RATE
        assert drift["acceptable"] is False
        with pytest.raises(ArtifactError, match="unseen"):
            validate_reuse(
                pd.DataFrame({"work_name": ["x"]}),
                {"vocabulary": drift},
                required_features=("work_name",),
            )

    def test_missing_required_feature_is_rejected(self) -> None:
        with pytest.raises(ArtifactError, match="required feature"):
            validate_reuse(
                pd.DataFrame({"other": [1]}), {}, required_features=("work_name",)
            )

    def test_excessive_strata_drift_is_rejected(self) -> None:
        artifact = StrataArtifact(
            edges_log=(1.0, 2.0),
            n_bins=3,
            n_reference=100,
            occupancy=(1.0, 0.0, 0.0),
        )
        drift = measure_strata_drift([5.0] * 100, artifact)
        assert drift["acceptable"] is False
        with pytest.raises(ArtifactError, match="cost strata"):
            validate_reuse(
                pd.DataFrame({"work_name": ["x"]}),
                {"strata": drift},
                required_features=("work_name",),
            )

    def test_corrupt_artifact_raises_rather_than_silently_refitting(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / TFIDF_VOCAB_FILE).write_text("{not json", encoding="utf-8")
        with pytest.raises(ArtifactError, match="Cannot read"):
            load_artifacts(tmp_path)

    def test_inconsistent_artifact_is_rejected(self) -> None:
        with pytest.raises(ArtifactError, match="inconsistent"):
            VocabularyArtifact.from_dict(
                {
                    "vocabulary": {"a": 0, "b": 1},
                    "idf": [1.0],
                    "ngram_range": [1, 2],
                }
            )

    def test_missing_artifacts_fall_back_to_fitting(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        """Reuse requested but nothing frozen yet: fit, do not crash."""
        outcome = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path, reuse_artifacts=True)
        ).run(corpus)
        assert outcome.reproducibility["reused_vocabulary"] is False
        assert len(outcome) == len(corpus)

    def test_reproducibility_report_contents(self, result: Any) -> None:
        report = result.reproducibility_report()
        for key in (
            "vocabulary_size",
            "strata_edges_log",
            "strata_edges_amount",
            "cluster_id_is_run_local",
            "stable_cluster_key",
        ):
            assert key in report, key
        json.dumps(report, default=str)

    def test_report_is_written_to_outputs(self, result: Any, tmp_path: Path) -> None:
        written = result.save_reports(tmp_path)
        assert "reproducibility_report" in written
        payload = json.loads(
            written["reproducibility_report"].read_text(encoding="utf-8")
        )
        assert payload["vocabulary_size"] > 0

    def test_save_can_be_disabled(self, corpus: Corpus, tmp_path: Path) -> None:
        SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path, save_artifacts=False)
        ).run(corpus)
        assert not (tmp_path / TFIDF_VOCAB_FILE).exists()


class TestNoRegression:
    """The hardening must not have moved a single score."""

    def test_contract_columns_only_grew(self) -> None:
        for column in (
            "cluster_id",
            "cost_stratum",
            "peer_cell_id",
            "peer_cell_stable",
            "peer_reference",
            "deviation_cell_cost",
            "duplicate_score",
        ):
            assert column in STAGE3_COLUMNS

    def test_scores_are_unchanged_by_instrumentation(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        """Calibration and artefact writing are observers, not participants."""
        instrumented = SemanticLayer(SemanticConfig(artifact_dir=tmp_path)).run(corpus)
        quiet = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path / "off", save_artifacts=False)
        ).run(corpus)
        for column in ("deviation_cell_cost", "deviation_cluster_cost", "duplicate_score"):
            pd.testing.assert_series_equal(
                instrumented.frame[column], quiet.frame[column]
            )

    def test_still_deterministic(self, corpus: Corpus, tmp_path: Path) -> None:
        config = SemanticConfig(artifact_dir=tmp_path)
        first = SemanticLayer(config).run(corpus)
        second = SemanticLayer(config).run(corpus)
        pd.testing.assert_frame_equal(first.frame, second.frame)
