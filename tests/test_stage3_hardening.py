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
    DEVIATION_BUCKETS,
    DEVIATION_REASON_CLUSTER_NOISE,
    NOISE_CLUSTER_ID,
    PEER_STAT_MIN_REFERENCE,
    Z_EXTREME_THRESHOLD,
    Z_HIGH_THRESHOLD,
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
from src.stage3.embedding import build_stopwords, embed_work_names
from src.stage3.evaluation import (
    DUPLICATE_ID_COLUMN,
    DuplicateTruth,
    assert_perturbations_are_real,
    evaluate_duplicates,
    inject_duplicate_pairs,
    perturb_work_name,
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
        from sklearn.preprocessing import normalize as _l2

        from src.core.constants import DUPLICATE_SIMILARITY_THRESHOLD

        frame = generate_dataset(n=6_000, seed=11)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=150, seed=3)
        built = Corpus.from_dataframe(augmented)
        attach_confidence(built)
        outcome = SemanticLayer(
            SemanticConfig(artifact_dir=tmp_path_factory.mktemp("dup"))
        ).run(built)
        # The detector's own view: untruncated, digits kept.
        embedding = embed_work_names(
            built.records, n_components=0, truncate_locality=False, keep_digits=True
        )
        vectors = _l2(embedding.record_tfidf().tocsr())
        similarity = [
            float((vectors[a] @ vectors[b].T).toarray()[0, 0])
            for a, b in zip(truth.source_rows, truth.injected_rows)
        ]
        report = evaluate_duplicates(
            outcome.frame["duplicate_group_id"],
            truth,
            outcome.frame["duplicate_score"],
            pair_similarity=similarity,
            threshold=DUPLICATE_SIMILARITY_THRESHOLD,
        )
        return {
            "truth": truth,
            "result": outcome,
            "report": report,
            "frame": augmented,
            "similarity": similarity,
        }

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
        """Every pair must differ AFTER preprocessing, not merely before it."""
        frame = generate_dataset(n=2_000, seed=5)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=50, seed=1)
        assert all(
            augmented.loc[source, "work_name"] != augmented.loc[injected, "work_name"]
            for source, injected in zip(truth.source_rows, truth.injected_rows)
        )
        check = assert_perturbations_are_real(
            augmented, truth, build_stopwords(augmented)
        )
        assert check["identical_after_preprocessing"] == 0

    @pytest.mark.parametrize("kind", ["typo", "swap", "truncate"])
    def test_each_perturbation_kind_changes_a_surviving_token(
        self, kind: str
    ) -> None:
        stopwords = build_stopwords(
            pd.DataFrame({"district": ["Mysuru"], "state": ["Karnataka"]})
        )
        name = "Construction of Overhead Water Tank at Ward No. 7, Mysuru"
        perturbed = perturb_work_name(name, kind, stopwords, position_seed=0)
        assert perturbed != name, kind

    def test_amount_jitter_is_applied(self) -> None:
        frame = generate_dataset(n=2_000, seed=5)
        augmented, truth = inject_duplicate_pairs(frame, n_pairs=50, seed=1)
        changed = 0
        for source, injected in zip(truth.source_rows, truth.injected_rows):
            a = augmented.loc[source, "sanction_amount"]
            b = augmented.loc[injected, "sanction_amount"]
            try:
                if float(a) != float(b):
                    changed += 1
            except (TypeError, ValueError):
                continue
        assert changed > 0

    def test_perturbation_kinds_are_recorded(self) -> None:
        frame = generate_dataset(n=2_000, seed=5)
        _, truth = inject_duplicate_pairs(frame, n_pairs=50, seed=1)
        assert len(truth.perturbations) == len(truth.source_rows)
        assert set(truth.perturbations) <= {"typo", "swap", "truncate"}

    def test_perturbations_survive_preprocessing(
        self, evaluated: Dict[str, Any]
    ) -> None:
        """AUDIT M2: the property whose absence invalidated the old F1.

        The first harness perturbed only the action verb, a stopword, so 60/60
        pairs were byte-identical in the detector's own text view and the
        reported 0.929 measured exact matching.
        """
        frame = evaluated["frame"]
        check = assert_perturbations_are_real(
            frame, evaluated["truth"], build_stopwords(frame)
        )
        assert check["identical_after_preprocessing"] == 0, check
        assert check["trivial"] is False

    def test_injected_pairs_are_not_identical_vectors(
        self, evaluated: Dict[str, Any]
    ) -> None:
        """Cosine must be strictly below 1.0, or the task is exact matching."""
        similarity = evaluated["similarity"]
        assert len(similarity) > 0
        assert max(similarity) < 1.0 - 1e-9, max(similarity)

    def test_evaluation_is_not_trivial_matching(
        self, evaluated: Dict[str, Any]
    ) -> None:
        """A meaningful spread of difficulty, not a pile of near-copies."""
        similarity = evaluated["similarity"]
        assert min(similarity) < 0.9
        assert float(np.median(similarity)) < 0.95

    def test_detector_still_finds_a_true_exact_duplicate(
        self, tmp_path: Path
    ) -> None:
        """Control: the low recall is a representation limit, not a dead detector.

        Given a genuine same-district, same-week, same-text pair, the detector
        must still group it. Without this control, M2's low recall could not be
        distinguished from a broken implementation.
        """
        rows = [
            {
                "work_name": "Construction of CC Road at Ward No. 5, Mysuru",
                "district": "Mysuru",
                "date_proposal": "2019-03-01",
            },
            {
                "work_name": "Construction of CC Road at Ward No. 5, Mysuru",
                "district": "Mysuru",
                "date_proposal": "2019-03-06",
            },
        ]
        frame = generate_dataset(n=3_000, seed=8)
        extra = pd.DataFrame(
            [
                dict(frame.iloc[0].to_dict(), **row, work_id=f"CTRL-{i}")
                for i, row in enumerate(rows)
            ],
            columns=list(FIELD_ORDER),
        ).astype("object")
        augmented = pd.concat([frame, extra], ignore_index=True)
        built = Corpus.from_dataframe(augmented)
        attach_confidence(built)
        outcome = SemanticLayer(SemanticConfig(artifact_dir=tmp_path)).run(built)
        groups = outcome.frame["duplicate_group_id"].iloc[-2:]
        assert (groups >= 0).all()
        assert groups.iloc[0] == groups.iloc[1]

    def test_metrics_are_reported_with_a_diagnosis(
        self, evaluated: Dict[str, Any]
    ) -> None:
        """Recall alone cannot separate a broken detector from a hard task."""
        report = evaluated["report"]
        assert "recall_by_perturbation" in report
        assert set(report["recall_by_perturbation"]) == {"typo", "swap", "truncate"}
        assert "pair_similarity" in report
        assert "pct_reachable" in report["pair_similarity"]

    def test_previous_result_is_explicitly_withdrawn(
        self, evaluated: Dict[str, Any]
    ) -> None:
        note = evaluated["report"]["_withdrawn"]
        assert "0.929" in note and "WITHDRAWN" in note

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
        assert "Stage 1" in note and "cannot validate" in note

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


# ---------------------------------------------------------------------------
# Audit remediation: M1, M3, M4
# ---------------------------------------------------------------------------


class TestNoiseNeverDefinesANorm:
    """AUDIT M1 - the invariant the cluster-level path was missing."""

    def test_no_statistics_exist_for_the_noise_cluster(self, result: Any) -> None:
        assert NOISE_CLUSTER_ID not in result.statistics.cluster_stats.index

    def test_cluster_has_norm_is_false_for_noise(self, result: Any) -> None:
        assert result.statistics.cluster_has_norm(NOISE_CLUSTER_ID) is False
        frame = result.frame
        noise = frame["cluster_is_noise"]
        assert len(frame.loc[noise]) > 0
        assert not frame.loc[noise, "cluster_has_norm"].any()
        assert frame.loc[~noise, "cluster_has_norm"].all()

    def test_all_noise_cluster_deviations_are_nan(self, result: Any) -> None:
        frame = result.frame
        noise = frame["cluster_is_noise"]
        assert frame.loc[noise, "deviation_cluster_cost"].isna().all()

    def test_reason_is_propagated_as_cluster_noise(self, result: Any) -> None:
        frame = result.frame
        reasons = frame.loc[frame["cluster_is_noise"], "deviation_cluster_cost_reason"]
        assert (reasons == DEVIATION_REASON_CLUSTER_NOISE).any()
        # feature_missing outranks it, and nothing else may appear.
        assert set(reasons) <= {DEVIATION_REASON_CLUSTER_NOISE, "feature_missing"}

    def test_noise_records_keep_their_assignments(self, result: Any) -> None:
        """Barred from defining a norm; not deleted, not relabelled."""
        frame = result.frame
        noise = frame.loc[frame["cluster_is_noise"]]
        assert (noise["cluster_id"] == NOISE_CLUSTER_ID).all()
        assert noise["peer_cell_id"].notna().all()
        assert not noise["peer_cell_stable"].any()

    def test_noise_no_longer_influences_any_statistic(self, result: Any) -> None:
        stats = result.statistics.cluster_stats
        assert (stats.index != NOISE_CLUSTER_ID).all()
        assert stats["n_reference"].gt(0).all()


class TestEffectiveSampleSize:
    """AUDIT M3 - the guard and the estimator must count the same thing."""

    def test_both_counts_are_reported(self, result: Any) -> None:
        stats = result.statistics.cell_stats
        assert "n_reference" in stats.columns
        for feature in ("log_cost", "spend_ratio", "duration_days"):
            assert f"{feature}_n_effective" in stats.columns

    def test_no_norm_is_emitted_below_the_effective_threshold(
        self, result: Any
    ) -> None:
        stats = result.statistics.cell_stats
        for feature in ("log_cost", "spend_ratio", "duration_days"):
            thin = stats[stats[f"{feature}_n_effective"] < PEER_STAT_MIN_REFERENCE]
            assert thin[f"{feature}_median"].isna().all(), feature
            assert thin[f"{feature}_mad"].isna().all(), feature

    def test_every_emitted_norm_meets_the_effective_threshold(
        self, result: Any
    ) -> None:
        stats = result.statistics.cell_stats
        for feature in ("log_cost", "spend_ratio", "duration_days"):
            emitted = stats[stats[f"{feature}_median"].notna()]
            assert (
                emitted[f"{feature}_n_effective"] >= PEER_STAT_MIN_REFERENCE
            ).all(), feature

    def test_counts_may_legitimately_differ(self, result: Any) -> None:
        """The gap is exactly what M3 was about: membership != usable values."""
        stats = result.statistics.cell_stats
        gap = stats["n_reference"] - stats["duration_days_n_effective"]
        assert gap.max() > 0, "expected coverage to differ between the two counts"

    def test_cluster_stats_use_the_same_rule(self, result: Any) -> None:
        stats = result.statistics.cluster_stats
        for feature in ("log_cost", "spend_ratio", "duration_days"):
            emitted = stats[stats[f"{feature}_median"].notna()]
            assert (
                emitted[f"{feature}_n_effective"] >= PEER_STAT_MIN_REFERENCE
            ).all(), feature

    def test_withheld_counts_are_diagnosed(self, result: Any) -> None:
        diagnostics = result.statistics.diagnostics
        assert "withheld_for_small_effective_n" in diagnostics
        assert diagnostics["noise_cluster_excluded"] is True


class TestExtremeDeviationFlags:
    """AUDIT M4 - surface the tail without destroying it."""

    def test_raw_values_are_never_clipped(self, result: Any) -> None:
        frame = result.frame
        assert float(frame["deviation_cell_cost"].abs().max()) > 100.0

    def test_buckets_exist_for_every_deviation(self, result: Any) -> None:
        for name in result.deviations.names:
            assert f"{name}_bucket" in result.frame.columns
            assert set(result.frame[f"{name}_bucket"]) <= set(DEVIATION_BUCKETS)

    def test_bucket_boundaries_are_correct(self, result: Any) -> None:
        frame = result.frame
        for name in result.deviations.names:
            magnitude = frame[name].abs()
            bucket = frame[f"{name}_bucket"]
            assert (magnitude[bucket == "extreme"] >= Z_EXTREME_THRESHOLD).all()
            high = magnitude[bucket == "high"]
            assert ((high >= Z_HIGH_THRESHOLD) & (high < Z_EXTREME_THRESHOLD)).all()
            assert (magnitude[bucket == "normal"] < Z_HIGH_THRESHOLD).all()

    def test_undefined_is_its_own_bucket(self, result: Any) -> None:
        frame = result.frame
        for name in result.deviations.names:
            undefined = frame[name].isna()
            assert (frame.loc[undefined, f"{name}_bucket"] == "undefined").all()
            assert (frame.loc[~undefined, f"{name}_bucket"] != "undefined").all()

    def test_extremes_are_the_injected_garbage(
        self, result: Any, corpus: Corpus
    ) -> None:
        frame = result.frame
        extreme = frame.index[frame["deviation_cell_cost_bucket"] == "extreme"]
        assert len(extreme) > 0
        amounts = corpus.records.loc[extreme, "sanction_amount"]
        assert (amounts.abs() > 1e15).any()

    def test_counts_are_reported(self, result: Any) -> None:
        payload = result.deviations.to_dict()["deviations"]
        for name, entry in payload.items():
            assert "buckets" in entry and "n_extreme" in entry


class TestGlobalInvariantsAfterRemediation:
    """The four fixes must not have disturbed anything else."""

    def test_no_anomaly_scoring_introduced(self) -> None:
        forbidden = {"anomaly_score", "is_anomaly", "anomaly_type", "risk_score"}
        assert not forbidden & set(STAGE3_COLUMNS)

    def test_no_feature_leakage_into_clustering(self, corpus: Corpus) -> None:
        embedding = embed_work_names(corpus.records)
        tokens: set = set()
        for term in embedding.vocabulary:
            tokens.update(term.split())
        for column in ("district", "state", "vendor_name"):
            values = {
                word
                for value in corpus.records[column].dropna().unique()
                for word in str(value).lower().split()
            }
            assert not (tokens & values), column

    def test_stage_two_signals_unchanged(self, corpus: Corpus, tmp_path: Path) -> None:
        from src.stage2.confidence import BREAKDOWN_COLUMNS
        from src.stage3.pipeline import attach_structure

        snapshot = corpus.records[list(BREAKDOWN_COLUMNS)].copy(deep=True)
        attach_structure(corpus, config=SemanticConfig(artifact_dir=tmp_path))
        pd.testing.assert_frame_equal(
            snapshot, corpus.records[list(BREAKDOWN_COLUMNS)]
        )

    def test_determinism_preserved(self, corpus: Corpus, tmp_path: Path) -> None:
        config = SemanticConfig(artifact_dir=tmp_path)
        first = SemanticLayer(config).run(corpus)
        second = SemanticLayer(config).run(corpus)
        pd.testing.assert_frame_equal(first.frame, second.frame)

    def test_nan_reason_mapping_intact(self, result: Any) -> None:
        frame = result.frame
        for name in result.deviations.names:
            reason = frame[f"{name}_reason"]
            assert (reason[frame[name].isna()] != "defined").all(), name
            assert frame[name][reason == "defined"].notna().all(), name

    def test_deviation_formula_unchanged(self, result: Any) -> None:
        """Recompute a sample by hand against the stored norms."""
        frame, features = result.frame, result.features.frame
        stats = result.statistics.cell_stats
        defined = frame.loc[frame["deviation_cell_cost_reason"] == "defined"].head(40)
        for row_id, row in defined.iterrows():
            cell = stats.loc[int(row["peer_cell_id"])]
            expected = (
                features.loc[row_id, "log_cost"] - cell["log_cost_median"]
            ) / cell["log_cost_mad"]
            assert row["deviation_cell_cost"] == pytest.approx(expected, rel=1e-9)
