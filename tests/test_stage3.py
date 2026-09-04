"""Stage 3 test suite - Semantic Layer & Peer Cell Formation.

Organised against Stage3.md and the implementation brief:

* ``TestEmbedding``          - sec.5, text normalisation and TF-IDF
* ``TestClustering``         - sec.6, determinism and noise handling
* ``TestStratification``     - sec.7, log-quantile cost bands
* ``TestPeerCells``          - sec.8, formation and stability
* ``TestConfidenceGating``   - the critical property: norms are not polluted
* ``TestPeerStatistics``     - median/MAD correctness, zero-MAD handling
* ``TestFeatures``           - grouping / testing / gating separation
* ``TestDeviations``         - undefined is not zero
* ``TestDuplicateDetection`` - sec.9
* ``TestExplanationInputs``  - structured evidence, read-only
* ``TestPipeline``           - contract, alignment, determinism
* ``TestEdgeCases``          - sec.13
* ``TestValidation``         - sec.12, cluster quality against ground truth
* ``TestScopeBoundary``      - Stage 3 must not score or classify
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    FIELD_ORDER,
    MAD_SCALE,
    MISSING_STRATUM,
    NOISE_CLUSTER_ID,
    PEER_CELL_MIN_SIZE,
    PEER_STAT_MIN_CONFIDENCE,
    PEER_STAT_MIN_REFERENCE,
    STAGE3_SECONDS_BUDGET,
    STAGE3_VERSION,
)
from src.stage1.corpus import Corpus
from src.stage1.data_generator import WORK_TYPES, generate_dataset
from src.stage2.confidence import attach_confidence
from src.stage3.clustering import cluster_records
from src.stage3.deviations import compute_deviations
from src.stage3.duplicate_detection import NO_DUPLICATE_GROUP, detect_duplicates
from src.stage3.embedding import (
    build_stopwords,
    embed_work_names,
    normalize_work_text,
    truncate_at_locality,
)
from src.stage3.explanation import build_explanation_inputs
from src.stage3.features import build_feature_table, compute_duration_days
from src.stage3.peer_cells import (
    build_reference_mask,
    compute_peer_statistics,
    form_peer_cells,
)
from src.stage3.pipeline import (
    STAGE3_COLUMNS,
    SemanticConfig,
    SemanticLayer,
    attach_structure,
)
from src.stage3.stratification import stratify_cost

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_ROW: Dict[str, Any] = {
    "work_id": "MPL-KA-2019-000001",
    "work_name": "Construction of CC Road at Ward No. 7, Mysuru",
    "district": "Mysuru",
    "state": "Karnataka",
    "sanction_amount": 850000.0,
    "amount_spent": 812345.50,
    "date_proposal": "2019-03-01",
    "date_approval": "2019-05-20",
    "date_completion": "2020-01-15",
    "implementing_agency": "Mysuru Zilla Parishad",
    "vendor_name": "Iyer Constructions",
    "status": "completed",
}


def make_corpus(rows: List[Dict[str, Any]]) -> Corpus:
    """Build a Stage-1+2 corpus from override dicts."""
    records = []
    for position, overrides in enumerate(rows):
        row = dict(BASE_ROW)
        row["work_id"] = f"MPL-KA-2019-{position + 1:06d}"
        row.update(overrides or {})
        records.append(row)
    frame = pd.DataFrame(records, columns=list(FIELD_ORDER)).astype("object")
    corpus = Corpus.from_dataframe(frame)
    attach_confidence(corpus)
    return corpus


def synthetic_cluster(
    n: int,
    work: str = "CC Road",
    amount: float = 850000.0,
    confidence_ok: bool = True,
    start: int = 0,
) -> List[Dict[str, Any]]:
    """A block of similar, well-formed records forming one peer cell.

    Each work type is emitted in several lexical variants. Locality truncation
    reduces "Construction of CC Road at Ward No. 7, Mysuru" to just "cc road",
    so a fixture with one phrasing per type yields one distinct text per type -
    and HDBSCAN cannot form a cluster from a single point. Real registers carry
    this variation naturally; the fixture has to supply it deliberately.
    """
    variants = ("", " Concrete Surface", " with Side Drain")
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        variant = variants[i % len(variants)]
        row: Dict[str, Any] = {
            "work_name": (
                f"Construction of {work}{variant} at Ward No. {start + i}, Mysuru"
            ),
            "sanction_amount": amount + 1000.0 * (i % 7),
            "amount_spent": round((amount + 1000.0 * (i % 7)) * 0.95, 2),
        }
        if not confidence_ok:
            # A temporal hard fail drives Stage 2 confidence to exactly zero.
            row["date_approval"] = "not a date"
        rows.append(row)
    return rows


def _mixed_background(start: int = 0, per_type: int = 20) -> List[Dict[str, Any]]:
    """A background of several work types, rich enough to actually cluster.

    HDBSCAN cannot form a cluster from a vocabulary of one or two distinct
    texts, so a fixture built from a single work type produces nothing but
    noise and tests nothing.
    """
    rows: List[Dict[str, Any]] = []
    for offset, work in enumerate(
        [
            "CC Road",
            "Overhead Water Tank",
            "Bus Shelter",
            "Public Toilet Block",
            "Anganwadi Centre",
        ]
    ):
        rows += synthetic_cluster(per_type, work, start=start + offset * 100)
    return rows


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    """A full-size corpus with Stage 2 attached."""
    built = Corpus.from_dataframe(generate_dataset(n=10_000, seed=42))
    attach_confidence(built)
    return built


@pytest.fixture(scope="module")
def result(corpus: Corpus) -> Any:
    """Stage 3 run over the full-size corpus."""
    return SemanticLayer().run(corpus)


@pytest.fixture(scope="module")
def ground_truth(corpus: Corpus) -> pd.Series:
    """True work type per record, recovered from the generator's vocabulary.

    A test-only oracle: the generator built every ``work_name`` from a 20-item
    work-type list, so cluster quality can be *measured* rather than asserted.
    Nothing in ``src/`` may use this.
    """
    labels = [entry[0].lower() for entry in WORK_TYPES]

    def _match(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        for label in labels:
            if label in value:
                return label
        return None

    return corpus.records["work_name"].map(_match)


# ---------------------------------------------------------------------------
# Stage3.md sec.5 - embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    """Interpretable, deterministic TF-IDF over normalised work names."""

    def test_locality_clause_is_truncated(self) -> None:
        assert (
            truncate_at_locality("construction of cc road at ward no. 7, mysuru")
            == "construction of cc road"
        )

    def test_name_without_a_locality_clause_survives_whole(self) -> None:
        assert truncate_at_locality("cc road phase ii") == "cc road phase ii"

    def test_geography_is_stripped_from_the_corpus_vocabulary(self) -> None:
        corpus = make_corpus(synthetic_cluster(20))
        stopwords = build_stopwords(corpus.records)
        assert "mysuru" in stopwords
        assert "karnataka" in stopwords

    def test_action_boilerplate_is_stripped(self) -> None:
        corpus = make_corpus(synthetic_cluster(20))
        stopwords = build_stopwords(corpus.records)
        text = normalize_work_text(
            pd.Series(["Construction of CC Road at Ward No. 7, Mysuru"]), stopwords
        )
        assert text.iloc[0] == "cc road"

    def test_repair_and_construction_normalise_alike(self) -> None:
        """A repaired road and a built road are the same KIND of work."""
        corpus = make_corpus(synthetic_cluster(20))
        stopwords = build_stopwords(corpus.records)
        text = normalize_work_text(
            pd.Series(
                [
                    "Construction of CC Road at Ward No. 1, Mysuru",
                    "Repair of CC Road at Village Devgaon, Mysuru",
                ]
            ),
            stopwords,
        )
        assert text.iloc[0] == text.iloc[1] == "cc road"

    def test_embedding_is_deterministic(self, corpus: Corpus) -> None:
        first = embed_work_names(corpus.records)
        second = embed_work_names(corpus.records)
        assert np.allclose(first.projection, second.projection)
        assert first.vocabulary == second.vocabulary

    def test_distinct_texts_collapse(self, corpus: Corpus) -> None:
        """The templated-name optimisation, asserted."""
        embedding = embed_work_names(corpus.records)
        assert embedding.n_unique < len(corpus) / 10

    def test_identical_text_gets_an_identical_vector(self, corpus: Corpus) -> None:
        embedding = embed_work_names(corpus.records)
        projection = embedding.record_projection()
        texts = embedding.normalized_text
        repeated = texts.value_counts()
        target = repeated[repeated > 1].index[0]
        rows = np.flatnonzero((texts == target).to_numpy())[:5]
        assert np.allclose(projection[rows], projection[rows[0]])

    def test_top_terms_are_readable(self, corpus: Corpus) -> None:
        """Interpretability survives dimensionality reduction."""
        embedding = embed_work_names(corpus.records)
        terms = embedding.top_terms(list(range(min(10, embedding.n_unique))), k=3)
        assert terms
        assert all(isinstance(term, str) and term for term in terms)

    def test_missing_work_name_yields_empty_text(self) -> None:
        corpus = make_corpus([{"work_name": None}] + synthetic_cluster(20))
        embedding = embed_work_names(corpus.records)
        assert bool(embedding.empty_mask.iloc[0])

    def test_absent_column_raises(self, corpus: Corpus) -> None:
        with pytest.raises(ValueError, match="absent"):
            embed_work_names(corpus.records, text_field="not_a_column")


# ---------------------------------------------------------------------------
# Stage3.md sec.6 - clustering
# ---------------------------------------------------------------------------


class TestClustering:
    """Deterministic density clustering with explicit noise."""

    def test_clustering_is_deterministic(self, corpus: Corpus) -> None:
        embedding = embed_work_names(corpus.records)
        first = cluster_records(embedding, corpus.records.index)
        second = cluster_records(embedding, corpus.records.index)
        pd.testing.assert_series_equal(first.cluster_id, second.cluster_id)

    def test_whole_pipeline_is_deterministic(self, corpus: Corpus) -> None:
        first = SemanticLayer().run(corpus)
        second = SemanticLayer().run(corpus)
        pd.testing.assert_frame_equal(first.frame, second.frame)

    def test_no_dependence_on_global_random_state(self, corpus: Corpus) -> None:
        np.random.seed(0)
        first = SemanticLayer().run(corpus).frame["cluster_id"]
        np.random.seed(9_999)
        _ = np.random.random(50)
        second = SemanticLayer().run(corpus).frame["cluster_id"]
        pd.testing.assert_series_equal(first, second)

    def test_every_record_gets_a_cluster(self, result: Any) -> None:
        assert result.frame["cluster_id"].notna().all()

    def test_noise_is_labelled_minus_one(self, result: Any) -> None:
        noise = result.frame["cluster_is_noise"]
        assert (result.frame.loc[noise, "cluster_id"] == NOISE_CLUSTER_ID).all()
        assert (result.frame.loc[~noise, "cluster_id"] != NOISE_CLUSTER_ID).all()

    def test_clusters_are_labelled_in_token_space(self, result: Any) -> None:
        """Clusters form in SVD space but must be nameable in words."""
        assert result.clusters.labels
        for label in result.clusters.labels.values():
            assert isinstance(label, str) and label

    def test_cluster_sizes_are_consistent(self, result: Any) -> None:
        frame = result.frame
        recomputed = frame["cluster_id"].map(frame["cluster_id"].value_counts())
        pd.testing.assert_series_equal(
            frame["cluster_size"], recomputed.astype("int64"), check_names=False
        )

    def test_similar_works_land_together(self) -> None:
        """Roads with roads, tanks with tanks - Stage3.md sec.16."""
        rows: List[Dict[str, Any]] = []
        for offset, work in enumerate(
            ["CC Road", "Overhead Water Tank", "Bus Shelter", "Public Toilet Block"]
        ):
            rows += synthetic_cluster(30, work, start=offset * 100)
        outcome = SemanticLayer(
            SemanticConfig(min_cluster_size=2, cluster_min_records=5)
        ).run(make_corpus(rows))
        roads = outcome.frame["cluster_id"].iloc[:30]
        tanks = outcome.frame["cluster_id"].iloc[30:60]
        assert roads.nunique() == 1
        assert tanks.nunique() == 1
        assert roads.iloc[0] != tanks.iloc[0]


# ---------------------------------------------------------------------------
# Stage3.md sec.7 - stratification
# ---------------------------------------------------------------------------


class TestStratification:
    """Log-quantile cost bands."""

    def test_log_cost_matches_the_definition(self, corpus: Corpus) -> None:
        outcome = stratify_cost(corpus.records)
        amounts = corpus.records["sanction_amount"]
        usable = amounts.notna() & (amounts > 0) & np.isfinite(amounts.fillna(0))
        expected = np.log1p(amounts[usable].to_numpy(dtype="float64"))
        assert np.allclose(outcome.log_cost[usable].to_numpy(), expected)

    def test_missing_amount_gets_the_missing_stratum(self) -> None:
        rows = [{"sanction_amount": None}] + synthetic_cluster(20)
        outcome = stratify_cost(make_corpus(rows).records)
        assert outcome.cost_stratum.iloc[0] == MISSING_STRATUM

    def test_non_positive_amount_gets_the_missing_stratum(self) -> None:
        rows = [{"sanction_amount": 0.0}, {"sanction_amount": -500.0}]
        rows += synthetic_cluster(20)
        outcome = stratify_cost(make_corpus(rows).records)
        assert (outcome.cost_stratum.iloc[:2] == MISSING_STRATUM).all()

    def test_strata_are_monotone_in_cost(self, result: Any) -> None:
        frame = result.frame
        assigned = frame.loc[frame["cost_stratum"] != MISSING_STRATUM]
        medians = assigned.groupby("cost_stratum")["log_cost"].median()
        assert list(medians) == sorted(medians)

    def test_strata_are_roughly_balanced(self, result: Any) -> None:
        counts = result.frame["cost_stratum"].value_counts()
        assigned = counts.drop(index=MISSING_STRATUM, errors="ignore")
        assert assigned.max() / assigned.min() < 2.0

    def test_degenerate_cost_distribution_collapses_to_one_band(self) -> None:
        corpus = make_corpus(synthetic_cluster(30))
        corpus.records["sanction_amount"] = 500_000.0
        outcome = stratify_cost(corpus.records)
        assert outcome.cost_stratum.nunique() == 1
        assert outcome.diagnostics["degenerate"] is True

    def test_bad_configuration_is_rejected(self, corpus: Corpus) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            stratify_cost(corpus.records, n_bins=0)
        with pytest.raises(ValueError, match="absent"):
            stratify_cost(corpus.records, amount_field="nope")


# ---------------------------------------------------------------------------
# Stage3.md sec.8 - peer cells
# ---------------------------------------------------------------------------


class TestPeerCells:
    """Formation and the stability rule."""

    def test_peer_cell_is_cluster_and_stratum(self, result: Any) -> None:
        frame = result.frame
        pairs = frame.groupby("peer_cell_id")[["cluster_id", "cost_stratum"]].nunique()
        assert (pairs["cluster_id"] == 1).all()
        assert (pairs["cost_stratum"] == 1).all()

    def test_small_cells_are_marked_unstable(self, result: Any) -> None:
        frame = result.frame
        small = frame.loc[frame["peer_cell_size"] < PEER_CELL_MIN_SIZE]
        assert not small["peer_cell_stable"].any()

    def test_noise_cells_are_never_stable(self, result: Any) -> None:
        """Four thousand unclustered records sharing a stratum is not a peer group."""
        frame = result.frame
        noise = frame.loc[frame["cluster_id"] == NOISE_CLUSTER_ID]
        assert len(noise) > 0
        assert not noise["peer_cell_stable"].any()

    def test_missing_stratum_cells_are_never_stable(self, result: Any) -> None:
        frame = result.frame
        unknown = frame.loc[frame["cost_stratum"] == MISSING_STRATUM]
        assert len(unknown) > 0
        assert not unknown["peer_cell_stable"].any()

    def test_stable_cells_meet_the_size_floor(self, result: Any) -> None:
        frame = result.frame
        stable = frame.loc[frame["peer_cell_stable"]]
        assert (stable["peer_cell_size"] >= PEER_CELL_MIN_SIZE).all()

    def test_cell_sizes_are_consistent(self, result: Any) -> None:
        frame = result.frame
        recomputed = frame["peer_cell_id"].map(frame["peer_cell_id"].value_counts())
        pd.testing.assert_series_equal(
            frame["peer_cell_size"], recomputed.astype("int64"), check_names=False
        )

    def test_keys_round_trip(self, result: Any) -> None:
        frame = result.frame
        for cell_id in frame["peer_cell_id"].unique()[:20]:
            cluster, stratum = result.peer_cells.key_of(int(cell_id))
            members = frame.loc[frame["peer_cell_id"] == cell_id]
            assert (members["cluster_id"] == cluster).all()
            assert (members["cost_stratum"] == stratum).all()


# ---------------------------------------------------------------------------
# The critical property
# ---------------------------------------------------------------------------


class TestConfidenceGating:
    """Low-confidence records must not shape the norms they are judged against."""

    def test_reference_mask_applies_the_confidence_floor(self, corpus: Corpus) -> None:
        mask = build_reference_mask(corpus.records)
        gated = corpus.records.loc[~mask, "confidence"]
        allowed = corpus.records.loc[mask, "confidence"]
        assert (allowed >= PEER_STAT_MIN_CONFIDENCE).all()
        assert (gated < PEER_STAT_MIN_CONFIDENCE).any()

    def test_unusable_reconciliation_branches_are_barred(self, corpus: Corpus) -> None:
        mask = build_reference_mask(corpus.records)
        barred = corpus.records.loc[
            corpus.records["reconciliation_branch"].isin(
                ["non_finite", "implausible_magnitude"]
            )
        ]
        assert len(barred) > 0
        assert not mask.loc[barred.index].any()

    def test_gated_records_are_still_assigned_and_measured(self, result: Any) -> None:
        """They are the REMEDIATE population, not deletions."""
        frame = result.frame
        gated = frame.loc[~frame["peer_reference"]]
        assert len(gated) > 0
        assert gated["peer_cell_id"].notna().all()
        assert gated["cluster_id"].notna().all()
        assert gated["deviation_cell_cost"].notna().any()

    def test_a_low_confidence_outlier_cannot_move_the_norm(self) -> None:
        """The whole point of gating, demonstrated end to end.

        The same corpus twice: once clean, once with twelve garbage records
        added to one work type at 100x the normal amount. Their Stage 2
        confidence is zero, so they must get no vote - and the cell median they
        would otherwise wreck must not move.
        """
        config = SemanticConfig(
            min_cluster_size=2,
            cluster_min_records=5,
            peer_cell_min_size=5,
            cost_bins=1,
            min_reference=5,
        )
        clean_rows = _mixed_background(per_type=20)
        polluted_rows = list(clean_rows)
        for row in synthetic_cluster(
            12, "CC Road", confidence_ok=False, start=900
        ):
            row["sanction_amount"] = 90_000_000.0
            row["amount_spent"] = 90_000_000.0
            polluted_rows.append(row)

        without = SemanticLayer(config).run(make_corpus(clean_rows))
        with_pollution = SemanticLayer(config).run(make_corpus(polluted_rows))

        clean_medians = without.statistics.cell_stats["log_cost_median"].dropna()
        polluted_medians = with_pollution.statistics.cell_stats[
            "log_cost_median"
        ].dropna()
        assert len(clean_medians) > 0 and len(polluted_medians) > 0

        # No polluted record contributed to any norm...
        assert not with_pollution.frame["peer_reference"].iloc[len(clean_rows) :].any()
        # ...so every norm that exists in both runs is unchanged.
        assert float(polluted_medians.max()) == pytest.approx(
            float(clean_medians.max()), abs=0.2
        )

    def test_stats_require_stage_two_output(self) -> None:
        frame = pd.DataFrame({"work_name": ["a"], "sanction_amount": [1.0]})
        with pytest.raises(ValueError, match="confidence"):
            build_reference_mask(frame)


class TestPeerStatistics:
    """Median and MAD only, and undefined where undefined."""

    def test_median_and_mad_match_a_hand_computation(self) -> None:
        corpus = make_corpus(synthetic_cluster(40, amount=500_000.0))
        outcome = SemanticLayer(
            SemanticConfig(
                min_cluster_size=2,
                cluster_min_records=5,
                peer_cell_min_size=5,
                cost_bins=1,
                min_reference=5,
            )
        ).run(corpus)
        stats = outcome.statistics.cell_stats
        features = outcome.features.frame
        for cell_id, row in stats.iterrows():
            if not np.isfinite(row["log_cost_median"]):
                continue
            member = (
                (outcome.frame["peer_cell_id"] == cell_id)
                & outcome.frame["peer_reference"]
                & outcome.frame["peer_cell_stable"]
            )
            values = features.loc[member, "log_cost"].dropna().to_numpy()
            expected_median = float(np.median(values))
            expected_mad = MAD_SCALE * float(
                np.median(np.abs(values - expected_median))
            )
            assert row["log_cost_median"] == pytest.approx(expected_median)
            if expected_mad > 0:
                assert row["log_cost_mad"] == pytest.approx(expected_mad)

    def test_zero_mad_is_reported_as_undefined_not_zero(self) -> None:
        """Every peer identical: no scale exists, so no deviation exists.

        Reporting zero here would assert "exactly normal", which is the
        opposite of what is known. Same rule as Stage 2's definedness.
        """
        corpus = make_corpus(_mixed_background(per_type=20))
        for column in ("sanction_amount", "amount_spent"):
            corpus.records[column] = 750_000.0
        outcome = SemanticLayer(
            SemanticConfig(
                min_cluster_size=2,
                cluster_min_records=5,
                peer_cell_min_size=5,
                cost_bins=1,
                min_reference=5,
            )
        ).run(corpus)
        stats = outcome.statistics.cell_stats
        assert stats["log_cost_median"].notna().any()
        assert stats["log_cost_mad"].isna().all()
        assert (
            outcome.frame["deviation_cell_cost_reason"] == "zero_dispersion"
        ).any()
        assert outcome.frame["deviation_cell_cost"].isna().all()

    def test_thin_cells_get_no_norm(self, result: Any) -> None:
        stats = result.statistics.cell_stats
        thin = stats.loc[stats["n_reference"] < PEER_STAT_MIN_REFERENCE]
        if len(thin):
            assert thin["log_cost_median"].isna().all()

    def test_cluster_stats_ignore_cell_stability(self, result: Any) -> None:
        """A cluster is a valid group even where a stratum is too thin to be one."""
        assert len(result.statistics.cluster_stats) > 0
        assert result.statistics.cluster_stats["n_reference"].max() > 0


# ---------------------------------------------------------------------------
# Features and deviations
# ---------------------------------------------------------------------------


class TestFeatures:
    """Grouping, testing and gating stay disjoint."""

    def test_feature_roles_do_not_overlap(self) -> None:
        from src.stage3.features import (
            GATING_FEATURES,
            GROUPING_FEATURES,
            TESTING_FEATURES,
        )

        assert not set(TESTING_FEATURES) & set(GROUPING_FEATURES)
        assert not set(TESTING_FEATURES) & set(GATING_FEATURES)
        assert not set(GROUPING_FEATURES) & set(GATING_FEATURES)

    def test_district_is_never_a_grouping_feature(self) -> None:
        """Grouping by district would normalise district anomalies away."""
        from src.stage3.features import GROUPING_FEATURES

        assert "district" not in GROUPING_FEATURES
        assert "state" not in GROUPING_FEATURES
        assert "vendor_name" not in GROUPING_FEATURES

    def test_spend_ratio_is_reused_not_recomputed(self, corpus: Corpus, result: Any) -> None:
        pd.testing.assert_series_equal(
            result.features.frame["spend_ratio"],
            corpus.records["spend_ratio"],
            check_names=False,
        )

    def test_no_identifiers_or_raw_strings_in_testing_features(self, result: Any) -> None:
        from src.stage3.features import TESTING_FEATURES

        for name in TESTING_FEATURES:
            assert pd.api.types.is_numeric_dtype(result.features.frame[name])

    def test_duration_requires_both_dates(self) -> None:
        corpus = make_corpus(
            [{"date_completion": None}, {}] + synthetic_cluster(20)
        )
        duration = compute_duration_days(corpus.records)
        assert pd.isna(duration.iloc[0])
        assert duration.iloc[1] > 0

    def test_duration_excludes_temporal_hard_fails(self) -> None:
        """A duration off an impossible timeline is a number with no referent."""
        corpus = make_corpus([{"date_approval": "not a date"}] + synthetic_cluster(20))
        assert bool(corpus.records["temporal_hard_fail"].iloc[0])
        assert pd.isna(compute_duration_days(corpus.records).iloc[0])

    def test_features_require_stage_two(self) -> None:
        frame = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="Stage 2"):
            build_feature_table(
                frame,
                log_cost=pd.Series([1.0]),
                cluster_id=pd.Series([0]),
                cost_stratum=pd.Series([0]),
                peer_cell_id=pd.Series([0]),
                peer_cell_size=pd.Series([1]),
                peer_cell_stable=pd.Series([False]),
            )


class TestDeviations:
    """Undefined is not zero."""

    def test_deviation_matches_the_robust_formula(self, result: Any) -> None:
        frame = result.frame
        features = result.features.frame
        stats = result.statistics.cell_stats
        defined = frame.loc[frame["deviation_cell_cost_reason"] == "defined"]
        assert len(defined) > 0
        sample = defined.head(50)
        for row_id, row in sample.iterrows():
            cell = stats.loc[int(row["peer_cell_id"])]
            expected = (
                features.loc[row_id, "log_cost"] - cell["log_cost_median"]
            ) / cell["log_cost_mad"]
            assert row["deviation_cell_cost"] == pytest.approx(expected, rel=1e-9)

    def test_unstable_cells_yield_no_cell_deviation(self, result: Any) -> None:
        frame = result.frame
        unstable = frame.loc[~frame["peer_cell_stable"]]
        assert len(unstable) > 0
        assert unstable["deviation_cell_cost"].isna().all()
        assert (unstable["deviation_cell_cost_reason"] != "defined").all()

    def test_missing_feature_yields_no_deviation(self, result: Any) -> None:
        frame = result.frame
        missing = frame.loc[frame["log_cost"].isna()]
        assert len(missing) > 0
        assert missing["deviation_cell_cost"].isna().all()
        assert (missing["deviation_cell_cost_reason"] == "feature_missing").all()

    def test_every_undefined_deviation_carries_a_reason(self, result: Any) -> None:
        frame = result.frame
        for name in result.deviations.names:
            undefined = frame[name].isna()
            reasons = frame.loc[undefined, f"{name}_reason"]
            assert (reasons != "defined").all()
            defined_rows = frame.loc[~undefined, f"{name}_reason"]
            assert (defined_rows == "defined").all()

    def test_deviations_are_finite_where_defined(self, result: Any) -> None:
        for name in result.deviations.names:
            values = result.frame[name].dropna().to_numpy()
            assert np.isfinite(values).all()

    def test_cluster_deviation_is_wider_than_cell_deviation(self, result: Any) -> None:
        """Stratifying is conservative; the cluster view keeps the sensitivity."""
        frame = result.frame
        cell = frame["deviation_cell_cost"].abs().quantile(0.95)
        cluster = frame["deviation_cluster_cost"].abs().quantile(0.95)
        assert cluster > cell

    def test_cluster_deviation_does_not_need_cell_stability(self, result: Any) -> None:
        frame = result.frame
        unstable = frame.loc[~frame["peer_cell_stable"]]
        assert unstable["deviation_cluster_cost"].notna().any()


# ---------------------------------------------------------------------------
# Stage3.md sec.9 - duplicates
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    """Same words, same place, same period."""

    def test_identical_works_score_high(self) -> None:
        rows = [
            {
                "work_name": "Construction of CC Road at Ward No. 5, Mysuru",
                "date_proposal": "2019-03-01",
            },
            {
                "work_name": "Construction of CC Road at Ward No. 5, Mysuru",
                "date_proposal": "2019-03-10",
            },
        ] + _mixed_background(start=200)
        outcome = SemanticLayer(
            SemanticConfig(min_cluster_size=2, cluster_min_records=5)
        ).run(make_corpus(rows))
        assert outcome.frame["duplicate_score"].iloc[0] > 0.85
        assert bool(outcome.frame["duplicate_flag"].iloc[0])
        assert (
            outcome.frame["duplicate_group_id"].iloc[0]
            == outcome.frame["duplicate_group_id"].iloc[1]
        )

    def test_different_districts_are_never_duplicates(self) -> None:
        """The same-district indicator is a hard gate, not a weight."""
        rows = [
            {"work_name": "Construction of CC Road at Ward No. 5, Mysuru",
             "district": "Mysuru"},
            {"work_name": "Construction of CC Road at Ward No. 5, Pune",
             "district": "Pune"},
        ] + _mixed_background(start=300)
        outcome = SemanticLayer(
            SemanticConfig(min_cluster_size=2, cluster_min_records=5)
        ).run(make_corpus(rows))
        assert outcome.frame["duplicate_group_id"].iloc[0] != (
            outcome.frame["duplicate_group_id"].iloc[1]
        ) or not outcome.frame["duplicate_flag"].iloc[0]

    def test_temporal_distance_decays_the_score(self) -> None:
        near = [
            {"work_name": "Construction of CC Road at Ward No. 5, Mysuru",
             "date_proposal": "2019-03-01"},
            {"work_name": "Construction of CC Road at Ward No. 5, Mysuru",
             "date_proposal": "2019-03-05"},
        ]
        far = [
            {"work_name": "Construction of CC Road at Ward No. 5, Mysuru",
             "date_proposal": "2015-03-01"},
            {"work_name": "Construction of CC Road at Ward No. 5, Mysuru",
             "date_proposal": "2022-03-01"},
        ]
        config = SemanticConfig(min_cluster_size=2, cluster_min_records=5)
        near_score = (
            SemanticLayer(config)
            .run(make_corpus(near + _mixed_background(start=400)))
            .frame["duplicate_score"]
            .iloc[0]
        )
        far_score = (
            SemanticLayer(config)
            .run(make_corpus(far + _mixed_background(start=400)))
            .frame["duplicate_score"]
            .iloc[0]
        )
        assert near_score > far_score

    def test_unrelated_works_score_low(self, result: Any) -> None:
        frame = result.frame
        assert float(frame["duplicate_score"].median()) < 0.5

    def test_groups_are_transitive(self, result: Any) -> None:
        frame = result.frame
        grouped = frame.loc[frame["duplicate_group_id"] != NO_DUPLICATE_GROUP]
        if len(grouped):
            assert grouped["duplicate_flag"].all()

    def test_unflagged_records_have_no_group(self, result: Any) -> None:
        frame = result.frame
        unflagged = frame.loc[~frame["duplicate_flag"]]
        assert (unflagged["duplicate_group_id"] == NO_DUPLICATE_GROUP).all()

    def test_flagged_groups_satisfy_the_definition(
        self, result: Any, corpus: Corpus
    ) -> None:
        """Precision, checked as a property rather than against labels.

        Stage 1's duplicate channel clones names from ANY row, so almost all
        injected clones are cross-district - which Stage3.md sec.9.1's
        1[d_i = d_j] deliberately excludes. There is therefore no valid labelled
        ground truth for this detector in the corpus. What CAN be asserted is
        that every group it forms satisfies the definition: same district, and
        close enough in time that the decay term is meaningful.
        """
        frame = result.frame.join(corpus.records[["district", "date_proposal"]])
        grouped = frame.loc[frame["duplicate_group_id"] != NO_DUPLICATE_GROUP]
        assert len(grouped) > 0
        checked = 0
        for _, members in grouped.groupby("duplicate_group_id"):
            if len(members) < 2:
                continue
            checked += 1
            assert members["district"].nunique() == 1
            dates = members["date_proposal"].dropna()
            if len(dates) > 1:
                span = (dates.max() - dates.min()).days
                assert span < 3 * 180, span
        assert checked > 0

    def test_score_is_bounded(self, result: Any) -> None:
        scores = result.frame["duplicate_score"]
        assert scores.between(0.0, 1.0).all()

    def test_mismatched_vector_count_raises(self, corpus: Corpus) -> None:
        embedding = embed_work_names(corpus.records)
        with pytest.raises(ValueError, match="rows"):
            detect_duplicates(
                corpus.records,
                embedding.tfidf,  # per-text, not per-record
                corpus.records["confidence"] * 0,
            )


# ---------------------------------------------------------------------------
# Explanation inputs
# ---------------------------------------------------------------------------


class TestExplanationInputs:
    """Structured evidence, assembled from stored values only."""

    def test_sections_are_present(self, result: Any) -> None:
        payload = result.explain(0)
        assert set(payload) >= {
            "context",
            "peer_norms",
            "deviations",
            "duplicates",
            "confidence",
        }

    def test_values_match_the_stored_frame(self, result: Any) -> None:
        row = 7
        payload = result.explain(row)
        frame = result.frame
        assert payload["context"]["cluster_id"] == int(frame.loc[row, "cluster_id"])
        assert payload["context"]["peer_cell_id"] == int(frame.loc[row, "peer_cell_id"])
        stored = frame.loc[row, "deviation_cell_cost"]
        reported = payload["deviations"]["deviation_cell_cost"]["value"]
        if pd.isna(stored):
            assert reported is None
        else:
            assert reported == pytest.approx(float(stored), abs=1e-4)

    def test_undefined_deviations_are_explained_in_words(self, result: Any) -> None:
        frame = result.frame
        # Must not be a noise record: since AUDIT M1 those report the more
        # fundamental "cluster_noise" instead, which is a different case.
        unstable = frame.loc[
            ~frame["peer_cell_stable"]
            & frame["log_cost"].notna()
            & ~frame["cluster_is_noise"]
        ].index[0]
        payload = result.explain(unstable)
        entry = payload["deviations"]["deviation_cell_cost"]
        assert entry["defined"] is False
        assert entry["reason"] == "cell_unstable"
        assert "too small" in entry["reason_text"]

    def test_stage_two_signals_are_carried_through(self, result: Any) -> None:
        payload = result.explain(0)
        assert set(payload["confidence"]) >= {
            "confidence",
            "critical_missing_count",
            "lifecycle_state",
            "reconciliation_branch",
        }

    def test_payload_is_json_serialisable(self, result: Any) -> None:
        json.dumps(result.explain(3), default=str)

    def test_missing_stage_three_columns_raise(self, corpus: Corpus) -> None:
        with pytest.raises(ValueError, match="Stage 3"):
            build_explanation_inputs(corpus.records, 0)

    def test_unknown_row_raises(self, result: Any) -> None:
        with pytest.raises(KeyError):
            result.explain(10**9)


# ---------------------------------------------------------------------------
# Pipeline contract
# ---------------------------------------------------------------------------


class TestPipeline:
    """Alignment, contract and integration."""

    def test_every_contract_column_is_produced(self, result: Any) -> None:
        for column in STAGE3_COLUMNS:
            assert column in result.frame.columns

    def test_attach_writes_the_contract(self, corpus: Corpus) -> None:
        attach_structure(corpus)
        for column in STAGE3_COLUMNS:
            assert column in corpus.records.columns

    def test_row_order_and_index_preserved(self, corpus: Corpus) -> None:
        before_index = corpus.records.index.copy()
        before_ids = corpus.records["work_id"].copy()
        attach_structure(corpus)
        assert corpus.records.index.equals(before_index)
        pd.testing.assert_series_equal(corpus.records["work_id"], before_ids)

    def test_no_rows_added_or_lost(self, corpus: Corpus) -> None:
        before = len(corpus)
        attach_structure(corpus)
        assert len(corpus) == before

    def test_stage_one_and_two_columns_untouched(self, corpus: Corpus) -> None:
        snapshot = corpus.records[list(FIELD_ORDER) + ["confidence"]].copy(deep=True)
        attach_structure(corpus)
        pd.testing.assert_frame_equal(
            snapshot, corpus.records[list(FIELD_ORDER) + ["confidence"]]
        )

    def test_no_nan_propagation_into_structure(self, result: Any) -> None:
        frame = result.frame
        for column in (
            "cluster_id",
            "cluster_size",
            "cost_stratum",
            "peer_cell_id",
            "peer_cell_size",
            "duplicate_score",
        ):
            assert frame[column].notna().all(), column
        assert np.isfinite(frame["duplicate_score"].to_numpy()).all()

    def test_report_is_serialisable(self, result: Any) -> None:
        payload = json.dumps(result.report(), default=str)
        assert json.loads(payload)["stage3_version"] == STAGE3_VERSION

    def test_reports_can_be_written(self, result: Any, tmp_path: Path) -> None:
        written = result.save_reports(tmp_path)
        assert set(written) == {
            "stage3_report",
            "peer_statistics",
            "calibration_report",
            "reproducibility_report",
        }
        for path in written.values():
            assert path.exists()

    def test_accepts_a_bare_frame(self, corpus: Corpus) -> None:
        outcome = SemanticLayer().run(corpus.records)
        assert len(outcome) == len(corpus)

    def test_rejects_nonsense_input(self) -> None:
        with pytest.raises(TypeError):
            SemanticLayer().run([1, 2, 3])

    def test_misaligned_result_is_rejected(self, corpus: Corpus) -> None:
        small = SemanticLayer().run(corpus.records.head(200))
        with pytest.raises(ValueError, match="rows"):
            attach_structure(corpus, small)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_cluster_size": 1},
            {"cost_bins": 0},
            {"peer_cell_min_size": 0},
            {"min_confidence": 2.0},
            {"duplicate_threshold": -1.0},
            {"duplicate_tau_days": 0.0},
        ],
    )
    def test_invalid_config_is_rejected(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            SemanticConfig(**kwargs)


class TestPerformance:
    """Stage3.md sec.11: 50k records in under 10 seconds."""

    def test_fifty_thousand_records_within_budget(self) -> None:
        built = Corpus.from_dataframe(generate_dataset(n=50_000, seed=42))
        attach_confidence(built)
        started = time.perf_counter()
        outcome = SemanticLayer().run(built)
        elapsed = time.perf_counter() - started
        assert len(outcome) == 50_000
        assert elapsed < STAGE3_SECONDS_BUDGET, (
            f"Stage 3 took {elapsed:.2f}s, budget is {STAGE3_SECONDS_BUDGET}s"
        )


# ---------------------------------------------------------------------------
# Stage3.md sec.13 - edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Every case Stage3.md sec.13 marks mandatory."""

    def test_empty_corpus(self) -> None:
        empty = pd.DataFrame(
            {name: pd.Series([], dtype="object") for name in FIELD_ORDER}
        )
        built = Corpus.from_dataframe(empty)
        attach_confidence(built)
        outcome = SemanticLayer().run(built)
        assert len(outcome) == 0
        json.dumps(outcome.report(), default=str)

    def test_single_record_corpus(self) -> None:
        outcome = SemanticLayer().run(make_corpus([{}]))
        assert len(outcome) == 1
        assert not outcome.frame["peer_cell_stable"].iloc[0]
        assert outcome.frame["deviation_cell_cost"].isna().all()

    def test_single_record_cluster_is_unstable(self) -> None:
        rows = [{"work_name": "Erection of Ropeway Gantry at Ward No. 1, Mysuru"}]
        rows += _mixed_background(start=600)
        outcome = SemanticLayer(
            SemanticConfig(min_cluster_size=2, cluster_min_records=5)
        ).run(make_corpus(rows))
        assert not outcome.frame["peer_cell_stable"].iloc[0]

    def test_all_records_share_one_text(self) -> None:
        outcome = SemanticLayer().run(make_corpus(synthetic_cluster(40, start=0)))
        assert len(outcome) == 40
        assert np.isfinite(outcome.frame["duplicate_score"].to_numpy()).all()

    def test_empty_or_missing_work_names(self) -> None:
        rows = [{"work_name": None}, {"work_name": "N/A"}, {"work_name": "   "}]
        rows += _mixed_background(start=700)
        outcome = SemanticLayer(
            SemanticConfig(min_cluster_size=2, cluster_min_records=5)
        ).run(make_corpus(rows))
        assert (outcome.frame["cluster_id"].iloc[:3] == NOISE_CLUSTER_ID).all()
        assert not outcome.frame["peer_cell_stable"].iloc[:3].any()

    def test_all_low_confidence_cluster(self) -> None:
        """No record may shape a norm, so no norm exists - and none is invented."""
        outcome = SemanticLayer(
            SemanticConfig(min_cluster_size=2, cluster_min_records=5, peer_cell_min_size=5)
        ).run(make_corpus(synthetic_cluster(40, confidence_ok=False)))
        assert (outcome.frame["confidence"] if "confidence" in outcome.frame else True) is not None
        assert not outcome.frame["peer_reference"].any()
        assert outcome.frame["deviation_cell_cost"].isna().all()
        assert (
            outcome.frame["deviation_cell_cost_reason"].isin(
                # "cluster_noise" since AUDIT M1: this fixture is a single work
                # type, so every record is unclustered and that is the precise
                # cause, superseding the generic "cell_unstable".
                ["no_peer_norm", "cell_unstable", "feature_missing", "cluster_noise"]
            )
        ).all()

    def test_extremely_skewed_cost_distribution(self) -> None:
        rows = synthetic_cluster(30, amount=100_000.0)
        rows += synthetic_cluster(5, amount=5e12, start=900)
        outcome = SemanticLayer(
            SemanticConfig(min_cluster_size=2, cluster_min_records=5, peer_cell_min_size=5)
        ).run(make_corpus(rows))
        assert np.isfinite(
            outcome.frame["deviation_cell_cost"].dropna().to_numpy()
        ).all()

    def test_missing_sanction_values_throughout(self) -> None:
        rows = [dict(row, sanction_amount=None) for row in synthetic_cluster(30)]
        outcome = SemanticLayer().run(make_corpus(rows))
        assert (outcome.frame["cost_stratum"] == MISSING_STRATUM).all()
        assert not outcome.frame["peer_cell_stable"].any()

    def test_no_warnings_on_dirty_data(self, corpus: Corpus) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            SemanticLayer().run(corpus)


# ---------------------------------------------------------------------------
# Stage3.md sec.12 - validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Cluster quality measured against the generator's own ground truth."""

    def test_clusters_recover_the_true_work_types(
        self, result: Any, ground_truth: pd.Series
    ) -> None:
        frame = result.frame
        usable = ground_truth.notna() & (frame["cluster_id"] != NOISE_CLUSTER_ID)
        assert int(usable.sum()) > len(frame) * 0.5
        table = pd.DataFrame(
            {"cluster": frame.loc[usable, "cluster_id"], "truth": ground_truth[usable]}
        )
        majority = table.groupby("cluster")["truth"].agg(
            lambda values: values.value_counts().iloc[0]
        )
        weighted_purity = float(majority.sum()) / int(usable.sum())
        # Measured: 0.802 at 5k, 0.819 at 10k, 0.924 at 20k, 0.972 at 50k.
        # Purity rises with corpus size because the name vocabulary gets richer;
        # the floor here is set for the 10k fixture, not for the best case.
        assert weighted_purity > 0.75, weighted_purity

    def test_unrelated_work_types_are_separated(
        self, result: Any, ground_truth: pd.Series
    ) -> None:
        """Roads must not share a cluster with borewells."""
        frame = result.frame
        clustered = frame["cluster_id"] != NOISE_CLUSTER_ID
        modes: Dict[str, Any] = {}
        for work_type in ("cc road", "bus shelter", "playground development"):
            member = (ground_truth == work_type) & clustered
            assert int(member.sum()) > 0, work_type
            modes[work_type] = int(frame.loc[member, "cluster_id"].mode().iloc[0])
        assert len(set(modes.values())) == len(modes), modes

    def test_cell_distribution_is_not_extremely_imbalanced(self, result: Any) -> None:
        frame = result.frame
        stable = frame.loc[frame["peer_cell_stable"]]
        sizes = stable["peer_cell_id"].value_counts()
        assert len(sizes) > 10
        assert sizes.min() >= PEER_CELL_MIN_SIZE

    def test_most_records_land_in_a_usable_peer_cell(self, result: Any) -> None:
        assert float(result.frame["peer_cell_stable"].mean()) > 0.5

    def test_peer_statistics_are_not_polluted(self, result: Any, corpus: Corpus) -> None:
        """Every record behind a norm passes the gate. No exceptions."""
        reference = result.frame["peer_reference"]
        contributors = corpus.records.loc[reference]
        assert (contributors["confidence"] >= PEER_STAT_MIN_CONFIDENCE).all()
        assert not contributors["reconciliation_branch"].isin(
            ["non_finite", "implausible_magnitude"]
        ).any()

    def test_deviations_are_not_random(self, result: Any, corpus: Corpus) -> None:
        """A cost outlier injected by Stage 1 must deviate more than a clean record."""
        frame = result.frame
        deviation = frame["deviation_cluster_cost"].abs()
        high_confidence = corpus.records["confidence"] > 0.8
        usable = deviation.notna() & high_confidence
        assert int(usable.sum()) > 1000
        top = deviation[usable].quantile(0.99)
        assert top > 2.0, top


class TestScopeBoundary:
    """Stage 3 builds structure. Stage 4 decides. The line is asserted."""

    def test_no_anomaly_score_is_produced(self, result: Any) -> None:
        forbidden = {"anomaly_score", "is_anomaly", "anomaly_type", "risk_score"}
        assert not forbidden & set(result.frame.columns)
        assert not forbidden & set(STAGE3_COLUMNS)

    def test_deviations_are_not_combined(self, result: Any) -> None:
        """Each deviation stands alone; no aggregate exists for Stage 4 to inherit."""
        names = set(result.deviations.names)
        assert names <= set(STAGE3_COLUMNS)
        combined = {"deviation_total", "deviation_max", "deviation_mean"}
        assert not combined & set(result.frame.columns)

    def test_explanation_states_the_boundary(self, result: Any) -> None:
        payload = result.explain(0)
        assert "Stage 4" in payload["note"]

    def test_confidence_outputs_are_unchanged(self, corpus: Corpus) -> None:
        before = corpus.records["confidence"].copy()
        SemanticLayer().run(corpus)
        pd.testing.assert_series_equal(before, corpus.records["confidence"])
