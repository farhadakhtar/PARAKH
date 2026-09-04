"""Stage 1 test suite.

Organised by the PRD's own structure:

* ``TestSyntheticGeneration``  - sec.3.1 / 3.2
* ``TestIngestion``            - sec.3.3
* ``TestSchema``               - sec.3.4
* ``TestValidation``           - sec.3.5
* ``TestCleaning``             - sec.3.6
* ``TestDefectPreservation``   - sec.11 (the key design principle)
* ``TestCorpus``               - sec.3.7 / 5.3
* ``TestEdgeCases``            - sec.7 (mandatory)
* ``TestPerformance``          - sec.4
* ``TestDeterminism``          - sec.4
* ``TestAcceptanceCriteria``   - sec.6
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.core.constants import (
    ALLOWED_STATUS,
    COST_OUTLIER_BAND,
    DATE_VIOLATION_BAND,
    DUPLICATE_NAME_BAND,
    FIELD_ORDER,
    IMPLAUSIBLE_AMOUNT_THRESHOLD,
    MISSING_RATE_BAND,
    PERFORMANCE_ROW_BUDGET,
    PERFORMANCE_SECONDS_BUDGET,
    RECOMMENDED_SIZE_BAND,
)
from src.stage1.cleaning import clean_frame
from src.stage1.corpus import Corpus
from src.stage1.data_generator import (
    GenerationConfig,
    generate_dataset,
    generate_with_ledger,
    indian_group,
    observe_frame,
    save_dataset,
)
from src.stage1.ingestion import (
    IngestionError,
    read_csv,
    read_dataframe,
    write_csv,
    write_parquet,
)
from src.stage1.schema import SCHEMA, NullReason, Record, SchemaError, null_reason_column
from src.stage1.validation import IssueCode, Severity, ISSUE_SEVERITY, issue_code_of, validate

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

#: A single clean, fully-populated, internally consistent record.
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


def make_frame(*rows: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Build a small object-dtype frame from overrides on :data:`BASE_ROW`.

    Args:
        *rows: One dict of field overrides per row. ``None`` yields a pristine
            base row.

    Returns:
        A schema-ordered DataFrame.
    """
    records: List[Dict[str, Any]] = []
    for index, overrides in enumerate(rows or [None]):
        row = dict(BASE_ROW)
        row["work_id"] = f"MPL-KA-2019-{index + 1:06d}"
        if overrides:
            row.update(overrides)
        records.append(row)
    return pd.DataFrame(records, columns=list(FIELD_ORDER)).astype("object")


def corpus_of(*rows: Optional[Dict[str, Any]]) -> Corpus:
    """Build a corpus directly from override dicts."""
    return Corpus.from_dataframe(make_frame(*rows))


def issue_codes(corpus: Corpus, row: int = 0) -> List[str]:
    """Bare issue codes carried by one row."""
    return [issue_code_of(item) for item in corpus.records["issues"].iloc[row]]


def reason_of(corpus: Corpus, field: str, row: int = 0) -> str:
    """The null reason recorded for one cell."""
    return str(corpus.records[null_reason_column(field)].iloc[row])


@pytest.fixture(scope="module")
def generated() -> Any:
    """A full-size generated dataset, shared across the module."""
    return generate_with_ledger(n=RECOMMENDED_SIZE_BAND[0], seed=42)


@pytest.fixture(scope="module")
def full_corpus(generated: Any) -> Corpus:
    """A corpus built from the full-size generated dataset."""
    return Corpus.from_dataframe(generated.frame)


# ---------------------------------------------------------------------------
# sec.3.1 / 3.2 - synthetic generation
# ---------------------------------------------------------------------------


class TestSyntheticGeneration:
    """Stage1.md sec.3.1 and sec.3.2."""

    def test_generates_requested_row_count(self) -> None:
        assert len(generate_dataset(n=500, seed=1)) == 500

    def test_emits_exactly_the_schema_columns_in_order(self) -> None:
        assert tuple(generate_dataset(n=50, seed=1).columns) == FIELD_ORDER

    def test_full_size_generation_is_within_prd_band(self, generated: Any) -> None:
        low, high = RECOMMENDED_SIZE_BAND
        assert low <= len(generated.frame) <= high

    def test_same_seed_is_byte_identical(self) -> None:
        first = generate_dataset(n=1200, seed=42)
        second = generate_dataset(n=1200, seed=42)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seed_produces_different_data(self) -> None:
        first = generate_dataset(n=1200, seed=42)
        second = generate_dataset(n=1200, seed=43)
        assert not first.equals(second)

    def test_ledger_is_deterministic(self) -> None:
        first = generate_with_ledger(n=1200, seed=42).ledger
        second = generate_with_ledger(n=1200, seed=42).ledger
        assert first.channel_counts == second.channel_counts
        assert first.rows_with("date_order:") == second.rows_with("date_order:")

    def test_no_global_random_state_is_touched(self) -> None:
        """Determinism must not depend on the caller's global RNG."""
        np.random.seed(0)
        first = generate_dataset(n=300, seed=42)
        np.random.seed(999)
        _ = np.random.random(50)
        second = generate_dataset(n=300, seed=42)
        pd.testing.assert_frame_equal(first, second)

    # --- noise budget --------------------------------------------------

    def test_missing_rate_within_prd_band(self, generated: Any) -> None:
        rate = generated.observed["missing_cell_rate"]
        assert MISSING_RATE_BAND[0] <= rate <= MISSING_RATE_BAND[1], rate

    def test_date_violation_rate_within_prd_band(self, generated: Any) -> None:
        observed = generated.observed["date_violation_rate"]
        injected = generated.observed["injected_date_violation_rate"]
        assert DATE_VIOLATION_BAND[0] <= observed <= DATE_VIOLATION_BAND[1], observed
        assert DATE_VIOLATION_BAND[0] <= injected <= DATE_VIOLATION_BAND[1], injected

    def test_cost_outlier_rate_within_prd_band(self, generated: Any) -> None:
        rate = generated.observed["cost_outlier_rate"]
        assert COST_OUTLIER_BAND[0] <= rate <= COST_OUTLIER_BAND[1], rate

    def test_duplicate_name_rate_within_prd_band(self, generated: Any) -> None:
        rate = generated.observed["duplicate_work_name_rate"]
        assert DUPLICATE_NAME_BAND[0] <= rate <= DUPLICATE_NAME_BAND[1], rate

    def test_every_field_has_some_missingness_except_the_key(
        self, generated: Any
    ) -> None:
        per_field = generated.observed["missing_pct_by_field"]
        assert per_field["work_id"] == 0.0
        for name in FIELD_ORDER:
            if name != "work_id":
                assert per_field[name] > 0.0, name

    def test_placeholder_tokens_are_present_in_output(self, generated: Any) -> None:
        flat = generated.frame.astype(str).to_numpy().ravel()
        assert "N/A" in flat
        assert "unknown" in flat
        assert "0000-00-00" in flat

    def test_unparseable_tokens_are_present_in_output(self, generated: Any) -> None:
        dates = generated.frame["date_proposal"].astype(str)
        assert dates.isin(["not a date", "31/02/2020", "pending"]).any()

    def test_duplicate_work_ids_are_injected(self, generated: Any) -> None:
        assert generated.frame["work_id"].duplicated().any()

    def test_near_duplicates_are_not_exact_duplicates(self, generated: Any) -> None:
        exact = generated.observed["exact_duplicate_work_name_rate"]
        total = generated.observed["duplicate_work_name_rate"]
        assert 0 < exact < total, (exact, total)

    # --- ledger hygiene -------------------------------------------------

    def test_ledger_is_not_a_column_of_the_dataset(self, generated: Any) -> None:
        """Leaking ground truth into the corpus would hand Stages 2-7 the answers."""
        assert tuple(generated.frame.columns) == FIELD_ORDER
        for column in generated.frame.columns:
            assert "defect" not in column and "ledger" not in column

    def test_ledger_rows_are_valid_positional_indices(self, generated: Any) -> None:
        payload = generated.ledger.to_dict(generated.config, generated.observed)
        n = len(generated.frame)
        for key in payload["defects_by_row"]:
            assert 0 <= int(key) < n

    def test_ledger_channel_counts_match_configured_rates(self, generated: Any) -> None:
        n = len(generated.frame)
        outliers = generated.ledger.channel_counts.get(
            "cost_outlier:high", 0
        ) + generated.ledger.channel_counts.get("cost_outlier:low", 0)
        assert outliers == round(n * generated.config.cost_outlier_rate)

    def test_ledger_serialises_to_json(self, generated: Any, tmp_path: Path) -> None:
        paths = save_dataset(generated, data_dir=tmp_path)
        payload = json.loads(paths["ledger"].read_text(encoding="utf-8"))
        assert payload["config"]["seed"] == 42
        assert payload["n_defective_rows"] > 0
        assert paths["csv"].exists()

    # --- size guards ----------------------------------------------------

    def test_zero_rows_is_allowed(self) -> None:
        frame = generate_dataset(n=0, seed=1)
        assert len(frame) == 0
        assert tuple(frame.columns) == FIELD_ORDER

    def test_single_row_is_allowed(self) -> None:
        assert len(generate_dataset(n=1, seed=1)) == 1

    def test_negative_row_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            generate_dataset(n=-5, seed=1)

    def test_out_of_band_size_warns_but_succeeds(self, caplog: Any) -> None:
        with caplog.at_level("WARNING", logger="parakh.data_generator"):
            frame = generate_dataset(n=100, seed=1)
        assert len(frame) == 100
        assert any("recommended" in record.message for record in caplog.records)

    def test_noise_channels_can_be_disabled(self) -> None:
        """A clean corpus must be producible, for downstream fixtures."""
        config = GenerationConfig(
            n=200,
            seed=3,
            missing_rates={name: 0.0 for name in FIELD_ORDER},
            date_order_violation_rate=0.0,
            cost_outlier_rate=0.0,
            duplicate_name_rate=0.0,
            duplicate_id_rate=0.0,
            negative_amount_rate=0.0,
            extreme_value_rate=0.0,
            pre_scheme_date_rate=0.0,
            recoverable_format_rate=0.0,
            unparseable_format_rate=0.0,
        )
        corpus = Corpus.from_dataframe(generate_dataset(config=config))
        assert corpus.validation_report.invalid_records == 0
        assert corpus.validation_report.missing_cell_rate_pct == 0.0

    def test_indian_grouping_helper(self) -> None:
        assert indian_group(125000) == "1,25,000"
        assert indian_group(1000) == "1,000"
        assert indian_group(999) == "999"
        assert indian_group(-125000.5) == "-1,25,000.50"


# ---------------------------------------------------------------------------
# sec.3.3 - ingestion
# ---------------------------------------------------------------------------


class TestIngestion:
    """Stage1.md sec.3.3."""

    def test_from_dataframe(self) -> None:
        assert len(corpus_of(None, None)) == 2

    def test_csv_round_trip_matches_dataframe_path(self, tmp_path: Path) -> None:
        frame = generate_dataset(n=800, seed=11)
        path = write_csv(frame, tmp_path / "d.csv")
        from_frame = Corpus.from_dataframe(frame)
        from_file = Corpus.from_csv(path)
        for name in FIELD_ORDER:
            pd.testing.assert_series_equal(
                from_frame.records[name], from_file.records[name]
            )
        assert (
            from_frame.validation_report.to_dict()
            == from_file.validation_report.to_dict()
        )

    def test_parquet_round_trip_matches_dataframe_path(self, tmp_path: Path) -> None:
        frame = generate_dataset(n=800, seed=11)
        path = write_parquet(frame, tmp_path / "d.parquet")
        from_frame = Corpus.from_dataframe(frame)
        from_file = Corpus.from_parquet(path)
        for name in FIELD_ORDER:
            pd.testing.assert_series_equal(
                from_frame.records[name], from_file.records[name]
            )

    def test_csv_reader_does_not_swallow_placeholder_tokens(
        self, tmp_path: Path
    ) -> None:
        """Regression: pandas' default NA handling erases "N/A" vs empty.

        If ``keep_default_na`` were left on, a deliberately-typed "N/A" would
        arrive as NaN and be misreported as MISSING rather than PLACEHOLDER,
        destroying the distinction Stage 2 depends on.
        """
        path = tmp_path / "p.csv"
        path.write_text(
            "work_id,work_name,district,state,sanction_amount,amount_spent,"
            "date_proposal,date_approval,date_completion,implementing_agency,"
            "vendor_name,status\n"
            "W1,Road,Mysuru,Karnataka,100,90,2019-01-01,2019-02-01,2019-03-01,"
            "Agency,N/A,completed\n"
            "W2,Road,Mysuru,Karnataka,100,90,2019-01-01,2019-02-01,2019-03-01,"
            "Agency,,completed\n",
            encoding="utf-8",
        )
        corpus = Corpus.from_csv(path)
        assert reason_of(corpus, "vendor_name", 0) == NullReason.PLACEHOLDER.value
        assert reason_of(corpus, "vendor_name", 1) == NullReason.MISSING.value

    def test_leading_zero_identifiers_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "z.csv"
        header = ",".join(FIELD_ORDER)
        path.write_text(
            f"{header}\n00123,Road,Mysuru,Karnataka,100,90,2019-01-01,"
            "2019-02-01,2019-03-01,Agency,Vendor,completed\n",
            encoding="utf-8",
        )
        assert Corpus.from_csv(path).records["work_id"].iloc[0] == "00123"

    def test_malformed_rows_are_quarantined_not_fatal(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        header = ",".join(FIELD_ORDER)
        good = "W1,Road,Mysuru,Karnataka,100,90,2019-01-01,2019-02-01,2019-03-01,A,V,completed"
        path.write_text(
            f"{header}\n{good}\nW2,too,few,fields\n{good}\n"
            "W3,a,b,c,d,e,f,g,h,i,j,k,l,m,n,EXTRA\n",
            encoding="utf-8",
        )
        result = read_csv(path)
        assert result.n_rows >= 2
        assert result.n_errors >= 1
        assert all(error["action"] == "skipped" for error in result.errors)

    def test_missing_required_column_is_a_hard_error(self) -> None:
        frame = make_frame(None).drop(columns=["work_id"])
        with pytest.raises(SchemaError, match="required column"):
            read_dataframe(frame)

    def test_missing_optional_column_is_synthesised_as_null(self) -> None:
        frame = make_frame(None).drop(columns=["vendor_name"])
        corpus = Corpus.from_dataframe(frame)
        assert corpus.records["vendor_name"].isna().all()
        assert "vendor_name" in corpus.metadata.synthesised_columns

    def test_extra_columns_are_dropped(self) -> None:
        frame = make_frame(None)
        frame["irrelevant"] = "x"
        corpus = Corpus.from_dataframe(frame)
        assert "irrelevant" not in corpus.records.columns
        assert corpus.metadata.dropped_columns == ["irrelevant"]

    def test_column_labels_are_normalised(self) -> None:
        frame = make_frame(None).rename(
            columns={"work_id": " Work ID ", "work_name": "Work-Name"}
        )
        corpus = Corpus.from_dataframe(frame)
        assert "work_id" in corpus.records.columns
        assert "work_name" in corpus.records.columns

    def test_colliding_column_labels_are_rejected(self) -> None:
        frame = make_frame(None)
        frame["Work ID"] = "dup"
        with pytest.raises(SchemaError, match="collide"):
            read_dataframe(frame)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="not found"):
            Corpus.from_csv(tmp_path / "nope.csv")

    def test_non_dataframe_input_raises(self) -> None:
        with pytest.raises(IngestionError):
            read_dataframe([1, 2, 3])  # type: ignore[arg-type]

    def test_empty_csv_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        assert len(Corpus.from_csv(path)) == 0

    def test_header_only_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "header.csv"
        path.write_text(",".join(FIELD_ORDER) + "\n", encoding="utf-8")
        assert len(Corpus.from_csv(path)) == 0


# ---------------------------------------------------------------------------
# sec.3.4 - schema
# ---------------------------------------------------------------------------


class TestSchema:
    """Stage1.md sec.3.4."""

    def test_python_types_match_the_prd_literal(self) -> None:
        import datetime

        assert SCHEMA.python_types() == {
            "work_id": str,
            "work_name": str,
            "district": str,
            "state": str,
            "sanction_amount": float,
            "amount_spent": float,
            "date_proposal": datetime.datetime,
            "date_approval": datetime.datetime,
            "date_completion": datetime.datetime,
            "implementing_agency": str,
            "vendor_name": str,
            "status": str,
        }

    def test_field_order_is_stable(self) -> None:
        assert SCHEMA.names == FIELD_ORDER

    def test_only_the_key_is_non_nullable(self) -> None:
        non_nullable = [spec.name for spec in SCHEMA if not spec.nullable]
        assert non_nullable == ["work_id"]

    def test_unknown_field_lookup_raises(self) -> None:
        with pytest.raises(SchemaError):
            SCHEMA.spec("not_a_field")

    def test_cleaned_columns_carry_the_declared_dtypes(self) -> None:
        corpus = corpus_of(None)
        for name, dtype in SCHEMA.pandas_dtypes().items():
            assert str(corpus.records[name].dtype) == dtype, name

    def test_schema_serialises(self) -> None:
        payload = SCHEMA.to_dict()
        assert len(payload["fields"]) == 12
        assert payload["version"] == SCHEMA.version


# ---------------------------------------------------------------------------
# sec.3.6 - cleaning
# ---------------------------------------------------------------------------


class TestCleaning:
    """Stage1.md sec.3.6."""

    def test_strings_are_lowercased_and_trimmed(self) -> None:
        corpus = corpus_of({"district": "  MYSURU  "})
        assert corpus.records["district"].iloc[0] == "mysuru"

    def test_internal_whitespace_is_collapsed(self) -> None:
        corpus = corpus_of({"work_name": "Repair   of    Road"})
        assert corpus.records["work_name"].iloc[0] == "repair of road"

    @pytest.mark.parametrize(
        "token", ["N/A", "n/a", " N/A ", "unknown", "NULL", "none", "-", "NIL"]
    )
    def test_placeholders_become_null_with_placeholder_reason(
        self, token: str
    ) -> None:
        corpus = corpus_of({"vendor_name": token})
        assert corpus.records["vendor_name"].iloc[0] is None
        assert reason_of(corpus, "vendor_name") == NullReason.PLACEHOLDER.value

    @pytest.mark.parametrize("blank", [None, "", "   ", np.nan])
    def test_blank_cells_get_the_missing_reason(self, blank: Any) -> None:
        corpus = corpus_of({"vendor_name": blank})
        assert corpus.records["vendor_name"].iloc[0] is None
        assert reason_of(corpus, "vendor_name") == NullReason.MISSING.value

    def test_zero_date_token_is_a_placeholder_not_an_unparseable_date(self) -> None:
        """The three null causes must stay distinguishable (sec.3.6 + Stage 2)."""
        corpus = corpus_of({"date_completion": "0000-00-00"})
        assert reason_of(corpus, "date_completion") == NullReason.PLACEHOLDER.value

    @pytest.mark.parametrize(
        "token", ["not a date", "31/02/2020", "2020-13-45", "pending"]
    )
    def test_garbage_dates_get_the_unparseable_reason(self, token: str) -> None:
        corpus = corpus_of({"date_completion": token})
        assert pd.isna(corpus.records["date_completion"].iloc[0])
        assert reason_of(corpus, "date_completion") == NullReason.UNPARSEABLE.value

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2019-03-01", "2019-03-01"),
            ("01-03-2019", "2019-03-01"),
            ("01/03/2019", "2019-03-01"),
            ("01 Mar 2019", "2019-03-01"),
            ("March 01, 2019", "2019-03-01"),
            ("2019/03/01", "2019-03-01"),
        ],
    )
    def test_date_formats_normalise_to_iso(self, raw: str, expected: str) -> None:
        corpus = corpus_of({"date_proposal": raw, "date_approval": "2020-01-01"})
        assert (
            corpus.records["date_proposal"].iloc[0].date().isoformat() == expected
        )

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1,25,000", 125000.0),
            ("Rs 1,25,000", 125000.0),
            ("Rs. 1,25,000.50", 125000.50),
            ("INR 850000", 850000.0),
            ("  850000  ", 850000.0),
            ("₹850000.00", 850000.0),
        ],
    )
    def test_numeric_symbols_and_separators_are_stripped(
        self, raw: str, expected: float
    ) -> None:
        corpus = corpus_of({"sanction_amount": raw})
        assert corpus.records["sanction_amount"].iloc[0] == pytest.approx(expected)

    @pytest.mark.parametrize("token", ["abcd", "to be decided", "as per estimate"])
    def test_garbage_numerics_get_the_unparseable_reason(self, token: str) -> None:
        corpus = corpus_of({"amount_spent": token})
        assert pd.isna(corpus.records["amount_spent"].iloc[0])
        assert reason_of(corpus, "amount_spent") == NullReason.UNPARSEABLE.value

    def test_overflowing_literal_becomes_infinity_not_null(self) -> None:
        """``1.2e400`` is a *present, parseable* value; nulling it would hide it."""
        corpus = corpus_of({"sanction_amount": "1.2e400"})
        assert np.isinf(corpus.records["sanction_amount"].iloc[0])
        assert reason_of(corpus, "sanction_amount") == NullReason.PRESENT.value

    def test_raw_snapshot_is_retained(self) -> None:
        corpus = corpus_of({"district": "  MYSURU  "})
        assert corpus.raw["district"].iloc[0] == "  MYSURU  "
        assert corpus.records["district"].iloc[0] == "mysuru"

    def test_clean_frame_rejects_an_unaligned_frame(self) -> None:
        with pytest.raises(ValueError, match="schema-aligned"):
            clean_frame(pd.DataFrame({"a": [1]}))

    def test_cleaning_reports_normalisation_activity(self) -> None:
        frame = make_frame({"district": "  MYSURU  "})
        result = clean_frame(frame)
        assert result.stats["normalization_changes"]["district"] == 1
        assert result.stats["rows_in"] == result.stats["rows_out"]


# ---------------------------------------------------------------------------
# sec.11 - the key design principle
# ---------------------------------------------------------------------------


class TestDefectPreservation:
    """Stage1.md sec.11: garbage in -> *controlled* garbage out.

    These are the tests that would catch a well-meaning "improvement" that
    quietly repairs the data and makes Stage 2 meaningless.
    """

    def test_out_of_order_dates_are_not_reordered(self) -> None:
        corpus = corpus_of(
            {"date_proposal": "2020-06-01", "date_approval": "2019-01-01"}
        )
        row = corpus.records.iloc[0]
        assert row["date_proposal"] > row["date_approval"]
        assert IssueCode.LOGICAL_DATE_ORDER.value in issue_codes(corpus)

    def test_missing_values_are_never_imputed(self) -> None:
        corpus = corpus_of({"sanction_amount": None, "vendor_name": None})
        assert pd.isna(corpus.records["sanction_amount"].iloc[0])
        assert corpus.records["vendor_name"].iloc[0] is None

    def test_outliers_are_never_clipped(self) -> None:
        corpus = corpus_of({"sanction_amount": 9.9e14})
        assert corpus.records["sanction_amount"].iloc[0] == pytest.approx(9.9e14)

    def test_negative_amounts_are_kept_and_flagged(self) -> None:
        corpus = corpus_of({"amount_spent": -500.0})
        assert corpus.records["amount_spent"].iloc[0] == -500.0
        assert IssueCode.VALUE_NEGATIVE.value in issue_codes(corpus)

    def test_no_rows_are_dropped_by_the_pipeline(self) -> None:
        frame = generate_dataset(n=900, seed=5)
        assert len(Corpus.from_dataframe(frame)) == len(frame)

    def test_row_order_is_preserved(self) -> None:
        frame = make_frame({"district": "A"}, {"district": "B"}, {"district": "C"})
        corpus = Corpus.from_dataframe(frame)
        assert corpus.records["district"].tolist() == ["a", "b", "c"]

    def test_duplicate_names_survive_cleaning(self) -> None:
        corpus = corpus_of({"work_name": "Same Road"}, {"work_name": "Same Road"})
        assert corpus.records["work_name"].nunique() == 1
        assert len(corpus) == 2


# ---------------------------------------------------------------------------
# sec.3.5 - validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Stage1.md sec.3.5."""

    def test_a_clean_record_is_valid_with_no_issues(self) -> None:
        corpus = corpus_of(None)
        assert bool(corpus.records["is_valid"].iloc[0])
        assert corpus.records["issues"].iloc[0] == ()

    def test_approval_before_proposal_is_invalid(self) -> None:
        corpus = corpus_of({"date_approval": "2018-01-01"})
        assert not bool(corpus.records["is_valid"].iloc[0])
        assert "logical_date_order:date_approval<date_proposal" in list(
            corpus.records["issues"].iloc[0]
        )

    def test_completion_before_approval_is_invalid(self) -> None:
        corpus = corpus_of({"date_completion": "2019-04-01"})
        assert "logical_date_order:date_completion<date_approval" in list(
            corpus.records["issues"].iloc[0]
        )

    def test_ordering_is_only_checked_when_both_dates_exist(self) -> None:
        """A missing date is a completeness defect, never an ordering defect."""
        corpus = corpus_of({"date_approval": None})
        assert IssueCode.LOGICAL_DATE_ORDER.value not in issue_codes(corpus)
        assert bool(corpus.records["is_valid"].iloc[0])
        assert corpus.validation_report.date_violations == 0.0

    def test_unparseable_date_does_not_create_a_phantom_ordering_violation(
        self,
    ) -> None:
        corpus = corpus_of({"date_approval": "not a date"})
        assert IssueCode.LOGICAL_DATE_ORDER.value not in issue_codes(corpus)
        assert IssueCode.TYPE_UNPARSEABLE.value in issue_codes(corpus)

    def test_equal_dates_are_not_a_violation(self) -> None:
        """The rule is ``<=``, so same-day milestones are legitimate."""
        corpus = corpus_of(
            {
                "date_proposal": "2019-03-01",
                "date_approval": "2019-03-01",
                "date_completion": "2019-03-01",
            }
        )
        assert bool(corpus.records["is_valid"].iloc[0])

    def test_negative_amount_is_invalid(self) -> None:
        corpus = corpus_of({"sanction_amount": -1.0})
        assert not bool(corpus.records["is_valid"].iloc[0])
        assert IssueCode.VALUE_NEGATIVE.value in issue_codes(corpus)

    def test_zero_amount_is_valid(self) -> None:
        corpus = corpus_of({"amount_spent": 0.0})
        assert IssueCode.VALUE_NEGATIVE.value not in issue_codes(corpus)

    def test_infinite_amount_is_flagged_non_finite(self) -> None:
        corpus = corpus_of({"sanction_amount": "1.2e400"})
        assert IssueCode.VALUE_NON_FINITE.value in issue_codes(corpus)
        assert corpus.validation_report.non_finite_amount_records == 1

    def test_implausible_magnitude_is_flagged(self) -> None:
        corpus = corpus_of({"sanction_amount": 1e300})
        assert IssueCode.VALUE_IMPLAUSIBLE_MAGNITUDE.value in issue_codes(corpus)
        assert corpus.validation_report.implausible_amount_records == 1

    def test_amount_just_below_the_threshold_is_accepted(self) -> None:
        corpus = corpus_of({"sanction_amount": IMPLAUSIBLE_AMOUNT_THRESHOLD - 1})
        assert IssueCode.VALUE_IMPLAUSIBLE_MAGNITUDE.value not in issue_codes(corpus)

    def test_pre_scheme_date_is_invalid(self) -> None:
        corpus = corpus_of(
            {
                "date_proposal": "1985-01-01",
                "date_approval": "1985-06-01",
                "date_completion": "1986-01-01",
            }
        )
        assert not bool(corpus.records["is_valid"].iloc[0])
        assert IssueCode.LOGICAL_DATE_BEFORE_SCHEME_START.value in issue_codes(corpus)
        assert corpus.validation_report.pre_scheme_date_records == 1

    def test_future_date_is_a_warning_not_an_error(self) -> None:
        corpus = corpus_of({"date_completion": "2030-01-01"})
        assert IssueCode.LOGICAL_DATE_IN_FUTURE.value in issue_codes(corpus)
        assert bool(corpus.records["is_valid"].iloc[0])

    def test_duplicate_key_invalidates_both_rows(self) -> None:
        frame = make_frame(None, None)
        frame.loc[1, "work_id"] = frame.loc[0, "work_id"]
        corpus = Corpus.from_dataframe(frame)
        assert corpus.validation_report.duplicate_key_records == 2
        assert not corpus.records["is_valid"].any()

    def test_missing_key_is_invalid(self) -> None:
        corpus = corpus_of({"work_id": None})
        assert not bool(corpus.records["is_valid"].iloc[0])
        assert IssueCode.SCHEMA_MISSING_KEY.value in issue_codes(corpus)

    def test_unknown_status_is_invalid(self) -> None:
        corpus = corpus_of({"status": "half done"})
        assert IssueCode.VALUE_UNKNOWN_STATUS.value in issue_codes(corpus)

    @pytest.mark.parametrize("status", ALLOWED_STATUS)
    def test_allowed_statuses_pass(self, status: str) -> None:
        assert IssueCode.VALUE_UNKNOWN_STATUS.value not in issue_codes(
            corpus_of({"status": status})
        )

    def test_missing_field_is_a_warning_and_does_not_invalidate(self) -> None:
        """Otherwise ~80% of a realistic corpus is "invalid" and the flag is useless."""
        corpus = corpus_of({"vendor_name": None, "district": None, "status": None})
        assert bool(corpus.records["is_valid"].iloc[0])
        assert IssueCode.COMPLETENESS_MISSING.value in issue_codes(corpus)

    def test_every_issue_code_has_a_declared_severity(self) -> None:
        for code in IssueCode:
            assert ISSUE_SEVERITY[code] in (Severity.ERROR, Severity.WARNING)

    # --- report shape ---------------------------------------------------

    def test_report_matches_the_prd_output_shape(self, full_corpus: Corpus) -> None:
        view = full_corpus.validation_report.prd_view()
        assert set(view) == {
            "total_records",
            "valid_records",
            "invalid_records",
            "missing_fields",
            "date_violations",
        }
        assert set(view["missing_fields"]) == set(FIELD_ORDER)

    def test_valid_and_invalid_counts_partition_the_corpus(
        self, full_corpus: Corpus
    ) -> None:
        report = full_corpus.validation_report
        assert report.valid_records + report.invalid_records == report.total_records
        assert report.total_records == len(full_corpus)

    def test_missing_percentages_match_the_null_counts(
        self, full_corpus: Corpus
    ) -> None:
        n = len(full_corpus)
        for name, pct in full_corpus.validation_report.missing_fields.items():
            observed = 100.0 * full_corpus.records[name].isna().sum() / n
            assert pct == pytest.approx(observed, abs=0.01), name

    def test_date_violation_percentage_matches_the_rows(
        self, full_corpus: Corpus
    ) -> None:
        rows = full_corpus.records_with_issue(IssueCode.LOGICAL_DATE_ORDER.value)
        expected = 100.0 * len(rows) / len(full_corpus)
        assert full_corpus.validation_report.date_violations == pytest.approx(
            expected, abs=0.01
        )

    def test_validate_preserves_row_count(self) -> None:
        frame = generate_dataset(n=400, seed=2)
        result = clean_frame(read_dataframe(frame).frame)
        outcome = validate(result)
        assert len(outcome.is_valid) == len(frame)
        assert outcome.n_valid + outcome.n_invalid == len(frame)


# ---------------------------------------------------------------------------
# sec.3.7 / 5.3 - corpus
# ---------------------------------------------------------------------------


class TestCorpus:
    """Stage1.md sec.3.7 and sec.5.3."""

    def test_corpus_retains_every_record(self, full_corpus: Corpus, generated: Any) -> None:
        assert len(full_corpus) == len(generated.frame)

    def test_valid_and_invalid_are_views_not_partitions(
        self, full_corpus: Corpus
    ) -> None:
        n_valid = len(full_corpus.valid_records)
        n_invalid = len(full_corpus.invalid_records)
        assert n_valid + n_invalid == len(full_corpus)
        assert n_invalid > 0
        # Taking the views must not shrink the corpus itself.
        assert len(full_corpus) == full_corpus.validation_report.total_records

    def test_head_defaults_and_bounds(self, full_corpus: Corpus) -> None:
        assert len(full_corpus.head()) == 5
        assert len(full_corpus.head(3)) == 3
        assert len(full_corpus.head(0)) == 0
        assert len(full_corpus.head(-4)) == 0
        assert len(full_corpus.head(10**9)) == len(full_corpus)

    def test_head_hides_diagnostics_by_default(self, full_corpus: Corpus) -> None:
        assert list(full_corpus.head(1).columns) == full_corpus.display_columns
        assert null_reason_column("state") in full_corpus.head(
            1, diagnostics=True
        ).columns

    def test_summary_is_json_serialisable(self, full_corpus: Corpus) -> None:
        payload = json.dumps(full_corpus.summary(), default=str)
        assert json.loads(payload)["n_records"] == len(full_corpus)

    def test_summary_has_no_wall_clock_field(self, full_corpus: Corpus) -> None:
        """A timestamp would break the "same input -> same output" contract."""
        blob = json.dumps(full_corpus.summary(), default=str).lower()
        for forbidden in ("ingested_at", "generated_at", "timestamp", "created_at"):
            assert forbidden not in blob

    def test_missing_report_columns_and_totals(self, full_corpus: Corpus) -> None:
        table = full_corpus.missing_report()
        assert set(table.columns) == {
            "n_present",
            "n_missing",
            "n_placeholder",
            "n_unparseable",
            "n_null_total",
            "pct_null",
            "pct_present",
        }
        assert len(table) == len(FIELD_ORDER)
        for name, row in table.iterrows():
            assert row["n_present"] + row["n_null_total"] == len(full_corpus), name

    def test_missing_report_as_dict(self, full_corpus: Corpus) -> None:
        payload = full_corpus.missing_report(as_dict=True)
        assert payload["total_records"] == len(full_corpus)
        assert set(payload["fields"]) == set(FIELD_ORDER)

    def test_describe_covers_all_fields(self, full_corpus: Corpus) -> None:
        table = full_corpus.describe()
        assert len(table) == len(FIELD_ORDER)
        assert table.loc["sanction_amount", "median"] > 0

    def test_records_with_issue_view(self, full_corpus: Corpus) -> None:
        rows = full_corpus.records_with_issue(IssueCode.LOGICAL_DATE_ORDER.value)
        assert len(rows) > 0
        assert not rows["is_valid"].any()

    def test_typed_records_use_none_not_nan(self) -> None:
        corpus = corpus_of({"sanction_amount": None, "date_completion": "N/A"})
        record = corpus.to_typed_records()[0]
        assert isinstance(record, Record)
        assert record.sanction_amount is None
        assert record.date_completion is None
        assert record.district == "mysuru"

    def test_typed_record_exposes_null_reasons(self) -> None:
        corpus = corpus_of({"vendor_name": "N/A", "district": None})
        record = corpus.to_typed_records()[0]
        assert record.null_reason("vendor_name") is NullReason.PLACEHOLDER
        assert record.null_reason("district") is NullReason.MISSING
        assert record.null_reason("state") is NullReason.PRESENT
        assert not record.is_field_valid("vendor_name")
        assert record.is_field_valid("state")

    def test_typed_records_are_immutable(self) -> None:
        record = corpus_of(None).to_typed_records()[0]
        with pytest.raises(Exception):
            record.district = "tampered"  # type: ignore[misc]

    def test_typed_record_count_matches_corpus(self) -> None:
        corpus = Corpus.from_dataframe(generate_dataset(n=200, seed=8))
        assert len(corpus.to_typed_records()) == len(corpus)

    def test_record_to_dict_round_trips(self) -> None:
        payload = corpus_of(None).to_typed_records()[0].to_dict()
        assert payload["date_proposal"] == "2019-03-01"
        assert payload["is_valid"] is True

    def test_repr_is_informative(self, full_corpus: Corpus) -> None:
        assert "Corpus" in repr(full_corpus)
        assert str(len(full_corpus)) in repr(full_corpus)

    def test_save_reports_writes_every_artefact(
        self, full_corpus: Corpus, tmp_path: Path
    ) -> None:
        written = full_corpus.save_reports(tmp_path)
        assert set(written) == {
            "summary",
            "validation_report",
            "missing_report",
            "metadata",
        }
        for path in written.values():
            assert path.exists()
            json.loads(path.read_text(encoding="utf-8"))

    def test_save_clean_csv_is_reingestable(self, tmp_path: Path) -> None:
        corpus = Corpus.from_dataframe(generate_dataset(n=300, seed=9))
        path = corpus.save_clean_csv(tmp_path / "clean.csv", diagnostics=False)
        assert len(Corpus.from_csv(path)) == len(corpus)


# ---------------------------------------------------------------------------
# sec.7 - mandatory edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Stage1.md sec.7. Every one of these is required to not crash."""

    def test_completely_empty_dataset(self) -> None:
        corpus = Corpus.from_dataframe(
            pd.DataFrame({name: pd.Series([], dtype="object") for name in FIELD_ORDER})
        )
        assert len(corpus) == 0
        report = corpus.validation_report
        assert report.total_records == 0
        assert report.valid_records == 0
        assert report.date_violations == 0.0
        assert all(pct == 0.0 for pct in report.missing_fields.values())
        assert corpus.head().empty
        assert len(corpus.describe()) == len(FIELD_ORDER)
        assert corpus.missing_report()["pct_null"].eq(0.0).all()
        assert corpus.summary()["n_records"] == 0
        assert corpus.to_typed_records() == []

    def test_single_record_dataset(self) -> None:
        corpus = corpus_of(None)
        assert len(corpus) == 1
        assert corpus.describe().loc["sanction_amount", "std"] == 0.0
        assert corpus.summary()["n_records"] == 1

    def test_all_null_column(self) -> None:
        frame = make_frame(None, None, None)
        frame["vendor_name"] = None
        corpus = Corpus.from_dataframe(frame)
        assert corpus.validation_report.missing_fields["vendor_name"] == 100.0
        assert pd.isna(corpus.describe().loc["vendor_name", "unique"])
        assert corpus.records["vendor_name"].isna().all()

    def test_every_column_null_except_the_key(self) -> None:
        frame = make_frame(None)
        for name in FIELD_ORDER:
            if name != "work_id":
                frame[name] = None
        corpus = Corpus.from_dataframe(frame)
        assert bool(corpus.records["is_valid"].iloc[0])
        assert len(issue_codes(corpus)) == len(FIELD_ORDER) - 1

    def test_extremely_large_values(self) -> None:
        corpus = corpus_of({"sanction_amount": 1e300, "amount_spent": 1e308})
        assert np.isfinite(corpus.records["sanction_amount"].iloc[0])
        assert corpus.validation_report.implausible_amount_records == 1
        stats = corpus.describe()
        assert int(stats.loc["sanction_amount", "n_extreme"]) == 1
        json.dumps(corpus.summary(), default=str)

    def test_invalid_date_formats_do_not_crash(self) -> None:
        corpus = corpus_of(
            {
                "date_proposal": "0000-00-00",
                "date_approval": "31/02/2020",
                "date_completion": "20200-01-01",
            }
        )
        assert corpus.records[list(SCHEMA.date_fields)].isna().all(axis=None)
        assert reason_of(corpus, "date_proposal") == NullReason.PLACEHOLDER.value
        assert reason_of(corpus, "date_approval") == NullReason.UNPARSEABLE.value
        assert reason_of(corpus, "date_completion") == NullReason.UNPARSEABLE.value

    def test_identical_records_have_no_variance(self) -> None:
        corpus = Corpus.from_dataframe(make_frame(*([None] * 10)))
        assert corpus.describe().loc["sanction_amount", "std"] == 0.0
        assert corpus.describe().loc["district", "unique"] == 1

    def test_unicode_and_symbols_survive(self) -> None:
        corpus = corpus_of(
            {
                "vendor_name": "Sri Vinayaka निर्माण & Co.",
                "sanction_amount": "₹ 12,34,567.89",
            }
        )
        assert "न" in corpus.records["vendor_name"].iloc[0]
        assert corpus.records["sanction_amount"].iloc[0] == pytest.approx(1234567.89)

    def test_very_long_text_field(self) -> None:
        corpus = corpus_of({"work_name": "road " * 2000})
        assert len(corpus.records["work_name"].iloc[0]) > 5000

    def test_observe_frame_on_empty_input(self) -> None:
        assert observe_frame(pd.DataFrame(columns=list(FIELD_ORDER))) == {"n": 0}


# ---------------------------------------------------------------------------
# sec.4 - non-functional
# ---------------------------------------------------------------------------


class TestPerformance:
    """Stage1.md sec.4: 50k rows in under 5 seconds."""

    def test_fifty_thousand_rows_within_budget(self) -> None:
        frame = generate_dataset(n=PERFORMANCE_ROW_BUDGET, seed=42)
        started = time.perf_counter()
        corpus = Corpus.from_dataframe(frame)
        elapsed = time.perf_counter() - started
        assert len(corpus) == PERFORMANCE_ROW_BUDGET
        assert elapsed < PERFORMANCE_SECONDS_BUDGET, (
            f"ingest+clean+validate took {elapsed:.2f}s, "
            f"budget is {PERFORMANCE_SECONDS_BUDGET}s"
        )


class TestDeterminism:
    """Stage1.md sec.4: same input -> same output, always."""

    def test_pipeline_reports_are_reproducible(self) -> None:
        first = Corpus.from_dataframe(generate_dataset(n=1500, seed=42))
        second = Corpus.from_dataframe(generate_dataset(n=1500, seed=42))
        assert first.summary() == second.summary()
        assert first.validation_report.to_dict() == second.validation_report.to_dict()
        pd.testing.assert_frame_equal(first.missing_report(), second.missing_report())

    def test_reingesting_the_same_file_is_stable(self, tmp_path: Path) -> None:
        path = write_csv(generate_dataset(n=600, seed=4), tmp_path / "d.csv")
        assert Corpus.from_csv(path).summary() == Corpus.from_csv(path).summary()


# ---------------------------------------------------------------------------
# sec.6 - acceptance criteria
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """One test per checkbox in Stage1.md sec.6."""

    def test_synthetic_dataset_generated_successfully(self, generated: Any) -> None:
        assert len(generated.frame) >= RECOMMENDED_SIZE_BAND[0]

    def test_data_loads_without_crashing(self, full_corpus: Corpus) -> None:
        assert len(full_corpus) > 0

    def test_schema_validation_works(self, full_corpus: Corpus) -> None:
        assert full_corpus.validation_report.schema_version == SCHEMA.version
        assert tuple(full_corpus.records.columns[: len(FIELD_ORDER)]) == FIELD_ORDER

    def test_invalid_records_are_detected(self, full_corpus: Corpus) -> None:
        report = full_corpus.validation_report
        assert report.invalid_records > 0
        assert report.duplicate_key_records > 0
        assert report.pre_scheme_date_records > 0
        assert report.negative_amount_records > 0
        assert report.implausible_amount_records > 0
        assert report.non_finite_amount_records > 0
        assert sum(report.unparseable_cells.values()) > 0

    def test_every_injected_channel_is_detected_on_real_generated_data(
        self, full_corpus: Corpus, generated: Any
    ) -> None:
        """Close the loop: what the generator injected, validation must find."""
        codes = {
            issue_code_of(item)
            for issues in full_corpus.records["issues"]
            for item in issues
        }
        for expected in (
            IssueCode.LOGICAL_DATE_ORDER,
            IssueCode.LOGICAL_DATE_BEFORE_SCHEME_START,
            IssueCode.TYPE_UNPARSEABLE,
            IssueCode.VALUE_NEGATIVE,
            IssueCode.VALUE_NON_FINITE,
            IssueCode.VALUE_IMPLAUSIBLE_MAGNITUDE,
            IssueCode.SCHEMA_DUPLICATE_KEY,
            IssueCode.COMPLETENESS_MISSING,
            IssueCode.COMPLETENESS_PLACEHOLDER,
        ):
            assert expected.value in codes, expected

    def test_injected_date_violations_are_found_where_still_visible(
        self, full_corpus: Corpus, generated: Any
    ) -> None:
        """Every surviving injected violation must be flagged - no false negatives.

        Rows whose dates were later blanked by the missing-value channel are
        excluded: the evidence is gone, so there is nothing to detect.
        """
        injected = set(generated.ledger.rows_with("date_order:"))
        flagged = set(
            full_corpus.records_with_issue(IssueCode.LOGICAL_DATE_ORDER.value).index
        )
        assert flagged.issubset(injected), "flagged a violation that was never injected"

        frame = full_corpus.records
        pair_of_label = {
            "date_order:approval_before_proposal": ("date_proposal", "date_approval"),
            "date_order:completion_before_approval": (
                "date_approval",
                "date_completion",
            ),
        }
        still_visible = set()
        for label, (earlier, later) in pair_of_label.items():
            for row in generated.ledger.rows_with(label):
                if pd.notna(frame[earlier].iloc[row]) and pd.notna(
                    frame[later].iloc[row]
                ):
                    still_visible.add(row)

        assert still_visible, "the masking channel hid every injected violation"
        assert still_visible == still_visible & flagged, (
            f"{len(still_visible - flagged)} injected violations were still "
            "visible in the data but not flagged"
        )

    def test_cleaning_pipeline_normalises_values(self, full_corpus: Corpus) -> None:
        text = full_corpus.records["district"].dropna()
        assert (text == text.str.lower()).all()
        assert not text.str.startswith(" ").any()

    def test_corpus_is_usable_downstream(self, full_corpus: Corpus) -> None:
        """Everything Stage 2 needs must be reachable from the corpus."""
        record = next(full_corpus.iter_records())
        assert hasattr(record, "sanction_amount")
        assert callable(record.is_field_valid)
        assert full_corpus.schema.date_fields
        assert full_corpus.validation_report.null_reason_counts
