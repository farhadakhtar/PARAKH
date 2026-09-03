import copy
import math
from collections import Counter
from datetime import datetime

import pytest

from stage1 import (
    CleanRecord,
    Cleaner,
    Corpus,
    Validator,
    build_corpus,
    generate_data,
    run_pipeline,
)


@pytest.fixture()
def valid_raw_record():
    return {
        "work_id": "work-test-001",
        "work_name": "Construction of Road at Ward 1",
        "district": "Patna",
        "state": "Bihar",
        "sanction_amount": 1000000.0,
        "amount_spent": 950000.0,
        "date_proposal": "2024-01-01",
        "date_approval": "2024-02-01",
        "date_completion": "2024-08-01",
        "implementing_agency": "Public Works Department",
        "vendor_name": "Sharma Constructions Pvt Ltd",
        "status": "completed",
        "work_type": "road",
        "record_category": "normal",
    }


# ---------------------------------------------------------------------------
# Data generation tests
# ---------------------------------------------------------------------------


def test_generate_data_exact_size_distribution_and_unique_work_ids():
    records = generate_data()

    assert len(records) == 10_000
    categories = Counter(record["record_category"] for record in records)
    assert categories == {"normal": 7000, "noisy": 2000, "anomalous": 1000}

    work_ids = [record["work_id"] for record in records]
    assert len(work_ids) == len(set(work_ids)) == 10_000
    assert all(work_id.startswith("work-") for work_id in work_ids)


def test_generate_data_is_deterministic():
    assert generate_data(seed=42)[:25] == generate_data(seed=42)[:25]
    assert generate_data(seed=42) != generate_data(seed=43)


def test_generate_data_rejects_non_stage1_record_count():
    with pytest.raises(ValueError, match="exactly 10,000"):
        generate_data(n=9999)


def test_generate_data_injects_required_anomaly_patterns():
    records = generate_data()
    anomalous = [record for record in records if record["record_category"] == "anomalous"]
    normal = [record for record in records if record["record_category"] == "normal"]

    normal_median = _median(record["sanction_amount"] for record in normal)
    anomalous_median = _median(record["sanction_amount"] for record in anomalous)
    assert anomalous_median > normal_median * 2.0, "anomalous costs should be visibly inflated"

    duplicate_names = [name for name, count in Counter(record["work_name"] for record in records).items() if count >= 50]
    assert duplicate_names, "duplicate/near-duplicate work names must be present"

    proposal_bursts = Counter(
        (record["district"], str(record["date_proposal"])[5:7]) for record in anomalous
    )
    assert proposal_bursts[("North Delhi", "02")] >= 100
    assert proposal_bursts[("Lucknow", "11")] >= 100
    assert proposal_bursts[("Patna", "03")] >= 100

    patna_anomalous_vendors = Counter(
        record["vendor_name"] for record in anomalous if record["district"] == "Patna"
    )
    dominant_share = sum(
        patna_anomalous_vendors[vendor]
        for vendor in ("Singh Infrastructure Ltd", "Patel Enterprises")
    ) / sum(patna_anomalous_vendors.values())
    assert dominant_share >= 0.40, "dominant vendors must create concentration signal"


def test_generated_numeric_values_are_finite_after_cleaning():
    cleaner = Cleaner()
    cleaned = [cleaner.clean(record) for record in generate_data()]
    for record in cleaned:
        for value in (record.sanction_amount, record.amount_spent):
            if value is not None:
                assert math.isfinite(value)


# ---------------------------------------------------------------------------
# Validator and schema constraint tests
# ---------------------------------------------------------------------------


def test_validator_accepts_valid_record(valid_raw_record):
    result = Validator().validate(valid_raw_record)

    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []


def test_validator_detects_missing_fields(valid_raw_record):
    record = dict(valid_raw_record)
    record["vendor_name"] = "N/A"
    record.pop("work_name")

    result = Validator().validate(record)

    assert result.is_valid is False
    assert "missing field: vendor_name" in result.errors
    assert "missing field: work_name" in result.errors


@pytest.mark.parametrize(
    "field,value,expected_error",
    [
        ("sanction_amount", -1, "invalid amount: sanction_amount must be finite and > 0"),
        ("sanction_amount", 0, "invalid amount: sanction_amount must be finite and > 0"),
        ("amount_spent", -1, "invalid amount: amount_spent must be finite and > 0"),
        ("amount_spent", 0, "invalid amount: amount_spent must be finite and > 0"),
        ("sanction_amount", float("inf"), "invalid amount: sanction_amount"),
    ],
)
def test_validator_rejects_invalid_amounts(valid_raw_record, field, value, expected_error):
    record = dict(valid_raw_record)
    record[field] = value

    result = Validator().validate(record)

    assert result.is_valid is False
    assert any(expected_error in error for error in result.errors)


def test_validator_enforces_financial_constraints(valid_raw_record):
    warning_record = dict(valid_raw_record, sanction_amount=1000, amount_spent=1100)
    warning_result = Validator().validate(warning_record)
    assert warning_result.is_valid is True
    assert "financial warning: amount_spent exceeds sanction_amount" in warning_result.warnings

    error_record = dict(valid_raw_record, sanction_amount=1000, amount_spent=1300)
    error_result = Validator().validate(error_record)
    assert error_result.is_valid is False
    assert "financial inconsistency: amount_spent exceeds sanction_amount by >25%" in error_result.errors


def test_validator_enforces_date_order_and_scheme_start(valid_raw_record):
    record = dict(
        valid_raw_record,
        date_proposal="1992-12-31",
        date_approval="2024-01-01",
        date_completion="2023-12-01",
    )

    result = Validator().validate(record)

    assert result.is_valid is False
    assert "date inconsistency: date_proposal before scheme start" in result.errors
    assert "date inconsistency: date_approval after date_completion" in result.errors


def test_validator_rejects_unparseable_dates_and_invalid_status(valid_raw_record):
    record = dict(valid_raw_record, date_approval="not-a-date", status="done")

    result = Validator().validate(record)

    assert result.is_valid is False
    assert "invalid date: date_approval" in result.errors
    assert "invalid status: must be proposed, approved, or completed" in result.errors


def test_validator_does_not_modify_input_record(valid_raw_record):
    record = copy.deepcopy(valid_raw_record)
    before = copy.deepcopy(record)

    Validator().validate(record)

    assert record == before


# ---------------------------------------------------------------------------
# Cleaner tests
# ---------------------------------------------------------------------------


def test_cleaner_normalizes_text_vendor_amounts_and_dates(valid_raw_record):
    dirty = dict(
        valid_raw_record,
        work_name="  Construction   OF Road  ",
        vendor_name="  SHARMA CONSTRUCTIONS PVT LTD  ",
        sanction_amount="₹1,000,000.50",
        date_approval="01/02/2024",
    )

    cleaned = Cleaner().clean(dirty)

    assert cleaned.work_name == "construction of road"
    assert cleaned.vendor_name == "sharma"
    assert cleaned.sanction_amount == 1_000_000.50
    assert cleaned.date_approval == datetime(2024, 2, 1)
    assert cleaned.to_dict()["date_approval"] == "2024-02-01"


def test_cleaner_does_not_silently_impute_missing_values(valid_raw_record):
    dirty = dict(
        valid_raw_record,
        vendor_name="",
        amount_spent="unknown",
        date_completion="0000-00-00",
    )

    cleaned = Cleaner().clean(dirty)

    assert cleaned.vendor_name is None
    assert cleaned.amount_spent is None
    assert cleaned.date_completion is None
    assert cleaned.work_id == "work-test-001", "non-missing fields should remain available"


# ---------------------------------------------------------------------------
# Corpus output and ingestion tests
# ---------------------------------------------------------------------------


def test_build_corpus_summary_statistics_are_correct(valid_raw_record):
    raw_records = [
        dict(valid_raw_record, work_id="dup", vendor_name="Vendor A", record_category="normal"),
        dict(valid_raw_record, work_id="dup", vendor_name="", record_category="noisy"),
        dict(valid_raw_record, work_id="unique", sanction_amount=-10, record_category="anomalous"),
    ]
    validator = Validator()
    cleaner = Cleaner()
    validations = [validator.validate(record) for record in raw_records]
    cleaned = [cleaner.clean(record) for record in raw_records]

    corpus = build_corpus(cleaned, validations, raw_records)

    assert len(corpus.records) == 3
    assert corpus.validation_summary["total_records"] == 3
    assert corpus.validation_summary["valid_records"] == 1
    assert corpus.validation_summary["invalid_records"] == 2
    assert corpus.validation_summary["invalid_pct"] == pytest.approx(66.6667, abs=0.0001)
    assert corpus.validation_summary["missing_fields_pct"]["vendor_name"] == pytest.approx(33.3333, abs=0.0001)
    assert corpus.validation_summary["category_distribution"] == {
        "normal": 1,
        "noisy": 1,
        "anomalous": 1,
    }
    assert corpus.validation_summary["duplicate_work_id_count"] == 1
    assert corpus.validation_summary["duplicate_work_ids"] == ["dup"]


def test_corpus_handles_empty_and_single_record_datasets(valid_raw_record):
    empty = Corpus.from_dataframe([])
    assert empty.records == []
    assert empty.validation_summary["total_records"] == 0
    assert all(value == 0.0 for value in empty.missing_report().values())

    single = Corpus.from_dataframe([valid_raw_record])
    assert len(single.records) == 1
    assert single.validation_summary["valid_records"] == 1
    assert single.head(1)[0]["work_id"] == "work-test-001"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_run_pipeline_end_to_end_outputs_connected_stage1_corpus():
    corpus = run_pipeline()

    assert len(corpus.records) == 10_000
    assert corpus.validation_summary["total_records"] == 10_000
    assert corpus.validation_summary["category_distribution"] == {
        "normal": 7000,
        "noisy": 2000,
        "anomalous": 1000,
    }
    assert corpus.validation_summary["invalid_records"] > 0
    assert corpus.validation_summary["duplicate_work_id_count"] == 0
    assert set(corpus.head(1)[0]).issuperset(
        {
            "work_id",
            "work_name",
            "district",
            "state",
            "sanction_amount",
            "amount_spent",
            "date_proposal",
            "date_approval",
            "date_completion",
            "implementing_agency",
            "vendor_name",
            "status",
        }
    )

    for record in corpus.records:
        for value in record.to_dict(iso_dates=False).values():
            if isinstance(value, float):
                assert math.isfinite(value)


def _median(values):
    ordered = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
