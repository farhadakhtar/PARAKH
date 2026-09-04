"""
PARAKH Stage 1 — Full Pipeline

Orchestrates: generate data → validate → clean → build corpus.

All operations are deterministic (fixed random seed).
Data is NEVER silently dropped; violations are recorded.
"""

from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timedelta

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fixed seed for deterministic output
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Sibling module imports (same directory)
from generator import generate_raw_records
from validator import Validator
from cleaner import clean_record
from corpus import CleanRecord, Corpus


def validate_record(
    validator: Validator,
    work_id: str,
    work_name: str,
    district: str,
    state: str,
    vendor_name,
    sanction_amount,
    amount_released,
    amount_utilized,
    date_sanction: datetime,
    date_start,
    date_completion,
    work_category: str,
) -> tuple:
    """Validate a single record.

    Returns (validation_result, errors_list, warnings_list).
    """
    result = validator.validate(
        work_id=work_id,
        work_name=work_name,
        district=district,
        state=state,
        vendor_name=vendor_name,
        sanction_amount=sanction_amount,
        amount_released=amount_released,
        amount_utilized=amount_utilized,
        date_sanction=date_sanction,
        date_start=date_start,
        date_completion=date_completion,
        work_category=work_category,
    )
    return result, result.errors, result.warnings


def build_clean_record_from_validated(
    raw_record: tuple,
    validation_result: object,
) -> CleanRecord:
    """Build a CleanRecord from raw data, applying cleaning rules.

    The record is stored AS-IS (including invalid values) — the pipeline
    does NOT silently impute or fix bad data. The ValidationResult carries
    the error/warning information.
    """
    (work_id, work_name, district, state, vendor_name,
     sanction_amount, amount_released, amount_utilized,
     date_sanction, date_start, date_completion, work_category) = raw_record

    cleaned = clean_record(
        work_id=work_id,
        work_name=work_name,
        district=district,
        state=state,
        vendor_name=vendor_name,
        sanction_amount=sanction_amount,
        amount_released=amount_released,
        amount_utilized=amount_utilized,
        date_sanction=date_sanction,
        date_start=date_start,
        date_completion=date_completion,
        work_category=work_category,
    )

    return CleanRecord(
        work_id=cleaned["work_id"],
        work_name=cleaned["work_name"],
        district=cleaned["district"],
        state=cleaned["state"],
        vendor_name=cleaned["vendor_name"],
        sanction_amount=cleaned["sanction_amount"],
        amount_released=cleaned["amount_released"],
        amount_utilized=cleaned["amount_utilized"],
        date_sanction=cleaned["date_sanction"],
        date_start=cleaned["date_start"],
        date_completion=cleaned["date_completion"],
        work_category=cleaned["work_category"],
    )


def pipeline(
    n_records: int = 10_000,
    seed: int = RANDOM_SEED,
) -> Corpus:
    """Full Stage 1 pipeline.

    Steps:
    1. Generate raw synthetic records (ensures distribution: 70% normal, 20% noisy, 10% anomalous)
    2. Validate each record, recording all violations
    3. Clean each record (standardize text, normalize vendor names, convert dates)
    4. Build Corpus with validation_summary tracking

    Returns a Corpus instance with all records (valid + invalid) stored,
    and summary statistics on validity, missing fields, and category distribution.
    """
    # Ensure reproducibility
    rng = random.Random(seed)

    # Step 1: Generate raw records
    raw_records = generate_raw_records(n_records, rng=rng)

    # Step 2: Validate each record
    validator = Validator()
    validation_results = []
    errors_across_records: list[list[str]] = []
    warnings_across_records: list[list[str]] = []

    for raw in raw_records:
        vr, errs, warns = validate_record(
            validator=validator,
            *raw,
        )
        validation_results.append(vr)
        errors_across_records.append(errs)
        warnings_across_records.append(warns)

    # Step 3: Clean each record (does NOT impute; normalizes what it can)
    clean_records = []
    for i, raw in enumerate(raw_records):
        # Reconstruct raw field names for cleaner
        (work_id, work_name, district, state, vendor_name,
         sanction_amount, amount_released, amount_utilized,
         date_sanction, date_start, date_completion, work_category) = raw

        cleaned = clean_record(
            work_id=work_id,
            work_name=work_name,
            district=district,
            state=state,
            vendor_name=vendor_name,
            sanction_amount=sanction_amount,
            amount_released=amount_released,
            amount_utilized=amount_utilized,
            date_sanction=date_sanction,
            date_start=date_start,
            date_completion=date_completion,
            work_category=work_category,
        )

        # Build CleanRecord; we track validity separately via validation_summary
        c_record = CleanRecord(
            work_id=cleaned["work_id"],
            work_name=cleaned["work_name"],
            district=cleaned["district"],
            state=cleaned["state"],
            vendor_name=cleaned["vendor_name"],
            sanction_amount=cleaned["sanction_amount"],
            amount_released=cleaned["amount_released"],
            amount_utilized=cleaned["amount_utilized"],
            date_sanction=cleaned["date_sanction"],
            date_start=cleaned["date_start"],
            date_completion=cleaned["date_completion"],
            work_category=cleaned["work_category"],
        )
        clean_records.append(c_record)

    # Step 4: Build validation summary
    total = n_records
    valid_count = sum(1 for vr in validation_results if vr.is_valid)
    invalid_count = total - valid_count

    all_errors: list[str] = []
    all_warnings: list[str] = []
    for errs in errors_across_records:
        all_errors.extend(errs)
    for warns in warnings_across_records:
        all_warnings.extend(warns)

    # Missing fields count: count None values across all clean records
    missing_field_count = 0
    for rec in clean_records:
        for attr in CleanRecord.__slots__:
            val = getattr(rec, attr, None)
            if val is None:
                missing_field_count += 1

    # Category distribution
    category_counts: dict[str, int] = Counter()
    for rec in clean_records:
        if rec.work_category:
            category_counts[rec.work_category] += 1

    validation_summary = {
        "total_records": total,
        "valid_records": valid_count,
        "invalid_records": invalid_count,
        "valid_percentage": (valid_count / total) * 100.0 if total else 0.0,
        "invalid_percentage": (invalid_count / total) * 100.0 if total else 0.0,
        "errors": all_errors[:20],  # first 20 for summary
        "warnings": all_warnings[:20],
        "missing_fields_count": missing_field_count,
        "missing_fields_percentage": (missing_field_count / (total * len(CleanRecord.__slots__))) * 100.0
        if total else 0.0,
        "category_distribution": dict(category_counts),
        "category_percentages": {
            cat: (count / total) * 100.0 for cat, count in category_counts.items()
        },
    }

    # Build Corpus — stores ALL records (valid + invalid)
    corpus = Corpus(clean_records=clean_records, validation_summary=validation_summary)

    return corpus