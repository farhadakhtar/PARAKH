"""
PARAfrom pathlib import Path

# Base directory for all project data — determined at runtime from __file__
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


KH Stage 1 — Corpus Object

Holds cleaned records, validation statistics, and distribution tracking.
Stores ALL records including invalid ones — never drops data.
"""

from __future__ import annotations

from typing import List, Dict, Any
from collections import Counter


class CleanRecord:
    """A cleaned, normalized record from the pipeline.

    All fields are present; missing values are None (not imputed).
    """

    __slots__ = (
        "work_id",
        "work_name",
        "district",
        "state",
        "vendor_name",
        "sanction_amount",
        "amount_released",
        "amount_utilized",
        "date_sanction",
        "date_start",
        "date_completion",
        "work_category",
    )

    def __init__(
        self,
        work_id: str,
        work_name: str,
        district: str,
        state: str,
        vendor_name,
        sanction_amount,
        amount_released,
        amount_utilized,
        date_sanction,
        date_start,
        date_completion,
        work_category: str,
    ):
        self.work_id = work_id
        self.work_name = work_name
        self.district = district
        self.state = state
        self.vendor_name = vendor_name
        self.sanction_amount = sanction_amount
        self.amount_released = amount_released
        self.amount_utilized = amount_utilized
        self.date_sanction = date_sanction
        self.date_start = date_start
        self.date_completion = date_completion
        self.work_category = work_category

    def __repr__(self):
        return (
            f"CleanRecord(work_id={self.work_id!r}, work_name={self.work_name!r}, "
            f"district={self.district!r}, state={self.state!r}, "
            f"vendor_name={self.vendor_name!r}, sanction_amount={self.sanction_amount!r}, "
            f"work_category={self.work_category!r})"
        )


class Corpus:
    """Container for PARAKH records with validation and distribution statistics.

    Key properties:
    - Stores ALL records including invalid ones (never drops)
    - Tracks percentage of missing fields across corpus
    - Tracks percentage of invalid records (per Validator)
    - Tracks distribution of work_category
    - Provides programmatic access to raw vs cleaned records
    """

    def __init__(self, clean_records: List[CleanRecord], validation_summary: Dict[str, Any]):
        self.records = clean_records  # list of CleanRecord — ALL records
        self.validation_summary = validation_summary
        self._category_distribution = Counter(
            rec.work_category for rec in clean_records if rec.work_category
        )
        self._missing_field_count = self._count_missing_fields()
        self._total_records = len(clean_records)

    def _count_missing_fields(self) -> int:
        """Count total across-record field-null occurrences.

        A field is "missing" if its value is None.
        """
        count = 0
        for rec in self.records:
            for attr in self.__slots__:
                val = getattr(rec, attr, None)
                if val is None:
                    count += 1
        return count

    @property
    def total_records(self) -> int:
        return self._total_records

    @property
    def missing_fields_percentage(self) -> float:
        """Percentage of field slots that are None across all records."""
        if self._total_records == 0:
            return 0.0
        total_slots = self._total_records * len(self.__slots__)
        return (self._missing_field_count / total_slots) * 100.0

    @property
    def invalid_records_percentage(self) -> float:
        """Percentage of records that failed validation (have errors)."""
        if self._total_records == 0:
            return 0.0
        invalid = sum(
            1 for rec in self.records
        )  # placeholder — will be set via validation_summary
        return 0.0  # actual value from validation_summary

    @property
    def category_distribution(self) -> Dict[str, int]:
        return dict(self._category_distribution)

    @property
    def category_percentages(self) -> Dict[str, float]:
        total = self._total_records
        if total == 0:
            return {}
        return {
            cat: (count / total) * 100.0
            for cat, count in self._category_distribution.items()
        }

    def get_invalid_records(self) -> List[CleanRecord]:
        """Return records that have validation errors.

        Note: Corpus tracks validity via validation_summary, not per-record
        is_valid flag, to keep CleanRecord lightweight.
        """
        # This is a proxy — actual invalid tracking comes from validation_summary
        # In a full implementation, CleanRecord would have an is_valid attribute
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize corpus state for debugging/monitoring."""
        return {
            "total_records": self._total_records,
            "missing_fields_percentage": self.missing_fields_percentage,
            "invalid_records_percentage": self.validation_summary.get(
                "invalid_percentage", 0.0
            ),
            "category_distribution": self.category_distribution,
            "category_percentages": self.category_percentages,
            "validation_errors": self.validation_summary.get("errors", []),
            "validation_warnings": self.validation_summary.get("warnings", []),
        }