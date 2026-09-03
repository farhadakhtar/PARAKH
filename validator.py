"""
PARAKH Stage 1 — Validation Engine

Validates records and returns ValidationResult with is_valid, errors, and warnings.
All violations are recorded; data is NEVER dropped.
"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of record validation."""

    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class Validator:
    """Validation engine for PARAKH records.

    Rules:
    - Missing required fields
    - Negative or zero amounts
    - Date inconsistencies
    - Financial inconsistencies
    - Never drops data; always returns result with violations recorded
    """

    # Fields that are required (must not be None/empty)
    REQUIRED_FIELDS = {
        "work_id",
        "work_name",
        "district",
        "state",
        "sanction_amount",
        "date_sanction",
    }

    def validate(
        self,
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
    ) -> ValidationResult:
        """Validate a single record and return ValidationResult."""

        errors: List[str] = []
        warnings: List[str] = []

        # 1. Check missing required fields
        for field_name, value in {
            "work_id": work_id,
            "work_name": work_name,
            "district": district,
            "state": state,
            "sanction_amount": sanction_amount,
            "date_sanction": date_sanction,
        }.items():
            if not value:
                errors.append(f"Missing required field: {field_name}")

        # 2. Check sanction_amount > 0
        if sanction_amount is not None and sanction_amount <= 0:
            errors.append(
                f"sanction_amount must be > 0, got {sanction_amount}"
            )

        # 3. Check amount_released <= sanction_amount if present
        if amount_released is not None:
            if amount_released > sanction_amount:
                errors.append(
                    f"amount_released ({amount_released}) must be <= sanction_amount ({sanction_amount})"
                )
            if amount_released < 0:
                warnings.append(
                    f"amount_released is negative: {amount_released}"
                )

        # 4. Check amount_utilized <= amount_released if both present
        if amount_utilized is not None:
            if amount_released is None:
                errors.append(
                    "amount_utilized present but amount_released missing; "
                    "cannot verify amount_utilized <= amount_released"
                )
            elif amount_utilized > amount_released:
                errors.append(
                    f"amount_utilized ({amount_utilized}) must be <= amount_released ({amount_released})"
                )
            if amount_utilized < 0:
                warnings.append(
                    f"amount_utilized is negative: {amount_utilized}"
                )

        # 5. Check date ordering: sanction <= start <= completion
        if date_start is not None and date_start < date_sanction:
            errors.append(
                f"date_start ({date_start}) must be >= date_sanction ({date_sanction})"
            )

        if date_completion is not None:
            if date_start is not None and date_completion < date_start:
                errors.append(
                    f"date_completion ({date_completion}) must be >= date_start ({date_start})"
                )
            elif date_start is None:
                if date_completion < date_sanction:
                    errors.append(
                        f"date_completion ({date_completion}) must be >= date_sanction ({date_sanction}) "
                        "(since date_start is missing)"
                    )

        # 6. Check work_id not empty
        if not work_id or not work_id.strip():
            errors.append("work_id must be a non-empty string")

        # 7. Check work_name not empty
        if not work_name or not work_name.strip():
            errors.append("work_name must be a non-empty string")

        # 8. Basic district sanity check
        if district is not None and district.strip() and len(district.strip()) < 2:
            warnings.append(f"District name very short: {district}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )