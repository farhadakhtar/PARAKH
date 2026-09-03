"""
PARAKH Stage 1 — Schema Definition

Strict schema for public fund records with validation rules that
RECORD violations (never drop data).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class RecordSchema:
    """Schema definition for a PARAKH work record."""

    work_id: str
    work_name: str
    district: str
    state: str
    vendor_name: Optional[str]
    sanction_amount: float
    amount_released: Optional[float]
    amount_utilized: Optional[float]
    date_sanction: datetime
    date_start: Optional[datetime]
    date_completion: Optional[datetime]
    work_category: str


def check_sanction_amount_greater_than_zero(value: float) -> bool:
    """Sanction amount must be greater than 0."""
    return value > 0


def check_amount_released_not_exceed_sanction(
    released: float | None, sanctioned: float
) -> bool:
    """If amount_released is present, it must be <= sanction_amount."""
    if released is None:
        return True
    return 0 < released <= sanctioned


def check_amount_utilized_not_exceed_released(
    utilized: float | None, released: float | None
) -> bool:
    """If amount_utilized is present, it must be <= amount_released."""
    if utilized is None or released is None:
        return True
    return 0 < utilized <= released


def check_date_ordering(
    sanction: datetime, start: datetime | None, completion: datetime | None
) -> bool:
    """Dates must follow: sanction <= start <= completion when present."""
    if start is not None and start < sanction:
        return False
    if completion is not None and completion < start:
        return False
    if completion is not None and start is not None and completion < start:
        return False
    return True