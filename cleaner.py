"""
PARAKH Sfrom pathlib import Path

# Base directory for all project data — determined at runtime from __file__
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


tage 1 — Cleaning and Normalization

Cleans and normalizes records without silently imputing missing values.
Rules:
- Standardize text by lowercasing and trimming
- Normalize vendor names using basic deduplication (remove punctuation, collapse whitespace)
- Convert dates to consistent format (ISO 8601 YYYY-MM-DD)
- Do NOT impute missing values silently
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional


def _standardize_text(value: Optional[str]) -> Optional[str]:
    """Lowercase, trim, collapse internal whitespace."""
    if value is None or value == "":
        return value
    trimmed = value.strip()
    collapsed = re.sub(r"\s+", " ", trimmed)
    return collapsed.lower()


def _normalize_vendor_name(value: Optional[str]) -> Optional[str]:
    """Basic vendor name deduplication.

    Removes punctuation, collapses whitespace, lowercases.
    Does NOT perform fuzzy matching — that is a downstream task.
    """
    if value is None or value == "":
        return value
    # Remove punctuation (keep alphanumeric and whitespace only)
    no_punct = re.sub(r"[^\w\s]", "", str(value))
    # Collapse whitespace
    collapsed = re.sub(r"\s+", " ", no_punct).strip()
    return collapsed.lower()


def _normalize_date(value: Optional[str]) -> Optional[datetime]:
    """Convert date string to datetime object (ISO format YYYY-MM-DD).

    Attempts multiple common formats. Returns None if truly unresolvable.
    Does NOT silently impute — if unparseable, returns None and warning is emitted externally.
    """
    if value is None or value == "":
        return None

    value = value.strip()

    # Try ISO format first
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        pass

    # Try common formats
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d %b %Y",
        "%b %d %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S.%f",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue

    # Unparseable — return None (caller decides how to handle)
    return None


def clean_record(
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
) -> dict:
    """Clean and normalize a single record.

    Returns a dict with normalized values. Missing values remain None;
    they are NOT imputed.
    """

    # Standardize text fields
    work_name_norm = _standardize_text(work_name)
    district_norm = _standardize_text(district)
    state_norm = _standardize_text(state)
    work_category_norm = _standardize_text(work_category)

    # Normalize vendor name
    vendor_name_norm = _normalize_vendor_name(vendor_name)

    # Normalize dates
    date_sanction_dt = _normalize_date(date_sanction)
    date_start_dt = _normalize_date(date_start)
    date_completion_dt = _normalize_date(date_completion)

    # Numeric fields — ensure types; if already float/int, keep as-is
    # If they came as strings, attempt conversion; if unparseable, keep as-is (validator will catch)
    sanction_amount_f = float(sanction_amount) if sanction_amount is not None else None
    amount_released_f = float(amount_released) if amount_released is not None else None
    amount_utilized_f = float(amount_utilized) if amount_utilized is not None else None

    # Build cleaned record
    clean = {
        "work_id": work_id.strip() if work_id else work_id,
        "work_name": work_name_norm,
        "district": district_norm,
        "state": state_norm,
        "vendor_name": vendor_name_norm,
        "sanction_amount": sanction_amount_f,
        "amount_released": amount_released_f,
        "amount_utilized": amount_utilized_f,
        "date_sanction": date_sanction_dt,
        "date_start": date_start_dt,
        "date_completion": date_completion_dt,
        "work_category": work_category_norm,
    }

    return clean