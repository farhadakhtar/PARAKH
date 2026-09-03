"""Confidence scoring primitives for PARAKH work records."""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Optional

from core_scoring.schema import WorkRecord

FIELD_WEIGHTS = {
    "sanctioned_amount": 3,
    "cost_estimate": 3,
    "sanction_date": 2,
    "completion_date": 2,
    "contractor_id": 2,
    "payment_id": 1,
    "utilization_certificate_status": 2,
    "uc_submission_date": 1,
    "location": 1,
}

TEMPORAL_KAPPA = 0.05
SOURCE_LAMBDA = 1.5


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _date_pair_score(earlier: Optional[date], later: Optional[date]) -> Optional[float]:
    if earlier is None or later is None:
        return None
    days_gap = (later - earlier).days
    if days_gap < 0:
        return _sigmoid(-TEMPORAL_KAPPA * abs(days_gap))
    return _sigmoid(TEMPORAL_KAPPA * days_gap)


def completeness(record: WorkRecord) -> float:
    """Weighted fraction of required fields present and well-formed."""
    numerator = 0.0
    denominator = float(sum(FIELD_WEIGHTS.values()))
    for field, weight in FIELD_WEIGHTS.items():
        if _present(getattr(record, field, None)):
            numerator += weight
    return numerator / denominator


def temporal_coherence(record: WorkRecord) -> float:
    """Product of causal date-ordering scores, excluding missing date pairs."""
    pairs = [
        (record.sanction_date, record.completion_date),
        (record.completion_date, record.uc_submission_date),
        (record.uc_submission_date, record.payment_release_date),
    ]
    product = 1.0
    for earlier, later in pairs:
        pair_score = _date_pair_score(earlier, later)
        if pair_score is not None:
            product *= pair_score
    return product


def source_agreement(record: WorkRecord) -> float:
    """Proxy source-agreement score based on provenance tags."""
    total = len(record.confidence_state)
    if total == 0:
        return 1.0
    uncertain = sum(
        1
        for tag in record.confidence_state.values()
        if tag in {"SELF_CERTIFIED", "INFERRED"}
    )
    # TODO: Replace with real cross-source reconciliation once a second source exists,
    # such as matching against payment ledger totals and independent UC records.
    return math.exp(-SOURCE_LAMBDA * (uncertain / total))


def confidence(record: WorkRecord) -> dict:
    completeness_score = completeness(record)
    temporal_score = temporal_coherence(record)
    source_score = source_agreement(record)
    return {
        "completeness": completeness_score,
        "temporal_coherence": temporal_score,
        "source_agreement": source_score,
        "confidence": completeness_score * temporal_score * source_score,
    }
