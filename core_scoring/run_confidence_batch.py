"""Run PARAKH confidence scoring for every work record in Postgres."""
from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import psycopg2
import psycopg2.extras

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core_scoring.confidence import confidence
from core_scoring.schema import WorkRecord

WORK_FIELDS = [
    "mp_id",
    "work_id",
    "location",
    "payment_id",
    "district_id",
    "work_status",
    "contractor_id",
    "cost_estimate",
    "sanction_date",
    "work_category",
    "completion_date",
    "constituency_id",
    "work_description",
    "sanctioned_amount",
    "uc_submission_date",
    "payment_release_date",
    "implementing_agency_id",
    "utilization_certificate_status",
]


def _json_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _load_records(cursor) -> List[WorkRecord]:
    cursor.execute("SELECT * FROM works")
    rows = cursor.fetchall()
    records: List[WorkRecord] = []
    for row in rows:
        payload = {}
        if "data" in row and row["data"] is not None:
            payload.update(_json_dict(row["data"]))
        for field in WORK_FIELDS:
            if field in row and row[field] is not None:
                payload[field] = row[field]
        if "description" in payload and "work_description" not in payload:
            payload["work_description"] = payload["description"]
        if "confidence_state" in row and row["confidence_state"] is not None:
            payload["confidence_state"] = _json_dict(row["confidence_state"])
        elif "confidence_state" in payload:
            payload["confidence_state"] = _json_dict(payload["confidence_state"])
        records.append(WorkRecord(**payload))
    return records


def _create_scores_table(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS confidence_scores (
            work_id TEXT PRIMARY KEY,
            completeness DOUBLE PRECISION NOT NULL,
            temporal_coherence DOUBLE PRECISION NOT NULL,
            source_agreement DOUBLE PRECISION NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            computed_at TIMESTAMPTZ NOT NULL
        )
        """
    )


def _write_scores(cursor, records: Iterable[WorkRecord]) -> List[float]:
    scores: List[float] = []
    computed_at = datetime.now(timezone.utc)
    for record in records:
        result = confidence(record)
        scores.append(result["confidence"])
        cursor.execute(
            """
            INSERT INTO confidence_scores (
                work_id, completeness, temporal_coherence, source_agreement,
                confidence, computed_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (work_id) DO UPDATE SET
                completeness = EXCLUDED.completeness,
                temporal_coherence = EXCLUDED.temporal_coherence,
                source_agreement = EXCLUDED.source_agreement,
                confidence = EXCLUDED.confidence,
                computed_at = EXCLUDED.computed_at
            """,
            (
                record.work_id,
                result["completeness"],
                result["temporal_coherence"],
                result["source_agreement"],
                result["confidence"],
                computed_at,
            ),
        )
    return scores


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is required")
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            records = _load_records(cursor)
            _create_scores_table(cursor)
            scores = _write_scores(cursor, records)
    mean_confidence = statistics.mean(scores) if scores else 0.0
    below_threshold = sum(1 for score in scores if score < 0.5)
    print(f"Scored {len(scores)} records")
    print(f"Mean confidence: {mean_confidence:.4f}")
    print(f"Records below 0.5 confidence: {below_threshold}")


if __name__ == "__main__":
    main()
