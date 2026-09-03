"""
PARAKH Stage 1 — Data Ingestion & Schema Layer

System-understanding plan:
- A record represents one public-works fund transaction/work item: identity, work text,
  geography, money sanctioned/spent, lifecycle dates, agency, vendor, and status.
- Stage 2 requires all fields for completeness; date_proposal/date_approval/date_completion
  for temporal coherence; sanction_amount and amount_spent for reconciliation.
- Stage 3 requires work_name, implementing_agency, district, sanction_amount, and approval
  timing for semantic clustering, cost strata, and duplicate scoring.
- Stage 4 requires sanction_amount, amount_spent, district, vendor_name, date_approval,
  and work identity/text for cost outliers, HHI, bursts, and duplicate signals.
- Assumptions: timestamps are scheme-era dates (>= 1993) when valid; monetary values are
  non-negative INR-like floats; vendor identity is name-based only and normalized text is
  a weak proxy (no beneficial ownership); missing/placeholder values are preserved as None.
- Critical constraint: Stage 1 never silently imputes, never mutates during validation,
  keeps unique work_id, emits normalized typed records plus validation evidence.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
for _directory in (DATA_DIR, OUTPUTS_DIR, LOGS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOGS_DIR / "stage1.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

PLACEHOLDERS = {"", "n/a", "na", "null", "none", "unknown", "0000-00-00", "-", "--"}
SCHEME_START = datetime(1993, 1, 1)

REQUIRED_FIELDS: Tuple[str, ...] = (
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
)

SCHEMA: Dict[str, type] = {
    "work_id": str,
    "work_name": str,
    "district": str,
    "state": str,
    "sanction_amount": float,
    "amount_spent": float,
    "date_proposal": datetime,
    "date_approval": datetime,
    "date_completion": datetime,
    "implementing_agency": str,
    "vendor_name": str,
    "status": str,
    "work_type": str,
    "record_category": str,
}


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidatedRecord:
    raw_record: Dict[str, Any]
    validation: ValidationResult


@dataclass(frozen=True)
class CleanRecord:
    work_id: Optional[str]
    work_name: Optional[str]
    district: Optional[str]
    state: Optional[str]
    sanction_amount: Optional[float]
    amount_spent: Optional[float]
    date_proposal: Optional[datetime]
    date_approval: Optional[datetime]
    date_completion: Optional[datetime]
    implementing_agency: Optional[str]
    vendor_name: Optional[str]
    status: Optional[str]
    work_type: Optional[str] = None
    record_category: Optional[str] = None

    def to_dict(self, iso_dates: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if iso_dates:
            for key in ("date_proposal", "date_approval", "date_completion"):
                if isinstance(data[key], datetime):
                    data[key] = data[key].date().isoformat()
        return data


@dataclass
class Corpus:
    records: List[CleanRecord]
    validation_summary: Dict[str, Any]
    schema: Dict[str, type] = field(default_factory=lambda: dict(SCHEMA))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dataframe(cls, df: Any) -> "Corpus":
        if hasattr(df, "to_dict"):
            records = df.to_dict(orient="records")
        else:
            records = list(df)
        validator = Validator()
        cleaner = Cleaner()
        validated = [validator.validate(r) for r in records]
        cleaned = [cleaner.clean(r) for r in records]
        return build_corpus(cleaned, validated, records)

    @classmethod
    def from_csv(cls, path: str | Path) -> "Corpus":
        safe_path = _ensure_project_path(path)
        rows: List[Dict[str, Any]] = []
        try:
            with safe_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            logging.exception("CSV ingestion failed: %s", safe_path)
            raise ValueError(f"Could not ingest CSV {safe_path}: {exc}") from exc
        return cls.from_dataframe(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "Corpus":
        safe_path = _ensure_project_path(path)
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:
            raise RuntimeError("Parquet ingestion requires pandas with a parquet engine") from exc
        return cls.from_dataframe(pd.read_parquet(safe_path))

    def summary(self) -> Dict[str, Any]:
        return {
            "total_records": len(self.records),
            "schema_fields": list(self.schema.keys()),
            "validation_summary": self.validation_summary,
            "metadata": self.metadata,
        }

    def head(self, n: int = 5) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.records[:n]]

    def missing_report(self) -> Dict[str, float]:
        if not self.records:
            return {field_name: 0.0 for field_name in REQUIRED_FIELDS}
        report: Dict[str, float] = {}
        for field_name in REQUIRED_FIELDS:
            missing = sum(1 for record in self.records if getattr(record, field_name) is None)
            report[field_name] = round(100.0 * missing / len(self.records), 4)
        return report

    def describe(self) -> Dict[str, Any]:
        amounts = [r.sanction_amount for r in self.records if _is_finite_number(r.sanction_amount)]
        spent = [r.amount_spent for r in self.records if _is_finite_number(r.amount_spent)]
        return {
            "sanction_amount": _numeric_description(amounts),
            "amount_spent": _numeric_description(spent),
            "status_distribution": dict(Counter(r.status for r in self.records)),
            "district_distribution": dict(Counter(r.district for r in self.records)),
        }

    def save(self) -> None:
        data_path = DATA_DIR / "stage1_clean_records.jsonl"
        summary_path = OUTPUTS_DIR / "stage1_validation_summary.json"
        with data_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(self.validation_summary, handle, indent=2, ensure_ascii=False)


class Validator:
    def validate(self, record: Dict[str, Any]) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        for field_name in REQUIRED_FIELDS:
            if field_name not in record or _is_missing(record.get(field_name)):
                errors.append(f"missing field: {field_name}")

        sanction = _parse_amount(record.get("sanction_amount"))
        spent = _parse_amount(record.get("amount_spent"))
        if sanction is None:
            errors.append("invalid amount: sanction_amount")
        elif sanction < 0 or not math.isfinite(sanction):
            errors.append("invalid amount: sanction_amount must be finite and >= 0")
        if spent is None:
            errors.append("invalid amount: amount_spent")
        elif spent < 0 or not math.isfinite(spent):
            errors.append("invalid amount: amount_spent must be finite and >= 0")
        if sanction is not None and spent is not None:
            if spent > sanction * 1.25:
                errors.append("financial inconsistency: amount_spent exceeds sanction_amount by >25%")
            elif spent > sanction:
                warnings.append("financial warning: amount_spent exceeds sanction_amount")

        dates = {name: _parse_datetime(record.get(name)) for name in ("date_proposal", "date_approval", "date_completion")}
        for name, value in dates.items():
            if value is None:
                errors.append(f"invalid date: {name}")
            elif value < SCHEME_START:
                errors.append(f"date inconsistency: {name} before scheme start")

        proposal, approval, completion = dates["date_proposal"], dates["date_approval"], dates["date_completion"]
        if proposal and approval and proposal > approval:
            errors.append("date inconsistency: date_proposal after date_approval")
        if approval and completion and approval > completion:
            errors.append("date inconsistency: date_approval after date_completion")

        status = _normalize_text_value(record.get("status"))
        if status is not None and status not in {"proposed", "approved", "completed"}:
            errors.append("invalid status: must be proposed, approved, or completed")

        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)


class Cleaner:
    VENDOR_SUFFIXES: ClassVar[Tuple[str, ...]] = (
        "private limited", "pvt ltd", "pvt. ltd.", "ltd", "limited", "contractors",
        "contractor", "construction", "constructions", "enterprises", "enterprise",
    )

    def clean(self, record: Dict[str, Any]) -> CleanRecord:
        return CleanRecord(
            work_id=_normalize_text_value(record.get("work_id"), keep_case=False),
            work_name=_normalize_text_value(record.get("work_name")),
            district=_normalize_text_value(record.get("district")),
            state=_normalize_text_value(record.get("state")),
            sanction_amount=_parse_amount(record.get("sanction_amount")),
            amount_spent=_parse_amount(record.get("amount_spent")),
            date_proposal=_parse_datetime(record.get("date_proposal")),
            date_approval=_parse_datetime(record.get("date_approval")),
            date_completion=_parse_datetime(record.get("date_completion")),
            implementing_agency=_normalize_text_value(record.get("implementing_agency")),
            vendor_name=self._normalize_vendor(record.get("vendor_name")),
            status=_normalize_text_value(record.get("status")),
            work_type=_normalize_text_value(record.get("work_type")),
            record_category=_normalize_text_value(record.get("record_category")),
        )

    def _normalize_vendor(self, value: Any) -> Optional[str]:
        text = _normalize_text_value(value)
        if text is None:
            return None
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        for suffix in self.VENDOR_SUFFIXES:
            text = re.sub(rf"\b{re.escape(suffix)}\b", "", text)
        return re.sub(r"\s+", " ", text).strip() or None


def generate_data(n: int = 10_000, seed: int = 42) -> List[Dict[str, Any]]:
    if n != 10_000:
        raise ValueError("Stage 1 synthetic generator is configured to generate exactly 10,000 records")
    rng = random.Random(seed)

    districts = ["North Delhi", "South Delhi", "Patna", "Gaya", "Lucknow", "Kanpur", "Jaipur", "Udaipur", "Pune", "Nagpur"]
    states = {
        "North Delhi": "Delhi", "South Delhi": "Delhi", "Patna": "Bihar", "Gaya": "Bihar",
        "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh", "Jaipur": "Rajasthan",
        "Udaipur": "Rajasthan", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    }
    district_weights = [0.16, 0.13, 0.12, 0.06, 0.15, 0.10, 0.11, 0.05, 0.08, 0.04]
    work_types = ["road", "school", "drainage", "water supply", "community hall", "health centre", "street light", "bridge"]
    work_weights = [0.30, 0.18, 0.16, 0.12, 0.10, 0.06, 0.05, 0.03]
    agencies = ["Public Works Department", "District Rural Development Agency", "Municipal Council", "Panchayat Samiti", "Urban Development Authority"]
    agency_weights = [0.35, 0.22, 0.20, 0.15, 0.08]
    vendors = [
        "Sharma Constructions Pvt Ltd", "Kumar Contractors", "Singh Infrastructure Ltd", "Patel Enterprises",
        "Gupta Builders", "Rao Civil Works", "Ali Construction", "Verma Projects", "Das Engineering", "Mishra Works",
        "Yadav Construction", "Khan Contractors", "Joshi Infrastructure", "Mehta Buildcon", "Nair Enterprises",
        "Iyer Civil Projects", "Reddy Builders", "Bose Construction", "Chauhan Works", "Agarwal Contractors",
    ]
    dominant_vendors = {
        "North Delhi": ["Sharma Constructions Pvt Ltd", "Kumar Contractors"],
        "Patna": ["Singh Infrastructure Ltd", "Patel Enterprises"],
        "Lucknow": ["Gupta Builders", "Rao Civil Works"],
        "Jaipur": ["Ali Construction", "Verma Projects"],
    }
    status_values = ["proposed", "approved", "completed"]
    status_weights = [0.10, 0.25, 0.65]
    categories = ["normal"] * 7000 + ["noisy"] * 2000 + ["anomalous"] * 1000
    rng.shuffle(categories)

    records: List[Dict[str, Any]] = []
    near_duplicate_bases = [
        "construction of cc road near market ward 4",
        "development of drainage line near school ward 7",
        "repair of community hall in central block",
        "installation of street lights on main road",
    ]
    burst_district_months = [("Patna", 3), ("Lucknow", 11), ("North Delhi", 2)]

    for i, category in enumerate(categories, start=1):
        district = _weighted_choice(rng, districts, district_weights)
        state = states[district]
        work_type = _weighted_choice(rng, work_types, work_weights)
        agency = _weighted_choice(rng, agencies, agency_weights)
        vendor_pool = vendors
        if category == "anomalous" and district in dominant_vendors and rng.random() < 0.70:
            vendor_pool = dominant_vendors[district]
        vendor = rng.choice(vendor_pool)

        sanction_amount = round(rng.lognormvariate(math.log(850_000), 0.85), 2)
        sanction_amount = max(25_000.0, min(sanction_amount, 25_000_000.0))
        amount_spent = round(sanction_amount * rng.uniform(0.72, 1.02), 2)

        year = rng.randint(2018, 2024)
        month = rng.randint(1, 12)
        if category == "anomalous" and rng.random() < 0.35:
            district, month = rng.choice(burst_district_months)
            state = states[district]
        proposal = datetime(year, month, rng.randint(1, 25))
        approval = proposal + timedelta(days=rng.randint(7, 120))
        completion = approval + timedelta(days=rng.randint(30, 540))

        locality = rng.choice(["ward 1", "ward 2", "ward 4", "block a", "main road", "near school", "market area", "village centre"])
        work_name = f"Construction of {work_type} at {locality} in {district}"
        if category == "anomalous" and rng.random() < 0.45:
            base = rng.choice(near_duplicate_bases)
            work_name = base if rng.random() < 0.50 else f"{base} phase {rng.choice(['i', 'ii', '2'])}"
        status = _weighted_choice(rng, status_values, status_weights)

        if category == "anomalous" and rng.random() < 0.55:
            sanction_amount = round(sanction_amount * rng.uniform(3.0, 10.0), 2)
            amount_spent = round(sanction_amount * rng.uniform(0.85, 1.15), 2)
        if category == "noisy":
            if rng.random() < 0.08:
                approval = proposal - timedelta(days=rng.randint(1, 90))
            if rng.random() < 0.06:
                completion = approval - timedelta(days=rng.randint(1, 120))
            if rng.random() < 0.08:
                amount_spent = round(sanction_amount * rng.uniform(1.26, 1.80), 2)
            if rng.random() < 0.08:
                work_name = rng.choice(near_duplicate_bases)

        record: Dict[str, Any] = {
            "work_id": f"work-{i:05d}",
            "work_name": work_name,
            "district": district,
            "state": state,
            "sanction_amount": sanction_amount,
            "amount_spent": amount_spent,
            "date_proposal": proposal.date().isoformat(),
            "date_approval": approval.date().isoformat(),
            "date_completion": completion.date().isoformat(),
            "implementing_agency": agency,
            "vendor_name": vendor,
            "status": status,
            "work_type": work_type,
            "record_category": category,
        }
        if category == "noisy":
            _inject_noise(record, rng)
        records.append(record)

    return records


def build_corpus(cleaned: List[CleanRecord], validated: List[ValidationResult], raw: Optional[List[Dict[str, Any]]] = None) -> Corpus:
    raw = raw or [r.to_dict(iso_dates=False) for r in cleaned]
    total = len(cleaned)
    invalid = sum(1 for result in validated if not result.is_valid)
    missing_counts: Dict[str, int] = {field_name: 0 for field_name in REQUIRED_FIELDS}
    for record in raw:
        for field_name in REQUIRED_FIELDS:
            if field_name not in record or _is_missing(record.get(field_name)):
                missing_counts[field_name] += 1
    error_counts = Counter(error for result in validated for error in result.errors)
    warning_counts = Counter(warning for result in validated for warning in result.warnings)
    summary = {
        "total_records": total,
        "valid_records": total - invalid,
        "invalid_records": invalid,
        "invalid_pct": round((100.0 * invalid / total) if total else 0.0, 4),
        "missing_fields_pct": {k: round((100.0 * v / total) if total else 0.0, 4) for k, v in missing_counts.items()},
        "category_distribution": dict(Counter(record.record_category for record in cleaned)),
        "status_distribution": dict(Counter(record.status for record in cleaned)),
        "error_counts": dict(error_counts),
        "warning_counts": dict(warning_counts),
    }
    return Corpus(records=cleaned, validation_summary=summary, metadata={"stage": 1, "deterministic_seed": 42})


def run_pipeline() -> Corpus:
    raw = generate_data()
    validator = Validator()
    cleaner = Cleaner()
    validated = [validator.validate(r) for r in raw]
    cleaned = [cleaner.clean(r) for r in raw]
    corpus = build_corpus(cleaned, validated, raw)
    _run_tests(raw, validated, corpus)
    corpus.save()
    return corpus


def _run_tests(raw: List[Dict[str, Any]], validated: List[ValidationResult], corpus: Corpus) -> None:
    assert len(raw) == 10_000, "total records must equal 10,000"
    work_ids = [record.get("work_id") for record in raw]
    assert len(work_ids) == len(set(work_ids)), "all work_id values must be unique"
    bad_record = {
        "work_id": "bad-1", "work_name": "bad", "district": "x", "state": "x",
        "sanction_amount": -1, "amount_spent": 100, "date_proposal": "2024-05-01",
        "date_approval": "2024-04-01", "date_completion": "0000-00-00",
        "implementing_agency": "x", "vendor_name": "x", "status": "completed",
    }
    bad_result = Validator().validate(bad_record)
    assert not bad_result.is_valid and bad_result.errors, "validator must catch bad data"
    assert any(not result.is_valid for result in validated), "synthetic data must include invalid records"
    for record in corpus.records:
        for value in record.to_dict(iso_dates=False).values():
            if isinstance(value, float):
                assert math.isfinite(value), "no NaN or inf values allowed"


def _inject_noise(record: Dict[str, Any], rng: random.Random) -> None:
    candidate_fields = [
        "work_name", "district", "sanction_amount", "amount_spent", "date_approval",
        "date_completion", "implementing_agency", "vendor_name",
    ]
    if rng.random() < 0.70:
        field_name = rng.choice(candidate_fields)
        record[field_name] = rng.choice(["N/A", "unknown", "", "null", "0000-00-00"])
    if rng.random() < 0.25 and isinstance(record.get("sanction_amount"), (int, float)):
        record["sanction_amount"] = f"₹{record['sanction_amount']:,.2f}"
    if rng.random() < 0.15:
        record["vendor_name"] = f"  {record['vendor_name'].upper()}  " if record.get("vendor_name") else record.get("vendor_name")


def _weighted_choice(rng: random.Random, items: List[str], weights: List[float]) -> str:
    return rng.choices(items, weights=weights, k=1)[0]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, str) and value.strip().lower() in PLACEHOLDERS:
        return True
    return False


def _normalize_text_value(value: Any, keep_case: bool = False) -> Optional[str]:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not keep_case:
        text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text or None


def _parse_amount(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    text = str(value).strip()
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", ".", "-"}:
        return None
    try:
        parsed = float(text)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    for attr in ("to_pydatetime",):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)().replace(hour=0, minute=0, second=0, microsecond=0)
            except Exception:
                pass
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            continue
    return None


def _numeric_description(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(sum(values) / len(values), 4),
    }


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _ensure_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = BASE_DIR / resolved
    resolved = resolved.resolve()
    if BASE_DIR.resolve() not in resolved.parents and resolved != BASE_DIR.resolve():
        raise ValueError("External paths are not allowed; use project-local ./data, ./outputs, or ./logs")
    return resolved


if __name__ == "__main__":
    stage1_corpus = run_pipeline()
    print("Stage 1 Completed Successfully")
    print(json.dumps(stage1_corpus.validation_summary, indent=2, ensure_ascii=False))
    print(json.dumps(stage1_corpus.head(5), indent=2, ensure_ascii=False))
