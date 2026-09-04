"""The :class:`Corpus` - Stage 1's central abstraction (Stage1.md sec.3.7).

A ``Corpus`` holds **every** record that was ingested. Nothing is filtered out.
``valid_records`` and ``invalid_records`` are *views* over the same rows, not
partitions that discard data.

That is not a stylistic choice. Stage 2 scores confidence precisely on the
records Stage 1 found defective - a record with an unparseable date is the one
whose ``C_temp`` must collapse to zero. Dropping invalid rows here would leave
those code paths unreachable and the REMEDIATE queue permanently empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

from src.core.constants import (
    AMOUNT_PRECISION,
    DEFAULT_HEAD_ROWS,
    FIELD_ORDER,
    IMPLAUSIBLE_AMOUNT_THRESHOLD,
    PERCENT_PRECISION,
)
from src.core.logger import get_logger
from src.stage1 import ingestion as ingestion_module
from src.stage1.cleaning import CleaningResult, clean_frame
from src.stage1.ingestion import IngestionResult
from src.stage1.schema import (
    SCHEMA,
    FieldType,
    NullReason,
    Record,
    Schema,
    null_reason_column,
)
from src.stage1.validation import (
    ValidationOutcome,
    ValidationReport,
    issue_code_of,
    validate,
)
from src.utils.helpers import safe_percentage, write_json

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

VALID_COLUMN = "is_valid"
ISSUES_COLUMN = "issues"


def _safe_round(value: float) -> float:
    """Round to the amount precision, passing ``inf``/``nan`` through intact.

    ``round(float("inf"), 2)`` is fine, but keeping the guard explicit
    documents that non-finite statistics are a real, expected outcome here
    rather than a bug: a corpus containing a 1e300 entry genuinely has an
    infinite variance in float64.
    """
    if not np.isfinite(value):
        return value
    return round(value, AMOUNT_PRECISION)


@dataclass(frozen=True)
class CorpusMetadata:
    """Provenance of a corpus.

    Deliberately carries **no wall-clock timestamp**: the determinism contract
    says the same input must always produce the same output, and an
    ``ingested_at`` field would break that for every report that embeds it.
    """

    source: str
    schema_version: str
    n_records: int
    n_fields: int
    source_path: Optional[str] = None
    renamed_columns: Dict[str, str] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    synthesised_columns: List[str] = field(default_factory=list)
    ingestion_errors: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "source": self.source,
            "source_path": self.source_path,
            "schema_version": self.schema_version,
            "n_records": self.n_records,
            "n_fields": self.n_fields,
            "renamed_columns": dict(self.renamed_columns),
            "dropped_columns": list(self.dropped_columns),
            "synthesised_columns": list(self.synthesised_columns),
            "n_ingestion_errors": len(self.ingestion_errors),
            "ingestion_errors": list(self.ingestion_errors[:50]),
            "extra": dict(self.extra),
        }


class Corpus:
    """Normalised, typed, validated collection of work records.

    Attributes are exposed read-only through properties; the underlying frames
    are never mutated after construction.
    """

    def __init__(
        self,
        cleaning_result: CleaningResult,
        validation_outcome: ValidationOutcome,
        metadata: CorpusMetadata,
        schema: Schema = SCHEMA,
    ) -> None:
        """Assemble a corpus from cleaning and validation output.

        Args:
            cleaning_result: Output of :func:`~src.stage1.cleaning.clean_frame`.
            validation_outcome: Output of :func:`~src.stage1.validation.validate`.
            metadata: Provenance.
            schema: Schema the corpus conforms to.
        """
        self._schema = schema
        self._cleaning = cleaning_result
        self._validation = validation_outcome
        self._metadata = metadata

        frame = cleaning_result.frame.copy()
        frame[VALID_COLUMN] = validation_outcome.is_valid
        frame[ISSUES_COLUMN] = validation_outcome.issues
        for name in schema.names:
            frame[null_reason_column(name)] = cleaning_result.null_reasons[name]
        self._frame = frame

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def _from_ingestion(
        cls, result: IngestionResult, schema: Schema = SCHEMA
    ) -> "Corpus":
        """Run the shared clean -> validate pipeline over an ingested frame."""
        cleaning_result = clean_frame(result.frame, schema=schema)
        validation_outcome = validate(cleaning_result, schema=schema)
        metadata = CorpusMetadata(
            source=result.source,
            source_path=result.source_path,
            schema_version=schema.version,
            n_records=len(cleaning_result.frame),
            n_fields=len(schema.names),
            renamed_columns=result.metadata.get("renamed_columns", {}),
            dropped_columns=result.metadata.get("dropped_columns", []),
            synthesised_columns=result.metadata.get("synthesised_columns", []),
            ingestion_errors=result.errors,
            extra={
                key: value
                for key, value in result.metadata.items()
                if key
                not in {
                    "renamed_columns",
                    "dropped_columns",
                    "synthesised_columns",
                    "source",
                    "source_path",
                    "n_rows",
                    "n_columns",
                    "schema_version",
                }
            },
        )
        return cls(cleaning_result, validation_outcome, metadata, schema=schema)

    @classmethod
    def from_dataframe(
        cls, frame: pd.DataFrame, schema: Schema = SCHEMA, source: str = "dataframe"
    ) -> "Corpus":
        """Build a corpus from an in-memory frame (Stage1.md sec.3.3)."""
        return cls._from_ingestion(
            ingestion_module.read_dataframe(frame, schema=schema, source=source),
            schema=schema,
        )

    @classmethod
    def from_csv(cls, path: PathLike, schema: Schema = SCHEMA) -> "Corpus":
        """Build a corpus from a CSV file (Stage1.md sec.3.3)."""
        return cls._from_ingestion(
            ingestion_module.read_csv(path, schema=schema), schema=schema
        )

    @classmethod
    def from_parquet(cls, path: PathLike, schema: Schema = SCHEMA) -> "Corpus":
        """Build a corpus from a Parquet file (Stage1.md sec.3.3)."""
        return cls._from_ingestion(
            ingestion_module.read_parquet(path, schema=schema), schema=schema
        )

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        report = self.validation_report
        return (
            f"<Corpus n={len(self)} source={self._metadata.source!r} "
            f"valid={report.valid_records} invalid={report.invalid_records}>"
        )

    @property
    def records(self) -> pd.DataFrame:
        """All records, cleaned and annotated. Never filtered."""
        return self._frame

    @property
    def raw(self) -> pd.DataFrame:
        """Pre-cleaning text of every cell - the evidence chain."""
        return self._cleaning.raw

    @property
    def null_reasons(self) -> pd.DataFrame:
        """Per-cell :class:`~src.stage1.schema.NullReason` values."""
        return self._cleaning.null_reasons

    @property
    def schema(self) -> Schema:
        """Schema this corpus conforms to."""
        return self._schema

    @property
    def metadata(self) -> CorpusMetadata:
        """Provenance of this corpus."""
        return self._metadata

    @property
    def validation_report(self) -> ValidationReport:
        """Corpus-level validation summary."""
        return self._validation.report

    @property
    def display_columns(self) -> List[str]:
        """Schema columns plus the two validation annotations."""
        return list(self._schema.names) + [VALID_COLUMN, ISSUES_COLUMN]

    # ------------------------------------------------------------------
    # Views (never filters)
    # ------------------------------------------------------------------

    @property
    def valid_records(self) -> pd.DataFrame:
        """View of records with no ERROR-severity issue. The rows stay in the corpus."""
        return self._frame.loc[self._frame[VALID_COLUMN]]

    @property
    def invalid_records(self) -> pd.DataFrame:
        """View of records carrying at least one ERROR-severity issue."""
        return self._frame.loc[~self._frame[VALID_COLUMN]]

    def records_with_issue(self, code: str) -> pd.DataFrame:
        """View of records carrying a given issue code.

        Args:
            code: Bare issue code, e.g. ``"logical_date_order"``.
        """
        mask = self._frame[ISSUES_COLUMN].map(
            lambda issues: any(issue_code_of(item) == code for item in issues)
        )
        return self._frame.loc[mask]

    # ------------------------------------------------------------------
    # Inspection API (Stage1.md sec.3.7 / sec.5.3)
    # ------------------------------------------------------------------

    def head(self, n: int = DEFAULT_HEAD_ROWS, diagnostics: bool = False) -> pd.DataFrame:
        """First ``n`` records.

        Args:
            n: Number of rows. Values below zero are treated as zero.
            diagnostics: Include the ``null_reason__*`` columns.

        Returns:
            A DataFrame; empty (with correct columns) when the corpus is empty.
        """
        n = max(0, int(n))
        columns = (
            list(self._frame.columns) if diagnostics else self.display_columns
        )
        return self._frame.loc[:, columns].head(n)

    def missing_report(self, as_dict: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Per-field completeness, broken down by cause.

        The breakdown is the point: ``missing`` (nobody filled it in),
        ``placeholder`` (somebody typed "N/A") and ``unparseable`` (somebody
        typed something unusable) are different failures with different
        remedies.

        Args:
            as_dict: Return a nested dict instead of a DataFrame.

        Returns:
            Rows indexed by field, sorted by null percentage descending.
        """
        n_rows = len(self._frame)
        rows: List[Dict[str, Any]] = []
        for name in self._schema.names:
            counts = self.validation_report.null_reason_counts.get(name, {})
            n_missing = int(counts.get(NullReason.MISSING.value, 0))
            n_placeholder = int(counts.get(NullReason.PLACEHOLDER.value, 0))
            n_unparseable = int(counts.get(NullReason.UNPARSEABLE.value, 0))
            n_null = n_missing + n_placeholder + n_unparseable
            rows.append(
                {
                    "field": name,
                    "n_present": n_rows - n_null,
                    "n_missing": n_missing,
                    "n_placeholder": n_placeholder,
                    "n_unparseable": n_unparseable,
                    "n_null_total": n_null,
                    "pct_null": safe_percentage(n_null, n_rows),
                    "pct_present": safe_percentage(n_rows - n_null, n_rows),
                }
            )

        table = pd.DataFrame(rows).set_index("field")
        if len(table):
            table = table.sort_values("pct_null", ascending=False)
        if as_dict:
            return {
                "total_records": n_rows,
                "fields": table.to_dict(orient="index"),
            }
        return table

    def describe(self) -> pd.DataFrame:
        """Per-field descriptive statistics, safe on empty and all-null input.

        Unlike ``DataFrame.describe`` this covers text, numeric and date fields
        in one table and never raises on a corpus with zero rows or a column
        that is entirely null.
        """
        n_rows = len(self._frame)
        rows: List[Dict[str, Any]] = []

        for spec in self._schema:
            column = self._frame[spec.name]
            present = column.notna()
            n_present = int(present.sum())
            entry: Dict[str, Any] = {
                "field": spec.name,
                "type": spec.dtype.value,
                "count": n_present,
                "n_null": n_rows - n_present,
                "pct_null": safe_percentage(n_rows - n_present, n_rows),
                "n_extreme": None,
                "unique": None,
                "top": None,
                "top_freq": None,
                "mean": None,
                "std": None,
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "max": None,
            }

            if n_present == 0:
                rows.append(entry)
                continue

            if spec.dtype is FieldType.STRING:
                values = column.loc[present]
                entry["unique"] = int(values.nunique())
                counts = values.value_counts()
                if len(counts):
                    entry["top"] = str(counts.index[0])
                    entry["top_freq"] = int(counts.iloc[0])
            elif spec.dtype is FieldType.FLOAT:
                values = column.loc[present].to_numpy(dtype="float64")
                finite = values[np.isfinite(values)]
                entry["unique"] = int(pd.unique(values).size)
                entry["n_extreme"] = int(
                    (np.abs(finite) > IMPLAUSIBLE_AMOUNT_THRESHOLD).sum()
                )
                if finite.size:
                    # An injected 1e300 makes mean/std overflow to inf. Those
                    # values are reported honestly rather than hidden, but the
                    # overflow is expected, so it must not raise a warning - and
                    # the robust percentiles beside them stay informative.
                    with np.errstate(over="ignore", invalid="ignore"):
                        entry["mean"] = _safe_round(float(finite.mean()))
                        entry["std"] = _safe_round(
                            float(finite.std(ddof=1)) if finite.size > 1 else 0.0
                        )
                    entry["min"] = _safe_round(float(finite.min()))
                    entry["p25"] = _safe_round(float(np.percentile(finite, 25)))
                    entry["median"] = _safe_round(float(np.percentile(finite, 50)))
                    entry["p75"] = _safe_round(float(np.percentile(finite, 75)))
                    entry["max"] = _safe_round(float(finite.max()))
            else:  # DATE
                values = column.loc[present]
                entry["unique"] = int(values.nunique())
                entry["min"] = values.min().date().isoformat()
                entry["max"] = values.max().date().isoformat()
                entry["median"] = values.quantile(0.5).date().isoformat()

            rows.append(entry)

        table = pd.DataFrame(rows).set_index("field")
        for column_name in ("count", "n_null", "n_extreme", "unique", "top_freq"):
            table[column_name] = table[column_name].astype("Int64")
        return table

    def summary(self) -> Dict[str, Any]:
        """Deterministic corpus overview (Stage1.md sec.5.3).

        Returns:
            A JSON-serialisable dict. Contains no wall-clock value, so two runs
            over the same input produce byte-identical output.
        """
        report = self.validation_report
        n_rows = len(self._frame)

        amounts: Dict[str, Any] = {}
        for name in self._schema.float_fields:
            values = self._frame[name].to_numpy(dtype="float64", na_value=np.nan)
            finite = values[np.isfinite(values)]
            amounts[name] = {
                "n_finite": int(finite.size),
                "n_non_finite": int(np.isinf(values).sum()),
                "n_extreme": int(
                    (np.abs(finite) > IMPLAUSIBLE_AMOUNT_THRESHOLD).sum()
                ),
                "total": _safe_round(float(finite.sum())) if finite.size else 0.0,
                "mean": _safe_round(float(finite.mean())) if finite.size else None,
                "median": _safe_round(float(np.median(finite)))
                if finite.size
                else None,
                "min": _safe_round(float(finite.min())) if finite.size else None,
                "max": _safe_round(float(finite.max())) if finite.size else None,
            }

        date_ranges: Dict[str, Any] = {}
        for name in self._schema.date_fields:
            column = self._frame[name].dropna()
            date_ranges[name] = {
                "n_present": int(len(column)),
                "min": column.min().date().isoformat() if len(column) else None,
                "max": column.max().date().isoformat() if len(column) else None,
            }

        status_counts = (
            self._frame["status"].value_counts(dropna=True).to_dict()
            if n_rows
            else {}
        )

        top_issues = sorted(
            report.issue_counts.items(), key=lambda item: (-item[1], item[0])
        )[:10]

        return {
            "schema_version": self._schema.version,
            "source": self._metadata.source,
            "source_path": self._metadata.source_path,
            "n_records": n_rows,
            "n_fields": len(self._schema.names),
            "valid_records": report.valid_records,
            "invalid_records": report.invalid_records,
            "validity_pct": report.validity_rate_pct,
            "records_with_warnings": report.warning_record_count,
            "missing_cell_pct": report.missing_cell_rate_pct,
            "date_violation_pct": report.date_violations,
            "duplicate_key_records": report.duplicate_key_records,
            "negative_amount_records": report.negative_amount_records,
            "non_finite_amount_records": report.non_finite_amount_records,
            "implausible_amount_records": report.implausible_amount_records,
            "pre_scheme_date_records": report.pre_scheme_date_records,
            "unique_work_names": int(self._frame["work_name"].nunique()),
            "duplicate_work_name_pct": safe_percentage(
                int(
                    self._frame["work_name"]
                    .dropna()
                    .duplicated(keep="first")
                    .sum()
                ),
                n_rows,
            ),
            "amounts": amounts,
            "date_ranges": date_ranges,
            "status_distribution": {str(k): int(v) for k, v in status_counts.items()},
            "top_issues": [{"issue": code, "count": count} for code, count in top_issues],
            "n_ingestion_errors": len(self._metadata.ingestion_errors),
        }

    # ------------------------------------------------------------------
    # Typed access for downstream stages
    # ------------------------------------------------------------------

    def iter_records(self) -> Iterator[Record]:
        """Yield each row as a frozen, strongly-typed :class:`Record`.

        Nulls surface as ``None`` (never ``NaN``/``NaT``), so downstream code
        can use plain ``is None`` checks.
        """
        schema_names = list(self._schema.names)
        reason_columns = [null_reason_column(name) for name in schema_names]
        frame = self._frame

        for row in frame.itertuples(index=False, name=None):
            lookup = dict(zip(frame.columns, row))
            values: Dict[str, Any] = {}
            for name in schema_names:
                value = lookup[name]
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    values[name] = None
                elif value is pd.NaT:
                    values[name] = None
                elif isinstance(value, pd.Timestamp):
                    values[name] = value.to_pydatetime()
                else:
                    values[name] = value
            null_reasons = {
                name: str(lookup[column])
                for name, column in zip(schema_names, reason_columns)
                if str(lookup[column]) != NullReason.PRESENT.value
            }
            yield Record(
                **values,
                is_valid=bool(lookup[VALID_COLUMN]),
                issues=tuple(lookup[ISSUES_COLUMN]),
                null_reasons=null_reasons,
            )

    def to_typed_records(self) -> List[Record]:
        """Materialise every row as a :class:`Record`."""
        return list(self.iter_records())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_clean_csv(self, path: PathLike, diagnostics: bool = True) -> Path:
        """Write the cleaned corpus to CSV.

        Args:
            path: Destination file.
            diagnostics: Include ``is_valid``, ``issues`` and the
                ``null_reason__*`` columns.
        """
        frame = self._frame if diagnostics else self._frame.loc[:, list(FIELD_ORDER)]
        exportable = frame.copy()
        if ISSUES_COLUMN in exportable.columns:
            exportable[ISSUES_COLUMN] = exportable[ISSUES_COLUMN].map(
                lambda items: "|".join(items)
            )
        for name in self._schema.date_fields:
            exportable[name] = self._frame[name].dt.strftime("%Y-%m-%d")
        return ingestion_module.write_csv(exportable, path)

    def save_reports(self, output_dir: PathLike) -> Dict[str, Path]:
        """Write summary, validation and missing reports as JSON.

        Args:
            output_dir: Directory to write into; created if absent.

        Returns:
            Mapping of report name to the path written.
        """
        directory = Path(output_dir)
        written = {
            "summary": write_json(self.summary(), directory / "stage1_summary.json"),
            "validation_report": write_json(
                self.validation_report.to_dict(),
                directory / "stage1_validation_report.json",
            ),
            "missing_report": write_json(
                self.missing_report(as_dict=True),
                directory / "stage1_missing_report.json",
            ),
            "metadata": write_json(
                self._metadata.to_dict(), directory / "stage1_metadata.json"
            ),
        }
        LOGGER.info("Wrote %d Stage 1 report(s) to %s", len(written), directory)
        return written


def percentage(numerator: float, denominator: float) -> float:
    """Convenience re-export used by report consumers."""
    return round(safe_percentage(numerator, denominator), PERCENT_PRECISION)
