"""Schema, value and logical validation (Stage1.md sec.3.5).

Validation *labels* records. It never deletes, repairs or reorders them.

Two severities exist and the distinction is deliberate:

``ERROR``
    A type, value or logical violation - something that is provably wrong
    regardless of how much data is missing. Any error makes ``is_valid`` False.

``WARNING``
    An incompleteness signal (missing field, placeholder, implausible-but-
    possible date). Recorded on the record, but does **not** invalidate it.

Treating missing fields as errors would mark ~80% of a realistically dirty
corpus "invalid" and make the flag useless. Completeness is measured
separately, in ``missing_fields``, and is what Stage 2 turns into ``C_comp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.core.constants import (
    ALLOWED_STATUS,
    IMPLAUSIBLE_AMOUNT_THRESHOLD,
    ORDERED_DATE_PAIRS,
    PERCENT_PRECISION,
    REFERENCE_DATE,
    SCHEMA_VERSION,
    SCHEME_START_DATE,
)
from src.core.logger import get_logger
from src.stage1.cleaning import CleaningResult
from src.stage1.schema import SCHEMA, FieldType, NullReason, Schema
from src.utils.helpers import safe_percentage

LOGGER = get_logger(__name__)


class Severity(str, Enum):
    """Whether an issue invalidates the record."""

    ERROR = "error"
    WARNING = "warning"


class IssueCode(str, Enum):
    """Stable machine-readable issue identifiers."""

    TYPE_UNPARSEABLE = "type_unparseable"
    VALUE_NEGATIVE = "value_negative"
    VALUE_NON_FINITE = "value_non_finite"
    VALUE_IMPLAUSIBLE_MAGNITUDE = "value_implausible_magnitude"
    VALUE_UNKNOWN_STATUS = "value_unknown_status"
    LOGICAL_DATE_ORDER = "logical_date_order"
    LOGICAL_DATE_BEFORE_SCHEME_START = "logical_date_before_scheme_start"
    LOGICAL_DATE_IN_FUTURE = "logical_date_in_future"
    SCHEMA_MISSING_KEY = "schema_missing_key"
    SCHEMA_DUPLICATE_KEY = "schema_duplicate_key"
    COMPLETENESS_MISSING = "completeness_missing"
    COMPLETENESS_PLACEHOLDER = "completeness_placeholder"


#: Severity of each issue code. Errors invalidate; warnings annotate.
ISSUE_SEVERITY: Dict[IssueCode, Severity] = {
    IssueCode.TYPE_UNPARSEABLE: Severity.ERROR,
    IssueCode.VALUE_NEGATIVE: Severity.ERROR,
    IssueCode.VALUE_NON_FINITE: Severity.ERROR,
    IssueCode.VALUE_IMPLAUSIBLE_MAGNITUDE: Severity.ERROR,
    IssueCode.VALUE_UNKNOWN_STATUS: Severity.ERROR,
    IssueCode.LOGICAL_DATE_ORDER: Severity.ERROR,
    IssueCode.LOGICAL_DATE_BEFORE_SCHEME_START: Severity.ERROR,
    IssueCode.SCHEMA_MISSING_KEY: Severity.ERROR,
    IssueCode.SCHEMA_DUPLICATE_KEY: Severity.ERROR,
    IssueCode.LOGICAL_DATE_IN_FUTURE: Severity.WARNING,
    IssueCode.COMPLETENESS_MISSING: Severity.WARNING,
    IssueCode.COMPLETENESS_PLACEHOLDER: Severity.WARNING,
}


def format_issue(code: IssueCode, detail: str = "") -> str:
    """Render an issue as ``code:detail`` (or just ``code`` when detail-free)."""
    return f"{code.value}:{detail}" if detail else code.value


def issue_code_of(issue: str) -> str:
    """Extract the bare code from a rendered issue string."""
    return issue.split(":", 1)[0]


class ValidationReport(BaseModel):
    """Corpus-level validation summary (Stage1.md sec.5.2).

    The first five fields are exactly the PRD's required output; everything
    after them is additional diagnostic detail.
    """

    model_config = ConfigDict(frozen=True)

    total_records: int = Field(ge=0)
    valid_records: int = Field(ge=0)
    invalid_records: int = Field(ge=0)
    missing_fields: Dict[str, float] = Field(default_factory=dict)
    date_violations: float = 0.0

    # --- diagnostics -----------------------------------------------------
    schema_version: str = SCHEMA_VERSION
    validity_rate_pct: float = 0.0
    missing_cell_rate_pct: float = 0.0
    null_reason_counts: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    issue_counts: Dict[str, int] = Field(default_factory=dict)
    error_record_count: int = 0
    warning_record_count: int = 0
    duplicate_key_records: int = 0
    unparseable_cells: Dict[str, int] = Field(default_factory=dict)
    negative_amount_records: int = 0
    non_finite_amount_records: int = 0
    implausible_amount_records: int = 0
    pre_scheme_date_records: int = 0
    future_date_records: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict form, suitable for ``json.dump``."""
        return self.model_dump()

    def prd_view(self) -> Dict[str, Any]:
        """Just the five keys Stage1.md sec.5.2 mandates."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "missing_fields": dict(self.missing_fields),
            "date_violations": self.date_violations,
        }


@dataclass(frozen=True)
class ValidationOutcome:
    """Per-record validation results plus the corpus report."""

    is_valid: pd.Series
    issues: pd.Series
    report: ValidationReport

    @property
    def n_valid(self) -> int:
        """Number of records with no ERROR-severity issue."""
        return int(self.is_valid.sum())

    @property
    def n_invalid(self) -> int:
        """Number of records carrying at least one ERROR-severity issue."""
        return int((~self.is_valid).sum())


class _IssueAccumulator:
    """Collects per-row issues from vectorised masks.

    Rules are evaluated as whole-column boolean masks; only the sparse
    ``True`` positions are visited, so the Python-level cost is proportional to
    the number of issues found, not to the number of rows.
    """

    def __init__(self, n_rows: int) -> None:
        self._issues: List[List[str]] = [[] for _ in range(n_rows)]
        self._has_error = np.zeros(n_rows, dtype=bool)
        self._has_warning = np.zeros(n_rows, dtype=bool)
        self.counts: Dict[str, int] = {}

    def add(self, mask: Any, code: IssueCode, detail: str = "") -> int:
        """Record ``code`` on every row where ``mask`` is True.

        Returns:
            Number of rows affected.
        """
        array = np.asarray(mask, dtype=bool)
        positions = np.flatnonzero(array)
        if positions.size == 0:
            self.counts.setdefault(format_issue(code, detail), 0)
            return 0

        rendered = format_issue(code, detail)
        for position in positions:
            self._issues[position].append(rendered)

        if ISSUE_SEVERITY[code] is Severity.ERROR:
            self._has_error[positions] = True
        else:
            self._has_warning[positions] = True

        self.counts[rendered] = self.counts.get(rendered, 0) + int(positions.size)
        return int(positions.size)

    def finalize(self, index: pd.Index) -> Tuple[pd.Series, pd.Series, np.ndarray]:
        """Materialise the accumulated state as pandas objects."""
        issues = pd.Series(
            [tuple(items) for items in self._issues], index=index, dtype="object"
        )
        is_valid = pd.Series(~self._has_error, index=index, dtype=bool)
        return is_valid, issues, self._has_warning


def validate(
    cleaning_result: CleaningResult, schema: Schema = SCHEMA
) -> ValidationOutcome:
    """Validate a cleaned frame.

    Applies, in order:

    1. **Schema validation** - key present, key unique.
    2. **Type validation** - cells that resisted coercion during cleaning.
    3. **Value validation** - amounts finite and non-negative, status in
       vocabulary.
    4. **Logical validation** - ``proposal <= approval <= completion``,
       evaluated only on pairs where *both* dates survived cleaning, plus
       scheme-start and future-date plausibility.
    5. **Completeness** - warnings, never errors.

    Args:
        cleaning_result: Output of :func:`~src.stage1.cleaning.clean_frame`.
        schema: Schema to validate against.

    Returns:
        A :class:`ValidationOutcome`. Row count is preserved exactly.
    """
    frame = cleaning_result.frame
    reasons = cleaning_result.null_reasons
    index = frame.index
    n_rows = len(frame)

    accumulator = _IssueAccumulator(n_rows)
    unparseable_cells: Dict[str, int] = {}

    # -- 1. schema --------------------------------------------------------
    key = schema.key_field
    key_series = frame[key]
    key_missing = key_series.isna().to_numpy()
    accumulator.add(key_missing, IssueCode.SCHEMA_MISSING_KEY, key)

    duplicate_key = (
        key_series.duplicated(keep=False) & key_series.notna()
    ).to_numpy()
    duplicate_key_records = accumulator.add(
        duplicate_key, IssueCode.SCHEMA_DUPLICATE_KEY, key
    )

    # -- 2. type ----------------------------------------------------------
    for name in schema.names:
        unparseable = (
            reasons[name].astype("object") == NullReason.UNPARSEABLE.value
        ).to_numpy()
        count = accumulator.add(unparseable, IssueCode.TYPE_UNPARSEABLE, name)
        unparseable_cells[name] = count

    # -- 3. value ---------------------------------------------------------
    negative_rows = np.zeros(n_rows, dtype=bool)
    non_finite_rows = np.zeros(n_rows, dtype=bool)
    implausible_rows = np.zeros(n_rows, dtype=bool)
    for spec in schema:
        if spec.dtype is not FieldType.FLOAT:
            continue
        values = frame[spec.name].to_numpy(dtype="float64", na_value=np.nan)
        present = ~np.isnan(values)

        non_finite = present & np.isinf(values)
        accumulator.add(non_finite, IssueCode.VALUE_NON_FINITE, spec.name)
        non_finite_rows |= non_finite

        implausible = (
            present
            & ~np.isinf(values)
            & (np.abs(values) > IMPLAUSIBLE_AMOUNT_THRESHOLD)
        )
        accumulator.add(implausible, IssueCode.VALUE_IMPLAUSIBLE_MAGNITUDE, spec.name)
        implausible_rows |= implausible

        if spec.non_negative:
            negative = present & ~np.isinf(values) & (values < 0.0)
            accumulator.add(negative, IssueCode.VALUE_NEGATIVE, spec.name)
            negative_rows |= negative

    status_spec = schema.spec("status")
    if status_spec.allowed_values:
        status_values = frame["status"]
        unknown_status = (
            status_values.notna() & ~status_values.isin(list(ALLOWED_STATUS))
        ).to_numpy()
        accumulator.add(unknown_status, IssueCode.VALUE_UNKNOWN_STATUS, "status")

    # -- 4. logical -------------------------------------------------------
    date_violation_rows = np.zeros(n_rows, dtype=bool)
    for earlier, later in ORDERED_DATE_PAIRS:
        left = frame[earlier]
        right = frame[later]
        both_present = (left.notna() & right.notna()).to_numpy()
        out_of_order = both_present & (right < left).to_numpy()
        accumulator.add(
            out_of_order, IssueCode.LOGICAL_DATE_ORDER, f"{later}<{earlier}"
        )
        date_violation_rows |= out_of_order

    scheme_start = pd.Timestamp(SCHEME_START_DATE)
    reference = pd.Timestamp(REFERENCE_DATE)
    pre_scheme_rows = np.zeros(n_rows, dtype=bool)
    future_rows = np.zeros(n_rows, dtype=bool)
    for name in schema.date_fields:
        column = frame[name]
        present = column.notna()
        before = (present & (column < scheme_start)).to_numpy()
        accumulator.add(before, IssueCode.LOGICAL_DATE_BEFORE_SCHEME_START, name)
        pre_scheme_rows |= before

        ahead = (present & (column > reference)).to_numpy()
        accumulator.add(ahead, IssueCode.LOGICAL_DATE_IN_FUTURE, name)
        future_rows |= ahead

    # -- 5. completeness (warnings only) ----------------------------------
    missing_fields_pct: Dict[str, float] = {}
    null_reason_counts: Dict[str, Dict[str, int]] = {}
    total_null_cells = 0
    for name in schema.names:
        reason_column = reasons[name].astype("object")
        missing = (reason_column == NullReason.MISSING.value).to_numpy()
        placeholder = (reason_column == NullReason.PLACEHOLDER.value).to_numpy()
        unparseable = (reason_column == NullReason.UNPARSEABLE.value).to_numpy()

        accumulator.add(missing, IssueCode.COMPLETENESS_MISSING, name)
        accumulator.add(placeholder, IssueCode.COMPLETENESS_PLACEHOLDER, name)

        n_null = int(missing.sum() + placeholder.sum() + unparseable.sum())
        total_null_cells += n_null
        missing_fields_pct[name] = safe_percentage(n_null, n_rows)
        null_reason_counts[name] = {
            NullReason.MISSING.value: int(missing.sum()),
            NullReason.PLACEHOLDER.value: int(placeholder.sum()),
            NullReason.UNPARSEABLE.value: int(unparseable.sum()),
            NullReason.PRESENT.value: int(
                n_rows - missing.sum() - placeholder.sum() - unparseable.sum()
            ),
        }

    is_valid, issues, has_warning = accumulator.finalize(index)
    n_valid = int(is_valid.sum())
    n_invalid = n_rows - n_valid
    total_cells = n_rows * len(schema.names)

    report = ValidationReport(
        total_records=n_rows,
        valid_records=n_valid,
        invalid_records=n_invalid,
        missing_fields=missing_fields_pct,
        date_violations=safe_percentage(int(date_violation_rows.sum()), n_rows),
        schema_version=schema.version,
        validity_rate_pct=safe_percentage(n_valid, n_rows),
        missing_cell_rate_pct=safe_percentage(total_null_cells, total_cells),
        null_reason_counts=null_reason_counts,
        issue_counts={
            code: count for code, count in sorted(accumulator.counts.items()) if count
        },
        error_record_count=n_invalid,
        warning_record_count=int(has_warning.sum()),
        duplicate_key_records=duplicate_key_records,
        unparseable_cells={k: v for k, v in unparseable_cells.items() if v},
        negative_amount_records=int(negative_rows.sum()),
        non_finite_amount_records=int(non_finite_rows.sum()),
        implausible_amount_records=int(implausible_rows.sum()),
        pre_scheme_date_records=int(pre_scheme_rows.sum()),
        future_date_records=int(future_rows.sum()),
    )

    LOGGER.info(
        "Validated %d records: %d valid (%.2f%%), %d invalid; "
        "date-order violations %.2f%%.",
        n_rows,
        n_valid,
        report.validity_rate_pct,
        n_invalid,
        report.date_violations,
    )

    return ValidationOutcome(is_valid=is_valid, issues=issues, report=report)


def round_percentage(value: float) -> float:
    """Round to the project-wide percentage precision."""
    return round(float(value), PERCENT_PRECISION)
