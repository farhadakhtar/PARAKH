"""Defect-preserving cleaning for Stage 1 (Stage1.md sec.3.6).

The single rule that governs this module:

    Cleaning changes *representation*, never *content*.

Permitted: trimming, whitespace collapsing, lowercasing, placeholder
recognition, currency/separator stripping, date parsing.

Forbidden: imputing a missing value, reordering dates that violate their
milestone ordering, clipping outliers, dropping rows. Every one of those would
repair a defect that Stage 2 exists to measure - a silent corruption, not a fix.

Each cell therefore carries a :class:`~src.stage1.schema.NullReason` explaining
why it is null, so that ``missing`` (a completeness defect), ``placeholder``
(the record author typed "N/A") and ``unparseable`` (a hard type failure) stay
distinguishable downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import pandas as pd

from src.core.logger import get_logger
from src.stage1.schema import (
    SCHEMA,
    FieldSpec,
    FieldType,
    NullReason,
    Schema,
    null_reason_column,
)
from src.utils.helpers import (
    normalize_text,
    normalize_whitespace,
    parse_date_series,
    parse_float_series,
    to_text_series,
)

LOGGER = get_logger(__name__)

_REASON_CATEGORIES = tuple(reason.value for reason in NullReason)


@dataclass(frozen=True)
class CleaningResult:
    """Output of :func:`clean_frame`.

    Attributes:
        frame: Cleaned, typed columns in schema order. Same row count and same
            index as the input - cleaning never drops a row.
        null_reasons: One categorical column per field naming the
            :class:`~src.stage1.schema.NullReason` for that cell.
        raw: The pre-cleaning text of every cell, retained as the evidence
            chain so any normalisation can be audited after the fact.
        stats: Per-field counts of each null reason plus normalisation
            activity.
    """

    frame: pd.DataFrame
    null_reasons: pd.DataFrame
    raw: pd.DataFrame
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Number of rows retained (always equal to the input row count)."""
        return len(self.frame)

    def reason_series(self, field_name: str) -> pd.Series:
        """Null reasons for one field, as a string Series."""
        return self.null_reasons[field_name].astype("object")

    def null_mask(self, field_name: str) -> pd.Series:
        """Boolean mask of cells that are null for any reason."""
        return self.null_reasons[field_name] != NullReason.PRESENT.value

    def to_frame_with_diagnostics(self) -> pd.DataFrame:
        """Cleaned columns plus their ``null_reason__*`` companions."""
        combined = self.frame.copy()
        for name in self.frame.columns:
            combined[null_reason_column(name)] = self.null_reasons[name]
        return combined


def _empty_reason_series(index: pd.Index, value: NullReason) -> pd.Series:
    """Build a categorical reason column filled with ``value``."""
    return pd.Series(
        pd.Categorical(
            [value.value] * len(index), categories=_REASON_CATEGORIES
        ),
        index=index,
    )


def clean_series(series: pd.Series, spec: FieldSpec) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Clean one column according to its :class:`FieldSpec`.

    Order of operations matters and is fixed:

    1. Snapshot the raw text (evidence chain).
    2. Trim and collapse whitespace.
    3. Classify genuinely empty cells as ``missing``.
    4. Classify absence tokens as ``placeholder``.
    5. Coerce the survivors to the declared type; failures become
       ``unparseable`` while keeping their raw text in the snapshot.

    Placeholder detection must precede coercion, otherwise ``"0000-00-00"``
    would be reported as an unparseable date rather than the deliberate
    absence token it is.

    Args:
        series: Raw column.
        spec: Declaration for this field.

    Returns:
        ``(values, reasons, raw_text)``.
    """
    index = series.index
    raw_text = to_text_series(series)

    if len(series) == 0:
        empty_values = pd.Series(
            pd.array([], dtype=spec.dtype.pandas_dtype), index=index
        )
        return empty_values, _empty_reason_series(index, NullReason.PRESENT), raw_text

    trimmed = normalize_whitespace(raw_text)

    missing_mask = trimmed.isna() | (trimmed.fillna("") == "")
    lowered = trimmed.str.lower()
    from src.core.constants import PLACEHOLDER_TOKENS  # local: avoids cycle noise

    placeholder_mask = (~missing_mask) & lowered.isin(PLACEHOLDER_TOKENS).fillna(False)

    # Cells that still hold a candidate value.
    candidate = trimmed.where(~(missing_mask | placeholder_mask), other=None)

    if spec.dtype is FieldType.STRING:
        values = normalize_text(candidate)
        values = values.where(values.notna(), other=None).astype("object")
        unparseable_mask = pd.Series(False, index=index, dtype=bool)
    elif spec.dtype is FieldType.FLOAT:
        values, unparseable_mask = parse_float_series(candidate)
    elif spec.dtype is FieldType.DATE:
        values, unparseable_mask = parse_date_series(candidate)
    else:  # pragma: no cover - FieldType is closed
        raise TypeError(f"Unhandled field type {spec.dtype!r}")

    reasons = _empty_reason_series(index, NullReason.PRESENT)
    reasons = reasons.astype("object")
    reasons[missing_mask.to_numpy()] = NullReason.MISSING.value
    reasons[placeholder_mask.to_numpy()] = NullReason.PLACEHOLDER.value
    reasons[unparseable_mask.to_numpy()] = NullReason.UNPARSEABLE.value
    reasons = pd.Series(
        pd.Categorical(reasons, categories=_REASON_CATEGORIES), index=index
    )

    return values, reasons, raw_text


def clean_frame(frame: pd.DataFrame, schema: Schema = SCHEMA) -> CleaningResult:
    """Clean an already schema-aligned frame.

    Args:
        frame: Frame whose columns are exactly ``schema.names``, in order.
        schema: Schema to clean against.

    Returns:
        A :class:`CleaningResult` with identical row count and index.

    Raises:
        ValueError: If ``frame`` is not schema-aligned. Alignment is
            :meth:`~src.stage1.schema.Schema.align`'s job and happens during
            ingestion.
    """
    if tuple(frame.columns) != schema.names:
        raise ValueError(
            "clean_frame expects a schema-aligned frame; got columns "
            f"{tuple(frame.columns)!r}"
        )

    index = frame.index
    cleaned: Dict[str, pd.Series] = {}
    reasons: Dict[str, pd.Series] = {}
    raws: Dict[str, pd.Series] = {}
    stats: Dict[str, Any] = {"per_field": {}, "normalization_changes": {}}

    for spec in schema:
        values, reason_series, raw_text = clean_series(frame[spec.name], spec)
        cleaned[spec.name] = values
        reasons[spec.name] = reason_series
        raws[spec.name] = raw_text

        counts = reason_series.value_counts()
        stats["per_field"][spec.name] = {
            reason.value: int(counts.get(reason.value, 0)) for reason in NullReason
        }

        if spec.dtype is FieldType.STRING and len(frame) > 0:
            present = reason_series == NullReason.PRESENT.value
            changed = int(
                (
                    present
                    & (raw_text.where(present, other=None) != values.where(present, other=None))
                ).sum()
            )
            stats["normalization_changes"][spec.name] = changed

    clean_df = pd.DataFrame(cleaned, index=index, columns=list(schema.names))
    for spec in schema:
        target = spec.dtype.pandas_dtype
        if str(clean_df[spec.name].dtype) != target:
            clean_df[spec.name] = clean_df[spec.name].astype(target)

    reason_df = pd.DataFrame(reasons, index=index, columns=list(schema.names))
    raw_df = pd.DataFrame(raws, index=index, columns=list(schema.names))

    total_cells = len(frame) * len(schema.names)
    null_cells = int(sum(
        stats["per_field"][name][reason.value]
        for name in schema.names
        for reason in NullReason
        if reason.is_null
    ))
    stats["total_cells"] = total_cells
    stats["null_cells"] = null_cells
    stats["rows_in"] = len(frame)
    stats["rows_out"] = len(clean_df)

    LOGGER.info(
        "Cleaned %d rows x %d fields; %d null cells (%.2f%%); no rows dropped.",
        len(clean_df),
        len(schema.names),
        null_cells,
        (100.0 * null_cells / total_cells) if total_cells else 0.0,
    )

    return CleaningResult(
        frame=clean_df, null_reasons=reason_df, raw=raw_df, stats=stats
    )
