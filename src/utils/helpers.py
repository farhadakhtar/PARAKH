"""Vectorised primitives shared across Stage 1.

Everything here is column-at-a-time pandas work. Stage1.md sec.4 gives a
50k-rows-in-5s budget, so no function in this module may iterate rows in
Python.

These helpers are deliberately *mechanical*: they normalise representation
only. Deciding what a normalised value means (missing? placeholder? invalid?)
is the job of :mod:`src.stage1.cleaning` and :mod:`src.stage1.validation`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    CURRENCY_TOKENS,
    DATE_FORMATS,
    PERCENT_PRECISION,
    PLACEHOLDER_TOKENS,
)

_WHITESPACE_PATTERN = r"\s+"


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def to_text_series(series: pd.Series) -> pd.Series:
    """Render any series as nullable text without inventing values.

    Genuine nulls (``None``, ``NaN``, ``NaT``) stay null; everything else is
    stringified. Floats keep their shortest round-trip representation, so a
    CSV round-trip is lossless.

    Args:
        series: Any pandas Series.

    Returns:
        Object-dtype Series of ``str`` or ``None``.
    """
    if len(series) == 0:
        return pd.Series([], dtype="object", index=series.index)
    null_mask = series.isna()
    text = series.astype("object").where(~null_mask, other=None)
    non_null = ~null_mask
    if non_null.any():
        text.loc[non_null] = text.loc[non_null].map(_stringify)
    return text


def _stringify(value: Any) -> str:
    """Stringify one scalar, avoiding float artefacts like ``123456.0``."""
    if isinstance(value, str):
        return value
    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        if math.isfinite(as_float) and as_float.is_integer():
            return str(int(as_float))
        return repr(as_float)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    return str(value)


def normalize_whitespace(series: pd.Series) -> pd.Series:
    """Trim ends and collapse internal whitespace runs to a single space."""
    if len(series) == 0:
        return series
    return series.str.strip().str.replace(_WHITESPACE_PATTERN, " ", regex=True)


def normalize_text(series: pd.Series) -> pd.Series:
    """Apply the full Stage1.md sec.3.6 string normalisation.

    Trim, collapse whitespace, then lowercase. Purely representational: no
    value is added, removed or corrected.
    """
    if len(series) == 0:
        return series
    return normalize_whitespace(series).str.lower()


def placeholder_mask(series: pd.Series) -> pd.Series:
    """Flag cells whose text is a known absence token.

    Comparison happens on the trimmed, whitespace-collapsed, lowercased form so
    that ``" N/A "`` and ``"n/a"`` are both caught.

    Args:
        series: Object-dtype text series.

    Returns:
        Boolean Series, ``False`` wherever the cell is already null.
    """
    if len(series) == 0:
        return pd.Series([], dtype=bool, index=series.index)
    normalized = normalize_text(series)
    return normalized.isin(PLACEHOLDER_TOKENS).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def strip_currency(series: pd.Series) -> pd.Series:
    """Remove currency markers, grouping separators and stray spacing.

    Handles the Indian grouping style (``1,25,000``) because the substitution is
    positional-agnostic.
    """
    cleaned = series.str.lower()
    for token in CURRENCY_TOKENS:
        cleaned = cleaned.str.replace(token, "", regex=False)
    return cleaned.str.strip()


def parse_float_series(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Coerce a column to float64, reporting which present cells failed.

    Fast path uses :func:`pandas.to_numeric` directly; only the cells that fail
    are re-attempted after currency/separator stripping.

    Non-finite results (``inf`` from an overflowing literal such as ``1.2e400``)
    are *kept*, not nulled: the value was present and parseable, and hiding it
    would be a silent repair. Validation flags it instead.

    Args:
        series: Object-dtype series with placeholders already nulled.

    Returns:
        ``(values, unparseable_mask)`` where ``values`` is float64 and
        ``unparseable_mask`` marks cells that were present but not numeric.
    """
    index = series.index
    if len(series) == 0:
        return (
            pd.Series([], dtype="float64", index=index),
            pd.Series([], dtype=bool, index=index),
        )

    present = series.notna()
    values = pd.to_numeric(series, errors="coerce").astype("float64")

    failed = values.isna() & present
    if bool(failed.any()):
        stripped = strip_currency(to_text_series(series[failed]))
        values.loc[failed] = pd.to_numeric(stripped, errors="coerce").astype("float64")

        # pandas' parser rejects a literal that overflows float64 ("1.2e400"),
        # but IEEE-754 defines the result as +/-inf. The cell IS present and IS
        # parseable, so it is kept and left for validation to flag as
        # non-finite. Nulling it here would pretend nobody wrote anything -
        # exactly the silent corruption Stage 1 exists to prevent.
        still_failing = values.index[(values.isna() & present).to_numpy()]
        if len(still_failing):
            values.loc[still_failing] = (
                stripped.reindex(still_failing).map(_float_or_nan).astype("float64")
            )

    unparseable = (values.isna() & present).astype(bool)
    return values, unparseable


def _float_or_nan(text: Any) -> float:
    """``float(text)`` with overflow preserved as ``inf`` and failure as NaN."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def parse_date_series(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Coerce a column to ``datetime64[ns]``, reporting present-but-bad cells.

    ISO-8601 is tried first, then each format in
    :data:`~src.core.constants.DATE_FORMATS` in a fixed order. The order is
    fixed rather than inferred so that an ambiguous string like ``03-04-2019``
    always resolves the same way.

    Args:
        series: Object-dtype series with placeholders already nulled.

    Returns:
        ``(values, unparseable_mask)``.
    """
    index = series.index
    if len(series) == 0:
        return (
            pd.Series(pd.array([], dtype="datetime64[ns]"), index=index),
            pd.Series([], dtype=bool, index=index),
        )

    present = series.notna()
    if not bool(present.any()):
        return (
            pd.Series(pd.NaT, index=index, dtype="datetime64[ns]"),
            pd.Series(False, index=index, dtype=bool),
        )

    values = pd.to_datetime(series, format="ISO8601", errors="coerce")
    values = pd.Series(values, index=index).astype("datetime64[ns]")

    for fmt in DATE_FORMATS:
        remaining = values.isna() & present
        if not bool(remaining.any()):
            break
        attempt = pd.to_datetime(
            to_text_series(series[remaining]), format=fmt, errors="coerce"
        )
        values.loc[remaining] = attempt.values

    unparseable = (values.isna() & present).astype(bool)
    return values, unparseable


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def safe_percentage(numerator: float, denominator: float) -> float:
    """Percentage that returns 0.0 instead of raising on an empty corpus.

    Args:
        numerator: Count of matching items.
        denominator: Total count; may legitimately be zero.

    Returns:
        ``100 * numerator / denominator`` rounded to
        :data:`~src.core.constants.PERCENT_PRECISION`, or ``0.0``.
    """
    if not denominator:
        return 0.0
    return round(100.0 * float(numerator) / float(denominator), PERCENT_PRECISION)


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if absent and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value: Any) -> Any:
    """Fallback encoder making numpy/pandas scalars JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else str(as_float)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def write_json(payload: Mapping[str, Any], path: Path, indent: int = 2) -> Path:
    """Write ``payload`` as UTF-8 JSON with stable key order.

    ``sort_keys`` is on so two runs of the same seed diff to nothing.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=indent,
            default=json_default,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")
    return path


def weighted_choice(
    rng: np.random.Generator, options: Iterable[str], weights: Iterable[float], size: int
) -> np.ndarray:
    """Draw ``size`` items from ``options`` using normalised ``weights``."""
    option_array = np.asarray(list(options), dtype=object)
    weight_array = np.asarray(list(weights), dtype="float64")
    weight_array = weight_array / weight_array.sum()
    return rng.choice(option_array, size=size, p=weight_array)
