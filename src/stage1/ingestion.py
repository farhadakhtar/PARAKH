"""File ingestion for Stage 1 (Stage1.md sec.3.3).

Reads CSV and Parquet into a schema-aligned *raw* frame. Cleaning and
validation happen afterwards, identically for every entry point, so
``from_csv``, ``from_parquet`` and ``from_dataframe`` cannot drift apart.

Two decisions here are load-bearing:

* **CSV is read with NA-detection disabled.** pandas' default
  ``keep_default_na`` silently rewrites ``"N/A"``, ``"NULL"`` and ``"None"``
  into ``NaN``. That would erase the difference between a cell nobody filled in
  and a cell somebody deliberately filled with an absence token - precisely the
  distinction Stage 1 exists to preserve.
* **Every column is read as text.** Letting pandas infer types would coerce
  ``"00123"`` to ``123`` and guess date formats per-chunk.

Malformed rows never abort a load: they are quarantined, logged, and reported
in :attr:`IngestionResult.errors`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.core.logger import get_logger
from src.stage1.schema import SCHEMA, Schema, SchemaError
from src.utils.helpers import ensure_dir

LOGGER = get_logger(__name__)

PathLike = Union[str, Path]

_COLUMN_CLEAN_PATTERN = re.compile(r"[\s\-]+")


class IngestionError(RuntimeError):
    """Raised when a source cannot be read at all."""


@dataclass(frozen=True)
class IngestionResult:
    """A schema-aligned raw frame plus everything that went wrong reading it."""

    frame: pd.DataFrame
    source: str
    source_path: Optional[str] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Rows successfully read."""
        return len(self.frame)

    @property
    def n_errors(self) -> int:
        """Rows or columns that could not be read cleanly."""
        return len(self.errors)


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Canonicalise column labels to ``snake_case`` lowercase.

    ``"Work ID"``, ``" work-id "`` and ``"work_id"`` all become ``work_id``.

    Args:
        frame: Frame whose columns may be inconsistently labelled.

    Returns:
        ``(renamed_frame, mapping)`` where ``mapping`` holds only the labels
        that actually changed.

    Raises:
        SchemaError: If two source columns collapse onto the same label.
    """
    mapping: Dict[str, str] = {}
    new_labels: List[str] = []
    for label in frame.columns:
        cleaned = _COLUMN_CLEAN_PATTERN.sub("_", str(label).strip().lower())
        new_labels.append(cleaned)
        if cleaned != label:
            mapping[str(label)] = cleaned

    duplicates = {
        label for label in new_labels if new_labels.count(label) > 1
    }
    if duplicates:
        raise SchemaError(
            f"Column labels collide after normalisation: {sorted(duplicates)!r}"
        )

    renamed = frame.copy()
    renamed.columns = new_labels
    if mapping:
        LOGGER.info("Renamed %d column(s): %s", len(mapping), mapping)
    return renamed, mapping


def _prepare(
    frame: pd.DataFrame,
    source: str,
    schema: Schema,
    source_path: Optional[Path] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> IngestionResult:
    """Normalise labels, align to schema and package the result."""
    errors = list(errors or [])
    renamed, rename_map = normalize_columns(frame)
    aligned, align_report = schema.align(renamed)
    aligned = aligned.reset_index(drop=True)

    if align_report["dropped_columns"]:
        LOGGER.warning(
            "Dropped %d column(s) not in schema: %s",
            len(align_report["dropped_columns"]),
            align_report["dropped_columns"],
        )
    for name in align_report["synthesised_columns"]:
        LOGGER.warning("Column %r absent from source; filled with nulls.", name)
        errors.append(
            {"kind": "missing_column", "column": name, "action": "filled_with_null"}
        )

    metadata: Dict[str, Any] = {
        "source": source,
        "source_path": str(source_path) if source_path else None,
        "n_rows": len(aligned),
        "n_columns": len(aligned.columns),
        "schema_version": schema.version,
        "renamed_columns": rename_map,
        "dropped_columns": align_report["dropped_columns"],
        "synthesised_columns": align_report["synthesised_columns"],
        "read_errors": len(errors),
    }
    metadata.update(extra_metadata or {})

    LOGGER.info(
        "Ingested %d row(s) from %s (%s); %d read error(s).",
        len(aligned),
        source,
        source_path or "in-memory",
        len(errors),
    )
    return IngestionResult(
        frame=aligned,
        source=source,
        source_path=str(source_path) if source_path else None,
        errors=errors,
        metadata=metadata,
    )


def read_dataframe(
    frame: pd.DataFrame, schema: Schema = SCHEMA, source: str = "dataframe"
) -> IngestionResult:
    """Ingest an in-memory frame.

    Args:
        frame: Any frame carrying at least the schema's required columns.
        schema: Schema to align against.
        source: Label recorded in metadata.

    Returns:
        Schema-aligned :class:`IngestionResult`.
    """
    if not isinstance(frame, pd.DataFrame):
        raise IngestionError(f"Expected a DataFrame, got {type(frame).__name__}")
    return _prepare(frame, source=source, schema=schema)


def read_csv(path: PathLike, schema: Schema = SCHEMA) -> IngestionResult:
    """Ingest a CSV file, quarantining malformed rows.

    The fast C parser is tried first. If it hits a structurally broken row the
    file is re-read with the Python parser, which can hand each bad line to a
    callback so it can be logged and skipped instead of aborting the load.

    Args:
        path: CSV path.
        schema: Schema to align against.

    Returns:
        Schema-aligned :class:`IngestionResult` whose ``errors`` list contains
        one entry per skipped line.

    Raises:
        IngestionError: If the file is missing or unreadable.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise IngestionError(f"CSV not found: {csv_path}")

    read_options: Dict[str, Any] = {
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "encoding": "utf-8",
    }

    errors: List[Dict[str, Any]] = []
    try:
        frame = pd.read_csv(csv_path, **read_options)
    except pd.errors.EmptyDataError:
        LOGGER.warning("CSV %s has no parseable content; treating as empty.", csv_path)
        frame = pd.DataFrame(columns=list(schema.names))
    except pd.errors.ParserError as exc:
        LOGGER.warning(
            "C parser failed on %s (%s); retrying with tolerant Python parser.",
            csv_path,
            exc,
        )

        def _collect_bad_line(line: List[str]) -> None:
            errors.append(
                {
                    "kind": "malformed_row",
                    "n_fields": len(line),
                    "preview": ",".join(line)[:200],
                    "action": "skipped",
                }
            )
            return None

        try:
            frame = pd.read_csv(
                csv_path,
                engine="python",
                on_bad_lines=_collect_bad_line,
                **read_options,
            )
        except Exception as inner:  # pragma: no cover - unrecoverable file
            raise IngestionError(f"Unable to parse CSV {csv_path}: {inner}") from inner
        LOGGER.warning("Skipped %d malformed row(s) in %s.", len(errors), csv_path)
    except UnicodeDecodeError as exc:
        raise IngestionError(f"CSV {csv_path} is not valid UTF-8: {exc}") from exc

    return _prepare(
        frame,
        source="csv",
        schema=schema,
        source_path=csv_path,
        errors=errors,
        extra_metadata={"file_bytes": csv_path.stat().st_size},
    )


def read_parquet(path: PathLike, schema: Schema = SCHEMA) -> IngestionResult:
    """Ingest a Parquet file.

    Parquet carries its own types, so no NA-detection workaround is needed;
    cleaning re-derives every value from its text form regardless, which keeps
    the CSV and Parquet paths behaviourally identical.

    Args:
        path: Parquet path.
        schema: Schema to align against.

    Returns:
        Schema-aligned :class:`IngestionResult`.

    Raises:
        IngestionError: If the file is missing or cannot be decoded.
    """
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise IngestionError(f"Parquet not found: {parquet_path}")
    try:
        frame = pd.read_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - pyarrow error surface
        raise IngestionError(f"Unable to read Parquet {parquet_path}: {exc}") from exc

    return _prepare(
        frame,
        source="parquet",
        schema=schema,
        source_path=parquet_path,
        extra_metadata={"file_bytes": parquet_path.stat().st_size},
    )


def write_csv(frame: pd.DataFrame, path: PathLike) -> Path:
    """Write a frame to CSV without inventing null tokens.

    ``na_rep=""`` keeps empty cells empty so a round-trip preserves the
    missing-versus-placeholder distinction exactly.
    """
    target = Path(path)
    ensure_dir(target.parent)
    frame.to_csv(target, index=False, na_rep="", encoding="utf-8", lineterminator="\n")
    LOGGER.info("Wrote %d row(s) to %s", len(frame), target)
    return target


def write_parquet(frame: pd.DataFrame, path: PathLike) -> Path:
    """Write a frame to Parquet, stringifying mixed columns first.

    The synthetic dataset deliberately mixes numbers and placeholder text in
    the same column, which Arrow cannot type. Serialising as text is lossless
    here because cleaning parses from text anyway.
    """
    target = Path(path)
    ensure_dir(target.parent)
    safe = frame.copy()
    for column in safe.columns:
        if safe[column].dtype == object:
            safe[column] = safe[column].map(
                lambda value: None if value is None or value != value else str(value)
            )
    safe.to_parquet(target, index=False)
    LOGGER.info("Wrote %d row(s) to %s", len(safe), target)
    return target
