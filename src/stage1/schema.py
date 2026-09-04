"""Strict schema definition for the Stage 1 corpus.

Implements Stage1.md sec.3.4. The schema is the single source of truth for
which columns exist, what type each carries, and which of them may be null.

Two representations are exposed:

* :data:`SCHEMA` - the rich :class:`Schema` object used internally.
* :meth:`Schema.python_types` - the plain ``{name: type}`` mapping the PRD
  writes out literally, kept so the spec can be checked against the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import pandas as pd

from src.core.constants import (
    ALLOWED_STATUS,
    DATE_FIELDS,
    FIELD_ORDER,
    FLOAT_FIELDS,
    KEY_FIELD,
    SCHEMA_VERSION,
    STRING_FIELDS,
)


class SchemaError(ValueError):
    """Raised when an input frame cannot be reconciled with the schema."""


class FieldType(str, Enum):
    """Logical type of a schema field."""

    STRING = "string"
    FLOAT = "float"
    DATE = "date"

    @property
    def python_type(self) -> type:
        """Return the Python type the PRD's schema literal names."""
        return {
            FieldType.STRING: str,
            FieldType.FLOAT: float,
            FieldType.DATE: datetime,
        }[self]

    @property
    def pandas_dtype(self) -> str:
        """Return the pandas dtype used for the cleaned column."""
        return {
            FieldType.STRING: "object",
            FieldType.FLOAT: "float64",
            FieldType.DATE: "datetime64[ns]",
        }[self]


class NullReason(str, Enum):
    """Why a cleaned cell ended up null.

    This distinction is the whole point of Stage 1. Collapsing these three
    causes into a bare ``None`` would destroy information Stage 2 needs:
    an *absent* date is a completeness defect, while an *unparseable* date is a
    hard temporal-coherence failure.
    """

    #: Cell carries a value and survived cleaning.
    PRESENT = "present"
    #: Cell was empty / NaN / NaT in the source.
    MISSING = "missing"
    #: Cell carried an absence token such as "N/A" or "0000-00-00".
    PLACEHOLDER = "placeholder"
    #: Cell carried a value that could not be coerced to the declared type.
    UNPARSEABLE = "unparseable"

    @property
    def is_null(self) -> bool:
        """True for every reason other than :attr:`PRESENT`."""
        return self is not NullReason.PRESENT


#: Column prefix under which per-field :class:`NullReason` values are stored.
NULL_REASON_PREFIX = "null_reason__"


def null_reason_column(field_name: str) -> str:
    """Return the diagnostic column name holding ``field_name``'s null reason."""
    return f"{NULL_REASON_PREFIX}{field_name}"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Declaration of a single schema field."""

    name: str
    dtype: FieldType
    nullable: bool = True
    description: str = ""
    #: Closed vocabulary, if the field has one (currently only ``status``).
    allowed_values: Optional[Tuple[str, ...]] = None
    #: Whether negative values constitute a violation (money columns).
    non_negative: bool = False

    @property
    def python_type(self) -> type:
        """Python type declared for this field."""
        return self.dtype.python_type


_FIELD_SPECS: Tuple[FieldSpec, ...] = (
    FieldSpec(
        name="work_id",
        dtype=FieldType.STRING,
        nullable=False,
        description="Unique identifier for the sanctioned work.",
    ),
    FieldSpec(
        name="work_name",
        dtype=FieldType.STRING,
        description="Free-text description of the work.",
    ),
    FieldSpec(
        name="district",
        dtype=FieldType.STRING,
        description="District in which the work is located.",
    ),
    FieldSpec(
        name="state",
        dtype=FieldType.STRING,
        description="State in which the work is located.",
    ),
    FieldSpec(
        name="sanction_amount",
        dtype=FieldType.FLOAT,
        non_negative=True,
        description="Approved cost in INR.",
    ),
    FieldSpec(
        name="amount_spent",
        dtype=FieldType.FLOAT,
        non_negative=True,
        description="Reported expenditure in INR.",
    ),
    FieldSpec(
        name="date_proposal",
        dtype=FieldType.DATE,
        description="Date the work was proposed.",
    ),
    FieldSpec(
        name="date_approval",
        dtype=FieldType.DATE,
        description="Date the work was sanctioned.",
    ),
    FieldSpec(
        name="date_completion",
        dtype=FieldType.DATE,
        description="Date the work was reported complete.",
    ),
    FieldSpec(
        name="implementing_agency",
        dtype=FieldType.STRING,
        description="Agency responsible for execution.",
    ),
    FieldSpec(
        name="vendor_name",
        dtype=FieldType.STRING,
        description="Contractor or vendor engaged.",
    ),
    FieldSpec(
        name="status",
        dtype=FieldType.STRING,
        allowed_values=ALLOWED_STATUS,
        description="Lifecycle stage of the work.",
    ),
)


@dataclass(frozen=True)
class Schema:
    """Ordered, typed collection of :class:`FieldSpec` objects."""

    version: str = SCHEMA_VERSION
    specs: Tuple[FieldSpec, ...] = field(default=_FIELD_SPECS)

    def __post_init__(self) -> None:
        declared = tuple(spec.name for spec in self.specs)
        if declared != FIELD_ORDER:
            raise SchemaError(
                "Schema field order drifted from constants.FIELD_ORDER: "
                f"{declared!r} != {FIELD_ORDER!r}"
            )

    # -- lookups -----------------------------------------------------------

    def __iter__(self) -> Iterator[FieldSpec]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, name: object) -> bool:
        return name in self.fields

    @property
    def fields(self) -> Mapping[str, FieldSpec]:
        """Field specs keyed by name, in declaration order."""
        return {spec.name: spec for spec in self.specs}

    @property
    def names(self) -> Tuple[str, ...]:
        """Field names in declaration order."""
        return tuple(spec.name for spec in self.specs)

    def spec(self, name: str) -> FieldSpec:
        """Return the spec for ``name``.

        Raises:
            SchemaError: If the field is not part of the schema.
        """
        try:
            return self.fields[name]
        except KeyError as exc:
            raise SchemaError(f"Unknown field {name!r}") from exc

    def names_of_type(self, dtype: FieldType) -> Tuple[str, ...]:
        """Field names carrying the given logical type."""
        return tuple(spec.name for spec in self.specs if spec.dtype is dtype)

    @property
    def string_fields(self) -> Tuple[str, ...]:
        """Names of all text fields."""
        return self.names_of_type(FieldType.STRING)

    @property
    def float_fields(self) -> Tuple[str, ...]:
        """Names of all numeric fields."""
        return self.names_of_type(FieldType.FLOAT)

    @property
    def date_fields(self) -> Tuple[str, ...]:
        """Names of all date fields."""
        return self.names_of_type(FieldType.DATE)

    @property
    def key_field(self) -> str:
        """Name of the identifier field."""
        return KEY_FIELD

    def python_types(self) -> Dict[str, type]:
        """Return the plain mapping written literally in Stage1.md sec.3.4."""
        return {spec.name: spec.python_type for spec in self.specs}

    def pandas_dtypes(self) -> Dict[str, str]:
        """Return the pandas dtype each cleaned column must carry."""
        return {spec.name: spec.dtype.pandas_dtype for spec in self.specs}

    def read_dtypes(self) -> Dict[str, str]:
        """Dtypes used when reading raw text sources.

        Every column is read as ``str`` so pandas cannot silently coerce
        ``"00123"`` into ``123`` or guess a date format on our behalf.
        """
        return {spec.name: "string" for spec in self.specs}

    # -- frame reconciliation ---------------------------------------------

    def align(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Project ``frame`` onto the schema's columns, in schema order.

        Args:
            frame: Raw input frame.

        Returns:
            ``(aligned, report)``. ``aligned`` holds exactly the schema columns
            in declaration order. ``report`` records dropped extras and any
            nullable columns that had to be synthesised.

        Raises:
            SchemaError: If a non-nullable column (the key) is absent.
        """
        present = set(frame.columns)
        missing = [name for name in self.names if name not in present]
        extra = sorted(present.difference(self.names))

        blocking = [name for name in missing if not self.spec(name).nullable]
        if blocking:
            raise SchemaError(
                "Input is missing required column(s) "
                f"{blocking!r}; found columns {sorted(present)!r}"
            )

        aligned = frame.copy()
        for name in missing:
            aligned[name] = pd.Series([None] * len(aligned), index=aligned.index,
                                      dtype="object")
        aligned = aligned.loc[:, list(self.names)]

        report: Dict[str, Any] = {
            "dropped_columns": extra,
            "synthesised_columns": missing,
        }
        return aligned, report

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable description of the schema."""
        return {
            "version": self.version,
            "fields": [
                {
                    "name": spec.name,
                    "type": spec.dtype.value,
                    "python_type": spec.python_type.__name__,
                    "nullable": spec.nullable,
                    "non_negative": spec.non_negative,
                    "allowed_values": list(spec.allowed_values)
                    if spec.allowed_values
                    else None,
                    "description": spec.description,
                }
                for spec in self.specs
            ],
        }


#: Canonical schema instance used everywhere in Stage 1.
SCHEMA = Schema()

# Belt-and-braces: the constants module and the schema must agree.
assert SCHEMA.string_fields == STRING_FIELDS
assert SCHEMA.float_fields == FLOAT_FIELDS
assert SCHEMA.date_fields == DATE_FIELDS


@dataclass(frozen=True, slots=True)
class Record:
    """A single strongly-typed, cleaned, validated work record.

    ``slots=True`` keeps materialising 50k of these cheap; ``frozen=True``
    guarantees no downstream stage can mutate ingested evidence in place.

    Null-valued fields are genuinely ``None``; ``null_reasons`` says *why*.
    """

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
    is_valid: bool = True
    issues: Tuple[str, ...] = ()
    null_reasons: Mapping[str, str] = field(default_factory=dict)

    def value(self, field_name: str) -> Any:
        """Return the cleaned value of ``field_name``."""
        if field_name not in FIELD_ORDER:
            raise SchemaError(f"Unknown field {field_name!r}")
        return getattr(self, field_name)

    def null_reason(self, field_name: str) -> NullReason:
        """Return why ``field_name`` is null, or :attr:`NullReason.PRESENT`."""
        return NullReason(self.null_reasons.get(field_name, NullReason.PRESENT.value))

    def is_field_valid(self, field_name: str) -> bool:
        """Field-level validity predicate reused verbatim by Stage 2's C_comp.

        A field is valid when it is not null, not a placeholder, and carries the
        declared type.
        """
        return not self.null_reason(field_name).is_null

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view of the record."""
        payload: Dict[str, Any] = {}
        for name in FIELD_ORDER:
            value = getattr(self, name)
            if isinstance(value, datetime):
                payload[name] = value.date().isoformat()
            else:
                payload[name] = value
        payload["is_valid"] = self.is_valid
        payload["issues"] = list(self.issues)
        payload["null_reasons"] = dict(self.null_reasons)
        return payload
