"""Stage 1 - Data Ingestion & Schema Layer.

Public surface:

* :func:`~src.stage1.data_generator.generate_dataset` - deterministic dirty data
* :class:`~src.stage1.corpus.Corpus` - the ingested, cleaned, validated corpus
* :class:`~src.stage1.schema.Schema` / :class:`~src.stage1.schema.Record`
* :class:`~src.stage1.validation.ValidationReport`
"""

from src.stage1.cleaning import CleaningResult, clean_frame
from src.stage1.corpus import Corpus, CorpusMetadata
from src.stage1.data_generator import (
    DefectLedger,
    GenerationConfig,
    GenerationResult,
    generate_dataset,
    generate_with_ledger,
    save_dataset,
)
from src.stage1.ingestion import (
    IngestionError,
    IngestionResult,
    read_csv,
    read_dataframe,
    read_parquet,
    write_csv,
    write_parquet,
)
from src.stage1.schema import (
    SCHEMA,
    FieldSpec,
    FieldType,
    NullReason,
    Record,
    Schema,
    SchemaError,
)
from src.stage1.validation import (
    IssueCode,
    Severity,
    ValidationOutcome,
    ValidationReport,
    validate,
)

__all__ = [
    "SCHEMA",
    "CleaningResult",
    "Corpus",
    "CorpusMetadata",
    "DefectLedger",
    "FieldSpec",
    "FieldType",
    "GenerationConfig",
    "GenerationResult",
    "IngestionError",
    "IngestionResult",
    "IssueCode",
    "NullReason",
    "Record",
    "Schema",
    "SchemaError",
    "Severity",
    "ValidationOutcome",
    "ValidationReport",
    "clean_frame",
    "generate_dataset",
    "generate_with_ledger",
    "read_csv",
    "read_dataframe",
    "read_parquet",
    "save_dataset",
    "validate",
    "write_csv",
    "write_parquet",
]
