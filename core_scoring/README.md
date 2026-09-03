# Core Scoring

This module computes record-level confidence scores for PARAKH work records.

## Components

- `schema.py` defines `WorkRecord`, the typed representation of a row read from Postgres, including its `confidence_state` provenance tags.
- `confidence.py` implements:
  - `completeness(record)`: weighted required-field coverage. Critical financial/date fields carry higher weights.
  - `temporal_coherence(record)`: product of logistic scores over causal date orderings: sanction to completion, completion to UC submission, and UC submission to payment release. Missing pairs are excluded.
  - `source_agreement(record)`: temporary provenance-based proxy that discounts records with more `SELF_CERTIFIED` or `INFERRED` fields. This should later be replaced by real cross-source reconciliation.
  - `confidence(record)`: product of completeness, temporal coherence, and source agreement.
- `run_confidence_batch.py` reads all `works` rows, scores each record, and upserts results into `confidence_scores`.

## Run

Set `DATABASE_URL` to the same Postgres URL used by Docker Compose, then run:

```bash
export DATABASE_URL=postgresql://parakh:parakh_dev@localhost:5432/parakh
python core_scoring/run_confidence_batch.py
```

On Windows PowerShell:

```powershell
$env:DATABASE_URL = "postgresql://parakh:parakh_dev@localhost:5432/parakh"
python core_scoring/run_confidence_batch.py
```

The script prints the mean confidence and the number of records below `0.5`, which are candidates for a data-quality audit queue.
