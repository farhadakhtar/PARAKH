# Synthetic data generator

From the repository root, install the backend requirements and run:

```bash
python data_ingestion/generator.py --count 2000
```

Use any larger count (for example `--count 10000`) to scale generation. The generator writes `output/works_sample.csv` and inserts records into the Postgres database selected by `DATABASE_URL`. It creates the `works` table if needed.

Each record includes a `confidence_state` mapping. Sanction and cost fields are `OBSERVED`; agency-reported status, completion, and utilization fields are `SELF_CERTIFIED`; identifiers and derived values are `INFERRED`.

`ground_truth` contains the planted anomaly labels and is reserved exclusively for later evaluation. It is stored separately in Postgres and is blank in the CSV backup so it is never exposed to a scoring pipeline.
