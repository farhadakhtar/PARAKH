"""PARAKH Stage 1 entry point.

Runs the pipeline end to end:

    Stage 1: generate -> persist -> ingest -> clean -> validate -> report
    Stage 2: score confidence -> attach to corpus -> report

Usage::

    python main.py                          # 20,000 synthetic records, seed 42
    python main.py --n 50000 --seed 7       # different size / seed
    python main.py --input data/real.csv    # ingest a real file instead
    python main.py --input data/real.parquet

Every artefact is written under ``data/`` and ``outputs/``. Nothing printed
here depends on wall-clock time, so two runs with the same arguments produce
identical output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.constants import (
    DATA_DIR,
    DEFAULT_HEAD_ROWS,
    DEFAULT_N_RECORDS,
    DEFAULT_SEED,
    OUTPUT_DIR,
    SYNTHETIC_CSV_NAME,
)
from src.core.logger import configure_logging, get_logger
from src.stage1.corpus import Corpus
from src.stage1.data_generator import generate_with_ledger, save_dataset
from src.stage2.confidence import (
    ConfidenceModel,
    attach_confidence,
    confidence_summary_frame,
)
from src.stage3.pipeline import STAGE3_COLUMNS, SemanticLayer, attach_structure
from src.utils.helpers import ensure_dir

LOGGER = get_logger(__name__)

_RULE = "=" * 78


def _banner(title: str) -> None:
    """Print a section header."""
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="parakh-stage1",
        description="PARAKH Stage 1 - Data Ingestion & Schema Layer",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N_RECORDS,
        help=f"Records to synthesise (default: {DEFAULT_N_RECORDS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Ingest this .csv/.parquet instead of generating data.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="Dataset directory."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Report directory."
    )
    parser.add_argument(
        "--head",
        type=int,
        default=DEFAULT_HEAD_ROWS,
        help=f"Rows to preview (default: {DEFAULT_HEAD_ROWS}).",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print reports without writing files."
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Stop after Stage 1; skip confidence scoring.",
    )
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Stop after Stage 2; skip peer structure.",
    )
    return parser


def load_corpus(path: Path) -> Corpus:
    """Ingest a file, dispatching on its extension.

    Args:
        path: A ``.csv`` or ``.parquet`` file.

    Returns:
        The ingested :class:`~src.stage1.corpus.Corpus`.

    Raises:
        SystemExit: If the extension is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return Corpus.from_csv(path)
    if suffix in {".parquet", ".pq"}:
        return Corpus.from_parquet(path)
    raise SystemExit(f"Unsupported input type {suffix!r}; expected .csv or .parquet")


def run(
    n: int = DEFAULT_N_RECORDS,
    seed: int = DEFAULT_SEED,
    input_path: Optional[Path] = None,
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    head_rows: int = DEFAULT_HEAD_ROWS,
    save: bool = True,
    run_stage2: bool = True,
    run_stage3: bool = True,
) -> Corpus:
    """Execute the Stage 1 pipeline and print its reports.

    Args:
        n: Records to synthesise when ``input_path`` is None.
        seed: Deterministic seed for generation.
        input_path: Existing dataset to ingest instead of generating.
        data_dir: Where the synthetic dataset and ledger are written.
        output_dir: Where JSON reports and the clean dataset are written.
        head_rows: Rows to preview.
        save: When False, nothing is written to disk.
        run_stage2: Also score evidentiary confidence.
        run_stage3: Also build peer structure.

    Returns:
        The constructed corpus.
    """
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    started = time.perf_counter()

    if input_path is None:
        _banner("STEP 1/5  SYNTHETIC DATA GENERATION")
        generation = generate_with_ledger(n=n, seed=seed)
        print(f"Generated {len(generation.frame):,} records (seed={seed}).")
        print("Injected noise channels:")
        for label, count in sorted(generation.ledger.channel_counts.items()):
            if count:
                print(f"  {label:<46} {count:>8,}")
        print(
            f"\nRows carrying >=1 injected defect: "
            f"{generation.ledger.n_defective_rows:,}"
        )
        print("\nSelf-check against the PRD noise budget:")
        for key in (
            "missing_cell_rate",
            "date_violation_rate",
            "duplicate_work_name_rate",
            "cost_outlier_rate",
        ):
            band_key = key.replace("_rate", "_rate_in_band").replace(
                "duplicate_work_name_rate_in_band", "duplicate_work_name_in_band"
            )
            value = generation.observed.get(key)
            in_band = generation.observed.get(band_key)
            mark = "OK " if in_band else ("-- " if in_band is None else "OUT")
            print(f"  [{mark}] {key:<28} {value:.4f}")

        if save:
            paths = save_dataset(generation, data_dir=data_dir)
            print(f"\nDataset -> {paths['csv']}")
            print(f"Ledger  -> {paths['ledger']}  (ground truth; NOT a corpus column)")
            source_path = paths["csv"]
        else:
            source_path = None

        _banner("STEP 2/5  INGESTION")
        if source_path is not None:
            print(f"Reading {source_path}")
            corpus = Corpus.from_csv(source_path)
        else:
            print("Ingesting the generated frame in memory (--no-save).")
            corpus = Corpus.from_dataframe(generation.frame)
    else:
        _banner("STEP 1/5  SYNTHETIC DATA GENERATION  [skipped: --input given]")
        _banner("STEP 2/5  INGESTION")
        print(f"Reading {input_path}")
        corpus = load_corpus(input_path)

    print(f"Ingested {len(corpus):,} records; {corpus.metadata.n_fields} fields.")
    if corpus.metadata.ingestion_errors:
        print(f"Ingestion errors: {len(corpus.metadata.ingestion_errors)}")

    _banner("STEP 3/5  SAMPLE INSPECTION  corpus.head()")
    with pd.option_context("display.max_colwidth", 34):
        print(corpus.head(head_rows).to_string())

    _banner("STEP 4/5  REPORTS")
    print("--- corpus.summary() ---")
    print(json.dumps(corpus.summary(), indent=2, default=str))

    print("\n--- validation_report (PRD sec.5.2 shape) ---")
    print(json.dumps(corpus.validation_report.prd_view(), indent=2))

    print("\n--- corpus.missing_report() ---")
    print(corpus.missing_report().to_string())

    print("\n--- corpus.describe() ---")
    print(corpus.describe().to_string())

    print("\n--- validity views (no rows were dropped) ---")
    report = corpus.validation_report
    print(f"  corpus rows          : {len(corpus):,}")
    print(f"  valid_records view   : {len(corpus.valid_records):,}")
    print(f"  invalid_records view : {len(corpus.invalid_records):,}")
    print(
        f"  sum of views         : "
        f"{len(corpus.valid_records) + len(corpus.invalid_records):,}  "
        f"(equals corpus rows: "
        f"{len(corpus.valid_records) + len(corpus.invalid_records) == len(corpus)})"
    )
    print(f"\n  top issues by frequency:")
    for entry in corpus.summary()["top_issues"]:
        print(f"    {entry['issue']:<46} {entry['count']:>8,}")

    _banner("STEP 5/5  PERSISTENCE")
    if save:
        ensure_dir(output_dir)
        written = corpus.save_reports(output_dir)
        clean_path = corpus.save_clean_csv(output_dir / "stage1_clean_dataset.csv")
        for name, path in written.items():
            print(f"  {name:<20} -> {path}")
        print(f"  {'clean_dataset':<20} -> {clean_path}")
    else:
        print("  --no-save: nothing written.")

    elapsed = time.perf_counter() - started
    print(f"\nStage 1 complete in {elapsed:.2f}s for {len(corpus):,} records.")

    if run_stage2:
        _run_stage2(corpus, output_dir=output_dir, head_rows=head_rows, save=save)
        if run_stage3:
            _run_stage3(
                corpus, output_dir=output_dir, head_rows=head_rows, save=save
            )

    return corpus


def _run_stage3(
    corpus: Corpus, output_dir: Path, head_rows: int, save: bool
) -> None:
    """Build peer structure and print the Stage 3 reports.

    Args:
        corpus: A corpus already carrying the Stage 2 breakdown.
        output_dir: Where the Stage 3 artefacts are written.
        head_rows: How many enriched records to preview.
        save: When False, nothing is written to disk.
    """
    _banner("STAGE 3  SEMANTIC LAYER & PEER CELL FORMATION")
    result = attach_structure(corpus)
    report = result.report()
    records = corpus.records

    print(
        f"Structured {len(result):,} records in {result.elapsed_seconds:.2f}s; "
        f"{len(STAGE3_COLUMNS)} contract columns."
    )

    print("\n--- semantic clusters, labelled by their own top TF-IDF terms ---")
    print(
        f"  {report['embedding']['n_unique']} distinct normalised names"
        f" -> {report['embedding']['n_terms']} terms"
        f" -> {report['embedding']['n_components']} dims"
    )
    sizes = records["cluster_id"].value_counts()
    for cluster_id, label in sorted(result.clusters.labels.items()):
        print(f"    {cluster_id:>3}  n={int(sizes.get(cluster_id, 0)):>6,}  {label}")
    print(f"    unclustered: {report['clustering']['noise_pct']}% of records")

    print("\n--- peer cells: (cluster k) x (cost stratum s) ---")
    print(
        f"  {report['peer_cells']['n_cells']} cells,"
        f" {report['peer_cells']['n_stable_cells']} stable"
        f" (>= {result.config.peer_cell_min_size} records), covering"
        f" {report['peer_cells']['stable_record_pct']}% of records"
    )
    print(f"  cost strata: {report['stratification']['counts']}")

    print("\n--- confidence gating of the peer norms ---")
    print(
        f"  {report['peer_statistics']['reference_record_pct']}% of records may shape"
        f" a norm (confidence >= {result.config.min_confidence}, usable amounts)"
    )
    print(
        f"  {report['peer_statistics']['usable_cells']} cell(s) carry enough"
        " high-confidence members to define one"
    )

    print("\n--- deviations from peer norms (raw material for Stage 4, not scores) ---")
    for name, entry in report["deviations"]["deviations"].items():
        print(
            f"  {name:<24} defined {entry['defined_pct']:>6.2f}%"
            f"  |p95|={entry['abs_p95']}  {entry['reasons']}"
        )

    print("\n--- near-duplicate candidates ---")
    print(
        f"  {report['duplicates']['n_flagged']} record(s) in"
        f" {report['duplicates']['n_groups']} group(s)"
    )
    grouped = records[records["duplicate_group_id"] >= 0]
    if len(grouped):
        first_id = grouped["duplicate_group_id"].iloc[0]
        example = grouped[grouped["duplicate_group_id"] == first_id]
        with pd.option_context("display.max_colwidth", 62):
            print(
                example[
                    ["work_name", "district", "date_proposal", "duplicate_score"]
                ].to_string(index=False)
            )

    print("\n--- sample enriched records ---")
    columns = [
        "work_id",
        "cluster_id",
        "cost_stratum",
        "peer_cell_id",
        "peer_cell_stable",
        "peer_reference",
        "deviation_cell_cost",
        "deviation_cluster_cost",
        "duplicate_score",
    ]
    print(
        records.loc[records["peer_cell_stable"], columns]
        .head(head_rows)
        .to_string(index=False)
    )

    if save:
        ensure_dir(output_dir)
        for name, path in result.save_reports(output_dir).items():
            print(f"\n  {name:<20} -> {path}")
        structure_path = output_dir / "stage3_structure.csv"
        result.frame.to_csv(structure_path, index=False, lineterminator="\n")
        print(f"  {'structure':<20} -> {structure_path}")
    else:
        print("\n  --no-save: nothing written.")

    print(
        f"\nStage 3 complete in {result.elapsed_seconds:.2f}s"
        f" for {len(result):,} records."
    )


def _run_stage2(
    corpus: Corpus, output_dir: Path, head_rows: int, save: bool
) -> None:
    """Score evidentiary confidence and print the Stage 2 reports.

    Args:
        corpus: The Stage 1 corpus, scored and annotated in place.
        output_dir: Where the Stage 2 reports are written.
        head_rows: How many extreme records to preview at each end.
        save: When False, nothing is written to disk.
    """
    _banner("STAGE 2  EVIDENTIARY CONFIDENCE ENGINE")
    started = time.perf_counter()
    model = ConfidenceModel()
    result = model.score(corpus)
    attach_confidence(corpus, result)
    elapsed = time.perf_counter() - started

    weights = model.config.weights
    print(f"Scored {len(result):,} records in {elapsed:.2f}s.")
    print(
        f"Weights: w_comp={weights[0]:.4f} w_temp={weights[1]:.4f} "
        f"w_recon={weights[2]:.4f}   (geometric mean, computed in log space)"
    )

    print("\n--- distribution ---")
    print(confidence_summary_frame(result).to_string())

    print("\n--- report (PRD sec.10.1 shape) ---")
    print(json.dumps(result.report.prd_view(), indent=2))

    print("\n--- histogram (sec.10.2) ---")
    for band, count in result.report.histogram.items():
        bar = "#" * int(60 * count / max(len(result), 1))
        print(f"  {band:<12} {count:>7,}  {bar}")

    print("\n--- components unmeasurable, hence dropped rather than scored 1.0 ---")
    for name, pct in result.report.components_dropped_pct.items():
        print(f"  {name:<30} {pct:>6.2f}%")

    field_weights = result.field_weights
    print("\n--- completeness field weights v_f ---")
    print(f"  entropy normalisation : {field_weights.normalization}")
    print(
        f"  structural floor      : {field_weights.structural_floor:.4f}"
        "   (contributed by fields that are never null)"
    )
    for name, share in sorted(field_weights.shares.items(), key=lambda kv: -kv[1]):
        print(
            f"    {name:<22} v={field_weights.weights[name]:.4f}"
            f"  share={100 * share:5.2f}%"
            f"  coverage={100 * field_weights.coverage[name]:6.2f}%"
        )

    columns = ["work_id", "confidence", "completeness", "temporal", "reconciliation"]
    print("\n--- lowest confidence (REMEDIATE candidates) ---")
    print(corpus.records.nsmallest(head_rows, "confidence")[columns].to_string(index=False))
    print("\n--- highest confidence ---")
    print(corpus.records.nlargest(head_rows, "confidence")[columns].to_string(index=False))

    if save:
        ensure_dir(output_dir)
        for name, path in result.save_reports(output_dir).items():
            print(f"\n  {name:<20} -> {path}")
        breakdown_path = output_dir / "stage2_confidence_scores.csv"
        result.breakdown.to_csv(breakdown_path, index=False, lineterminator="\n")
        print(f"  {'confidence_scores':<20} -> {breakdown_path}")
    else:
        print("\n  --no-save: nothing written.")

    print(f"\nStage 2 complete in {elapsed:.2f}s for {len(result):,} records.")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        run(
            n=args.n,
            seed=args.seed,
            input_path=args.input,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            head_rows=args.head,
            save=not args.no_save,
            run_stage2=not args.stage1_only,
            run_stage3=not (args.stage1_only or args.stage2_only),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        LOGGER.exception("Stage 1 failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
