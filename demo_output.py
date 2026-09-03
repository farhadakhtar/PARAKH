"""
PARAKH Stage 1 — Demonstration Output

This script runs the full pipeline and outputs:
1. Complete Python implementation (imports and class structure)
2. Synthetic dataset generation code (already in generator.py)
3. Sample output of 5 records from the Corpus
4. Validation summary printout

Run: python /tmp/parakh_stage1/demo_output.py
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
from pipeline import pipeline


def main():
    t0 = time.time()

    # Generate corpus with 10000 records (default is 10000)
    print("=" * 60)
    print("PARAKH Stage 1 — Full Pipeline Execution")
    print("=" * 60)

    print("\n--- Generating corpus with 10000 records ---")
    corpus = pipeline(n_records=10_000)

    t1 = time.time()
    elapsed = t1 - t0
    print(f"--- Pipeline completed in {elapsed:.3f} seconds ---\n")

    # 3. Sample output of 5 records
    print("--- Sample of 5 records from Corpus ---")
    for i, rec in enumerate(corpus.records[:5]):
        # Format output nicely
        sid = rec.work_id
        wname = rec.work_name[:40] if rec.work_name else "None"
        dist = rec.district[:15] if rec.district else "None"
        state = rec.state[:10] if rec.state else "None"
        vendor = rec.vendor_name[:30] if rec.vendor_name else "None"
        sa = rec.sanction_amount
        ar = rec.amount_released
        au = rec.amount_utilized
        ds = rec.date_sanction
        dstart = rec.date_start
        dcomp = rec.date_completion
        cat = rec.work_category
        # validity info from summary
        print(
            f"  Record {i+1}: work_id={sid}, work_name={wname}, "
            f"district={dist}, state={state}, vendor={vendor}, "
            f"sanction_amount={sa}, amount_released={ar}, "
            f"amount_utilized={au}, date_sanction={ds}, "
            f"date_start={dstart}, date_completion={dcomp}, "
            f"category={cat}"
        )

    # 4. Validation summary printout
    print("\n--- Validation Summary ---")
    vs = corpus.validation_summary
    print(f"  Total records: {vs.get('total_records', 'N/A')}")
    print(f"  Valid records: {vs.get('valid_records', 'N/A')} ({vs.get('valid_percentage', 0):.1f}%)")
    print(f"  Invalid records: {vs.get('invalid_records', 'N/A')} ({vs.get('invalid_percentage', 0):.1f}%)")
    print(f"  Missing fields count: {vs.get('missing_fields_count', 'N/A')}")
    print(f"  Missing fields %: {vs.get('missing_fields_percentage', 0):.2f}%")
    print(f"  Category distribution: {vs.get('category_distribution', {})}")
    print(f"  Category percentages: {vs.get('category_percentages', {})}")
    print(f"  Sample errors (first 10): {vs.get('errors', [])[:10]}")
    print(f"  Sample warnings (first 10): {vs.get('warnings', [])[:10]}")

    # Non-functional checks
    print("\n--- Non-Functional Checks ---")
    # No NaN or infinite values
    has_nan = any(
        rec.sanction_amount != rec.sanction_amount  # NaN check
        or rec.amount_released != rec.amount_released
        or rec.amount_utilized != rec.amount_utilized
        for rec in corpus.records
        if rec.sanction_amount is not None
        and isinstance(rec.sanction_amount, float)
    )
    print(f"  No NaN values in sanctions: {not has_nan}")

    # All records stored (including invalid)
    print(f"  Corpus stores ALL records (valid + invalid): {corpus.total_records == 10_000}")

    # Deterministic: running again with same seed should produce same results
    # (We use fixed seed inside pipeline; re-running may not be identical due to
    #  pipeline internal state, but the seed is set at RANDOM_SEED=42)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()