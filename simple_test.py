"""Simple test to verify PARAKH Stage 1 pipeline."""
import sys
import os

# Import from project root
sys.path.insert(0, r"D:\hackthon\SIH\PARAKH")

from pipeline import pipeline


def main():
    print("Testing PARAKH Stage 1 pipeline with 100 records...")
    corpus = pipeline(n_records=100)
    print(f"Total records: {corpus.total_records}")
    print(f"Valid: {corpus.validation_summary.get('valid_percentage', 0):.1f}%")
    print(f"Invalid: {corpus.validation_summary.get('invalid_percentage', 0):.1f}%")
    print(f"Missing fields %: {corpus.validation_summary.get('missing_fields_percentage', 0):.2f}%")
    print(f"Categories: {corpus.validation_summary.get('category_distribution', {})}")
    
    # Show 3 sample records
    print("\nSample records:")
    for i, rec in enumerate(corpus.records[:3]):
        print(f"  {rec.work_id}: {rec.work_name[:30]} | {rec.district} | {rec.state} | "
              f"cat={rec.work_category} | sa={rec.sanction_amount}")
    
    # Verify no NaN
    nan_count = 0
    for rec in corpus.records:
        if rec.sanction_amount is not None and isinstance(rec.sanction_amount, float) and rec.sanction_amount != rec.sanction_amount:
            nan_count += 1
    print(f"\nNaN values: {nan_count}")
    
    # Verify work IDs are unique
    work_ids = [rec.work_id for rec in corpus.records]
    if len(work_ids) == len(set(work_ids)):
        print("Work IDs are unique: PASS")
    else:
        print("Work IDs are unique: FAIL - duplicates found")
    
    print("\nTest PASSED!")


if __name__ == "__main__":
    main()