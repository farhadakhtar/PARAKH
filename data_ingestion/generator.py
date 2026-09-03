"""Generate reproducible synthetic MPLADS/MGNREGA work records."""
import argparse
import csv
import json
import os
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

DISTRICTS = [f"D{i:03d}" for i in range(1, 26)]
AGENCIES = [f"AGENCY_{i:02d}" for i in range(1, 13)]
CONTRACTORS = [f"CONTRACTOR_{i:03d}" for i in range(1, 41)]
CATEGORIES = ["road", "water", "sanitation", "school", "health", "irrigation", "community hall"]
CEILINGS = {"MPLADS": 5000000.0, "MGNREGA": 2500000.0}


def generate_records(count=2000, seed=42):
    fake = Faker("en_IN")
    Faker.seed(seed)
    random.seed(seed)
    records = []
    # Popular contractors are legitimate high-volume participants; planted contractors are narrower.
    popular = CONTRACTORS[:5]
    planted = CONTRACTORS[5:8]
    duplicate_templates = [fake.sentence(nb_words=5) for _ in range(max(1, count // 100))]
    for i in range(count):
        scheme = random.choice(list(CEILINGS))
        district = random.choice(DISTRICTS)
        agency = random.choice(AGENCIES)
        contractor = random.choices(CONTRACTORS, weights=[8 if c in popular else 3 for c in CONTRACTORS])[0]
        ceiling = CEILINGS[scheme]
        sanctioned = round(random.uniform(150000, ceiling * .92), 2)
        estimate = round(sanctioned * random.uniform(.82, .99), 2)
        sanction_date = date.today() - timedelta(days=random.randint(30, 1500))
        status = random.choices(["COMPLETED", "IN_PROGRESS", "SANCTIONED"], [55, 30, 15])[0]
        completion = sanction_date + timedelta(days=random.randint(30, 500)) if status == "COMPLETED" else None
        record = {
            "work_id": f"WORK-{i+1:07d}", "mp_id": f"MP-{random.randint(1, 250):04d}",
            "constituency_id": f"C-{random.randint(1, 550):04d}", "district_id": district,
            "implementing_agency_id": agency, "contractor_id": contractor,
            "description": random.choice(duplicate_templates) if i % 37 == 0 else fake.sentence(nb_words=7),
            "work_category": random.choice(CATEGORIES), "sanctioned_amount": sanctioned,
            "cost_estimate": estimate, "sanction_date": sanction_date.isoformat(),
            "work_status": status, "completion_date": completion.isoformat() if completion else None,
            "payment_id": f"PAY-{i+1:07d}",
            "utilization_certificate_status": random.choice(["PENDING", "FILED", "ACCEPTED"]),
            "location": f"{fake.city()}, {district}",
        }
        truth = []
        # Explicit, deterministic planted patterns (about 6% total, with overlap allowed).
        if i < max(1, int(count * .06)):
            pattern = i % 6
            if pattern == 0:  # near duplicate, different MP and same period
                record["description"] = duplicate_templates[i % len(duplicate_templates)]
                record["cost_estimate"] = round(sanctioned * .94, 2)
                truth.append("duplicate_near_duplicate")
            elif pattern == 1:  # unrelated districts share a small contractor pool
                record["contractor_id"] = planted[i % len(planted)]
                record["district_id"] = DISTRICTS[i % 20]
                truth.append("contractor_concentration")
            elif pattern == 2:
                record["cost_estimate"] = round(ceiling * random.uniform(.995, 1.0), 2)
                truth.append("cost_ceiling_clustering")
            elif pattern == 3:
                record["completion_date"] = (sanction_date - timedelta(days=1)).isoformat()
                record["utilization_certificate_status"] = "FILED_BEFORE_PAYMENT"
                truth.append("timeline_anomaly")
            elif pattern == 4:
                record["implementing_agency_id"] = AGENCIES[0]
                truth.append("agency_volume_deviation")
            else:
                record["implementing_agency_id"] = AGENCIES[1]
                record["contractor_id"] = planted[0]
                record["district_id"] = DISTRICTS[i % 3]
                truth.append("connected_suspicious_cluster")
        record["ground_truth"] = truth
        record["confidence_state"] = {k: ("OBSERVED" if k in {"sanctioned_amount", "cost_estimate", "sanction_date"} else "SELF_CERTIFIED" if k in {"work_status", "completion_date", "utilization_certificate_status"} else "INFERRED") for k in record if k not in {"ground_truth", "confidence_state"}}
        records.append(record)
    return records


def write_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) + ["confidence_state"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            output = dict(row)
            output["ground_truth"] = ""  # never expose truth in the scoring-facing CSV
            output["confidence_state"] = json.dumps(output["confidence_state"], sort_keys=True)
            writer.writerow(output)


def insert_postgres(records):
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is required for database output")
    import psycopg2
    with psycopg2.connect(url) as conn, conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS works (work_id TEXT PRIMARY KEY, data JSONB NOT NULL, ground_truth JSONB NOT NULL)""")
        for record in records:
            cur.execute("INSERT INTO works (work_id, data, ground_truth) VALUES (%s, %s, %s) ON CONFLICT (work_id) DO NOTHING", (record["work_id"], json.dumps({k: v for k, v in record.items() if k != "ground_truth"}), json.dumps(record["ground_truth"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    rows = generate_records(args.count)
    write_csv(rows, Path(__file__).parent / "output" / "works_sample.csv")
    insert_postgres(rows)
    print(f"Generated {len(rows)} records")
