"""
PARAKH Stage 1 — Synthetic Data Generator

Generates realistic synthetic MPLADS-like work records with controllable
distribution: 70% normal, 20% noisy, 10% anomalous.

Deterministic when seed is fixed.

Rules:
- district: 20 to 50 distinct values
- vendor_name: skewed distribution (few dominant vendors via Zipf-like weighting)
- work_category: 5 to 10 categories
- sanction_amount: log normal distribution, scaled per record type
- dates: spanning multiple years (2019–2025)
- Normal: clean, reasonable costs, valid timelines
- Noisy: missing fields, inconsistent dates, partial payments
- Anomalous: inflated amounts (3-10x peer median), vendor dominance,
  duplicate/near-duplicate work names, temporal bursts (many in same month)
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List, Tuple, Any

# ---------------------------------------------------------------------------
# Configuration (adjustable; keep consistent for deterministic output)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "community hall",
    "borewell",
    "culvert",
    "road stretch",
    "school classroom",
    "health post",
    "irrigation channel",
    "market shed",
    "rural bridge",
    "power sub-station",
]

CATEGORY_COUNT = len(CATEGORIES)  # 10; we'll sample 5-10 from this

DISTRICTS = [
    f"District_{i:03d}" for i in range(1, 51)
]  # 50 districts; we'll sample 20-50

START_YEAR = 2019
END_YEAR = 2025

# vendor surnames for skewed distribution
VENDOR_SURNAMES = [
    "Kumar", "Sharma", "Patel", "Singh", "Reddy", "Naidu", "Yadav",
    "Roy", "Gupta", "Jain", "Ali", "Saifi", "Mondal", "Hossain",
    "Ahmed", "Hussain", "Singh", "Rawat", "Meena", "Chowdhury",
]

# Anomalous vendor names that create dominance patterns
ANOMALOUS_VENDORS = [
    "M/s Prime Constructions",
    "M/s Global Enterprises",
    "M/s Royal Traders",
    "M/s Heritage Works",
]

# Probability weights for vendor selection (skewed: few get most picks)
VENDOR_WEIGHTS = [
    0.25, 0.20, 0.15, 0.10, 0.08, 0.05, 0.05, 0.04, 0.04, 0.03,
    0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
]


def _random_date_in_range(
    start: datetime, end: datetime, rng: random.Random
) -> datetime:
    """Generate a random datetime between start and end (inclusive)."""
    delta_days = (end - start).days
    random_days = rng.randint(0, delta_days)
    return start + timedelta(days=random_days)


def _sample_district(rng: random.Random, num_districts: int) -> str:
    """Sample `num_districts` distinct district names from the pool."""
    sample_size = min(num_districts, len(DISTRICTS))
    return rng.sample(DISTRICTS, sample_size)


def _generate_sanction_amount(
    base_median: float, is_anomalous: bool, rng: random.Random, scale: float = 1.0
) -> float:
    """Generate sanction_amount.

    - Normal: log-normal around base_median
    - Anomalous: 3-10x the peer median (inflated)
    - Noisy: may be negative, zero, or extreme
    """
    import math
    sigma = 0.5  # log-normal sigma
    mu = math.log(base_median) if base_median > 0 else 0.0

    if is_anomalous:
        # Inflated: 3 to 10 times the peer median
        inflation_factor = rng.uniform(3.0, 10.0)
        return max(0.1, base_median * inflation_factor) * scale
    else:
        # Normal log-normal
        val = rng.lognormvariate(mu, sigma) * scale
        return max(0.1, val)


def _generate_vendor_name(is_anomalous: bool, rng: random.Random, is_noisy: bool = False) -> str | None:
    """Generate a vendor name.

    - Skewed distribution: most common surnames get picked most often
    - Anomalous: a few dominant vendor firm names
    - Noisy: may have punctuation, mixed case, odd characters
    """
    if is_anomalous:
        # Dominant vendors appear repeatedly across districts
        vendor_name = rng.choice(ANOMALOUS_VENDORS)
        # Occasionally add noise
        if rng.random() < 0.3:
            vendor_name += f"_{rng.randint(100, 999)}"
        return vendor_name

    # Normal vendor from skewed distribution
    if rng.random() < 0.7:
        # Use weighted surname
        surname = rng.choices(VENDOR_SURNAMES, weights=VENDOR_WEIGHTS, k=1)[0]
        # Add "Construction" or "Works" suffix
        suffix = rng.choice(["Construction", "Works", "Enterprises", "Industries"])
        return f"{surname} {suffix}"
    else:
        # Small local vendor
        return f"M/s Local Works_{rng.randint(1, 99)}"


def _generate_date_start(date_sanction: datetime, rng: random.Random) -> datetime | None:
    """Generate date_start after date_sanction.

    Returns None with 15% probability (noisy record pattern).
    """
    if rng.random() < 0.15:
        return None
    # Start should be after sanction, typically within 45 days per guidelines,
    # but we'll allow a wider window for synthetic data
    delta_max = timedelta(days=180)
    delta = rng.uniform(timedelta(days=1), delta_max)
    return date_sanction + delta


def _generate_date_completion(
    date_sanction: datetime, date_start: datetime | None, is_noisy: bool, rng: random.Random
) -> datetime | None:
    """Generate date_completion after date_start.

    - Noisy: may be None, or before date_start, or far in the future
    - Normal: always after date_start, within 365 days of sanction
    """
    if date_start is None:
        # 50% chance of None when start is None (noisy), 100% if we want some normals without completion
        if is_noisy and rng.random() < 0.5:
            return None
        # If start is None through other path, completion also None
        return None

    # Normal case: completion after start
    if not is_noisy:
        delta_max = timedelta(days=365)
        delta = rng.uniform(timedelta(days=1), delta_max)
        return date_start + delta

    # Noisy case: various bad patterns
    noisy_pattern = rng.choice([
        "none",  # None
        "before_start",  # before date_start
        "far_future",  # years after sanction
        "same_as_start",  # identical to start (unlikely but possible)
    ], p=[0.4, 0.2, 0.2, 0.2])

    if noisy_pattern == "none":
        return None
    elif noisy_pattern == "before_start":
        # completion before start — illogical
        delta = rng.uniform(timedelta(days=-365), timedelta(days=-1))
        return date_start + delta
    elif noisy_pattern == "far_future":
        delta = timedelta(days=rng.randint(365, 1825))  # 1-5 years after sanction
        return date_sanction + delta
    else:  # same_as_start
        return date_start


def _generate_temporal_burst_pattern(
    record_index: int, total_records: int, rng: random.Random
) -> bool:
    """Determine if this record participates in a temporal burst.

    About 10% of normal records will be clustered in a few months
    to simulate fiscal-year-end or monsoon-season bursts.
    """
    # Select a burst month (e.g., March for fiscal year-end, or August for monsoon)
    burst_months = [3, 8]  # March (fy-end), August (monsoon)
    burst_month = burst_months[rng.randint(0, len(burst_months) - 1)]

    # If this record's eventual approval month is the burst month, it participates
    # We'll influence this later when we set the approval month
    # For now, mark ~10% of records as burst participants
    participates = rng.random() < 0.10
    return participates


def generate_raw_records(
    n_records: int = 10_000,
    rng: random.Random | None = None,
) -> List[Tuple]:
    """Generate n_records raw synthetic MPLADS-like records.

    Distribution:
    - 70% normal (clean, valid)
    - 20% noisy (missing fields, inconsistent dates, partial payments)
    - 10% anomalous (inflated amounts, vendor dominance, duplicates, bursts)

    Returns list of tuples with fields matching the RecordSchema order.
    Each tuple: (work_id, work_name, district, state, vendor_name,
    sanction_amount, amount_released, amount_utilized,
    date_sanction, date_start, date_completion, work_category)
    """
    if rng is None:
        rng = random.Random(RANDOM_SEED)
    else:
        # override seed if desired but keep deterministic per call
        pass

    # ------------------------------------------------------------------
    # Pre-compute distributions
    # ------------------------------------------------------------------
    # Select 20-50 districts for this dataset
    num_districts = rng.randint(20, min(50, len(DISTRICTS)))
    available_districts = rng.sample(DISTRICTS, num_districts)

    # Select 5-10 work categories
    num_categories = rng.randint(5, min(10, CATEGORY_COUNT))
    available_categories = rng.sample(CATEGORIES, num_categories)

    # State values (simulate 5-10 states)
    states = [f"State_{i:02d}" for i in rng.sample(range(1, 20), rng.randint(5, 10))]

    # ------------------------------------------------------------------
    # Record generation
    # ------------------------------------------------------------------
    records: List[Tuple] = []

    # We'll track peer medians for anomaly inflation later
    # For now, compute base median after generating normal records
    # But we need a two-pass approach: first generate normals, compute median,
    # then generate normals/anomalous/noisy with knowledge of that median.

    # STRATEGY:
    # 1. Generate 70% n_normal normal records
    # 2. Compute median sanction_amount from normals
    # 3. Generate 20% n_noisy noisy records
    # 4. Generate 10% n_anomalous anomalous records using the median
    # 5. Shuffle all together

    n_normal = int(0.70 * n_records)
    n_noisy = int(0.20 * n_records)
    n_anomalous = n_records - n_normal - n_noisy  # remainder

    # --- Step 1: Generate normal records ---
    normal_records: List[Tuple] = []
    for _ in range(n_normal):
        # Work ID: unique
        work_id = f"WORK_{rng.randint(100_000_000, 999_999_999)}"

        # Work name: from categories, slightly varied
        base_cat = rng.choice(available_categories)
        # Add some variation
        work_name = f"{base_cat}_{rng.randint(1, 50)}"

        # District
        district = rng.choice(available_districts)

        # State
        state = rng.choice(states)

        # Vendor name (skewed, normal)
        vendor_name = _generate_vendor_name(is_anomalous=False, rng=rng)

        # Sanction amount: log-normal, reasonable
        # We'll use a base median that we'll refine after all normal records
        base_median_for_normals = 5_000_000  # 5 lakh base
        sanction_amount = _generate_sanction_amount(
            base_median_for_normals, is_anomalous=False, rng=rng
        )

        # Amount released: <= sanction, typically 80-100%
        released_pct = rng.uniform(0.8, 1.0)
        amount_released = sanction_amount * released_pct

        # Amount utilized: <= amount_released, typically 70-100%
        utilized_pct = rng.uniform(0.7, 1.0)
        amount_utilized = amount_released * utilized_pct

        # date_sanction: spanning 2019-2025
        date_sanction = _random_date_in_range(
            datetime(START_YEAR, 1, 1), datetime(END_YEAR, 12, 31), rng
        )

        # date_start: after sanction, or None (15% noisy pattern even in "normal")
        is_start_none = rng.random() < 0.10
        if is_start_none:
            date_start = None
        else:
            date_start = _generate_date_start(date_sanction, rng)

        # date_completion: after start, or None
        is_completion_none = rng.random() < 0.10
        if is_completion_none:
            date_completion = None
        else:
            date_completion = _generate_date_completion(
                date_sanction, date_start, is_noisy=False, rng=rng
            )

        # work_category
        work_category = rng.choice(available_categories)

        normal_records.append((
            work_id, work_name, district, state, vendor_name,
            sanction_amount, amount_released, amount_utilized,
            date_sanction, date_start, date_completion, work_category,
        ))

    # --- Step 2: Compute median sanction from normal records ---
    normal_sanctions = [r[5] for r in normal_records]  # index 5 is sanction_amount
    if normal_sanctions:
        normal_sanctions_sorted = sorted(normal_sanctions)
        n = len(normal_sanctions_sorted)
        if n % 2 == 1:
            median_sanction = normal_sanctions_sorted[n // 2]
        else:
            median_sanction = (normal_sanctions_sorted[n // 2 - 1] + normal_sanctions_sorted[n // 2]) / 2
    else:
        median_sanction = 5_000_000  # fallback

    # --- Step 3: Generate noisy records (20%) ---
    noisy_records: List[Tuple] = []
    for _ in range(n_noisy):
        work_id = f"WORK_{rng.randint(100_000_000, 999_999_999)}"

        # Work name: may be generic or slightly garbled
        work_name_option = [
            f"{rng.choice(available_categories)}_work",
            "",  # empty work name — missing field
            f"Work_{rng.randint(1, 999)}",
        ]
        work_name = rng.choice(work_name_option)

        district = rng.choice(available_districts) if rng.random() > 0.2 else None  # 20% missing district

        state = rng.choice(states) if rng.random() > 0.15 else None  # 15% missing state

        # Vendor name: may be None or noisy
        if rng.random() < 0.4:
            vendor_name = None  # missing
        else:
            vendor_name = _generate_vendor_name(is_anomalous=False, rng=rng, is_noisy=True)
            # Add noise: odd punctuation, mixed case will be handled by cleaner

        # Sanction amount: may be negative, zero, or extreme
        # 50% chance of negative/zero (bad data)
        if rng.random() < 0.5:
            sanction_amount = rng.uniform(-10_000_000, 0)  # negative or zero
        else:
            # Normal-ish but possibly inflated
            sanction_amount = _generate_sanction_amount(
                median_sanction, is_anomalous=False, rng=rng, scale=rng.uniform(0.5, 1.5)
            )

        # Amount released: may be missing, or exceed sanction (bad), or partial
        if rng.random() < 0.4:
            amount_released = None  # missing
        elif rng.random() < 0.3:
            # Exceed sanction (financial inconsistency)
            amount_released = sanction_amount * rng.uniform(1.1, 1.5)
        else:
            # Partial/normal
            released_pct = rng.uniform(0.3, 0.9)
            amount_released = sanction_amount * released_pct

        # Amount utilized: may be missing, exceed amount_released, or negative
        if rng.random() < 0.4:
            amount_utilized = None  # missing
        elif rng.random() < 0.3:
            # Exceed amount_released
            amount_utilized = amount_released * rng.uniform(1.1, 1.5) if amount_released else None
        else:
            # Partial
            if amount_released and amount_released > 0:
                utilized_pct = rng.uniform(0.2, 0.9)
                amount_utilized = amount_released * utilized_pct
            else:
                amount_utilized = None

        # date_sanction: may be None or extreme
        if rng.random() < 0.3:
            date_sanction = None
        else:
            date_sanction = _random_date_in_range(
                datetime(START_YEAR, 1, 1), datetime(END_YEAR, 12, 31), rng
            )

        # date_start: often None, or before date_sanction
        if rng.random() < 0.6:
            date_start = None
        elif rng.random() < 0.2:
            # Before sanction
            delta = timedelta(days=rng.randint(-365, -1))
            date_start = datetime(START_YEAR, 1, 1) + delta
        else:
            date_start = _generate_date_start(date_sanction, rng) if date_sanction else None

        # date_completion: many noisy patterns
        noisy_comp = rng.choice([
            None,
            "before_start",
            "far_future",
            "random_unrelated",
        ], p=[0.5, 0.2, 0.2, 0.1])

        if noisy_comp == "none":
            date_completion = None
        elif noisy_comp == "before_start" and date_start:
            delta = timedelta(days=rng.randint(-365, -1))
            date_completion = date_start + delta
        elif noisy_comp == "far_future":
            delta = timedelta(days=rng.randint(365, 1825))
            date_completion = date_sanction + delta
        else:  # random_unrelated
            date_completion = _random_date_in_range(
                datetime(START_YEAR, 1, 1), datetime(END_YEAR, 12, 31), rng
            )

        # work_category: may be None or wrong category
        if rng.random() < 0.3:
            work_category = None
        else:
            work_category = rng.choice(available_categories) if available_categories else "unknown"

        noisy_records.append((
            work_id, work_name, district, state, vendor_name,
            sanction_amount, amount_released, amount_utilized,
            date_sanction, date_start, date_completion, work_category,
        ))

    # --- Step 4: Generate anomalous records (10%) ---
    anomalous_records: List[Tuple] = []
    for _ in range(n_anomalous):
        work_id = f"WORK_{rng.randint(100_000_000, 999_999_999)}"

        # Work name: may be duplicated/near-duplicated from other records
        # Use a common anomalous pattern: same work name in different districts
        dup_templates = [
            "Construction of community hall",
            "Borewell drilling",
            "Culvert construction",
            "Road repair works",
        ]
        work_name = rng.choice(dup_templates)

        # District: concentrate anomalous records in a few districts
        # Pick from a subset to create "vendor dominance" patterns
        dominant_districts = rng.sample(available_districts, k=min(5, len(available_districts)))
        district = rng.choice(dominant_districts)

        # State
        state = rng.choice(states)

        # Vendor name: dominant vendors that appear repeatedly
        vendor_name = rng.choice(ANOMALOUS_VENDORS)

        # Sanction amount: INFLATED 3-10x the normal peer median
        inflation_factor = rng.uniform(3.0, 10.0)
        sanction_amount = median_sanction * inflation_factor

        # Amount released: may be close to sanction (sophisticated fraud) or also inflated
        released_pct = rng.uniform(0.8, 1.0)
        amount_released = sanction_amount * released_pct

        # Amount utilized: may be close to amount_released
        utilized_pct = rng.uniform(0.7, 1.0)
        amount_utilized = amount_released * utilized_pct

        # date_sanction: cluster in specific months for temporal bursts
        # 70% of anomalous records in March (fy-end burst), rest spread
        if rng.random() < 0.7:
            date_sanction = datetime(rng.randint(2020, 2024), 3, rng.randint(1, 31))
        else:
            date_sanction = _random_date_in_range(
                datetime(START_YEAR, 1, 1), datetime(END_YEAR, 12, 31), rng
            )

        # date_start: after sanction, but may be missing
        if rng.random() < 0.2:
            date_start = None
        else:
            date_start = _generate_date_start(date_sanction, rng)

        # date_completion: may be missing or illogical
        if rng.random() < 0.4:
            date_completion = None
        else:
            date_completion = _generate_date_completion(
                date_sanction, date_start, is_noisy=True, rng=rng
            )

        # work_category: may be from the normal set or None
        work_category = rng.choice(available_categories) if rng.random() > 0.2 else None

        anomalous_records.append((
            work_id, work_name, district, state, vendor_name,
            sanction_amount, amount_released, amount_utilized,
            date_sanction, date_start, date_completion, work_category,
        ))

    # --- Step 5: Shuffle all records together ---
    all_records = normal_records + noisy_records + anomalous_records
    rng.shuffle(all_records)

    # ------------------------------------------------------------------
    # Post-shuffle: assign approval dates (date_of_administrative_approval)
    # that will be used by downstream stages. We'll set these now.
    # ------------------------------------------------------------------
    # The raw tuple order we've been using is:
    # (work_id, work_name, district, state, vendor_name,
    #  sanction_amount, amount_released, amount_utilized,
    #  date_sanction, date_start, date_completion, work_category)
    #
    # For downstream Stage 7 we need date_of_administrative_approval.
    # We'll add it as an additional element or shift the tuple.
    # Here, we'll prepend/append as needed. Let's append it:
    # New tuple order: (date_of_administrative_approval, work_id, work_name, ...)
    # Actually, let's just ensure date_sanction serves as the approval date too
    # for simplicity, or we generate a separate approval date.

    # Let's generate admin approval dates: typically shortly after sanction,
    # or sometimes delayed (noisy/burst patterns)
    final_records: List[Tuple] = []
    for rec in all_records:
        (work_id, work_name, district, state, vendor_name,
         sanction_amount, amount_released, amount_utilized,
         date_sanction, date_start, date_completion, work_category) = rec

        # Generate date_of_administrative_approval:
        # - Normal: a few days to weeks after date_sanction
        # - Noisy: may be None, or far after, or before
        # - Anomalous: may cluster in burst months
        if rng.random() < 0.7:
            # Normal: a few weeks after sanction
            weeks_after = rng.randint(0, 12)
            admin_approval = date_sanction + timedelta(weeks=weeks_after)
        elif rng.random() < 0.85:
            # Delayed approval (noisy)
            months_after = rng.randint(1, 36)
            admin_approval = date_sanction + timedelta(days=30 * months_after)
        else:
            # Very delayed or None
            admin_approval = None

        # Build final tuple with admin_approval as the FIRST element,
        # matching the order expected by the pipeline validator.
        # We'll put admin_approval at position 0, shift everything else.
        final_tuple = (admin_approval, work_id, work_name, district, state, vendor_name,
                       sanction_amount, amount_released, amount_utilized,
                       date_sanction, date_start, date_completion, work_category)
        final_records.append(final_tuple)

    return final_records