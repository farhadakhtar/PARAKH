"""Deterministic synthetic dataset generator (Stage1.md sec.3.1 / sec.3.2).

Produces a corpus of MPLADS-like public works records that is *realistically
dirty*: the noise is not decoration, it is the point. Stage 2 cannot be
developed or validated without data whose defects are known in advance.

Determinism
-----------
All randomness flows from a single :class:`numpy.random.Generator` seeded once.
No global ``random``/``np.random`` state is touched, and no wall-clock value is
read (see :data:`~src.core.constants.REFERENCE_DATE`). ``seed=42`` therefore
yields byte-identical output on every machine and every day.

Noise channels
--------------
Channels are applied in a fixed order and each is recorded in a
:class:`DefectLedger`:

1. duplicate / near-duplicate work names   (5%)
2. cost outliers                            (5%)
3. negative amounts                         (1%)
4. extreme magnitudes                       (0.3%)
5. duplicate work ids                       (0.5%)
6. milestone ordering violations            (9% injected)
7. pre-scheme dates                         (1%)
8. recoverable formatting noise             (3% of date/amount cells)
9. unparseable garbage                      (2% of date/amount cells)
10. missing values and placeholders         (10-20% of cells, per-field rates)

The missing-value channel runs **last** and deliberately masks some earlier
defects behind a null. That is what real data looks like, and it is why the
injected ordering-violation rate sits at the top of the PRD's 5-10% band: the
*observed* rate after masking lands mid-band.

The ledger is written to a sidecar file and is never a column of the dataset -
leaking it into the corpus would hand Stages 2-7 the answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    ALLOWED_STATUS,
    COST_OUTLIER_BAND,
    COST_OUTLIER_HIGH_RANGE,
    COST_OUTLIER_HIGH_SHARE,
    COST_OUTLIER_LOW_RANGE,
    COST_OUTLIER_RATE,
    DATE_FIELDS,
    DATE_ORDER_SHIFT_DAYS,
    DATE_ORDER_VARIANT_WEIGHTS,
    DATE_ORDER_VIOLATION_RATE,
    DATE_PLACEHOLDERS,
    DATE_VIOLATION_BAND,
    DEFAULT_N_RECORDS,
    DEFAULT_SEED,
    DUPLICATE_ID_RATE,
    DUPLICATE_NAME_BAND,
    DUPLICATE_NAME_RATE,
    EXTREME_VALUE_MAGNITUDE,
    EXTREME_VALUE_RATE,
    FIELD_ORDER,
    FLOAT_FIELDS,
    GEN_APPROVAL_LAG_DAYS,
    GEN_COMPLETION_LAG_DAYS,
    GEN_PROPOSAL_END,
    GEN_PROPOSAL_START,
    GROUND_TRUTH_LEDGER_NAME,
    MISSING_RATE_BAND,
    MISSING_RATES,
    NEAR_DUPLICATE_SHARE,
    NEGATIVE_AMOUNT_RATE,
    NUMERIC_PLACEHOLDERS,
    PLACEHOLDER_SHARE_OF_MISSING,
    PRE_SCHEME_DATE_RATE,
    PRE_SCHEME_SHIFT_YEARS,
    RECOMMENDED_SIZE_BAND,
    RECOVERABLE_FORMAT_RATE,
    RUPEE_SIGN,
    STRING_FIELDS,
    SYNTHETIC_CSV_NAME,
    TEXT_PLACEHOLDERS,
    UNPARSEABLE_DATE_TOKENS,
    UNPARSEABLE_FORMAT_RATE,
    UNPARSEABLE_NUMERIC_TOKENS,
)
from src.core.constants import DATA_DIR
from src.core.logger import get_logger
from src.stage1.ingestion import write_csv
from src.utils.helpers import safe_percentage, weighted_choice, write_json

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------

#: State -> districts. Twelve states x eight districts = 96 peer cells, enough
#: for Stage 3's clustering to have something to work with.
STATE_DISTRICTS: Mapping[str, Tuple[str, ...]] = {
    "Uttar Pradesh": (
        "Lucknow", "Varanasi", "Gorakhpur", "Kanpur Nagar",
        "Prayagraj", "Bareilly", "Meerut", "Jhansi",
    ),
    "Maharashtra": (
        "Pune", "Nagpur", "Nashik", "Chhatrapati Sambhajinagar",
        "Solapur", "Kolhapur", "Latur", "Amravati",
    ),
    "Karnataka": (
        "Belagavi", "Kalaburagi", "Mysuru", "Tumakuru",
        "Ballari", "Vijayapura", "Shivamogga", "Hassan",
    ),
    "Tamil Nadu": (
        "Coimbatore", "Madurai", "Salem", "Tiruchirappalli",
        "Thanjavur", "Vellore", "Erode", "Tirunelveli",
    ),
    "West Bengal": (
        "Murshidabad", "Purba Bardhaman", "Nadia", "Hooghly",
        "Malda", "Bankura", "Purulia", "Jalpaiguri",
    ),
    "Bihar": (
        "Patna", "Gaya", "Muzaffarpur", "Darbhanga",
        "Bhagalpur", "Purnia", "Saran", "Nalanda",
    ),
    "Rajasthan": (
        "Jaipur", "Jodhpur", "Udaipur", "Bikaner",
        "Ajmer", "Alwar", "Bhilwara", "Barmer",
    ),
    "Madhya Pradesh": (
        "Indore", "Jabalpur", "Gwalior", "Ujjain",
        "Sagar", "Rewa", "Satna", "Chhindwara",
    ),
    "Gujarat": (
        "Ahmedabad", "Surat", "Rajkot", "Vadodara",
        "Bhavnagar", "Junagadh", "Kachchh", "Mehsana",
    ),
    "Andhra Pradesh": (
        "Guntur", "Kurnool", "Visakhapatnam", "Anantapur",
        "Chittoor", "Nellore", "Kadapa", "Srikakulam",
    ),
    "Odisha": (
        "Cuttack", "Ganjam", "Sambalpur", "Balasore",
        "Mayurbhanj", "Koraput", "Puri", "Bolangir",
    ),
    "Punjab": (
        "Ludhiana", "Amritsar", "Patiala", "Jalandhar",
        "Bathinda", "Hoshiarpur", "Sangrur", "Ferozepur",
    ),
}

STATE_CODES: Mapping[str, str] = {
    "Uttar Pradesh": "UP",
    "Maharashtra": "MH",
    "Karnataka": "KA",
    "Tamil Nadu": "TN",
    "West Bengal": "WB",
    "Bihar": "BR",
    "Rajasthan": "RJ",
    "Madhya Pradesh": "MP",
    "Gujarat": "GJ",
    "Andhra Pradesh": "AP",
    "Odisha": "OD",
    "Punjab": "PB",
}

#: Rough relative volume per state, so the corpus is not uniformly distributed.
STATE_WEIGHTS: Mapping[str, float] = {
    "Uttar Pradesh": 1.60,
    "Maharashtra": 1.30,
    "Karnataka": 0.95,
    "Tamil Nadu": 1.05,
    "West Bengal": 1.15,
    "Bihar": 1.25,
    "Rajasthan": 0.90,
    "Madhya Pradesh": 1.00,
    "Gujarat": 0.85,
    "Andhra Pradesh": 0.80,
    "Odisha": 0.70,
    "Punjab": 0.55,
}

#: (work type, min cost INR, max cost INR, sampling weight).
WORK_TYPES: Tuple[Tuple[str, float, float, float], ...] = (
    ("CC Road", 300_000, 2_500_000, 1.60),
    ("Bituminous Road", 500_000, 4_000_000, 1.10),
    ("Culvert", 150_000, 900_000, 0.80),
    ("Drainage Line", 200_000, 1_500_000, 1.05),
    ("School Building Block", 800_000, 5_000_000, 0.75),
    ("Additional Classroom", 400_000, 1_800_000, 0.90),
    ("Community Hall", 600_000, 3_500_000, 0.70),
    ("Borewell with Hand Pump", 80_000, 400_000, 1.20),
    ("Overhead Water Tank", 500_000, 3_000_000, 0.65),
    ("Solar Street Lighting", 100_000, 900_000, 1.00),
    ("Street Light", 60_000, 500_000, 0.95),
    ("Public Toilet Block", 150_000, 1_200_000, 0.85),
    ("Library Building", 400_000, 2_500_000, 0.40),
    ("Playground Development", 200_000, 1_500_000, 0.55),
    ("Bus Shelter", 80_000, 600_000, 0.70),
    ("Health Sub Centre", 700_000, 4_000_000, 0.50),
    ("Crematorium Shed", 250_000, 1_500_000, 0.45),
    ("Footpath and Paver Block", 150_000, 1_000_000, 0.75),
    ("Check Dam", 400_000, 3_000_000, 0.45),
    ("Anganwadi Centre", 300_000, 1_800_000, 0.65),
)

WORK_ACTIONS: Tuple[str, ...] = (
    "Construction of",
    "Repair of",
    "Renovation of",
    "Upgradation of",
    "Installation of",
    "Extension of",
    "Providing and Fixing of",
    "Strengthening of",
    "Improvement of",
)

VILLAGE_NAMES: Tuple[str, ...] = (
    "Rampur", "Sultanpur", "Bhagwanpur", "Chandpur", "Devgaon", "Mahadevpura",
    "Kishanganj", "Nandgaon", "Hariharpur", "Basantpur", "Gopalpur", "Naganahalli",
    "Kotwali", "Shivpuri", "Mangalwada", "Ambedkar Nagar", "Indranagar",
    "Sundarpur", "Bhairavpalli", "Kalyanpur", "Lakshmipuram", "Narsinghpur",
    "Peddapalli", "Thottiyam", "Vadakkur", "Alipur", "Barigaon", "Chikkanahalli",
    "Dharampur", "Etawah Khurd", "Fatehpur", "Ganeshpura", "Hazaripur",
    "Islampur", "Jamalpur", "Karanjgaon", "Lodhipur", "Madhavpur",
)

AGENCY_TEMPLATES: Tuple[str, ...] = (
    "{district} Zilla Parishad",
    "Public Works Department, {state}",
    "Rural Engineering Service, {district}",
    "Municipal Corporation of {district}",
    "District Rural Development Agency, {district}",
    "Panchayati Raj Engineering Department, {state}",
    "{district} Municipal Council",
    "State Water Supply and Sewerage Board, {state}",
    "Block Development Office, {district}",
)

VENDOR_SURNAMES: Tuple[str, ...] = (
    "Sharma", "Patel", "Reddy", "Iyer", "Nair", "Gupta", "Verma", "Deshmukh",
    "Kulkarni", "Banerjee", "Chatterjee", "Mishra", "Yadav", "Rathore",
    "Choudhary", "Pillai", "Mehta", "Shetty", "Naidu", "Rao", "Joshi", "Singh",
    "Bhatt", "Sethi", "Kapoor", "Malhotra", "Gowda", "Prasad", "Das", "Bose",
    "Thakur", "Solanki", "Chauhan", "Dubey", "Pandey", "Saxena", "Trivedi",
    "Ghosh", "Sinha", "Menon",
)

VENDOR_SUFFIXES: Tuple[str, ...] = (
    "Constructions", "Infratech Pvt Ltd", "Builders", "Engineering Works",
    "and Sons", "Enterprises", "Contractors", "Infra Projects", "Associates",
    "Construction Company", "Developers", "Civil Works",
)

#: Number of distinct vendors. Kept well below the record count so vendor
#: concentration (Stage 5's HHI) has something real to measure.
VENDOR_POOL_SIZE: int = 900

#: Zipf-like exponent shaping vendor market share.
VENDOR_SKEW: float = 1.15

STATUS_WEIGHTS: Tuple[float, ...] = (0.12, 0.23, 0.65)


# ---------------------------------------------------------------------------
# Configuration and ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationConfig:
    """Knobs for :func:`generate_dataset`.

    Defaults reproduce the PRD's noise budget exactly. Tests override
    individual rates to isolate a channel.
    """

    n: int = DEFAULT_N_RECORDS
    seed: int = DEFAULT_SEED
    missing_rates: Mapping[str, float] = field(
        default_factory=lambda: dict(MISSING_RATES)
    )
    placeholder_share: float = PLACEHOLDER_SHARE_OF_MISSING
    date_order_violation_rate: float = DATE_ORDER_VIOLATION_RATE
    cost_outlier_rate: float = COST_OUTLIER_RATE
    duplicate_name_rate: float = DUPLICATE_NAME_RATE
    duplicate_id_rate: float = DUPLICATE_ID_RATE
    negative_amount_rate: float = NEGATIVE_AMOUNT_RATE
    extreme_value_rate: float = EXTREME_VALUE_RATE
    pre_scheme_date_rate: float = PRE_SCHEME_DATE_RATE
    recoverable_format_rate: float = RECOVERABLE_FORMAT_RATE
    unparseable_format_rate: float = UNPARSEABLE_FORMAT_RATE

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view of the configuration."""
        return {
            "n": self.n,
            "seed": self.seed,
            "missing_rates": dict(self.missing_rates),
            "placeholder_share": self.placeholder_share,
            "date_order_violation_rate": self.date_order_violation_rate,
            "cost_outlier_rate": self.cost_outlier_rate,
            "duplicate_name_rate": self.duplicate_name_rate,
            "duplicate_id_rate": self.duplicate_id_rate,
            "negative_amount_rate": self.negative_amount_rate,
            "extreme_value_rate": self.extreme_value_rate,
            "pre_scheme_date_rate": self.pre_scheme_date_rate,
            "recoverable_format_rate": self.recoverable_format_rate,
            "unparseable_format_rate": self.unparseable_format_rate,
        }


class DefectLedger:
    """Ground truth: which defect was injected into which row.

    Rows are keyed by **positional index** in the emitted frame, because
    ``work_id`` is deliberately duplicated by one of the noise channels and so
    is not a usable key. Row order is preserved through CSV/Parquet
    serialisation, which makes the positional key a valid join.

    This object is written to a sidecar file and must never become a column of
    the dataset.
    """

    def __init__(self) -> None:
        self._defects: Dict[int, List[str]] = {}
        self.channel_counts: Dict[str, int] = {}

    def record(self, rows: Sequence[int], label: str) -> None:
        """Attach ``label`` to every row index in ``rows``."""
        count = 0
        for row in rows:
            self._defects.setdefault(int(row), []).append(label)
            count += 1
        if count:
            self.channel_counts[label] = self.channel_counts.get(label, 0) + count

    @property
    def n_defective_rows(self) -> int:
        """Number of rows carrying at least one injected defect."""
        return len(self._defects)

    def rows_with(self, prefix: str) -> List[int]:
        """Row indices carrying a defect label starting with ``prefix``."""
        return sorted(
            row
            for row, labels in self._defects.items()
            if any(label.startswith(prefix) for label in labels)
        )

    def to_dict(self, config: GenerationConfig, observed: Dict[str, Any]) -> Dict[str, Any]:
        """Full JSON payload for the sidecar file."""
        return {
            "_note": (
                "Ground truth for the synthetic corpus. Keyed by positional row "
                "index in the emitted dataset. NEVER join this into the corpus "
                "before Stage 7 evaluation - it leaks labels."
            ),
            "config": config.to_dict(),
            "channel_counts": dict(sorted(self.channel_counts.items())),
            "n_defective_rows": self.n_defective_rows,
            "observed": observed,
            "defects_by_row": {
                str(row): sorted(labels)
                for row, labels in sorted(self._defects.items())
            },
        }


@dataclass(frozen=True)
class GenerationResult:
    """A generated dataset together with its ground-truth ledger."""

    frame: pd.DataFrame
    ledger: DefectLedger
    config: GenerationConfig
    observed: Dict[str, Any]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def indian_group(value: float) -> str:
    """Format a number with Indian digit grouping (``1,25,000.50``).

    Args:
        value: Amount to format.

    Returns:
        Grouped string, with a leading minus preserved.
    """
    negative = value < 0
    magnitude = abs(float(value))
    whole = int(magnitude)
    fraction = magnitude - whole

    digits = str(whole)
    if len(digits) <= 3:
        grouped = digits
    else:
        head, tail = digits[:-3], digits[-3:]
        parts: List[str] = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        grouped = ",".join(parts + [tail])

    if fraction > 0:
        grouped = f"{grouped}.{int(round(fraction * 100)):02d}"
    return f"-{grouped}" if negative else grouped


def _perturb_name(name: str, variant: int) -> str:
    """Turn an exact duplicate into a near-duplicate.

    Six deterministic perturbations mimicking real data-entry drift:
    abbreviation, case shift, whitespace damage, suffixing, truncation and a
    character transposition.
    """
    if variant == 0:
        return name.replace("Construction of", "Constn. of").replace(
            "Providing and Fixing of", "Prov. & Fixing of"
        )
    if variant == 1:
        return name.upper()
    if variant == 2:
        return name.replace(" at ", "  at  ")
    if variant == 3:
        return f"{name} - Phase II"
    if variant == 4:
        return name.split(",")[0]
    if len(name) > 12:
        cut = len(name) // 2
        return name[:cut] + name[cut + 1] + name[cut] + name[cut + 2 :]
    return name


# ---------------------------------------------------------------------------
# Base generation
# ---------------------------------------------------------------------------


def _empty_frame() -> pd.DataFrame:
    """Zero-row frame carrying the full schema - the empty-dataset edge case."""
    return pd.DataFrame({name: pd.Series([], dtype="object") for name in FIELD_ORDER})


def _build_vendor_pool(rng: np.random.Generator) -> np.ndarray:
    """Create the fixed vendor universe."""
    surnames = rng.choice(
        np.asarray(VENDOR_SURNAMES, dtype=object), size=VENDOR_POOL_SIZE
    )
    suffixes = rng.choice(
        np.asarray(VENDOR_SUFFIXES, dtype=object), size=VENDOR_POOL_SIZE
    )
    serials = rng.integers(1, 60, size=VENDOR_POOL_SIZE)
    return np.asarray(
        [
            f"{surname} {suffix}"
            if serial > 12
            else f"{surname} {suffix} (Unit {serial})"
            for surname, suffix, serial in zip(surnames, suffixes, serials)
        ],
        dtype=object,
    )


def _generate_base(
    n: int, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Generate the clean, internally consistent base corpus.

    Every record here has all three milestone dates present and correctly
    ordered, both amounts populated, and a plausible cost for its work type.
    All defects are introduced afterwards, by the noise channels, so the noise
    budget stays exactly auditable.
    """
    # -- geography ------------------------------------------------------
    pairs: List[Tuple[str, str]] = []
    pair_weights: List[float] = []
    for state, districts in STATE_DISTRICTS.items():
        share = STATE_WEIGHTS[state] / len(districts)
        for district in districts:
            pairs.append((state, district))
            pair_weights.append(share)

    weights = np.asarray(pair_weights, dtype="float64")
    weights /= weights.sum()
    pair_idx = rng.choice(len(pairs), size=n, p=weights)
    states = np.asarray([pairs[i][0] for i in pair_idx], dtype=object)
    districts = np.asarray([pairs[i][1] for i in pair_idx], dtype=object)

    # -- work type and cost ---------------------------------------------
    type_weights = np.asarray([entry[3] for entry in WORK_TYPES], dtype="float64")
    type_weights /= type_weights.sum()
    type_idx = rng.choice(len(WORK_TYPES), size=n, p=type_weights)

    lows = np.asarray([entry[1] for entry in WORK_TYPES], dtype="float64")[type_idx]
    highs = np.asarray([entry[2] for entry in WORK_TYPES], dtype="float64")[type_idx]
    type_labels = np.asarray([entry[0] for entry in WORK_TYPES], dtype=object)[type_idx]

    # Log-uniform within the band, pulled toward the centre by a Beta draw, so
    # costs are right-skewed like real tenders rather than flat.
    quantile = rng.beta(2.0, 2.5, size=n)
    sanction = lows * np.power(highs / lows, quantile)
    sanction = np.round(sanction / 100.0) * 100.0

    spend_ratio = np.clip(rng.normal(0.93, 0.10, size=n), 0.35, 1.25)
    spent = np.round(sanction * spend_ratio, 2)

    # -- milestone dates -------------------------------------------------
    span_days = (GEN_PROPOSAL_END - GEN_PROPOSAL_START).days
    proposal_offset = rng.integers(0, span_days + 1, size=n)
    approval_lag = rng.integers(
        GEN_APPROVAL_LAG_DAYS[0], GEN_APPROVAL_LAG_DAYS[1] + 1, size=n
    )
    completion_lag = rng.integers(
        GEN_COMPLETION_LAG_DAYS[0], GEN_COMPLETION_LAG_DAYS[1] + 1, size=n
    )

    origin = np.datetime64(GEN_PROPOSAL_START.isoformat(), "D")
    date_proposal = origin + proposal_offset.astype("timedelta64[D]")
    date_approval = date_proposal + approval_lag.astype("timedelta64[D]")
    date_completion = date_approval + completion_lag.astype("timedelta64[D]")

    # -- names, agencies, vendors ---------------------------------------
    actions = rng.choice(np.asarray(WORK_ACTIONS, dtype=object), size=n)
    villages = rng.choice(np.asarray(VILLAGE_NAMES, dtype=object), size=n)
    ward_numbers = rng.integers(1, 41, size=n)
    locality_style = rng.integers(0, 3, size=n)

    localities = np.asarray(
        [
            f"Ward No. {ward}"
            if style == 0
            else (f"Village {village}" if style == 1 else f"Gram Panchayat {village}")
            for style, ward, village in zip(locality_style, ward_numbers, villages)
        ],
        dtype=object,
    )
    work_names = np.asarray(
        [
            f"{action} {work_type} at {locality}, {district}"
            for action, work_type, locality, district in zip(
                actions, type_labels, localities, districts
            )
        ],
        dtype=object,
    )

    agency_idx = rng.integers(0, len(AGENCY_TEMPLATES), size=n)
    agencies = np.asarray(
        [
            AGENCY_TEMPLATES[idx].format(district=district, state=state)
            for idx, district, state in zip(agency_idx, districts, states)
        ],
        dtype=object,
    )

    vendor_pool = _build_vendor_pool(rng)
    vendor_weights = 1.0 / np.power(
        np.arange(1, VENDOR_POOL_SIZE + 1, dtype="float64"), VENDOR_SKEW
    )
    vendor_weights /= vendor_weights.sum()
    vendors = vendor_pool[rng.choice(VENDOR_POOL_SIZE, size=n, p=vendor_weights)]

    statuses = weighted_choice(rng, ALLOWED_STATUS, STATUS_WEIGHTS, n)

    # -- identifiers -----------------------------------------------------
    years = date_proposal.astype("datetime64[Y]").astype(int) + 1970
    work_ids = np.asarray(
        [
            f"MPL-{STATE_CODES[state]}-{year}-{seq:06d}"
            for state, year, seq in zip(states, years, range(1, n + 1))
        ],
        dtype=object,
    )

    return {
        "work_id": work_ids,
        "work_name": work_names,
        "district": districts,
        "state": states,
        "sanction_amount": sanction,
        "amount_spent": spent,
        "date_proposal": date_proposal,
        "date_approval": date_approval,
        "date_completion": date_completion,
        "implementing_agency": agencies,
        "vendor_name": vendors,
        "status": statuses,
    }


def _select_rows(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    """Pick ``round(n * rate)`` distinct row indices without replacement."""
    if n <= 0 or rate <= 0:
        return np.asarray([], dtype=int)
    k = int(round(n * rate))
    k = max(0, min(k, n))
    if k == 0:
        return np.asarray([], dtype=int)
    return np.sort(rng.choice(n, size=k, replace=False))


# ---------------------------------------------------------------------------
# Noise channels
# ---------------------------------------------------------------------------


def _inject_duplicate_names(
    columns: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    config: GenerationConfig,
    ledger: DefectLedger,
) -> None:
    """Clone work names onto other rows, sometimes with a perturbation."""
    targets = _select_rows(rng, n, config.duplicate_name_rate)
    if targets.size == 0:
        return

    sources = rng.choice(n, size=targets.size, replace=True)
    sources = np.where(sources == targets, (sources + 1) % max(n, 1), sources)

    names = columns["work_name"]
    perturb = rng.random(targets.size) < NEAR_DUPLICATE_SHARE
    variants = rng.integers(0, 6, size=targets.size)

    exact_rows: List[int] = []
    near_rows: List[int] = []
    for position, (target, source) in enumerate(zip(targets, sources)):
        cloned = names[source]
        if perturb[position]:
            names[target] = _perturb_name(str(cloned), int(variants[position]))
            near_rows.append(int(target))
        else:
            names[target] = cloned
            exact_rows.append(int(target))

    ledger.record(exact_rows, "duplicate_name:exact")
    ledger.record(near_rows, "duplicate_name:near")


def _inject_cost_outliers(
    columns: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    config: GenerationConfig,
    ledger: DefectLedger,
) -> None:
    """Scale both amounts on selected rows.

    Both columns are scaled by the *same* factor on purpose. Scaling only one
    would also manufacture a reconciliation mismatch, entangling two noise
    channels and making Stage 2's ``C_recon`` untestable in isolation.
    """
    rows = _select_rows(rng, n, config.cost_outlier_rate)
    if rows.size == 0:
        return

    is_high = rng.random(rows.size) < COST_OUTLIER_HIGH_SHARE
    factors = np.where(
        is_high,
        rng.uniform(*COST_OUTLIER_HIGH_RANGE, size=rows.size),
        rng.uniform(*COST_OUTLIER_LOW_RANGE, size=rows.size),
    )
    columns["sanction_amount"][rows] = np.round(
        columns["sanction_amount"][rows] * factors, 2
    )
    columns["amount_spent"][rows] = np.round(
        columns["amount_spent"][rows] * factors, 2
    )

    ledger.record(rows[is_high].tolist(), "cost_outlier:high")
    ledger.record(rows[~is_high].tolist(), "cost_outlier:low")


def _inject_amount_anomalies(
    columns: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    config: GenerationConfig,
    ledger: DefectLedger,
) -> None:
    """Introduce negative and astronomically large amounts.

    These exercise value validation: ``amount >= 0`` and finiteness.
    """
    negative_rows = _select_rows(rng, n, config.negative_amount_rate)
    if negative_rows.size:
        which = rng.integers(0, 2, size=negative_rows.size)
        for row, field_idx in zip(negative_rows, which):
            name = FLOAT_FIELDS[int(field_idx)]
            columns[name][row] = -abs(columns[name][row])
        ledger.record(negative_rows.tolist(), "negative_amount")

    extreme_rows = _select_rows(rng, n, config.extreme_value_rate)
    if extreme_rows.size:
        which = rng.integers(0, 2, size=extreme_rows.size)
        for row, field_idx in zip(extreme_rows, which):
            name = FLOAT_FIELDS[int(field_idx)]
            columns[name][row] = EXTREME_VALUE_MAGNITUDE
        ledger.record(extreme_rows.tolist(), "extreme_value")


def _inject_duplicate_ids(
    columns: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    config: GenerationConfig,
    ledger: DefectLedger,
) -> None:
    """Copy an existing work_id onto another row, breaking key uniqueness."""
    targets = _select_rows(rng, n, config.duplicate_id_rate)
    if targets.size == 0:
        return
    sources = rng.choice(n, size=targets.size, replace=True)
    sources = np.where(sources == targets, (sources + 1) % max(n, 1), sources)
    columns["work_id"][targets] = columns["work_id"][sources]
    ledger.record(targets.tolist(), "duplicate_work_id")


def _inject_date_order_violations(
    columns: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    config: GenerationConfig,
    ledger: DefectLedger,
) -> None:
    """Break milestone ordering on selected rows.

    Three flavours: approval before proposal, completion before approval, or
    both. Dates are moved *backwards*, never forwards, so the violation is
    unambiguous.
    """
    rows = _select_rows(rng, n, config.date_order_violation_rate)
    if rows.size == 0:
        return

    variants = weighted_choice(
        rng, ("approval", "completion", "both"), DATE_ORDER_VARIANT_WEIGHTS, rows.size
    )
    shifts = rng.integers(
        DATE_ORDER_SHIFT_DAYS[0], DATE_ORDER_SHIFT_DAYS[1] + 1, size=rows.size
    ).astype("timedelta64[D]")

    approval_rows: List[int] = []
    completion_rows: List[int] = []
    for row, variant, shift in zip(rows, variants, shifts):
        if variant in ("approval", "both"):
            columns["date_approval"][row] = columns["date_proposal"][row] - shift
            approval_rows.append(int(row))
        if variant in ("completion", "both"):
            columns["date_completion"][row] = columns["date_approval"][row] - shift
            completion_rows.append(int(row))

    ledger.record(approval_rows, "date_order:approval_before_proposal")
    ledger.record(completion_rows, "date_order:completion_before_approval")


def _inject_pre_scheme_dates(
    columns: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    config: GenerationConfig,
    ledger: DefectLedger,
) -> None:
    """Shift whole records back before the 1993 scheme start.

    All three dates move together, so relative ordering is preserved and the
    only defect introduced is the impossible epoch.
    """
    rows = _select_rows(rng, n, config.pre_scheme_date_rate)
    if rows.size == 0:
        return
    years = rng.integers(
        PRE_SCHEME_SHIFT_YEARS[0], PRE_SCHEME_SHIFT_YEARS[1] + 1, size=rows.size
    )
    shifts = (years * 365).astype("timedelta64[D]")
    for name in DATE_FIELDS:
        columns[name][rows] = columns[name][rows] - shifts
    ledger.record(rows.tolist(), "pre_scheme_date")


def _to_object_columns(columns: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Freeze typed arrays into object arrays holding text/float cell values.

    Dates become ISO strings so the in-memory frame and its CSV serialisation
    contain literally the same tokens - which is what lets ``from_dataframe``
    and ``from_csv`` be proven equivalent.
    """
    frozen: Dict[str, np.ndarray] = {}
    for name in FIELD_ORDER:
        values = columns[name]
        if name in DATE_FIELDS:
            frozen[name] = np.asarray(
                np.datetime_as_string(values, unit="D"), dtype=object
            )
        elif name in FLOAT_FIELDS:
            frozen[name] = np.asarray(
                [float(value) for value in values], dtype=object
            )
        else:
            frozen[name] = np.asarray(values, dtype=object)
    return frozen


def _inject_format_noise(
    columns: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    config: GenerationConfig,
    ledger: DefectLedger,
) -> None:
    """Rewrite date and amount cells in odd formats.

    Two sub-channels:

    * **recoverable** - ``"Rs 1,25,000"``, ``"15-03-2019"``. Cleaning must
      recover these; they exercise sec.3.6's numeric and date normalisation.
    * **unparseable** - ``"31/02/2020"``, ``"abcd"``. Cleaning must *not*
      recover these; they must surface as ``NullReason.UNPARSEABLE``.
    """
    for name in FLOAT_FIELDS:
        rows = _select_rows(rng, n, config.recoverable_format_rate)
        styles = rng.integers(0, 3, size=rows.size)
        for row, style in zip(rows, styles):
            value = float(columns[name][row])
            if style == 0:
                columns[name][row] = f"Rs {indian_group(value)}"
            elif style == 1:
                columns[name][row] = indian_group(value)
            else:
                columns[name][row] = f"{RUPEE_SIGN}{value:.2f}"
        ledger.record(rows.tolist(), f"format_recoverable:{name}")

        bad_rows = _select_rows(rng, n, config.unparseable_format_rate)
        tokens = rng.choice(
            np.asarray(UNPARSEABLE_NUMERIC_TOKENS, dtype=object), size=bad_rows.size
        )
        for row, token in zip(bad_rows, tokens):
            columns[name][row] = token
        ledger.record(bad_rows.tolist(), f"format_unparseable:{name}")

    for name in DATE_FIELDS:
        rows = _select_rows(rng, n, config.recoverable_format_rate)
        styles = rng.integers(0, 4, size=rows.size)
        for row, style in zip(rows, styles):
            iso = str(columns[name][row])
            try:
                parsed = date.fromisoformat(iso)
            except ValueError:  # already polluted by an earlier channel
                continue
            if style == 0:
                columns[name][row] = parsed.strftime("%d-%m-%Y")
            elif style == 1:
                columns[name][row] = parsed.strftime("%d/%m/%Y")
            elif style == 2:
                columns[name][row] = parsed.strftime("%d %b %Y")
            else:
                columns[name][row] = parsed.strftime("%B %d, %Y")
        ledger.record(rows.tolist(), f"format_recoverable:{name}")

        bad_rows = _select_rows(rng, n, config.unparseable_format_rate)
        tokens = rng.choice(
            np.asarray(UNPARSEABLE_DATE_TOKENS, dtype=object), size=bad_rows.size
        )
        for row, token in zip(bad_rows, tokens):
            columns[name][row] = token
        ledger.record(bad_rows.tolist(), f"format_unparseable:{name}")


def _inject_missing(
    columns: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
    config: GenerationConfig,
    ledger: DefectLedger,
) -> None:
    """Blank cells per-field, some as true nulls and some as placeholders.

    Runs last so it can mask defects injected by earlier channels, exactly as
    happens in real registers where the same clerk who mis-entered a date also
    left the vendor blank.
    """
    for name in FIELD_ORDER:
        rate = float(config.missing_rates.get(name, 0.0))
        rows = _select_rows(rng, n, rate)
        if rows.size == 0:
            continue

        as_placeholder = rng.random(rows.size) < config.placeholder_share
        if name in DATE_FIELDS:
            vocabulary = DATE_PLACEHOLDERS
        elif name in FLOAT_FIELDS:
            vocabulary = NUMERIC_PLACEHOLDERS
        else:
            vocabulary = TEXT_PLACEHOLDERS
        tokens = rng.choice(np.asarray(vocabulary, dtype=object), size=rows.size)

        placeholder_rows: List[int] = []
        null_rows: List[int] = []
        for position, row in enumerate(rows):
            if as_placeholder[position]:
                columns[name][row] = tokens[position]
                placeholder_rows.append(int(row))
            else:
                columns[name][row] = None
                null_rows.append(int(row))

        ledger.record(null_rows, f"missing:{name}")
        ledger.record(placeholder_rows, f"placeholder:{name}")


# ---------------------------------------------------------------------------
# Observation / self-check
# ---------------------------------------------------------------------------


def observe_frame(
    frame: pd.DataFrame, ledger: Optional[DefectLedger] = None
) -> Dict[str, Any]:
    """Measure the defect rates actually present in a generated frame.

    Measured on the *raw* frame, before cleaning, so the numbers describe what
    the generator emitted rather than what ingestion made of it.

    Two rates need the ledger to be measured honestly:

    * **duplicate names** - a *near*-duplicate is no longer string-equal to its
      source, so counting exact string duplicates would understate the channel.
      The PRD's 5% budget covers duplicates *and* near-duplicates together.
    * **cost outliers** - indistinguishable from a genuinely expensive work
      without knowing which rows were scaled.

    Args:
        frame: Generated dataset.
        ledger: Ground truth, when available.

    Returns:
        Observed rates and counts, including whether each lands in its PRD band.
    """
    n = len(frame)
    if n == 0:
        return {"n": 0}

    null_like = frame.map(
        lambda value: value is None
        or (isinstance(value, float) and value != value)
        or (isinstance(value, str) and value.strip().lower() in {
            "", "n/a", "na", "null", "none", "nil", "unknown", "-", "0000-00-00",
        })
    )
    per_field = {
        name: safe_percentage(int(null_like[name].sum()), n) for name in FIELD_ORDER
    }
    overall_missing = float(null_like.to_numpy().mean())

    parsed = {
        name: pd.to_datetime(frame[name], errors="coerce", format="mixed")
        for name in DATE_FIELDS
    }
    violation = np.zeros(n, dtype=bool)
    violation |= (
        parsed["date_approval"].notna()
        & parsed["date_proposal"].notna()
        & (parsed["date_approval"] < parsed["date_proposal"])
    ).to_numpy()
    violation |= (
        parsed["date_completion"].notna()
        & parsed["date_approval"].notna()
        & (parsed["date_completion"] < parsed["date_approval"])
    ).to_numpy()

    names = frame["work_name"].dropna()
    exact_duplicate_rate = float(names.duplicated(keep="first").sum()) / n

    observed_missing_rate = overall_missing
    observed_violation_rate = float(violation.mean())

    result: Dict[str, Any] = {
        "n": n,
        "missing_cell_rate": round(observed_missing_rate, 4),
        "missing_cell_rate_in_band": (
            MISSING_RATE_BAND[0] <= observed_missing_rate <= MISSING_RATE_BAND[1]
        ),
        "missing_pct_by_field": per_field,
        "date_violation_rate": round(observed_violation_rate, 4),
        "date_violation_rate_in_band": (
            DATE_VIOLATION_BAND[0] <= observed_violation_rate <= DATE_VIOLATION_BAND[1]
        ),
        "exact_duplicate_work_name_rate": round(exact_duplicate_rate, 4),
        "unique_work_ids": int(frame["work_id"].nunique(dropna=True)),
    }

    if ledger is not None:
        duplicate_rate = len(ledger.rows_with("duplicate_name:")) / n
        outlier_rate = len(ledger.rows_with("cost_outlier:")) / n
        result.update(
            {
                "duplicate_work_name_rate": round(duplicate_rate, 4),
                "duplicate_work_name_in_band": (
                    DUPLICATE_NAME_BAND[0] <= duplicate_rate <= DUPLICATE_NAME_BAND[1]
                ),
                "cost_outlier_rate": round(outlier_rate, 4),
                "cost_outlier_rate_in_band": (
                    COST_OUTLIER_BAND[0] <= outlier_rate <= COST_OUTLIER_BAND[1]
                ),
                "duplicate_work_id_rate": round(
                    len(ledger.rows_with("duplicate_work_id")) / n, 4
                ),
                "injected_date_violation_rate": round(
                    len(ledger.rows_with("date_order:")) / n, 4
                ),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_with_ledger(
    n: int = DEFAULT_N_RECORDS,
    seed: int = DEFAULT_SEED,
    config: Optional[GenerationConfig] = None,
) -> GenerationResult:
    """Generate a dirty synthetic corpus plus its ground-truth ledger.

    Args:
        n: Number of records. Values outside
            :data:`~src.core.constants.RECOMMENDED_SIZE_BAND` are permitted (the
            edge-case tests need them) but logged as a warning.
        seed: Seed for the single :class:`numpy.random.Generator`.
        config: Full configuration. When given, ``n`` and ``seed`` are ignored
            in favour of the config's own values.

    Returns:
        A :class:`GenerationResult`.

    Raises:
        ValueError: If ``n`` is negative.
    """
    resolved = config or GenerationConfig(n=n, seed=seed)
    if resolved.n < 0:
        raise ValueError(f"n must be non-negative, got {resolved.n}")

    ledger = DefectLedger()
    if resolved.n == 0:
        LOGGER.warning("Generating an empty dataset (n=0).")
        frame = _empty_frame()
        return GenerationResult(
            frame=frame, ledger=ledger, config=resolved, observed={"n": 0}
        )

    low, high = RECOMMENDED_SIZE_BAND
    if not low <= resolved.n <= high:
        LOGGER.warning(
            "n=%d is outside the PRD's recommended %d-%d band; noise rates "
            "will be noisier than their nominal targets.",
            resolved.n,
            low,
            high,
        )

    rng = np.random.default_rng(resolved.seed)
    columns = _generate_base(resolved.n, rng)

    _inject_duplicate_names(columns, rng, resolved.n, resolved, ledger)
    _inject_cost_outliers(columns, rng, resolved.n, resolved, ledger)
    _inject_amount_anomalies(columns, rng, resolved.n, resolved, ledger)
    _inject_duplicate_ids(columns, rng, resolved.n, resolved, ledger)
    _inject_date_order_violations(columns, rng, resolved.n, resolved, ledger)
    _inject_pre_scheme_dates(columns, rng, resolved.n, resolved, ledger)

    frozen = _to_object_columns(columns)

    _inject_format_noise(frozen, rng, resolved.n, resolved, ledger)
    _inject_missing(frozen, rng, resolved.n, resolved, ledger)

    frame = pd.DataFrame(
        {name: pd.Series(frozen[name], dtype="object") for name in FIELD_ORDER},
        columns=list(FIELD_ORDER),
    )

    observed = observe_frame(frame, ledger)
    LOGGER.info(
        "Generated %d records (seed=%d): missing %.2f%%, date violations %.2f%%, "
        "duplicate names %.2f%%, %d rows carry >=1 injected defect.",
        resolved.n,
        resolved.seed,
        100 * observed.get("missing_cell_rate", 0.0),
        100 * observed.get("date_violation_rate", 0.0),
        100 * observed.get("duplicate_work_name_rate", 0.0),
        ledger.n_defective_rows,
    )
    for key in (
        "missing_cell_rate_in_band",
        "date_violation_rate_in_band",
        "duplicate_work_name_in_band",
        "cost_outlier_rate_in_band",
    ):
        if observed.get(key) is False:
            LOGGER.warning("Noise self-check outside PRD band: %s", key)

    return GenerationResult(
        frame=frame, ledger=ledger, config=resolved, observed=observed
    )


def generate_dataset(
    n: int = DEFAULT_N_RECORDS,
    seed: int = DEFAULT_SEED,
    config: Optional[GenerationConfig] = None,
) -> pd.DataFrame:
    """Generate a dirty synthetic corpus (Stage1.md sec.9's entry point).

    Args:
        n: Number of records.
        seed: Deterministic seed.
        config: Optional full configuration.

    Returns:
        An object-dtype DataFrame with the twelve schema columns. Values are
        intentionally heterogeneous (floats, ISO date strings, placeholder
        tokens, garbage) because that is what a real export looks like.
    """
    return generate_with_ledger(n=n, seed=seed, config=config).frame


def save_dataset(
    result: GenerationResult,
    data_dir: Path = DATA_DIR,
    csv_name: str = SYNTHETIC_CSV_NAME,
    ledger_name: str = GROUND_TRUTH_LEDGER_NAME,
) -> Dict[str, Path]:
    """Write the dataset and its ground-truth ledger to disk.

    Args:
        result: Output of :func:`generate_with_ledger`.
        data_dir: Destination directory.
        csv_name: Dataset filename.
        ledger_name: Sidecar ledger filename.

    Returns:
        Mapping of artefact name to written path.
    """
    directory = Path(data_dir)
    csv_path = write_csv(result.frame, directory / csv_name)
    ledger_path = write_json(
        result.ledger.to_dict(result.config, result.observed),
        directory / ledger_name,
    )
    LOGGER.info("Saved dataset to %s and ledger to %s", csv_path, ledger_path)
    return {"csv": csv_path, "ledger": ledger_path}
