"""Immutable constants for the PARAKH system.

Every magic number used anywhere in Stage 1 is declared here exactly once.
Nothing in this module depends on wall-clock time, environment variables or
random state, which is what makes the whole pipeline reproducible.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final, Mapping

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

#: Repository root, resolved from this file (src/core/constants.py -> root).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

SYNTHETIC_CSV_NAME: Final[str] = "synthetic_dataset.csv"
SYNTHETIC_PARQUET_NAME: Final[str] = "synthetic_dataset.parquet"
GROUND_TRUTH_LEDGER_NAME: Final[str] = "ground_truth_ledger.json"

SCHEMA_VERSION: Final[str] = "stage1.schema.v1"

# ---------------------------------------------------------------------------
# Temporal anchors
#
# REFERENCE_DATE is a FROZEN "today". Using the real clock here would make the
# synthetic dataset change from one day to the next and silently destroy the
# determinism guarantee (seed=42 must always yield the same bytes).
# ---------------------------------------------------------------------------

REFERENCE_DATE: Final[date] = date(2024, 12, 31)

#: MPLADS scheme start. Stage 2 treats any date before this as a hard failure.
SCHEME_START_DATE: Final[date] = date(1993, 1, 1)

#: Window used by the synthetic generator for proposal dates.
GEN_PROPOSAL_START: Final[date] = date(2015, 1, 1)
GEN_PROPOSAL_END: Final[date] = date(2022, 12, 31)

#: Lag windows (in days) between consecutive milestones in the base data.
GEN_APPROVAL_LAG_DAYS: Final[tuple[int, int]] = (5, 210)
GEN_COMPLETION_LAG_DAYS: Final[tuple[int, int]] = (30, 720)

# ---------------------------------------------------------------------------
# Schema vocabulary
# ---------------------------------------------------------------------------

FIELD_ORDER: Final[tuple[str, ...]] = (
    "work_id",
    "work_name",
    "district",
    "state",
    "sanction_amount",
    "amount_spent",
    "date_proposal",
    "date_approval",
    "date_completion",
    "implementing_agency",
    "vendor_name",
    "status",
)

STRING_FIELDS: Final[tuple[str, ...]] = (
    "work_id",
    "work_name",
    "district",
    "state",
    "implementing_agency",
    "vendor_name",
    "status",
)
FLOAT_FIELDS: Final[tuple[str, ...]] = ("sanction_amount", "amount_spent")
DATE_FIELDS: Final[tuple[str, ...]] = (
    "date_proposal",
    "date_approval",
    "date_completion",
)

#: The identifier column. It is the only field that may never be null.
KEY_FIELD: Final[str] = "work_id"

ALLOWED_STATUS: Final[tuple[str, ...]] = ("proposed", "approved", "completed")

#: Ordered milestone pairs. Stage 2 reuses this exact tuple for C_temp so that
#: the two stages can never disagree about what "out of order" means.
ORDERED_DATE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("date_proposal", "date_approval"),
    ("date_approval", "date_completion"),
)

#: The two independently-reported money columns compared by Stage 2 (C_recon).
RECONCILIATION_PAIR: Final[tuple[str, str]] = ("sanction_amount", "amount_spent")

# ---------------------------------------------------------------------------
# Placeholder / null vocabulary
# ---------------------------------------------------------------------------

#: Case-insensitive tokens that look like data but encode absence. Compared
#: after trimming and whitespace collapsing.
PLACEHOLDER_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "",
        "-",
        "--",
        ".",
        "?",
        "n/a",
        "n.a.",
        "na",
        "null",
        "none",
        "nil",
        "nan",
        "unknown",
        "not available",
        "not applicable",
        "no data",
        "0000-00-00",
        "00-00-0000",
        "9999-99-99",
    }
)

#: Placeholder strings the generator injects into text columns.
TEXT_PLACEHOLDERS: Final[tuple[str, ...]] = ("N/A", "unknown", "NULL", "-", "NA")
#: Placeholder strings the generator injects into date columns.
DATE_PLACEHOLDERS: Final[tuple[str, ...]] = ("0000-00-00", "N/A", "unknown", "-")
#: Placeholder strings the generator injects into numeric columns.
NUMERIC_PLACEHOLDERS: Final[tuple[str, ...]] = ("N/A", "unknown", "-", "NIL")

#: Rupee sign (U+20B9), kept as an escape so every source file stays ASCII.
RUPEE_SIGN: Final[str] = "₹"

#: Substrings stripped from numeric strings before float() is attempted.
#: Longer tokens must precede their own prefixes ("rs." before "rs").
CURRENCY_TOKENS: Final[tuple[str, ...]] = (
    RUPEE_SIGN,
    "rs.",
    "rs",
    "inr",
    "$",
    ",",
    "_",
)

#: Explicit date formats attempted, in order, after ISO-8601 parsing fails.
#: Order is fixed so parsing is deterministic for ambiguous strings.
DATE_FORMATS: Final[tuple[str, ...]] = (
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%Y%m%d",
)

# ---------------------------------------------------------------------------
# Synthetic generation defaults
# ---------------------------------------------------------------------------

DEFAULT_N_RECORDS: Final[int] = 20_000
DEFAULT_SEED: Final[int] = 42

#: Size band mandated by Stage1.md sec.3.1. Smaller sizes are allowed (the
#: edge-case tests need them) but emit a warning.
RECOMMENDED_SIZE_BAND: Final[tuple[int, int]] = (10_000, 50_000)

# --- noise channel rates ---------------------------------------------------
# Per-field probability that a cell is blanked. The corpus-wide mean of these
# rates must land inside MISSING_RATE_BAND (Stage1.md sec.3.2: 10-20%).
MISSING_RATES: Final[Mapping[str, float]] = {
    "work_id": 0.00,  # the key is never blanked by this channel
    "work_name": 0.05,
    "district": 0.08,
    "state": 0.06,
    "sanction_amount": 0.12,
    "amount_spent": 0.19,
    "date_proposal": 0.10,
    "date_approval": 0.15,
    "date_completion": 0.24,
    "implementing_agency": 0.18,
    "vendor_name": 0.26,
    "status": 0.09,
}

#: Share of blanked cells that become a visible placeholder token rather than a
#: truly empty cell.
PLACEHOLDER_SHARE_OF_MISSING: Final[float] = 0.40

#: Fraction of records given a broken milestone ordering. Chosen at the top of
#: the 5-10% band because the missing-value channel runs afterwards and masks
#: roughly a quarter of the injected violations behind a null date.
DATE_ORDER_VIOLATION_RATE: Final[float] = 0.09

#: Relative weights for the three flavours of ordering violation:
#: (approval before proposal, completion before approval, both).
DATE_ORDER_VARIANT_WEIGHTS: Final[tuple[float, float, float]] = (0.40, 0.35, 0.25)
DATE_ORDER_SHIFT_DAYS: Final[tuple[int, int]] = (1, 400)

COST_OUTLIER_RATE: Final[float] = 0.05
COST_OUTLIER_HIGH_SHARE: Final[float] = 0.75
COST_OUTLIER_HIGH_RANGE: Final[tuple[float, float]] = (15.0, 60.0)
COST_OUTLIER_LOW_RANGE: Final[tuple[float, float]] = (0.005, 0.05)

DUPLICATE_NAME_RATE: Final[float] = 0.05
#: Probability that a cloned name is perturbed into a *near*-duplicate.
NEAR_DUPLICATE_SHARE: Final[float] = 0.60

DUPLICATE_ID_RATE: Final[float] = 0.005
NEGATIVE_AMOUNT_RATE: Final[float] = 0.01
EXTREME_VALUE_RATE: Final[float] = 0.003
EXTREME_VALUE_MAGNITUDE: Final[float] = 1e300

#: Any amount above this is not a data point, it is a data-entry accident.
#: No public work in the MPLADS universe costs 1e15 INR (~1000x India's GDP),
#: so a value beyond it is a validation ERROR rather than an outlier for
#: Stage 5 to rank. Named here so the threshold is arguable, not hidden.
IMPLAUSIBLE_AMOUNT_THRESHOLD: Final[float] = 1e15
PRE_SCHEME_DATE_RATE: Final[float] = 0.01
PRE_SCHEME_SHIFT_YEARS: Final[tuple[int, int]] = (25, 50)

#: Cells rewritten in an odd-but-recoverable format ("Rs 1,25,000", "15-03-2019").
RECOVERABLE_FORMAT_RATE: Final[float] = 0.03
#: Cells rewritten as genuine garbage ("31/02/2020", "abcd").
UNPARSEABLE_FORMAT_RATE: Final[float] = 0.02

UNPARSEABLE_DATE_TOKENS: Final[tuple[str, ...]] = (
    "31/02/2020",
    "2020-13-45",
    "not a date",
    "pending",
    "20200-01-01",
    "date awaited",
)
#: Note: "1.2e400" is a *valid* float literal that overflows float64 to +inf.
#: It is grouped here because it is the same kind of data-entry garbage, but it
#: exercises the non-finite validation path rather than the unparseable one.
UNPARSEABLE_NUMERIC_TOKENS: Final[tuple[str, ...]] = (
    "abcd",
    "to be decided",
    "as per estimate",
    "1.2e400",
    "12-34-56",
)

# --- acceptance bands used by tests and by the generator self-check --------
MISSING_RATE_BAND: Final[tuple[float, float]] = (0.10, 0.20)
DATE_VIOLATION_BAND: Final[tuple[float, float]] = (0.05, 0.10)
COST_OUTLIER_BAND: Final[tuple[float, float]] = (0.04, 0.06)
DUPLICATE_NAME_BAND: Final[tuple[float, float]] = (0.04, 0.06)

# ---------------------------------------------------------------------------
# Reporting / performance
# ---------------------------------------------------------------------------

#: Rounding used in every emitted percentage, so reports diff cleanly.
PERCENT_PRECISION: Final[int] = 2
AMOUNT_PRECISION: Final[int] = 2
DEFAULT_HEAD_ROWS: Final[int] = 5

#: Stage1.md sec.4: 50k rows must ingest, clean and validate in under 5s.
PERFORMANCE_ROW_BUDGET: Final[int] = 50_000
PERFORMANCE_SECONDS_BUDGET: Final[float] = 5.0

LOG_FILE_NAME: Final[str] = "parakh.log"
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"

# ===========================================================================
# STAGE 2 - Evidentiary Confidence Engine
#
# Every value below is a CALIBRATION PARAMETER, not a constant of nature.
# README.md is explicit that the system is non-operational until these are
# estimated against real data; the defaults here reproduce Stage2.md exactly
# so that scores are structurally correct while remaining, in the README's
# words, "operationally meaningless" until calibrated.
# ===========================================================================

STAGE2_VERSION: Final[str] = "stage2.confidence.v2"

# --- aggregation -----------------------------------------------------------

#: (w_comp, w_temp, w_recon). Must be non-negative and sum to 1.
# REFINEMENT (stage2.confidence.v2): rebalanced from equal thirds.
#
# Measured on the v1 engine, the variance decomposition of log C was
# temporal 50.3% / reconciliation 48.0% / completeness 1.8% - two components
# carried 98.2% of all ranking signal. C_recon is also the weakest *evidence*:
# it rests on a single budget-vs-outcome comparison whose semantics are
# approximate (see RECON_MODE), whereas completeness and temporal coherence
# rest on direct field-level observation. It therefore gets the smaller weight.
CONFIDENCE_WEIGHTS: Final[tuple[float, float, float]] = (0.4, 0.4, 0.2)

#: v1 weights, retained so the previous behaviour is exactly reproducible.
CONFIDENCE_WEIGHTS_V1: Final[tuple[float, float, float]] = (1 / 3, 1 / 3, 1 / 3)

#: Tolerance when checking that the weights sum to one.
WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9

# --- completeness ----------------------------------------------------------

#: Evidentiary credit per null reason, keyed by NullReason.value.
#:
#: Ordering (Stage 2 brief): missing < placeholder < unparseable in severity.
#:   present     - usable evidence.
#:   missing     - an honest gap. No evidence, but no false assertion either.
#:   placeholder - "completeness theatre": an assertion of no-data that
#:                 satisfies a form check while conveying nothing, and which
#:                 can mask an omission.
#:   unparseable - the record asserts a value that cannot be read. This is
#:                 evidence that the entry/export pipeline itself failed,
#:                 which casts doubt on the cells that *did* parse.
COMPLETENESS_CREDIT: Final[Mapping[str, float]] = {
    "present": 1.00,
    "missing": 0.20,
    "placeholder": 0.08,
    "unparseable": 0.00,
}

#: Force C_comp = 0 for a record with no PRESENT field at all.
#:
#: Necessary because the credits above are non-zero: without this rule a record
#: in which every single field is blank scores C_comp = 0.20 rather than 0, and
#: the core principle ("if the data is unreliable, the system MUST REFUSE
#: confidence") would be violated by the very worst record in the corpus.
#:
#: The justification is that credit for a defective cell measures *residual*
#: evidentiary value, and residual value is only meaningful relative to some
#: actual evidence. A record with no present field has no evidence base at all,
#: and a fraction of nothing is still nothing.
COMPLETENESS_REQUIRE_EVIDENCE: Final[bool] = True

#: Fields present in fewer than this share of records are dropped from the
#: completeness basis.
#:
#: This guards a genuine non-monotonicity in the PRD's weight formula:
#: (1 - H_null) is HIGH at both p_null -> 0 and p_null -> 1, and low only in
#: the middle. A field present 1% of the time therefore earns weight ~0.92 and
#: would uniformly depress every record's score without discriminating between
#: any of them.
MIN_FIELD_COVERAGE: Final[float] = 0.02

#: How H_value is normalised into [0,1].
#:   "cardinality" - divide by log2(k), k = number of distinct present values.
#:                   Scale-invariant: a record's score does not drift as the
#:                   corpus grows. This is the default.
#:   "sample"      - divide by log2(n_present). More discriminative between
#:                   low- and high-cardinality fields, but corpus-size
#:                   dependent, so the same record scores differently in a
#:                   1k-row and a 100k-row corpus.
ENTROPY_NORMALIZATION: Final[str] = "cardinality"
ENTROPY_NORMALIZATIONS: Final[tuple[str, ...]] = ("cardinality", "sample")

# --- temporal --------------------------------------------------------------

#: Decay rate for an ordering violation, per DAY.
#:
#: kappa is DIMENSIONAL. The unit is not optional: the same 0.01 applied to
#: seconds would give exp(-864) = 0 for a one-day inversion, collapsing the
#: soft penalty into a hard fail. Calibration at this value:
#:   1 day -> 0.990,  90 days -> 0.407,  400 days -> 0.018.
TEMPORAL_KAPPA_PER_DAY: Final[float] = 0.01

#: Factor applied to a milestone pair when either date is absent.
#:
#: 1.0 (neutral) by default: absence is a COMPLETENESS defect, already priced
#: by C_comp. Charging it again here would double-bill one defect across two
#: components that must stay orthogonal and independently interpretable.
#:
#: Known consequence: a record with no dates at all has an empty product and
#: scores C_temp = 1.0, i.e. perfect temporal coherence on zero evidence. It is
#: survivable only because the geometric mean multiplies it against a badly
#: damaged C_comp on the same record. `temporal_pairs_evaluated` is reported
#: per record so that "coherent" and "nothing to check" stay distinguishable.
TEMPORAL_MISSING_PAIR_CREDIT: Final[float] = 1.0

#: Whether a date after REFERENCE_DATE forces C_temp = 0. Stage2.md lists only
#: pre-scheme dates and unparseable dates as hard fails, so this defaults off.
TEMPORAL_HARD_FAIL_ON_FUTURE: Final[bool] = False

# --- reconciliation --------------------------------------------------------

#: Overspend decay rate for the plausibility model, and the legacy
#: disagreement rate under RECON_MODE = "agreement".
#:
#: HISTORY: under the v1 equality model this penalised any divergence between
#: budget and outcome, so a routine 30% underspend cost ~30% of C_recon. The
#: plausibility model retired that reading - underspend is now free down to
#: RECON_UNDERSPEND_FLOOR and gated on lifecycle stage, and lambda applies only
#: past RECON_OVERSPEND_TOLERANCE. Calibration target regardless.
RECON_LAMBDA: Final[float] = 2.0

#: Stabiliser in the denominator; also makes 0-vs-0 well defined without a branch.
RECON_EPSILON: Final[float] = 1e-6

#: Score when exactly one of the two amounts is null.
#:
#: This flat constant is doing a great deal of work: it fires on roughly 28% of
#: a realistically dirty corpus and caps those records at 0.2^(1/3) = 0.585 no
#: matter how perfect everything else is. Stage2.md words it as "e.g. 0.2" - a
#: suggestion, not a derivation. Prime candidate for calibration.
# REFINEMENT: raised 0.2 -> 0.7.
#
# HISTORY: the old value of 0.2 fired on 28.27% of the corpus and, under the
# equal 1/3 weights then in force, capped every one of those records at
# 0.2^(1/3) = 0.585 however sound the rest of the record was. Those 4,255
# records formed 96.1% of the [0.5,0.6) histogram bin - an artefact spike
# manufactured by one hard-coded constant rather than by anything about the
# records themselves. One absent amount is a partial-information penalty, not
# a verdict.
RECON_ONE_SIDED_CREDIT: Final[float] = 0.7

#: Score when both amounts are null: nothing is asserted, so nothing can
#: contradict. Per Stage2.md sec.5.4 ("Both values null -> ignore component").
RECON_BOTH_NULL_CREDIT: Final[float] = 1.0

#: Score when either amount is non-finite. Explicit, because the symmetric
#: ratio evaluates inf/inf = NaN, which would silently poison the log-sum.
# CORRECTION (audit finding 3): restored to 0.0.
#
# History: v1 used 0.0, the v2 refinement brief asked for a "strong penalty
# (<0.3)" and it became 0.25. The audit found that too weak - a record whose
# amount is literally infinite was still able to produce moderate confidence,
# because 0.25 does not trigger zero-dominance. Garbage must be refused, not
# discounted. Back to 0.0.
RECON_NON_FINITE_CREDIT: Final[float] = 0.0

#: Denominator form.
#:   "symmetric" - |x1| + |x2| + eps. Stage2.md sec.5.4 and README. Bounded,
#:                 sign-safe, stable near zero. Default.
#:   "max"       - max(x1, x2, eps). Alternative form; NOT sign-safe - with the
#:                 negative amounts Stage 1 injects, max(-1000, -500, eps) = eps
#:                 makes the ratio explode and the score underflow to 0.
RECON_NORMALIZATION: Final[str] = "symmetric"
RECON_NORMALIZATIONS: Final[tuple[str, ...]] = ("symmetric", "max")

# --- reporting -------------------------------------------------------------

#: Reporting bands for Stage2.md sec.10.1.
CONFIDENCE_LOW_THRESHOLD: Final[float] = 0.2
CONFIDENCE_HIGH_THRESHOLD: Final[float] = 0.8

#: Bin count for the confidence histogram (Stage2.md sec.10.2).
CONFIDENCE_HISTOGRAM_BINS: Final[int] = 10

#: Stage2.md sec.7: 50k records scored in under 3 seconds.
CONFIDENCE_SECONDS_BUDGET: Final[float] = 3.0

# ===========================================================================
# STAGE 2 REFINEMENT (stage2.confidence.v2)
#
# Three corrections to the v1 engine, each traced to a measurement:
#
#   1. C_recon was an EQUALITY test on a budget-vs-outcome pair. 74.57% of
#      comparable records sit in the normal execution band (0.2 <= r <= 1.0)
#      and were charged a mean penalty of 0.8875 for behaving correctly. It is
#      now a PLAUSIBILITY test.
#   2. Var(log C) decomposed as temporal 50.3% / recon 48.0% / comp 1.8%.
#      Weights rebalanced to (0.4, 0.4, 0.2) and the one-sided credit raised.
#   3. C_comp had an algebraic floor of 0.3449 (observed min 0.5150) because
#      work_id - never null, proving nothing - carried 18.11% of all weight
#      while the three dates and two amounts carried 30.56% between them.
#
# A later audit round added the lifecycle gate, the overspend tolerance and
# the restoration of outright refusal for garbage; see the STAGE 2 FINAL
# CORRECTIONS block at the end of this module.
# ===========================================================================

STAGE2_REFINEMENT_VERSION: Final[str] = "stage2.confidence.v2"

# --- C_recon: financial plausibility ---------------------------------------

#: Scoring semantics for reconciliation.
#:   "plausibility" - r = spent / (sanction + eps), asymmetric penalties.
#:                    Overspend is a control failure; underspend is normal
#:                    until it becomes implausible. This is the v2 default.
#:   "agreement"    - the v1 symmetric |x1-x2|/(|x1|+|x2|+eps) equality test,
#:                    retained so v1 behaviour is exactly reproducible and the
#:                    two can be compared on the same corpus.
RECON_MODE: Final[str] = "plausibility"
RECON_MODES: Final[tuple[str, ...]] = ("plausibility", "agreement")

#: Overspend decay. r = 1.10 -> 0.819;  r = 1.50 -> 0.368;  r = 2.00 -> 0.135.
#: Spending beyond sanction is not a reporting quirk: it requires a sanction
#: revision that should itself be on record, so its absence is a real signal.
RECON_OVERSPEND_LAMBDA: Final[float] = RECON_LAMBDA

#: Underspend is unpenalised until the ratio falls below this floor. Set at
#: 0.2 because a work reported against a sanction while having consumed under a
#: fifth of it is asserting something the money does not support. Measured: only
#: 0.62% of comparable records fall below it, against 74.57% in the band above.
RECON_UNDERSPEND_FLOOR: Final[float] = 0.2

#: Underspend decay, applied to max(0, floor - r), whose range is [0, 0.2] for
#: non-negative spend. gamma = 6.0 places total underspend (r = 0) at
#: exp(-1.2) = 0.301 - deliberately the same severity tier as a non-finite
#: amount, since both say the financial record cannot be believed.
RECON_UNDERSPEND_GAMMA: Final[float] = 6.0

#: Score when sanction <= 0, which makes the ratio meaningless.
#:
#: BEHAVIOUR CHANGE from v1: sanction = spent = 0 previously scored 1.0
#: ("zero equals zero, perfect agreement"). Under a plausibility reading a
#: non-positive budget is not plausible, so it is penalised. This follows
#: directly from the redefinition and is covered by an updated test.
RECON_NON_POSITIVE_SANCTION_CREDIT: Final[float] = 0.25

# --- C_comp: criticality weighting ------------------------------------------

#: How field weights v_f are formed.
#:   "criticality" - v_f = criticality_f * H_value(f). The v2 default.
#:   "entropy"     - v_f = (1 - H_null(f)) * H_value(f). The v1 behaviour.
#:   "hybrid"      - the product of all three factors.
#:
#: Why criticality replaces (1 - H_null): that term down-weighted precisely the
#: fields most likely to be absent, so the evidentiary spine of a work (dates
#: and money) ended up holding 30.56% of weight while the identifier held
#: 18.11%. It was defended as an artifact-invariance device, but README sec.9
#: places that guarantee on R, not C, and low-confidence records route to
#: REMEDIATE rather than INVESTIGATE. Confidence is *supposed* to track
#: documentation quality; suppressing that solved a problem the routing layer
#: already solves, at the cost of making C_comp nearly constant.
COMPLETENESS_WEIGHT_MODE: Final[str] = "criticality"
COMPLETENESS_WEIGHT_MODES: Final[tuple[str, ...]] = (
    "criticality",
    "entropy",
    "hybrid",
)

#: Fields whose absence removes the evidentiary spine of a work: when it was
#: proposed, sanctioned and completed, and what it cost.
CRITICAL_FIELDS: Final[tuple[str, ...]] = (
    "date_proposal",
    "date_approval",
    "date_completion",
    "sanction_amount",
    "amount_spent",
)

#: Per-field criticality. Critical fields 0.15-0.20; everything else 0.05.
#:
#: work_id sits at 0.05 deliberately. Stage 1 guarantees it is never null, so
#: whatever weight it carries is an identical constant added to every record -
#: pure range compression with zero discriminating power. At 18.11% under v1 it
#: was the single largest term in the whole score.
FIELD_CRITICALITY: Final[Mapping[str, float]] = {
    "work_id": 0.05,
    "work_name": 0.05,
    "district": 0.05,
    "state": 0.05,
    "sanction_amount": 0.20,
    "amount_spent": 0.15,
    "date_proposal": 0.20,
    "date_approval": 0.20,
    "date_completion": 0.15,
    "implementing_agency": 0.05,
    "vendor_name": 0.05,
    "status": 0.05,
}

#: Decay rate for the critical-field cluster penalty.
#:
#: Evidence loss is super-additive: losing one date is a gap, but losing all
#: three dates and both amounts destroys the record's ability to be
#: cross-checked at all, which a linear weighted average cannot express.
#: With delta = 0.35 the extra factor runs 1.00, 0.70, 0.50, 0.35, 0.25 as the
#: critical deficit grows from 1 through 5.
CLUSTER_PENALTY_DELTA: Final[float] = 0.35

#: Critical-field deficit allowed before the cluster penalty engages. At 1.0 a
#: single missing critical field costs only its weighted share, as before.
CLUSTER_PENALTY_ALLOWANCE: Final[float] = 1.0


# ===========================================================================
# STAGE 2 FINAL CORRECTIONS (audit response)
#
# Three findings, all confined to C_recon:
#   1. Lifecycle blindness - a proposed work legitimately has spent ~ 0 and was
#      charged the underspend penalty for being normal.
#   2. No overspend tolerance - penalty began at the first rupee past sanction,
#      charging rounding and routine price variation as anomaly.
#   3. Weak refusal for garbage - a non-finite amount scored 0.25, moderate
#      enough to survive aggregation.
# ===========================================================================

#: Column carrying the lifecycle stage of a work.
STATUS_FIELD: Final[str] = "status"

#: Statuses at which spending is not expected to have completed. Underspend
#: carries no information about data reliability for these records: a proposed
#: work with zero expenditure is behaving exactly as it should.
#:
#: "pending" is included for forward-compatibility with real MPLADS exports.
#: It is not in Stage 1's ALLOWED_STATUS, so Stage 1 will flag it
#: VALUE_UNKNOWN_STATUS - but Stage 2 will still route it correctly rather than
#: penalising a work for a vocabulary mismatch.
RECON_PRE_COMPLETION_STATUSES: Final[tuple[str, ...]] = (
    "proposed",
    "approved",
    "pending",
    "ongoing",
    "in progress",
)

#: Statuses at which the money should have been spent, so a low execution rate
#: genuinely contradicts the claim of completion.
RECON_TERMINAL_STATUSES: Final[tuple[str, ...]] = ("completed", "closed")

#: Multiplier applied to gamma when the lifecycle stage cannot be determined -
#: status null, placeholder, unparseable, or outside both vocabularies.
#:
#: A mild penalty, not the full one: we do not know whether the underspend is
#: legitimate, so we neither excuse it nor condemn it. At 0.5 a total
#: underspend scores exp(-0.6) = 0.549 instead of exp(-1.2) = 0.301.
RECON_UNKNOWN_STATUS_GAMMA_SCALE: Final[float] = 0.5

#: Tolerance band above the sanctioned amount before overspend is penalised.
#:
#: Rounding, minor price variation and final-bill adjustments routinely put a
#: work a percent or two over its sanction. Penalising from the first rupee
#: treated ordinary accounting noise as a control failure.
#: r <= 1.05 -> no penalty;  r > 1.05 -> exp(-lambda * (r - 1.05)).
RECON_OVERSPEND_TOLERANCE: Final[float] = 0.05

#: Score for an amount beyond IMPLAUSIBLE_AMOUNT_THRESHOLD (1e15).
#:
#: Stage 1 already classifies these as VALUE_IMPLAUSIBLE_MAGNITUDE errors. They
#: are finite, so they survive the non-finite branch, but a 1e300 sanction is
#: not a number this system should reason about - it is a data-entry accident.
#: Refused on the same terms as an infinity.
RECON_IMPLAUSIBLE_MAGNITUDE_CREDIT: Final[float] = 0.0

# ===========================================================================
# STAGE 3 - Semantic Layer & Peer Cell Formation
#
# Builds the comparison groups every downstream signal is measured against.
# Stage 3 computes STRUCTURE and DEVIATIONS; it does not score or classify
# anomalies - that is Stage 4's responsibility.
# ===========================================================================

STAGE3_VERSION: Final[str] = "stage3.peer.v1"

#: Single seed for every seeded operation in Stage 3 (currently only the
#: truncated SVD solver). HDBSCAN and TF-IDF are already deterministic.
STAGE3_SEED: Final[int] = 42

# --- text normalisation ----------------------------------------------------

#: Action verbs and connectives stripped before embedding. A repaired road and
#: a constructed road are the same KIND of work, so the action must not drive
#: the clustering.
STAGE3_BOILERPLATE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "construction", "constn", "repair", "renovation", "upgradation",
        "installation", "extension", "providing", "fixing", "strengthening",
        "improvement", "development", "supply", "provision",
        "of", "and", "at", "in", "on", "the", "for", "to", "by", "with",
        "work", "works", "phase", "unit", "i", "ii", "iii", "no",
    }
)

#: Locality markers. These prefix a place name and carry no work-type meaning.
STAGE3_LOCALITY_TOKENS: Final[frozenset[str]] = frozenset(
    {"ward", "village", "gram", "panchayat", "sector", "block", "nagar", "puram"}
)

#: Delimiters introducing the LOCATION clause of a work name.
#:
#: Public-works names follow "<action> <work type> at <locality>, <district>".
#: Everything from the delimiter onward names a place, not a kind of work.
#:
#: Truncating there is not a nicety. Measured without it, village names
#: survived normalisation and split single work types across several clusters -
#: "check dam" became one cluster for Peddapalli and another for Nandgaon - so
#: geography was silently becoming a grouping feature, which is precisely what
#: the grouping/testing separation forbids. Noise also ran at 28.4%.
#:
#: District and state stripping (below) cannot catch this: village names appear
#: nowhere in the district or state columns, and in a register where localities
#: are spread evenly across districts no statistical test distinguishes them
#: from work-type tokens either. Position is the signal, so position is used.
STAGE3_LOCALITY_DELIMITERS: Final[tuple[str, ...]] = (
    " at ",
    " in ",
    " near ",
    " opposite ",
    " opp ",
    " behind ",
    " adjacent to ",
)

#: Whether to truncate a work name at its first locality delimiter.
STAGE3_TRUNCATE_AT_LOCALITY: Final[bool] = True

#: Whether to strip every district and state name found in the corpus.
#:
#: NOT cosmetic. Every work_name in an MPLADS-style register ends with its
#: district. Left in, TF-IDF clusters by geography, and district-level
#: anomalies are normalised out of existence before Stage 4 can see them. The
#: grouping features must stay disjoint from the testing features.
STAGE3_STRIP_GEOGRAPHY: Final[bool] = True

# --- embedding -------------------------------------------------------------

TFIDF_NGRAM_RANGE: Final[tuple[int, int]] = (1, 2)
TFIDF_MIN_DF: Final[int] = 1
TFIDF_SUBLINEAR_TF: Final[bool] = True

#: Truncated-SVD width for the clustering projection.
#:
#: Measured against generator ground truth at min_cluster_size = 3: 8 dims gave
#: 0.704 weighted purity at 20k, 16 gave 0.924; 24 and above collapse into 27-41%
#: noise. HDBSCAN's density estimate degrades sharply with dimension, so 16 is
#: the measured optimum, not a guess.
#:
#: The sparse TF-IDF matrix is retained alongside the projection: duplicate
#: detection and per-cluster top-term labels both read it, so token-level
#: interpretability survives the reduction.
SVD_COMPONENTS: Final[int] = 16

# --- clustering ------------------------------------------------------------

#: HDBSCAN operates on UNIQUE normalised texts, not on records.
#:
#: Public-works names are heavily templated. After locality truncation, 20,000
#: records reduce to 91 distinct normalised strings and 50,000 to 184.
#: Clustering the distinct strings and broadcasting labels back runs in well
#: under a second where clustering records took 5.0s at 20k and 27.0s at 50k -
#: inside Stage3.md sec.11's 10s budget instead of 2.7x over it.
#:
#: It is also the better statistics. Identical text must receive an identical
#: cluster regardless, and collapsing duplicates removes the artificial density
#: spikes that templated naming creates in a density-based algorithm.
#:
#: NOTE the unit: this counts distinct TEXTS, not records. Stage3.md sec.6.2's
#: min_cluster_size = 20 is a record count; CLUSTER_MIN_RECORDS enforces that
#: separately.
#:
#: Chosen by sweep against generator ground truth, at four corpus sizes.
#: Weighted purity is over clustered records; noise is the share left
#: unclustered, and those records get no usable peer cell at all:
#:
#:      n      mcs=2 (k/noise/purity)     mcs=3 (k/noise/purity)
#:    5,000     13 / 11.2% / 0.802         5 / 64.9% / 0.916
#:   10,000     16 /  5.0% / 0.819         9 / 32.9% / 0.688
#:   20,000     17 /  7.6% / 0.924        17 /  7.6% / 0.924
#:   50,000     17 / 18.3% / 0.972        17 /  5.0% / 0.917
#:
#: 2 is selected. On the product of coverage and purity - the share of records
#: landing in a cluster that is actually correct - it wins at three of the four
#: sizes (0.712/0.778/0.854/0.794 against 0.321/0.462/0.854/0.871) and is far
#: more stable at small corpus sizes, where mcs=3 collapses to 65% noise.
#:
#: 2 is also the semantically right floor. After locality truncation, distinct
#: texts ARE distinct work types plus their typo variants, so two distinct
#: spellings of "borewell with hand pump" is sufficient evidence that such a
#: work type exists.
HDBSCAN_MIN_CLUSTER_SIZE: Final[int] = 2

#: Clusters holding fewer than this many RECORDS are merged into the nearest
#: retained cluster by centroid cosine (Stage3.md sec.6.4).
CLUSTER_MIN_RECORDS: Final[int] = 30

#: Label for records HDBSCAN could not place (Stage3.md sec.6.3).
NOISE_CLUSTER_ID: Final[int] = -1

#: Top TF-IDF terms kept per cluster as its human-readable label.
CLUSTER_LABEL_TERMS: Final[int] = 4

# --- cost stratification ---------------------------------------------------

#: Quantile bins over log(sanction_amount + 1), per Stage3.md sec.7.3 option A.
COST_STRATA_BINS: Final[int] = 5

#: Stratum for a record whose sanctioned amount is absent or unusable.
MISSING_STRATUM: Final[int] = -1

# --- peer cells ------------------------------------------------------------

#: Minimum records in a peer cell before its statistics may be trusted
#: (Stage3.md sec.8.1).
PEER_CELL_MIN_SIZE: Final[int] = 15

# --- peer statistics (confidence gating) -----------------------------------

#: Minimum Stage 2 confidence for a record to contribute to a peer norm.
#:
#: A cell's median and MAD are the yardstick every member is judged against. A
#: record with an unreadable amount or a fabricated timeline must not bend that
#: yardstick, because the corruption then propagates to every honest record in
#: the cell. Gated records are still ASSIGNED a cell and still MEASURED against
#: the clean norm - they are the REMEDIATE population - they simply get no vote
#: on what normal looks like.
PEER_STAT_MIN_CONFIDENCE: Final[float] = 0.5

#: Reconciliation branches barred from the statistics basis outright: their
#: amounts are not numbers this system should reason about.
PEER_STAT_EXCLUDED_BRANCHES: Final[tuple[str, ...]] = (
    "non_finite",
    "implausible_magnitude",
)

#: Minimum high-confidence members required before a peer norm is computed at
#: all. A 15-record cell with two usable members cannot define a median.
PEER_STAT_MIN_REFERENCE: Final[int] = 8

#: Consistency constant making MAD a consistent estimator of sigma under
#: normality. Median + MAD tolerate up to 50% contamination (README sec.2),
#: which mean and standard deviation do not.
MAD_SCALE: Final[float] = 1.4826

# --- duplicate detection ---------------------------------------------------

#: Cosine similarity above which two work names are near-duplicates.
DUPLICATE_SIMILARITY_THRESHOLD: Final[float] = 0.85

#: Temporal decay for the Stage3.md sec.9.1 duplicate score, in days.
DUPLICATE_TAU_DAYS: Final[float] = 180.0

#: Largest block compared pairwise. Blocking on (cluster, district) keeps the
#: comparison O(N*b) rather than O(N^2); this caps the worst block.
DUPLICATE_MAX_BLOCK: Final[int] = 600

# --- performance -----------------------------------------------------------

#: Stage3.md sec.11: 50k records processed in under 10 seconds.
STAGE3_SECONDS_BUDGET: Final[float] = 10.0

# ===========================================================================
# STAGE 3 HARDENING - calibration, evaluation and reproducibility
#
# Purely additive infrastructure. None of these values changes a score: they
# make the existing behaviour observable, measurable and repeatable.
# ===========================================================================

ARTIFACT_DIR: Final[Path] = PROJECT_ROOT / "artifacts"

TFIDF_VOCAB_FILE: Final[str] = "tfidf_vocab.json"
COST_STRATA_FILE: Final[str] = "cost_strata.json"
STAGE3_CONFIG_SNAPSHOT_FILE: Final[str] = "stage3_config.json"

STAGE3_CALIBRATION_REPORT: Final[str] = "stage3_calibration_report.json"
STAGE3_DUPLICATE_EVAL_REPORT: Final[str] = "stage3_duplicate_eval.json"
STAGE3_REPRODUCIBILITY_REPORT: Final[str] = "stage3_reproducibility_report.json"

#: Percentiles reported for every deviation distribution. These are diagnostic
#: only - Stage 3 does not threshold on them, and Stage 4 must not inherit them
#: as thresholds without calibrating first.
DEVIATION_PERCENTILES: Final[tuple[int, ...]] = (50, 90, 95, 99)

#: Share of a new corpus's tokens that may be absent from a frozen vocabulary
#: before the run is rejected.
#:
#: Beyond this the frozen feature space no longer describes the new data:
#: records would be embedded largely as zero vectors, cluster as noise, and
#: silently lose their peer cells. Failing loudly is the correct response.
MAX_UNSEEN_TOKEN_RATE: Final[float] = 0.35

#: Share of records that may fall outside the frozen strata's occupancy profile
#: before the run is rejected. Measured as total variation distance between the
#: recorded and observed bin occupancies.
MAX_STRATA_DRIFT: Final[float] = 0.35

#: Reproducibility artefacts are WRITTEN by default and REUSED only on request.
#: Silently reusing a stale vocabulary would be far worse than recomputing one.
STAGE3_REUSE_ARTIFACTS_DEFAULT: Final[bool] = False
STAGE3_SAVE_ARTIFACTS_DEFAULT: Final[bool] = True

# --- duplicate evaluation harness (test-only) ------------------------------

#: Injected duplicate pairs share a district and sit within this many days of
#: each other, matching Stage3.md sec.9.1's 1[d_i=d_j] and exp(-|dt|/tau).
EVAL_DUPLICATE_MAX_DAY_GAP: Final[int] = 30

#: Action verbs swapped in to make an injected duplicate a NEAR duplicate
#: rather than a byte-identical copy - the realistic case, and the harder one.
EVAL_DUPLICATE_ACTIONS: Final[tuple[str, ...]] = (
    "Construction of",
    "Renovation of",
    "Repair of",
    "Improvement of",
    "Upgradation of",
)

# ===========================================================================
# STAGE 3 AUDIT REMEDIATION
#
# Four correctness fixes from the Stage 3 deep audit. None changes a formula,
# a clustering decision or a threshold that was already load-bearing: they
# close guard gaps and make existing behaviour legible.
# ===========================================================================

#: Reason recorded when a deviation is undefined because the record's cluster
#: is the noise pool.
#:
#: AUDIT M1. form_peer_cells forces noise CELLS unstable, but the cluster-level
#: statistics path had no equivalent guard, so cluster -1 was emitting a median
#: and MAD pooled from records HDBSCAN judged similar to nothing. Measured on
#: the 20k corpus: 1,331 reference records, MAD 0.803 against 0.499 for a
#: typical real cluster - 61% wider, so every noise record was systematically
#: compressed toward zero and under-flagged. Noise is now barred from defining
#: a norm, and its records carry this reason rather than the generic
#: "no_peer_norm".
DEVIATION_REASON_CLUSTER_NOISE: Final[str] = "cluster_noise"

#: |z| above which a deviation is marked extreme.
#:
#: AUDIT M4. Deviations are NOT clipped - a 1e300 sanction genuinely is
#: thousands of MADs from its peers, and hiding that would be the silent
#: corruption Stage 1 exists to prevent. But Stage 4 inherits a distribution
#: whose maximum is 255x its own p99, so any magnitude-weighted aggregation
#: would be decided by 22 records. The flag lets Stage 4 see the tail coming
#: without Stage 3 destroying information.
Z_EXTREME_THRESHOLD: Final[float] = 20.0

#: |z| above which a deviation is marked high but not yet extreme.
Z_HIGH_THRESHOLD: Final[float] = 5.0

#: Magnitude buckets, in ascending severity. "undefined" is a distinct bucket
#: rather than a missing value, so the NaN-carries-a-reason rule holds here too.
DEVIATION_BUCKETS: Final[tuple[str, ...]] = ("undefined", "normal", "high", "extreme")

# --- duplicate evaluation: real perturbations (AUDIT M2) -------------------

#: AUDIT M2. The previous harness perturbed only the ACTION VERB - and action
#: verbs are stopwords in normalize_work_text, so the perturbation was erased
#: before the detector ever saw it. Verified: 60/60 injected pairs were
#: byte-identical in the detector's own text view, which means the reported
#: F1 of 0.929 measured EXACT-MATCH RETRIEVAL, not near-duplicate detection.
#: That figure is withdrawn.
#:
#: These perturbations act on tokens that SURVIVE preprocessing, so the
#: duplicate is genuinely a near match rather than a copy.

#: Synonym pairs among surviving content tokens.
EVAL_TOKEN_SWAPS: Final[tuple[tuple[str, str], ...]] = (
    ("centre", "center"),
    ("block", "blk"),
    ("road", "rd"),
    ("building", "bldg"),
    ("light", "lamp"),
    ("tank", "tanks"),
    ("line", "lines"),
    ("shelter", "stand"),
)

#: Fractional amount jitter applied to an injected duplicate, as a realistic
#: re-estimate rather than a copy of the sanctioned figure.
EVAL_AMOUNT_JITTER: Final[tuple[float, float]] = (0.05, 0.20)

#: Perturbation kinds cycled over injected duplicates. Every one of them
#: survives preprocessing, which is the whole point.
EVAL_PERTURBATIONS: Final[tuple[str, ...]] = ("typo", "swap", "truncate")

# ===========================================================================
# STAGE 4 - Contextual Anomaly Interpretation
#
# Stage 4 recomputes nothing. It consumes Stage 2 confidence and Stage 3 peer
# deviations, decides what they MEAN, and routes.
#
# SCOPE NOTE (recorded, not resolved silently): Stage4.md excludes "Final risk
# scoring" and "Routing decisions"; Stage5.md owns R(r) and Stage6.md owns the
# INVESTIGATE/REMEDIATE/MONITOR/CLEAR routing. The implementation brief for
# this stage requires both here. Containment: the score is named SEVERITY, not
# risk, and the decision uses INSUFFICIENT_CONTEXT rather than Stage 6's CLEAR,
# so neither downstream stage's output is pre-empted or overwritten.
# ===========================================================================

STAGE4_VERSION: Final[str] = "stage4.anomaly.v1"

# --- confidence gating (the core PARAKH rule) ------------------------------

#: Confidence below which a record may never be escalated to INVESTIGATE.
#:
#: README sec.8: "The system never emits a fraud hypothesis on low-confidence
#: evidence." A record under this line is routed to REMEDIATE whatever its
#: deviations look like - the deviations are not erased, they are simply not
#: allowed to mean "fraud" until the evidence supporting them is fixed.
#:
#: Matches PEER_STAT_MIN_CONFIDENCE by construction: a record that was not
#: trusted to SHAPE a peer norm is not trusted to be ACCUSED by one either.
CONFIDENCE_GATE_THRESHOLD: Final[float] = PEER_STAT_MIN_CONFIDENCE

# --- deviation thresholds --------------------------------------------------

#: |z| at which a deviation earns an anomaly TYPE.
#:
#: 3.0 robust MADs. Under normality that is roughly a 0.3% two-tailed tail, but
#: the distribution here is not normal and this is not a calibrated false-
#: positive rate - it is a starting judgement, like every other Stage 3/4
#: parameter. Stage 3's calibration report carries the observed percentiles
#: (cell cost: p95 1.69, p99 11.95) and explicitly forbids adopting them as
#: thresholds without calibration against real outcomes.
Z_TYPE_THRESHOLD: Final[float] = 3.0

#: |z| at which a high-confidence record is escalated to INVESTIGATE. Set above
#: the type threshold so that being unusual enough to NAME is deliberately a
#: lower bar than being unusual enough to ACCUSE.
Z_INVESTIGATE_THRESHOLD: Final[float] = 3.5

#: Divisor mapping |z| into [0,1] for severity. |z| >= this contributes fully.
Z_SEVERITY_SCALE: Final[float] = 5.0

# --- severity composition --------------------------------------------------

#: Weights over the four signals. Duplicate is deliberately the smallest: the
#: brief and Stage3.md both treat it as supporting evidence only, and Stage 3's
#: own evaluation measured ~1% recall on realistic near-duplicates, so it is
#: nowhere near strong enough to drive a decision.
#:
#: Weights are renormalised over the VALID signals of each record, so a record
#: missing its duration signal is scored on what it has rather than penalised
#: or credited for what it lacks.
SEVERITY_WEIGHTS: Final[Mapping[str, float]] = {
    "cost": 0.45,
    "spend": 0.30,
    "duration": 0.15,
    "duplicate": 0.10,
}

# --- vocabularies ----------------------------------------------------------

#: Anomaly types. A record may carry several; they are not mutually exclusive
#: and severity never overrides them.
ANOMALY_TYPES: Final[tuple[str, ...]] = (
    "cost_outlier",
    "overspend_anomaly",
    "underspend_anomaly",
    "temporal_outlier",
    "duplicate_suspect",
    "low_confidence",
    "insufficient_context",
)

#: Provisional triage classes. Stage 6 owns the final routing and may supersede
#: these; INSUFFICIENT_CONTEXT is used in place of Stage 6's CLEAR so the two
#: vocabularies never get conflated.
DECISION_CLASSES: Final[tuple[str, ...]] = (
    "INVESTIGATE",
    "REMEDIATE",
    "MONITOR",
    "INSUFFICIENT_CONTEXT",
)

#: Which cost deviation was used, in preference order.
COST_SCOPES: Final[tuple[str, ...]] = ("cell", "cluster", "none")

#: Stage 2 lifecycle labels that mean "the money should already have been
#: spent", so a low execution rate genuinely contradicts the record.
#:
#: Stage 2 emits "terminal"; the brief says "completed". Both are accepted so a
#: vocabulary difference between the two documents cannot silently disable the
#: underspend gate.
LIFECYCLE_TERMINAL_STATES: Final[tuple[str, ...]] = ("terminal", "completed", "closed")

#: Lifecycle labels at which low spend is expected and must NOT be an anomaly.
LIFECYCLE_PRE_COMPLETION_STATES: Final[tuple[str, ...]] = (
    "pre_completion",
    "proposed",
    "approved",
    "pending",
    "ongoing",
)

#: The deviation-derived signals counted by valid_signal_count. The duplicate
#: signal is excluded on purpose: it is supporting evidence, and letting it
#: satisfy the "has context" test would let a record with no peer comparison at
#: all escape the insufficient_context finding.
CORE_SIGNALS: Final[tuple[str, ...]] = ("cost", "spend", "duration")

STAGE4_ANOMALY_REPORT: Final[str] = "stage4_anomaly_report.json"


# ===========================================================================
# Stage 4 hardening - measurement, exposure and contract completion
#
# Nothing in this block influences a Stage 4 decision. Every constant here
# exists so that a judgement already being made can be SEEN. The z thresholds,
# severity weights and the confidence gate above remain uncalibrated
# judgements; this block is what makes calibrating them possible later.
# ===========================================================================

#: Quantiles reported for every distribution in the Stage 4 calibration report.
#: Fixed rather than configurable so two reports are always comparable.
CALIBRATION_QUANTILES: Final[tuple[float, ...]] = (0.50, 0.75, 0.90, 0.95, 0.99)

#: Why a record has no severity. Exhaustive and mutually exclusive: exactly one
#: applies to every record, so "no severity" is never an unexplained gap.
SEVERITY_DEFINED_REASONS: Final[tuple[str, ...]] = (
    "ok",
    "no_peer_norm",
    "cluster_noise",
    "no_valid_deviation",
    "insufficient_features",
)

#: Stage 3 deviation reasons that mean "a peer norm could not be established",
#: as opposed to "this record lacks the input". The distinction matters: the
#: first is a corpus-structure problem, the second is a data-quality problem,
#: and they are fixed by different people.
PEER_NORM_ABSENT_REASONS: Final[tuple[str, ...]] = (
    "cell_unstable",
    "no_peer_norm",
    "zero_dispersion",
)

#: Stage 3's reason for "the record does not carry the underlying value".
FEATURE_MISSING_REASON: Final[str] = "feature_missing"

#: Stage 3's reason for "this record is in the noise cluster".
CLUSTER_NOISE_REASON: Final[str] = "cluster_noise"

# --- duplicate observability ----------------------------------------------
#
# DUPLICATE_SIMILARITY_THRESHOLD (0.85) above is the DETECTION threshold and is
# not touched here. The constants below only describe what the detector can
# see, so that its ~1% measured recall can be attributed rather than guessed at.

#: Raw cosine at which a pair is considered *reachable* - within sight of the
#: detector had nothing else attenuated it. Well below the detection threshold
#: on purpose: the gap between reachable and flagged is the diagnostic.
DUPLICATE_REACHABLE_THRESHOLD: Final[float] = 0.60

#: Cosine cut points reported in the duplicate diagnostics, ascending.
DUPLICATE_DIAGNOSTIC_THRESHOLDS: Final[tuple[float, ...]] = (
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
)

STAGE4_CALIBRATION_REPORT: Final[str] = "stage4_calibration.json"
STAGE4_DUPLICATE_DIAGNOSTICS: Final[str] = "stage4_duplicate_diagnostics.json"


# ===========================================================================
# Stage 5 - Risk Scoring Layer
#
# Stage 4 says what deviates and how far. Stage 5 says how much that is worth
# acting on, GIVEN how much the record can be trusted. It labels nothing as
# fraud: a risk score is an estimate under uncertainty, not an accusation.
#
# Every constant here is a JUDGEMENT, not an estimate. None is fitted to a
# distribution, and the Stage 5 calibration report exists precisely so that
# these numbers can be argued with rather than assumed.
# ===========================================================================

STAGE5_VERSION: Final[str] = "stage5.risk.v1"

# --- step 1: signal strength (WHAT is wrong) -------------------------------
#
# Severity is the base. The three boosts fill the REMAINING headroom above it
# (`base + (1 - base) * boost`), which keeps the result in [0,1], keeps it
# strictly increasing in severity, and means no boost can ever lower a score.

#: Weight given to anomaly breadth - several distinct findings on one record
#: are worth more than one, but never more than the severity itself.
RISK_BREADTH_WEIGHT: Final[float] = 0.20

#: Weight given to the extreme-magnitude bucket. A z of 30 and a z of 4 both
#: saturate Stage 4's severity; this restores part of that lost ordering.
RISK_EXTREME_WEIGHT: Final[float] = 0.30

#: Weight given to a flagged near-duplicate. Capped at 0.10 by the brief and
#: by Stage 3's own measured ~1% recall: it may support a case, never make one.
RISK_DUPLICATE_WEIGHT: Final[float] = 0.10

#: Anomaly types that count toward breadth. `low_confidence` is excluded
#: because it is a statement about the EVIDENCE, not about the work, and it is
#: already priced in the data-quality term; counting it in both places would
#: penalise a record twice for one defect. `insufficient_context` is excluded
#: for the same reason via the uncertainty term.
RISK_BREADTH_TYPES: Final[tuple[str, ...]] = (
    "cost_outlier",
    "overspend_anomaly",
    "underspend_anomaly",
    "temporal_outlier",
    "duplicate_suspect",
)

#: Breadth saturates here: three simultaneous findings is already "broad".
RISK_BREADTH_SATURATION: Final[int] = 3

# --- step 2: data quality (CAN we trust it) --------------------------------

#: Below this, a record cannot carry a risk score at all. Same value as the
#: Stage 4 gate and Stage 3's PEER_STAT_MIN_CONFIDENCE, deliberately: a record
#: not trusted to shape a norm, or to be escalated, is not trusted to be
#: scored either.
MIN_CONFIDENCE_FOR_RISK: Final[float] = PEER_STAT_MIN_CONFIDENCE

#: Decay rate on critical_deficit. Stage 2 charges roughly 0.8 per missing
#: critical field, so k = 0.5 makes one missing critical field cost about a
#: third of the record's quality. A judgement about how much a missing date or
#: amount should matter - stated so it can be disputed.
RISK_CRITICAL_DEFICIT_DECAY: Final[float] = 0.5

#: Quality ceiling for a record whose dates are internally impossible. Not 0:
#: the record still exists and its other evidence is still readable. Near-zero
#: because a corpus that cannot order its own events cannot support a finding.
RISK_TEMPORAL_HARD_FAIL_QUALITY: Final[float] = 0.05

#: Multiplier applied when confidence sits below the gate. Such records never
#: receive a risk score, so this only shapes the reported COMPONENT - it exists
#: so the component stays meaningful for the records the gate excludes.
RISK_LOW_CONFIDENCE_PENALTY: Final[float] = 0.25

# --- step 3: uncertainty (HOW stable is the judgement) ---------------------
#
# Additive contributions, clipped into [0,1]. Undefined severity alone
# saturates: if the central quantity could not be computed, nothing about the
# record is stable.

#: No severity at all - nothing to be uncertain around.
RISK_UNCERTAINTY_NO_SEVERITY: Final[float] = 1.0

#: The work type carries no norm, so there is no baseline to deviate from.
RISK_UNCERTAINTY_NO_NORM: Final[float] = 0.60

#: The peer cell is too small to be relied on, though a coarser norm exists.
RISK_UNCERTAINTY_UNSTABLE_CELL: Final[float] = 0.25

#: Weight on missing signal coverage: a judgement resting on one of three
#: possible comparisons is less stable than one resting on all three.
RISK_UNCERTAINTY_COVERAGE_WEIGHT: Final[float] = 0.30

#: A record flagged as a duplicate that the detector could not actually have
#: seen. PROVABLY IMPOSSIBLE while Stage 4 holds - the temporal decay lies in
#: [0,1], so a blended score above the detection threshold implies a cosine
#: above the reachability cut. Implemented and measured anyway: if it ever
#: fires, Stage 3 and Stage 4 have diverged and the risk should say so.
RISK_UNCERTAINTY_UNREACHABLE_DUPLICATE: Final[float] = 0.40

# --- step 5: risk bands ----------------------------------------------------
#
# NOT tuned. Round numbers on the [0,1] scale, chosen before the distribution
# was looked at, and left alone afterwards. The calibration report states what
# they actually select; if that turns out to be uncomfortable, the honest move
# is to argue about the number in the open, not to slide it.

#: At or above: high risk.
R_HIGH: Final[float] = 0.50

#: At or above (and below R_HIGH): moderate risk. Below: low risk.
R_LOW: Final[float] = 0.20

#: Why a record has no risk score. Exhaustive and mutually exclusive.
RISK_UNDEFINED_REASONS: Final[tuple[str, ...]] = (
    "ok",
    "severity_undefined",
    "confidence_below_gate",
    "no_cluster_norm",
)

#: Mutually exclusive risk bands. These are NOT decisions - Stage 6 routes.
RISK_FLAGS: Final[tuple[str, ...]] = (
    "high_risk",
    "moderate_risk",
    "low_risk",
    "insufficient_data",
)

STAGE5_RISK_REPORT: Final[str] = "stage5_risk_report.json"
STAGE5_CALIBRATION_REPORT: Final[str] = "stage5_calibration.json"


#: Printed on every Stage 5 report. Removing it is a deliberate act that should
#: require validating the layer against real outcomes first.
CALIBRATION_STATUS_BANNER: Final[str] = (
    "UNFIT FOR PRODUCTION - NOT CALIBRATED. Every threshold and weight in Stage 5 is a stated judgement; none has been fitted to, or validated against, real outcomes. The system has only ever been run on synthetic data with injected defects, so no number here estimates a real-world rate of anything."
)


#: How each uncertainty component behaves INSIDE the score, established by
#: measurement (Stage 5 hardening) rather than by assumption.
#:
#: * ``active`` - fires on records that receive a score, and changes them.
#: * ``gate_redundant`` - can only fire on records the gate has already
#:   excluded. Retained because the uncertainty column is reported for EVERY
#:   record, scored or not, and there it is the whole answer.
#: * ``structurally_impossible`` - cannot fire while the upstream stages agree.
#:   Retained as an invariant guard: if it ever fires, Stage 3 and Stage 4 have
#:   diverged and the risk layer must say so rather than absorb it.
UNCERTAINTY_COMPONENT_CLASS: Final[dict[str, str]] = {
    "no_severity": "gate_redundant",
    "no_norm": "gate_redundant",
    "unstable_cell": "active",
    "coverage": "active",
    "unreachable_duplicate": "structurally_impossible",
}

#: A component contributing below this share of the score's log-variance is
#: FLAGGED in the calibration report. A flag is a prompt to examine, never an
#: instruction to remove: the decisive test for removal is whether dropping the
#: component changes any decision, not whether its variance is small.
CONTRIBUTION_FLAG_THRESHOLD_PCT: Final[float] = 1.0

#: Printed beside every risk distribution.
RISK_NOT_A_THRESHOLD_NOTE: Final[str] = (
    "Risk values are NOT calibrated thresholds. A risk of 0.5 does not mean a "
    "50% chance of anything; it is a position on an uncalibrated ordinal scale "
    "produced by this corpus and these judgements."
)


# ===========================================================================
# Stage 6 - Action & Routing Layer
#
# POLICY, not intelligence. Stage 6 computes nothing: it maps the Stage 4
# decision and the Stage 5 risk band onto an action, a priority and a queue,
# and writes a sentence a human can act on. Every table below is a policy
# choice that an operations lead should be able to change without a developer.
# ===========================================================================

STAGE6_VERSION: Final[str] = "stage6.routing.v1"

#: The five actions. Ordered most to least urgent.
ACTION_CLASSES: Final[tuple[str, ...]] = (
    "ESCALATE_IMMEDIATE",
    "ESCALATE_REVIEW",
    "DATA_QUALITY_REVIEW",
    "REQUEST_CORRECTION",
    "PASSIVE_MONITOR",
)

#: Priorities, most urgent first.
PRIORITY_LEVELS: Final[tuple[str, ...]] = ("P0", "P1", "P2", "P3")

#: Action -> priority. Fixed by the Stage 6 policy table.
ACTION_TO_PRIORITY: Final[dict[str, str]] = {
    "ESCALATE_IMMEDIATE": "P0",
    "ESCALATE_REVIEW": "P1",
    "DATA_QUALITY_REVIEW": "P1",
    "REQUEST_CORRECTION": "P2",
    "PASSIVE_MONITOR": "P3",
}

#: Action -> the team that owns the work.
ACTION_TO_QUEUE: Final[dict[str, str]] = {
    "ESCALATE_IMMEDIATE": "fraud_investigation_team",
    "ESCALATE_REVIEW": "audit_team",
    "REQUEST_CORRECTION": "field_officer",
    "PASSIVE_MONITOR": "automated_monitoring",
    "DATA_QUALITY_REVIEW": "data_quality_team",
}

REVIEWER_QUEUES: Final[tuple[str, ...]] = tuple(
    sorted(set(ACTION_TO_QUEUE.values()))
)

#: Actions that put a record in front of an investigator. Invariant 6 applies
#: to exactly these: an escalated record must carry at least one finding.
ESCALATING_ACTIONS: Final[tuple[str, ...]] = (
    "ESCALATE_IMMEDIATE",
    "ESCALATE_REVIEW",
)

#: The M1 correction label. Stage 4 gates `underspend_anomaly` on lifecycle, so
#: a large underspend on a work that is not yet complete escalates while
#: carrying no named finding. That is a labelling gap, not a new signal: the
#: deviation was already measured, already scored and already escalated
#: upstream. This label states plainly that something drove the escalation
#: which Stage 4 declined to name.
M1_CORRECTION_LABEL: Final[str] = "unexplained_deviation"

STAGE6_ACTION_REPORT: Final[str] = "stage6_action_report.json"


# ===========================================================================
# Stage 6 hardening - contract alignment and self-validation
#
# Nothing here changes a routing decision. Every entry either exposes an
# existing decision under a second name, or states an assumption that was
# previously being trusted in silence.
# ===========================================================================

#: The action vocabulary named in the Stage 6 audit specification.
#:
#: THREE vocabularies exist for this layer, and they do not agree:
#:
#: 1. ``Stage6.md`` (the PRD): INVESTIGATE / REMEDIATE / MONITOR / CLEAR.
#:    Those are *decision* names, and Stage 4 already implements them as
#:    DECISION_CLASSES (with INSUFFICIENT_CONTEXT in place of CLEAR).
#: 2. The Stage 6 build brief: :data:`ACTION_CLASSES`, which is what this
#:    system emits and what every downstream test binds to.
#: 3. The Stage 6 audit specification: the names below.
#:
#: The as-built names are authoritative because they are what the pipeline
#: produces; these are provided as an ALIAS so a consumer written against the
#: audit specification resolves. No routing logic reads them.
SPEC_ACTION_CLASSES: Final[tuple[str, ...]] = (
    # --- currently produced by SPEC_ACTION_ALIAS -------------------------
    "INVESTIGATE",
    "ROUTE_AUDIT",
    "REMEDIATE",
    "MONITOR",
    # --- named by a specification but never produced ---------------------
    "ESCALATE_INVESTIGATION",
    "ESCALATE_REVIEW",
    "ROUTE_REMEDIATE",
    "MONITOR_PASSIVE",
    "HOLD_NO_ACTION",
)

#: As-built action -> the closest specification name.
#:
#: ``DATA_QUALITY_REVIEW -> ROUTE_AUDIT`` is the weakest of the five: the
#: specification offers no data-quality action, and ROUTE_AUDIT is the nearest
#: remaining sense of "a team must look at this record before it can be
#: judged". Stated here rather than buried so the imprecision is visible.
#:
#: ``HOLD_NO_ACTION`` has no producer. Stage 6 never concludes that a record
#: needs nothing: the quietest outcome it emits is PASSIVE_MONITOR, which is a
#: standing watch rather than a dismissal. Representable, never emitted, and
#: that is a deliberate property of the policy, not an omission.
#: SUPERSEDED MAPPING. An earlier specification asked for a one-to-one
#: rename into ESCALATE_INVESTIGATION / ROUTE_REMEDIATE / MONITOR_PASSIVE.
#: Kept only so the change is traceable; nothing reads it.
SPEC_ACTION_ALIAS_V1: Final[dict[str, str]] = {
    "ESCALATE_IMMEDIATE": "ESCALATE_INVESTIGATION",
    "ESCALATE_REVIEW": "ESCALATE_REVIEW",
    "REQUEST_CORRECTION": "ROUTE_REMEDIATE",
    "DATA_QUALITY_REVIEW": "ROUTE_AUDIT",
    "PASSIVE_MONITOR": "MONITOR_PASSIVE",
}

#: As-built action -> the specification vocabulary, as currently mandated.
#:
#: **This mapping is deliberately NOT injective.** Both escalating actions
#: collapse to ``INVESTIGATE``, so ``action_spec`` alone cannot distinguish a
#: P0 fraud referral from a P1 audit review - 291 records from 128. That
#: distinction survives in ``action_class`` and ``priority_level``, which are
#: unchanged, and a consumer that needs it must read one of those. Recorded
#: here because a lossy alias is a real cost, not a detail.
SPEC_ACTION_ALIAS: Final[dict[str, str]] = {
    "ESCALATE_IMMEDIATE": "INVESTIGATE",
    "ESCALATE_REVIEW": "INVESTIGATE",
    "DATA_QUALITY_REVIEW": "ROUTE_AUDIT",
    "REQUEST_CORRECTION": "REMEDIATE",
    "PASSIVE_MONITOR": "MONITOR",
}

#: Specification column name -> the as-built column it aliases. Pure renames:
#: both columns carry identical values, verified by assertion on every run.
SPEC_COLUMN_ALIAS: Final[dict[str, str]] = {
    "action": "action_class",
    "priority": "priority_level",
    "action_reason": "action_rule",
}
