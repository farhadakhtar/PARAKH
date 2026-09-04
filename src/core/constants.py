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
