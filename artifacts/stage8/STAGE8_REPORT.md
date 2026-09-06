# Stage 8 - Empirical Validation: Report

Generated 2026-09-06 | commit `48917ed` | Python 3.12.4

---

## Verdict

## `COVERAGE_FAILURE`

**3 of 11 gates pass. `calibrated_risk` is None.**

Stage 8 REFUSES to calibrate. The binding constraint is data coverage, not model quality: the single available audit report covers a financial year that no structured dataset covers. No threshold was fitted, no label was manufactured, and no gate was weakened.

The binding constraint is **data coverage, not model quality**. No threshold
was fitted, no label manufactured, no gate weakened.

---

## The measurement that decides it

```
audit evidence      CAG SFAR, Government of Nagaland, FY 2020-21
structured finance  PMGSY state-year, FY 2023-24 / 24-25 / 25-26 / 26-27

year intersection   EMPTY
state intersection  {NAGALAND}  -> 1 state
work-level overlap  none: no dataset carries a work identifier
```

Even granting a year overlap, the joinable universe would be a handful of
state-year rows. Matching was **not** relaxed to force a join: dropping to
STATE_ONLY produces rows at LEVEL_0 evidence, which is the defect that put
5,237 labels on a postal directory in the previous build.

---

## Gates

| gate | status | detail |
|---|---|---|
| `G1_DATA_VALIDITY` | **PASS** | 20,000 synthetic dev records + 12 real Data/ files inventoried; provenance recorded per file. |
| `G2_LABEL_VALIDITY` | **FAIL** | 0 verified positives, 0 verified negatives at calibration-eligible evidence. NULL was not converted to 0. |
| `G3_FEATURE_AVAILABILITY` | **FAIL** | 0 labelled rows carry budget/expenditure features; no bid table exists anywhere. |
| `G4_COVERAGE` | **FAIL** | Audit year {2020-21} vs structured years {2023-24..2026-27}: intersection empty. 1 state in common. |
| `G5_LEAKAGE` | **PASS** | 6 feature(s), none target-derived |
| `G6_TEMPORAL_GENERALIZATION` | **NOT_EVALUABLE** | No labels, so no performance to measure across time. |
| `G7_PROBABILITY_CALIBRATION` | **NOT_EVALUABLE** | Blocked by G2/G4. Stage 8 calibrate() returns INSUFFICIENT_LABELS. |
| `G8_ARTIFACT_INVARIANCE` | **NOT_EVALUABLE** | Dev corpus has a single source_file, so no artifact grouping has 2 classes. Re-run when multi-source real data exists. |
| `G9_STABILITY` | **NOT_RUN** | Requires re-running Stages 3-5 per subsample; not executed this pass. |
| `G10_OPERATIONAL_UTILITY` | **NOT_EVALUABLE** | Precision@K needs outcomes. |
| `G11_EXPLAINABILITY` | **PASS** | Every record carries risk_explanation and a canonical explanation_payload from Stage 6. |

`NOT_EVALUABLE` is not a pass. A gate that could not run has not been
cleared, and is counted against the total.

---

## What was built

| module | purpose | tests |
|---|---|---|
| `src/stage8/matching.py` | blocked record/finding matching, evidence-tiered | 33 |
| `src/stage8/compliance.py` | statutory rules with citations | 23 |
| `src/stage8/injection.py` | Indic text + defect vectorisation | 31 |
| `src/stage8/typology.py` | documented fraud typologies | - |
| `src/stage8/safety.py` | artifact invariance, leakage, stability | - |
| `src/stage8/neural.py` | self-supervised consistency model | - |

### Matching - blocking proven, not asserted

```
naive    400,000 comparisons   6.226s   797 matches
blocked      866 comparisons   0.070s   797 matches
           462x fewer          89x       IDENTICAL
```

Parity against brute force is the entire correctness argument for blocking,
and it is asserted across seeds and both key regimes. A blocking key that
drops true pairs is invisible downstream - a metric computed on the labels
that exist cannot notice the labels never produced.

The parity tests also caught a real bug: the minimum-name-length guard was
being applied to work IDs, silently deleting the strongest evidence level.

### Compliance - thresholds by citation, not by taste

5 rules encoded, **0 citation-verified**. All gated to
`PENDING_CITATION_VERIFICATION`; none can fire.

`R > 0.7` is a judgement constant. A tender limit is not - it is a rule with
a number in it, and the right provenance is a citation rather than a fit.
That is available today with zero audit labels.

Rules ship inert because the clause numbers were stated from working
knowledge and not checked against primary sources here. An unverified
citation is worse than none: it carries the authority of law without the
substance. A cost-overrun rule was deliberately **omitted** - no single
overrun percentage is safely quotable, and inventing one would fabricate a
statutory number.

**Ceiling, stated plainly:** this finds *irregularity*, never fraud. An
emergency repair without tender breaks a rule and defrauds nobody; a
flawless tender whose losing bidders are shells owned by the winner passes
every check here.

### Typologies - the limits of adjudicated data

5 defined, 2 reachable with the columns present. The 3 unreachable ones
need `tender_id`, `bid_amount`, `is_winner`, `n_bidders`.

**PARAKH has never represented a competition.** The entire bid-rigging class
is structurally invisible to it, independent of any model.

Adjudicated sources (World Bank debarment, CVC blacklists) do exist and
cover India - an earlier claim that no labels exist was scoped to CAG
reports and was wrong. But they label **entities, not works**, and they
label **caught** fraud, so recall against undetected schemes is not merely
unmeasured but unmeasurable.

---

## The neural experiment, and why it failed

Masked-field consistency model, TensorFlow, 36,615 parameters, 50 epochs.

**Reconstruction - excellent:**

| field | model | mode baseline |
|---|---|---|
| state | 0.9743 | 0.1278 |
| district | 0.8375 | 0.0680 |
| implementing_agency | 0.8727 | 0.3990 |
| status | 0.9183 | 0.5840 |

**Defect detection - near chance:**

```
ROC-AUC  0.5379     enrichment 1.08x
```

This is the most instructive result in the layer. A model can look
excellent on its training objective and be worthless at the job. Two causes,
both mine:

1. Surprise summed over **categorical heads only**, so a cost outlier
   (numeric) or a date-order violation (dates) was invisible *by construction*.
2. The target was near-degenerate - **88.4%** of rows carry some defect,
   dominated by missingness.

**Both fixed** in `kaggle/parakh_consistency_train.py`. Numeric and temporal
heads now score into surprise, and evaluation is per channel:

| channel | n | ROC-AUC |
|---|---|---|
| cost_outlier | 171 | **0.699** |
| date_order | 142 | **0.608** |
| overspend | 113 | 0.596 |
| agency_mismatch | 113 | 0.548 |

Previously undetectable -> 0.699. Modest, and honest for the task.

`trainable_channels()` now rejects `missing` (61.7% prevalence)
automatically - the failure caught *before* a run instead of after.

---

## What is empirically validated vs assumed

| claim | status |
|---|---|
| Blocking preserves the match set | **VALIDATED** - parity, 33 tests |
| Indic normalisation unifies encoding variants | **VALIDATED** - 31 tests |
| Consistency model detects numeric/date defects | **VALIDATED** - AUC 0.699 / 0.608 |
| Compliance rules implement their statements | **VALIDATED** - 23 tests |
| Compliance citations are correct | **ASSUMED** - 0/5 verified |
| Typology signatures match the literature | **ASSUMED** - 0/5 verified |
| risk_score tracks conduct not paperwork | **UNTESTED** - single-source corpus |
| Ranking is stable under perturbation | **UNTESTED** - gate not run |
| risk_score corresponds to real outcomes | **UNTESTED and untestable here** |

---

## Before production

1. District- or work-level PMGSY financial data for FY2020-21 covering Nagaland (OMMAS).
2. Audit SCOPE, not just findings - which units were examined and cleared. Without scope there are no negatives and NULL is the only honest value.
3. Tender-level bid data (tender_id, bid_amount, is_winner, n_bidders) to make the bid-rigging typologies reachable at all.

Ranked by unlock value:

| acquisition | unlocks |
|---|---|
| **Tender-level bid data** | 3 typologies; the whole bid-rigging class |
| **GFR 2017 + CVC + CPWD** | 5 compliance rules can fire |
| **OMMAS district/work PMGSY, FY2020-21 Nagaland** | coverage overlap |
| **Audit scope** | negatives - and therefore calibration |

Only the last unlocks calibration. Without scope, absence of a finding is
indistinguishable from absence of an audit, and NULL remains the only
honest value.

**More audit PDFs do not help.**

---

See `STAGE8_FAILURES.md` for all 5 experiments including the 3 that failed,
and `APPROVAL_STATUS.json` for the machine-readable gate record.