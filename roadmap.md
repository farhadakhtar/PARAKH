# PARAKH — Implementation Roadmap

Progress tracker for the seven-stage build. Updated after every step.
Nothing is marked COMPLETE until its tests pass.

| Stage | Name | Status |
| ----- | ---- | ------ |
| 1 | Data Ingestion & Schema Layer | **COMPLETE** |
| 2 | Evidentiary Confidence Engine | **COMPLETE** |
| 3 | Clustering & Peer Formation | NOT STARTED |
| 4 | Duplicate & Entity Resolution | NOT STARTED |
| 5 | Risk Metrics | NOT STARTED |
| 6 | Risk Aggregation & Routing | NOT STARTED |
| 7 | Validation & Artifact-Invariance | NOT STARTED |

---

## Stage 1 — Data Ingestion & Schema Layer

**Status: COMPLETE**
**Tests: 157 passing, 0 failing** (`python -m pytest tests/test_stage1.py`)

### Components implemented

| Component | Where | Notes |
| --------- | ----- | ----- |
| Synthetic data generator | `src/stage1/data_generator.py` | 10 noise channels, single seeded RNG, ground-truth ledger |
| Schema definition | `src/stage1/schema.py` | 12 typed fields, `Record`, `NullReason`, frame alignment |
| Cleaning (normalisation only) | `src/stage1/cleaning.py` | Trim / collapse / lowercase / placeholder / currency / date parsing |
| Validation | `src/stage1/validation.py` | 12 issue codes across 2 severities, `ValidationReport` |
| Ingestion | `src/stage1/ingestion.py` | CSV + Parquet + DataFrame, malformed-row quarantine |
| Corpus | `src/stage1/corpus.py` | `head` / `summary` / `describe` / `missing_report`, valid & invalid **views** |
| Constants | `src/core/constants.py` | Every threshold and rate named once |
| Logging | `src/core/logger.py` | Idempotent, stderr + `logs/parakh.log` |
| Vectorised helpers | `src/utils/helpers.py` | Text, numeric and date primitives; no row loops |
| CLI | `main.py` | `generate → persist → ingest → clean → validate → report` |

### Files created

```
.gitignore
conftest.py
main.py
requirements.txt
roadmap.md
src/core/constants.py
src/core/logger.py
src/stage1/cleaning.py
src/stage1/corpus.py
src/stage1/data_generator.py
src/stage1/ingestion.py
src/stage1/schema.py
src/stage1/validation.py
src/utils/helpers.py
tests/test_stage1.py
```

Generated at runtime (gitignored, reproducible from `--seed 42`):
`data/synthetic_dataset.csv`, `data/ground_truth_ledger.json`,
`outputs/stage1_*.json`, `outputs/stage1_clean_dataset.csv`, `logs/parakh.log`.

### Tests added

157 tests in `tests/test_stage1.py`, organised against the PRD's own sections:

| Class | Covers | Count |
| ----- | ------ | ----- |
| `TestSyntheticGeneration` | sec.3.1 / 3.2 — size, determinism, every noise band, ledger hygiene | 26 |
| `TestIngestion` | sec.3.3 — CSV / Parquet / DataFrame equivalence, malformed rows, column handling | 14 |
| `TestSchema` | sec.3.4 — PRD type literal, nullability, dtypes | 6 |
| `TestCleaning` | sec.3.6 — normalisation, placeholders, date & numeric formats | 25 |
| `TestDefectPreservation` | sec.11 — no repair, no imputation, no clipping, no row loss | 7 |
| `TestValidation` | sec.3.5 — type / value / logical rules, report shape | 27 |
| `TestCorpus` | sec.3.7 / 5.3 — views, inspection API, typed records, persistence | 22 |
| `TestEdgeCases` | sec.7 — all six mandatory cases plus unicode and long text | 11 |
| `TestPerformance` | sec.4 — 50k row budget | 1 |
| `TestDeterminism` | sec.4 — reproducible reports | 2 |
| `TestAcceptanceCriteria` | sec.6 — one test per checkbox, plus closed-loop detection | 10 |

### Acceptance criteria (Stage1.md sec.6)

- [x] Synthetic dataset generated successfully — 20,000 records at `seed=42`, byte-identical across runs
- [x] Data loads without crashing — four entry points, all six mandatory edge cases under test
- [x] Schema validation works — type, value and logical rules, each with unit tests
- [x] Invalid records are detected — 16.6% invalid, matching the PRD's own worked example
- [x] Cleaning pipeline normalises values — and provably repairs nothing
- [x] Corpus object is usable downstream — exposes exactly what Stage 2 sec.3 requires

### Measured behaviour (n=20,000, seed=42)

| Metric | Value | PRD target | |
| ------ | ----- | ---------- | - |
| Missing cells | 13.37% | 10–20% | OK |
| Date-order violations | 6.45% | 5–10% | OK |
| Cost outliers | 5.00% | 5% | OK |
| Duplicate / near-duplicate names | 5.00% | 5% | OK |
| Valid records | 16,638 (83.2%) | ~84% in the PRD's example | OK |
| 50k ingest + clean + validate | 1.80s | < 5s | OK |

### Design decisions worth knowing

1. **The corpus retains every record.** `valid_records` / `invalid_records` are
   views. Filtering here would make Stage 2's REMEDIATE queue unreachable, since
   the records it must score are exactly the ones Stage 1 found defective.
2. **Three null causes stay distinct** — `missing`, `placeholder`,
   `unparseable`. Stage 2 treats an unparseable date (hard `C_temp` failure) and
   an absent date (completeness penalty) differently, so collapsing them into a
   bare `None` would destroy information.
3. **Missing fields are warnings, not errors.** Treating them as errors would
   mark ~81% of a realistically dirty corpus "invalid" and make the flag
   useless. Completeness is measured separately in `missing_fields`.
4. **CSV is read with `keep_default_na=False`.** pandas' default silently
   rewrites `"N/A"` and `"NULL"` into `NaN`, erasing the placeholder
   distinction. There is a dedicated regression test for this.
5. **No wall-clock anywhere.** `REFERENCE_DATE` is frozen and `CorpusMetadata`
   carries no `ingested_at`, so reports are byte-reproducible. A test asserts no
   timestamp key exists in `summary()`.
6. **The ground-truth ledger is a sidecar**, keyed by positional row index, and
   is never a corpus column — joining it in before Stage 7 would leak labels.

### Known issues / accepted limitations

1. **`status` is not consistent with date presence.** The base generator draws
   `status` independently and populates all three dates, so a `proposed` work
   can carry a completion date. This was chosen deliberately: deriving
   missingness from status would make the 10–20% noise budget unauditable.
   Stage 1 does not validate status-versus-date coherence. *If a later stage
   needs it, add it as its own labelled noise channel.*
2. **`describe()` reports `+inf` for `mean`/`std`** on an amount column
   containing an injected 1e300. That is arithmetically honest rather than a
   bug; the robust `median`/`p25`/`p75` beside it stay informative, and
   `n_extreme` counts the offenders. Consumers should prefer the percentiles.
3. **Malformed-CSV recovery is slow.** The tolerant re-read uses pandas' Python
   engine (roughly 5–10x slower than the C parser). It only triggers on files
   the C parser rejects outright.
4. **`to_typed_records()` costs ~0.9s per 50k rows.** It is opt-in; the
   vectorised `records` frame is the default path and stays within budget.
5. **The ledger JSON is large** (~10 MB at 50k rows) because it lists every
   defective row. It is gitignored and regenerable.
6. **Near-duplicate perturbation can collapse.** The truncation variant
   occasionally maps two distinct names onto the same string, so exact string
   duplicates slightly exceed the number of exact clones. Duplicate rates are
   therefore measured from the ledger, not from string equality.
7. **No real MPLADS export has been ingested.** The schema follows the PRD, not
   a live file. Column-name normalisation and the malformed-row path are the
   intended shock absorbers, but first contact with real data will likely
   surface new placeholder tokens — add them to `PLACEHOLDER_TOKENS`.

### Handoff to Stage 2

Stage 2 needs, and now has:

* `Corpus.records` — cleaned, typed frame, all rows retained
* `null_reason__<field>` columns — the `C_comp` field-validity predicate,
  already implemented as `Record.is_field_valid`
* `SCHEMA.date_fields` and `ORDERED_DATE_PAIRS` — the same pair definition
  `C_temp` must use, shared so the two stages cannot disagree
* `RECONCILIATION_PAIR` — the two amount columns `C_recon` compares
* `SCHEME_START_DATE` — the 1993 hard-fail boundary, already flagged per record
* `Corpus.iter_records()` — frozen `Record` objects with `None` (never `NaN`)

---

## Stage 2 — Evidentiary Confidence Engine

**Status: COMPLETE**
**Tests: 142 passing, 0 failing** (`python -m pytest tests/test_stage2.py`)
**Whole suite: 299 passing** (Stage 1 + Stage 2)

`C(r) = exp( w1·log C_comp + w2·log C_temp + w3·log C_recon )`

### Components implemented

| Component | Where | Notes |
| --------- | ----- | ----- |
| Entropy primitives | `src/stage2/completeness.py` | Bernoulli + Shannon, two normalisation modes |
| Field weights `v_f` | `src/stage2/completeness.py` | `(1-H_null)·H_value`, frozen and emitted for audit |
| `C_comp` | `src/stage2/completeness.py` | Defect-aware credit over the null-reason taxonomy |
| `C_temp` | `src/stage2/temporal.py` | Asymmetric decay + two hard-fail classes |
| `C_recon` | `src/stage2/reconciliation.py` | Symmetric normalisation, four branches |
| Log-space aggregation | `src/stage2/confidence.py` | Zero dominance, definedness renormalisation |
| `ConfidenceModel` / `ConfidenceReport` | `src/stage2/confidence.py` | Stage2.md sec.6.2 / sec.10 |
| Corpus integration | `src/stage2/confidence.py` | `attach_confidence`, index-checked |
| Calibration constants | `src/core/constants.py` | Stage 2 block; every parameter named once |
| CLI | `main.py` | Stage 2 section; `--stage1-only` to skip |

### Files created

```
src/stage2/__init__.py
src/stage2/completeness.py
src/stage2/temporal.py
src/stage2/reconciliation.py
src/stage2/confidence.py
tests/test_stage2.py
```

Modified: `src/core/constants.py` (new Stage 2 block), `main.py` (Stage 2 section).
Stage 1 source was **not touched** — `Corpus.records` returns a live frame
reference, so attachment needed no change to a locked stage.

### Tests added

142 tests, organised against Stage2.md's own sections:

| Class | Covers | Count |
| ----- | ------ | ----- |
| `TestEntropy` | sec.5.2 entropy primitives, both normalisations | 10 |
| `TestFieldWeights` | sec.5.2 `v_f`, degenerate fallback, coverage floor, freezing | 11 |
| `TestCompleteness` | sec.5.2 `C_comp`, defect ordering, no imputation | 12 |
| `TestTemporal` | sec.5.3 all five cases, both hard fails, definedness | 16 |
| `TestReconciliation` | sec.5.4 four branches, both normalisations, non-finite | 18 |
| `TestLogSpaceAggregation` | sec.5.1 zero dominance, underflow ranking, NaN guards | 13 |
| `TestConfigValidation` | Calibration rejected at construction | 7 |
| `TestFunctional` | sec.8.1 the four mandated cases | 7 |
| `TestEdgeCases` | sec.9 all four, plus inf/NaN and division-by-zero | 10 |
| `TestDistribution` | sec.8.2 sanity checks, sec.8.3 synthetic validation | 8 |
| `TestDeterminism` | sec.7 reproducibility, permutation invariance | 5 |
| `TestIntegration` | Attachment, alignment, order preservation | 9 |
| `TestPerformance` | sec.7, 50k under 3s | 1 |
| `TestAcceptanceCriteria` | sec.11, one per checkbox | 6 |

### Acceptance criteria (Stage2.md sec.11)

- [x] Confidence computed for all records — 20,000/20,000, no NaN
- [x] Component scores available — per-record breakdown plus evidence base
- [x] Edge cases handled correctly — every sec.9 case under test
- [x] Outputs stable and interpretable — byte-identical artefacts across runs
- [x] Works on the synthetic dataset — end to end from `main.py`

### Measured behaviour (n=20,000, seed 42)

| metric | confidence | completeness | temporal | reconciliation |
| --- | --- | --- | --- | --- |
| mean | 0.7425 | 0.9197 | 0.8934 | 0.6943 |
| median | 0.9057 | 0.9310 | 1.0000 | 0.8732 |
| sd | 0.2845 | 0.0670 | 0.2970 | 0.3314 |
| % exactly 0 | 6.38 | 0.00 | 5.81 | 0.60 |

`C < 0.2` = 7.29% · `C > 0.8` = 61.59% · 50k scored in **0.05 s** (budget 3 s).

**Ground-truth validation** — mean `C` grouped by the defect Stage 1 actually
injected. This is the strongest evidence that the engine measures what it
claims to:

| injected defect | n | mean C |
| --- | --- | --- |
| none | 2,318 | **0.9657** |
| missing value | 12,338 | 0.7133 |
| placeholder | 9,454 | 0.7032 |
| date-order violation | 1,800 | 0.4957 |
| unparseable value | 1,939 | 0.2619 |
| pre-scheme date | 200 | **0.0954** |

Monotone in defect count too: mean `C` falls 0.902 → 0.790 → 0.697 → 0.623 →
0.533 as nulls accumulate; `corr(n_null, C) = -0.359`.

### Assumptions

1. **`is_valid` / `issues` are deliberately not consumed.** Confidence is
   derived only from field state, so Stage 7's artifact-invariance test is not
   comparing two views that are correlated by construction.
2. **`kappa` is per day.** The parameter is dimensional; the same 0.01 applied
   to seconds would turn every soft penalty into a hard fail.
3. **`v_f` is corpus-level state.** Scores are deterministic given a fixed
   corpus but shift if the corpus changes, so weights are frozen, emitted with
   every score set, and accepted back via
   `ConfidenceModel.score(..., field_weights=...)` to keep batches comparable.
4. **No wall-clock anywhere**, matching Stage 1, so reports are byte-reproducible.

### Deviations from the literal spec, and why

| # | Decision | Reason |
| --- | --- | --- |
| 1 | Symmetric denominator; `max` available as a mode | Stage2.md sec.5.4 and README both specify symmetric. `max` is not sign-safe — with Stage 1's negative amounts, `max(-1000,-500,eps) = eps` explodes the ratio and underflows the score. Both are implemented and tested. |
| 2 | Graded credit kappa = (1.00, 0.20, 0.08, 0.00) | Required by the defect-awareness rule; the PRD's binary indicator is recovered exactly with kappa = (1,0,0,0). |
| 3 | **Undefined components are dropped and weights renormalised** | The most consequential change. sec.5.4 says "ignore component" when both amounts are null — in a geometric mean, *ignoring* means dropping the term and renormalising, not substituting 1.0. Without this a wholly empty record collects two vacuous perfect scores and lands at **C = 0.63**, failing the mandated "all missing → C ≈ 0". |
| 4 | `C_comp = 0` when no field is present | Credit for a defective cell is *residual* value, meaningful only against some actual evidence. With kappa(missing) > 0 the worst possible record would otherwise score 0.20. |
| 5 | Uniform-weight fallback when the weight sum is 0 | A corpus of identical *perfect* records would otherwise compute 0/0 and score zero. Covers sec.9's "identical values (no variance)". |
| 6 | Coverage floor tau = 0.02 | `(1 - H_null)` is non-monotonic — high at both p→0 *and* p→1 — so a field present 1% of the time earns weight 0.92 and uniformly depresses every score. Inert on this corpus (max p_null = 0.26); required on real data. |
| 7 | Non-finite amounts hard-fail `C_recon = 0` | `inf/inf = NaN` would propagate silently through the log-sum and poison the output rather than lowering it. |

### Known limitations

1. **`C_recon` compares a budget to an outcome.** `sanction_amount` is
   sanctioned; `amount_spent` is spent. They are not two measurements of one
   quantity, so a legitimate 30% underspend costs ~30% of `C_recon` at
   lambda = 2. The schema exposes no genuine second source. **lambda must be
   calibrated against the observed underspend distribution before deployment.**
2. **The one-sided penalty dominates a whole mode of the distribution.** The
   flat 0.2 fires on 21.4% of records and pins 4,255 of them into the
   `[0.5, 0.6)` histogram bin — 95.5% of that bin is the `one_null` branch. The
   value is Stage2.md's "e.g. 0.2", a suggestion rather than a derivation, and
   it is the single highest-leverage calibration target.
3. **`C_comp` has a narrow dynamic range** — observed [0.515, 1.0], sd 0.067,
   with a structural floor of 0.181 contributed by `work_id` alone (never null,
   18.1% of all weight). This is the surprisal design working as intended, and
   it is what stops the score encoding administrative capacity — but it means
   **theta_C must be calibrated on the empirical distribution**, never on an
   absolute intuition such as "0.5 means half the data is there".
4. **`H_value` is nearly inert under cardinality normalisation** — it spans only
   [0.67, 1.0] across fields, so in practice `v_f ~= (1 - H_null(f))`. The
   `"sample"` mode discriminates far more but makes scores corpus-size
   dependent. Measured impact on final `C` for this corpus: none to 3 dp.
5. **A field constant across the corpus cannot affect any score.** `H_value = 0`
   implies `v_f = 0`, so losing it costs nothing. Correct — its presence proved
   nothing — but surprising enough that an explicit test pins it down.
6. **Temporal coverage is reported, not priced.** 16.4% of records have no
   evaluable milestone pair and are dropped from the mean rather than scored,
   which is right; but a record checked on one pair and one checked on two are
   otherwise treated alike. `temporal_pairs_evaluated` is exposed for any later
   stage that wants to weight by evidence depth.
7. **Nothing is calibrated.** `w = (1/3, 1/3, 1/3)`, kappa, lambda and every
   credit are Stage2.md defaults. Per README the system is **non-operational**
   until these are estimated: the scores are structurally correct and
   operationally meaningless.

### Handoff to Stage 3

Stage 3 needs, and now has:

* `corpus.records["confidence"]` plus the three component columns, index-aligned
  and in original row order
* `ConfidenceResult.breakdown` — per-record components with the evidence base
  (`temporal_pairs_evaluated`, `reconciliation_branch`, `n_valid_fields`)
* `FieldWeights` — frozen `v_f`, serialisable and re-injectable for batch stability
* `outputs/stage2_confidence_scores.csv` — the full scored corpus

---

## Stage 2 Refinement — `stage2.confidence.v2`

**Status: COMPLETE** · **Tests: 188 Stage 2 (was 142), 345 total, 0 failing**

Model correction for real-world validity. No component was removed, no public
signature changed, and every v1 behaviour is recoverable through configuration.

### What was changed, and why it was needed

#### 1. `C_recon`: equality → financial plausibility

`sanction_amount` is a budget and `amount_spent` is an outcome. v1 compared
them with a symmetric difference, i.e. as two measurements of one quantity, so
correct budget execution registered as data unreliability:

| spend ratio band | share of comparable records | v1 mean `C_recon` |
| --- | --- | --- |
| `0.2 ≤ r ≤ 1.0` (normal execution) | **74.57%** | **0.8875** |
| `r > 1.0` (overspend) | 24.81% | penalised identically |
| `r < 0.2` (implausible) | 0.62% | penalised identically |

Being symmetric it also could not distinguish an overspend from an underspend
of the same size — discarding the one direction that carries signal.

v2 scores plausibility, asymmetrically:

```
r = spent / (sanction + eps)
C_recon = exp(-λ·max(0, r − 1)) · exp(-γ·max(0, 0.2 − r))     λ = 2.0, γ = 6.0
```

| r | 1.00 | 0.95 | 0.70 | 0.30 | 0.15 | 0.00 | 2.00 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `C_recon` | 1.000 | 1.000 | 1.000 | 1.000 | 0.741 | 0.301 | 0.135 |

#### 2. Component dominance

Measured on v1, reconciliation carried **69.4%** of all penalty mass, and its
flat `one_null` credit of 0.2 fired on 28.27% of records — capping them at
`0.2^(1/3) = 0.585` and manufacturing a 4,255-record artefact spike in
`[0.5, 0.6)` that was 96.1% one branch.

Fixed by weights `(0.4, 0.4, 0.2)` and `one_sided_credit` 0.2 → 0.7.

#### 3. `C_comp` structural floor

v1's `(1 − H_null)·H_value` weighting produced an algebraic floor of 0.3449
(observed min 0.5150, sd 0.0670) and contributed just **1.77%** of the variance
of `log C` — it was very nearly a constant. Two faults:

| | v1 | v2 |
| --- | --- | --- |
| `work_id` share (never null, proves nothing) | **18.11%** | **4.14%** |
| dates + amounts share (the evidentiary spine) | **30.56%** | **73.55%** |
| `vendor_name` (missing 26% of the time) | 2.12% | 2.80% |

Fixed by `v_f = criticality_f · H_value(f)`, plus a cluster penalty
`exp(-0.35·max(0, m − 1))` where `m` is the *fractional* critical-field
deficit — built from the same credit vector, so `missing < placeholder <
unparseable` flows into the cluster term as well.

**On artifact invariance.** v1 defended `(1 − H_null)` as stopping the score
encoding administrative capacity. That defence was misapplied: README §9 places
the artifact-invariance guarantee on **R**, not **C**, and low-confidence
records route to REMEDIATE rather than INVESTIGATE. Confidence is *supposed* to
track documentation quality. `H_value` is retained (it correctly zeroes
constant fields and keeps the degenerate-corpus fallback working).

### Impact on system behaviour

| | v1 | v2 |
| --- | --- | --- |
| mean `C` | 0.7425 | **0.8010** |
| median | 0.9057 | 0.8974 |
| sd | 0.2845 | 0.2692 |
| min / max | 0.0 / 1.0 | 0.0 / 1.0 |
| `C < 0.2` | 7.30% | 6.89% |
| `C > 0.8` | 61.60% | 69.26% |
| exactly 0 | 6.38% | 5.88% |
| `C_comp` min / sd | 0.5150 / 0.0670 | **0.1016 / 0.1761** |

Histogram — the artefact spike is gone and mass is spread across the middle:

| band | v1 | v2 | | band | v1 | v2 |
| --- | --- | --- | --- | --- | --- | --- |
| `[0.3,0.4)` | 470 | 274 | | `[0.7,0.8)` | 215 | **2,043** |
| `[0.4,0.5)` | 950 | 543 | | `[0.8,0.9)` | 1,992 | **3,971** |
| `[0.5,0.6)` | **4,255** | **415** | | `[0.9,1.0)` | 10,327 | 9,881 |
| `[0.6,0.7)` | 112 | **1,142** | | | | |

Ground-truth separation is preserved and sharper at the clean end:

| injected defect | v1 | v2 |
| --- | --- | --- |
| none | 0.9657 | **0.9941** |
| missing / placeholder | 0.713 / 0.703 | 0.777 / 0.763 |
| date-order violation | 0.4957 | 0.4911 |
| unparseable value | 0.2619 | 0.3513 |
| pre-scheme date | 0.0954 | 0.0979 |

### Balance: what actually improved, and what did not

Three metrics disagree, so all three are reported rather than the flattering one:

| dominance metric | v1 max | v2 max | |
| --- | --- | --- | --- |
| **mean penalty share** — which component takes confidence away on a typical record | recon **69.4%** | comp **45.8%** | **fixed** |
| binding constraint per record | recon 70.3% | comp 73.1% | shifted, arguably correct |
| variance of `log C` | temporal 50.3% | temporal **83.4%** | **worse** |

**The variance figure is honest and I am not tuning it away.** It is an
outlier effect: among non-zero records 93.34% have `C_temp = 1.0` exactly, and
the variance comes from a 6.66% tail whose log penalties reach −8 nats. That is
non-compensatory design working as intended — a 400-day backdating *should*
dominate that record's score. The metric that describes typical behaviour is
penalty share, and it improved from 69.4% to 45.8% with every component now
between 17.9% and 45.8%.

### Known limitations

1. **The spend ratio ignores lifecycle stage.** A *proposed* work legitimately
   has `amount_spent` at or near zero and is charged the underspend penalty for
   being normal. Stage 1 supplies `status`; v2 does not condition on it, to
   keep the components uncoupled. **This is the first thing to revisit on real
   data.**
2. **Overspend has no tolerance band.** `max(0, r − 1)` starts penalising at
   the first rupee over sanction, so rounding and routine price variation are
   charged. A tolerance `τ ≈ 0.05` belongs in the calibration set.
3. **Non-finite amounts no longer trigger zero-dominance.** Now 0.25 rather
   than 0.0, per the refinement brief. Stage 1 still marks these records
   invalid via `VALUE_NON_FINITE`, so the signal is not lost, but `C` alone
   will no longer refuse on them.
4. **Behaviour change: `sanction = spent = 0`** scored 1.0 under v1's equality
   reading ("zero equals zero") and scores 0.25 under v2's plausibility reading
   (no budget to have executed against). Covered by an updated test.
5. **Granularity fell**: 14,553 → 6,267 distinct scores, because `C_recon` is
   now exactly 1.0 across the whole normal band. That is deliberate — the lost
   discrimination was spurious — but it does mean ties are more common, which
   matters for any downstream rank aggregation.
6. **Criticality weights are a domain judgement, not an estimate.** The 0.20 /
   0.15 / 0.05 assignment is defensible but unvalidated; `δ = 0.35` and
   `γ = 6.0` likewise. All are named constants in `src/core/constants.py`.
7. **Still uncalibrated.** Per README the system remains non-operational until
   `θ_R`, `θ_C`, `λ`, `κ`, `γ`, `δ` and the weights are estimated against real
   data. v2 is more *correct*; it is not yet *operational*.

### Reversibility

Every change is a configuration option, and a test asserts the rollback works:

```python
ConfidenceConfig(
    weights=CONFIDENCE_WEIGHTS_V1, recon_mode="agreement",
    completeness_weight_mode="entropy", cluster_delta=0.0,
    one_sided_credit=0.2, non_finite_credit=0.0,
    non_positive_sanction_credit=1.0,
)   # reproduces stage2.confidence.v1 exactly
```

### Tests added (46 new, 8 updated)

| Class | Covers |
| --- | --- |
| `TestCriticalityWeighting` | weight basis, floor reduction, spread, mode reversibility |
| `TestClusterPenalty` | super-additivity, null-reason ordering in the deficit, disable switch |
| `TestComponentBalance` | penalty-share dominance bounds, artefact-spike removal |
| `TestRefinementInvariants` | API, zero dominance, no imputation, determinism, log-space, rollback |

Eight reconciliation tests were rewritten for the new semantics; two of those
encode deliberate behaviour changes (zero sanction, non-finite amounts).

---

## Stage 2 Final Corrections — audit response

**Status: COMPLETE** · **Tests: 238 Stage 2 (was 188), 395 total, 0 failing**

Three audit findings, all three previously self-reported as v2 limitations #1,
#2 and #3. Every change is confined to `C_recon`: `completeness.py`,
`temporal.py`, the `(0.4, 0.4, 0.2)` weights and the log-space aggregation were
not touched.

### 1. Lifecycle blindness (MAJOR) — `status` is now read

`status` appeared in Stage 2 only inside a docstring. The underspend penalty
applied unconditionally, so a *proposed* work with no expenditure — behaving
exactly as it should — was charged for it.

The gate scales `γ` per record, and **only** `γ`:

| lifecycle | `γ` scale | `C_recon` at r = 0 |
| --- | --- | --- |
| `proposed` / `approved` / `pending` / `ongoing` / `in progress` | 0.0 | **1.000** |
| `completed` / `closed` | 1.0 | **0.301** |
| unknown — null, placeholder, unparseable, unrecognised | 0.5 | **0.549** |

Overspend is **not** gated: spending past sanction is a control failure at any
stage. `pending`, `ongoing` and `in progress` are outside Stage 1's
`ALLOWED_STATUS`, so Stage 1 will flag them `VALUE_UNKNOWN_STATUS` — but
Stage 2 routes them correctly rather than penalising a work for a vocabulary
mismatch. An absent `status` column degrades to *unknown* rather than raising.

### 2. No overspend tolerance (MAJOR)

`max(0, r − 1)` charged from the first rupee over sanction, treating rounding
and final-bill adjustment as anomaly. Now `max(0, r − (1 + τ))`, τ = 0.05.

| r | 1.00 | 1.02 | 1.05 | 1.06 | 1.20 | 2.00 |
| --- | --- | --- | --- | --- | --- | --- |
| `C_recon` | 1.000 | **1.000** | **1.000** | 0.980 | 0.741 | 0.150 |

### 3. Weak refusal for non-finite values (MAJOR)

`RECON_NON_FINITE_CREDIT` was 0.0 in v1, became 0.25 under the v2 brief's
"strong penalty (<0.3)", and the audit found that too weak — 0.25 does not
trigger zero-dominance, so a record whose amount was literally infinite could
still produce respectable confidence. **Restored to 0.0.**

Extended, under Step 3's "extreme invalid numeric" clause, to amounts beyond
`IMPLAUSIBLE_AMOUNT_THRESHOLD` (1e15). These are *finite*, so they slipped past
the non-finite check entirely: a `1e300` sanction against a `1e300` spend gave
r = 1.0 and scored a **perfect** reconciliation on two data-entry accidents.
They now take their own branch, `implausible_magnitude`, and are refused on the
same terms. Stage 1 already flags them `VALUE_IMPLAUSIBLE_MAGNITUDE`.

### Distribution: before vs after

| | v1 original | v2 pre-audit | **v3 corrected** |
| --- | --- | --- | --- |
| mean | 0.7432 | 0.8017 | **0.7994** |
| median | 0.9061 | 0.8974 | 0.9023 |
| sd | 0.2844 | 0.2685 | **0.2768** |
| `C < 0.2` | 7.30% | 6.83% | **7.50%** |
| `C > 0.8` | 61.73% | 69.38% | 69.36% |
| exactly 0 | 6.38% | 5.81% | **6.53%** |
| distinct values | 14,553 | 6,257 | 4,649 |

Against the Step 6 targets: mean stays in the 0.70–0.80 band, low-confidence
share **increases** (6.83% → 7.50%), no collapse (sd rose to 0.2768), and no
artificial inflation — the mean actually moved slightly *down* despite 3,200
records being raised, because 144 were refused outright.

### Isolated effect of each fix

| fix | records moved | up | down | mean Δ |
| --- | --- | --- | --- | --- |
| 1 lifecycle gate | 22 | 22 | 0 | +0.00059 |
| 2 overspend tolerance | 3,178 | 3,178 | 0 | +0.00221 |
| 3 garbage refusal | 144 | 0 | 144 | −0.00516 |
| **all three** | **3,344** | 3,200 | 144 | −0.00237 |

**The lifecycle fix moves only 22 records here, and that is not evidence it is
unimportant.** Stage 1's generator draws `spend_ratio ~ N(0.93, 0.10)`
*independently of status*, so almost no synthetic record is both
pre-completion and below the 0.2 underspend floor — only 0.62% of the corpus
falls below that floor at all. The fix is therefore validated by 11 unit tests
across the full status vocabulary rather than by distribution shift. On real
MPLADS data, where proposed works genuinely carry near-zero expenditure, this
is expected to be the largest of the three corrections. Closing that gap is a
**Stage 1 generator** issue, already logged as Stage 1 known limitation #1.

Corpus lifecycle split: terminal 11,790 · pre-completion 6,410 · unknown 1,800.

### Behaviour changes encoded in updated tests

| test | change |
| --- | --- |
| `test_overspend_matches_the_closed_form` | penalty now measured from 1 + τ |
| `test_infinite_amount_is_refused` | 0.25 → 0.0 (was `..._takes_a_strong_penalty`) |
| `test_extreme_finite_amount_is_refused` | 1e300 vs 1e300 was 1.0, now 0.0 |

### Tests added (50 new, 3 updated)

| Class | Covers |
| --- | --- |
| `TestLifecycleAwareness` | all five pre-completion statuses, both terminal, four unknown forms, overspend never excused, absent column, vocabulary coverage |
| `TestOverspendTolerance` | inside/outside the band, exact boundary, closed form, monotonicity, disable switch |
| `TestGarbageRefusal` | non-finite and implausible both drive `C = 0`, zero-dominance restored, threshold does not swallow legitimately large works |
| `TestCorrectionInvariants` | weights, completeness, temporal, log-space, determinism, API, no imputation, no inflation |

### Known limitations after these corrections

1. **Granularity fell again**: 6,257 → 4,649 distinct scores, because the
   tolerance band puts more records at exactly 1.0. Deliberate — the lost
   discrimination was spurious — but ties matter for downstream rank
   aggregation in Stage 6.
2. **τ = 0.05, γ-scale = 0.5 and the status vocabularies are judgements**, not
   estimates. All are named constants and configurable.
3. **`RECON_NON_POSITIVE_SANCTION_CREDIT` remains 0.25**, deliberately: the
   audit did not flag it, and a zero budget is degenerate rather than garbage.
4. **Still uncalibrated.** Per README the system is non-operational until
   `θ_R`, `θ_C`, `λ`, `γ`, `κ`, `δ`, `τ` and the weights are estimated.

### Reversibility

Every correction is a config field; a test asserts each can be switched off:

```python
ConfidenceConfig(
    pre_completion_statuses=(), terminal_statuses=(),
    unknown_status_gamma_scale=1.0,      # disable the lifecycle gate
    overspend_tolerance=0.0,             # disable the tolerance band
    non_finite_credit=0.25, implausible_credit=1.0,   # restore the weak refusal
)
```

---

## Stage 2 Finalization

**Status: COMPLETE** · **Tests: 265 Stage 2 (was 238), 423 total, 0 failing**

Hardening only. **No formula, weight, constant or threshold was touched**, and
the score vector is byte-identical before and after — verified by hashing
`ConfidenceModel().score(corpus).scores` at every step
(`f85d33ad00081a03e8a0c9ca58e06432` throughout).

### 1. Breakdown exposure — 5 signals → 17

`attach_confidence` was writing 5 columns while the engine computed 16.
Eleven were discarded at the corpus boundary, so Stage 3 could see *that* a
record scored 0.4 but never *why*.

`BREAKDOWN_COLUMNS` is now the single source of truth, written identically by
`attach_confidence` and `ConfidenceResult.breakdown`:

```
confidence  completeness  temporal  reconciliation
completeness_defined  temporal_defined  reconciliation_defined  n_components_used
n_valid_fields  critical_missing_count  critical_deficit  cluster_penalty_factor
temporal_pairs_evaluated  temporal_hard_fail
reconciliation_branch  lifecycle_state  spend_ratio
```

`outputs/stage2_confidence_scores.csv` now exports all 17.

**`critical_missing_count` is new, and it is deliberately not the same as
`critical_deficit`.** The deficit is *fractional* — it sums `1 − κ` so the
missing/placeholder/unparseable ordering flows into the cluster penalty — and
is what the formula uses. The count is its human-readable companion. Three
placeholders are a count of 3 but a deficit of 2.76. A test asserts they differ.

Columns are only ever added to this tuple; removing one is a breaking change to
the downstream contract, and a test guards the original five.

### 2. Explainability — `explain_confidence`

Answers "why is this confidence what it is" from **stored outputs only**. It
never invokes a scorer, and raises rather than silently recomputing if the
breakdown columns are absent — an explanation derived from a fresh computation
could disagree with the stored score, which is worse than no explanation.

Returns per-component scores, effective weights after definedness
renormalisation, penalty attribution in nats, the evidence behind each
component, ordered human-readable reasons, and a one-sentence summary.
`ConfidenceResult.explain(row)` supplies its own weights.

The strongest test asserts the explanation **reconstructs the score exactly**:
`exp(Σ effective_weight · log score)` equals the stored confidence to 1e-6,
including records where a component was dropped and the weights renormalised
(e.g. row 42: 2 components used, effective weights 0.667 / 0.333). The
explanation therefore describes the aggregation actually performed, not a
plausible story about it.

### 3. Documentation drift — 5 contradictions resolved

| # | Location | Said | Actually |
| --- | --- | --- | --- |
| 1 | `reconciliation.py` module formula | `exp(-λ·max(0, r−1))` | **τ was missing entirely** |
| 2 | `reconciliation.py` "Remaining caveat" | *"v2 does not condition on status, so the components stay uncoupled"* | **documented a fixed bug as open** |
| 3 | `reconciliation.py` Args | `lam: Disagreement penalty rate` | λ is the **overspend** rate |
| 4 | `constants.py` `RECON_LAMBDA` | *"a 30% underspend costs ~30% of C_recon"* | a 30% underspend now costs **nothing** |
| 5 | `constants.py` `RECON_ONE_SIDED_CREDIT` | `0.2^(1/3) = 0.585` | true of the retired 1/3 weights; now marked HISTORY |

### 4. README — operating semantics

New section covering all five mandated topics plus the Stage 3 contract:
confidence interpretation (high C ≠ clean data), lifecycle awareness, tolerance
behaviour, garbage handling, and the calibration disclaimer stated verbatim:

> **Confidence scores are comparative, not absolute. The system requires
> real-world calibration to be operational.**

### 5. Stage 3 contract

Six guarantees, each asserted by test rather than merely written down: range,
monotonicity against ground truth, zero-means-reject, row alignment,
determinism, and breakdown availability.

> **Stage 3 MUST use the breakdown, not only the scalar confidence.**

Three cases where the scalar alone misleads:

1. `temporal = 1.0` means *coherent* **or** *nothing to check*. Only
   `temporal_pairs_evaluated` / `temporal_defined` separate them — 16.4% of the
   corpus has no evaluable milestone pair.
2. Two records at `C = 0` are not alike: a fabricated timeline
   (`temporal_hard_fail`) and an unreadable amount (`reconciliation_branch`)
   belong in different remediation queues.
3. A low `reconciliation` may be entirely legitimate — check `lifecycle_state`
   before treating a low `spend_ratio` as signal.

`n_components_used < 3` means a component was **dropped as unmeasurable** and
the weights renormalised, not that it scored badly.

### 6. Safe optimisation

`resolve_reasons` was called **3 times** per `compute_completeness_result`
(lines 319, 419, 539), each building a 12-column frame. Collapsed to one shared
pass threaded through `compute_field_weights` and `credit_matrix`. Score hash
unchanged; 50k end-to-end 0.167 s → **0.147 s** against a 3 s budget. This was
cleanliness, not need.

### Tests added (27 new)

| Class | Covers |
| --- | --- |
| `TestStage3Contract` | all 17 columns attached, breakdown/corpus agreement, alignment, serialisability, zero-means-reject, monotonicity, defined mask, idempotence |
| `TestExplainConfidence` | explanation matches computed values, **reconstructs the score exactly**, refuses to recompute, JSON-serialisable, refusal and lifecycle narration |
| `TestFinalizationInvariants` | dedup changed nothing, determinism, weights/constants untouched, log-space intact, no column removed |

### Known limitations carried forward

1. **Nothing is calibrated.** `w`, `κ`, `λ`, `γ`, `δ`, `τ`, `θ_u` and the
   null-reason credits are defaults, not estimates. Per README the system is
   non-operational until fitted. This is the single largest open item.
2. **Generator bias.** Stage 1 draws `spend_ratio ~ N(0.93, 0.10)`
   *independently of status*, so the lifecycle gate moves only 22 of 20,000
   synthetic records. It is validated by 11 unit tests across the full status
   vocabulary rather than by distribution shift, and is expected to matter far
   more on real MPLADS data. **Fixing this is Stage 1 work**, logged as Stage 1
   known limitation #1.
3. **`C_comp` is corpus-relative.** `v_f` is estimated from the corpus, so the
   same record scores differently in a different corpus. Weights are frozen,
   emitted with every score set, and re-injectable — but a consumer that
   forgets to pass them back will silently get incomparable scores.
4. **Ties.** 4,649 distinct scores over 20,000 records, because `C_recon` is
   exactly 1.0 across the whole normal band. Deliberate — the lost
   discrimination was spurious — but Stage 6's rank aggregation must handle
   ties explicitly.
5. **No real data has been scored.** Every number in this roadmap comes from a
   synthetic corpus.

### Stage 2 is closed

`stage2.confidence.v2` · 265 tests · 17-column contract · explainable per
record · deterministic · 50k in 0.147 s. Ready for Stage 3.

---

## Stage 3 — Peer Structure Layer

**Status: COMPLETE** · **Tests: 109 Stage 3, 532 total, 0 failing**
**Runtime: 0.80s at 20k, well inside `Stage3.md` §11's 10s budget**

Builds the comparison groups every downstream signal is measured against:

```
peer_cell = (semantic cluster k, cost stratum s)
```

Stage 3 produces **structure**. It ends at deviations from peer norms and does
not score or classify — a boundary asserted by `TestScopeBoundary`.

### Clustering method

| step | choice | why |
| --- | --- | --- |
| Embedding | **TF-IDF**, not a sentence transformer | Every dimension is a token you can name. `Stage3.md` §5.2 suggests `all-MiniLM-L6-v2`; for a system whose thesis is auditability, an explanation that says *"similar because both are 'cc road' works"* beats a 384-dim vector that can say nothing. Also deterministic, no download. |
| Normalisation | truncate at locality delimiter, strip geography + action boilerplate | See below — this is the load-bearing part |
| Projection | TruncatedSVD, 16 dims, seed 42 | Measured optimum: 8 dims → 0.704 purity, 16 → 0.924, 24+ → 27–41% noise |
| Clustering | **HDBSCAN** over *distinct texts* | No `k`, deterministic, explicit noise |
| Labels | top TF-IDF terms per cluster | Clusters form in SVD space, are **named in token space** — interpretability survives the reduction |

**Two decisions that carried the result.**

*Cluster distinct texts, not records.* Names are heavily templated: 20,000
records reduce to 91 distinct normalised strings. Clustering the strings and
broadcasting labels back took **0.08s vs 5.01s** at 20k, and **0.04s vs 27.0s**
at 50k — the difference between sitting inside the budget and missing it by
2.7×. It is also better statistics: identical text must get an identical cluster
anyway, and collapsing removes the artificial density spikes templated naming
creates in a density-based algorithm.

*Truncate the locality clause.* Every name reads
`"<action> <work type> at <locality>, <district>"`. Left whole, village names
split single work types across places — `"check dam"` became one cluster for
Peddapalli and another for Nandgaon — so **geography was silently becoming a
grouping feature**, which is exactly what the grouping/testing separation
forbids. District stripping cannot catch this: village names appear in no
district column, and where localities are spread evenly across districts no
statistical test distinguishes them from work-type tokens either. Position is
the signal, so position is used.

**Measured against generator ground truth** (a test-only oracle: the generator
built every name from a 20-item work-type vocabulary):

| n | clusters | noise | weighted purity |
| --- | --- | --- | --- |
| 5,000 | 13 | 11.2% | 0.802 |
| 10,000 | 16 | 5.0% | 0.819 |
| **20,000** | **17** | **7.6%** | **0.924** |
| 50,000 | 17 | 5.0% | 0.917 |

`cc road` / `bituminous road`, `school building` / `library building`,
`street light` / `solar street lighting` all separate correctly.

### Peer cell definition

`k` = cluster, `s` = quintile of `log(sanction+1)`; missing amount → `s = -1`.
`peer_cell_stable = size ≥ 15` (`Stage3.md` §8.1). At 20k: **108 cells, 71
stable, covering 78.6% of records**.

A cell built on `k = -1` is **forced unstable** however large: noise points are
not similar to one another, and 1,500 of them sharing a stratum is a bucket, not
a peer group.

### Confidence gating logic — the critical property

A cell's median and MAD are the yardstick every member is judged against. Let a
record with an unreadable amount or a fabricated timeline shape that yardstick
and **the corruption propagates to every honest record in the cell**.

The basis excludes `confidence < 0.5`, `reconciliation_branch ∈ {non_finite,
implausible_magnitude}`, and any non-finite or implausible amount. **86.7% of
records may shape a norm.**

Gated records are **not dropped**. They keep their cell and are measured against
the clean norm — they are the REMEDIATE population, and discarding them would
repeat Stage 1's silent-corruption mistake one layer up. They simply get no vote
on what normal looks like. `TestConfidenceGating` proves it end to end: twelve
garbage records at 100× the normal amount added to a cell move its median by
less than 0.2 in log space.

### Deviation preparation — **not** scoring

```
deviation = (x − median_group(x)) / (1.4826 · MAD_group(x))
```

Median and MAD only, never mean and σ — robust to 50% contamination (README §2),
because a cluster half-full of fraud must still yield a usable norm.

| deviation | level | defined at 20k | |p95| |
| --- | --- | --- | --- |
| `deviation_cell_cost` | (k,s) | 78.55% | 1.69 |
| `deviation_cluster_cost` | k only | 85.77% | 4.64 |
| `deviation_spend_ratio` | (k,s) | 62.22% | 2.00 |
| `deviation_duration` | (k,s) | 48.05% | 1.46 |

**Undefined is never zero.** A deviation is `NaN` with a recorded reason
(`feature_missing`, `cell_unstable`, `no_peer_norm`, `zero_dispersion`) whenever
it cannot be measured — reporting zero would say "exactly normal", the opposite
of what is known. Same rule Stage 2 enforces for its components.

Both levels are kept on purpose: stratifying is conservative and hides gross
cost inflation (an inflated work just lands in a higher stratum), so the
cluster-level view recovers the sensitivity. Its wider p95 is that difference.

### Three defects validation caught, and how

Validation against ground truth found problems that passing tests did not:

1. **18 records with `inf`/1e300 sanctions were shaping peer norms.** Stage 2
   labels a record `implausible_magnitude` only when *both* amounts are present;
   one with a 1e300 sanction and a missing spend is `one_null` and slipped the
   branch gate. Fixed with an explicit magnitude check → **0 remaining**.
2. **Duplicate detection was reusing the clustering vectors** — which have the
   locality stripped, so every road in a district looked identical. Precision
   0.047. Fixed by giving duplicates their own **untruncated, digit-preserving**
   embedding: `"ward no. 11"` and `"ward no. 35"` are otherwise identical after
   stopword removal, and the number is the entire distinction.
3. **`elapsed_seconds` in the serialised report broke byte-determinism** — my
   own no-wall-clock rule from Stages 1 and 2, violated. Removed from the
   report, kept on the result object.

### Files created

```
src/stage3/{embedding,clustering,stratification,peer_cells,features,
            deviations,duplicate_detection,explanation,pipeline}.py
tests/test_stage3.py
```
Modified: `src/core/constants.py` (Stage 3 block), `main.py` (Stage 3 section,
`--stage2-only`). Stages 1 and 2 were **not touched**.

### Stage 4 contract — 21 columns

```
cluster_id  cluster_size  cluster_is_noise
log_cost  cost_stratum
peer_cell_id  peer_cell_size  peer_cell_stable  peer_reference
duration_days
deviation_cell_cost(_reason)  deviation_cluster_cost(_reason)
deviation_spend_ratio(_reason)  deviation_duration(_reason)
duplicate_score  duplicate_flag  duplicate_group_id
```

`Stage4.md` §5 defines its cost outlier over `(k,s)` — that is
`deviation_cell_cost`, already computed here. **Stage 4 should consume it, not
recompute it.** `duplicate_score` is `Stage3.md` §9.4's `D_max`, which
`Stage4.md` §8 consumes.

### Known limitations

1. **No cross-lingual synonymy.** TF-IDF will not match "sadak" to "road". On
   code-mixed registers this is a real gap, accepted in exchange for
   auditability.
2. **Duplicate detection has no valid ground truth in this corpus.** Stage 1's
   duplicate channel clones names from *any* row, so only 70 of 1,000 injected
   clones land in the same district — and `Stage3.md` §9.1's `1[dᵢ=dⱼ]`
   deliberately excludes the rest. Precision is therefore validated as a
   *property* (every group shares a district and is temporally close) and by
   inspection: all 18 flagged groups are textbook near-duplicates, e.g.
   `"construction of cc road at village kishanganj, mehsana"` (2015-09-26) and
   `"renovation of cc road at village kishanganj, mehsana"` (2015-09-27).
   **Fixing this is Stage 1 generator work** — the channel should clone
   within-district.
3. **Cluster quality depends on corpus size** (0.80 at 5k → 0.97 at 50k),
   because the name vocabulary gets richer. A small register will cluster worse.
4. **Locality truncation assumes a naming convention.** A register that does not
   put the place after "at"/"in" loses the benefit, though it loses nothing
   relative to not truncating.
5. **`min_cluster_size` counts distinct texts, not records** — a genuinely
   different unit from `Stage3.md` §6.2's record-level floor, which
   `CLUSTER_MIN_RECORDS` enforces separately.
6. **Nothing is calibrated.** `PEER_STAT_MIN_CONFIDENCE`, `PEER_CELL_MIN_SIZE`,
   the duplicate threshold and τ are defaults, not estimates.

---

## Stage 3 Hardening — calibration, evaluation, reproducibility

**Status: COMPLETE** · **Tests: 158 Stage 3 (109 + 49 new), 581 total, 0 failing**

Purely additive. No clustering, deviation or gating math changed; a test asserts
the scores are identical with instrumentation on and off.

### 1. Calibration framework

`src/stage3/calibration.py` + `outputs/stage3_calibration_report.json`.

Every parameter is now declared with what it governs, **where it came from**,
the value used, and what goes wrong if it is wrong. Six of ten carry
`"source": "default"` — the explicit admission that nobody estimated them.

**Nothing was tuned**, and a test enforces it: every parameter must still equal
its default.

The distributions those defaults produce are now on paper:

| | value |
| --- | --- |
| cluster size | min 518, median 981, max 1,892 (17 clusters) |
| peer cell size | min 5, median 142, max 951 (108 cells) |
| stable cells | 71 of 108 (65.7%), covering 78.6% of records |
| reference records | 86.7% may shape a norm |
| `deviation_cell_cost` \|p50/p90/p95/p99\| | 0.68 / 1.34 / 1.69 / 11.95 |

That p99 of 11.95 against a p95 of 1.69 is the kind of thing calibration exists
to notice — a very thin, very heavy tail. The report labels these percentiles
**descriptive, not thresholds**, because Stage 4 lifting p99 as a flag boundary
would fit the cut to this corpus.

`ConfigSnapshot` saves the exact parameter set to `artifacts/stage3_config.json`
and reloads it.

### 2. Duplicate detection is finally measurable

`src/stage3/evaluation.py` + `outputs/stage3_duplicate_eval.json`.

The previous figures (precision 0.047, recall 0.119) were **measuring the wrong
thing**. Stage 1's duplicate channel clones names from *any* row, so only 70 of
1,000 injected clones share a district — and `Stage3.md` §9.1's `1[dᵢ=dⱼ]`
excludes the rest by design. The numbers described a definition mismatch, not
the detector.

The harness injects duplicates matching the detector's own definition: same
district, near-identical text (action verb swapped), within 30 days. Measured on
20,000 records with 300 injected pairs:

| metric | value |
| --- | --- |
| **precision** | **0.939** |
| **recall** | **0.920** |
| **F1** | **0.929** |
| injected median score | 0.920 vs corpus median 0.119 |

**`duplicate_id` cannot reach the pipeline.** It is returned as a separate
object, never a frame column — a structural guarantee rather than a convention,
asserted by test.

**Deviation from the brief, flagged:** the task said to extend the *Stage 1
generator*. I built the harness on top of Stage 1 instead. Stage 1 is locked and
157 tests depend on it, and post-generation injection is strictly stronger — a
hidden column could still be read by accident, a separate return value cannot.

### 3. Reproducibility contract

`src/stage3/artifacts.py` + `outputs/stage3_reproducibility_report.json`.

```
artifacts/tfidf_vocab.json     vocabulary + IDF, full precision
artifacts/cost_strata.json     quantile edges, log and rupee scale
artifacts/stage3_config.json   parameter snapshot
```

Default is compute-and-save; **reuse is opt-in**, because silently scoring a new
corpus against a stale vocabulary is worse than recomputing one. Drift is
measured (unseen-token rate, stratum-occupancy total-variation distance) and the
run is **rejected** beyond `MAX_UNSEEN_TOKEN_RATE` (0.35) or `MAX_STRATA_DRIFT`
(0.35).

**Measured: what freezing does and does not fix.**

| | reproduces? |
| --- | --- |
| cost stratum | exactly |
| cluster partition | exactly — **adjusted Rand index 1.0** |
| `cluster_label` | exactly |
| `cluster_id` (integer) | **no** |

`cluster_id` is run-local: HDBSCAN numbers clusters in an order that turns on
float ties at the 1e-16 level, so the integers permute even when the grouping is
bit-identical. **`cluster_label` was added to the contract (22 columns) as the
stable key**, and the reproducibility report says so explicitly.

### Three bugs this work exposed

Instrumentation found defects that a passing test suite had not:

1. **My own artefacts rounded to 10 decimal places.** A reproducibility artefact
   that rounds does not reproduce: the perturbed IDF shifted the SVD projection
   enough to move cost strata. Now saved at full precision.
2. **The duplicate injector misaligned its own ground truth.** Sources were
   recorded by slicing the candidate array, so skipping one unparseable date
   silently paired every later duplicate with the wrong original.
3. **The injector's usability filter used `notna()` on raw string dates**, so
   Stage 1's deliberate garbage (`"pending"`) passed as a valid date.

### Performance

`run()` measured at **0.838s** with artefact saving and **0.839s** without,
against a 0.80s pre-hardening baseline — within noise, and far inside
`Stage3.md` §11's 10s budget. The reproducibility contract costs nothing
measurable.

### Files added / modified

**Added:** `src/stage3/{calibration,artifacts,evaluation}.py`,
`tests/test_stage3_hardening.py`.
**Modified:** `src/core/constants.py` (hardening block),
`src/stage3/{embedding,stratification,pipeline}.py` (optional parameters only,
all defaulting to previous behaviour), `README.md`, `roadmap.md`,
`tests/test_stage3.py` (one assertion widened for the grown report set).
**Stages 1 and 2 untouched.**

### Known limitations after hardening

1. **Calibration is now possible, not done.** Every default is still a default.
   The system remains non-operational until fitted to real MPLADS outcomes.
2. **The duplicate evaluation measures the detector against duplicates it was
   designed to find.** Real double-claiming may look different — a rewritten
   description, a split across financial years. The 0.93 F1 is a floor on
   competence, not a claim about field performance.
3. **Freezing does not make cluster ids stable**, only the partition and the
   label. Anything keying on the integer will break across runs.
4. **Drift thresholds are themselves uncalibrated** (0.35 / 0.35), chosen to be
   permissive enough not to block legitimate corpus growth.
5. **Peer norms remain corpus-relative even with frozen artefacts** — a
   deviation of 2.5 means "unusual against *this* reference population". That is
   inherent to contextual anomaly detection, not a defect, but it means
   deviations are not comparable across corpora without pinning the population.

---

## Stage 3 Audit Remediation — M1–M4

**Status: COMPLETE** · **Tests: 614 total, 0 failing** (was 581; +33)
**Contract: 22 → 27 columns.** No formula, clustering decision or threshold changed.

### M1 — The noise cluster no longer defines a norm

`form_peer_cells` forced noise *cells* unstable, but `compute_peer_statistics`
had no equivalent guard at cluster level, so `cluster_id = -1` was emitting a
median and MAD pooled from records HDBSCAN judged similar to nothing.

| | before | after |
|---|---|---|
| `-1` row in `cluster_stats` | present, `n_reference` 1,331 | **absent** |
| records measured against the noise pool | 1,309 (7.6% of defined cluster deviations) | **0** |
| noise-pool MAD vs typical cluster | 0.803 vs 0.499 (**61% wider** → systematic under-flagging) | n/a |

Noise records keep `cluster_id = -1`, keep their peer-cell assignment, and now
report reason **`cluster_noise`** rather than the generic `no_peer_norm` — the
cause differs and so does the remedy. New column **`cluster_has_norm`** (False
for exactly the 1,524 noise records).

### M2 — The duplicate evaluation was invalid; it is withdrawn

> **The previously reported F1 of 0.929 is WITHDRAWN.** The harness perturbed
> only the *action verb*, which is a stopword in `normalize_work_text`, so the
> perturbation was erased before the detector saw it. Verified: **60 of 60**
> injected pairs were byte-identical in the detector's own text view. It
> measured exact-match retrieval, not near-duplicate detection.

Perturbations now act on tokens that **survive** preprocessing — a typo inside a
content word, a synonym swap, a dropped token — plus ±5–20% amount jitter and a
time shift. `assert_perturbations_are_real` checks the property directly.

| | before | after |
|---|---|---|
| identical after preprocessing | 60/60 | **0/282** |
| cosine(original, duplicate) | 1.000 for all | median **0.389**, max **0.897** |
| **F1** | 0.929 (invalid) | **0.020** |
| precision / recall | 0.939 / 0.920 | 0.143 / 0.011 |

**The detector is not broken — this is a representation limit, and the report
now says which.** Only **11.3%** of injected pairs reach the 0.85 cosine
threshold at all, so recall is bounded above by that before temporal decay even
applies. Work names normalise to ~3 content tokens, so a single-token change
costs most of the cosine. Recall by perturbation: truncate 0.034, typo 0.000,
swap 0.000. A control test confirms a genuine same-district, same-week, same-text
pair is still grouped correctly.

The detector itself was **not modified**, per the brief.

### M3 — Statistics gate on effective sample size

`n_reference` counted group membership; `_robust_stats` then independently
dropped non-finite values. A cell of 15 could emit a median from 2 points while
reporting `n_reference = 15`. Largest observed gap: **276**.

The guard now uses the **effective** count — finite values of the field actually
being summarised. Both are stored: `n_reference` and `<field>_n_effective`
(renamed from the ambiguous `<field>_n`, which was zeroed on withholding and so
conflated "not computed" with "no values"). 37 cells per field are now correctly
withheld.

### M4 — Extreme deviations flagged, never clipped

Raw values are **preserved**: a 1e300 sanction genuinely is thousands of MADs
from its peers, and truncating that would be the silent corruption Stage 1
exists to prevent. Metadata added beside it:

`<deviation>_bucket` ∈ `undefined` / `normal` (<5) / `high` (5–20) / `extreme` (≥20)

On the 20k corpus: 15,327 normal · 340 high · **43 extreme** · 4,290 undefined.
Max |z| = **3049.7, unchanged**. `undefined` is a bucket rather than a gap, so
the NaN-carries-a-reason rule holds here too.

### One bug introduced and caught

Skipping the noise key left `rows` empty on an **all-noise corpus**, and
`pd.DataFrame([]).set_index("cluster_id")` raised `KeyError`. Five edge-case
tests caught it; an `_empty_stats` guard returns the correctly-shaped empty
frame.

### Invariant check

- ✅ **No noise contributes to norms** — `-1` absent from `cluster_stats`; all its deviations NaN
- ✅ **Duplicate evaluation is non-trivial** — 0/282 identical post-preprocessing, max cosine 0.897 < 1.0
- ✅ **All statistics use effective sample size** — every emitted norm has `n_effective ≥ 8`
- ✅ No feature leakage into clustering (0 district/state/vendor tokens in vocabulary)
- ✅ No anomaly scoring introduced
- ✅ Stage 2's 17 columns byte-identical
- ✅ Determinism preserved; deviation formula recomputed by hand on 40 records
- ✅ Clusters 17, noise 7.62%, stable coverage 78.55% — **unchanged**

### Known limitations after remediation

1. **Duplicate detection has ~1% recall on realistic near-duplicates.** The
   honest number. Fixing it means changing the detector (out of scope here):
   character n-grams, a lower threshold, or blocking on locality.
2. **The perturbation may be harsher than reality** — a single-token change on a
   3-token name is a large relative edit. Real double-claims often repeat more
   text. The 11.3% reachable figure bounds the task, not the detector.
3. **`Z_EXTREME_THRESHOLD = 20` and `Z_HIGH_THRESHOLD = 5` are judgements**, not
   estimates, like every other Stage 3 parameter.
4. Audit findings **N1–N4 remain open**: 42 records where locality truncation
   fails, 26 junk vocabulary fragments polluting 2 of 17 cluster labels, the
   roadmap's earlier "six of ten defaults" miscount (actual: **3 of 11**), and
   steep small-corpus degradation.

---

## Stage 4 — Contextual Anomaly Interpretation

Stage 3 says *how far from its peers* a record sits. Stage 4 says *what that
means, given how much the record can be trusted*. It recomputes nothing: every
number it reads was produced upstream.

### Scope collision, and how it was contained

`Stage4.md` explicitly excludes "Final risk scoring" and "Routing decisions".
`Stage5.md` owns `R(r)`. `Stage6.md` owns INVESTIGATE / REMEDIATE / MONITOR /
CLEAR. The Stage 4 brief pulls both into Stage 4. The brief was followed, with
three containment choices so later stages are not pre-empted:

1. The composed number is **`severity_score`**, never `risk_score`.
2. The fourth decision class is **`INSUFFICIENT_CONTEXT`**, not Stage 6's
   `CLEAR` — the vocabularies stay distinct.
3. Every report labels the output **provisional triage**.

`Stage4.md`'s HHI and temporal-burst signals are **not** built: both require new
computation over raw data, which the brief forbids.

### Two documented mismatches with the brief

| Brief says | Upstream actually emits | Resolution |
|---|---|---|
| `is_duplicate` | `duplicate_flag` | Read the real column |
| lifecycle `completed` | `terminal` / `pre_completion` / `unknown` | `LIFECYCLE_TERMINAL_STATES` accepts `terminal`, `completed`, `closed` |

### The one rule the design turns on

**Undefined is not zero.** A signal is usable only when its value is finite
**and** Stage 3 recorded the reason `defined`. Both are checked rather than one
trusted: if they ever disagree the signal is dropped and the disagreement
counted, because a contract violation upstream must not become a silent anomaly
downstream.

Everything else follows from that:

- Severity is a weighted mean over the **valid** signals, renormalised per
  record, so a record measured on one signal is neither penalised nor flattered.
- Severity is **NaN, never 0**, when nothing was measurable. A record nobody
  could check has *unknown* severity, not low severity.
- Unmeasurable records route to `INSUFFICIENT_CONTEXT`, never `MONITOR` —
  "monitored" reads as *checked and fine*, which would be a lie.
- Every explanation carries a **"Not assessed:"** clause naming what could not
  be measured and why, ending with the sentence that absence of a signal means
  it could not be measured, not that it was normal.

### Confidence gates interpretation, not value

A low-confidence record keeps its deviations and its severity at **full
magnitude** — damping them would destroy what a remediator needs. What it
cannot do is escalate. Routing is a **precedence chain**, not a weighted score,
so no accumulation of deviations can outvote the gate:

```
confidence < 0.5              -> REMEDIATE              (applied last; wins outright)
no valid signal               -> INSUFFICIENT_CONTEXT
any |z| >= 3.5                -> INVESTIGATE
otherwise                     -> MONITOR
```

The threshold is `PEER_STAT_MIN_CONFIDENCE` deliberately: a record not trusted
to *shape* a peer norm is not trusted to be *accused* by one. The property is
asserted in code, not merely tested.

### Two design corrections the assertions forced

1. **The duplicate signal can no longer be the sole basis for a severity.** A
   record with no valid core deviation but a flagged duplicate was getting a
   severity number — the weak supporting signal driving the whole result,
   which is exactly what "duplicate is never a primary anomaly" forbids. It now
   keeps its `duplicate_suspect` type and its score, and has no severity.
2. **`insufficient_context` now also fires when the work type has no norm at
   all** (`cluster_has_norm == False`), alongside no-measurable-signal and
   unstable-cell. Three distinct ways to lack context, all worth saying aloud.

### Files created

| File | Contents |
|---|---|
| `src/stage4/anomaly.py` | contract check, signal validation, type classification |
| `src/stage4/decision.py` | severity composition, confidence-gated routing |
| `src/stage4/explanation.py` | per-record narrative; recomputes nothing |
| `src/stage4/pipeline.py` | `AnomalyLayer`, `attach_anomalies`, 13-column contract |
| `tests/test_stage4.py` | 130 tests |

### Behaviour on 20,000 records (seed 42), 0.98s

| Triage | n | % |
|---|---|---|
| MONITOR | 13,541 | 67.70 |
| INSUFFICIENT_CONTEXT | 3,402 | 17.01 |
| REMEDIATE | 2,638 | 13.19 |
| INVESTIGATE | 419 | 2.10 |

419 escalations, **none** on low-confidence evidence. Severity is defined for
79.22% of records; the other **4,156 have no severity at all** — undefined,
not zero.

Anomaly types (a record may carry several — no single-score collapse):
cost_outlier 499, underspend_anomaly 54, duplicate_suspect 36, overspend 24,
temporal_outlier 5, low_confidence 2,638, insufficient_context 4,290.

### Known limitations

1. **Nothing here is calibrated.** `Z_TYPE_THRESHOLD = 3.0`,
   `Z_INVESTIGATE_THRESHOLD = 3.5`, `Z_SEVERITY_SCALE = 5.0` and the four
   severity weights are **judgements, not estimates**. The 2.10% escalation rate
   is a consequence of those choices, not evidence for them.
2. **A lifecycle-gated underspend can still escalate on magnitude.** Routing
   reads `|z|`, so a pre-completion record with `z_spend = -9` is not *accused*
   of underspending but is still investigated. Deliberate — suppressing a
   label is not the same as suppressing evidence — and tested as such.
3. **`duplicate_suspect` inherits Stage 3's ~1% recall.** It is weighted lowest
   (0.10), cannot escalate alone, and cannot supply context. Its absence means
   almost nothing.
4. **17.01% of records reach no conclusion**, driven by Stage 3's deviation
   definedness (duration 48.05%, spend 62.22%). That is the data's fault, not a
   pipeline fault, and the system says so rather than defaulting them to safe.
5. **Explanations report the cell reason for cost** even when the cluster-level
   fallback was also unavailable; precise, but not exhaustive.

---

## Stage 4 Hardening — measurement, exposure, contract completion

No behaviour changed. The pass added instrumentation, made an implicit state
explicit, and made an unmeasurable signal measurable. Every pre-existing Stage 4
column is **byte-identical** with all measurement passes on and off, verified in
`TestNothingChanged`; the 20,000-record triage distribution is unchanged
(13,541 / 3,402 / 2,638 / 419).

### One place the brief contradicted itself, and how it was resolved

FIX 2 asks for three things that cannot all hold on one input:

1. `severity_defined = True` **only if** `cluster_has_norm == True`
2. `severity_defined == False` implies `severity_score is NaN`
3. existing outputs preserved exactly

A record with a defined deviation but no cluster norm satisfies at most two.
Enforcing rule 1 would blank an already-computed severity, breaking rule 3 —
in a brief whose first success criterion is that nothing changes.

**Resolution:** `severity_defined` is read off the severity that was already
computed, so rule 2 holds *by construction* rather than by enforcement. Rule 1
is checked against it and any divergence is counted, logged and reported
(`rule_divergence`) — never silently resolved. On real data the divergence is
**0**, and structurally must be: `cluster_has_norm` is False only when no
cluster-wide median exists for any metric, and a peer cell is a subset of its
cluster, so no cell deviation can be defined either.

### FIX 1 — calibration instrumentation

`src/stage4/calibration.py`, `compute_stage4_calibration_report(df)`. Purely
descriptive; the module refuses the obvious temptation — reading a p95 back in
as a threshold would let the corpus define its own normality, so a corpus with
systematic fraud would calibrate that fraud into the baseline.

Every statistic reports its own `count_defined`, and an empty input returns
`None`, never `0.0`, because 0.0 reads as a measured value.

**What it immediately exposed.** A single z threshold is applied to three
signals with completely different tails:

| signal | n defined | abs z p50 | p95 | p99 | at 3.0 | at 3.5 |
|---|---|---|---|---|---|---|
| cell_cost | 15,710 | 0.676 | 1.687 | 11.951 | 2.105% | 2.035% |
| spend_ratio | 12,444 | 0.675 | 2.004 | 2.858 | 0.540% | 0.350% |
| duration | 9,609 | 0.674 | 1.462 | 1.725 | 0.025% | 0.015% |

Two things fall out of this table that were invisible before:

1. **`Z_TYPE_THRESHOLD = 3.0` is a ~p98 cut for cost and a ~p99.97 cut for
   duration.** The same nominal threshold is roughly **100x more selective** on
   one signal than another. Whether that is right is a judgement — but it was
   being made without anyone seeing it.
2. **The gap between the type and investigate thresholds barely discriminates
   for cost** (2.105% vs 2.035%): almost everything that earns the label also
   escalates. For duration the gap does most of the work.

A third observation is reassuring rather than alarming: abs-z p50 is 0.675 on
all three signals, and 0.6745 is exactly the median abs-z of a standard normal.
Stage 3's median/MAD scaling is behaving as designed.

### FIX 2 — explicit severity definedness

Two columns added, `severity_defined` and `severity_defined_reason`, over the
exhaustive vocabulary `ok / no_peer_norm / cluster_noise / no_valid_deviation /
insufficient_features`. A NaN severity previously forced the consumer to guess
whether the record was unmeasurable, noise, or absent from the peer structure.
Those have different owners and different fixes:

| reason | n | who fixes it |
|---|---|---|
| ok | 15,844 | — |
| no_peer_norm | 1,574 | corpus structure |
| cluster_noise | 1,524 | naming / clustering |
| insufficient_features | 1,058 | data entry |

Four assertions now hold on every run: undefined implies NaN, defined implies
not NaN, defined implies reason is `ok`, undefined implies reason is not `ok`.

### FIX 3 — duplicate observability

`compute_duplicate_diagnostics(df)`. The detector is untouched: same embedding,
same blocking, same similarity, same 0.85 threshold. `DUPLICATE_SIMILARITY_THRESHOLD`
already existed and is used everywhere — a test now asserts no literal `0.85`
survives in `duplicate_detection.py`.

The detector scores `cosine x 1[same district] x exp(-dt/tau)` and retains only
the product, so *too dissimilar in text* and *too far apart in time* are
indistinguishable downstream. The diagnostics separate them, over 106,592
candidate pairs:

| cosine cut | pairs |
|---|---|
| 0.60 | 1,406 |
| 0.70 | 1,406 |
| 0.80 | 1,404 |
| 0.85 | 1,404 |
| 0.90 | 1,404 |

**The distribution is effectively binary.** Two pairs — out of 106,592 — sit
anywhere in the band [0.60, 0.90). Dropping the cosine threshold from 0.85 to
0.60 would gain **two pairs**. Threshold tuning cannot fix duplicate recall, and
now there is a number saying so rather than an intuition.

**Where the recall actually goes.** 1,404 pairs clear 0.85 on text alone;
**1,386 of them (98.72%) are then killed by the temporal decay term**. With
`tau = 180` days, a cosine-1.0 pair needs a gap of 29.3 days or less to survive.
2,532 records have a near-identical partner by text; 36 are flagged.

This does **not** overturn the earlier M2 finding — it sits beside it. They
measure different populations:

- **Injected near-duplicates** (M2 harness, perturbed text): fail on **text**;
  only 11.3% reach 0.85 cosine.
- **Naturally occurring pairs** (this measurement, mostly identical text): clear
  the text bar easily and fail on **time**.

Two independent bottlenecks, two different fixes. Neither was applied.

`duplicate_reachable` (cosine at or above 0.60) and `duplicate_cosine` are
attached when diagnostics run. The invariant *flagged implies reachable* holds
by construction — the decay factor lies in [0,1], so the blended score can
never exceed its own cosine — and is asserted rather than assumed.

### Files

| File | Change |
|---|---|
| `src/stage4/calibration.py` | **new** — calibration report, duplicate diagnostics |
| `src/stage4/decision.py` | `severity_definedness()` added; no formula touched |
| `src/stage4/pipeline.py` | 2 contract columns, 2 config flags, 6 invariants, 2 reports |
| `src/core/constants.py` | hardening block; no existing constant altered |
| `main.py` | `--duplicate-diagnostics`; two new report sections |
| `tests/test_stage4_hardening.py` | **new** — 67 tests |

811 tests pass (744 + 67). The 130 existing Stage 4 tests were **not modified**.

### One bug this pass introduced and caught

`records = diagnostics["records"]` in the new CLI block shadowed
`records = corpus.records`, crashing the worked-examples section — but only
under `--duplicate-diagnostics`, which is why the default path stayed green.
Caught by running the flag end-to-end rather than trusting the test suite.

### Did any metric improve?

**No.** Every decision, severity and flag is byte-identical. The only figures
that moved are ones that did not exist before. This was checked, not assumed:
`test_pre_hardening_columns_are_byte_identical` compares all 13 pre-existing
columns with instrumentation on and off.

### Known limitations after hardening

1. **Still nothing is calibrated.** The report makes the thresholds arguable; it
   does not make them right. `Z_TYPE_THRESHOLD`, `Z_INVESTIGATE_THRESHOLD`,
   `Z_SEVERITY_SCALE` and the four severity weights remain judgements.
2. **The per-signal selectivity gap above is unaddressed.** Fixing it means
   per-signal thresholds — a behaviour change, out of scope here.
3. **Duplicate recall is measured, not fixed**, as instructed. The measurement
   says the fix is `tau` or the blending rule, not the cosine threshold.
4. **`DUPLICATE_REACHABLE_THRESHOLD = 0.60` is itself a judgement.** It is a
   diagnostic cut with no effect on detection, but it is not derived.
5. **The diagnostics rebuild Stage 3's duplicate embedding** because Stage 3
   does not retain it. Deterministic and identical, but it is recomputation —
   accepted because it produces no pipeline output.
6. Stage 3 audit findings **N1–N4 remain open**.

---

## Stage 5 — Risk Scoring Layer

Stage 4 says what deviates and how far. Stage 5 says how much that is worth
acting on, **given how much of it can be believed**. No record is labelled
fraud; risk is an estimate under uncertainty, and the bands are descriptive —
Stage 6 owns routing.

### The score

```
risk_score = signal_strength × data_quality × (1 − uncertainty)
```

A product, not a sum, and that is the whole argument. A sum lets a strong
anomaly compensate for unreadable data — exactly the inference this system
exists to refuse. A product collapses toward zero the moment any factor is
weak, so a serious-looking finding on a record nobody can verify scores low.
Not because the finding is dismissed, but because *risk* is a claim about what
is worth acting on, and acting on unverifiable evidence is not. That record is
a remediation case, and Stage 4 already routes it as one.

The three components are **always reported alongside the score**, never
collapsed. "Risk 0.08" on its own is unusable: it could be a clean record, a
filthy record nobody can read, or a real finding on a corpus with no peers.
Those need three different responses.

### Signal strength — boosts fill headroom, they do not add

```
strength = severity + (1 − severity) × (0.20·breadth + 0.30·extreme + 0.10·duplicate)
```

Chosen over a weighted sum for three reasons: it stays inside [0,1] without
clipping, it is **strictly increasing in severity** so Stage 4's ordering is
never inverted, and a boost can never *lower* a score. It also bounds the
duplicate contribution at `(1 − severity) × 0.10 ≤ 0.10` — the cap the design
requires, and the right one given Stage 3's measured ~1% duplicate recall.

`low_confidence` is deliberately **excluded** from breadth. It describes the
evidence, not the work, and it is already priced into the data-quality term;
counting it in both places would charge a record twice for one defect.

### Data quality — non-compensatory, like Stage 2

```
quality = confidence × min(defined Stage 2 components) × exp(−0.5·critical_deficit)
          × cluster_penalty_factor
```

then floored at 0.05 where the dates are internally impossible. The component
term is a **minimum, not a mean**: Stage 2's whole philosophy is that a zero
component dominates, and averaging would let a perfect completeness score paper
over a broken reconciliation. An undefined component is skipped, never scored
1.0 — that is the vacuous-perfection bug Stage 2 was built to avoid.

**Known double count, applied deliberately.** `critical_deficit` and
`cluster_penalty_factor` are *already inside* Stage 2's `completeness`, so
applying them again charges the same defect twice. The design names all three
as inputs and the direction is conservative — it lowers risk on poor records,
never raises it — so it is applied as specified rather than silently dropped.
Flagged here because it is a real modelling choice, not an oversight.

### Uncertainty — additive, because these are independent ways of not knowing

| contribution | weight |
|---|---|
| severity undefined | 1.00 (saturates alone) |
| work type has no norm | 0.60 |
| peer cell unstable | 0.25 |
| missing signal coverage | up to 0.30 |
| flagged duplicate that was unreachable | 0.40 |

Additive rather than multiplicative so defects accumulate: no norm *and* one
signal *and* an unstable cell is worse than any one of them.

### A term that provably cannot fire

The last uncertainty contribution — flagged duplicate, not reachable — is
**empty by construction**. Stage 4's temporal decay lies in [0,1], so a blended
score above the 0.85 detection threshold implies a cosine above the 0.60
reachability cut: flagged ⇒ reachable, always. Measured count on 20,000
records: **0**.

It is implemented anyway, and counted, because the day it stops being empty is
the day Stage 3 and Stage 4 have diverged — and a risk score should say so
rather than quietly absorb the contradiction.

### The gate

A score exists only where severity is defined **and** confidence clears 0.5
**and** the work type has a norm. Otherwise the score is **NaN with a stated
reason**, never 0.

The third conjunct is **structurally redundant** — no norm ⇒ no defined
deviation ⇒ no severity, so the first conjunct always fires first. Measured:
it binds **0 times**. Kept as defence in depth, and the diagnostics report how
often it actually binds so the redundancy stays visible rather than assumed.

### Behaviour on 20,000 records (0.46s)

| | n | % |
|---|---|---|
| low_risk | 12,680 | 63.40 |
| insufficient_data | 6,040 | 30.20 |
| moderate_risk | 1,112 | 5.56 |
| high_risk | 168 | 0.84 |

Risk defined for 69.80%; median 0.081, p95 0.238, p99 0.567, max 0.727. The
6,040 unscored records split 4,156 severity-undefined and 1,884
confidence-below-gate — **no score, which is not a low score**.

Spearman correlations, which are the numbers that say whether the composition
does what it claims:

| relationship | ρ |
|---|---|
| risk vs severity | +0.630 |
| risk vs confidence | +0.684 |
| risk vs data quality | +0.698 |
| risk vs uncertainty | −0.604 |

Risk tracks severity and is attenuated by weak evidence, as designed. That
confidence correlates *more strongly* than severity is worth stating plainly:
**on this corpus, risk is driven more by how readable a record is than by how
anomalous it is.** That is the intended philosophy taken to its conclusion, and
it is also a warning — on a cleaner corpus the ordering would reverse, so this
number should be re-read whenever the input data changes.

The product retains a median **0.523** of the raw signal. Roughly half of every
finding is discounted as the price of conditioning on evidence.

### Two bugs found and fixed during the build

1. **`np.nanmax` warned on all-undefined rows**, violating invariant 5 ("no
   RuntimeWarning"). Caught by running the pipeline under
   `warnings.filterwarnings("error", RuntimeWarning)` rather than by a test.
   Replaced with a `-inf` substitution — equivalent, silent, and a row with no
   defined z correctly scores 0 on the extreme bucket.
2. **The explanation contradicted its own arithmetic.** It read anomaly types
   from `type_*` columns, which Stage 4 computes but does **not** attach — only
   the `anomaly_types` list is in the contract. So on real data the narrative
   claimed "deviations stayed within normal range" while simultaneously
   reporting a breadth boost from findings it had failed to see. It also called
   a single finding "several". Both fixed; a test now asserts the text can
   never claim a breadth boost the stored `risk_breadth` does not support.

A third gap was fixed upstream: **Stage 4 computed `duplicate_reachable` and
`duplicate_cosine` and then stranded them on the result**, never attaching them
to the corpus. Now attached when the diagnostics run, as `OPTIONAL_STAGE4_COLUMNS`.

### Files

| File | Contents |
|---|---|
| `src/stage5/components.py` | contract check, signal strength, data quality, uncertainty |
| `src/stage5/risk.py` | gating, composition, banding, invariants |
| `src/stage5/explanation.py` | narrative that reconstructs its own arithmetic |
| `src/stage5/calibration.py` | descriptive report, rank correlations |
| `src/stage5/pipeline.py` | `RiskLayer`, `attach_risk`, 8-column contract |
| `tests/test_stage5.py` | 128 tests |

939 tests pass (811 + 128). `main.py` gains `--stage4-only`.

### The explanation must show the multiplication

Every scored record's narrative prints the arithmetic:

> Risk 0.498 (moderate risk), composed as signal 1.000 × data quality 0.906 ×
> stability 0.550.

0.498 = 1.000 × 0.906 × 0.550. A reader who does not trust the number can check
it from the sentence without opening the code — and a whole class of bug becomes
impossible to hide, because a drifting explanation would not close arithmetically.
A test asserts the product holds on every defined record to 1e-12.

### Known limitations

1. **Nothing is calibrated.** Every weight, band and decay rate is a judgement:
   `R_HIGH = 0.50`, `R_LOW = 0.20`, `RISK_CRITICAL_DEFICIT_DECAY = 0.5`, the
   three signal weights, the five uncertainty weights. They were chosen before
   the distribution was measured and were **not** adjusted afterwards. The
   0.84% high-risk rate is a consequence of those choices, not evidence for them.
2. **The multiplicative form is structurally deflationary.** Three factors each
   below 1 compound: the median score retains 52% of its raw signal, and the
   maximum observed risk is 0.727 — nothing reaches 1.0 in practice. If a
   future reviewer wants `R_HIGH` to mean "top decile", that is a threshold
   argument, not a formula argument, and it should be had in the open.
3. **Confidence outweighs severity in driving the score** (ρ 0.684 vs 0.630).
   Philosophically intended; empirically worth re-checking on cleaner data.
4. **The double count in data quality** (see above) is applied as specified.
5. **30.20% of records receive no risk score at all.** That is the data's fault,
   not the layer's, and the system says so rather than defaulting them to safe.
6. Stage 3 audit findings **N1–N4 remain open**; duplicate recall is still ~1%.

---

## Stage 5 Audit — one structural fix, two refusals

Verdict: **CONDITIONAL PASS**. One critical flaw found, proven and fixed. Two
mandated changes refused with proof, because applying them would have broken
guarantees the same brief required preserving.

### CRITICAL — data_quality double-counted Stage 2, two to four times over

`quality = confidence × component_floor × exp(−0.5·critical_deficit) ×
cluster_penalty_factor`. All three modifiers are *already inside* confidence:
`critical_deficit` and `cluster_penalty_factor` are inputs to `C_comp`, and the
component floor is built from the very three scores confidence aggregates.

Proof:

| measurement | value |
|---|---|
| `component_floor == completeness` | 13,795 / 20,000 (68.97%) |
| `deficit_factor < 1.0` | 12,419 (62.09%) |
| median quality, as shipped | 0.495 |
| median quality, duplication removed | 0.754 |
| **median suppression** | **34.29%** |

Consequence, and the reason this was critical rather than cosmetic: it inverted
the ordering the layer exists to produce. Risk correlated more strongly with
**confidence (0.684)** than with **severity (0.630)**.

**Fix — a minimum, not a re-weighted product:**

```
quality = min(confidence, completeness, temporal, reconciliation)   # defined only
```

`min` is **idempotent**, so a quantity appearing twice contributes exactly once.
That fixes the class of bug structurally rather than by tuning a weight: a
product would need every input audited for overlap forever. It is also strictly
non-compensatory — the property the product was reached for in the first place.
`critical_deficit` and `cluster_penalty_factor` are no longer applied; both are
still **reported** so an auditor reading a low score can still see them.

### The fix resolved the dominance imbalance without touching a weight

| metric | before | after |
|---|---|---|
| corr(risk, severity) | 0.630 | **0.888** |
| corr(risk, confidence) | 0.684 | **0.367** |
| log-variance: signal | 59.34% | **92.06%** |
| log-variance: quality | 40.07% | 7.02% |
| log-variance: stability | 0.59% | 0.92% |

No reweighting, no transformation, no capping was needed. The imbalance was a
symptom of the double count, not an independent defect — which is why removing
the cause was the right move and tuning a weight would have masked it.

### REFUSED 1 — "signal_strength == 0 → risk MUST be NaN"

Five records have `signal_strength == 0`. Their profile: `severity_defined =
True`, `valid_signal_count ≥ 1`, confidence 0.738–0.771, `anomaly_count = 0`.

They were **measured, and found clean**. Reporting NaN would assert "could not
be assessed", which is false, and would destroy the Stage 4/5 distinction that
the whole system rests on — `0` means measured-and-normal, `NaN` means unknown.
The brief that mandates this invariant also mandates not breaking prior
guarantees; on this input the two conflict, and the prior guarantee is the one
worth keeping. Not applied. A test documents the reasoning at the point of
refusal.

### REFUSED 2 — a decompressing transform for the max<0.9 trigger

After the fix, `p99 = 0.605` clears its trigger. `max = 0.730` still does not.

Option A (geometric mean, `(S·Q·(1−U))^(1/3)`) was measured rather than argued
about: it exceeds the record's own signal strength on **13,811 of 13,960 scored
records (98.93%)**, worst case inflating a signal of 0.192 to 0.577. It
directly violates "never inflates weak evidence". Options B and C decompress by
the same mechanism.

The residual ceiling is a **fact about the corpus, not the formula**: no record
simultaneously has maximal signal, perfect evidence and full coverage. A
synthetic record that does scores exactly **1.0**, and there is a test proving
it. Rescaling to reach 0.9 would manufacture a number no record earns. Not
applied.

### MAJOR — three of five uncertainty terms cannot fire inside the score

Measured on the **scored subset**, which is the only place uncertainty affects
anything:

| term | fires corpus-wide | fires on scored records |
|---|---|---|
| coverage | 61.89% | 32.52% |
| unstable_cell | 21.45% | 0.60% |
| no_severity | 20.78% | **0.00%** |
| no_norm | 7.62% | **0.00%** |
| unreachable_duplicate | 0.00% | **0.00%** |

`no_severity` and `no_norm` are redundant with the gate — it excludes exactly
the records that would trigger them. `unreachable_duplicate` is provably empty.
Effective uncertainty on a scored record is therefore
`0.25·unstable + 0.30·(1 − coverage/3)`, bounded at 0.55, carrying 0.92% of the
log-variance.

Retained as invariant guards — they are **alive in the reported column**, which
covers every record including unscored ones, and they would fire if the gate
were loosened or if Stage 3 and Stage 4 diverged. Their deadness is now
**published** in `uncertainty_liveness` rather than assumed.

### MINOR — the explanation attributed quality to a factor it no longer uses

After the fix the narrative still said "Data quality 0.906 follows from ...
critical fields missing (factor 0.61)". Corrected: quality is now described as
"the lowest of ..." and the deficit is reported separately as "already inside
its confidence and reported here rather than charged again".

### TASK 7 — calibration honesty

Both Stage 5 reports now carry `CALIBRATION_STATUS_BANNER`:

> **UNFIT FOR PRODUCTION — NOT CALIBRATED.** Every threshold and weight in
> Stage 5 is a stated judgement; none has been fitted to, or validated against,
> real outcomes. The system has only ever been run on synthetic data with
> injected defects, so no number here estimates a real-world rate of anything.

`R_HIGH = 0.50`, `R_LOW = 0.20`, `MIN_CONFIDENCE_FOR_RISK = 0.5` are unchanged
from before the audit, and a test asserts they are still the round untouched
constants — a tuned threshold would show up there as a non-round number.

### Before vs after

| | before | after |
|---|---|---|
| risk p50 | 0.0812 | 0.1137 |
| risk p95 | 0.2376 | 0.2581 |
| risk p99 | 0.5668 | **0.6053** |
| risk max | 0.7271 | 0.7304 |
| high_risk | 168 (0.84%) | **291 (1.46%)** |
| moderate_risk | 1,112 | 1,452 |
| low_risk | 12,680 | 12,217 |
| insufficient_data | 6,040 | 6,040 (unchanged) |
| corr(risk, severity) | 0.630 | **0.888** |

Spearman(risk_before, risk_after) = 0.892 — deliberately **not** 1.0. This was a
correction, not a rescale; records penalised differently for the same defect
change rank relative to one another, which is the fix working.

Gating is untouched: the same 6,040 records are unscored, for the same reasons.

### Files

| File | Change |
|---|---|
| `src/stage5/components.py` | `data_quality` is now a minimum; uncertainty firing rates published |
| `src/stage5/explanation.py` | no longer attributes quality to the deficit factor |
| `src/stage5/calibration.py` | banner, `uncertainty_liveness` section |
| `src/stage5/pipeline.py` | banner on the risk report |
| `src/core/constants.py` | `CALIBRATION_STATUS_BANNER` |
| `tests/test_stage5_audit.py` | **new** — 50 tests |

989 tests pass (939 + 50). The 128 existing Stage 5 tests were **not modified**.

### Known limitations after the audit

1. **Still not calibrated**, and now says so on every report.
2. **`RISK_CRITICAL_DEFICIT_DECAY` is now decorative** — it shapes a reported
   diagnostic and nothing else. Kept so the column keeps its meaning; a future
   pass may remove it entirely.
3. **Uncertainty is nearly inert inside the score** (0.92% of log-variance).
   Whether it deserves to be a full factor in the product, rather than a
   reported caveat, is a live design question this audit did not settle.
4. **max risk 0.730 < 0.9** persists, by the data, not the formula.
5. Stage 3 audit findings **N1–N4 remain open**; duplicate recall is still ~1%.

---

## Stage 5 Hardening — surgical, and numerically inert

Verdict: **PASS**. Every score, band and ranking is **byte-identical** to before
the pass. One mandated restructuring was refused with proof; two measurement
bugs in the audit's own diagnostics were found and fixed.

### The headline finding: "uncertainty is dead" was wrong

The audit reported uncertainty at **0.92% of variance** and the brief called it
"functionally dead". Both the number and the conclusion were wrong.

**The conclusion was wrong** because variance share cannot see decisions. The
decisive test is whether removing the factor changes an outcome:

| test | result |
|---|---|
| spearman(risk with U, risk without U) | 0.99305 — **not** 1.0 |
| scored records with U > 0 | 6,503 / 13,960 (46.58%) |
| records whose rank changes | 13,915 (99.68%) |
| **records whose BAND changes** | **294** |
| high_risk with U vs without | 291 → **367** (+26% escalation queue) |

A factor carrying 2% of the spread still moves 294 records across a band
boundary, 76 of them into the investigate queue. **Uncertainty is load-bearing.
No restructuring applied.** Options A, B and C were all rejected — and the
report now runs this removal test itself and publishes the verdict, so the
question is settled by measurement on every run rather than by argument.

**The number was wrong** for two independent reasons, both found in this pass:

1. **Variance share ignores covariance.** For a sum in log space the honest
   attribution is `cov(x, log R) / var(log R)`, which sums to exactly 100%.
2. **Two bugs in that computation**, caught by tightening the sum-to-100% test
   from `abs=0.5` to `abs=1e-6`:
   - Records with a zero factor were `clip`ped to `1e-12`, breaking the identity
     `log R = log S + log Q + log(1−U)`. They are now **excluded and counted**,
     never approximated.
   - `np.cov` defaults to `ddof=1` while `np.var` defaults to `ddof=0`. Mixing
     them scaled every share by `n/(n−1)` — the corpus reported **100.91%**,
     and a 500-row test reproduced `500/499 = 1.002004` exactly.

Corrected attribution:

| factor | variance share | covariance share |
|---|---|---|
| signal_strength | 88.31% | **85.10%** |
| data_quality | 10.34% | **12.65%** |
| stability | 1.35% | **2.25%** |
| | | **sums to 100.0000%** |

Stability is 2.25%, not 0.92% — 2.4× the reported figure and above the 1% flag
threshold. The flag itself is documented as **"FLAGGED FOR REVIEW, NOT FOR
REMOVAL"**, because this pass is the demonstration of why those differ.

### Fix 2 — the explanation names only what the score uses

Tightened from the audit's "name the inactive factor but say it is not charged"
to **"do not name it"**. A narrative claiming to reconstruct the arithmetic
cannot also discuss a term outside the arithmetic, however carefully caveated.
`risk_deficit_factor` and `cluster_penalty_factor` are gone from the text; both
remain as columns and in the calibration report. Gate-redundant uncertainty
terms are likewise never named on a scored record, where they cannot be true.

The round-trip test never reads a component column. It parses the numbers back
out of the English and re-multiplies them, with a tolerance **derived** from the
3-decimal print precision (`3 × 5e-4`), not chosen. Over the full corpus:

| check | result |
|---|---|
| scored explanations parsed and re-multiplied | 13,960 |
| arithmetic or band mismatches | **0** |
| explanations naming an inactive factor | **0** |

### Fix 3 — every uncertainty component classified, and the class verified

| component | class | corpus | scored |
|---|---|---|---|
| coverage | active | 61.89% | 32.52% |
| unstable_cell | active | 21.45% | 0.87% |
| no_severity | gate_redundant | 20.78% | **0.00%** |
| no_norm | gate_redundant | 7.62% | **0.00%** |
| unreachable_duplicate | structurally_impossible | 0.00% | **0.00%** |

All retained, each against a stated criterion: the gate-redundant pair **applies
to unscored records**, where the reported uncertainty column is the entire
answer; the impossible one **protects an invariant** across the Stage 3/4
boundary. The classification is published, and a test asserts the published
class matches measured behaviour — a claim of "gate-redundant" that stopped
being true would fail.

### Fix 4 — the zero-signal records are measured-normal, proven from the reason

Five records have `signal_strength == 0`. The proof is Stage 3's reason column,
not the value:

```
deviation_cell_cost = 0.0    deviation_cell_cost_reason = "defined"
valid_signal_count  = 1      severity_defined = True
confidence          in [0.738, 0.771]
```

Stage 3 **computed** a peer comparison and it landed on the median. That is a
measurement, not a gap. `risk = 0` stays; NaN would be a false claim of
ignorance and would collapse the distinction the whole system rests on.

### Fix 5 — extreme inputs

All four mandated shapes plus numeric extremes (`1e-300`, `±1e308`,
`critical_deficit = 1e6`) produce no negative, inflated, infinite or undefined
value, under `warnings.simplefilter("error", RuntimeWarning)`. Monotonicity in
severity holds across `0 → 1e-9 → 0.001 → 0.5 → 0.999 → 1.0`.

### Fix 6 — distribution honesty

Unchanged and un-rescaled. p50 0.1137 · p90 0.2126 · p95 0.2581 · p99 0.6053 ·
max 0.7304. The multiplicative structure was verified intact rather than
assumed: `E[log risk] = −2.272832` against `Σ E[log factor] = −2.273096`, the
residual being exactly the zero-factor records now excluded from attribution.

Added beside every distribution:

> **Risk values are NOT calibrated thresholds.** A risk of 0.5 does not mean a
> 50% chance of anything; it is a position on an uncalibrated ordinal scale
> produced by this corpus and these judgements.

### No-change confirmation

| | before | after |
|---|---|---|
| p50 / p95 / p99 / max | 0.113716 / 0.258133 / 0.605283 / 0.730351 | **identical** |
| high / moderate / low | 291 / 1,452 / 12,217 | **identical** |
| unscored | 6,040 | **identical** |

1,038 tests pass. The 128 Stage 5 tests and 49 of the 50 audit tests were not
modified; one audit test was updated because Fix 2 **superseded** its
requirement, and its docstring records that.

### Files

| File | Change |
|---|---|
| `src/stage5/explanation.py` | names only active factors |
| `src/stage5/calibration.py` | `compute_contribution_analysis`, liveness detail, threshold note |
| `src/core/constants.py` | `UNCERTAINTY_COMPONENT_CLASS`, flag threshold, note |
| `tests/test_stage5_hardening.py` | **new** — 49 tests |

### Known limitations after hardening

1. **Still not calibrated**, and every report says so twice.
2. **`RISK_CRITICAL_DEFICIT_DECAY` remains decorative** — it shapes a reported
   diagnostic and nothing else.
3. **Stability contributes 2.25%** of the spread. Load-bearing, but whether it
   deserves to be a full multiplicative factor rather than a reported caveat is
   still an open design question — now with a number attached.
4. **max risk 0.730 < 0.9**, by the data and not the formula.
5. Stage 3 audit findings **N1–N4 remain open**; duplicate recall is still ~1%.

---

## Stage 6 — Action & Routing Layer

Policy, not inference. Stage 6 computes nothing: it maps the Stage 4 decision
and the Stage 5 risk band onto an action, a priority, a queue and a sentence a
human can act on. Every rule is a table lookup an operations lead could change
without a developer.

### Three defects in the specification, all measured before coding

| defect | evidence | resolution |
|---|---|---|
| **Gap** — `INVESTIGATE + low_risk` matches none of CASES 1–5 | **6 records** | Filled from EDGE CASE 3 → `ESCALATE_REVIEW` |
| **Collision** — every REMEDIATE also satisfies CASE 5's `risk_defined == False` | **2,638 records** | CASE 3 wins; see below |
| **Missing column** — the brief requires `anomaly_reason` | Stage 4 emits `decision_reason` | Treated as optional, never required |

**On the collision.** Stage 4 routes to REMEDIATE precisely when confidence is
too low for Stage 5 to score, so *all* 2,638 REMEDIATE records satisfy both
cases. CASE 3 names the decision class; CASE 5 is a fallback — the explicit
rule wins. It is also more actionable: REMEDIATE means *this record's own
evidence is weak*, which the field officer who filed it can fix, whereas the
data-quality queue is for records nothing could be said about. Resolving the
other way would have put **30% of the corpus into P1** and emptied the word
"priority" of meaning.

### A fourth defect, found by an exhaustive test rather than by reading

`TestPolicyTotality` enumerates every `decision_class × risk_flag` combination,
including ones the corpus does not contain. It failed:

> `AssertionError: a high_risk record was given the lowest priority`

`MONITOR + high_risk` falls to CASE 4 → `PASSIVE_MONITOR` → **P3**, breaching
invariant 4 (*high_risk must never map to P3*). CASE 4 and invariant 4
contradict each other, and the reference corpus hides it — all 291 high_risk
records are INVESTIGATE.

**It is reachable, not hypothetical.** Stage 4 monitors below `|z| = 3.5`, so
severity can reach ≈0.7; with breadth boosting the signal to ≈0.76 and clean,
fully-covered evidence, Stage 5 can band such a record `high_risk`.

Resolved with one rule, `disagreement_high_risk` → `ESCALATE_REVIEW`. It is the
**mirror of EDGE CASE 3**, which resolves the opposite disagreement the same
way: when the two stages differ, a human looks. Stage 6 invents no verdict — it
declines to pick the quieter of two upstream answers.

### M1 — the fix this stage exists for

Stage 5's audit found records escalated with **no named finding**: Stage 4
gates `underspend_anomaly` on lifecycle, but its routing and Stage 5's severity
both read `|z|`, so a large underspend on an unfinished work escalates unlabelled.
**18 records** on the reference corpus (4 of them P0). They now carry
`unexplained_deviation`.

Two deliberate deviations from the letter of the brief:

1. **Stage 4's `anomaly_types` is not mutated.** The corrected list goes to a
   new column, `action_anomaly_types`. Stage 4 is locked with byte-identical
   guarantees, and a downstream layer quietly rewriting it would make a re-run
   of Stage 5 read different inputs than the first run did.
2. **The correction keys off the action, not `decision_class`.** Invariant 6
   requires a finding on every `ESCALATE_*` record. Every INVESTIGATE escalates,
   but the reverse is not guaranteed once `disagreement_high_risk` exists.
   Keying on the action covers the mandated case exactly and closes that one too.

### Routing on 20,000 records (0.24s)

| queue | n | % | priority |
|---|---|---|---|
| automated_monitoring | 13,541 | 67.70 | P3 |
| data_quality_team | 3,402 | 17.01 | P1 |
| field_officer | 2,638 | 13.19 | P2 |
| fraud_investigation_team | 291 | 1.46 | P0 |
| audit_team | 128 | 0.64 | P1 |

Rules fired: monitor 13,541 · insufficient_context 3,402 · remediate 2,638 ·
investigate_high 291 · investigate_moderate 122 · investigate_low 6. The two
backstop rules (`investigate_unscored`, `disagreement_high_risk`) fired **0**
times — correct, and retained because both are reachable in principle.

**291 records reach a human investigator; 128 more reach an auditor.** That is
2.1% of the corpus, from 20,000 records nobody could read by hand.

### The explanation is machine-checkable

Five lines, fixed order, one field each — and `parse_action_explanation` is the
exact inverse of `explain_action`, shipped in the same module so the two cannot
drift:

```
Record routed to ESCALATE_IMMEDIATE because:
- Findings: cost_outlier
- Severity: 0.601
- Risk: 0.730
- Decision basis: INVESTIGATE with high_risk
```

A test parses **every** generated explanation and compares each field against
its stored column. A narrative that stops matching its record fails the build.
The definedness flag is authoritative, not the number: a stray `severity_score`
on a record with `severity_defined == False` prints as `not defined`, and a
test asserts it.

### Files

| File | Contents |
|---|---|
| `src/stage6/routing.py` | contract check, M1 correction, the 8-rule policy table, invariants |
| `src/stage6/explanation.py` | the fixed format, and its parser |
| `src/stage6/pipeline.py` | `ActionLayer`, `attach_actions`, queue and priority views |
| `tests/test_stage6.py` | 81 tests |

1,119 tests pass (1,038 + 81). `main.py` gains `--stage5-only`.

### Known limitations

1. **The policy is judgement, like everything upstream.** Which team owns which
   action, and that P0 means fraud investigation, are organisational choices
   with no evidence behind them. They are in `constants.py` so changing them is
   a one-line edit, not a code change.
2. **17.65% of the corpus lands in P1.** Driven by the 3,402 unscorable
   records, not by anomaly volume. If a data-quality team cannot absorb that,
   the fix is upstream data collection, not a threshold here.
3. **`unexplained_deviation` names a gap, it does not close it.** The real fix
   is for Stage 4 to say *what* deviated — a lifecycle-gated underspend is a
   specific, nameable thing. Stage 6 can only report that Stage 4 declined to
   name it.
4. **Nothing is calibrated**, and Stage 5's reports still say so on every run.
5. Stage 3 audit findings **N1–N4 remain open**; duplicate recall is still ~1%.

---

## Stage 6 Hardening — self-validation and contract alignment

No routing decision changed. Every action, priority, rule and M1 count is
**identical** to before the pass, verified record by record on 20,000 rows. The
pass added aliases, a machine-readable payload, and three checks for
assumptions that were previously being trusted in silence.

### A third vocabulary, found by reading the PRD

`Stage6.md` names its outputs **INVESTIGATE / REMEDIATE / MONITOR / CLEAR**.
That matches neither the build brief's five action names nor the audit brief's
six. Those PRD names are *decision* names, and Stage 4 already implements them
as `DECISION_CLASSES` (with `INSUFFICIENT_CONTEXT` in place of `CLEAR`) — so
Stage 6 as built is an action layer sitting on top of the PRD's decision layer,
not a competing implementation of it. All three vocabularies are now documented
together in `SPEC_ACTION_CLASSES`.

### C1 — contract alignment, additively

| spec name | as-built source |
|---|---|
| `action` | `action_class` |
| `priority` | `priority_level` |
| `action_reason` | `action_rule` |
| `action_spec` | `SPEC_ACTION_ALIAS[action_class]` |

Nothing renamed, nothing removed. Every alias is **asserted equal to its
source on each run** — an alias that drifted would be worse than none at all:
two columns disagreeing about one decision.

`ESCALATE_IMMEDIATE → ESCALATE_INVESTIGATION`, `REQUEST_CORRECTION →
ROUTE_REMEDIATE`, `PASSIVE_MONITOR → MONITOR_PASSIVE`, `ESCALATE_REVIEW`
unchanged. **`DATA_QUALITY_REVIEW → ROUTE_AUDIT` is the weakest of the five**
and is documented as such: the specification offers no data-quality action, and
ROUTE_AUDIT is the nearest remaining sense of "a team must look before this can
be judged".

**`HOLD_NO_ACTION` has no producer, deliberately.** Stage 6 never concludes a
record needs nothing: its quietest outcome is `PASSIVE_MONITOR`, a standing
watch rather than a dismissal. Representable, never emitted, and asserted so.

### M2 + M3 — a canonical machine form, alongside the human one

The five-line sentence is written for a person and pays for it twice: it omits
`priority` (recoverable 0 / 20,000) and its delimiters are ambiguous. The audit
proved three collisions by construction — a finding containing `", "` parsed
back as two, a `decision_class` containing `" with "` split in the wrong place,
and a finding literally named `"none recorded"` was indistinguishable from
having none.

**Resolution: a new `explanation_payload` column carrying canonical JSON.** It
escapes every delimiter, distinguishes an empty list from any string, and
round-trips arbitrary content — including quotes, backslashes, newlines and
unicode. Keys are sorted and separators compact, so two runs produce identical
bytes.

The human `explanation` is left **byte-identical** (all 20,000 still exactly
five lines). That is a deliberate reading of the constraint *"ALL existing tests
must pass unchanged"*: `test_the_format_is_exactly_five_lines` pins the format,
so the machine form had to arrive as a new column rather than a rewrite.
`parse_action_explanation` is now documented as best-effort for humans and
**not** the machine contract.

| | before | after |
|---|---|---|
| priority recoverable | 0 / 20,000 | **20,000 / 20,000** |
| payload field mismatches | — | **0** across 100,000 field comparisons |
| `NaN` in any payload | — | **0** (absent numbers are `null`) |

### M1 + M4 + m1 — Stage 6 now validates itself

Three checks run at pipeline entry, ordered so a failure is cheapest to
diagnose: configuration, then shape, then cross-field consistency.

* **`assert_gate_alignment()`** — refuses to route when
  `CONFIDENCE_GATE_THRESHOLD != MIN_CONFIDENCE_FOR_RISK`. Both derive from
  `PEER_STAT_MIN_CONFIDENCE`, so it passes by construction today; it exists so
  a future edit fails here rather than silently breaking an invariant three
  stages away.
* **`validate_stage5_contract()`** — verifies
  `risk_flag == "insufficient_data" ⟺ ¬risk_defined`. Eight policy predicates
  read `risk_flag` and five read `risk_defined`; nothing previously checked
  they agree, so a disagreement surfaced as an internal `AssertionError` from
  deep inside the invariant block instead of a contract error at the door.
* **`require_unique_index()`** — replaces pandas' opaque
  *"cannot reindex on an axis with duplicate labels"* with a message naming the
  duplicated labels and the requirement broken.

All three raise `Stage6ContractError`, a subclass of `Stage6InputError`, because
the remedy differs: a missing column means a stage was not run, a contradictory
one means a stage produced something impossible.

### One fix deliberately NOT made, and why

The audit's headline break — `RiskConfig(min_confidence=0.80)` producing **73
records** that are `INVESTIGATE` yet `insufficient_data`, every one escalated —
is **still reachable**.

Closing it means rejecting `decision_class == INVESTIGATE ∧ ¬risk_defined`. That
check was written, and it failed three existing policy tests that construct
exactly that combination on purpose, and it made the `investigate_unscored`
backstop rule unreachable. More importantly, routing such a record requires
breaking one of two invariants — *never downgrade an escalation* or *never
escalate insufficient data* — and choosing which is a **policy decision, not a
validation one**.

The brief's own instruction settled it: *"Only fix what is proven broken.
Everything else is preserved."* `assert_gate_alignment` catches the
constant-level case; the configured case is documented in
`validate_stage5_contract`'s docstring and remains open for a deliberate
decision.

### Regression

| | before | after |
|---|---|---|
| PASSIVE_MONITOR / DATA_QUALITY_REVIEW / REQUEST_CORRECTION | 13,541 / 3,402 / 2,638 | **identical** |
| ESCALATE_IMMEDIATE / ESCALATE_REVIEW | 291 / 128 | **identical** |
| P0 / P1 / P2 / P3 | 291 / 3,530 / 2,638 / 13,541 | **identical** |
| all six firing rules | unchanged | **identical** |
| M1 corrections | 18 | **identical** |

1,174 tests pass (1,119 + 55). The 81 existing Stage 6 tests were **not
modified**.

### Files

| File | Change |
|---|---|
| `src/stage6/routing.py` | `Stage6ContractError`, three entry checks |
| `src/stage6/explanation.py` | JSON payload builder and parser; human form untouched |
| `src/stage6/pipeline.py` | five alias columns, alias equality asserted |
| `src/core/constants.py` | `SPEC_ACTION_CLASSES`, `SPEC_ACTION_ALIAS`, `SPEC_COLUMN_ALIAS` |
| `tests/test_stage6_hardening.py` | **new** — 55 tests |

### Second hardening pass — specification realignment

A later specification revised three things this pass implemented:

* **`action_spec` now maps to the PRD vocabulary and is deliberately lossy.**
  `ESCALATE_IMMEDIATE` and `ESCALATE_REVIEW` both become `INVESTIGATE`, so the
  alias alone cannot separate 291 P0 fraud referrals from 128 P1 audit reviews.
  That distinction survives in `action_class` and `priority_level`, both
  unchanged, and a test pins that it survives somewhere. The prior one-to-one
  mapping is retained as `SPEC_ACTION_ALIAS_V1` so the change is traceable.
* **Two dedicated exception types.** `Stage6ConfigError` for a threshold that
  cannot support the invariants (no record is at fault, no rerun helps), and
  `Stage6InvariantError` for a post-routing guarantee — a `RuntimeError`, not an
  `AssertionError`, so the checks survive `python -O` and a caller can catch a
  Stage 6 failure specifically.
* **The payload gained `rule`, `findings` and `reason`.** `anomaly_types` is
  kept as a synonym of `findings` so the earlier payload contract still
  resolves, and an I5 assertion checks the two agree on every record.

Three tests changed, each carrying a docstring recording what superseded it:
the injectivity assertion (the spec revoked it) and two that expected
`Stage6ContractError` where a `Stage6ConfigError` is now mandated.

1,185 tests pass. Routing counts, priorities and M1 corrections are byte-identical.

### Known limitations after hardening

1. **The configured gate-drift vector remains open** (see above). It is a
   policy decision awaiting an owner, not an oversight.
2. **`DATA_QUALITY_REVIEW → ROUTE_AUDIT` is an imprecise alias.** The
   specification has no data-quality action; a consumer reading `action_spec`
   will see audit work and data-quality work under one name.
3. **Two explanation formats now exist.** The human one is not injection-safe
   and never will be; the payload is authoritative. A consumer that parses the
   sentence instead of the payload gets the old ambiguity.
4. **P1 still conflates two meanings** — 3,402 data-quality records and 128
   audit escalations share a priority band.
5. Stage 3 audit findings **N1–N4 remain open**; nothing downstream is calibrated.

---

## Stage 7

Not started. Strict order is enforced: no stage begins until the previous one's
tests pass.
