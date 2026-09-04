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

## Stages 3–7

Not started. Strict order is enforced: no stage begins until the previous one's
tests pass.
