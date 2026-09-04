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

## Stages 3–7

Not started. Strict order is enforced: no stage begins until the previous one's
tests pass.
