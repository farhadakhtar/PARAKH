# PARAKH Stage 8 — handoff

Written 2026-09-06 for whoever picks this up next. Everything here is measured
unless marked otherwise. Where a number was wrong earlier, the correction is
recorded rather than the number silently replaced.

---

## 1. What PARAKH is

An **evidentiary-confidence fraud-analytics engine** for Indian public works
(MPLADS/PMGSY-style registers). Stages 1–7 are built, tested and stable:

| stage | does |
|---|---|
| 1 | ingestion, cleaning, schema validation |
| 2 | evidentiary **confidence** — completeness, temporal coherence, reconciliation |
| 3 | peer structure — TF-IDF over `work_name`, HDBSCAN clusters, peer cells, duplicate detection |
| 4 | anomaly interpretation — `z_cost`, `z_spend`, `z_duration`, severity |
| 5 | **risk** = signal_strength × data_quality × (1 − uncertainty) |
| 6 | action & routing — 8-rule ordered policy, safety layer S1–S5 |
| 7 | decision consumption — read-only over 1–6, canonical JSON payload |
| 8 | **empirical validation — this session** |

**The architectural thesis, which must not be broken:** confidence (C) and
risk (R) are different quantities. High C ≠ clean. A record that cannot be
evidenced is never a fraud allegation. Low confidence + high risk routes to
REMEDIATE, never to FRAUD.

**Every threshold in Stages 4–7 is a stated judgement**, labelled
`UNCALIBRATED` in `constants.py`. None has been fitted to real outcomes.

---

## 2. Stage 8 verdict

`artifacts/stage8/APPROVAL_STATUS.json` → **`COVERAGE_FAILURE`, 3 of 11 gates
pass**, `calibrated_risk = None`.

The binding constraint is **data coverage, not model quality**:

```
audit evidence      CAG SFAR, Nagaland, FY 2020-21
structured finance  PMGSY state-year, FY 2023-24 .. 2026-27
year intersection   EMPTY
state intersection  {NAGALAND} — one state
work-level match    impossible, no dataset carries a work ID
```

Matching was **not** relaxed to force a join. Dropping to STATE_ONLY would
produce rows at LEVEL_0 evidence — the defect that put 5,237 labels on a
postal-office directory in a previous session.

---

## 3. What was built this session

New modules under `src/stage8/` (all tracked in git):

| module | purpose | tests |
|---|---|---|
| `matching.py` | blocked record↔finding matching, evidence-tiered cascade | 33 |
| `compliance.py` | statutory rules with citations (GFR 2017) | 23 |
| `injection.py` | Indic text normalisation + defect vectorisation | 31 |
| `typology.py` | documented fraud typologies, source-gated | — |
| `safety.py` | artifact invariance, leakage, ranking stability | — |
| `neural.py` | self-supervised masked-field consistency model (TF) | — |
| `calibration.py` | pre-existing refusal scaffold, unchanged | — |

Kaggle trainers under `kaggle/`: `parakh_consistency_train.py`,
`parakh_longrun_train.py`, `parakh_nlp_train.py`, `parakh_cartel_train.py`,
plus `KAGGLE_PLAN.md`.

**Full suite: 1,576 tests passing, zero regressions.**
Run `pytest tests/` — NOT bare `pytest`. `simple_test.py` at repo root is a
stale Stage-1 leftover importing a `validator` module that no longer exists;
it breaks collection. Pre-existing, unrelated.

---

## 4. Measured results

### Matcher — blocking proven, not asserted
```
naive    400,000 comparisons   6.226s   797 matches
blocked      866 comparisons   0.070s   797 matches
           462x fewer          89x       IDENTICAL
```
Parity against brute force is the whole correctness argument for blocking. A
blocking key that drops true pairs is invisible downstream — a metric computed
on the labels that exist cannot notice the labels never produced.

### Compliance — 3 of 5 rules LIVE
Every rule shipped `citation_verified=False`, gated to
`PENDING_CITATION_VERIFICATION`. Downloading GFR 2017 from doe.gov.in proved
**3 of 5 citations wrong**:

| stated | GFR 2017 actually says |
|---|---|
| Rule 155 = Advertised Tender | Rule 155 = purchase by committee. Advertised Tender is **161** |
| Rule 154 = Limited Tender | Rule 154 = purchase without quotations. Limited Tender is **162** |
| Rule 230 = Utilisation Certificates | Rule 230 = grants-in-aid principles. UC is **238**, anchored to financial-year close, not completion |
| `n_bidders >= 3` | Rule 162 says "more than three" → minimum **4** |

Now live: 161, 162, 166. Still gated: 238 (UC), CPWD work-before-approval.
PDF is at `Data/legal/GFR_2017_original.pdf`, extracted text alongside.

### Consistency model — weak, and the ceiling is the data
500k records, 100 epochs, converged epoch 94, **10 minutes** of an 8-hour
budget. AUC **flat from epoch 10 to 100** — more compute changes nothing.

| channel | n | ROC-AUC |
|---|---|---|
| date_order | 2,754 | 0.577 ± 0.012 |
| cost_outlier | 3,382 | 0.573 ± 0.011 |
| agency_mismatch | 2,210 | 0.538 ± 0.013 |
| overspend | 2,319 | 0.534 ± 0.013 |

### Cartel screen — the first genuinely supervised fraud task
73 prosecuted EU cartels, 18,807 labelled bids (8,045 / 10,762).

Grouped CV (by `cartel_id`): gradient boosting **0.581 ± 0.008**,
MLP 0.566, logistic 0.520.

Leave-one-country-out — the India proxy:
```
BG 0.73  LV 0.74  SE 0.63  HU 0.63  PT 0.58  FR 0.57  ES 0.51
```
**Screens do not transfer reliably between jurisdictions.** The mechanism
travels; the calibration does not.

### Method survey (26-agent workflow)
58 methods surveyed, 28 claimed label-free implementable, **0 survived
adversarial refutation**. Almost everything needs bid data or labels. Plan and
engineering-hacks section: `artifacts/stage8/METHOD_SURVEY_PLAN.md`.

---

## 5. Errors I made — distrust these patterns

Four of ten logged experiments are my own code bugs. Full detail in
`STAGE8_FAILURES.md`. The instructive ones:

**EXP-009 — tender reconstruction gave groups of size 1.** The cartel dataset
has no `tender_id`, so I reconstructed one, keyed over the *labelled subset
only*. Median group size 1, 8–11% agreement with `lot_bidscount` against a
true median of 4. Every within-tender screen computed on a single bid. It
scored 0.52 and looked like it worked. Nothing in the output flagged it — a
degenerate group returns a number, not an error. **Check any invented join key
against a published count.**

**EXP-008 — published an AUC from 171 positives with no interval.** Reported
`cost_outlier 0.699` as the target to beat; at n=3,382 the value is 0.573 and
the intervals do not overlap. Always report n and a CI.

**EXP-002 — 0.97 reconstruction accuracy beside 0.538 defect AUC.** Loss
convergence is not task performance. The surprise score summed categorical
heads only, so numeric and temporal defects were undetectable by construction.

**I claimed no Kaggle access when it was configured.** Checked for
`~/.kaggle/kaggle.json`; the file is `credentials.json` and the CLI is not on
PATH. `python -c "import kaggle"` printing nothing means success, not failure.

**I shipped a from-scratch BiLSTM** whose own docstring said its purpose was
cross-lingual synonymy — an architecture that cannot learn it. Pretrained
weights are not optional for that claim.

---

## 6. Data assets acquired

| path / source | what | licence |
|---|---|---|
| `Data/legal/GFR_2017_original.pdf` | General Financial Rules 2017, 208pp | Govt of India |
| `Data/cartels/full_cartels.parquet` | 3.86M bids, 73 cartels, 7 countries, 18,807 labelled | CC-BY-NC (Zenodo 17595875) |
| `Data/prozorro/bids_2022.csv` | 240k Ukrainian bids w/ `is_winner` | CC-BY (Kaggle) |
| `Data/courts/*.parquet` | Indian HC case metadata | CC-BY-4.0 (AWS open data) |
| `s3://indian-high-court-judgments` | 25 courts, 1950–2025, ~1 TB, **no credentials needed** | CC-BY-4.0 |

**ProZorro measured:** single-bidder share 50.8% (inflated — 2022 is the
invasion year, emergency rules); 25% of losing bids within **0.04%** of the
winner, which is the cover-bidding signature.

**Court data caveat:** `title`/`description` are case metadata, not judgment
text. Allahabad 2023 = 443,845 rows but only **2** mention corruption and
**0%** Devanagari. The `acts` column — which would identify PC Act cases — is
`[]` for all rows. Statutes and facts are in the PDFs (~1 TB).

---

## 7. Current state

**Running:** `parakh-cartel-screen` on Kaggle.
**Complete:** `parakh-consistency-long-run`.
**Blocked:** `parakh-encoder-bench` — kernel gets no DNS despite
`enable_internet: true` stored server-side. That signature is an **unverified
phone number** on the Kaggle account. Two-minute user action at
kaggle.com/settings; not fixable in code. `--model-path` was added as an
offline fallback, but Kaggle's own `google/muril` is a **TF-Hub SavedModel**,
not a HuggingFace checkpoint, so it will not load via `AutoModel`.

Status page for non-technical readers:
https://claude.ai/code/artifact/6e633a50-07c0-481a-b9af-93c6998cfceb
(source: `docs/parakh_status.html`)

---

## 8. Environment gotchas

- Windows 11, Git Bash. Working dir `D:\hackthon\SIH\PARAKH`.
- **Bash heredocs fail** on non-ASCII and long content. Use the Write tool, or
  write a Python patch script to the scratchpad and execute it.
- `tensorflow 2.20` installed. **No torch.** `transformers` absent locally.
- `transformers` v5 **removed all TF classes** — `TFAutoModel` does not exist.
  Use `AutoModel` + torch on Kaggle.
- Kaggle CLI at `~/AppData/Roaming/Python/Python312/Scripts/kaggle.exe`, not
  on PATH. Auth via `~/.kaggle/credentials.json`, user `farhadastrodent`.
- **Always wait for `kaggle datasets status` to report `ready`** before
  pushing a kernel. That race killed two kernels.
- Kernel `title` must slugify to the `id` suffix or the push 409s.
- Deleting `data/synthetic_dataset.csv` breaks 5 tests; regenerate with
  `generate_with_ledger(n=20000, seed=42)`.

---

## 9. What to do next, ranked

1. **Verify the Kaggle phone number.** Unblocks the encoder bench permanently.
2. **Verify GFR Rule 238 and the CPWD manual** — two clause readings by
   someone who knows procurement, ~20 minutes, flips the last two compliance
   rules live. Each carries a `verification_note` saying exactly what to check.
3. **Extract typologies from Indian sources** — major scams and PC Act
   judgments give *what to look for*, with citations. `typology.py` is
   currently **0 of 5 source-verified**. This is the India half of the split.
4. **Wire the 5 typologies against ProZorro** — the international half. Real
   bid distributions to set thresholds against instead of taste.
5. **Do not** spend more compute on the consistency model. Flat across 90
   epochs; the data is the limit.

### The division that holds
- **India → the WHAT.** Scams and judgments define which patterns matter, with
  citations. Fixes the unverified typology library.
- **International → the HOW.** ProZorro and the cartel set supply bid
  structure, distributions and validation.

Calibration stays refused either way. It needs **audit scope** — which units
an audit examined and *cleared* — because that is the only thing that creates
negatives. Without it, "no finding" and "never audited" are the same row and
NULL is the only honest value. More audit PDFs do not help.

---

## 10. The rule that governs everything here

A refusal backed by a measurement is a successful outcome. A result backed by
invented labels is a failure. Never convert NULL to 0, never fit a threshold
to absent outcomes, never present synthetic ground truth as real-world
validation, never weaken a gate to make a pipeline pass.
