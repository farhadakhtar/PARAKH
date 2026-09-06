# PARAKH — Kaggle training round

Local box cannot run this: **no torch, no `transformers`, no GPU, no model
downloads.** Verified 2026-09-06. TensorFlow 2.20 is present, which is why
both trainers are Keras. Everything below runs on Kaggle as-is.

---

## What is actually being trained, and what it can claim

Two models. Neither predicts fraud, because no fraud label exists — see
`artifacts/stage8/APPROVAL_STATUS.json`.

| model | task | ground truth | claim it supports |
|---|---|---|---|
| **Consistency NN** | mask a field, predict it from the rest | the field's own value — free, exact | "this record disagrees with itself" |
| **Work-text encoder** | description → agency / cost band / status | the record's own columns | "these two works are peers" |

Both are *feature* work. Neither moves the calibration gate.

---

## Round 1 — Consistency NN (no accelerator, ~10 min)

```bash
!python parakh_consistency_train.py --demo --epochs 60
```

Then on real data:

```bash
!python parakh_consistency_train.py \
    --csv /kaggle/input/parakh-corpus/synthetic_dataset.csv --epochs 60
```

**Measured on Kaggle, 500k records, 75k holdout** (seed 20260906):

| defect channel | n positives | ROC-AUC (95% CI) |
|---|---|---|
| date_order | 2,754 | 0.577 ± 0.012 |
| cost_outlier | 3,382 | 0.573 ± 0.011 |
| agency_mismatch | 2,210 | 0.538 ± 0.013 |
| overspend | 2,319 | 0.534 ± 0.013 |

⚠ **Superseded numbers.** An earlier local run reported cost_outlier 0.699 from
**171 positives with no error bar**, and that figure was quoted as a target
here. Its interval (±0.045) does not overlap the large-sample estimate. The
0.699 was unreliable; 0.573 is the value.

AUC was flat from epoch 10 to epoch 100 — the ceiling is the data, not the
compute. A longer run will not move it. Reconstruction accuracy is **not** the
success metric — an earlier version scored 0.97 on reconstruction and 0.538
on defect detection.

**Do not** report accuracy on `state`: districts nest inside states, so that
head is a lookup table and scored 1.0000 on demo data.

---

## Round 2 — Work-text encoder (GPU T4, ~25 min)

Accelerator: **GPU T4 x2**. Internet: **ON** for the first run (model
download); afterwards attach the model as a Kaggle Dataset and turn it off.

```bash
!pip -q install transformers
!python parakh_nlp_train.py --encoder muril --demo --epochs 30
```

### Open weights for Hindi / code-mixed text

| model | HF id | params | why |
|---|---|---|---|
| **MuRIL** ← start here | `google/muril-base-cased` | 236M | 17 Indian languages **plus transliterated text** — the register work names are actually written in |
| IndicBERT | `ai4bharat/indic-bert` | 33M | ALBERT-based, much smaller; fallback if MuRIL OOMs |
| XLM-R base | `xlm-roberta-base` | 270M | broader multilingual, weaker on transliteration |
| LaBSE | `sentence-transformers/LaBSE` | 471M | sentence-level; good if you want retrieval instead of heads |

MuRIL first specifically because Indian works registers mix Devanagari,
English, and romanised Hindi *in the same field* ("Nirman of sadak"), and
transliteration coverage is the axis the others are weakest on.

### The gate that decides the whole round

`synonymy_probe()` runs **before** training. It embeds known pairs
(`road`/`sadak`, `school`/`vidyalaya`, `building`/`bhavan`) against controls
(`road`/`vidyalaya`) and reports the separation.

```
separation > 0.05   → encoder has the capability TF-IDF lacks. Continue.
separation ≤ 0.05   → STOP. Do not train heads.
```

If the probe fails, the encoder does not supply cross-lingual matching, and
that is the *entire* reason to replace TF-IDF. Training heads anyway burns
GPU to produce a number that cannot justify the swap.

### Success criterion

Beat `TF-IDF + logistic regression` on a **decisive** target — the script
marks ties and saturated targets as `tie/degenerate` and excludes them.
Local ablation (`--encoder scratch`) tied on all three, which is the expected
result and the reason pretrained weights are mandatory.

Winning is not sufficient. Stage 3's explanations name the tokens that made
two works similar; a dense encoder cannot. **Adopt only if the margin is
large enough to pay for losing that.**

---

## Round 3 — only if a bid table is acquired

Blocked on data, not compute. `detect_typologies()` reports **3 of 5**
typologies unreachable for want of `tender_id`, `bid_amount`, `is_winner`,
`n_bidders`. No model changes this; PARAKH's schema has never represented a
competition.

With a bid table, in order: single-bidder rate vs peer base rate → bid-spread
clustering (cover bidding) → winner rotation within recurring bidder sets.

Supervised fraud training still stays shut. A bid table gives *features*, not
outcomes.

---

## Setup

```
Dataset:     parakh-corpus  (data/synthetic_dataset.csv, data/ground_truth_ledger.json)
Round 1:     CPU, internet off
Round 2:     GPU T4 x2, internet on for first run only
Seed:        20260906  (set in both scripts; TF is not bitwise deterministic)
Persist:     /kaggle/working/{parakh_consistency_out,parakh_nlp_out}/
```

Each run writes `metrics.json`, the saved model, and — for Round 1 —
`holdout_surprise.csv`. Commit those back; they are what the Stage 8 report
cites.

---

## Things that will waste the round

- **Reporting reconstruction accuracy as success.** 0.97 reconstruction sat
  beside 0.538 defect detection. Different metrics, and only one is the job.
- **Skipping the synonymy probe.** It is the cheapest test in the pipeline
  and it can end the round in 30 seconds.
- **Training on `defect_channel`.** It is a synthetic label describing
  injected data-quality faults. It is a *validation* target here and never a
  fraud target.
- **Fine-tuning MuRIL on 20k four-word phrases.** 236M parameters will
  memorise it. Keep the encoder frozen; train the heads.
- **Reading either model's output as a fraud score.** Neither has seen a
  fraud label.

---

# Data sources for the Kaggle round

**Verification discipline.** `official URL` = a primary-source domain I am
confident of. `SEARCH` = I know the dataset exists in substance but will not
invent a Kaggle slug; find it and pin the exact version. Nothing below is a
fabricated link — an unverifiable citation is worse than none, which is the
same rule the compliance registry runs on.

Upload each as a **private Kaggle Dataset**, then attach it to the notebook.
That keeps Round 2 runnable with internet off after the first model pull.

## A. Statutory / legal — India

Feeds `src/stage8/compliance.py`. Every rule there ships
`citation_verified=False` and **cannot fire** until confirmed against these.
That is the single cheapest unlock in the project: 5 clauses, one reviewer.

| what | where | status |
|---|---|---|
| General Financial Rules 2017 | `https://doe.gov.in/order-circular/general-financial-rules-2017` | official URL |
| GFR amendments (thresholds change) | Dept. of Expenditure circulars, same domain | official URL |
| CVC circulars — procurement, single-tender | `https://www.cvc.gov.in/` | official URL |
| CPWD Works Manual | `https://cpwd.gov.in/` | official URL |
| Manual for Procurement of Works | Dept. of Expenditure | official URL |
| State PWD manuals | per-state PWD sites | varies — the manual that governs a work depends on its executing agency |

**Verify these five first** (each has a `verification_note` in the registry
saying exactly what to check):
`OPEN_TENDER_ABOVE_THRESHOLD` · `LIMITED_TENDER_MIN_BIDDERS` ·
`UC_SUBMITTED_WITHIN_PERIOD` · `SINGLE_SOURCE_JUSTIFICATION_RECORDED` ·
`WORK_NOT_STARTED_BEFORE_APPROVAL`

## B. Adjudicated fraud outcomes — the only real labels available

Feeds `src/stage8/typology.py`. **Entity-level, not work-level**, and biased
toward *caught* fraud. Both limits are structural — see the module docstring.

| what | where | status |
|---|---|---|
| World Bank debarred & cross-debarred firms | `https://www.worldbank.org/en/projects-operations/procurement/debarred-firms` | official URL — best single source; procurement domain, adjudicated, includes Indian firms |
| World Bank Sanctions Board decisions | `https://www.worldbank.org/en/about/unit/sanctions-system` | official URL — published reasoning, which is what makes a typology citable |
| ADB sanctions list | `https://www.adb.org/who-we-are/integrity/sanctions` | official URL |
| CVC / ministry blacklists | `https://www.cvc.gov.in/` | official URL — Indian, adjudicated |
| CAG audit reports | `https://cag.gov.in/` | official URL |
| OECD bid-rigging guidelines | `https://www.oecd.org/competition/bidrigging.htm` | official URL — defines the typology signatures |

⚠ Match debarment on **entity identity, never fuzzy name similarity**. A
fuzzy match against a debarment list is a defamation risk, not a finding.

## C. Procurement / bid data — the highest-value acquisition

This unlocks **3 of 5 typologies** that are currently unreachable. Measured:
the corpus has no `tender_id`, `bid_amount`, `is_winner`, or `n_bidders`, so
the entire bid-rigging class is invisible to PARAKH regardless of model.

| what | where | status |
|---|---|---|
| CPPP — Central Public Procurement Portal | `https://eprocure.gov.in/` | official URL — tender notices + awards |
| GeM | `https://gem.gov.in/` | official URL |
| State e-procurement portals | per state | varies |
| OMMAS (PMGSY district/work level) | `https://pmgsy.dord.gov.in/dbweb` | official URL — named in the PIB release as where district-wise data lives |

## D. Financial / scheme data

| what | where | status |
|---|---|---|
| PMGSY state-year (already in `Data/`) | PIB release, in repo | held |
| data.gov.in scheme datasets | `https://data.gov.in/` | official URL |
| Union Budget expenditure | `https://www.indiabudget.gov.in/` | official URL |

## E. Hindi / code-mixed text — for the NLP tier

**Measured 2026-09-06:** the current corpus is **100% Latin script, 0%
code-mixed** (`corpus_script_report`). The Indic pipeline in
`src/stage8/injection.py` is implemented and has 31 passing tests, but it has
never seen real Devanagari. Real work names are needed before any claim about
Hindi handling is supportable.

| what | where | status |
|---|---|---|
| MuRIL | `google/muril-base-cased` (HuggingFace) | verified id — 17 Indian languages **plus transliterated text** |
| IndicBERT | `ai4bharat/indic-bert` (HuggingFace) | verified id |
| LaBSE | `sentence-transformers/LaBSE` (HuggingFace) | verified id |
| IndicCorp (AI4Bharat) | `https://ai4bharat.iitm.ac.in/` | official URL — Hindi corpus, only if pretraining is ever needed |
| Real Hindi work names | MPLADS / state PWD registers | **SEARCH** — this is the gap; nothing here has it |

## Attach order

```
parakh-corpus     data/synthetic_dataset.csv + ground_truth_ledger.json   → Round 1
parakh-legal      A: GFR / CVC / CPWD PDFs                                → citation verification
parakh-sanctions  B: World Bank debarment + CVC blacklists (CSV)          → typology verification
parakh-tenders    C: CPPP / OMMAS extracts                                → unlocks Round 3
muril-base        google/muril-base-cased snapshot                        → Round 2 offline
```

## What each source actually unlocks

| source | unlocks | still blocked after |
|---|---|---|
| A. Legal | 5 compliance rules can fire | fraud detection — these find *irregularity* |
| B. Sanctions | vendor-level typologies | work-level labels; caught-fraud bias |
| C. Bid data | 3 bid-rigging typologies | supervised fraud training — features, not outcomes |
| D. Financial | coverage overlap with audits | needs FY2020-21 Nagaland specifically |
| E. Hindi text | the Indic tier gets real input | nothing — this one is clean |

**None of these unlocks calibration.** That needs audit *scope* — which units
were examined and cleared — so that negatives exist. Without scope, absence of
a finding is indistinguishable from absence of an audit, and NULL stays the
only honest value.
