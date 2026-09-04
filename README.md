# PARAKH: An Evidentiary-Confidence Layer for Public Fund Anomaly Detection

**A fraud-analytics engine that scores the trustworthiness of each record before it scores the risk of each transaction — because on self-certified government data, a confident anomaly detector mostly learns who files paperwork late.**

---

## Abstract

We present PARAKH, an analytics engine for public fund disbursement schemes that separates *evidentiary confidence* from *fraud risk* and refuses to emit a high-confidence flag on a low-confidence record.

The core insight: in systems where the **data-generating process is unreliable**, anomaly detection without epistemic gating learns *administrative artifacts*, not fraud.

PARAKH computes:

* a per-record **confidence score** \(C \in [0,1]\)
* a **risk score** \(R \in [0,1]\)

and routes decisions on the **pair \((R, C)\)** rather than collapsing them.

---

## Theoretical Foundation

---

### 1) Evidentiary Confidence

For record \(r\) with fields \(F\):

$$
C(r) = \exp\left(
w_1 \log C_{\text{comp}} +
w_2 \log C_{\text{temp}} +
w_3 \log C_{\text{recon}}
\right), \quad \sum_i w_i = 1
$$

> Implemented in log-space to avoid numerical underflow and preserve ranking among low-confidence records.

---

### 1.1 Completeness

$$
C_{\text{comp}}(r) =
\frac{\sum_{f \in F} v_f \cdot \mathbb{1}[r_f \text{ valid}]}
{\sum_{f \in F} v_f}
$$

Field weights:

$$
v_f = (1 - H_{\text{null}}(f)) \cdot H_{\text{value}}(f)
$$

Where:

* \(H_{\text{null}}\): entropy of null-pattern
* \(H_{\text{value}}\): entropy of actual values

> This prevents consistently-filled but non-informative fields from dominating confidence.

---

### 1.2 Temporal Coherence

Let \(\mathcal{O}\) be ordered event pairs:

$$
C_{\text{temp}}(r) =
\prod_{(a,b) \in \mathcal{O}}
\begin{cases}
1 & t_b \ge t_a \\
\exp(-\kappa |t_b - t_a|) & t_b < t_a
\end{cases}
$$

> Enforces **asymmetric penalty**:

* Valid ordering → no penalty
* Violations → exponential decay

Impossible timestamps (e.g., pre-scheme dates) force:

$$
C_{\text{temp}} \to 0
$$

---

### 1.3 Reconciliation

$$
C_{\text{recon}}(r) =
\exp\left(
-\lambda \cdot
\frac{|x_r^{(1)} - x_r^{(2)}|}
{|x_r^{(1)}| + |x_r^{(2)}| + \epsilon}
\right)
$$

> Uses symmetric normalization to avoid instability near zero.

---

### 2) Peer-Conditioned Cost Outlier

$$
z_{\text{cost}}(r) =
\frac{\log a_r - \operatorname{med}_{k,s}(\log a)}
{1.4826 \cdot \operatorname{MAD}_{k,s}(\log a)}
$$

Where:

* \(k\): semantic cluster
* \(s\): cost stratum

> Median + MAD ensure robustness up to 50% contamination.

---

### 3) Entity Resolution

Fellegi–Sunter linkage:

$$
\log \Lambda(a,b) =
\sum_j \log \frac{m_j(\gamma_j)}{u_j(\gamma_j)}
$$

* Parameters estimated via EM
* Blocking on trigram + district

> Note: Precision bounded by absence of ownership-level identifiers.

---

### 4) Work-Type Clustering & Duplicate Detection

For embeddings \(e_i, e_j\):

$$
D(i,j) =
\cos(e_i, e_j)
\cdot \mathbb{1}[d_i = d_j]
\cdot \exp\left(-\frac{|t_i - t_j|}{\tau_d}\right)
$$

Per-record duplicate score:

$$
D_{\max}(i) = \max_{j \neq i} D(i,j)
$$

> Explicit definition removes ambiguity in downstream risk model.

---

### 5) Vendor Concentration

$$
\text{HHI}_{c,k} =
\sum_{v \in V_{c,k}}
\left(\frac{a_v}{\sum_u a_u}\right)^2
$$

> Screening statistic only; not causal evidence.

---

### 6) Temporal Burst Detection

$$
B(d,m) =
\log \frac{n_{d,m} + \alpha}
{\lambda_d \Delta + \alpha}
$$

* Seasonally adjusted baseline required
* \(\alpha\): Laplace smoothing

---

### 7) Risk Aggregation

Due to lack of labels, **parametric estimation is not identifiable**.

Instead:

$$
R(r) =
\text{rank\_agg}\big(
z_{\text{cost}}, \text{HHI}, B, D_{\max}
\big)
$$

Where:

* Each component is normalized to percentile rank
* Aggregation = mean or Borda count

> Avoids arbitrary β-weight selection under unsupervised conditions.

---

### 8) Confidence-Gated Routing

$$
\text{route}(r) =
\begin{cases}
\text{INVESTIGATE} & R \ge \theta_R \;\wedge\; C \ge \theta_C \\
\text{REMEDIATE} & C < \theta_C \\
\text{MONITOR} & R \ge \theta_R \;\wedge\; C < \theta_C \text{ (post-fix)} \\
\text{CLEAR} & \text{otherwise}
\end{cases}
$$

> The system never emits a fraud hypothesis on low-confidence evidence.

---

### 9) Artifact-Invariance Test

Goal: ensure risk does not encode administrative capacity.

Let \(A\) be artifact features.

Train:

$$
A \sim R
$$

Constraint:

$$
\text{AUC}(R \rightarrow A) \le 0.5 + \epsilon
$$

> If violated, model is detecting reporting artifacts and must be disabled.

---

## Implementation Notes

* Confidence computed in log-space
* Blocking reduces entity resolution complexity from \(O(N^2)\) to \(O(Nb)\)
* Full graph rebuild preferred over incremental updates for correctness
* All outputs must include full **evidence chain**

---

## Calibration (Required Before Deployment)

The system is **non-operational without calibration**.

Must estimate:

* \(\theta_R, \theta_C\)
* \(\lambda, \kappa, \tau, \tau_d\)
* minimum peer-cell size

Without these:

> Scores are structurally correct but operationally meaningless.

---

---

# Stage 2 — Confidence Engine: Operating Semantics

Everything below describes **implemented, tested behaviour** of
`src/stage2/`, not intent. Where a number is quoted it was measured on the
20,000-record reference corpus (`seed=42`).

---

## 1. Confidence Interpretation

> **High confidence does not mean clean data.**
> It means: *internally consistent given the available evidence.*

\(C(r)\) answers one question only:

> If we asserted something about this record, could we defend the assertion
> with the evidence in hand?

This is a statement about the **evidentiary state of a record**, not about the
world. The consequences are deliberate and must not be read as bugs:

| situation | score | why |
| --- | --- | --- |
| Genuine fraud, filed perfectly | **high** | The evidence is intact. Detecting the fraud is \(R\)'s job, not \(C\)'s |
| An honest culvert, three placeholders and a reversed date | **near zero** | Nothing here can be defended |
| A record with almost nothing in it | **near zero** | Absence of evidence is not evidence of correctness |
| A record with *no* dates at all | temporal **dropped**, not scored 1.0 | Nothing to check is not the same as coherent |

That inversion is the design. PARAKH refuses to allege on evidence it cannot
stand behind, and confidence is the gate that enforces it.

**A high \(C\) is a licence to reason about a record. It is not a clean bill of
health.**

---

## 2. Lifecycle Awareness

\(C_{\text{recon}}\) reads Stage 1's `status` field, because the same spend
ratio means different things at different stages of a work's life.

| lifecycle | statuses | underspend | overspend |
| --- | --- | --- | --- |
| pre-completion | `proposed`, `approved`, `pending`, `ongoing`, `in progress` | **not penalised** | penalised |
| terminal | `completed`, `closed` | fully penalised | penalised |
| unknown | null, placeholder, unparseable, unrecognised | **half** penalised | penalised |

**Why proposed works are not penalised for low spend.** A work that has been
proposed but not executed *should* show near-zero expenditure. Charging it
would be penalising a record for behaving correctly — and worse, it would make
confidence a proxy for project age rather than for data reliability.

**Why overspend is always penalised.** Spending beyond the sanctioned amount is
not a reporting quirk at any stage. It requires a sanction revision, and that
revision should itself be on the record. Its absence is a real signal, so the
overspend term is never gated by status.

**Why unknown gets half.** When the stage cannot be determined we do not know
whether the gap is legitimate, so the model neither excuses nor condemns it.

---

## 3. Tolerance Behaviour

Overspend is measured from \(1 + \tau\), with \(\tau = 0.05\):

$$\text{overspend penalty} = \exp\big(-\lambda \cdot \max(0,\; r - (1+\tau))\big)$$

| \(r\) | 1.00 | 1.02 | 1.05 | 1.06 | 1.20 | 2.00 |
| --- | --- | --- | --- | --- | --- | --- |
| \(C_{\text{recon}}\) | 1.000 | **1.000** | **1.000** | 0.980 | 0.741 | 0.150 |

**Why small overspend is ignored.** Rounding, minor price variation and
final-bill adjustment routinely put a public work a percent or two over its
sanction. Penalising from the first rupee treated ordinary accounting noise as
a control failure and charged 3,178 records in the reference corpus for it.
\(\tau\) marks where accounting noise ends and a control failure begins.

Underspend has a matching threshold: it is free down to \(\theta_u = 0.2\), and
penalised below it — a work reported against a sanction while having consumed
under a fifth of it asserts something the money does not support.

---

## 4. Garbage Handling

Unusable financial values are **refused, not discounted**:

| condition | \(C_{\text{recon}}\) | effect on \(C\) |
| --- | --- | --- |
| `inf` / `-inf` (e.g. an overflowing literal) | **0.0** | \(C = 0\) by zero dominance |
| \(\lvert x\rvert >\) `IMPLAUSIBLE_AMOUNT_THRESHOLD` (1e15) | **0.0** | \(C = 0\) |
| sanction \(\le 0\) | 0.25 | strong penalty |

A moderate score on an unreadable number lets garbage survive aggregation. An
interim value of 0.25 for non-finite amounts was tried and rejected in audit for
exactly that reason.

The implausibility branch exists because `1e300` is *finite* and slips past a
non-finite check: a `1e300` sanction against a `1e300` spend gives \(r = 1.0\)
and would otherwise score a **perfect** reconciliation on two data-entry
accidents.

---

## 5. Calibration Disclaimer

> ### Confidence scores are comparative, not absolute.
> ### The system requires real-world calibration to be operational.

A score of 0.83 does **not** mean "83% trustworthy". It means this record ranks
where it ranks *against this corpus, under these parameters*. Specifically:

- **\(C_{\text{comp}}\) is corpus-relative.** Field weights \(v_f\) are
  estimated from the corpus, so the same record scores differently in a
  different corpus. Weights are frozen and emitted with every score set
  (`outputs/stage2_field_weights.json`) and can be re-injected via
  `ConfidenceModel.score(..., field_weights=...)` to keep batches comparable.
- **Every parameter is a default, not an estimate**: \(w\), \(\kappa\),
  \(\lambda\), \(\gamma\), \(\delta\), \(\tau\), \(\theta_u\) and the
  null-reason credits. None has been fitted to observed data.
- **\(\theta_C\) must be set from the empirical distribution**, never from an
  absolute intuition such as "0.5 means half the data is there".
- **The reference corpus is synthetic.** No real MPLADS export has been scored.

Until calibrated, the scores are **structurally correct and operationally
meaningless** — usable for ranking and triage, not for any absolute claim about
a record.

---

## 6. Stage 3 Contract

### What Stage 3 may rely on

| guarantee | detail |
| --- | --- |
| **Range** | `confidence` \(\in [0,1]\). Never NaN, never inf — asserted in `ConfidenceModel.score` |
| **Monotonicity** | Lower means less trustworthy. Verified against Stage 1's ground-truth ledger: no injected defect 0.994 → missing 0.777 → date-order 0.491 → unparseable 0.351 → pre-scheme 0.098 |
| **Zero means reject** | `confidence == 0.0` is a refusal, not a low score. Reached only by an unevidenced record or a refused component. Never route a zero to INVESTIGATE |
| **Alignment** | Row count, order and index identical to `corpus.records`; `attach_confidence` raises rather than misalign |
| **Determinism** | Same corpus, same config, same bytes. No wall-clock, no RNG |
| **Breakdown** | All 17 columns of `BREAKDOWN_COLUMNS` are present after `attach_confidence` |

### The columns

```
confidence  completeness  temporal  reconciliation
completeness_defined  temporal_defined  reconciliation_defined  n_components_used
n_valid_fields  critical_missing_count  critical_deficit  cluster_penalty_factor
temporal_pairs_evaluated  temporal_hard_fail
reconciliation_branch  lifecycle_state  spend_ratio
```

### > Stage 3 MUST use the breakdown, not only the scalar confidence.

The scalar is a summary and it is lossy. Three cases where reading it alone
gives the wrong answer:

1. **`temporal = 1.0` can mean "coherent" or "nothing to check".** Only
   `temporal_pairs_evaluated` and `temporal_defined` separate them. 16.4% of
   the reference corpus has no evaluable milestone pair.
2. **Two records at \(C = 0\) are not alike.** One may have a fabricated
   timeline (`temporal_hard_fail`), another an unreadable amount
   (`reconciliation_branch == "non_finite"`). They belong in different
   remediation queues.
3. **A low `reconciliation` may be entirely legitimate.** Check
   `lifecycle_state` before treating a low `spend_ratio` as a signal.

`n_components_used < 3` means a component was **dropped as unmeasurable** and
the weights renormalised for that record — not that it scored badly.

### Explaining a record

`explain_confidence(records, row)` returns a JSON-serialisable dict with
per-component scores, effective weights, penalty attribution, the evidence
behind each component and ordered human-readable reasons. It **reads stored
outputs and recomputes nothing**, so an explanation can never disagree with the
score it explains.

```python
from src.stage2.confidence import attach_confidence, explain_confidence

result = attach_confidence(corpus)
explain_confidence(corpus.records, 0)["summary"]
# 'Confidence refused (0.0): temporal could not be evidenced at all,
#  and no other component can compensate for it.'
```

---

# Stage 3 — Peer Structure: Calibration, Evaluation and Reproducibility

## 1. Calibration Disclaimer

> ### The system is not operational until calibrated on real MPLADS data.

Every Stage 3 threshold is a **default, not an estimate**. Nothing here has been
fitted to observed outcomes, and several were chosen against a *synthetic*
corpus whose own defect rates were themselves chosen by a generator.

`outputs/stage3_calibration_report.json` makes each one observable: what it
governs, where it came from, the value actually used, and what goes wrong if it
is wrong. Parameters marked `"source": "default"` are the admission that nobody
estimated them.

| parameter | default | source |
| --- | --- | --- |
| `PEER_STAT_MIN_CONFIDENCE` | 0.5 | **default** |
| `PEER_CELL_MIN_SIZE` | 15 | Stage3.md §8.1 |
| `PEER_STAT_MIN_REFERENCE` | 8 | **default** |
| `DUPLICATE_SIMILARITY_THRESHOLD` | 0.85 | **default** |
| `HDBSCAN_MIN_CLUSTER_SIZE` | 2 | swept against ground truth |
| `SVD_COMPONENTS` | 16 | swept against ground truth |

The report also carries the distributions those defaults produce — cluster sizes,
peer-cell sizes, norm coverage per feature, and deviation percentiles. That
matters more than it sounds: whether `PEER_CELL_MIN_SIZE = 15` is right depends
on a distribution nobody had looked at. On the reference corpus the median cell
holds 142 records, so the floor is doing little; on a smaller register it could
be discarding everything.

**The deviation percentiles are descriptive, not thresholds.** Stage 4 must not
lift p99 and call it a flag boundary — that fits the cut to whatever happens to
be in this corpus.

## 2. Duplicate Detection — how it is measured, and what that does not prove

Stage 1's duplicate channel clones work names from *any* row, so only 70 of
1,000 injected clones land in the same district — and `Stage3.md` §9.1's
`1[dᵢ = dⱼ]` deliberately excludes the rest. Precision and recall against that
channel measured the mismatch between two definitions, not the detector.
Reported earlier as 0.047 / 0.119, those figures were meaningless.

`src/stage3/evaluation.py` injects duplicates matching the detector's own
definition — **same district, near-identical text, within 30 days** — and
carries a `duplicate_id` that is returned as a *separate object*, never a frame
column, so it is structurally impossible for it to reach the pipeline.

Measured on 20,000 records with 300 injected pairs
(`outputs/stage3_duplicate_eval.json`):

| metric | value |
| --- | --- |
| precision | **0.939** |
| recall | **0.920** |
| F1 | **0.929** |
| injected-pair median score | 0.920 vs corpus median 0.119 |

**What this does not prove.** The evaluation measures the detector against
duplicates *it was designed to find*. Real double-claiming may look different —
a rewritten description, a split across two financial years, a different
implementing agency. These numbers are a floor on competence, not a claim about
field performance.

## 3. Corpus-Relative Behaviour

Stage 3 estimates its feature space from whatever corpus is in front of it: the
TF-IDF vocabulary, the IDF weights and the cost-strata quantiles all come from
the data being scored. Three consequences:

1. **The same record can change cost stratum between runs** without changing at
   all, because the quantile boundaries moved.
2. **A record's embedding depends on the corpus's vocabulary**, so its cluster
   can change when the corpus does.
3. **Peer norms are corpus-relative by construction** — that is intended, since
   "unusual given similar records" has no meaning without a reference
   population, but it means a deviation of 2.5 is not comparable across runs
   unless the reference population is pinned.

For an audit system this is a real defect: a finding must survive being
re-derived next quarter.

## 4. Reproducibility Contract

Freezing the feature space fixes (1) and (2).

```
artifacts/tfidf_vocab.json     vocabulary + IDF weights
artifacts/cost_strata.json     quantile edges, in log and rupee scale
artifacts/stage3_config.json   the exact parameter set used
```

**Default is compute-and-save. Reuse is opt-in** — silently scoring a new corpus
against a stale vocabulary is worse than recomputing one.

```python
SemanticLayer(SemanticConfig(reuse_artifacts=True)).run(corpus)
```

Freezing the vocabulary alone would not be enough: IDF is re-estimated from
document frequencies, so the same token would carry a different weight on a
different corpus. Both are frozen, at full precision — an artefact that exists
to reproduce a run must not round.

**Drift is measured and the run is gated on it**
(`outputs/stage3_reproducibility_report.json`): unseen-token rate against the
frozen vocabulary, and total-variation distance between the frozen and observed
stratum occupancies. Beyond `MAX_UNSEEN_TOKEN_RATE` (35%) or `MAX_STRATA_DRIFT`
(0.35) the run is **rejected**, because a corpus the frozen space cannot
describe would embed as near-zero vectors, cluster as noise, and silently lose
every peer cell. Failing loudly is the correct response.

### What reuse reproduces, and what it does not

Measured by re-running the same corpus against frozen artefacts:

| | reproduces? |
| --- | --- |
| cost stratum | **exactly** |
| cluster *partition* | **exactly** — adjusted Rand index 1.0 |
| `cluster_label` | **exactly** |
| `cluster_id` (the integer) | **no** |

`cluster_id` is **run-local**. HDBSCAN numbers clusters in an order that turns
on floating-point ties at the 1e-16 level, so the integers permute even when the
grouping is bit-identical.

> **Downstream code that must survive across runs keys on `cluster_label`,
> never on `cluster_id`.**

`cluster_label` is part of the Stage 4 column contract for exactly this reason.

## Limitations

* No beneficial ownership → no true collusion detection
* Confidence gating reduces recall
* Peer-cell sparsity limits coverage
* HHI confounded by market structure
* No causal claims — outputs are screening hypotheses only

---

## Status

**Mathematically consistent, not yet empirically validated.**

All performance metrics:

* TBD via controlled evaluation

---

## License

MIT
