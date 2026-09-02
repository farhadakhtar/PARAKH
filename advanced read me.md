Great pair of drafts to work from. Before the merged README, here's exactly what I fixed mathematically (mostly pulling v2 forward, plus a few errors that were in *both* versions):

**Math corrections carried in:**
1. **§1.2 (biggest error in v1):** the logistic $\sigma(x) < 1$ for *any* finite $x$, so it penalized **correct** orderings too. Replaced with v2's asymmetric penalty (exactly 1 when satisfied), with v1's tolerance $\tau$ folded back in as a shift.
2. **§1.3:** `max()` normalization is asymmetric and undefined at $(0,0)$ → symmetric normalization $+\ \epsilon$, with the $[e^{-\lambda}, 1]$ bound stated.
3. **§1.1:** v2's formula was right but its rationale didn't match it; both entropy terms are now defined, normalized, and given true justifications, plus the all-null convention.
4. **§4/§7:** v1 used $D_{\max}$ in the risk model **without ever defining it** — now defined.
5. **§7:** the $\beta$-weighted logistic is unidentifiable without labels (and mixed record-level scores with cell-level HHI) → v2's percentile-rank aggregation, with the tradeoff honestly stated.
6. **§8:** REMEDIATE and MONITOR **overlapped** (both matched $C < \theta_C \wedge R \ge \theta_R$). Now a non-overlapping 3-way rule with MONITOR as the post-remediation re-entry, shown as a 2×2 matrix.
7. **§2:** peer-cell strata defined as *sanction slabs*, not realized-cost bands (banding on realized cost would absorb the outlier being tested); unscored-cell rule made explicit.
8. **§9:** test direction fixed — AUC of predicting artifact proxies *from* $R$, max over proxies.

**Kept from v1:** all the prose, the CAG/MGNREGS/eSAKSHI background, the prior-art table, the results-honesty posture, and the full limitations. **Added:** architecture diagram, component map, routing matrix, calibration parameter table.

````markdown
# PARAKH: An Evidentiary-Confidence Layer for Public Fund Anomaly Detection

**A fraud-analytics engine that scores the trustworthiness of each record before it scores the risk of each transaction — because on self-certified government data, a confident anomaly detector mostly learns who files paperwork late.**
`https://github.com/farhadakhtar/PARAKH`

---

## Abstract

We present PARAKH, an analytics engine for public fund disbursement schemes that separates *evidentiary confidence* from *fraud risk* and refuses to emit a high-confidence flag on a low-confidence record. The core insight is that in schemes like MPLADS the binding constraint is not the absence of anomaly detection but the unreliability of the source records: the CAG's 2010 performance audit found handing-over of assets unrecorded for 14,828 of 15,049 sampled works (98.53%), and work completion on the current eSAKSHI portal is self-certified by the implementing agency being monitored. An unsupervised detector trained on such data maximizes its objective by learning reporting artifacts — district-level data-entry latency, retrospective backfill — which correlate with administrative capacity, not corruption.

PARAKH computes a per-record confidence $C \in [0,1]$ from field completeness, temporal coherence and cross-source reconciliation; a raw risk $R \in [0,1]$ from peer-conditioned cost outliers, near-duplicate works, vendor-graph concentration and temporal bursts; and routes on the *pair* $(R, C)$: high-$R$/high-$C$ to investigation, low-$C$ to data remediation. It is a Rust core with Python bindings and a DuckDB-backed store.

---

## Background

### Why an evidentiary layer instead of a fraud classifier?

**The label problem is not the hard part.** Every practitioner reaches for unsupervised methods here — Isolation Forest, autoencoders, robust z-scores — because labelled fraud is essentially nonexistent in Indian scheme data. That reflex is correct and insufficient. Unsupervised detection answers "which records are statistically unusual?" What an auditor needs answered is "which records are unusual *for reasons that are not administrative*?" Those are different questions, and on this data they have substantially different answers.

**Reporting artifacts dominate the variance.** Consider what actually varies across MPLADS work records: whether a district staffed its 2%-capped monitoring cell, whether an implementing agency backfilled six months of entries in one sitting before an audit, whether `date_of_administrative_approval` was entered as a real date or a placeholder. The CAG found dates predating the 1993 scheme launch. These artifacts are large, systematic, and geographically clustered — precisely the signature an anomaly detector rewards. The resulting model is a well-calibrated predictor of administrative dysfunction wearing a fraud label, and it will concentrate its flags on poor districts.

**The self-certification loop is the root defect.** eSAKSHI states that "only those works that are marked as complete by the Implementing Agencies are reflected as completed works on the public dashboard." The entity being monitored controls the monitoring signal. No amount of modelling on the downstream record repairs an upstream signal that the subject generates. MGNREGS is the proof by counterexample: after Aadhaar seeding reached 99.67% of active workers and geo-tagged twice-daily photo attendance was mandated, the Ministry of Rural Development in July 2025 ordered *manual* re-verification of digital attendance following documented app manipulation. Digitising a gameable signal produces gameable digital records at higher throughput.

**So confidence must be a first-class output, not a preprocessing step.** The conventional pipeline treats data quality as a cleaning stage that happens before modelling and is then discarded. PARAKH carries $C$ through to the decision boundary and uses it to gate the flag. This costs recall — genuinely fraudulent records with poor documentation are suppressed rather than flagged — and that trade is deliberate: a suppressed record enters the remediation queue, which is the correct destination for a work whose completion cannot be evidenced.

### Prior Art

Operational systems in this class, with the columns that show where PARAKH does *not* win.

| System | Jurisdiction | Graph / entity resolution | Confidence-gated scoring | Beneficial ownership data | Operational scale |
|---|---|---|---|---|---|
| ARACHNE (EC) | EU cohesion funds | Yes — >100 indicators, 7 risk categories | No | Yes (Orbis, WorldCompliance) | ~1.2M projects, ~€330bn (EC-reported) |
| Recovery Operations Center (US, 2009–15) | Recovery Act | Yes — link analysis on entity relationships | No | Yes (federal registries) | 1.7M entities / $36.4bn (GAO-15-814) |
| GRAS (World Bank / Paraíba pilot) | Brazil sub-national | Yes — ~60 red-flag indicators | No | Yes (supplier registries) | Pilot: 850+ collusion-flagged suppliers |
| eSAKSHI (MoSPI) | MPLADS | No | No | No | All States/UTs, FY2023-24 onward |
| NREGASoft + Social Audit | MGNREGS | No | No (human social audit instead) | No | National, statutory |
| `PARAKH` | MPLADS (scheme-agnostic core) | Yes — probabilistic linkage + bipartite projection | **Yes** | **No — public data lacks it** | **TODO: measure** |

The honest reading of this table: ARACHNE, ROC and GRAS are all more capable than PARAKH on the dimension that matters most for collusion detection, because each ingests beneficial-ownership data that MPLADS public data does not expose. PARAKH's contribution is the confidence-gating column, and its handicap is the ownership column. ARACHNE's own charter — that it "does not supply any proof of error, irregularity or fraud" — is the correct posture and PARAKH inherits it.

---

## Architecture

```
                ┌────────────────────────────────────────┐
                │    INGESTION · Arrow record batches    │
                │    Dataful / data.gov.in MPLADS dumps  │
                └───────────────────┬────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
 ┌─────────────────────────────┐          ┌─────────────────────────────┐
 │  CONFIDENCE BRANCH (§1)     │          │  RISK BRANCH (§2–§7)        │
 │  C_comp · C_temp · C_recon  │          │  §2 cost outlier    z_cost  │
 │  → C ∈ [0,1], log-space     │          │  §3 entity graph (linkage)  │
 └──────────────┬──────────────┘          │  §4 duplicates      D_max   │
                │                         │  §5 concentration   HHI     │
                │                         │  §6 bursts          B       │
                │                         │  §7 rank aggregation → R    │
                │                         └──────────────┬──────────────┘
                │                                        ▼
                │                         ┌─────────────────────────────┐
                │                         │  ARTIFACT-INVARIANCE GATE   │
                │                         │  §9  AUC(R→A) ≤ 0.5 + ε     │
                │                         │  fail → R-branch disabled,  │
                │                         │  remediation continues      │
                │                         └──────────────┬──────────────┘
                │                                        │
                └────────────────────┬───────────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │   ROUTER (§8) on (R, C)   │
                       └─────┬────────┬───────┬────┘
                             ▼        ▼       ▼
                       INVESTIGATE  REMEDIATE  CLEAR
                                        │
                                        ▼  re-scored after remediation
                                   MONITOR (R ≥ θ_R persists)
```

Every component, the level it operates at, and what it feeds:

| § | Component | Symbol | Granularity | Feeds |
|---|---|---|---|---|
| 1.1 | Completeness | $C_{\text{comp}}$ | record | $C$ |
| 1.2 | Temporal coherence | $C_{\text{temp}}$ | record | $C$ |
| 1.3 | Reconciliation | $C_{\text{recon}}$ | record | $C$ |
| 2 | Cost outlier | $z_{\text{cost}}$ | record vs. peer cell | $R$ |
| 3 | Entity resolution | linkage graph | entity | §5, §6 |
| 4 | Duplicate score | $D_{\max}$ | record | $R$ |
| 5 | Vendor concentration | HHI | (constituency, work-type) cell → record | $R$ |
| 6 | Temporal burst | $B$ | (district, month) → record | $R$ |
| 7 | Rank aggregation | $R$ | record | routing |
| 8 | Routing | route(r) | record | output queues |
| 9 | Artifact-invariance | AUC gate | corpus | enables/disables R-branch |

---

## Theoretical Foundation

### 1) Evidentiary confidence

For record $r$, confidence is a weighted geometric mean of three independently-computable factors:

$$
C(r) = C_{\text{comp}}(r)^{w_1}\, C_{\text{temp}}(r)^{w_2}\, C_{\text{recon}}(r)^{w_3}
     = \exp\!\Big(w_1 \ln C_{\text{comp}} + w_2 \ln C_{\text{temp}} + w_3 \ln C_{\text{recon}}\Big),
\qquad w_1 + w_2 + w_3 = 1
$$

The geometric form is deliberate: no factor can compensate for another. An arithmetic mean would let a strong reconciliation score rescue a record with impossible dates; here any factor $\to 0$ forces $C \to 0$. The log-space expression on the right is the implementation form (see Implementation).

#### 1.1 Completeness

$$
C_{\text{comp}}(r) = \frac{\sum_{f \in F} v_f\, \mathbb{1}[r_f \text{ valid}]}{\sum_{f \in F} v_f},
\qquad v_f = \big(1 - H_{\text{null}}(f)\big)\cdot H_{\text{value}}(f)
$$

where $H_{\text{null}}(f)$ is the normalized entropy of the null/not-null pattern of field $f$ across the corpus and $H_{\text{value}}(f)$ the normalized entropy of its observed values, both in $[0,1]$. The two factors do different jobs. $H_{\text{value}}(f) \to 0$ for a field filled with a constant — a scheme name stamped on every row certifies nothing, and its weight vanishes. $1 - H_{\text{null}}(f)$ down-weights fields whose fill-status varies unpredictably across the corpus: for those fields, *whether a value is present* is entangled with reporting behaviour, which is precisely the artifact channel §9 tests for. Fields never observed get $v_f = 0$ by convention; a record with $\sum_f v_f = 0$ is assigned $C_{\text{comp}} = 0$ and routes to remediation.

#### 1.2 Temporal coherence

Let $\mathcal{O}$ be the scheme's mandated ordering pairs — proposal receipt $\le$ administrative approval $\le$ completion. For each pair, with tolerance $\tau$ in days:

$$
C_{\text{temp}}(r) = \prod_{(a,b) \in \mathcal{O}} g\big(t_b - t_a + \tau\big),
\qquad
g(\Delta) = \begin{cases} 1 & \Delta \ge 0 \\ \exp(-\kappa\,|\Delta|) & \Delta < 0 \end{cases}
$$

with $\kappa > 0$ in day⁻¹. The penalty is **asymmetric**: a satisfied ordering costs exactly nothing (a logistic would penalize every finite gap), and a violation decays exponentially in its magnitude, with sub-tolerance slips effectively free. Dates outside the scheme's domain — `date_of_administrative_approval` preceding the 1993-04-01 launch, as the CAG found — are treated as maximally violating, so $C_{\text{temp}} \to 0$ and the record never reaches the fraud queue.

#### 1.3 Reconciliation

For a quantity reported by two independent sources — work-level sanction totals against MoSPI fund-release aggregates, IA-reported completion against district totals:

$$
C_{\text{recon}}(r) = \exp\!\left(-\lambda\,
\frac{|x_r^{(1)} - x_r^{(2)}|}{|x_r^{(1)}| + |x_r^{(2)}| + \epsilon}\right)
\;\in\; \big[e^{-\lambda},\, 1\big]
$$

Symmetric normalization with $\epsilon$ guarding the denominator. Perfect agreement — including both sources reporting zero — scores 1; maximal disagreement scores $e^{-\lambda}$.

### 2) Peer-conditioned cost outlier

Raw cost comparison across districts is confounded by genuine input-price variation. We compare each work against its *peer cell* — works in the same semantic cluster $k$ (from §4) and the same sanction slab $s$. Slabs are fixed administrative amount bands, **not** realized-cost bands: banding on realized cost would partially absorb the very outlier being tested.

$$
z_{\text{cost}}(r) = \frac{\log a_r - \operatorname{med}_{k,s}(\log a)}{1.4826 \cdot \operatorname{MAD}_{k,s}(\log a)}
$$

The constant $1.4826$ makes MAD a consistent estimator of $\sigma$ under normality. Median and MAD are used rather than mean and standard deviation because the contamination we are searching for is in the same statistic we would otherwise estimate from — a 20% inflated cell breaks a mean and leaves a median intact up to a 50% breakdown point. Cells with fewer than $n_{\min}$ works, or with $\operatorname{MAD}(\log a) = 0$, are **not scored**: $z_{\text{cost}}$ is undefined there, the aggregate in §7 is taken over the remaining components, and the omission is written to the evidence chain.

### 3) Entity resolution

Vendor and implementing-agency names arrive as unnormalized free text with no persistent identifier. We use Fellegi–Sunter probabilistic linkage over blocked candidate pairs. For record pair $(a,b)$ with agreement vector $\gamma$:

$$
\log \Lambda(a,b) = \sum_{j} \log \frac{m_j(\gamma_j)}{u_j(\gamma_j)}
$$

where $m_j = P(\gamma_j \mid \text{match})$ and $u_j = P(\gamma_j \mid \text{non-match})$, estimated by EM on the unlabelled candidate set. Blocking is on normalized-name trigram and district to keep candidate generation at $O(N \cdot b)$ rather than $O(N^2)$; at $N \approx 10^5$ works the unblocked comparison space is $\approx 5 \times 10^9$ pairs and blocking is not optional.

Matched records collapse into entity nodes. **Where the public data ends:** $\gamma$ can only include name, district and IA co-occurrence. Bank account, PAN, GSTIN and director overlap — the features that make ARACHNE work — are unavailable, so linkage precision is bounded well below what an ownership-enriched system achieves.

### 4) Work-type clustering and duplicate detection

Free-text `work_name` is embedded and clustered to produce the peer cells of §2 and to surface near-duplicate works (the same asset sanctioned twice). For works $i,j$ with embeddings $e_i, e_j$:

$$
D(i,j) = \big[\cos(e_i, e_j)\big]_+ \cdot \mathbb{1}[d_i = d_j] \cdot \exp\!\left(-\frac{|t_i - t_j|}{\tau_d}\right),
\qquad
D_{\max}(i) = \max_{j \neq i} D(i,j)
$$

with $[x]_+ = \max(x, 0)$. Conjunction rather than a weighted sum is deliberate: two culverts with identical descriptions in different districts are not a duplicate, and a soft weighting lets high semantic similarity overwhelm the geographic mismatch. $D_{\max}$ is the per-record quantity that enters the risk aggregate in §7.

### 5) Vendor concentration

Award concentration per (constituency, work-type) cell, via Herfindahl–Hirschman index over resolved entities:

$$
\text{HHI}_{c,k} = \sum_{v \in V_{c,k}} \left(\frac{a_v}{\sum_{u \in V_{c,k}} a_u}\right)^2
\;\in\; \left[\tfrac{1}{|V_{c,k}|},\, 1\right]
$$

where $a_v$ is the sanctioned amount awarded to vendor $v$ in the cell. A record inherits the HHI of its cell $(c(r), k(r))$. $\text{HHI} \to 1$ indicates a single vendor capturing a cell. This is a *screening* statistic, not evidence: legitimate concentration arises from thin local supplier markets, and the report must say so on every flag.

### 6) Temporal burst detection

Sanction arrivals per (district, month) modelled as a Poisson process. The baseline rate is **seasonal** — estimated from the same calendar month across trailing years, so March is compared to March:

$$
B(d,m) = \log \frac{n_{d,m} + \alpha}{\hat{\lambda}_{d,m}\, \Delta + \alpha}
$$

with $\alpha$ a Laplace smoothing term and $\Delta$ the window length in the rate's time units. $B > 0$ means above seasonal baseline. Fiscal-year-end March bursts are a documented, near-universal pattern in Indian public expenditure and are therefore *expected*; the seasonal baseline removes them rather than flagging them.

### 7) Risk aggregation

**Why not learned weights?** The obvious parametric form $R = \sigma(\beta_0 + \sum_j \beta_j x_j)$ is unidentifiable here: on unlabelled data the likelihood is flat in $\beta$, so every weight vector fits equally well and any specific choice is arbitrary authority dressed as calibration. It would also mix quantities at different granularities — record-level scores with cell-level statistics. PARAKH aggregates by rank instead:

$$
R(r) = \frac{1}{|\mathcal{J}(r)|} \sum_{j \in \mathcal{J}(r)} \rho_j(r),
\qquad
\rho_j(r) = \text{percentile rank of component } j \text{ among all scored records}
$$

with $\mathcal{J}(r) = \{\, z_{\text{cost}}(r),\; \text{HHI}_{c(r),k(r)},\; B_{d(r),m(r)},\; D_{\max}(r) \,\}$ minus any undefined components (per §2), each omission recorded in the evidence chain. $R \in [0,1]$; the mean of percentile ranks is equivalent to a Borda count.

The tradeoff is stated, not hidden: rank aggregation is invariant to monotone rescaling — which removes every normalization dispute and is robust to heavy tails — but discards metric spacing within the tail. The difference between a 3.5σ and a 40σ cost outlier survives only as the number of records ranked between them. The evidence chain always exposes the raw components for this reason.

### 8) Confidence-gated routing

The emitted disposition is a function of the pair $(R, C)$, evaluated with REMEDIATE taking precedence over every risk-based route — a record that cannot be evidenced is never a fraud allegation, regardless of its risk score:

$$
\text{route}(r) = \begin{cases}
\text{INVESTIGATE} & C \ge \theta_C \;\wedge\; R \ge \theta_R \\
\text{REMEDIATE} & C < \theta_C \\
\text{CLEAR} & C \ge \theta_C \;\wedge\; R < \theta_R
\end{cases}
$$

| | **$C \ge \theta_C$** (evidence adequate) | **$C < \theta_C$** (evidence inadequate) |
|---|---|---|
| **$R \ge \theta_R$** (high risk) | INVESTIGATE | REMEDIATE → MONITOR if still $R \ge \theta_R$ after repair |
| **$R < \theta_R$** (low risk) | CLEAR | REMEDIATE |

**MONITOR is not a primary disposition.** A record routed to REMEDIATE is re-scored after remediation: if $C$ recovers past $\theta_C$, the primary rule applies again; if $C$ remains below $\theta_C$ while $R$ remains $\ge \theta_R$, the record is held in MONITOR for manual escalation rather than re-queued mechanically.

A high-risk, low-confidence record does not become a fraud allegation. It becomes a documentation request — which is both the epistemically honest response and, per the CAG findings, the one that addresses the actual dominant failure.

### 9) Artifact-invariance test

The claim that PARAKH is not learning reporting artifacts is falsifiable and must be tested, not asserted. Let $A = \{A_1, \dots, A_p\}$ be administrative-capacity proxies (district data-entry latency, backfill burstiness, field null-rates). For each proxy, use $R$ alone as the scorer for the binary task "$A_j$ above its corpus median", and require:

$$
\max_{j} \; \operatorname{AUC}\big(A_j \leftarrow R\big) \;\le\; 0.5 + \epsilon
$$

If any artifact proxy is predicted from the risk score above chance by more than $\epsilon$, the model is an artifact detector and the R-branch ships disabled — confidence scoring and remediation continue to run. **TODO: measure** — this test is specified and not yet run.

---

## Implementation

### Core (`parakh-core`, Rust)

Columnar ingestion and scoring over Arrow record batches. Confidence and risk components are computed as vectorized passes; the per-record cost is dominated by the §3 blocking pass, not the arithmetic. One non-obvious decision: confidence factors are stored as `f32` log-space accumulators — exactly the right-hand form of §1 — rather than products, because $C_{\text{temp}}$ is a product over ordering constraints and underflows to zero on records with many violations — which is semantically correct but destroys the ability to rank *among* bad records for the remediation queue.

### Entity graph (`parakh-graph`)

`petgraph` over resolved entity nodes with award edges. HHI and bipartite projection are computed per cell. The graph is rebuilt rather than incrementally maintained — at MPLADS scale (~$10^5$–$10^6$ works) a full rebuild is cheaper than maintaining incremental correctness, and correctness matters more than latency for an audit tool that runs nightly.

### Semantic layer (`parakh-nlp`, Python)

Multilingual sentence embeddings over `work_name`, HDBSCAN for cluster discovery. Indian scheme text is code-mixed (English work descriptions with transliterated Hindi/regional terms), so a monolingual English encoder degrades badly. **TODO: measure** cluster purity against a hand-labelled sample.

### Case store (`parakh-store`, DuckDB)

Every flag persists with its full evidence chain: which rule fired, the peer cell it was compared against, the cell's median and MAD, the contributing records, and $C$ with its three factors. A flag with no reproducible derivation is not admissible in an audit context and is treated as a bug.

### API surface

```rust
let corpus = Corpus::from_parquet("mplads_17ls.parquet")?;
let conf   = ConfidenceModel::default().score(&corpus)?;         // §1
let graph  = EntityGraph::resolve(&corpus, LinkageConfig::default())?; // §3
let risk   = RiskModel::new(&graph).score(&corpus)?;             // §2, §4–§7
let queue  = Router::new(theta_r, theta_c).route(&risk, &conf)?; // §8
```

---

## Calibration

The system is **non-operational without calibration**. Scores are structurally correct but operationally meaningless until every parameter below is estimated on a pilot corpus with a sensitivity analysis, and versioned — an audit tool whose flags move because someone moved a knob is worse than no tool.

| Parameter | Role | Section |
|---|---|---|
| $w_1, w_2, w_3$ | confidence factor weights, $\sum = 1$ | §1 |
| $\kappa$ | temporal-violation decay (day⁻¹) | §1.2 |
| $\tau$ | ordering tolerance (days) | §1.2 |
| $\lambda$ | reconciliation penalty | §1.3 |
| $n_{\min}$ | minimum peer-cell size | §2 |
| $\tau_d$ | duplicate temporal window | §4 |
| $\alpha$ | Laplace smoothing | §6 |
| $\epsilon$ | artifact-invariance tolerance | §9 |
| $\theta_R, \theta_C$ | routing thresholds | §8 |

---

## Results

Not yet measured. Per this repository's standard, no figure appears here until it is produced by a stated command.

| Metric | Value | Notes |
|---|---|---|
| Ingest throughput (works/s) | **TODO: measure** | 17th LS dataset, 73,305 rows |
| Entity resolution precision | **TODO: measure** | vs. hand-labelled sample, 18th LS vendor names |
| Entity resolution recall | **TODO: measure** | same sample |
| Peer-cell cluster purity | **TODO: measure** | hand-labelled work-type sample |
| Artifact-invariance AUC (§9) | **TODO: measure** | must be ≤ 0.5 + ε to ship enabled |
| Flag precision @ CAG-confirmed | **TODO: measure** | proxy labels from CAG Report 31/2010 annexures |
| Full-corpus scoring latency | **TODO: measure** | single node, core-pinned |

Reproduction command once implemented: `cargo bench --bench full_corpus` and `python -m parakh.eval --split holdout`.

**The number that will matter most is the artifact-invariance AUC**, not the flag precision. A high flag precision against CAG proxy labels is achievable by a model that has learned which districts get audited, and the two are separable only by §9.

**Data substrate.** Work-level records from Dataful/data.gov.in sourced from MoSPI: 17th LS (dataset 18533, ~73,305 rows, 17 columns), 16th LS (18534), 15th LS (18535), sitting Rajya Sabha (18539, ~13,638 rows). Vendor names exist **only** for the 18th LS (dataset 22565). Bank account, PAN, GSTIN and payment-level records are not public in any dataset.

---

## Quick Start

```bash
cargo add parakh-core   # TODO: not yet published
pip install parakh      # TODO: not yet published
```

```python
from parakh import Corpus, Pipeline

corpus = Corpus.from_csv("mplads_17ls.csv")
result = Pipeline.default().run(corpus)

# Records that cannot be evidenced — the dominant bucket on real MPLADS data
print(result.queue("REMEDIATE").head())

# Flags that survive confidence gating, each with its evidence chain
for flag in result.queue("INVESTIGATE"):
    print(flag.explain())   # peer cell, median, MAD, contributing records, C factors
```

---

## Requirements

- Rust 1.75+ (MSRV), stable channel. No nightly features.
- Python 3.10+ for the semantic layer.
- x86_64 or aarch64. No CPU-feature requirements beyond baseline; the hot paths are memory-bound, not SIMD-bound.
- DuckDB 0.10+.
- ~8 GB RAM for full-corpus scoring at $10^5$-work scale (estimated from Arrow batch sizing, **not measured**).

---

## Limitations

**No beneficial-ownership data, therefore no genuine collusion detection.** This is the central limitation and it is not fixable in the open-data setting. PARAKH can detect that one *named* vendor won repeatedly; it cannot detect that three differently-named vendors share a director, an address or a bank account — which is how collusion is actually structured, and precisely what ARACHNE and GRAS were built to catch. Any claim that this system detects vendor collusion on public MPLADS data is false. Closing this requires MCA21/GSTN linkage that only the Ministry can authorize.

**Confidence gating trades recall for defensibility.** A sophisticated actor who files complete, temporally-coherent, internally-reconciled paperwork while stealing will pass §1 with a high $C$ and be evaluated purely on the risk branch. Good documentation is not evidence of good faith, and the gate is one-directional: it suppresses false confidence, it does not detect competent fraud.

**Rank aggregation is scale-free, and pays for it.** Percentile ranks remove every normalization dispute — and also remove metric spacing: the difference between a 3.5σ and a 40σ cost outlier survives only as the number of records ranked between them. The evidence chain exposes raw components for exactly this reason, but consumers of $R$ alone should know what it aggregates.

**Peer cells require enough peers.** Districts or work types with few comparable works produce unstable MAD estimates and unreliable $z_{\text{cost}}$. The current design suppresses scoring below a minimum cell size rather than reporting a noisy score, which systematically under-covers rare work types — where unusual costs are arguably most likely.

**HHI concentration is confounded by thin markets.** A single vendor in a remote district may be the only qualified contractor. Every concentration flag is a hypothesis requiring local knowledge to adjudicate, and the system has none.

**Satellite verification is not implemented and should not be promised.** The obvious answer to self-certification is independent imagery verification of asset existence. Free Sentinel-2 is 10 m resolution; typical MPLADS works are culverts, borewells and single community halls, which are below that resolution. Verification is feasible only for larger footprints (buildings, ponds, road stretches) and reliably only with commercial sub-metre imagery. Building it as a general capability on free imagery would be an overclaim.

**Proxy labels are weak and possibly circular.** CAG-derived labels reflect what was *audited and found*, which is a biased sample of what occurred. Precision measured against them may partly measure agreement with audit site-selection.

**No causal claim is made anywhere.** Outputs are screening hypotheses for human verification. The system flags risk for investigation; it never adjudicates fraud, and it must not be deployed in a configuration that treats a score as a finding — particularly given that works are attributable to named sitting legislators.

**Untested at production scale.** All figures above are unmeasured. The design has not been validated against a live eSAKSHI feed, only against static open datasets.

---

## License

MIT
````

Want me to also produce a mermaid version of the architecture diagram (GitHub renders it natively, and it's editable rather than hand-aligned ASCII), or a condensed one-page version of this README for the repo landing page?
