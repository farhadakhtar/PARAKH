# PARAKH: An Evidentiary-Confidence Layer for Public Fund Anomaly Detection

**A fraud-analytics engine that scores the trustworthiness of each record before it scores the risk of each transaction — because on self-certified government data, a confident anomaly detector mostly learns who files paperwork late.**
`https://github.com/farhadakhtar/PARAKH`

---

## Abstract

We present PARAKH, an analytics engine for public fund disbursement schemes that separates *evidentiary confidence* from *fraud risk* and refuses to emit a high-confidence flag on a low-confidence record. The core insight is that in schemes like MPLADS the binding constraint is not the absence of anomaly detection but the unreliability of the source records: the CAG's 2010 performance audit found handing-over of assets unrecorded for 14,828 of 15,049 sampled works (98.53%), and work completion on the current eSAKSHI portal is self-certified by the implementing agency being monitored. An unsupervised detector trained on such data maximizes its objective by learning reporting artifacts — district-level data-entry latency, retrospective backfill — which correlate with administrative capacity, not corruption. PARAKH computes a per-record confidence $C \in [0,1]$ from field completeness, temporal coherence and cross-source reconciliation, computes a raw risk $R$ from peer-conditioned cost outliers, vendor-graph concentration and temporal bursts, then routes on the *pair* $(R, C)$: high-$R$/high-$C$ to investigation, low-$C$ to data remediation. It is a Rust core with Python bindings and a DuckDB-backed store.

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

## Theoretical Foundation

### 1) Evidentiary confidence

For record $r$ with field set $F$, confidence is a product of three independently-computable factors:

$$
C(r) = C_{\text{comp}}(r)^{w_1} \cdot C_{\text{temp}}(r)^{w_2} \cdot C_{\text{recon}}(r)^{w_3}, \qquad \sum_i w_i = 1
$$

**Completeness** $C_{\text{comp}}$ is the mass of non-null, non-placeholder fields weighted by each field's diagnostic value $v_f$ (a field that is null for 90% of records carries little information about the 10%):

$$
C_{\text{comp}}(r) = \frac{\sum_{f \in F} v_f \cdot \mathbb{1}[r_f \text{ valid}]}{\sum_{f \in F} v_f}, \qquad v_f = 1 - H(f)
$$

where $H(f)$ is the normalized entropy of field $f$'s null-pattern across the corpus.

**Temporal coherence** $C_{\text{temp}}$ penalizes violations of the scheme's own causal ordering — proposal receipt $\le$ administrative approval $\le$ completion — and impossible dates:

$$
C_{\text{temp}}(r) = \prod_{(a,b) \in \mathcal{O}} \sigma\!\left(\frac{t_b - t_a}{\tau}\right)
$$

with $\mathcal{O}$ the mandated ordering pairs, $\sigma$ the logistic function, and $\tau$ a tolerance in days. A record with `date_of_administrative_approval` preceding 1993 scores $C_{\text{temp}} \to 0$ and never reaches the fraud queue.

**Reconciliation** $C_{\text{recon}}$ measures agreement across independent sources for the same entity — work-level sanction totals against MoSPI fund-release aggregates, IA-reported completion against district totals:

$$
C_{\text{recon}}(r) = \exp\!\left(-\lambda \left| \frac{x_r^{(1)} - x_r^{(2)}}{\max(x_r^{(1)}, x_r^{(2)})} \right|\right)
$$

### 2) Peer-conditioned cost outlier

Raw cost comparison across districts is confounded by genuine input-price variation. We compare each work against its *peer cell* — works in the same semantic cluster $k$ (from §4) and the same cost stratum $s$ — using a robust score on log cost:

$$
z_{\text{cost}}(r) = \frac{\log a_r - \operatorname{med}_{k,s}(\log a)}{1.4826 \cdot \operatorname{MAD}_{k,s}(\log a)}
$$

The constant $1.4826$ makes MAD a consistent estimator of $\sigma$ under normality. Median and MAD are used rather than mean and standard deviation because the contamination we are searching for is in the same statistic we would otherwise estimate from — a 20% inflated cell breaks a mean and leaves a median intact up to a 50% breakdown point.

### 3) Entity resolution

Vendor and implementing-agency names arrive as unnormalized free text with no persistent identifier. We use Fellegi–Sunter probabilistic linkage over blocked candidate pairs. For record pair $(a,b)$ with agreement vector $\gamma$:

$$
\log \Lambda(a,b) = \sum_{j} \log \frac{m_j(\gamma_j)}{u_j(\gamma_j)}
$$

where $m_j = P(\gamma_j \mid \text{match})$ and $u_j = P(\gamma_j \mid \text{non-match})$, estimated by EM on the unlabelled candidate set. Blocking is on normalized-name trigram and district to keep candidate generation at $O(N \cdot b)$ rather than $O(N^2)$; at $N \approx 10^5$ works the unblocked comparison space is $\approx 5 \times 10^9$ pairs and blocking is not optional.

Matched records collapse into entity nodes. **Where the public data ends:** $\gamma$ can only include name, district and IA co-occurrence. Bank account, PAN, GSTIN and director overlap — the features that make ARACHNE work — are unavailable, so linkage precision is bounded well below what an ownership-enriched system achieves.

### 4) Work-type clustering and duplicate detection

Free-text `work_name` is embedded and clustered to produce the peer cells of §2 and to surface near-duplicate works (the same asset sanctioned twice). For works $i,j$ with embeddings $e_i, e_j$, duplicate candidacy requires semantic and geographic and temporal proximity jointly:

$$
D(i,j) = \cos(e_i, e_j) \cdot \mathbb{1}[d_i = d_j] \cdot \exp\!\left(-\frac{|t_i - t_j|}{\tau_d}\right)
$$

Conjunction rather than a weighted sum is deliberate: two culverts with identical descriptions in different districts are not a duplicate, and a soft weighting lets high semantic similarity overwhelm the geographic mismatch.

### 5) Vendor concentration

Award concentration per (constituency, work-type) cell, via Herfindahl–Hirschman index over resolved entities:

$$
\text{HHI}_{c,k} = \sum_{v \in V_{c,k}} \left(\frac{a_v}{\sum_{u} a_u}\right)^2
$$

$\text{HHI} \to 1$ indicates a single vendor capturing a cell. This is a *screening* statistic, not evidence: legitimate concentration arises from thin local supplier markets, and the report must say so on every flag.

### 6) Temporal burst detection

Sanction arrivals per (district, month) modelled as a Poisson process with rate $\lambda$ estimated from the district's trailing baseline. Burst score is the log rate ratio:

$$
B(d,m) = \log \frac{n_{d,m} + \alpha}{\lambda_d \Delta + \alpha}
$$

with $\alpha$ a Laplace smoothing term. Fiscal-year-end March bursts are a documented, near-universal pattern in Indian public expenditure and are therefore *expected*; the burst score is computed against a seasonally-adjusted baseline so that March is compared to March.

### 7) Confidence-gated routing — the decision rule

Raw risk aggregates the components with weights $\beta$:

$$
R(r) = \sigma\!\left(\beta_0 + \beta_1 z_{\text{cost}} + \beta_2 \text{HHI} + \beta_3 B + \beta_4 D_{\max}\right)
$$

The emitted disposition is a function of the pair, not the product:

$$
\text{route}(r) = \begin{cases}
\text{INVESTIGATE} & R \ge \theta_R \;\wedge\; C \ge \theta_C \\
\text{REMEDIATE} & C < \theta_C \\
\text{MONITOR} & R \ge \theta_R \;\wedge\; C < \theta_C \text{ (after remediation)} \\
\text{CLEAR} & \text{otherwise}
\end{cases}
$$

A high-risk, low-confidence record does not become a fraud allegation. It becomes a documentation request — which is both the epistemically honest response and, per the CAG findings, the one that addresses the actual dominant failure.

### 8) Artifact-invariance test

The claim that PARAKH is not learning reporting artifacts is falsifiable and must be tested, not asserted. Let $A$ be a set of administrative-capacity proxies (district data-entry latency, backfill burstiness, null-rate). We require the risk score to be approximately conditionally independent of $A$ given the substantive features $X$:

$$
\text{AUC}\big(R \mid A\big) - 0.5 < \epsilon
$$

If risk score alone predicts administrative capacity above chance by more than $\epsilon$, the model is an artifact detector and ships disabled. **TODO: measure** — this test is specified and not yet run.

## Implementation

### Core (`parakh-core`, Rust)

Columnar ingestion and scoring over Arrow record batches. Confidence and risk components are computed as vectorized passes; the per-record cost is dominated by the §3 blocking pass, not the arithmetic. One non-obvious decision: confidence factors are stored as `f32` log-space accumulators rather than products, because $C_{\text{temp}}$ is a product over ordering constraints and underflows to zero on records with many violations — which is semantically correct but destroys the ability to rank *among* bad records for the remediation queue.

### Entity graph (`parakh-graph`)

`petgraph` over resolved entity nodes with award edges. HHI and bipartite projection are computed per cell. The graph is rebuilt rather than incrementally maintained — at MPLADS scale (~$10^5$–$10^6$ works) a full rebuild is cheaper than maintaining incremental correctness, and correctness matters more than latency for an audit tool that runs nightly.

### Semantic layer (`parakh-nlp`, Python)

Multilingual sentence embeddings over `work_name`, HDBSCAN for cluster discovery. Indian scheme text is code-mixed (English work descriptions with transliterated Hindi/regional terms), so a monolingual English encoder degrades badly. **TODO: measure** cluster purity against a hand-labelled sample.

### Case store (`parakh-store`, DuckDB)

Every flag persists with its full evidence chain: which rule fired, the peer cell it was compared against, the cell's median and MAD, the contributing records, and $C$ with its three factors. A flag with no reproducible derivation is not admissible in an audit context and is treated as a bug.

### API surface

```rust
let corpus = Corpus::from_parquet("mplads_17ls.parquet")?;
let conf   = ConfidenceModel::default().score(&corpus)?;   // §1
let graph  = EntityGraph::resolve(&corpus, LinkageConfig::default())?; // §3
let risk   = RiskModel::new(&graph).score(&corpus)?;       // §2,5,6
let queue  = Router::new(theta_r, theta_c).route(&risk, &conf)?; // §7
```

## Results

Not yet measured. Per this repository's standard, no figure appears here until it is produced by a stated command.

| Metric | Value | Notes |
|---|---|---|
| Ingest throughput (works/s) | **TODO: measure** | 17th LS dataset, 73,305 rows |
| Entity resolution precision | **TODO: measure** | vs. hand-labelled sample, 18th LS vendor names |
| Entity resolution recall | **TODO: measure** | same sample |
| Peer-cell cluster purity | **TODO: measure** | hand-labelled work-type sample |
| Artifact-invariance AUC (§8) | **TODO: measure** | must be < 0.5 + ε to ship enabled |
| Flag precision @ CAG-confirmed | **TODO: measure** | proxy labels from CAG Report 31/2010 annexures |
| Full-corpus scoring latency | **TODO: measure** | single node, core-pinned |

Reproduction command once implemented: `cargo bench --bench full_corpus` and `python -m parakh.eval --split holdout`.

**The number that will matter most is the artifact-invariance AUC**, not the flag precision. A high flag precision against CAG proxy labels is achievable by a model that has learned which districts get audited, and the two are separable only by §8.

**Data substrate.** Work-level records from Dataful/data.gov.in sourced from MoSPI: 17th LS (dataset 18533, ~73,305 rows, 17 columns), 16th LS (18534), 15th LS (18535), sitting Rajya Sabha (18539, ~13,638 rows). Vendor names exist **only** for the 18th LS (dataset 22565). Bank account, PAN, GSTIN and payment-level records are not public in any dataset.

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

## Requirements

- Rust 1.75+ (MSRV), stable channel. No nightly features.
- Python 3.10+ for the semantic layer.
- x86_64 or aarch64. No CPU-feature requirements beyond baseline; the hot paths are memory-bound, not SIMD-bound.
- DuckDB 0.10+.
- ~8 GB RAM for full-corpus scoring at $10^5$-work scale (estimated from Arrow batch sizing, **not measured**).

## Limitations

**No beneficial-ownership data, therefore no genuine collusion detection.** This is the central limitation and it is not fixable in the open-data setting. PARAKH can detect that one *named* vendor won repeatedly; it cannot detect that three differently-named vendors share a director, an address or a bank account — which is how collusion is actually structured, and precisely what ARACHNE and GRAS were built to catch. Any claim that this system detects vendor collusion on public MPLADS data is false. Closing this requires MCA21/GSTN linkage that only the Ministry can authorize.

**Confidence gating trades recall for defensibility.** A sophisticated actor who files complete, temporally-coherent, internally-reconciled paperwork while stealing will pass §1 with a high $C$ and be evaluated purely on §2/§5/§6. Good documentation is not evidence of good faith, and the gate is one-directional: it suppresses false confidence, it does not detect competent fraud.

**Peer cells require enough peers.** Districts or work types with few comparable works produce unstable MAD estimates and unreliable $z_{\text{cost}}$. The current design suppresses scoring below a minimum cell size rather than reporting a noisy score, which systematically under-covers rare work types — where unusual costs are arguably most likely.

**HHI concentration is confounded by thin markets.** A single vendor in a remote district may be the only qualified contractor. Every concentration flag is a hypothesis requiring local knowledge to adjudicate, and the system has none.

**Satellite verification is not implemented and should not be promised.** The obvious answer to self-certification is independent imagery verification of asset existence. Free Sentinel-2 is 10 m resolution; typical MPLADS works are culverts, borewells and single community halls, which are below that resolution. Verification is feasible only for larger footprints (buildings, ponds, road stretches) and reliably only with commercial sub-metre imagery. Building it as a general capability on free imagery would be an overclaim.

**Proxy labels are weak and possibly circular.** CAG-derived labels reflect what was *audited and found*, which is a biased sample of what occurred. Precision measured against them may partly measure agreement with audit site-selection.

**No causal claim is made anywhere.** Outputs are screening hypotheses for human verification. The system flags risk for investigation; it never adjudicates fraud, and it must not be deployed in a configuration that treats a score as a finding — particularly given that works are attributable to named sitting legislators.

**Untested at production scale.** All figures above are unmeasured. The design has not been validated against a live eSAKSHI feed, only against static open datasets.

## License

MIT

*Invented by [Teerth Sharma](https://teerthsharma.vercel.app)*
