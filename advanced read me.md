# PARAKH

### An Evidentiary-Confidence Layer for Public Fund Anomaly Detection

**A fraud analytics engine that scores the *trustworthiness of each record* before scoring the *risk of each transaction*.**

> Because on self-certified government data, a confident anomaly detector mostly learns who files paperwork late.

---

## Abstract

PARAKH is an analytics engine for public fund disbursement schemes that explicitly separates **evidentiary confidence** from **fraud risk**, and enforces that no high-risk flag is emitted on low-confidence data.

In schemes like MPLADS, the dominant failure mode is not the absence of anomaly detection—but the **unreliability of source records**. The CAG’s 2010 audit found missing asset handover records in **98.53%** of sampled works, while systems like eSAKSHI rely on **self-certification by the implementing agency**.

Traditional unsupervised anomaly detection (Isolation Forests, autoencoders, etc.) fails here—not because of weak models, but because it learns **administrative artifacts** (data-entry delays, backfills, missing fields) instead of fraud.

PARAKH solves this by:

* Computing a **confidence score** $C \in [0,1]$ per record
* Computing a **risk score** $R \in [0,1]$
* Routing decisions based on the **pair $(R, C)$**

**Key principle:**

> A record that cannot be evidenced is never a fraud allegation.

---

## Core Idea

PARAKH introduces a structural separation:

| Layer              | Purpose                                      |
| ------------------ | -------------------------------------------- |
| **Confidence (C)** | Can this record be trusted at all?           |
| **Risk (R)**       | Is this record suspicious relative to peers? |
| **Router**         | Decides action using both                    |

This prevents a critical failure mode:

> High-risk scores on low-quality data → false accusations driven by reporting artifacts.

---

## Architecture Overview

The system consists of two independent branches:

* **Confidence Branch** → produces $C$
* **Risk Branch** → produces $R$

These converge in a **routing layer**.

### Component Flow

1. Data ingestion (Arrow batches)
2. Parallel computation:

   * Confidence metrics (§1)
   * Risk metrics (§2–§7)
3. Artifact-invariance validation (§9)
4. Routing based on $(R, C)$

---

## Theoretical Foundation

### 1. Evidentiary Confidence

Confidence is defined as a **weighted geometric mean**:

$$
C(r) = C_{comp}^{w_1} \cdot C_{temp}^{w_2} \cdot C_{recon}^{w_3}
$$

This ensures:

* No factor can compensate for another
* Any failure → $C \to 0$

---

### 1.1 Completeness

Measures presence and informativeness of fields:

* Weighted by entropy:

  * Penalizes constant fields
  * Penalizes inconsistent fill patterns

---

### 1.2 Temporal Coherence

Uses asymmetric penalty:

* Valid ordering → no penalty
* Violations → exponential decay

Fix applied:

* Removed logistic penalty (which incorrectly penalized valid cases)

---

### 1.3 Reconciliation

Measures agreement across independent sources:

$$
C_{recon} = \exp\left(-\lambda \cdot \frac{|x_1 - x_2|}{|x_1| + |x_2| + \epsilon}\right)
$$

* Symmetric normalization
* Well-defined at zero
* Bounded in $[e^{-\lambda}, 1]$

---

## Risk Modeling

### Components

| Component            | Description                    |
| -------------------- | ------------------------------ |
| Cost Outlier         | Peer-normalized robust z-score |
| Duplicate Detection  | Semantic + temporal similarity |
| Vendor Concentration | HHI per cell                   |
| Temporal Burst       | Seasonal Poisson deviation     |

---

### Key Design Choice: Rank Aggregation

Instead of learned weights:

$$
R(r) = \text{mean percentile rank of components}
$$

Reason:

* No labels → weights are unidentifiable
* Avoids arbitrary calibration
* Robust to scale differences

Tradeoff:

* Loses magnitude differences in extreme values

---

## Routing Logic (Critical)

Routing is based on $(R, C)$:

| Condition      | Action      |
| -------------- | ----------- |
| High C, High R | INVESTIGATE |
| Low C          | REMEDIATE   |
| High C, Low R  | CLEAR       |

**Important rule:**

> REMEDIATE overrides everything.

This resolves a major flaw in earlier versions:

* Overlapping conditions between REMEDIATE and MONITOR fixed

---

## Artifact-Invariance Test

The system explicitly tests whether it is learning **administrative artifacts**.

Requirement:

$$
\max \text{AUC}(artifact \leftarrow R) \le 0.5 + \epsilon
$$

If violated:

* Risk model is **disabled**
* Only confidence + remediation continues

This is the most important validation in the system.

---

## Implementation

### Stack

* **Core:** Rust (high-performance scoring)
* **Graph:** Entity linkage + vendor relationships
* **NLP:** Python embeddings + clustering
* **Storage:** DuckDB (audit-friendly persistence)

---

### Key Engineering Decisions

* Log-space computation for stability
* Full graph rebuild instead of incremental updates
* Evidence-chain persistence for every flag

---

## Calibration (Non-Optional)

The system is unusable without calibration.

Parameters include:

* Confidence weights ($w_1, w_2, w_3$)
* Temporal decay ($\kappa$)
* Reconciliation penalty ($\lambda$)
* Routing thresholds ($\theta_R, \theta_C$)

> Any uncalibrated deployment is invalid.

---

## Results

Not yet measured.

All metrics are explicitly marked **TODO** until reproducible benchmarks exist.

---

## Limitations (Non-Negotiable)

### 1. No Beneficial Ownership Data

* Cannot detect real collusion
* Only surface-level vendor patterns

### 2. Confidence Gating Reduces Recall

* Fraud with good documentation may pass

### 3. Rank Aggregation Tradeoff

* Loses magnitude sensitivity

### 4. Sparse Peer Cells

* Rare work types under-covered

### 5. HHI Misinterpretation Risk

* High concentration ≠ fraud

### 6. No Satellite Verification

* Not feasible with free imagery at required resolution

### 7. Weak Proxy Labels

* CAG data is biased and incomplete

### 8. No Causal Claims

* System produces **hypotheses**, not conclusions

### 9. Not Production Tested

* No live validation yet

---

## Quick Start

```bash
cargo add parakh-core   # TODO
pip install parakh      # TODO
```

```python
from parakh import Corpus, Pipeline

corpus = Corpus.from_csv("mplads_17ls.csv")
result = Pipeline.default().run(corpus)

print(result.queue("REMEDIATE").head())

for flag in result.queue("INVESTIGATE"):
    print(flag.explain())
```

---

## Requirements

* Rust 1.75+
* Python 3.10+
* DuckDB 0.10+
* ~8GB RAM (estimated)

---

## License

MIT
