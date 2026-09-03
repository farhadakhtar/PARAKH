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
