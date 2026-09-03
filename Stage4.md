# PARAKH — Stage 4 PRD

## Risk Signal Computation Layer

---

## 1. Objective

Compute **raw anomaly signals** for each record that capture statistically unusual behavior **independent of confidence**.

Each record must produce a feature vector:

$$
X(r) = \{z_{\text{cost}}, \text{HHI}, B, D_{\max}\}
$$

These are **not final risk scores**—they are interpretable signals used in Stage 5.

---

## 2. Scope

### Included

* Cost outlier detection
* Vendor concentration (HHI)
* Temporal burst detection
* Duplicate signal integration
* Feature normalization (pre-aggregation)

### Excluded

* Final risk scoring
* Confidence computation
* Routing decisions

---

## 3. Inputs

From Stage 1:

```python
Corpus.records
```

From Stage 3:

```python
SemanticResult:
  - cluster_id
  - cost_stratum
  - peer_cell
  - duplicate_score (D_max)
```

---

## 4. Outputs

Each record must produce:

```python
{
  "z_cost": float,
  "hhi": float,
  "burst_score": float,
  "duplicate_score": float
}
```

---

## 5. Cost Outlier (Primary Signal)

---

### 5.1 Definition

$$
z_{\text{cost}}(r) =
\frac{\log a_r - \operatorname{med}_{k,s}(\log a)}
{1.4826 \cdot \operatorname{MAD}_{k,s}(\log a)}
$$

Where:

* \(k,s\): peer cell
* \(a_r\): sanction_amount

---

### 5.2 Implementation Steps

1. Group records by peer cell
2. Compute:

   * median(log cost)
   * MAD(log cost)
3. Compute z-score per record

---

### 5.3 Edge Handling

If:

* `MAD < 1e-6` OR
* `cell_size < n_min`

Then:

```python
z_cost = None
```

---

### 5.4 Output Range

* Normal: ~[-3, +3]
* Extreme outliers: > |5|

---

## 6. Vendor Concentration (HHI)

---

### 6.1 Definition

For each cell \((c,k)\):

$$
\text{HHI}_{c,k} =
\sum_{v} \left(\frac{a_v}{\sum a}\right)^2
$$

Where:

* \(a_v\): total awarded to vendor v

---

### 6.2 Implementation

1. Group by:

```python
(constituency/district, cluster_id)
```

2. Aggregate:

* total spend per vendor
* total spend in cell

3. Compute HHI

---

### 6.3 Assignment

Each record inherits:

```python
hhi = HHI(cell)
```

---

### 6.4 Edge Cases

* Single vendor → HHI = 1
* Missing vendor → exclude or assign neutral value

---

## 7. Temporal Burst Detection

---

### 7.1 Definition

$$
B(d,m) =
\log \frac{n_{d,m} + \alpha}
{\lambda_d(m) + \alpha}
$$

Where:

* \(n_{d,m}\): records in district d, month m
* \(\lambda_d(m)\): seasonal baseline

---

### 7.2 Baseline Estimation

For each district:

* Compute monthly averages across years
* Use **month-specific baseline** (March vs March)

---

### 7.3 Parameters

```python
alpha = 1.0
```

---

### 7.4 Assignment

Each record gets:

```python
burst_score = B(district, month(date_approval))
```

---

### 7.5 Interpretation

* > 0 → higher-than-expected activity
* < 0 → lower-than-expected

---

## 8. Duplicate Signal

---

Already computed in Stage 3:

```python
duplicate_score = D_max
```

---

### 8.1 Validation

Ensure:

* identical names → high score (~0.9+)
* unrelated → near 0

---

## 9. Feature Normalization

---

Before Stage 5, normalize all signals:

### 9.1 Method

```python
percentile_rank(feature)
```

---

### 9.2 Rules

* Apply globally (NOT mixed scopes)
* Ignore None values
* Clip extreme values

---

### 9.3 Output

```python
{
  "z_cost_norm": [0,1],
  "hhi_norm": [0,1],
  "burst_norm": [0,1],
  "dup_norm": [0,1]
}
```

---

## 10. Implementation Architecture

---

### 10.1 Modules

* `cost_outlier.py`
* `hhi.py`
* `burst.py`
* `normalization.py`

---

### 10.2 Core Class

```python
class RiskSignals:
    def compute(self, corpus, semantic_result):
        return RiskFeatureResult
```

---

### 10.3 Output Object

```python
class RiskFeatureResult:
    raw_features: Dict
    normalized_features: Dict
```

---

## 11. Non-Functional Requirements

### Performance

* 50k records processed in < 10 seconds

### Stability

* No NaN/inf values

### Determinism

* Same input → same output

---

## 12. Validation & Testing

---

### 12.1 Cost Outlier Checks

* median-cost records → z ≈ 0
* extreme cost → large |z|

---

### 12.2 HHI Checks

* uniform vendors → low HHI
* dominant vendor → high HHI

---

### 12.3 Burst Checks

* uniform distribution → near 0
* spikes → positive

---

### 12.4 Duplicate Checks

* identical entries → high score
* random entries → low

---

## 13. Edge Cases (MANDATORY)

Handle:

* Missing cost values
* Single-record peer cell
* No vendor info
* All records in same month
* Zero variance in cost

---

## 14. Acceptance Criteria

Stage complete if:

* [ ] All signals computed per record
* [ ] Invalid cases handled safely
* [ ] Normalized features in [0,1]
* [ ] No crashes on edge cases
* [ ] Works on synthetic dataset

---

## 15. Definition of Done

* All anomaly signals computed
* Ready for aggregation (Stage 5)

---

## 16. Key Design Principle

> **Signals, not decisions.**

This stage does not decide fraud.
It produces **independent, interpretable indicators** of unusual behavior.

---
