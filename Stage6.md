# PARAKH — Stage 6 PRD

## Confidence-Gated Decision & Routing Engine

---

## 1. Objective

Convert the pair:

$$
(R(r), C(r))
$$

into a **clear, auditable decision** for each record.

This stage defines the **final system output**:

* INVESTIGATE
* REMEDIATE
* MONITOR
* CLEAR

---

## 2. Core Principle

> **Do not act on data you cannot trust.**

A high-risk signal on low-confidence data is **not a fraud signal**—it is a **data quality problem**.

---

## 3. Scope

### Included

* Decision logic using \(R\) and \(C\)
* Threshold definition
* Routing categories
* Decision explanations
* Output formatting

### Excluded

* Confidence computation (Stage 2)
* Risk computation (Stage 5)
* UI (Stage 7)

---

## 4. Inputs

From Stage 2:

```python
confidence_score: float  # C(r)
```

From Stage 5:

```python
risk_score: float  # R(r)
```

---

## 5. Output

Each record must produce:

```python
{
  "decision": str,  # INVESTIGATE | REMEDIATE | MONITOR | CLEAR
  "confidence": float,
  "risk": float,
  "reason": str
}
```

---

## 6. Decision Logic

---

### 6.1 Thresholds

Define:

```python
theta_C = 0.6   # confidence threshold
theta_R = 0.7   # risk threshold
```

---

### 6.2 Routing Function

$$
\text{route}(r) =
\begin{cases}
\text{INVESTIGATE} & R \ge \theta_R \;\wedge\; C \ge \theta_C \\
\text{REMEDIATE} & C < \theta_C \\
\text{MONITOR} & R \ge \theta_R \;\wedge\; C < \theta_C \\
\text{CLEAR} & \text{otherwise}
\end{cases}
$$

---

### 6.3 Priority Rule (IMPORTANT)

Apply rules in order:

```python
if C < theta_C:
    decision = "REMEDIATE"
elif R >= theta_R and C >= theta_C:
    decision = "INVESTIGATE"
elif R >= theta_R:
    decision = "MONITOR"
else:
    decision = "CLEAR"
```

> Ensures low-confidence always routes to remediation first.

---

## 7. Decision Semantics

---

### INVESTIGATE

* High risk
* High confidence
  → **credible anomaly**

---

### REMEDIATE

* Low confidence
  → **data unreliable**

---

### MONITOR

* High risk but low confidence
  → **potential issue after data fix**

---

### CLEAR

* Low risk
  → **no action required**

---

## 8. Explanation Generation (CRITICAL FOR DEMO)

Each decision must include a human-readable reason.

---

### 8.1 Format

```python
reason = f"""
Record classified as {decision} because:
- Confidence = {C:.2f}
- Risk = {R:.2f}
- Key signals:
  - Cost anomaly: {z_cost}
  - Vendor concentration: {hhi}
  - Burst activity: {burst}
  - Duplicate similarity: {dup}
"""
```

---

### 8.2 Rules

* Always include:

  * top contributing signals
* Keep explanation < 5 lines
* Must be understandable by non-technical user

---

## 9. Implementation Architecture

---

### 9.1 Modules

* `router.py`
* `decision_engine.py`
* `explain.py`

---

### 9.2 Core Class

```python
class Router:
    def __init__(self, theta_r=0.7, theta_c=0.6):
        self.theta_r = theta_r
        self.theta_c = theta_c

    def route(self, risk_scores, confidence_scores):
        return DecisionResult
```

---

### 9.3 Output Object

```python
class DecisionResult:
    decisions: List[str]
    explanations: List[str]
```

---

## 10. Non-Functional Requirements

### Determinism

* Same input → same decision

### Interpretability

* Every decision must be explainable

### Performance

* < 1 second for 50k records

---

## 11. Validation & Testing

---

### 11.1 Rule Testing

Test all cases:

| C    | R    | Expected    |
| ---- | ---- | ----------- |
| High | High | INVESTIGATE |
| Low  | High | REMEDIATE   |
| High | Low  | CLEAR       |
| Low  | Low  | REMEDIATE   |

---

### 11.2 Boundary Testing

* C = theta_C
* R = theta_R

Ensure correct classification

---

### 11.3 Distribution Check

```python
count(decisions)
```

Ensure:

* Not all records in one category

---

## 12. Edge Cases (MANDATORY)

Handle:

* Missing R or C → assign REMEDIATE
* All records low confidence
* All records high risk
* Uniform scores

---

## 13. Acceptance Criteria

Stage complete if:

* [ ] Every record assigned a decision
* [ ] Explanations generated
* [ ] Threshold logic works correctly
* [ ] Edge cases handled safely
* [ ] Output interpretable

---

## 14. Definition of Done

* Full decision pipeline operational
* Ready for final presentation layer (Stage 7)

---

## 15. Key Design Principle

> **Separate uncertainty from suspicion.**

* Confidence = “Can we trust this data?”
* Risk = “Is this behavior unusual?”

The system acts only when **both are high**.

---
