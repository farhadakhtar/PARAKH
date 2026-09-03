# PARAKH — Stage 7 PRD

## Explainability, Interface & Demo Layer

---

## 1. Objective

Transform system outputs into a **clear, interactive, human-understandable interface** that:

* Explains *why* a record was flagged
* Shows **confidence vs risk separation visually**
* Enables **quick inspection and storytelling**

This stage is critical for:

* SIH judging
* usability
* trust

---

## 2. Scope

### Included

* CLI or simple UI (web or notebook)
* Record-level explanations
* Summary dashboards
* Filtering & inspection tools
* Visualization of scores

### Excluded

* Core computation (Stages 1–6)
* Model changes

---

## 3. Inputs

From previous stages:

```python
Corpus
ConfidenceResult
RiskFeatureResult
RiskResult
DecisionResult
```

---

## 4. Outputs

User-facing system that allows:

* Viewing records
* Filtering by decision
* Inspecting explanations
* Visualizing distributions

---

## 5. Interface Options

---

### Option A (Recommended for SIH)

**Streamlit Web App**

Fast, simple, demo-friendly

---

### Option B

**Jupyter Notebook Dashboard**

Simpler but less polished

---

### Option C

**CLI Tool**

Backup/demo fallback

---

## 6. Core Features

---

### 6.1 Record Table View

Display:

| work_id | work_name | district | C | R | decision |
| ------- | --------- | -------- | - | - | -------- |

---

### 6.2 Filters

User must be able to filter by:

* decision type (INVESTIGATE / REMEDIATE / etc.)
* confidence range
* risk range
* district

---

### 6.3 Record Drill-Down (MOST IMPORTANT)

Clicking a record shows:

```text
Work: Road construction in X district

Decision: INVESTIGATE

Confidence: 0.82
Risk: 0.91

Why flagged:
- Cost is 3.2x higher than peer median
- Vendor concentration is high (HHI = 0.76)
- Similar work found within 30 days (duplicate score = 0.88)

Data quality:
- All fields present
- Dates valid
- Amounts consistent
```

---

### 6.4 Confidence vs Risk Plot

Scatter plot:

* X-axis → Confidence
* Y-axis → Risk

Quadrants:

* Top-right → INVESTIGATE
* Bottom-left → CLEAR
* Bottom-right → MONITOR
* Top-left → (rare)

---

### 6.5 Distribution Charts

* Histogram of confidence
* Histogram of risk
* Decision breakdown (bar chart)

---

### 6.6 Summary Metrics

```python
{
  "total_records": N,
  "investigate_count": X,
  "remediate_count": Y,
  "avg_confidence": ...,
  "avg_risk": ...
}
```

---

## 7. Explainability Engine

---

### 7.1 Top Signal Extraction

For each record:

* Identify top contributing signals:

  * highest z_cost
  * high HHI
  * high burst
  * high duplicate score

---

### 7.2 Template

```python
reason = f"""
High risk due to:
- Cost anomaly (z = {z_cost:.2f})
- Vendor concentration (HHI = {hhi:.2f})
- Temporal burst detected

Confidence is {C:.2f}, indicating reliable data.
"""
```

---

### 7.3 Rules

* Max 4 bullet points
* Plain English only
* No math symbols in UI

---

## 8. Implementation Architecture

---

### 8.1 Modules

* `ui_app.py` (Streamlit)
* `visualization.py`
* `explain.py`

---

### 8.2 Example (Streamlit)

```python
import streamlit as st

st.title("PARAKH Dashboard")

df = load_results()

decision_filter = st.selectbox("Decision", df["decision"].unique())
filtered = df[df["decision"] == decision_filter]

st.dataframe(filtered)

selected = st.selectbox("Select Record", filtered["work_id"])

record = df[df["work_id"] == selected].iloc[0]

st.write(record["explanation"])
```

---

## 9. Non-Functional Requirements

### Speed

* UI loads < 2 seconds

### Clarity

* Non-technical user can understand output

### Stability

* No crashes during demo

---

## 10. Demo Flow (CRITICAL)

---

### Step 1

Show dataset overview

---

### Step 2

Show confidence distribution

---

### Step 3

Show risk vs confidence plot

---

### Step 4

Click INVESTIGATE record

---

### Step 5

Explain:

* why it’s risky
* why data is trustworthy

---

### Step 6

Show REMEDIATE case:

* “we don’t trust this data”

---

## 11. Acceptance Criteria

Stage complete if:

* [ ] UI displays all records
* [ ] Filters work correctly
* [ ] Record explanations readable
* [ ] Charts render correctly
* [ ] Demo flow works smoothly

---

## 12. Definition of Done

* Fully working demo interface
* End-to-end system visible
* Ready for SIH presentation

---

## 13. Key Design Principle

> **If you can’t explain it, it doesn’t exist.**

The system is not judged by:

* math
* models

It is judged by:

* clarity
* trust
* explanation

---
