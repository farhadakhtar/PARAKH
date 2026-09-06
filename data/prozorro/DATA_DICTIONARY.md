# Data Dictionary - Ukraine Prozorro Procurement Dataset

## Overview

This document describes all columns across the dataset files:
- **4 tenders files** (tenders_2022.csv - tenders_2025.csv): 55 columns each
- **4 bids files** (bids_2022.csv - bids_2025.csv): 8 columns each
- **3 reference tables** (buyers.csv, suppliers.csv, bidders.csv): 16, 6, 7 columns respectively

---

## Table: tenders_YYYY.csv

Main tender information with **55 features**.

### Identification (2 columns)

| Column | Type | Description | Example | Missing % |
|--------|------|-------------|---------|-----------|
| tender_id | string | Unique tender identifier | "abc123def456..." | 0% |
| ocid | string | Open Contracting ID (OCDS standard) | "ocds-be6bcu-abc123..." | 0% |

### Temporal Features - Raw Dates (6 columns)

| Column | Type | Description | Example | Missing % |
|--------|------|-------------|---------|-----------|
| published_date | string (ISO 8601) | Tender publication date/time | "2023-05-15T10:30:00+03:00" | 0% |
| tender_start_date | string (ISO 8601) | Tender period start date | "2023-05-20T00:00:00+03:00" | 90% |
| tender_end_date | string (ISO 8601) | Tender period end date | "2023-05-30T00:00:00+03:00" | 90% |
| award_period_start | string (ISO 8601) | Award period start | "2023-06-01T00:00:00+03:00" | 93% |
| award_period_end | string (ISO 8601) | Award period end | "2023-06-15T00:00:00+03:00" | 93% |
| award_date | string (ISO 8601) | Award decision date | "2023-06-10T14:00:00+03:00" | 3% |

**Note:** High missing % for tender dates is normal - not all procurement types require these fields.

### Temporal Features - Normalized (7 columns)

Derived from `published_date` for easier analysis.

| Column | Type | Description | Example | Missing % |
|--------|------|-------------|---------|-----------|
| year | integer | Year (4 digits) | 2023 | 0% |
| month | integer | Month (1-12) | 5 | 0% |
| quarter | integer | Quarter (1-4) | 2 | 0% |
| day_of_week | integer | Day of week (0=Monday, 6=Sunday) | 0 | 0% |
| is_q4 | integer | Flag: 1 if Q4 (Oct-Dec), 0 otherwise | 0 | 0% |
| is_december | integer | Flag: 1 if December, 0 otherwise | 0 | 0% |
| is_weekend | integer | Flag: 1 if Saturday/Sunday, 0 otherwise | 0 | 0% |

**Use for:** Seasonal analysis, Q4 spending spikes detection, working day patterns.

### Buyer Information (2 columns)

| Column | Type | Description | Example | Missing % |
|--------|------|-------------|---------|-----------|
| buyer_id | integer | Buyer organization ID (EDRPOU code) | 12345678 | 0% |
| procuring_entity_id | integer | Procuring entity ID (usually same as buyer_id) | 12345678 | 0% |

**Note:** Buyer name and region are in `buyers.csv` (join on buyer_id).

**EDRPOU:** Ukrainian tax identification number (8 digits).

### Geographic (2 columns)

| Column | Type | Description | Example | Missing % |
|--------|------|-------------|---------|-----------|
| locality | string | City or town name | "Kyiv" | 7% |
| postal_code | string | Postal/ZIP code | "01001" | 7% |

**Note:** Region information available in `buyers.csv`.

### Supplier (1 column)

| Column | Type | Description | Example | Missing % |
|--------|------|-------------|---------|-----------|
| supplier_id | string | Winning supplier ID (EDRPOU) | "12345678" | 3% |

**Note:** Supplier details (name, region, statistics) in `suppliers.csv` to avoid data leakage in ML models.

### Tender Characteristics (3 columns)

| Column | Type | Description | Possible Values | Missing % |
|--------|------|-------------|-----------------|-----------|
| procurement_method | string | Procurement procedure type | limited, open, selective | 0% |
| main_procurement_category | string | Category of goods/services | goods, services, works | 0% |
| award_criteria | string | Winner selection criteria | lowestCost, bestProposal, bestValueToGovernment | 0% |

**Procurement methods:**
- `limited`: Simplified procurement (restricted competition)
- `open`: Open competitive bidding
- `selective`: Pre-qualified suppliers only

### CPV Classification (3 columns)

CPV = Common Procurement Vocabulary (EU standard classification system).

| Column | Type | Description | Example | Missing % |
|--------|------|-------------|---------|-----------|
| main_cpv_code | string | Full CPV code (8 digits + check digit) | "09130000-9" | 0% |
| main_cpv_2_digit | integer | 2-digit category code | 9 | 0% |
| main_cpv_4_digit | integer | 4-digit subcategory code | 913 | 0% |

**Example categories:**
- 09: Petroleum products, fuel
- 30: Office equipment
- 33: Medical equipment
- 45: Construction work

### Price Features (5 columns)

| Column | Type | Description | Example | Missing % |
|--------|------|-------------|---------|-----------|
| tender_value | float | Expected tender value (UAH) | 50000.0 | 3% |
| award_value | float | Actual awarded contract value (UAH) | 48000.0 | 3% |
| price_change_amount | float | Difference: tender_value - award_value (UAH) | 2000.0 | 6% |
| price_change_pct | float | Discount percentage | 4.0 | 0% |
| currency | string | Currency code (ISO 4217) | "UAH" | 0% |

**Important notes:**
- Negative `price_change_pct` means price **increased** (red flag!)
- Missing award_value occurs when tender is unsuccessful/cancelled
- Currency is 99.9% UAH (Ukrainian Hryvnia)

**USD conversion:** Approximately 1 USD = 40 UAH (varies by year).

### Competition Metrics (6 columns)

| Column | Type | Description | Example | Missing % |
|--------|------|-------------|---------|-----------|
| number_of_items | integer | Number of line items in tender | 5 | 0% |
| number_of_tenderers | integer | Number of organizations that submitted bids | 3 | 0% |
| number_of_bids | integer | Total number of bids received | 3 | 0% |
| number_of_awards | integer | Number of awards issued | 1 | 0% |
| number_of_contracts | integer | Number of contracts signed | 1 | 0% |
| number_of_documents | integer | Number of attached documents | 10 | 0% |

**Analysis tips:**
- `number_of_tenderers = 0` → No participants (tender failed)
- `number_of_tenderers = 1` → Single bidder (potential anomaly)
- `number_of_bids > number_of_tenderers` → Some organizations submitted multiple bids

### Critical Flags (6 columns)

Binary flags for identifying suspicious patterns and critical tender characteristics.

| Column | Type | Description | Values | Missing % |
|--------|------|-------------|--------|-----------|
| is_single_bidder | integer | Only one bidder participated | 1=yes, 0=no | 0% |
| is_cross_region | integer | Supplier from different region than buyer | 1=yes, 0=no | 0% |
| has_enquiries | integer | Tender had clarification questions | 1=yes, 0=no | 0% |
| is_buyer_masked | integer | Buyer data is masked/redacted | 1=yes, 0=no | 0% |
| is_supplier_masked | integer | Supplier data is masked/redacted | 1=yes, 0=no | 0% |
| is_competitive | integer | Competitive tender (>1 bidder, not single-source) | 1=yes, 0=no | 0% |

**Anomaly patterns:**
- `is_single_bidder=1` + high value → Potential lack of competition
- `is_buyer_masked=1` OR `is_supplier_masked=1` → Military/sensitive procurement
- `is_cross_region=1` + `is_single_bidder=1` → Unusual (investigate)
- `is_competitive=0` + high value → Non-competitive procurement (higher risk)

**Awards Aggregated Features (12 columns)**

These columns are **pre-aggregated** from the original awards data and included directly in the tenders table:

| Column | Type | Description |
|--------|------|-------------|
| has_multiple_awards | integer | 1 if tender has >1 award, 0 otherwise |
| award_value_total | float | Sum of all award values (UAH) |
| award_value_max | float | Maximum award value (UAH) |
| award_value_min | float | Minimum award value (UAH) |
| award_value_mean | float | Mean award value (UAH) |
| award_value_std | float | Standard deviation of award values |
| award_concentration | float | Concentration ratio (max/total), range 0-1 |
| discount_percentage_avg | float | Average discount % across all awards |
| discount_percentage_max | float | Maximum discount % |
| has_unsuccessful_awards | integer | 1 if any award has unsuccessful status, 0 otherwise |
| has_cancelled_awards | integer | 1 if any award has cancelled status, 0 otherwise |
| active_awards_count | integer | Count of awards with active status |

**Note:** Awards data has been aggregated at the tender level to avoid creating separate files. This saves ~1.7 GB and simplifies analysis.

**Use for:** Multi-award tender analysis, award concentration detection, competitive dynamics.

---

## Table: bids_YYYY.csv

Bidding information for all participants (not just winners).

### Schema (8 columns)

| Column | Type | Description | Missing % |
|--------|------|-------------|-----------|
| tender_id | string | Links to tenders table (FK) | 0% |
| bid_id | string | Unique bid identifier | 0% |
| bidder_id | string | Bidder EDRPOU code (can be linked to suppliers) | 0.5% |
| bid_date | string (ISO 8601) | Bid submission date | 5% |
| bid_status | string | Bid status (valid, withdrawn, pending, disqualified) | 0% |
| bid_amount | float | Bid value in UAH | 0.6% |
| is_winner | integer | 1 if this bidder won the tender, 0 otherwise | 0% |
| is_bidder_masked | integer | Data quality flag (1=masked) | 0% |

**Key features:**
- `bid_amount`: Actual bid value submitted by participant
- `is_winner`: Flag indicating if bidder won (matched by EDRPOU with award suppliers)
- `bidder_id`: Real EDRPOU code — can be joined with `suppliers.supplier_id`

**Use for:** Competition analysis, bidder behavior patterns, winner prediction, price analysis.

---

## Table: suppliers.csv

Aggregated supplier statistics across all years.

### Schema (6 columns)

| Column | Type | Description |
|--------|------|-------------|
| supplier_id | string | Supplier ID (EDRPOU) - PRIMARY KEY |
| supplier_name | string | Organization name |
| supplier_region | string | Region (84% missing - data quality issue) |
| total_awards | integer | Total number of won tenders (2022-2025) |
| total_value | float | Total contract value in UAH (2022-2025) |
| is_masked | integer | Data quality flag (1=masked) |

**Sorted by:** `total_awards` descending (most active suppliers first).

**Join with tenders:**
```python
merged = tenders.merge(suppliers, on='supplier_id', how='left', suffixes=('', '_supplier'))
```

**Top suppliers:**
Top 10 suppliers account for significant market share (potential monopolization).

---

## Table: buyers.csv

Buyer (procuring organization) statistics and advanced metrics.

### Schema (16 columns)

**Base columns (5):**

| Column | Type | Description |
|--------|------|-------------|
| buyer_id | integer | Buyer ID (EDRPOU) - PRIMARY KEY |
| buyer_name | string | Organization name |
| buyer_region | string | Region |
| total_tenders | integer | Total tenders created (2022-2025) |
| total_value | float | Total procurement value in UAH (2022-2025) |

**Supply chain diversity (3 columns):**

| Column | Type | Description |
|--------|------|-------------|
| unique_suppliers | integer | Number of unique suppliers used in all tenders |
| supplier_diversity_index | float | Diversity ratio: unique_suppliers / total_tenders (0-1) |
| avg_tenderers_per_tender | float | Average number of bidders per tender (competition level) |

**Competition metrics (3 columns):**

| Column | Type | Description |
|--------|------|-------------|
| competitive_tenders_count | integer | Count of tenders with >1 bid |
| competitive_rate | float | % of competitive tenders (0-1) |
| single_bidder_rate | float | % of tenders with single bidder (0-1) |

**Financial metrics (5 columns):**

| Column | Type | Description |
|--------|------|-------------|
| avg_discount_pct | float | Average savings percentage across all tenders |
| total_savings | float | Total savings in UAH (sum of price_change_amount) |
| avg_tender_value | float | Average tender value in UAH |
| first_tender_date | string | Date of first procurement (ISO 8601) |
| last_tender_date | string | Date of last procurement (ISO 8601) |

**Sorted by:** `total_tenders` descending.

**Join with tenders:**
```python
merged = tenders.merge(buyers, on='buyer_id', how='left', suffixes=('', '_buyer'))
```

**Top buyers:** Mostly military units and large municipalities.

---

## Table: bidders.csv

Bidder participation statistics across all years (72K unique bidders).

**Note:** `bidder_id` is EDRPOU code — can be linked to `suppliers.supplier_id` for win statistics.

### Schema (7 columns)

| Column | Type | Description |
|--------|------|-------------|
| bidder_id | string | Bidder EDRPOU code (PRIMARY KEY) |
| total_bids | integer | Total number of bids submitted (2022-2025) |
| unique_tenders | integer | Number of unique tenders participated in |
| unique_buyers | integer | Number of unique buyers interacted with |
| first_bid_date | string (ISO 8601) | Date of first bid |
| last_bid_date | string (ISO 8601) | Date of last bid |
| is_masked | integer | Data quality flag (1=masked) |

**Sorted by:** `total_bids` descending.

**Join with bids:**
```python
bids = pd.read_csv('bids_2023.csv')
bidders = pd.read_csv('bidders.csv')

merged = bids.merge(bidders, on='bidder_id', how='left')
print(merged[['bid_id', 'bidder_id', 'total_bids']])
```

**Join with suppliers (for win statistics):**
```python
suppliers = pd.read_csv('suppliers.csv')
bidders_with_wins = bidders.merge(suppliers, left_on='bidder_id', right_on='supplier_id', how='left')
```

**Use for:** Identifying repeat bidders, bidder diversity analysis, competition patterns.

---

## Data Types Reference

| Type | Description | Example |
|------|-------------|---------|
| **string** | Text values | "Kyiv", "UA-123456" |
| **integer** | Whole numbers | 123, 0, -5 |
| **float** | Decimal numbers | 123.45, 0.0, -5.5 |
| **ISO 8601** | Date-time format | "2023-05-15T10:30:00+03:00" |

---

## Missing Values

Missing values are represented as:
- **CSV:** Empty cells (blank)
- **Pandas:** `NaN` or `None`
- **Databases:** `NULL`

**High missing % fields:**
- `tender_start_date`, `tender_end_date`: 90% (not required for all procurement types)
- `award_period_start/end`: 93% (optional fields)
- `supplier_region`: 84% (data quality issue in Prozorro system)

---

## Joining Tables

### Example 1: Tenders + Suppliers
```python
import pandas as pd

tenders = pd.read_csv('tenders_2023.csv')
suppliers = pd.read_csv('suppliers.csv')

merged = tenders.merge(suppliers, on='supplier_id', how='left')
print(merged[['tender_id', 'supplier_name', 'total_awards']])
```

### Example 2: Tenders + Buyers
```python
buyers = pd.read_csv('buyers.csv')

merged = tenders.merge(buyers, on='buyer_id', how='left')
print(merged[['tender_id', 'buyer_name', 'buyer_region']])
```

### Example 3: Full Join (All Tables)
```python
# Tenders + Suppliers + Buyers
full = (tenders
    .merge(suppliers, on='supplier_id', how='left', suffixes=('', '_supplier'))
    .merge(buyers, on='buyer_id', how='left', suffixes=('', '_buyer'))
)
```

---

## Relationships

```
buyers (36K)
    └─ buyer_id (PK)
           ↓ (1:many)
    tenders (12.9M)
        ├─ tender_id (PK)
        │     ↓ (1:many)
        │  bids (2.64M)
        │
        └─ supplier_id (FK)
               ↓ (many:1)
    suppliers (358K)
           ↑
    bidders (72K) ── bidder_id = supplier_id (EDRPOU)
```

---

## Currency Conversion

All monetary values are in **Ukrainian Hryvnia (UAH)**.

Approximate USD conversion rates:
- **2022:** 1 USD ≈ 30 UAH (pre-invasion)
- **2023:** 1 USD ≈ 37 UAH
- **2024:** 1 USD ≈ 40 UAH
- **2025:** 1 USD ≈ 42 UAH

**Important:** Exchange rates fluctuated significantly due to war. Use official NBU (National Bank of Ukraine) rates for accurate conversion.

---

## Analysis Tips

### Anomaly Detection

**Single bidder + high value:**
```python
anomalies = tenders[
    (tenders['is_single_bidder'] == 1) &
    (tenders['tender_value'] > 100000)
]
```

**Negative discount (price increase):**
```python
price_increase = tenders[tenders['price_change_pct'] < 0]
```

**Q4 spending spike:**
```python
q4_tenders = tenders[tenders['is_q4'] == 1]
print(q4_tenders['tender_value'].sum() / tenders['tender_value'].sum())
```

### Competition Analysis

**Average bidders by year:**
```python
tenders.groupby('year')['number_of_tenderers'].mean()
```

**Single bidder trend:**
```python
tenders.groupby('year')['is_single_bidder'].mean() * 100
```

---

## Version Information

- **Dataset Version:** 1.2
- **Last Updated:** January 2026
- **Coverage:** January 2022 - December 2025
- **Records:** 12,877,960 tenders (completed only)
- **Source:** Prozorro (prozorro.gov.ua)

---

## Questions?

For questions about specific fields or data quality issues, please:
1. Check Prozorro documentation: https://prozorro.gov.ua
2. Review OCDS standard: https://standard.open-contracting.org
3. Open discussion on Kaggle dataset page

---

**See also:** README.md for dataset overview and quick start guide.
