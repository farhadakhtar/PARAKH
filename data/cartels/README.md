
# **Public procurement cartels: A large-sample testing of screens using machine learning**
Replication Package
Total Run time without Super Learner ~86.44 mins

This package allows for the full replication of results presented in the study **“Public procurement cartels: A large-sample testing of screens using machine learning.”**  

### System Information

This analysis was conducted in an R environment under the following system configuration:
Specification
**R version** 4.3.0 (2023-04-21 ucrt)
**RStudio version** 2024.04.2 Build 764
**Platform** x86_64-w64-mingw32/x64 (64-bit)
**Operating System** Windows 11 x64 (build 26100)
**Time Zone** Europe/Budapest

###  R Packages and Versions
All computations rely on a curated set of R packages for data wrangling, statistical modeling, and machine learning. The table below lists the main packages and their versions as used in the reference environment.
**Data Processing** tidyr (1.3.0), dplyr (1.1.2), janitor (2.2.0), forcats (1.0.0)
**Visualization** ggplot2 (3.4.2), cowplot (1.1.1), gridExtra (2.3), ggcorrplot (0.1.4.1)
**Statistical Analysis** statar (0.7.5), gmodels (2.18.1.1), arm (1.13-1), stargazer (5.2.3)
**Machine Learning** tree, gbm, xgboost, randomForest, ranger, caret, SuperLearner, nnet, kernlab, LogicReg, biglasso, party, earth, bartMachine
**Explainability & Diagnostics** ROCR, fastshap, shapviz, shapr
**Utilities & Output** writexl, benford.analysis, lattice, skimr, officer, flextable

##  Folder Structure

To maintain consistency use the following directory structure:
(Note: The script creates the desired data structure)
<pre>
```
├── data/
│   ├── tables/
│   └── descriptive_stats/
├── figures/
└── [analysis scripts]
```
 </pre>

-   **data/** — Stores all raw and processed data files.
-   **data/tables/** — Contains analytical tables and model outputs (e.g., regression results).
-   **data/descriptive_stats/** — Holds descriptive summaries and diagnostic tables.
-   **figures/** — Saves visualizations generated during analysis.

## Data Description

The input data is on the contract level for all countries - except Hungary, Sweden and Portugal - it's on the bid-level. To enable analysis we transform the data to  a **contract-level dataset** by retaining only the **winning bid** for each contract. Losing bids in those 3 countries are removed before training the models, as the prediction task focuses on identifying whether a _contract_ is cartel-affected.

Each observation in the cleaned dataset represents **one awarded contract** with features derived from the winning bid, characteristics of the tender, and metadata about competition indicators.

| **Column name** | **Description** |
|-----------------|-----------------|
| `row_id` | Unique row identifier assigned during preprocessing. |
| `country` | Country where the procurement procedure was carried out. |
| `tender_year` | Year when the tender was published. |
| `buyer_id` | Unique identifier of the contracting authority/buyer. |
| `bidder_id` | Identifier of the winning supplier or economic operator. |
| `main_cpv_2` | CPV code aggregated at the 2-digit level (broad sector). |
| `main_cpv_3` | CPV code aggregated at the 3-digit level (mid-level sector) - Missing coded as 990. |
| `main_cpv_3x` | Extended 3-digit CPV grouping used - Missing kept as NA. |
| `main_cpv_4` | CPV code aggregated at the 4-digit level (detailed product/service category). |
| `tender_publications_firstcallfortenderdate` | Date of the first call for tender publication. |
| `tender_publications_firstdcontractawarddate` | Date of the contract award notice. |
| `cartel_id` | Identifier linking the contract to a known cartel case. |
| `is_cartel` | Binary label (1 = During the cartel period, 0 = After the cartel period). |
| `bid_iswinning` | Indicator showing whether the bid is the winning bid (always 1 in final dataset). |
| `lot_bidscount` | Number of bids submitted for the lot. |
| `singleb` | Indicator for single bidding (1 = only one bid submitted). |
| `bid_isconsortium` | Whether the winning bid is submitted by a consortium. |
| `bid_issubcontracted` | Whether subcontracting was declared for the winning bid. |
| `bid_price` | Winning bid price. |
| `lot_estimatedprice` | Estimated value of the lot prior to receiving bids. |
| `tender_estimatedprice` | Estimated total value of the tender (sum of lots). |
| `tender_finalprice` | Final contracted amount reported after award. |


## 1. Reproducibility & Computational Environment

This section prepares a clean R session, loads all required packages, and sets up the folder structure for data, tables, and figures.  
It ensures that analyses can be fully replicated under the documented system and package versions.

## II. Data Preparation
Run time: 3.89 mins
This section loads the contract-level dataset and prepares it for analysis.  
It creates bidder-level indicators (e.g., number of buyers, markets, and contracts per year), handles missing or consortium bids, and filters out losing bids from specific countries (HU, SE, PT).  
Finally, it restricts the data to relevant years and prepares the filtered dataset for indicator calculation.

## III. Building and Checking Indicators
Run time:  8.76 mins

This section constructs and validates key analytical indicators from the cleaned dataset.
**Main steps:**

1.  **Cartel variable:**  
    Converted `is_cartel` from logical to numeric (1/0) and ensured consistent factor format.

2.  **Single bidding indicator:**  
    Created `singleb` = 1 for tenders with only one bid, handling missing bid counts.

3.  **Bid count cleanup:**  
    Checked and truncated `lot_bidscount` (set implausible or coded values like 999999 to `NA` and capped at 50).

4.  **Relative bid value:**  
    Computed bidder’s price relative to estimated tender value; replaced outliers and missing ratios with `NA`.

5.  **Consortium and subcontract indicators:**  
    Standardized and converted `bid_isconsortium` and `bid_issubcontracted` to numeric, filling missing values appropriately.

6.  **Benford’s Law conformity:**  
    Computed Benford-based indicators (`MAD` and conformity) for buyers and markets (with ≥100 records) to detect irregular pricing patterns, then merged back to the main dataset.

7.  **Bidder-level yearly aggregates:**  
    Created bidder-year averages and one-year lag values for key metrics (e.g., single bidding rate, bid count, relative value, Benford conformity).

8.  **Filtering small-sample bidders:**  
    Excluded aggregate and lag values for bidders with ≤3 contracts per year.

9.  **Final subset:**  
    Extracted `df` containing only observations with non-missing `is_cartel` for model estimation.

## IV. Variable Cleaning, Aggregation, and Summary Statistics
Run time:  1.73 secs

This section refines variables, prepares descriptive summaries, and finalizes the dataset for modeling.

**Main steps:**

3.  Creating **cartel_tender:**  
    Created a unique bidder tender-year identifier (`cartel_tender`) for later train/test splits. This ensures that there is no leakage between the train/test splits.

4.  **Summary statistics generation:**  
    Produced global and country-level descriptive statistics using a custom `generate_summary()` function:

    -   Compiled means, SDs, medians, and missing rates.

    -   Exported results as Word and CSV tables.

5.  **Handling missing data:**  
    Replaced missing numeric values with variable means and converted categorical variables to factors with placeholder codes.

6.  **Market re-aggregation:**  
    Collapsed detailed CPV2 categories into 34 broader groups (`cpv2_smp`) for modeling compatibility with random forest constraints.

7.  **Final dataset:**  
    The cleaned and aggregated data (`df` and `df_full`) can be saved as the finalized input for subsequent modeling stages.

## V. Model Building and Hyperparameter Tuning
Run time (without GBM tuning + Super Learner): 14.43 mins

This section focuses on constructing, tuning, and comparing machine learning models to detect cartel behavior.  
The main goal is to identify the optimal model specifications — particularly for **Random Forest (RF)** and **Gradient Boosting (GBM)** — to be later used in the **5-fold cross-validation** exercise.
The best parameter configurations identified here were stored in `model_functions.R` and subsequently used for the final predictive modeling stage.

**Main steps:**

1.  **Data splitting:**  
    The dataset was filtered to labeled observations (`is_cartel ∈ {0,1}`) and split into training (≈80%) and testing (≈20%) sets via **cluster sampling** at the tender level (`cartel_tender`), ensuring that bids from the same tender did not appear in both sets.

2.  **Random Forest (RF):**

    -   Baseline model trained with `ntree = 3000` and `mtry = √p`, where p is the number of predictors.

    -   Hyperparameter tuning via `tuneRF()` optimized mtry  by minimizing the out-of-bag (OOB) error.

    -   The best-performing configuration was mtry=10, ntree=1000.

    -   Model diagnostics included **accuracy**, **specificity**, **F1**, **ROC-AUC**, and threshold optimization.

    -   Feature importance was evaluated via Mean Decrease Gini, with partial dependence (PDP) and SHAP plots illustrating marginal and interaction effects of key predictors (e.g., number of bids, Benford conformity, subcontracting).

3.  **Boosting (GBM):**

    -   Grid search across parameters
        ntrees∈{600,800,3000},
        interaction.depth∈{4,6},
        shrinkage∈{0.01,0.1,0.2},
        n.minobsinnode∈{10,20},
        using 5-fold cross-validation (`caret::train()` with `metric = "ROC"`).

    -   The optimal GBM had ntrees=3000, interaction.depth=6, shrinkage =0.01, and  n.minobsinnode=10.
    -   Results showed strong predictive stability and comparable AUC to the tuned RF.

4.  **Benchmark models:**

    -   **Logistic regression** (baseline) was estimated for interpretability and comparison.

    -   **Super Learner ensemble** combined multiple learners (GLM, RF, SVM, XGBoost, etc.) using AUC-based weighting to benchmark overall performance.

**Purpose:**  
This tuning phase ensured that both tree-based models (RF and GBM) were optimally specified for subsequent **5-fold cross-validation** and performance comparison in the final modeling stage.

## VI. Test 1 – Contract-Level Sampling (5-Fold Validation)
Run time (without Super Learner):  22.89 mins

This section performs the first cross-validation test, where **sampling is done at the contract level**

**Design:**
-   The labeled dataset (`is_cartel ∈ {0,1}`) is repeatedly split into **5 folds** via **cluster sampling** on the variable `cartel_tender`.
-   Each fold holds out ≈20% of tenders for testing, ensuring that no contract appears in both training and test data.

**Models evaluated:**
-   **Random Forest (RF)** using tuned `mtry` values depending on feature set complexity.
-   **Gradient Boosting (GBM)** with parameters from Section V (optimized for AUC).
-   **Logistic Regression (Logit)** for benchmark.
-  **Super Learner** for comparison.
**Feature sets:**  
Five configurations were compared — from simple controls to the full model to assess the predictive contribution of key indicator groups (e.g., single bidding, subcontracting, Benford indicators etc.).

**Outputs:**

-   Main results (Table 5): average **AUC**, **accuracy**, and **specificity** across folds for the “All Features” specification.
-   Supplementary results (Table A8): feature-set comparisons across models.

This test establishes baseline predictive performance using **contract-level cross-validation**, ensuring model generalizability without cross-contract contamination.

## VII. Test 2 – Cartel-Level Sampling (5-Fold Validation)
Run time (without Super Learner):  8.12 mins

This test evaluates model performance using **cartel-level cross-validation**, where all contracts from a given cartel are either in the training or test set.

**Design:**
-   The unique list of `cartel_id`s is randomly shuffled and split into **5 folds**.
-   For each fold, all contracts from the selected cartels are used as the **test set**, while the rest form the **training set**.

**Models evaluated:**
-   **Random Forest (RF)** with `mtry = 10`.
-   **Gradient Boosting (GBM)** using binary outcome (`is_cartel == 1`).
-   **Logistic Regression (Logit)** as benchmark.
-   **Super Learner models** tested.

**Outputs:**

-   Fold-wise results include accuracy, AUC, specificity, F1 scores, and best thresholds.
-   **Table 5 (Test 2)** reports mean metrics across folds for each model.

This approach tests **generalization across different cartels**, highlighting whether models rely on cartel-specific patterns.

## VIII. Test 3 – Country-Level Sampling (Leave-One-Country-Out Validation)
Run time (without Super Learner):  12.47 mins

This section evaluates **geographic generalization**, testing whether models trained on all but one country can predict cartel behavior in the held-out country.

**Design:**
-   Each country is sequentially held out as the **test set**.
-   Training data includes all other countries.
-   Checks effect of  country-specific patterns

**Models evaluated:**
-   **Random Forest (RF)**, tuned similarly to previous tests.
-   **Gradient Boosting (GBM)** adapted for binary outcomes.
-   **Logistic Regression (Logit)**.
-   **Super Learner models**.

**Outputs:**

-   Country-wise results include accuracy, specificity, F1 scores, AUC, and thresholds.
-   **Table 5 (Test 3)** reports the average metrics per model across all countries.
-
## IX. Test 4: Incrementally Adding Countries
Run time (without Super Learner):  6.61 mins

This section evaluates the model performance when adding countries one by one to the training set.

-   **Design:**

    1.  Start with Hungary (`HU`) and incrementally add countries: `SE`, `ES`, `PT`, `LV`, `BG`.
    2.  Ensure **80/20 train-test split** at the bidder-year level for each country.
    3.  Simplify product codes and lump infrequent categories to achieve ≤32 distinct market categories.
    4.  Train multiple models on the growing dataset:
        -   Random Forest (RF)

        -   Gradient Boosting (Boost)

        -   Super Learner

    5.  Evaluate models using **Accuracy, AUC, F1 Score, Best Threshold**.


## X. Prediction Profiles Using Random Forest
Run time:  ~9.24 mins

This section generates and  explores the predicted probabilities of collusion for all contracts and suppliers.
To run this section you will need
- cpv_levels_RF from Section IV Data Check _ Summary (Subset)  -> (Re)-Aggregating Markets in data subset
- RF_best from Section V -> Random Forest -> Main Random forest Model


-   **Data Preparation:**

    -   Factorize categorical variables and impute missing values using mean of training data.
    -   Ensure Market levels in the prediction dataset match those used in RF training.

-   **Prediction Steps:**

    1.  Predict **collusion probability** (`is_cartel_pred1`) using trained Random Forest.

    2.  Convert probabilities to binary predictions (`is_cartel_pred`) using `best_threshold`.

    3.  Evaluate performance with **confusion matrix** (Accuracy, F1 Score).

-   **Visualization:**

    -   **Contract-level histogram** (`fig_5_cartel_prediction_contracts.png`)

    -   **Supplier-level histogram** – average predicted probabilities for suppliers with >10 contracts.

    -   **Country-level distributions:**

        -   Violin plots with mean values (`fig_6_cartel_prediction_country_violin.png`)

        -   Average predicted probability bar chart per country

    -   **Market-level analysis:**

        -   Mean probability vs. total spending per market (`fig_7_cartel_prediction_markets_spending.png`)

        -   Trends over time for top markets (`fig_8_cartel_prediction_top_markets_country.png`)

-   **Notes:**
    -   Probabilities are continuous `[0,1]`; thresholds can be adjusted for different FPR/Sensitivity trade-offs.
