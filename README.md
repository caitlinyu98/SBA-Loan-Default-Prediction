# SBA-Loan-Default-Prediction
Analyzing 400K+ SBA loans to test whether franchises default less than independent businesses and predict loan defaults using machine learning (Random Forest AUC: 0.970).

## Overview

This project analyzes over 400,000 historical SBA (Small Business Administration) loans to predict whether a small business will default. Using Python, I performed end-to-end data cleaning, exploratory data analysis, machine learning, borrower clustering, and model stress testing.

The target variable is **ChargedOff** — derived from `MIS_Status`: `CHGOFF` → 1 (defaulted), `P I F` → 0 (paid in full).

**Core question:** Are franchised businesses safer bets for SBA loans than independent businesses?

---

## Dataset

- **Source:** [Should This Loan Be Approved or Denied?](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied/data) — Kaggle
- **File:** `SBAnational.csv`
- **Size:** 400,000+ rows, 34 columns
- **Class distribution:** ~78% paid in full, ~22% defaulted

> The full dataset was used — no sampling was applied.

---

## Data Cleaning

All preprocessing was done in Python:

- Stripped `$` signs and commas from currency columns (`DisbursementGross`, `BalanceGross`, `ChgOffPrinGr`, `GrAppv`, `SBA_Appv`) and cast to numeric
- Dropped rows with nulls in all columns except `NoEmp`, `CreateJob`, `RetainedJob`, `ChgOffDate`
- Removed invalid rows where:
  - `City`, `Zip`, or `NAICS` contained only a single character
  - `ApprovalFY` was non-numeric
  - `Term` was 0
  - `NewExist` was not 1 or 2
  - `RevLineCr` or `LowDoc` was not `Y` or `N`

---

## Feature Engineering

| Feature | Description |
|---|---|
| `ChargedOff` | Target: 1 = defaulted, 0 = paid in full (from `MIS_Status`) |
| `Franchised` | 1 = franchise, 0 = independent (derived from `FranchiseCode`) |
| `NewExist` | Recoded: 0 = existing business, 1 = new business |
| `RevLineCr` | Recoded: Y → 1, N → 0 |
| `LowDoc` | Recoded: Y → 1, N → 0 |
| `NAICS_2Digit_Code` | First 2 digits of NAICS code (industry sector) |
| `SBA_Guarantee_Pct` | `SBA_Appv / GrAppv` — share of loan guaranteed by SBA |

---

## Exploratory Data Analysis

Visualizations produced:

- Overall loan outcome distribution
- Default rate: Franchise vs. Independent
- Default rate by industry (NAICS sector)
- Default rate over time by `ApprovalFY` (with 2008 financial crisis callout)
- Default rate: New vs. Existing business
- Default rate: LowDoc vs. Standard loans
- Loan amount distribution by outcome
- SBA Guarantee % distribution by outcome
- Default rate by industry, broken down by franchise status
- Correlation heatmap across numeric features

---

## Machine Learning

### Features Used

```
Franchised, NewExist, Term, DisbursementGross, GrAppv,
SBA_Guarantee_Pct, LowDoc, RevLineCr, UrbanRural,
NoEmp, NAICS_2Digit_Code, ApprovalFY
```

### Pipeline

1. One-hot encode `NAICS_2Digit_Code` and `UrbanRural`
2. 70/30 train/test split (stratified by `ChargedOff`)
3. MinMax normalization
4. Train three models:
   - **Logistic Regression** (`max_iter=1000`)
   - **Decision Tree** (`max_depth=5`)
   - **Random Forest** (`n_estimators=100`, `n_jobs=-1`)

### Results

| Model | AUC | Recall (Defaulted) |
|---|---|---|
| Logistic Regression | 0.856 | 38% |
| Decision Tree | 0.947 | 74% |
| **Random Forest** | **0.970** | **81%** |

Evaluation included ROC curves, confusion matrices, classification reports, and Random Forest feature importance.

> **Note:** Accuracy alone is misleading here — a model that always predicts "Paid in Full" would score ~78% accuracy but catch zero defaults. ROC-AUC and recall are the primary metrics.

---

## Clustering (Borrower Profiles)

K-means clustering (`k=4`) was applied to identify distinct borrower segments:

**Features used:** `DisbursementGross`, `Term`, `SBA_Guarantee_Pct`, `NoEmp`, `Franchised`, `NewExist`

Each cluster was profiled by default rate, average loan size, average term, and franchise share.

---

## Model Stress Testing

Three checks were run to validate model reliability:

1. **Overfitting check** — Train AUC vs. Test AUC gap (flagged if > 0.05)
2. **Dummy classifier baseline** — compared against a model that always predicts the majority class
3. **5-fold cross-validation** — mean AUC and standard deviation across folds

---

## Key Findings

- **Loan term** is the strongest predictor of default.
- **Franchise businesses** default at a lower rate than independents, but the difference narrows when controlling for industry.
- **Business age** (new vs. existing) showed virtually no difference — 21.5% vs. 21.4% default rate — and is not a meaningful predictor on its own.
- **LowDoc loans** counterintuitively have *lower* default rates than standard loans, likely due to selection bias (more established borrowers) and the program's discontinuation post-2008.
- The **2008–09 financial crisis** is clearly visible as a spike in default rates by approval year.
- All three models passed overfitting and cross-validation checks.

---

## Project Structure

```
sba-loan-default-prediction/
│
├── SBA_Loans.ipynb    # Full pipeline: cleaning, EDA, ML, clustering, stress tests
└── README.md
```

---

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn jupyter
   ```

2. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sba-loan-default-prediction.git
   cd sba-loan-default-prediction
   ```

3. Download `SBAnational.csv` from [Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied/data) and place it in the project folder.

4. Launch Jupyter and run all cells:
   ```bash
   jupyter notebook SBA_Loans.ipynb
   ```

---

## License

This project was created for academic purposes as part of QST IS 823 at Boston University.
