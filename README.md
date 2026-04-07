# SBA-Loan-Default-Prediction
Analyzing 400K+ SBA loans to test whether franchises default less than independent businesses and predict loan defaults using machine learning (Random Forest AUC: 0.970).

---

## Why This Matters

The SBA guaranteed over **$27 billion in small business loans** in 2023 alone. When a loan defaults, the SBA absorbs the loss which means that taxpayer money covers the gap. Yet approval decisions today still rely heavily on manual judgment and simple credit checks.

This project asks: **can we predict loan default before disbursement, using only information available at the time of approval?**

If yes, a model like this could help SBA loan officers flag high-risk applications earlier, reducing losses without turning away creditworthy borrowers. It also tests a common assumption in lending: that **franchised businesses are safer bets than independent ones** — and finds the answer is more complicated than a yes or no.

---

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

## Results

### Model Comparison

| Model | AUC | Accuracy | Recall (Defaulted) |
|---|---|---|---|
| Logistic Regression | 0.856 | 0.83 | 38% |
| Decision Tree | 0.947 | 0.91 | 74% |
| **Random Forest** | **0.970** | **0.94** | **81%** |

> Fill in Accuracy column from your classification report outputs.

**Random Forest** is the best model. At 81% recall, it correctly flags 4 out of 5 loans that will actually default — catching far more risk than the other models.

> ⚠️ **Why not just use accuracy?** A model that predicts "Paid in Full" for every loan would score ~78% accuracy but catch zero defaults. Recall and AUC are what actually matter here.

### ROC Curves

<img width="630" height="470" alt="roc_curves" src="https://github.com/user-attachments/assets/51192d75-5c1e-46ab-99e4-cc0debce4f76" />

Random Forest (green) pulls away from the other two models, hugging the top-left corner — the ideal zone for a classifier.

### Confusion Matrices

<img width="1416" height="390" alt="confusion_matrices" src="https://github.com/user-attachments/assets/7d2faf41-e53e-4cb6-a52e-7a9abe090bf7" />


### Feature Importance

**Term** is the single most dominant predictor by a wide margin, followed by approval year and loan amount — all information available before disbursement.

<img width="989" height="590" alt="feature_importance" src="https://github.com/user-attachments/assets/d2beaa70-824c-432e-b155-f2186f552c0c" />


---

## Key Findings 

**1. Loan term is the #1 predictor.**
Longer loan terms are strongly associated with higher default rates. This is a signal lenders can act on at approval time.

**2. Franchises default less — but lose more when they do.**
Franchise businesses show a lower overall default rate than independents (~15.8% vs ~21.2%). However, when controlling for industry sector, the gap narrows significantly. And when franchises do default, the loss-per-dollar tends to be higher. The "franchise = safe" assumption is an oversimplification.

<img width="630" height="470" alt="franchise_vs_independent" src="https://github.com/user-attachments/assets/8ecda6df-99e2-42f6-a615-a7f3bd043dcf" />


**3. Business age is not a useful predictor.**
New vs. existing businesses defaulted at virtually identical rates — 21.5% vs. 21.4%. This feature added no predictive power.

**4. LowDoc loans are counterintuitively safer.**
Less paperwork ≠ more risk. LowDoc loans had lower default rates than standard loans, likely because only established, lower-risk borrowers qualified for the program before it was discontinued post-2008.

**5. The 2008–09 financial crisis is clearly visible in the data.**
Default rates spiked sharply for loans approved around 2006–2008, confirming the model's ability to capture macroeconomic risk patterns over time.

<img width="987" height="390" alt="default_rate_over_time" src="https://github.com/user-attachments/assets/7907dd27-f7bf-4a96-b650-431cf21ba6c5" />


---

## So What? — Practical Implications

**For SBA loan officers:**
Loan term and SBA guarantee percentage are strong pre-approval signals. Applications with very long terms or unusually high SBA exposure relative to total loan size warrant extra scrutiny even if other indicators look clean.

**For franchise lenders:**
Don't give franchises a blanket pass. The default rate advantage is real but small once you control for industry. Focus on the specific sector and loan structure, not just the franchise flag.

**For policy:**
The LowDoc finding suggests that simplifying the loan application process does not necessarily increase risk — in fact, it may select for more reliable borrowers. This could inform future program design.

**For the model itself:**
An 81% recall Random Forest means 1 in 5 defaults still slips through. Before deploying in a real lending context, the cost of a missed default (false negative) vs. a wrongly rejected borrower (false positive) needs to be explicitly calibrated into the threshold. This model is a starting point and not a decision-maker.

---

## Methods Summary

### Data Cleaning
- Stripped `$` and commas from currency columns; cast to numeric
- Dropped nulls across all columns except `NoEmp`, `CreateJob`, `RetainedJob`, `ChgOffDate`
- Removed invalid rows (single-character city/zip/NAICS, non-numeric `ApprovalFY`, `Term = 0`, invalid categorical values)

### Feature Engineering
| Feature | Description |
|---|---|
| `ChargedOff` | Target: 1 = defaulted, 0 = paid in full |
| `Franchised` | 1 = franchise, 0 = independent |
| `NewExist` | 0 = existing business, 1 = new business |
| `RevLineCr` / `LowDoc` | Y → 1, N → 0 |
| `NAICS_2Digit_Code` | Industry sector (first 2 digits of NAICS) |
| `SBA_Guarantee_Pct` | `SBA_Appv / GrAppv` |

### ML Pipeline
1. One-hot encode `NAICS_2Digit_Code` and `UrbanRural`
2. 70/30 stratified train/test split
3. MinMax normalization
4. Train: Logistic Regression, Decision Tree (`max_depth=5`), Random Forest (`n=100`)
5. Evaluate: classification report, ROC-AUC, confusion matrix, feature importance

### Clustering
K-means (`k=4`) on loan size, term, SBA guarantee %, employee count, franchise status, and business age — to identify distinct borrower risk profiles.

## Model Stress Testing

Three checks were run to validate model reliability:

1. **Overfitting check** — Train AUC vs. Test AUC gap (flagged if > 0.05)
2. **Dummy classifier baseline** — compared against a model that always predicts the majority class
3. **5-fold cross-validation** — mean AUC and standard deviation across folds

---

## Project Structure

```
sba-loan-default-prediction/
├── SBA_Loans.ipynb
├── images/
│   ├── franchise_vs_independent.png
│   ├── default_rate_over_time.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── feature_importance.png
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
   git clone https://github.com/caitlinyu98/SBA-Loan-Default-Prediction.git
   cd SBA-Loan-Default-Prediction
   ```

3. Download `SBAnational.csv` from [Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied/data) and place it in the project folder.

4. Run all cells in `SBA_Loans.ipynb`.

---

## License

Created for academic purposes — QST IS 823, Boston University, Spring 2026.
[SBA_README.md](https://github.com/user-attachments/files/26545381/SBA_README.md)
than independent ones** — and finds the answer is more complicated than a yes or no.
