# AI-Driven-Credit-Risk-Prediction-and-Crisis-Resilience-Analysis (Logistic Regression + XGBoost + SHAP)

Predicting whether a loan applicant is likely to default using real-world financial data from LendingClub.

---

## Project Objective

The goal is to develop robust machine learning models to predict **loan defaults**, enabling lenders to:

- Minimize credit risk exposure
- Improve underwriting decisions
- Optimize expected credit loss (ECL) under Basel II/III
- Ensure explainability for compliance and audit purposes

---

## Dataset

- **Source**: LendingClub public loan data
- **Size**: ~2.6 million rows, 100+ features
- **Target Variable**: `loan_default` (1 = defaulted, 0 = non-defaulted)


---

## ML Pipeline

1. **Data Cleaning + Imputation**
2. **EDA** (FICO Score, Annual Income, DTI, Grade, etc.)
3. **Feature Engineering**
4. **Class Imbalance Handling**:
   - SMOTE (Synthetic Minority Oversampling)
   - Class Weights
5. **Modeling**:
   - Logistic Regression (base, SMOTE, weights)
   - XGBoost
6. **Evaluation**:
   - Accuracy, Recall, Precision, F1-Score
   - ROC Curve
   - Confusion Matrix
7. **Model Explainability**:
   - Logistic Coefficients
   - SHAP values (XGBoost)

---

## Visual Results

### 1️⃣ Confusion Matrices
![Confusion Matrix](images/confusion_matrices.png)

---

### 2️⃣ ROC Curves
![ROC Curves](images/roc_curves.png)

---

### 3️⃣ Precision-Recall Curves
![Precision-Recall Curves](images/precision_recall_curves.png)

---

### 4️⃣ Model Metrics Bar Plot
![Model Comparison](images/model_comparison_barplot.png)

---

## Insights & Interpretation

| Model              | Accuracy | Recall (Defaulters) | Precision | ROC-AUC | Best For |
|-------------------|----------|---------------------|-----------|---------|----------|
| Logistic (Base)    | 89%      | 64%                 | 79%       | 0.94    | High precision use cases |
| Logistic (SMOTE)   | 89%      | 86%                 | 67%       | 0.94    | Balanced risk detection |
| Logistic (Weights) | 89%      | 87%                 | 66%       | 0.94    | Audit-safe, interpretable |
| **XGBoost**        | 88%      | **91%**             | 65%       | **0.955** | Best risk capture & financial modeling ✅ |

- **SHAP Interpretation**:  
  Key features affecting default probability include:
  - `last_fico_range_high`
  - `last_fico_range_low`, `dti`
  - `term_60_months`, `int_rate`, `mo_sin_old_rev_tl_op`

---

##  Financial Relevance

- **Fits Basel IRB Models** for capital requirement calculation
- **Probabilities of Default (PD)** from models feed into:
  - Expected Credit Loss (ECL)
  - Risk-Based Pricing
  - Loan approval workflows
----
###  Crisis Analysis & Real-World Validation (2020–2024)

This project doesn't stop at model performance — it validates whether the model's learned patterns actually align with real-world economic behavior, especially during crisis periods like COVID-19 and the post-stimulus inflationary period.

####  What the Model Learned (SHAP Feature Importance)
Using SHAP interpretability on the XGBoost classifier, it was founded that the **top feature influencing default** was `last_fico_range_high` — the most recent FICO score of the borrower. Other highly impactful features included:

- `annual_inc`: Lower income leads to higher default risk
- `dti`: High debt-to-income ratio significantly raises risk
- `term__60_months`: Longer loan terms increase vulnerability
- `int_rate` and `installment`: Larger loan payments increase financial stress

These features align strongly with fundamental credit risk theory — lower creditworthiness and higher financial burden lead to higher chances of default.

####  Real-World Delinquency Trends (Fed Data)
We used official **Federal Reserve data** from FRED:

> `DRCLACBS`: *Delinquency Rate on Consumer Loans at All Commercial Banks*

Key observations:

| Period      | Delinquency Rate (%) | Key Events |
|-------------|----------------------|------------|
| Early 2020  | ~2.4%                | Pre-COVID baseline |
| Mid 2020    | ~flat/slightly up    | Start of COVID, but mitigated by stimulus + forbearance |
| 2021        | ↓ Below 2.0%         | Forbearance + stimulus helped consumers pay on time |
| 2022–2024   | ↑ to ~2.6%           | Stimulus ends, inflation rises, job losses increase stress |

#### 🔗 Model vs. Macro Comparison
Despite the Fed delinquency rate not spiking immediately in 2020, the model **correctly learned the underlying risk signals** that emerged *before* defaults climbed in 2022+:

- **Low recent FICO scores** (captured in `last_fico_range_high`)
- **Rising DTI**, falling real income, and increasing financial burdens
- **Loan structure (term, interest rate)** exacerbating repayment stress

The model essentially mirrors the risk buildup that the macro data eventually reflected.

####  Conclusion: Realistic and Crisis-Resilient
This proves the model doesn't just overfit historical patterns — it picks up **genuine economic signals** that align with **real-world financial stress events**. This makes it well-suited for:

- Early warning systems
- Adaptive credit risk management
- Crisis scenario testing and capital planning
