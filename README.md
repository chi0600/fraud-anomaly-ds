# Fraud / Anomaly Detection – Credit Card Transactions

An end-to-end data science project focused on fraud detection under severe class imbalance, with emphasis on PR-AUC evaluation and precision–recall trade-offs.

---

## Problem

Fraud detection datasets are typically highly imbalanced.  
In this dataset:

- Total transactions: 284,807
- Fraud cases: 492 (≈0.17%)

Under such imbalance, **accuracy becomes misleading**, and evaluation must focus on ranking and recall–precision trade-offs.

---

## Approach

### 1. Exploratory Data Analysis (EDA)

- Verified severe class imbalance (0.17% fraud)
- Examined transaction time distribution
- Analyzed transaction amount (log-scale visualization)
- Identified right-skewed distributions and heavy-tailed non-fraud behavior

---

### 2. Baseline – Logistic Regression (Leakage-Free Pipeline)

- Stratified train/test split
- StandardScaler + Logistic Regression using `Pipeline`
- `class_weight='balanced'` to address imbalance
- Evaluation using **PR-AUC**

**Logistic Regression PR-AUC:** ~0.72

Threshold tuning demonstrated clear precision–recall trade-offs:
- Lower thresholds increased recall but caused excessive false positives
- Higher thresholds (e.g., 0.9) provided more practical operating points

---

### 3. Model Comparison – Random Forest

- Tree-based model (no scaling required)
- Captures nonlinear feature interactions

**Random Forest PR-AUC:** ~0.85

Random Forest significantly improved ranking capability over Logistic Regression, indicating that fraud detection in this dataset involves nonlinear feature relationships.

---

## Key Insights

- Accuracy is not appropriate for highly imbalanced fraud detection.
- PR-AUC provides a more meaningful evaluation metric.
- Threshold selection is an operational decision balancing fraud capture vs user impact.
- Tree-based ensemble models outperform linear models in this setting.

---

## Project Structure
- `notebooks/01_eda.ipynb` – Exploratory data analysis and initial observations
- `requirements.txt` – Python dependencies
- `data/` – Local data (ignored by git)

---

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt