# 🛡️ Credit Card Default Prediction: Cost-Sensitive Learning

**👉 [Click here to read the full Research Paper (PDF)](./Credit_Risk_Prediction_Cost_Sensitive_Learning.pdf)**

## 📌 Executive Summary

This research develops machine learning classifiers to predict credit card client default probability using Taiwan's Default of Credit Card Clients Dataset (30,000 samples, ~22% default rate). The study addresses severe class imbalance through stratified sampling, feature engineering, and cost-sensitive optimization, prioritizing **Recall** over raw accuracy to align with financial risk management imperatives.

**Random Forest** emerges as the strongest baseline (F1=0.68, AUC=0.77), while **cost-sensitive XGBoost**—tuned with `scale_pos_weight` ≈3.5—elevates Recall to 0.71, reducing false negatives (missed defaulters) at controlled precision cost. Engineered features like **Credit Utilization Ratio** (`BILL_AMT1/LIMIT_BAL`) and **Payment Ratio** (`PAY_AMT1/BILL_AMT2`) rank among top predictors, enhancing both performance and financial interpretability.

---

## 💡 Key Highlights & Methodology

### 1. Dataset Characteristics

**Taiwan Credit Card Dataset** (30K samples):
- **Demographics**: Age, education, marriage status
- **Payment behavior**: 6-month repayment delays (`PAY_0` to `PAY_6`)
- **Financials**: Bill amounts (`BILL_AMT1-6`), payment amounts (`PAY_AMT1-6`), credit limits (`LIMIT_BAL`)
- **Severe imbalance**: 77.8% non-default vs 22.2% default

Stratified 70/30 train-test split preserves class ratios for robust evaluation.

### 2. Targeted Feature Engineering

Three domain-motivated financial stress indicators:

| Engineered Feature | Formula | Financial Interpretation |
|-------------------|---------|--------------------------|
| **Credit Utilization** | `BILL_AMT1 / LIMIT_BAL` | Near-limit exhaustion |
| **Payment Ratio** | `PAY_AMT1 / BILL_AMT2` | Repayment adequacy |
| **Bill Growth** | `(BILL_AMT1-BILL_AMT2)/BILL_AMT2` | Sudden spending surge |

### 3. Progressive Modeling Pipeline

| Model | Accuracy | Precision | **Recall** | F1-Score |
|-------|----------|-----------|------------|----------|
| Logistic Regression | 0.802 | 0.74 | 0.59 | 0.61 |
| Decision Tree | 0.723 | 0.60 | 0.61 | 0.61 |
| **Random Forest** | **0.820** | **0.75** | **0.65** | **0.68** |
| **XGBoost (Cost-Sensitive)** | 0.760 | 0.67 | **0.71** | **0.68** |

**Key insight**: Cost-sensitive XGBoost trades accuracy for +9% Recall uplift—critical as false negatives cost far more than false positives in credit risk.

---

## 📊 Core Findings

### 🔍 Feature Importance & Interpretability

**Dominant predictors** (Random Forest ranking):
1. `PAY_0` (recent repayment delay) 
2. **Credit Utilization** (engineered)
3. `PAY_2`, `LIMIT_BAL`, **Payment Ratio**

Confirms financial intuition: **recent payment behavior + utilization stress** drive defaults.

### 📈 Business Impact

**Production deployment** enables:
- **Proactive intervention** on high-risk clients (Recall=71%)
- **Expected loss reduction** via earlier default detection
- **Portfolio stability** through systematic risk transfer


