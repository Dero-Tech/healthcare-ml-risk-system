# Healthcare Risk Prediction System

> Production-style ML pipeline for cost-sensitive patient risk scoring using synthetic EHR data.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.97-brightgreen) ![Cost-Sensitive](https://img.shields.io/badge/Cost--Sensitive-FN%2010x%20FP-orange) ![Azure ML](https://img.shields.io/badge/Cloud-Azure%20ML%20Ready-0078D4)

---

## For Non-Technical Readers

This system predicts which patients are at highest risk of generating high healthcare costs — enabling proactive care management before expensive events occur.

Rather than treating all prediction errors equally, the model is intentionally tuned to flag more patients for clinical review rather than miss a high-risk individual. A missed high-risk patient (false negative) was weighted 10 times more costly than a false alarm (false positive) — directly reflecting how real utilization management decisions work in clinical and insurance settings.

---

## Key Results

| Metric | Value |
|---|---|
| ROC-AUC | **0.97** |
| Brier score improvement after calibration | Improved (Platt scaling) |
| Theoretical optimal threshold | 0.09 |
| Empirically deployed threshold | **0.05** |
| False negative cost weight | **10x false positive** |

### Key Insight: Theory vs. Empirical Threshold

> Although theoretical decision analysis suggested an optimal threshold of **0.09**, empirical cost minimization on the actual score distribution selected **0.05**.
>
> This divergence demonstrates how finite sample effects and probability distribution shape influence real deployment policy — and why threshold decisions in production systems require empirical validation, not just theoretical calculation. In a clinical context, this difference determines how many patients are flagged for utilization review.

---

## Pipeline Architecture

This was built as a full end-to-end production pipeline — not just a model training notebook.

```
Raw EHR Data
    │
    ▼
1. Data Ingestion + Cleaning
   - Schema validation
   - Missing value handling
   - Categorical encoding
    │
    ▼
2. Feature Engineering
   - Categorical encoding
   - Clinical feature construction
    │
    ▼
3. Stratified Train / Test Split
   - Preserves class imbalance ratio
    │
    ▼
4. Model Training
   - Classification model (scikit-learn)
   - Cross-validation
    │
    ▼
5. Probability Calibration
   - Platt scaling (sigmoid calibration)
   - Brier score evaluation pre/post
    │
    ▼
6. Cost-Sensitive Threshold Optimization
   - FN cost = 10x FP cost
   - Empirical vs. theoretical threshold comparison
    │
    ▼
7. Evaluation + Reporting
   - ROC-AUC, precision/recall, confusion matrix
   - SHAP feature importance
   - Threshold cost curve visualization
```

---

## Core Concepts Demonstrated

### Discrimination vs. Calibration
A model can rank patients correctly (high AUC) but still produce poorly calibrated probabilities. This pipeline evaluates both — discrimination via ROC-AUC and calibration via Brier score — and applies Platt scaling to improve probability estimates before threshold decisions are made.

### Cost-Aware Decision Systems
Standard ML evaluation metrics (accuracy, F1) treat all errors equally. In healthcare, a missed high-risk patient is far more costly than a false alarm. This pipeline builds cost asymmetry directly into threshold selection.

### Ranking vs. Probability Estimation
The model serves two purposes: ranking patients by risk (discrimination) and estimating the actual probability of a high-cost event (calibration). These require different evaluation methods and optimization strategies.

---

## Connection to Medical Claims Data

This project uses synthetic EHR data. In a production healthcare setting (e.g., health insurance, utilization management), the same pipeline would operate on:

| Synthetic Feature | Real Claims Equivalent |
|---|---|
| Diagnosis codes | ICD-10-CM codes |
| Procedure history | CPT / HCPCS codes |
| Utilization frequency | Claims volume, admission counts |
| Cost history | Paid claims amounts, PMPM spend |
| Demographic features | Member enrollment data |

The cost-sensitive framework directly mirrors real utilization management workflows:
- **Prior authorization** — flag high-risk procedures before approval
- **Concurrent review** — monitor high-risk inpatient stays
- **Retrospective analysis** — identify patterns in historical spending

---

## Model Interpretability (SHAP)

Clinical and business stakeholders require explanations for model predictions. This pipeline includes SHAP (SHapley Additive exPlanations) analysis to identify which features drive individual risk scores — supporting the auditability requirements of healthcare AI systems.

*(See `/assets/shap_summary.png`)*

---

## Cloud Deployment — Azure ML

The model is designed for transition to Azure ML Studio, including:
- Model registration via `azureml-sdk`
- Managed endpoint deployment
- Data drift monitoring for production retraining triggers

See `azure/deploy.py` for the deployment stub.

---

## Repository Structure

```
healthcare-ml-risk-system/
├── scripts/
│   ├── 01_ingest.py              # Data ingestion + schema validation
│   ├── 02_feature_engineering.py # Feature construction + encoding
│   ├── 03_train.py               # Model training + cross-validation
│   ├── 04_calibrate.py           # Platt scaling calibration
│   ├── 05_threshold_optimize.py  # Cost-sensitive threshold selection
│   └── 06_evaluate.py            # Full evaluation report + plots
├── azure/
│   └── deploy.py                 # Azure ML deployment stub
├── assets/
│   ├── roc_curve.png
│   ├── threshold_cost_curve.png
│   └── shap_summary.png
├── data/
│   └── synthetic_ehr_sample.csv
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Dero-Tech/healthcare-ml-risk-system.git
cd healthcare-ml-risk-system
pip install -r requirements.txt
python scripts/01_ingest.py
python scripts/03_train.py
python scripts/05_threshold_optimize.py
```

---

## About This Project

Built independently to demonstrate production ML thinking in the healthcare domain — including the real-world tradeoffs between discrimination, calibration, and cost-aware deployment that are often absent from academic ML projects.

**Topics:** `healthcare-ml` `risk-prediction` `cost-sensitive-learning` `scikit-learn` `python` `utilization-management` `azure-ml`
