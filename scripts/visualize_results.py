import json
import numpy as np
import matplotlib.pyplot as plt

# Load metrics
with open("reports/metrics_day9.json", "r") as f:
    metrics = json.load(f)

raw_auc = metrics["raw_auc"]
cal_auc = metrics["calibrated_auc"]

raw_brier = metrics["raw_brier"]
cal_brier = metrics["calibrated_brier"]

best_threshold = metrics["cost_analysis"]["best_threshold"]["threshold"]
theoretical_threshold = metrics["cost_analysis"]["theoretical_threshold"]

# --- 1. AUC Comparison ---
plt.figure()
plt.bar(["Raw AUC", "Calibrated AUC"], [raw_auc, cal_auc])
plt.title("ROC-AUC Comparison")
plt.ylim(0.9, 1.0)
plt.ylabel("AUC")
plt.savefig("reports/auc_comparison.png")
plt.close()

# --- 2. Brier Score Comparison ---
plt.figure()
plt.bar(["Raw Brier", "Calibrated Brier"], [raw_brier, cal_brier])
plt.title("Brier Score Improvement After Calibration")
plt.ylabel("Brier Score (Lower is Better)")
plt.savefig("reports/brier_comparison.png")
plt.close()

# --- 3. Cost Curve ---
thresholds = np.linspace(0.01, 0.99, 99)
costs = []

# Recompute costs using stored FN/FP logic
from sklearn.metrics import roc_auc_score  # dummy import to ensure sklearn installed
import pandas as pd

df = pd.read_csv("data/processed/features_v1.csv")
y = df["label_risk"].astype(int)

# We need calibrated probabilities again
import joblib
model = joblib.load("models/baseline_calibrated.pkl")

drop_cols = ["label_risk"]
if "patient_id" in df.columns:
    drop_cols.append("patient_id")
X = df.drop(columns=drop_cols)

# Split same way as training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

cal_prob = model.predict_proba(X_test)[:, 1]

for t in thresholds:
    y_pred = (cal_prob >= t).astype(int)
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    total_cost = 10 * fn + 1 * fp
    costs.append(total_cost)

plt.figure()
plt.plot(thresholds, costs)
plt.axvline(best_threshold)
plt.axvline(theoretical_threshold)
plt.title("Expected Cost vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Total Cost")
plt.savefig("reports/cost_curve.png")
plt.close()

print("Charts saved in reports/ folder.")

