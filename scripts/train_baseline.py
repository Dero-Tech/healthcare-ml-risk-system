def expected_cost(y_true, y_prob, threshold, c_fn=10, c_fp=1):
    y_pred = (y_prob >= threshold).astype(int)

    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    total_cost = c_fn * fn + c_fp * fp
    return total_cost, fn, fp

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import numpy as np
import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

LABEL_COL = "label_risk"
ID_COL = "patient_id"


def eval_at_threshold(y_true, cal_prob, threshold: float):
    y_pred = (cal_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Train + evaluate baseline healthcare risk model.")
    p.add_argument("--input", required=True, help="Path to features_v1.csv")
    p.add_argument("--model_out", required=True, help="Where to save trained model (.pkl)")
    p.add_argument("--metrics_out", required=True, help="Where to save metrics (.json)")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive class")
    args = p.parse_args()

    in_path = Path(args.input)
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading features: {in_path}")
    df = pd.read_csv(in_path)
    df.columns = df.columns.str.strip()

    if LABEL_COL not in df.columns:
        logging.error(f"Missing label column: {LABEL_COL}")
        return 2

    y = df[LABEL_COL].astype(int)

    drop_cols = [LABEL_COL]
    if ID_COL in df.columns:
        drop_cols.append(ID_COL)
    X = df.drop(columns=drop_cols)

    logging.info(f"Rows={len(df)} Features={X.shape[1]} PosRate={y.mean():.3f}")

    # Proper split for real evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logging.info(f"Train={len(X_train)} Test={len(X_test)}")
    logging.info(f"PosRate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    base_model = LogisticRegression(max_iter=400, class_weight="balanced")

    # Fit base model
    base_model.fit(X_train, y_train)

    # Platt scaling calibration (sigmoid)
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5
    )

    calibrated_model.fit(X_train, y_train)
   
     # Raw probabilities
    raw_prob = base_model.predict_proba(X_test)[:, 1]

    # Calibrated probabilities
    cal_prob = calibrated_model.predict_proba(X_test)[:, 1]

    # --- Discrimination ---
    raw_auc = float(roc_auc_score(y_test, raw_prob))
    cal_auc = float(roc_auc_score(y_test, cal_prob))

    # --- Calibration quality ---
    raw_brier = float(brier_score_loss(y_test, raw_prob))
    cal_brier = float(brier_score_loss(y_test, cal_prob))

    # --- Cost analysis ---
    thresholds = np.linspace(0.01, 0.99, 99)
    cost_results = []

    for t in thresholds:
        y_pred = (cal_prob >= t).astype(int)
        fn = ((y_test == 1) & (y_pred == 0)).sum()
        fp = ((y_test == 0) & (y_pred == 1)).sum()
        total_cost = 10 * fn + 1 * fp

        cost_results.append({
            "threshold": float(t),
            "total_cost": int(total_cost),
            "false_negatives": int(fn),
            "false_positives": int(fp),
        })

    best = min(cost_results, key=lambda x: x["total_cost"])

    metrics = {
        "model": "logistic_regression",
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "positive_rate_overall": float(y.mean()),
        "raw_auc": raw_auc,
        "calibrated_auc": cal_auc,
        "raw_brier": raw_brier,
        "calibrated_brier": cal_brier,
        "cost_analysis": {
            "best_threshold": best,
            "theoretical_threshold": 1 / 11
        }
    }

    logging.info(f"Raw AUC={raw_auc:.3f} | Calibrated AUC={cal_auc:.3f}")
    logging.info(f"Raw Brier={raw_brier:.4f} | Calibrated Brier={cal_brier:.4f}")
    logging.info(f"Best threshold by cost: {best['threshold']:.3f}")
    logging.info(f"Total cost at best threshold: {best['total_cost']}")

    logging.info(f"Saving metrics: {metrics_out}")
    metrics_out.write_text(json.dumps(metrics, indent=2))

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
