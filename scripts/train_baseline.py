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


def eval_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
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

    model = LogisticRegression(max_iter=400, class_weight="balanced")
    model.fit(X_train, y_train)

    # Probabilities for evaluation
    y_prob = model.predict_proba(X_test)[:, 1]

    # AUC is threshold-independent
    auc = float(roc_auc_score(y_test, y_prob))

    # Evaluate at default threshold 0.5 AND at user threshold
    eval_05 = eval_at_threshold(y_test, y_prob, 0.5)
    eval_user = eval_at_threshold(y_test, y_prob, args.threshold)

    metrics = {
        "model": "logistic_regression",
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "positive_rate_overall": float(y.mean()),
        "roc_auc": auc,
        "eval_at_0_5": eval_05,
        "eval_at_threshold": eval_user,
    }

    logging.info(f"AUC={auc:.3f}")
    logging.info(
        f"@0.5 Precision={eval_05['precision']:.3f} Recall={eval_05['recall']:.3f} F1={eval_05['f1']:.3f}"
    )
    logging.info(
        f"@{args.threshold:.2f} Precision={eval_user['precision']:.3f} Recall={eval_user['recall']:.3f} F1={eval_user['f1']:.3f}"
    )

    logging.info(f"Saving model: {model_out}")
    joblib.dump(model, model_out)

    logging.info(f"Saving metrics: {metrics_out}")
    metrics_out.write_text(json.dumps(metrics, indent=2))

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
