import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

FEATURE_VERSION = "v1"

# Columns we expect coming from clean.csv
INPUT_COLUMNS = [
    "patient_id", "age", "sex", "admit_type",
    "prior_visits", "has_diabetes", "label_risk"
]

CATEGORICAL_COLS = ["sex", "admit_type"]
NUMERIC_COLS = ["age", "prior_visits", "has_diabetes"]
LABEL_COL = "label_risk"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Defensive: strip whitespace again (real-world exports)
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    # One-hot encode categoricals
    cat = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=False)

    # Numeric features
    num = df[NUMERIC_COLS].copy()

    # Keep label and patient_id (patient_id not a feature, but useful for tracing)
    out = pd.concat(
        [df[["patient_id"]], num, cat, df[[LABEL_COL]]],
        axis=1
    )

    # Final sanity: no nulls in features/label
    if out.isna().any().any():
        bad_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Null values found after feature build in columns: {bad_cols}")

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Build model-ready features from clean healthcare CSV.")
    p.add_argument("--input", required=True, help="Path to clean.csv")
    p.add_argument("--output", required=True, help="Path to features CSV")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading clean data: {in_path}")
    if not in_path.exists():
        logging.error(f"Input file not found: {in_path}")
        return 2

    df = pd.read_csv(in_path)
    logging.info(f"Loaded rows={len(df)} cols={len(df.columns)}")

    try:
        feats = build_features(df)
    except Exception as e:
        logging.exception(f"Feature build failed: {e}")
        return 3

    logging.info(f"Built features: rows={len(feats)} cols={len(feats.columns)} version={FEATURE_VERSION}")
    logging.info(f"Writing features: {out_path}")
    feats.to_csv(out_path, index=False)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
