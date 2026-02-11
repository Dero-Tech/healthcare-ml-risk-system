import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

REQUIRED_COLUMNS = [
    "patient_id", "age", "sex", "admit_type",
    "prior_visits", "has_diabetes", "label_risk"
]

ALLOWED_SEX = {"M", "F"}
ALLOWED_LABELS = {0, 1}


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["age"].isna().any():
        raise ValueError("age has missing values")
    if (df["age"] < 0).any() or (df["age"] > 120).any():
        raise ValueError("age out of expected range 0-120")

    sex_vals = set(df["sex"].dropna().unique())
    if not sex_vals.issubset(ALLOWED_SEX):
        raise ValueError(f"Unexpected sex values: {sex_vals} (allowed: {ALLOWED_SEX})")

    label_vals = set(df["label_risk"].dropna().unique())
    if not label_vals.issubset(ALLOWED_LABELS):
        raise ValueError(f"label_risk must be 0/1; got: {label_vals}")


def main() -> int:
    p = argparse.ArgumentParser(description="Load and validate raw healthcare-style CSV.")
    p.add_argument("--input", required=True, help="Path to raw CSV")
    p.add_argument("--output", required=True, help="Path to cleaned CSV")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading input: {in_path}")
    if not in_path.exists():
        logging.error(f"Input file not found: {in_path}")
        return 2

    df = pd.read_csv(in_path)
    df.columns = df.columns.str.strip()
    logging.info(f"Loaded rows={len(df)} cols={len(df.columns)}")

    try:
        validate_schema(df)
    except Exception as e:
        logging.exception(f"Schema validation failed: {e}")
        return 3

    df = df.copy()
    for col in ["patient_id", "age", "prior_visits", "has_diabetes", "label_risk"]:
        df[col] = df[col].astype(int)

    logging.info(f"Writing cleaned data: {out_path}")
    df.to_csv(out_path, index=False)
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
