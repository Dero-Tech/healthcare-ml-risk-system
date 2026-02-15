import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main() -> int:
    p = argparse.ArgumentParser(description="Generate synthetic EHR-like tabular data.")
    p.add_argument("--n", type=int, default=2000, help="Number of rows")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n

    patient_id = np.arange(1, n + 1)

    # Demographics / utilization
    age = rng.integers(18, 91, size=n)
    sex = rng.choice(["M", "F"], size=n, p=[0.48, 0.52])
    admit_type = rng.choice(["ED", "Clinic"], size=n, p=[0.35, 0.65])
    prior_visits = rng.poisson(lam=2.0, size=n)
    has_diabetes = rng.binomial(1, p=sigmoid((age - 55) / 12), size=n)

    # Risk label generation (imperfect, realistic-ish)
    # Higher risk with: older age, ED admits, more prior visits, diabetes
    linear = (
        -4.0
        + 0.045 * age
        + 0.9 * (admit_type == "ED").astype(int)
        + 0.25 * prior_visits
        + 1.1 * has_diabetes
        + rng.normal(0, 0.6, size=n)
    )
    p_risk = sigmoid(linear)

    # Create mild class imbalance (common in healthcare)
    # By shifting the threshold slightly, we can make positives ~10-25%
    label_risk = (p_risk > 0.55).astype(int)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "admit_type": admit_type,
        "prior_visits": prior_visits,
        "has_diabetes": has_diabetes,
        "label_risk": label_risk
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logging.info(f"Wrote {len(df)} rows to {out_path}")
    logging.info(f"Positive rate={df['label_risk'].mean():.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
