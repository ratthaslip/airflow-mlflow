"""
generate_sample_data.py
------------------------
Create a synthetic `flood_dataset.csv` with the SAME schema the notebook /
pipeline expects, so `flood_ml_pipeline` runs end-to-end without the real data.

Schema (columns consumed by the pipeline):
    province                            (str)   -> label-encoded
    month                               (int)
    MinRain, MaxRain, AvgRain           (float) -> outlier-capped
    AvgFloodRiskArea(Square meter)      (float) -> outlier-capped
    flooding                            (0/1)   -> imbalanced target

Replace this file with your real `flood_dataset.csv` for production.

Usage:
    python include/generate_sample_data.py data/flood_dataset.csv
"""

import sys

import numpy as np
import pandas as pd

PROVINCES = [
    "Bangkok", "Chiang Mai", "Khon Kaen", "Nakhon Ratchasima", "Ayutthaya",
    "Surat Thani", "Ubon Ratchathani", "Songkhla", "Phitsanulok", "Nonthaburi",
]


def generate(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    province = rng.choice(PROVINCES, size=n)
    month = rng.integers(1, 13, size=n)

    avg_rain = rng.gamma(shape=2.0, scale=40.0, size=n)            # right-skewed -> has outliers
    spread = rng.uniform(5, 60, size=n)
    min_rain = np.clip(avg_rain - spread, 0, None)
    max_rain = avg_rain + spread
    risk_area = rng.gamma(shape=2.0, scale=5000.0, size=n)

    # Flood probability rises with rain & risk area + seasonal (monsoon) effect.
    monsoon = np.isin(month, [7, 8, 9, 10]).astype(float)
    logit = (
        -4.0
        + 0.020 * avg_rain
        + 0.00005 * risk_area
        + 1.2 * monsoon
        + rng.normal(0, 0.5, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    flooding = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "province": province,
            "month": month,
            "MinRain": min_rain.round(2),
            "MaxRain": max_rain.round(2),
            "AvgRain": avg_rain.round(2),
            "AvgFloodRiskArea(Square meter)": risk_area.round(2),
            "flooding": flooding,
        }
    )
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/flood_dataset.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print("Target distribution:\n", df["flooding"].value_counts())
